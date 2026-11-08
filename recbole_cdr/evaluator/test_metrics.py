"""Test-only recommendation diversity/popularity metrics."""

from __future__ import annotations

import csv
import os
from collections import OrderedDict

import numpy as np
import torch


def _config_get(config, key, default=None):
    return config[key] if key in config else default


def _dataset_name(config):
    dataset = _config_get(config, "dataset", "")
    if isinstance(dataset, dict):
        source = dataset.get("source_domain", "")
        target = dataset.get("target_domain", "")
        return f"{source}->{target}"
    return str(dataset)


def _metric_key(name, k):
    return f"{name.lower()}@{k}"


def _gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - (n + 1.0) / n)


def _target_item_popularity(model, train_dataset, num_items):
    if hasattr(model, "target_interaction_matrix"):
        item_ids = model.target_interaction_matrix.col
    elif hasattr(train_dataset, "inter_matrix"):
        item_ids = train_dataset.inter_matrix(form="coo", value_field=None, domain="target").col
    else:
        target_dataset = getattr(train_dataset, "target_domain_dataset", train_dataset)
        iid_field = target_dataset.iid_field
        item_ids = target_dataset.inter_feat[iid_field].numpy()
    return np.bincount(np.asarray(item_ids, dtype=np.int64), minlength=num_items).astype(np.float64)


def _target_item_features(model, num_items):
    if hasattr(model, "target_text_feat_emb") and hasattr(model, "target_visual_feat_emb"):
        text = model.target_text_feat_emb.weight[:num_items].detach()
        visual = model.target_visual_feat_emb.weight[:num_items].detach()
        return torch.cat([text, visual], dim=1)
    if hasattr(model, "target_item_embedding"):
        return model.target_item_embedding.weight[:num_items].detach()
    return None


def _test_uid_list(test_data, sample_size=None, seed=2022):
    uid_list = test_data.uid_list
    if hasattr(uid_list, "detach"):
        uid_list = uid_list.detach().cpu()
    else:
        uid_list = torch.as_tensor(uid_list, dtype=torch.long)

    if sample_size is None or sample_size <= 0 or sample_size >= len(uid_list):
        return uid_list

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(uid_list), generator=generator)[:sample_size]
    return uid_list[indices].sort().values


def _iter_topk(model, test_data, max_k, batch_size, uid_list):
    from recbole.data.interaction import Interaction

    device = model.device
    uid_field = getattr(test_data, "uid_field", getattr(model, "TARGET_USER_ID"))
    num_items = getattr(model, "target_num_items", None)
    if num_items is None:
        num_items = test_data.dataset.num(test_data.iid_field)
    k = min(max_k, num_items)

    for start in range(0, len(uid_list), batch_size):
        users = uid_list[start:start + batch_size].long()
        interaction = Interaction({uid_field: users}).to(device)
        scores = model.full_sort_predict(interaction)
        scores = scores.view(users.size(0), -1)
        scores[:, 0] = -torch.inf

        history = getattr(test_data, "uid2history_item", None)
        if history is not None:
            for row, uid in enumerate(users.tolist()):
                history_items = history[uid]
                if history_items is not None and len(history_items) > 0:
                    scores[row, history_items.to(device)] = -torch.inf

        _, topk_items = torch.topk(scores, k=k, dim=1)
        yield users.detach().cpu(), topk_items.detach().cpu()


def calculate_validation_coverage_arp(config, model, train_dataset, valid_data):
    """Compute full-ranking validation popularity and warm/cold diagnostics."""
    topk = list(_config_get(config, "topk", [20]))
    num_items = getattr(model, "target_num_items", valid_data.dataset.num(valid_data.iid_field))
    item_pop = _target_item_popularity(model, train_dataset, num_items)
    users = _test_uid_list(valid_data)
    batch_size = int(_config_get(config, "valid_extra_metrics_batch_size", 256))
    if batch_size < 1:
        raise ValueError("valid_extra_metrics_batch_size must be positive")
    coverage = {k: set() for k in topk}
    pop_sum = {k: 0.0 for k in topk}
    ratio = float(_config_get(config, "long_tail_item_ratio", 0.8))
    if not 0 < ratio <= 1:
        raise ValueError("long_tail_item_ratio must be in (0, 1]")
    catalog = np.arange(1, num_items, dtype=np.int64)
    warm = catalog[item_pop[catalog] > 0]
    warm_tail = np.zeros(num_items, dtype=bool)
    if warm.size:
        boundary = np.sort(item_pop[warm])[int(np.ceil(warm.size * ratio)) - 1]
        warm_tail[warm] = item_pop[warm] <= boundary
    cold = np.zeros(num_items, dtype=bool)
    cold[catalog] = item_pop[catalog] == 0
    groups = {"WarmTail": warm_tail, "ColdItem": cold}
    group_coverage = {name: {k: set() for k in topk} for name in groups}
    group_hits = {name: {k: 0 for k in topk} for name in groups}
    group_positives = {name: {k: 0 for k in topk} for name in groups}
    user_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_users, rec_items in _iter_topk(model, valid_data, max(topk), batch_size, users):
                rec = rec_items.numpy()
                user_count += len(rec)
                for k in topk:
                    selected = rec[:, :k]
                    coverage[k].update(selected[selected > 0].tolist())
                    pop_sum[k] += float(item_pop[selected].sum())
                    for name, mask in groups.items():
                        valid_items = selected[selected > 0]
                        group_coverage[name][k].update(
                            valid_items[mask[valid_items]].tolist()
                        )
                    for row, uid in enumerate(batch_users.tolist()):
                        positive = valid_data.uid2positive_item[uid]
                        if positive is None or len(positive) == 0:
                            continue
                        positive_np = positive.detach().cpu().numpy()
                        for name, mask in groups.items():
                            relevant = positive_np[mask[positive_np]]
                            group_positives[name][k] += len(relevant)
                            group_hits[name][k] += int(np.isin(
                                selected[row], relevant
                            ).sum())
    finally:
        if was_training:
            model.train()
    result = OrderedDict()
    for k in topk:
        result[_metric_key("Coverage", k)] = round(
            len(coverage[k]) / max(num_items - 1, 1), 7
        )
        result[_metric_key("ARP", k)] = round(
            pop_sum[k] / (user_count * k) if user_count else 0.0, 7
        )
        for name, mask in groups.items():
            catalog_count = int(mask.sum())
            positive_count = group_positives[name][k]
            result[_metric_key(f"{name}Coverage", k)] = round(
                len(group_coverage[name][k]) / catalog_count
                if catalog_count else 0.0, 7
            )
            result[_metric_key(f"{name}Recall", k)] = round(
                group_hits[name][k] / positive_count
                if positive_count else 0.0, 7
            )
            result[_metric_key(f"{name}PositiveCount", k)] = positive_count
    return result


def _popularity_item_groups(item_pop, num_items, head_ratio, middle_ratio):
    """Assign catalog items to head/middle/tail by train popularity rank."""
    item_ids = np.arange(1, num_items, dtype=np.int64)
    order = item_ids[
        np.argsort(-item_pop[item_ids], kind="stable")
    ]
    head_size = int(np.ceil(len(order) * head_ratio))
    middle_size = int(np.ceil(len(order) * middle_ratio))
    head_end = min(head_size, len(order))
    middle_end = min(head_end + middle_size, len(order))
    groups = np.full(num_items, 2, dtype=np.int8)
    groups[0] = -1
    groups[order[:head_end]] = 0
    groups[order[head_end:middle_end]] = 1
    return groups


def _user_popularity_groups(values, group_count):
    """Quantile-group users by their train-history popularity indicator."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or group_count <= 1:
        return np.zeros(values.size, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, group_count + 1)[1:-1]
    boundaries = np.quantile(values, quantiles)
    return np.searchsorted(boundaries, values, side="right")


def _safe_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.size < 2
        or second.size != first.size
        or np.std(first) <= 1e-12
        or np.std(second) <= 1e-12
    ):
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def calculate_test_extra_metrics(config, model, train_dataset, test_data):
    """Calculate AvgPop/Coverage/ILD/Gini from test recommendations only."""
    if not _config_get(config, "test_extra_metrics", True):
        return OrderedDict()

    topk = list(_config_get(config, "topk", [10]))
    max_k = max(topk)
    batch_size = int(_config_get(config, "test_extra_metrics_batch_size", 256))
    sample_size = _config_get(config, "test_extra_metrics_sample_size", None)
    sample_seed = _config_get(config, "test_extra_metrics_seed", 2022)
    num_items = getattr(model, "target_num_items", test_data.dataset.num(test_data.iid_field))
    effective_item_count = max(num_items - 1, 1)
    uid_list = _test_uid_list(test_data, sample_size=sample_size, seed=sample_seed)

    item_pop = _target_item_popularity(model, train_dataset, num_items)
    long_tail_enabled = bool(_config_get(
        config, "test_long_tail_metrics",
        _config_get(config, "test_long_tail_coverage", False),
    ))
    long_tail_ratio = float(
        _config_get(config, "long_tail_item_ratio", 0.8)
    )
    if not 0.0 < long_tail_ratio <= 1.0:
        raise ValueError("long_tail_item_ratio must be in (0, 1].")
    catalog_items = np.arange(1, num_items, dtype=np.int64)
    long_tail_size = int(np.ceil(catalog_items.size * long_tail_ratio))
    long_tail_items = catalog_items[
        np.argsort(item_pop[catalog_items], kind="stable")[:long_tail_size]
    ]
    long_tail_mask = np.zeros(num_items, dtype=bool)
    long_tail_mask[long_tail_items] = True
    # Cold is defined strictly by zero target-domain training interactions.
    # Warm-tail is the bottom fraction of training-observed items by count;
    # all items tied at the boundary belong to the same group.
    cold_mask = np.zeros(num_items, dtype=bool)
    cold_mask[catalog_items] = item_pop[catalog_items] == 0
    warm_items = catalog_items[item_pop[catalog_items] > 0]
    warm_tail_mask = np.zeros(num_items, dtype=bool)
    if warm_items.size:
        boundary = np.sort(item_pop[warm_items])[
            int(np.ceil(warm_items.size * long_tail_ratio)) - 1
        ]
        warm_tail_mask[warm_items] = item_pop[warm_items] <= boundary
    item_features = _target_item_features(model, num_items)
    if item_features is not None:
        item_features = torch.nn.functional.normalize(item_features, p=2, dim=1).cpu()

    popularity_breakdown = bool(
        _config_get(config, "test_popularity_breakdown", False)
    )
    head_ratio = float(_config_get(config, "popularity_head_ratio", 0.2))
    middle_ratio = float(
        _config_get(config, "popularity_middle_ratio", 0.3)
    )
    user_group_count = int(
        _config_get(config, "popularity_user_group_count", 3)
    )
    if (
        head_ratio < 0.0
        or middle_ratio < 0.0
        or head_ratio + middle_ratio > 1.0
    ):
        raise ValueError(
            "popularity_head_ratio and popularity_middle_ratio must be "
            "non-negative and sum to at most 1."
        )
    if user_group_count < 1:
        raise ValueError("popularity_user_group_count must be positive.")
    item_groups = _popularity_item_groups(
        item_pop, num_items, head_ratio, middle_ratio
    )
    target_user_popularity = getattr(
        model, "target_user_popularity", None
    )
    source_user_popularity = getattr(
        model, "source_user_popularity", None
    )
    source_user_has_history = getattr(
        model, "source_user_has_history", None
    )
    normalized_target_popularity = getattr(
        model, "target_item_popularity", None
    )
    if target_user_popularity is not None:
        target_user_popularity = target_user_popularity.detach().cpu().numpy()
    if source_user_popularity is not None:
        source_user_popularity = source_user_popularity.detach().cpu().numpy()
    if source_user_has_history is not None:
        source_user_has_history = (
            source_user_has_history.detach().cpu().numpy()
        )
    if normalized_target_popularity is not None:
        normalized_target_popularity = (
            normalized_target_popularity.detach().cpu().numpy()
        )
    else:
        log_popularity = np.log1p(item_pop)
        normalized_target_popularity = log_popularity / max(
            float(log_popularity.max()), 1e-8
        )

    uid_numpy = uid_list.numpy()
    if target_user_popularity is not None:
        user_pop_values = target_user_popularity[uid_numpy]
    else:
        user_pop_values = np.zeros(len(uid_numpy), dtype=np.float64)
    user_group_by_position = _user_popularity_groups(
        user_pop_values, user_group_count
    )
    user_group_by_id = {
        int(uid): int(group)
        for uid, group in zip(uid_numpy, user_group_by_position)
    }

    user_count = 0
    pop_sums = {k: 0.0 for k in topk}
    ild_sums = {k: 0.0 for k in topk}
    coverage_sets = {k: set() for k in topk}
    long_tail_coverage_sets = {k: set() for k in topk}
    long_tail_exposure_counts = {k: 0 for k in topk}
    long_tail_hit_counts = {k: 0 for k in topk}
    long_tail_positive_counts = {k: 0 for k in topk}
    warm_mask = np.zeros(num_items, dtype=bool)
    warm_mask[warm_items] = True
    group_masks = {
        "WarmItem": warm_mask,
        "WarmTail": warm_tail_mask,
        "ColdItem": cold_mask,
    }
    group_coverage = {name: {k: set() for k in topk} for name in group_masks}
    group_hits = {name: {k: 0 for k in topk} for name in group_masks}
    group_positives = {name: {k: 0 for k in topk} for name in group_masks}
    group_positive_users = {name: {k: 0 for k in topk} for name in group_masks}
    exposure_counts = {k: np.zeros(num_items, dtype=np.float64) for k in topk}
    item_group_hits = {k: np.zeros(3, dtype=np.float64) for k in topk}
    item_group_relevant = {k: np.zeros(3, dtype=np.float64) for k in topk}
    user_group_recall = {
        k: np.zeros(user_group_count, dtype=np.float64) for k in topk
    }
    user_group_ndcg = {
        k: np.zeros(user_group_count, dtype=np.float64) for k in topk
    }
    user_group_counts = {
        k: np.zeros(user_group_count, dtype=np.float64) for k in topk
    }
    source_pop_values = {k: [] for k in topk}
    recommended_pop_values = {k: [] for k in topk}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_users, rec_items in _iter_topk(
            model, test_data, max_k, batch_size, uid_list
        ):
            rec_np = rec_items.numpy()
            user_count += rec_np.shape[0]
            for k in topk:
                items_k = rec_np[:, :k]
                pop_sums[k] += float(item_pop[items_k].sum())
                valid_items = items_k[items_k > 0]
                coverage_sets[k].update(valid_items.tolist())
                if long_tail_enabled:
                    long_tail_recommended = valid_items[
                        long_tail_mask[valid_items]
                    ]
                    long_tail_coverage_sets[k].update(
                        long_tail_recommended.tolist()
                    )
                    long_tail_exposure_counts[k] += int(
                        long_tail_mask[items_k].sum()
                    )
                    for row, uid in enumerate(batch_users.tolist()):
                        positive = test_data.uid2positive_item[uid]
                        if positive is None or len(positive) == 0:
                            continue
                        positive_np = positive.detach().cpu().numpy()
                        positive_tail = positive_np[
                            long_tail_mask[positive_np]
                        ]
                        long_tail_positive_counts[k] += len(positive_tail)
                        long_tail_hit_counts[k] += int(np.isin(
                            items_k[row], positive_tail
                        ).sum())
                    for name, mask in group_masks.items():
                        group_coverage[name][k].update(
                            valid_items[mask[valid_items]].tolist()
                        )
                    for row, uid in enumerate(batch_users.tolist()):
                        positive = test_data.uid2positive_item[uid]
                        if positive is None or len(positive) == 0:
                            continue
                        positive_np = positive.detach().cpu().numpy()
                        for name, mask in group_masks.items():
                            relevant = positive_np[mask[positive_np]]
                            group_positives[name][k] += len(relevant)
                            group_positive_users[name][k] += int(len(relevant) > 0)
                            group_hits[name][k] += int(np.isin(
                                items_k[row], relevant
                            ).sum())
                exposure_counts[k] += np.bincount(items_k.reshape(-1), minlength=num_items)

                if item_features is not None and k > 1:
                    feats = item_features[torch.as_tensor(items_k, dtype=torch.long)]
                    sim = torch.bmm(feats, feats.transpose(1, 2))
                    tri = torch.triu_indices(k, k, offset=1)
                    dissim = 1.0 - sim[:, tri[0], tri[1]]
                    ild_sums[k] += float(dissim.mean(dim=1).sum().item())

                if not popularity_breakdown:
                    continue
                for row, uid in enumerate(batch_users.tolist()):
                    positive = test_data.uid2positive_item[uid]
                    if positive is None or len(positive) == 0:
                        continue
                    positive_np = positive.detach().cpu().numpy()
                    recommended = items_k[row]
                    hit_mask = np.isin(recommended, positive_np)
                    hit_items = recommended[hit_mask]

                    for group in range(3):
                        item_group_relevant[k][group] += float(
                            np.sum(item_groups[positive_np] == group)
                        )
                        item_group_hits[k][group] += float(
                            np.sum(item_groups[hit_items] == group)
                        )

                    user_group = user_group_by_id.get(int(uid), 0)
                    recall = float(hit_mask.sum()) / float(len(positive_np))
                    discounts = 1.0 / np.log2(
                        np.arange(2, len(recommended) + 2)
                    )
                    dcg = float((hit_mask.astype(np.float64) * discounts).sum())
                    ideal_length = min(len(positive_np), len(recommended))
                    idcg = float(discounts[:ideal_length].sum())
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    user_group_recall[k][user_group] += recall
                    user_group_ndcg[k][user_group] += ndcg
                    user_group_counts[k][user_group] += 1.0

                    if (
                        source_user_popularity is not None
                        and uid < len(source_user_popularity)
                        and uid < getattr(model, "overlapped_num_users", 0)
                        and (
                            source_user_has_history is None
                            or bool(source_user_has_history[uid])
                        )
                    ):
                        source_pop_values[k].append(
                            float(source_user_popularity[uid])
                        )
                        recommended_pop_values[k].append(
                            float(
                                normalized_target_popularity[recommended].mean()
                            )
                        )

    if was_training:
        model.train()

    result = OrderedDict()
    if user_count == 0:
        for k in topk:
            result[_metric_key("AvgPop", k)] = 0.0
            result[_metric_key("ARP", k)] = 0.0
            result[_metric_key("Coverage", k)] = 0.0
            if long_tail_enabled:
                result[_metric_key("LongTailCoverage", k)] = 0.0
                result[_metric_key("LongTailExposure", k)] = 0.0
                result[_metric_key("LongTailRecall", k)] = 0.0
                result[_metric_key("LongTailPositiveCount", k)] = 0
                for name, mask in group_masks.items():
                    result[_metric_key(f"{name}Coverage", k)] = 0.0
                    result[_metric_key(f"{name}Recall", k)] = 0.0
                    result[_metric_key(f"{name}PositiveCount", k)] = 0
                    result[_metric_key(f"{name}PositiveUserCount", k)] = 0
                    result[_metric_key(f"{name}CatalogCount", k)] = int(mask.sum())
            result[_metric_key("ILD", k)] = 0.0
            result[_metric_key("Gini", k)] = 0.0
        return result

    for k in topk:
        average_popularity = round(pop_sums[k] / (user_count * k), 7)
        result[_metric_key("AvgPop", k)] = average_popularity
        result[_metric_key("ARP", k)] = average_popularity
        result[_metric_key("Coverage", k)] = round(len(coverage_sets[k]) / effective_item_count, 7)
        if long_tail_enabled:
            result[_metric_key("LongTailCoverage", k)] = round(
                len(long_tail_coverage_sets[k]) / max(long_tail_size, 1),
                7,
            )
            result[_metric_key("LongTailExposure", k)] = round(
                long_tail_exposure_counts[k] / (user_count * k), 7
            )
            result[_metric_key("LongTailRecall", k)] = round(
                long_tail_hit_counts[k] / long_tail_positive_counts[k]
                if long_tail_positive_counts[k] else 0.0,
                7,
            )
            result[_metric_key("LongTailPositiveCount", k)] = (
                long_tail_positive_counts[k]
            )
            for name, mask in group_masks.items():
                catalog_count = int(mask.sum())
                positive_count = group_positives[name][k]
                result[_metric_key(f"{name}Coverage", k)] = round(
                    len(group_coverage[name][k]) / catalog_count
                    if catalog_count else 0.0, 7
                )
                result[_metric_key(f"{name}Recall", k)] = round(
                    group_hits[name][k] / positive_count
                    if positive_count else 0.0, 7
                )
                result[_metric_key(f"{name}PositiveCount", k)] = positive_count
                result[_metric_key(f"{name}PositiveUserCount", k)] = (
                    group_positive_users[name][k]
                )
                result[_metric_key(f"{name}CatalogCount", k)] = catalog_count
        result[_metric_key("ILD", k)] = round(ild_sums[k] / user_count if item_features is not None and k > 1 else 0.0, 7)
        exposure = exposure_counts[k][1:num_items] if num_items > 1 else exposure_counts[k]
        result[_metric_key("Gini", k)] = round(_gini(exposure), 7)
        if popularity_breakdown:
            for group_name, group in zip(
                ("Head", "Middle", "Tail"), range(3)
            ):
                denominator = item_group_relevant[k][group]
                value = (
                    item_group_hits[k][group] / denominator
                    if denominator > 0
                    else 0.0
                )
                result[_metric_key(f"{group_name}Recall", k)] = round(
                    float(value), 7
                )

            if user_group_count == 3:
                group_names = ("LowPopUser", "MidPopUser", "HighPopUser")
            else:
                group_names = tuple(
                    f"PopUserGroup{group + 1}"
                    for group in range(user_group_count)
                )
            for group, group_name in enumerate(group_names):
                count = user_group_counts[k][group]
                recall = user_group_recall[k][group] / count if count else 0.0
                ndcg = user_group_ndcg[k][group] / count if count else 0.0
                result[_metric_key(f"{group_name}Recall", k)] = round(
                    float(recall), 7
                )
                result[_metric_key(f"{group_name}NDCG", k)] = round(
                    float(ndcg), 7
                )
            result[_metric_key("SourcePopRecPopCorr", k)] = round(
                _safe_correlation(
                    source_pop_values[k], recommended_pop_values[k]
                ),
                7,
            )
    return result


def append_metrics_csv(config, metrics):
    include_params = bool(_config_get(config, "metrics_csv_include_params", False))
    csv_path = _config_get(config, "metrics_csv_param_path", None) if include_params else None
    if not csv_path:
        csv_path = _config_get(config, "metrics_csv_path", None)
    if not csv_path:
        return None

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    append = bool(_config_get(config, "metrics_csv_append", True))
    mode = "a" if append else "w"
    write_header = (not append) or (not os.path.exists(csv_path))
    dataset = _dataset_name(config)
    model = _config_get(config, "model", "")
    param_keys = list(_config_get(config, "metrics_csv_param_keys", [])) if include_params else []
    param_values = OrderedDict()
    for key in param_keys:
        value = _config_get(config, key, "")
        if isinstance(value, (list, tuple, dict)):
            value = str(value)
        param_values[key] = value

    metric_keys = list(metrics.keys())
    fieldnames = ["dataset", "model"] + param_keys + metric_keys
    if append and os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header != fieldnames:
            mode = "w"
            write_header = True

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "dataset": dataset,
            "model": model,
        }
        row.update(param_values)
        row.update(metrics)
        writer.writerow(row)
    return csv_path
