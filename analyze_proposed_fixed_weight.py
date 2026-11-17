"""Evaluation-only diagnostics for saved Proposed checkpoints.

All model parameters stay frozen. Graph thinning is an inference-time stress
test, and timestamp groups are slices of the original random test split.
Neither is equivalent to retraining on a sparse or chronological dataset.
"""

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from analyze_proposed_disentanglement import load_checkpoint, flatten


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _valid_positives(loader, user, num_items):
    values = loader.uid2positive_item[int(user)]
    if values is None:
        return np.empty(0, dtype=np.int64)
    result = np.asarray(_numpy(values), dtype=np.int64)
    return result[(result > 0) & (result < num_items)]


def item_popularity_groups(degrees):
    """Head 20%, middle 30%, tail 50% of train-warm items; cold separately."""
    groups = np.full(len(degrees), 3, dtype=np.int8)
    warm = np.flatnonzero(degrees > 0)
    warm = warm[warm > 0]
    ordered = warm[np.argsort(-degrees[warm], kind="stable")]
    h = int(np.ceil(.2 * len(ordered)))
    m = int(np.ceil(.3 * len(ordered)))
    groups[ordered[:h]] = 0
    groups[ordered[h:h + m]] = 1
    groups[ordered[h + m:]] = 2
    groups[0] = -1
    return groups


def gini(counts):
    values = np.sort(np.asarray(counts, dtype=np.float64))
    total = values.sum()
    if total <= 0 or len(values) == 0:
        return 0.0
    index = np.arange(1, len(values) + 1)
    return float(2 * np.dot(index, values) / (len(values) * total) - (len(values) + 1) / len(values))


def source_popularity_groups(source_matrix, users):
    matrix = source_matrix.tocsr()
    item_degrees = np.asarray(matrix.sum(axis=0)).ravel()
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    mean = np.asarray(matrix @ np.log1p(item_degrees)).ravel() / np.maximum(counts, 1)
    active = users[counts[users] > 0]
    boundaries = np.quantile(mean[active], [1 / 3, 2 / 3]) if len(active) else [0, 0]
    groups = np.searchsorted(boundaries, mean, side="right").astype(np.int8)
    groups[counts == 0] = -1
    return groups, [float(x) for x in boundaries]


def timestamp_lookup(config, train_dataset, loader, num_items):
    """Map raw target timestamps to held-out pairs without altering the split."""
    settings = config["target_domain"]
    path = Path(settings["data_path"]) / settings["dataset"] / f"{settings['dataset']}.inter"
    if not path.is_file():
        return {}, {"available": False, "reason": f"raw file missing: {path}"}
    wanted = {(int(user), int(item)) for user in _numpy(loader.uid_list)
              for item in _valid_positives(loader, user, num_items)}
    user_map = train_dataset.target_user_ID_remap_dict
    item_map = train_dataset.target_item_ID_remap_dict
    user_field = f"{settings['USER_ID_FIELD']}:token"
    item_field = f"{settings['ITEM_ID_FIELD']}:token"
    time_field = f"{settings['TIME_FIELD']}:float"
    found = {}
    duplicate_pairs = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter=config["field_separator"])
        if not {user_field, item_field, time_field}.issubset(reader.fieldnames or []):
            return {}, {"available": False, "reason": "timestamp columns missing"}
        for row in reader:
            user = user_map.get(row[user_field])
            item = item_map.get(row[item_field])
            if user is None or item is None:
                continue
            key = (int(user), int(item))
            if key in wanted:
                stamp = float(row[time_field])
                if key in found:
                    duplicate_pairs.add(key)
                found[key] = max(stamp, found.get(key, -np.inf))
    # Duplicate raw pairs have no unique timestamp after interaction deduplication.
    for key in duplicate_pairs:
        found.pop(key, None)
    metadata = {"available": bool(found), "raw_file": str(path),
                "evaluation_pairs": len(wanted), "matched_pairs": len(found),
                "ambiguous_duplicate_pairs": len(duplicate_pairs),
                "meaning": "Timestamp terciles within the original random test split; not future generalization."}
    return found, metadata


def time_bins(stamps):
    if not stamps:
        return {}, []
    cuts = np.quantile(np.fromiter(stamps.values(), dtype=np.float64), [1 / 3, 2 / 3])
    return {key: int(np.searchsorted(cuts, value, side="right"))
            for key, value in stamps.items()}, [float(x) for x in cuts]


def _score_summary(acc, k):
    if acc["users"] == 0:
        return {"users": 0, "positives": 0, f"recall@{k}": None, f"ndcg@{k}": None}
    return {"users": acc["users"], "positives": acc["positives"],
            f"recall@{k}": acc["hits"] / acc["positives"] if acc["positives"] else None,
            f"ndcg@{k}": acc["ndcg"] / acc["users"]}


def _empty_acc():
    return {"users": 0, "positives": 0, "hits": 0, "ndcg": 0.0}


def _add_user(acc, hit_count, positive_count, ndcg):
    acc["users"] += 1
    acc["positives"] += positive_count
    acc["hits"] += hit_count
    acc["ndcg"] += ndcg


def evaluate(users_embedding, items_embedding, loader, target_degrees, target_counts, source_groups,
             stamp_groups, k, max_users, seed):
    """One full-sort pass; all groups share identical candidates and masking."""
    n_items = len(target_degrees)
    k = min(k, n_items - 1)
    if k < 1:
        raise ValueError("Target catalog has no items")
    all_users = np.asarray(_numpy(loader.uid_list), dtype=np.int64)
    if max_users and max_users < len(all_users):
        all_users = np.sort(np.random.default_rng(seed).choice(all_users, max_users, replace=False))
    item_groups = item_popularity_groups(target_degrees)
    group_names = ("head", "middle", "tail", "cold")
    global_acc = _empty_acc()
    user_history = {name: _empty_acc() for name in ("zero", "1_to_3", "4_to_10", "11_plus")}
    source_pop = {name: _empty_acc() for name in ("low", "middle", "high", "none")}
    item_acc = {name: {"positives": 0, "hits": 0} for name in group_names}
    period_acc = {name: {"positives": 0, "hits": 0} for name in ("older", "middle", "recent")}
    recommended = np.zeros(n_items, dtype=np.int64)
    popularity_sum = 0.0
    for user in all_users:
        positive = _valid_positives(loader, user, n_items)
        if len(positive) == 0:
            continue
        scores = items_embedding[:n_items] @ users_embedding[int(user)]
        scores = scores.copy()
        scores[0] = -np.inf
        history = loader.uid2history_item[int(user)]
        if history is not None:
            scores[np.asarray(_numpy(history), dtype=np.int64)] = -np.inf
        candidates = np.argpartition(-scores, kth=k - 1)[:k]
        top = candidates[np.argsort(-scores[candidates], kind="stable")]
        hit_mask = np.isin(top, positive)
        hit_items = top[hit_mask]
        ideal = np.sum(1 / np.log2(np.arange(2, min(len(positive), k) + 2)))
        ndcg = float(np.sum(hit_mask / np.log2(np.arange(2, k + 2))) / ideal)
        _add_user(global_acc, len(hit_items), len(positive), ndcg)
        count = int(target_counts[user])
        history_name = ("zero" if count == 0 else "1_to_3" if count <= 3
                        else "4_to_10" if count <= 10 else "11_plus")
        _add_user(user_history[history_name], len(hit_items), len(positive), ndcg)
        source_name = ("none" if source_groups[user] < 0 else
                       ("low", "middle", "high")[int(source_groups[user])])
        _add_user(source_pop[source_name], len(hit_items), len(positive), ndcg)
        recommended += np.bincount(top, minlength=n_items)
        popularity_sum += float(target_degrees[top].sum())
        for group_id, name in enumerate(group_names):
            item_acc[name]["positives"] += int(np.sum(item_groups[positive] == group_id))
            item_acc[name]["hits"] += int(np.sum(item_groups[hit_items] == group_id))
        for period_id, name in enumerate(period_acc):
            period_positive = np.array([item for item in positive
                                        if stamp_groups.get((int(user), int(item))) == period_id], dtype=np.int64)
            period_acc[name]["positives"] += len(period_positive)
            period_acc[name]["hits"] += int(np.isin(hit_items, period_positive).sum())
    catalog = max(n_items - 1, 1)
    total_recs = int(recommended.sum())
    result = {
        "overall": _score_summary(global_acc, k),
        "target_history_groups": {name: _score_summary(acc, k) for name, acc in user_history.items()},
        "source_popularity_groups": {name: _score_summary(acc, k) for name, acc in source_pop.items()},
        "item_popularity_groups": {name: {"positives": acc["positives"], "hits": acc["hits"],
                                          f"recall@{k}": acc["hits"] / acc["positives"]
                                          if acc["positives"] else None} for name, acc in item_acc.items()},
        "test_timestamp_groups": {name: {"positives": acc["positives"], "hits": acc["hits"],
                                         f"recall@{k}": acc["hits"] / acc["positives"]
                                         if acc["positives"] else None} for name, acc in period_acc.items()},
        "popularity_diversity": {
            f"average_train_popularity@{k}": popularity_sum / total_recs if total_recs else None,
            f"catalog_coverage@{k}": float(np.count_nonzero(recommended[1:]) / catalog),
            f"exposure_gini@{k}": gini(recommended[1:]),
            f"warm_tail_exposure@{k}": float(recommended[item_groups == 2].sum() / total_recs)
            if total_recs else None,
        },
    }
    return result


def correspondence(source_users, target_users, source_matrix, target_matrix, seed, sample_size):
    """Compare correct cross-domain user pairing with random pairing."""
    source_counts = np.asarray(source_matrix.sum(axis=1)).ravel()
    target_counts = np.asarray(target_matrix.sum(axis=1)).ravel()
    eligible = np.flatnonzero((source_counts > 0) & (target_counts > 0))
    eligible = eligible[eligible > 0]
    rng = np.random.default_rng(seed)
    if len(eligible) > sample_size:
        eligible = np.sort(rng.choice(eligible, sample_size, replace=False))
    if len(eligible) < 2:
        return {"users": len(eligible), "error": "fewer than two active overlap users"}
    a = source_users[eligible].astype(np.float64)
    b = target_users[eligible].astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    similarity = a @ b.T
    diagonal = similarity.diagonal().copy()
    permutation = rng.permutation(len(eligible))
    rank = 1 + (similarity > diagonal[:, None]).sum(axis=1)
    return {"users": int(len(eligible)),
            "same_user_mean_cosine": float(diagonal.mean()),
            "random_pair_mean_cosine": float(similarity[np.arange(len(eligible)), permutation].mean()),
            "same_user_retrieval@1": float(np.mean(rank <= 1)),
            "same_user_retrieval@10": float(np.mean(rank <= min(10, len(eligible)))),
            "random_retrieval@10": float(min(10, len(eligible)) / len(eligible)),
            "meaning": "Representation correspondence, not a causal source-shuffle recommender test."}


def thin_matrix(matrix, rate, priorities):
    """Nested thinning; keep at least one original edge per active user."""
    matrix = matrix.tocsr()
    selected = np.zeros(matrix.nnz, dtype=bool)
    for user in range(1, matrix.shape[0]):
        begin, end = matrix.indptr[user:user + 2]
        count = end - begin
        if count == 0:
            continue
        keep = min(count, max(1, int(np.ceil(rate * count))))
        if keep == count:
            selected[begin:end] = True
        else:
            local = np.argpartition(priorities[begin:end], keep - 1)[:keep]
            selected[begin + local] = True
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))[selected]
    cols = matrix.indices[selected]
    return sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=matrix.shape)


def thinned_target_embeddings(model, matrix, rate, priorities):
    if rate == 1:
        with torch.no_grad():
            users, items = model._gcn(model.target_norm_adj, model.pref_u_emb_tgt, model.pref_i_emb_tgt)
        count = matrix.nnz
    else:
        subset = thin_matrix(matrix, rate, priorities)
        adjacency = model._normalized_adj(subset).to(model.device)
        with torch.no_grad():
            users, items = model._gcn(adjacency, model.pref_u_emb_tgt, model.pref_i_emb_tgt)
        count = subset.nnz
    return users.detach().cpu().numpy(), items.detach().cpu().numpy(), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/proposed_fixed_weight"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--alignment-users", type=int, default=2000)
    parser.add_argument("--keep-rates", default="1.0,0.75,0.5,0.25")
    args = parser.parse_args()
    if args.topk < 1 or args.max_users < 0 or args.alignment_users < 2:
        parser.error("Invalid topk, max-users, or alignment-users")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    rates = [float(x) for x in args.keep_rates.split(",") if x.strip()]
    if not rates or any(not 0 < value <= 1 for value in rates) or len(set(rates)) != len(rates):
        parser.error("--keep-rates must contain unique values in (0,1]")
    if 1.0 not in rates:
        rates.insert(0, 1.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for checkpoint in args.checkpoints:
        print(f"Analyzing {checkpoint}", flush=True)
        started = time.perf_counter()
        config, model, train, valid, test, source, target = load_checkpoint(checkpoint, args.device)
        seed = int(config["seed"])
        train_dataset = train.dataset
        source_matrix = train_dataset.inter_matrix(form="csr", value_field=None, domain="source")
        target_matrix = train_dataset.inter_matrix(form="csr", value_field=None, domain="target")
        target_degrees = np.asarray(target_matrix.sum(axis=0)).ravel()[:model.target_num_items]
        target_counts = np.asarray(target_matrix.sum(axis=1)).ravel()
        loader = valid[-1] if args.split == "valid" and isinstance(valid, (tuple, list)) else (
            valid if args.split == "valid" else test)
        eval_users = np.asarray(_numpy(loader.uid_list), dtype=np.int64)
        source_groups, source_cuts = source_popularity_groups(source_matrix, eval_users)
        stamps, time_metadata = timestamp_lookup(config, train_dataset, loader, model.target_num_items)
        stamp_groups, time_cuts = time_bins(stamps)
        report = {
            "checkpoint": str(checkpoint.resolve()), "dataset": str(config["dataset"]),
            "seed": seed, "split": args.split,
            "design": "Frozen saved weights; original random split and candidate masks.",
            "group_definitions": {
                "target_history": "Original target train interactions: 0, 1-3, 4-10, 11+.",
                "item_popularity": "Among target train-warm items: top 20% head, next 30% middle, rest tail; zero-degree items cold.",
                "source_popularity": "Terciles of mean log(1 + source item train degree) consumed by the user.",
                "metric_averaging": "Recall uses pooled positive interactions; NDCG averages over users with positives.",
            },
            "source_popularity_cutoffs": source_cuts,
            "timestamp": {**time_metadata, "tercile_cutoffs": time_cuts},
            "cross_domain_correspondence": correspondence(
                source[0], target[0], source_matrix, target_matrix, seed, args.alignment_users),
            "inference_graph_thinning": {},
        }
        priorities = np.random.default_rng(seed).random(target_matrix.nnz)
        original_users, original_items = target[0], target[1]
        for rate in rates:
            if rate == 1:
                users, items = original_users, original_items
                retained_edges = target_matrix.nnz
            else:
                users, items, retained_edges = thinned_target_embeddings(model, target_matrix, rate, priorities)
            # Keep history-based groups anchored to the *original* train graph.
            result = evaluate(users, items, loader, target_degrees, target_counts, source_groups,
                              stamp_groups, args.topk, args.max_users, seed)
            report["inference_graph_thinning"][str(rate)] = {
                "requested_keep_rate": rate,
                "retained_train_edge_fraction": retained_edges / target_matrix.nnz,
                "meaning": "Inference-time graph thinning with frozen weights; no sparse-data retraining.",
                **result,
            }
            print(f"  keep={rate}: {result['overall']}", flush=True)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        name = f"{checkpoint.parent.name}__{checkpoint.stem}__{args.split}.json"
        path = args.output_dir / name
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for rate in rates:
            summaries.append({
                "checkpoint": report["checkpoint"], "dataset": report["dataset"],
                "seed": seed, "split": args.split, "requested_keep_rate": rate,
                "same_user_mean_cosine": report["cross_domain_correspondence"].get("same_user_mean_cosine"),
                "same_user_retrieval@10": report["cross_domain_correspondence"].get("same_user_retrieval@10"),
                "timestamp_matched_pairs": report["timestamp"].get("matched_pairs", 0),
                **dict(flatten(report["inference_graph_thinning"][str(rate)])),
            })
        print(f"Wrote {path}", flush=True)
        del model, train, valid, test, source, target, train_dataset, source_matrix, target_matrix
        del report, original_users, original_items, users, items, loader
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary_path = args.output_dir / f"summary_{args.split}.csv"
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
