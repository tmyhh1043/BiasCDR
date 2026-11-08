"""Select a popularity penalty on validation, with the Phase 1 score as fallback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from recbole.utils import init_seed
from scipy.stats import spearmanr

from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.Proposed import Proposed
from recbole_cdr.quick_start.quick_start import _torch_load_checkpoint


def evaluate_split(loader, components, alphas, k, batch_users, max_users=0):
    base_user, base_item, delta_user, delta_item, bias, popularity = components
    device = base_user.device
    user_ids = loader.uid_list
    if max_users > 0:
        user_ids = user_ids[:max_users]
    candidates = ["baseline"] + [f"alpha={alpha:g}" for alpha in alphas]
    totals = {
        name: {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0,
               "arp_sum": 0.0, "items": set()}
        for name in candidates
    }
    with torch.inference_mode():
        for start in range(0, len(user_ids), batch_users):
            ids = user_ids[start:start + batch_users].long().to(device)
            base = base_user[ids] @ base_item.T
            delta = delta_user[ids] @ delta_item.T
            pref = base + delta
            for name, alpha in [("baseline", None)] + [
                (f"alpha={value:g}", value) for value in alphas
            ]:
                scores = (base if alpha is None else pref - alpha * bias).clone()
                scores[:, 0] = -torch.inf
                for row, uid in enumerate(ids.tolist()):
                    history = loader.uid2history_item[uid]
                    if history is not None and len(history):
                        scores[row, history.to(device)] = -torch.inf
                top = scores.topk(k, dim=1).indices.cpu()
                stat = totals[name]
                stat["items"].update(top.reshape(-1).tolist())
                stat["arp_sum"] += popularity[top].sum().item()
                for uid, row in zip(ids.tolist(), top):
                    positives = loader.uid2positive_item[uid]
                    hits = torch.isin(row, positives)
                    hit_count = hits.sum().item()
                    stat["recall"] += hit_count / max(len(positives), 1)
                    if hit_count:
                        ranks = torch.nonzero(hits, as_tuple=False).flatten() + 1
                        stat["mrr"] += 1.0 / ranks[0].item()
                        discounts = 1.0 / torch.log2(ranks.float() + 1.0)
                        ideal_ranks = torch.arange(
                            1, min(len(positives), k) + 1, dtype=torch.float32
                        )
                        ideal = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()
                        stat["ndcg"] += (discounts.sum() / ideal).item()
    user_count = len(user_ids)
    return {
        name: {
            "users": user_count,
            "recall@20": round(stat["recall"] / user_count, 7),
            "ndcg@20": round(stat["ndcg"] / user_count, 7),
            "mrr@20": round(stat["mrr"] / user_count, 7),
            "arp@20": round(stat["arp_sum"] / (user_count * k), 5),
            "coverage@20": round(len(stat["items"]) / (len(base_item) - 1), 7),
        }
        for name, stat in totals.items()
    }


def diagnose_scores(loader, components, sample_users=256, k=20):
    """Phase 1 との順位相関と、少数アイテムへの集中を測る。"""
    base_user, base_item, delta_user, delta_item, _, _ = components
    user_ids = loader.uid_list[:sample_users].long()
    correlations = []
    frequencies = torch.zeros(len(base_item), dtype=torch.long)
    baseline_frequencies = torch.zeros(len(base_item), dtype=torch.long)
    with torch.inference_mode():
        for uid in user_ids.tolist():
            base = (base_user[uid] @ base_item.T).clone()
            preference = base + delta_user[uid] @ delta_item.T
            valid = torch.ones(len(base_item), dtype=torch.bool, device=base.device)
            valid[0] = False
            history = loader.uid2history_item[uid]
            if history is not None and len(history):
                valid[history.to(base.device)] = False
            correlation = spearmanr(
                base[valid].cpu().numpy(), preference[valid].cpu().numpy()
            ).correlation
            if np.isfinite(correlation):
                correlations.append(float(correlation))
            top = preference.masked_fill(~valid, -torch.inf).topk(k).indices.cpu()
            frequencies.scatter_add_(0, top, torch.ones_like(top))
            baseline_top = base.masked_fill(~valid, -torch.inf).topk(k).indices.cpu()
            baseline_frequencies.scatter_add_(
                0, baseline_top, torch.ones_like(baseline_top)
            )
    item_norm = torch.cat((base_item, delta_item), dim=1).norm(dim=1).cpu()
    norm_frequency = spearmanr(
        item_norm[1:].numpy(), frequencies[1:].numpy()
    ).correlation
    return {
        "sample_users": len(user_ids),
        "phase1_preference_rank_spearman_mean": float(np.mean(correlations)),
        "phase1_preference_rank_spearman_median": float(np.median(correlations)),
        "preference_top20_unique_items": int((frequencies > 0).sum()),
        "preference_top20_max_item_frequency": int(frequencies.max()),
        "phase1_top20_unique_items": int((baseline_frequencies > 0).sum()),
        "phase1_top20_max_item_frequency": int(baseline_frequencies.max()),
        "item_norm_recommendation_frequency_spearman": float(norm_frequency),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--recall-tolerance", type=float, default=0.0)
    parser.add_argument("--ndcg-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-users", type=int, default=32)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alphas = sorted({float(value) for value in args.alphas.split(",")})
    if not alphas or min(alphas) < 0 or args.recall_tolerance < 0 or args.ndcg_tolerance < 0:
        parser.error("alphas, recall-tolerance, and ndcg-tolerance must be nonnegative")
    if args.batch_users < 1 or args.max_users < 0:
        parser.error("batch-users must be positive and max-users nonnegative")

    checkpoint = _torch_load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device(args.device)
    config["device"] = device
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    if isinstance(valid, (tuple, list)):
        valid = valid[-1]
    model = Proposed(config, train.dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        representations = model.forward()
        count = model.target_num_items
        components = (
            model.target_z_user,
            model.target_z_item[:count],
            representations["target_user_preference"],
            representations["target_item_preference"][:count],
            model.popularity_head["target"](
                representations["target_item_bias"][:count]
            ).squeeze(-1),
            model.target_item_log_popularity[:count].expm1().cpu(),
        )
        validation = evaluate_split(
            valid, components, alphas, 20, args.batch_users, args.max_users
        )
        minimum_recall = validation["baseline"]["recall@20"] - args.recall_tolerance
        minimum_ndcg = validation["baseline"]["ndcg@20"] - args.ndcg_tolerance
        eligible = [
            name for name, values in validation.items()
            if values["recall@20"] >= minimum_recall
            and values["ndcg@20"] >= minimum_ndcg
        ]
        selected = max(
            eligible,
            key=lambda name: (
                validation[name]["coverage@20"],
                validation[name]["ndcg@20"],
            ),
        )
        selected_alpha = None if selected == "baseline" else float(
            selected.removeprefix("alpha=")
        )
        test_alphas = [] if selected_alpha is None else [selected_alpha]
        test_result = evaluate_split(
            test, components, test_alphas, 20, args.batch_users, args.max_users
        )
        diagnostics = diagnose_scores(valid, components)

    report = {
        "checkpoint": str(args.checkpoint),
        "selection_rule": "max validation coverage subject to Recall and NDCG >= Phase 1 - tolerance",
        "recall_tolerance": args.recall_tolerance,
        "ndcg_tolerance": args.ndcg_tolerance,
        "selected": selected,
        "validation": validation,
        "test": test_result,
        "diagnostics": diagnostics,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    output = args.output or Path("results/additive_alpha_selection") / (
        args.checkpoint.stem + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
