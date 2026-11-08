"""Search popularity reranking rules against a specified ARP target.

Candidate selection uses validation data only. Test data is evaluated once for
the selected candidate and the unpenalized Proposed score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from recbole.utils import init_seed

from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.Proposed import Proposed
from recbole_cdr.quick_start.quick_start import _torch_load_checkpoint


TARGET_ARP = {
    "AmazonPhone_AmazonSport": 78.30,  # Phone → Sport, P2M2CDR
    "AmazonSport_AmazonPhone": 132.71,  # Sport → Phone, P2M2CDR
    "AmazonToy_AmazonSport": 32.16,    # Toy → Sport, P2M2CDR
    "AmazonSport_AmazonToy": 32.89,    # Sport → Toy, P2M2CDR
}


def evaluate(loader, components, specs, batch_users=64):
    base_user, base_item, delta_user, delta_item, penalties, popularity, user_weights = components
    device = base_user.device
    n_users = len(loader.uid_list)
    totals = {
        name: {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0,
               "arp": 0.0, "items": set()}
        for name in specs
    }
    with torch.inference_mode():
        for start in range(0, n_users, batch_users):
            users = loader.uid_list[start:start + batch_users].long().to(device)
            base = base_user[users] @ base_item.T
            delta = delta_user[users] @ delta_item.T
            valid = torch.ones_like(base, dtype=torch.bool)
            valid[:, 0] = False
            for row, uid in enumerate(users.tolist()):
                history = loader.uid2history_item[uid]
                if history is not None and len(history):
                    valid[row, history.to(device)] = False
            for name, (gamma, penalty_type, alpha, weight_mode) in specs.items():
                scores = base + gamma * delta
                if alpha:
                    scores = scores - alpha * user_weights[weight_mode][users, None] * penalties[penalty_type]
                top = scores.masked_fill(~valid, -torch.inf).topk(20, dim=1).indices.cpu()
                stat = totals[name]
                stat["items"].update(top.flatten().tolist())
                stat["arp"] += popularity[top].sum().item()
                for uid, row in zip(users.tolist(), top):
                    positive = loader.uid2positive_item[uid]
                    hits = torch.isin(row, positive)
                    stat["recall"] += hits.sum().item() / max(len(positive), 1)
                    if hits.any():
                        ranks = torch.nonzero(hits, as_tuple=False).flatten() + 1
                        stat["mrr"] += 1.0 / ranks[0].item()
                        ideal_ranks = torch.arange(
                            1, min(len(positive), 20) + 1, dtype=torch.float32
                        )
                        ideal = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()
                        stat["ndcg"] += (
                            (1.0 / torch.log2(ranks.float() + 1.0)).sum() / ideal
                        ).item()
    return {
        name: {
            "users": n_users,
            "recall@20": round(stat["recall"] / n_users, 7),
            "ndcg@20": round(stat["ndcg"] / n_users, 7),
            "mrr@20": round(stat["mrr"] / n_users, 7),
            "arp@20": round(stat["arp"] / (20 * n_users), 5),
            "coverage@20": round(len(stat["items"]) / (len(base_item) - 1), 7),
        }
        for name, stat in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=TARGET_ARP, required=True)
    parser.add_argument("--target-arp", type=float)
    parser.add_argument("--alphas", default="0.01,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--gammas", default="0,0.5,1")
    parser.add_argument("--penalties", default="learned,headtail,rawpop")
    parser.add_argument("--weight-modes", default="uniform")
    parser.add_argument("--batch-users", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.target_arp if args.target_arp is not None else TARGET_ARP[args.dataset]
    alphas = sorted({float(value) for value in args.alphas.split(",")})
    gammas = sorted({float(value) for value in args.gammas.split(",")})
    penalty_names = [name.strip() for name in args.penalties.split(",")]
    weight_modes = [name.strip() for name in args.weight_modes.split(",")]
    if target <= 0 or not alphas or min(alphas) < 0 or not gammas:
        parser.error("target, alphas, and gammas must be valid nonnegative values")

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
        representation = model.forward()
        count = model.target_num_items
        log_pop = model.target_item_log_popularity[:count]
        nonpadding = log_pop[1:]
        cutoff = torch.quantile(nonpadding, 0.9)
        penalties = {
            "learned": model.popularity_head["target"](
                representation["target_item_bias"][:count]
            ).squeeze(-1),
            "logpop": log_pop,
            # 最上位10%の人気アイテムだけを抑える。
            "headtail": (log_pop - cutoff).clamp_min(0),
            # 生の人気度を尺度調整し、極端に人気の高いアイテムを重点的に抑える。
            "rawpop": log_pop.expm1() / log_pop[1:].expm1().std().clamp_min(1),
        }
        if any(name not in penalties for name in penalty_names):
            parser.error("Unknown penalty name")
        # 訓練履歴の人気度が低いユーザーほど、人気アイテムへの罰則を強くする。
        history = train.dataset.inter_matrix(form="csr", domain="target")[:, :count]
        history_count = np.asarray(history.getnnz(axis=1)).reshape(-1)
        history_mean = np.asarray(
            history @ log_pop.detach().cpu().numpy()
        ).reshape(-1) / np.maximum(history_count, 1)
        order = np.argsort(history_mean, kind="stable")
        percentile = np.empty(len(order), dtype=np.float32)
        percentile[order] = np.arange(len(order), dtype=np.float32) / max(len(order) - 1, 1)
        percentile = torch.as_tensor(percentile, device=device)
        user_weights = {
            "uniform": torch.ones(len(order), device=device),
            "tail_linear": 2.0 * (1.0 - percentile),
            "tail_quadratic": 3.0 * (1.0 - percentile).square(),
        }
        if any(name not in user_weights for name in weight_modes):
            parser.error("Unknown weight mode")
        components = (
            model.target_z_user,
            model.target_z_item[:count],
            representation["target_user_preference"],
            representation["target_item_preference"][:count],
            penalties,
            log_pop.expm1().cpu(),
            user_weights,
        )
        specs = {
            "preference": (1.0, "learned", 0.0, "uniform"),
            "phase1": (0.0, "learned", 0.0, "uniform"),
        }
        for gamma in gammas:
            for penalty in penalty_names:
                for weight_mode in weight_modes:
                    for alpha in alphas:
                        specs[f"g={gamma:g},p={penalty},w={weight_mode},a={alpha:g}"] = (
                            gamma, penalty, alpha, weight_mode
                        )
        validation = evaluate(valid, components, specs, args.batch_users)
        reference = validation["preference"]
        feasible = [
            name for name, values in validation.items()
            if values["arp@20"] <= target
            and values["recall@20"] >= reference["recall@20"]
            and values["ndcg@20"] >= reference["ndcg@20"]
            and values["coverage@20"] >= reference["coverage@20"]
        ]
        selected = max(
            feasible,
            key=lambda name: (
                validation[name]["coverage@20"],
                validation[name]["recall@20"],
                validation[name]["ndcg@20"],
            ),
        ) if feasible else None
        # 条件未達なら目標ARP内で精度損失が最小の候補を診断用に残す。
        near = [name for name, values in validation.items()
                if values["arp@20"] <= target]
        closest = max(
            near,
            key=lambda name: (
                validation[name]["recall@20"],
                validation[name]["ndcg@20"],
            ),
        ) if near else None
        test_specs = {"preference": specs["preference"]}
        if selected is not None:
            test_specs[selected] = specs[selected]
        elif closest is not None:
            test_specs[closest] = specs[closest]
        test_result = evaluate(test, components, test_specs, args.batch_users)

    report = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "target_arp@20": target,
        "reference": "unpenalized Proposed preference score, same checkpoint and seed",
        "selected": selected,
        "target_feasible_on_validation": selected is not None,
        "closest_at_target": closest,
        "validation": validation,
        "test": test_result,
    }
    output = args.output or Path("results/arp_target_search") / (
        args.dataset + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset,
        "target_arp@20": target,
        "selected": selected,
        "closest_at_target": closest,
        "reference_validation": reference,
        "selected_validation": validation.get(selected) if selected else None,
        "closest_validation": validation.get(closest) if closest else None,
        "test": test_result,
        "report": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
