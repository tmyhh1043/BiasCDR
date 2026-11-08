"""Evaluate popularity leakage and recommendation popularity from saved Proposed models.

Run with ``python analyze_disentanglement.py`` inside the project environment.
The recommender is never updated; only independent diagnostic probes are fit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recbole.utils import init_seed
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.Proposed import Proposed


DATASETS = (
    "AmazonSport_AmazonPhone",
    "AmazonPhone_AmazonSport",
    "AmazonSport_AmazonToy",
    "AmazonToy_AmazonSport",
)
PENALTIES = (0.25, 0.5, 1.0, 2.0)


def load_checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(prediction, target):
    prediction = prediction.reshape(-1).double()
    target = target.reshape(-1).double()
    error = prediction - target
    denominator = ((target - target.mean()) ** 2).sum()
    return {
        "r2": float(1 - error.square().sum() / denominator) if denominator > 0 else float("nan"),
        "mae_log_popularity": float(error.abs().mean()),
    }


def popularity_probe(features, labels, seed, epochs, device):
    """Item-disjoint 60/20/20 probe split, shared by all representations."""
    n = len(labels)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    train_end, valid_end = int(0.6 * n), int(0.8 * n)
    train, valid, test = order[:train_end], order[train_end:valid_end], order[valid_end:]
    x = features.detach().float().cpu().clone()
    y = labels.detach().float().cpu()
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
    x = ((x - mean) / std).clamp(-10, 10).to(device)
    y = y.to(device)
    train, valid, test = train.to(device), valid.to(device), test.to(device)

    augmented = torch.cat((x[train], torch.ones((len(train), 1), device=device)), dim=1)
    penalty = torch.eye(augmented.shape[1], device=device) * 1e-2
    penalty[-1, -1] = 0
    coefficients = torch.linalg.solve(
        augmented.T @ augmented + penalty, augmented.T @ y[train]
    )
    linear_prediction = x[test] @ coefficients[:-1] + coefficients[-1]
    results = {"linear": regression_metrics(linear_prediction, y[test])}

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    # Learn nonlinear information beyond the linear probe. A zero-initialized
    # correction makes the linear fit a valid starting point and fallback.
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    baseline = x @ coefficients[:-1] + coefficients[-1]
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_loss = float((baseline[valid] - y[valid]).square().mean())
    best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
    patience = 0
    for _ in range(epochs):
        net.train()
        shuffled = train[torch.randperm(len(train), device=device)]
        for batch in shuffled.split(8192):
            loss = (baseline[batch] + net(x[batch]).squeeze(1) - y[batch]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            valid_loss = float((baseline[valid] + net(x[valid]).squeeze(1) - y[valid]).square().mean())
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        results["ffn"] = regression_metrics(baseline[test] + net(x[test]).squeeze(1), y[test])
    return results, (len(train), len(valid), len(test))


def as_item_set(values):
    if values is None:
        return set()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {int(value) for value in values if int(value) > 0}


def score_popularity_correlation(scores, log_popularity):
    """Per-user Pearson correlation over available candidate items."""
    valid = torch.isfinite(scores)
    count = valid.sum(1).clamp_min(2)
    safe_scores = torch.where(valid, scores, 0)
    left_mean = safe_scores.sum(1) / count
    right_mean = (valid * log_popularity).sum(1) / count
    left = torch.where(valid, scores - left_mean[:, None], 0)
    right = torch.where(valid, log_popularity[None, :] - right_mean[:, None], 0)
    numerator = (left * right).sum(1)
    denominator = left.square().sum(1).sqrt() * right.square().sum(1).sqrt()
    return (numerator / denominator.clamp_min(1e-12)).cpu().numpy()


def ranking_experiment(model, train_data, test_data, users, item_count, k, batch_size):
    device = model.target_z_user.device
    n_items = model.target_num_items
    popularity = torch.as_tensor(item_count[:n_items], device=device, dtype=torch.float32)
    log_popularity = popularity.log1p()
    pop_z = (log_popularity - log_popularity[1:].mean()) / log_popularity[1:].std().clamp_min(1e-6)
    head_count = max(1, int(0.2 * (n_items - 1)))
    head_ids = torch.topk(popularity[1:], head_count).indices + 1
    head = torch.zeros(n_items, dtype=torch.bool, device=device)
    head[head_ids] = True
    representations = model.forward()
    user_pref = representations["target_user_preference"]
    item_pref = representations["target_item_preference"]
    item_bias = representations["target_item_bias"]
    bias_score = model.popularity_head["target"](
        item_bias[:n_items]
    ).squeeze(-1)
    modulation = 1.0 + getattr(model, "observation_modulation_strength", 1.0) * F.elu(bias_score)
    counterfactual_modulation = F.elu(bias_score) + 1.0
    gcn_user, gcn_item = model.target_z_user, model.target_z_item
    train_matrix = train_data.dataset.inter_matrix(form="csr", domain="target")
    mean_history_pop = np.zeros(len(users), dtype=np.float64)
    for index, uid in enumerate(users):
        ids = train_matrix.indices[train_matrix.indptr[uid]:train_matrix.indptr[uid + 1]]
        mean_history_pop[index] = float(np.log1p(item_count[ids]).mean()) if len(ids) else 0.0
    cutoffs = np.quantile(mean_history_pop, [1 / 3, 2 / 3])
    groups = np.searchsorted(cutoffs, mean_history_pop, side="right")
    rows = []
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        ids = torch.as_tensor(batch_users, device=device, dtype=torch.long)
        history = [as_item_set(test_data.uid2history_item[uid]) for uid in batch_users]
        positives = [as_item_set(test_data.uid2positive_item[uid]) for uid in batch_users]
        gcn_scores = gcn_user[ids] @ gcn_item[:n_items].T
        preference_scores = gcn_scores + user_pref[ids] @ item_pref[:n_items].T
        observed_scores = preference_scores * modulation[None, :]
        counterfactual_scores = (
            preference_scores - model.counterfactual_reference
        ) * counterfactual_modulation[None, :]
        variants = {
            "gcn": gcn_scores,
            "preference": preference_scores,
            "observed": observed_scores,
            "counterfactual": counterfactual_scores,
            "popularity_subtract": preference_scores - getattr(model, "inference_popularity_alpha", 0.0) * bias_score[None, :],
        }
        with torch.no_grad():
            for variant, original_scores in list(variants.items()):
                variants[variant] = original_scores.clone()
            for row, seen in enumerate(history):
                for scores in variants.values():
                    scores[row, 0] = -torch.inf
                    if seen:
                        scores[row, list(seen)] = -torch.inf
            base = variants["preference"]
            finite = torch.isfinite(base)
            count = finite.sum(1).clamp_min(1)
            mean = torch.where(finite, base, 0).sum(1) / count
            variance = torch.where(finite, (base - mean[:, None]).square(), 0).sum(1) / count
            scale = variance.sqrt().clamp_min(1e-6)
            for alpha in PENALTIES:
                score = base - alpha * scale[:, None] * pop_z[None, :]
                variants[f"penalty_{alpha:g}"] = score.masked_fill(~finite, -torch.inf)
            for variant, scores in variants.items():
                correlation = score_popularity_correlation(scores, log_popularity)
                top = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
                for row, uid in enumerate(batch_users):
                    recommended = top[row].tolist()
                    relevant = positives[row]
                    hits = [item in relevant for item in recommended]
                    ideal = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
                    tail_relevant = sum(not bool(head[item]) for item in relevant)
                    rows.append({
                        "user_id": uid, "user_group": int(groups[start + row]),
                        "history_mean_log_popularity": mean_history_pop[start + row],
                        "variant": variant, "arp": float(popularity[recommended].mean()),
                        "recall": sum(hits) / len(relevant) if relevant else 0.0,
                        "ndcg": sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits)) / ideal if ideal else 0.0,
                        "tail_hits": sum(hit and not bool(head[item]) for item, hit in zip(recommended, hits)),
                        "tail_relevant": tail_relevant,
                        "head_share": float(head[recommended].float().mean()),
                        "score_popularity_corr": float(correlation[row]),
                        "recommended_items": " ".join(map(str, recommended)),
                        "positive_mean_popularity": float(popularity[list(relevant)].mean()) if relevant else float("nan"),
                    })
    summary = []
    gcn_items = {
        row["user_id"]: set(map(int, row["recommended_items"].split()))
        for row in rows if row["variant"] == "gcn"
    }
    for row in rows:
        recommended = set(map(int, row["recommended_items"].split()))
        row["overlap_with_gcn"] = len(recommended & gcn_items[row["user_id"]]) / k
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant]
        unique_items = {int(item) for row in selected for item in row["recommended_items"].split()}
        summary.append({
            "variant": variant, "users": len(selected), "k": k,
            "arp": float(np.mean([row["arp"] for row in selected])),
            "recall": float(np.mean([row["recall"] for row in selected])),
            "ndcg": float(np.mean([row["ndcg"] for row in selected])),
            "tail_recall": sum(row["tail_hits"] for row in selected) / max(1, sum(row["tail_relevant"] for row in selected)),
            "coverage": len(unique_items) / max(1, n_items - 1),
            "head_share": float(np.mean([row["head_share"] for row in selected])),
            "overlap_with_gcn": float(np.mean([row["overlap_with_gcn"] for row in selected])),
            "score_popularity_corr": float(np.mean([row["score_popularity_corr"] for row in selected])),
            "positive_mean_popularity": float(np.nanmean([row["positive_mean_popularity"] for row in selected])),
        })
    return rows, summary


def summarize_user_groups(rows):
    output = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        for variant in dict.fromkeys(row["variant"] for row in rows if row["dataset"] == dataset):
            for group in range(3):
                selected = [row for row in rows if row["dataset"] == dataset
                            and row["variant"] == variant and row["user_group"] == group]
                if not selected:
                    continue
                output.append({
                    "dataset": dataset, "variant": variant, "user_group": group,
                    "users": len(selected),
                    "arp": float(np.mean([row["arp"] for row in selected])),
                    "recall": float(np.mean([row["recall"] for row in selected])),
                    "ndcg": float(np.mean([row["ndcg"] for row in selected])),
                    "head_share": float(np.mean([row["head_share"] for row in selected])),
                    "overlap_with_gcn": float(np.mean([row["overlap_with_gcn"] for row in selected])),
                })
    return output


def analyze_dataset(name, args, device):
    checkpoint_file = args.checkpoint_dir / f"{name}{args.checkpoint_suffix}"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing saved Phase 2 model: {checkpoint_file}")
    checkpoint = load_checkpoint(checkpoint_file)
    config = checkpoint["config"]
    config["device"] = device
    config["save_dataset"] = False
    config["save_dataloaders"] = False
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = Proposed(config, train_data.dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    item_count = np.bincount(
        train_data.dataset.inter_matrix(form="coo", domain="target").col,
        minlength=model.target_num_items,
    ).astype(np.float64)
    item_ids = torch.as_tensor(np.flatnonzero(item_count[1:] > 0) + 1, dtype=torch.long)
    # Both probes use the same item split and the exact training interaction count.
    target = torch.from_numpy(np.log1p(item_count[item_ids.numpy()]).astype(np.float32))
    with torch.inference_mode():
        values = model.forward()
        representations = {
            "gcn": model.target_z_item[item_ids],
            "preference": values["target_item_preference"][item_ids],
            "bias": values["target_item_bias"][item_ids],
        }
    probe_rows = []
    for component, features in representations.items():
        results, sizes = popularity_probe(features, target, args.seed, args.probe_epochs, device)
        for probe, metrics in results.items():
            probe_rows.append({"dataset": name, "component": component, "probe": probe,
                               "items": len(item_ids), "train_items": sizes[0],
                               "valid_items": sizes[1], "test_items": sizes[2], **metrics})
    users = np.asarray(test_data.uid_list, dtype=np.int64)
    if args.max_users and len(users) > args.max_users:
        rng = np.random.default_rng(args.seed)
        users = np.sort(rng.choice(users, args.max_users, replace=False))
    with torch.inference_mode():
        per_user, ranking = ranking_experiment(
            model, train_data, test_data, users, item_count, args.k, args.batch_users
        )
    for row in per_user:
        row["dataset"] = name
    for row in ranking:
        row["dataset"] = name
    return probe_rows, per_user, ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("saved"))
    parser.add_argument("--checkpoint-suffix", default="-PROPOSED-ADDITIVE.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("results/disentanglement"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--max-users", type=int, default=512,
                        help="Deterministic test-user sample; 0 evaluates all users")
    parser.add_argument("--batch-users", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    if args.k < 1 or args.batch_users < 1 or args.max_users < 0 or args.probe_epochs < 1:
        parser.error("k, batch-users and probe-epochs must be positive; max-users must be nonnegative")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_probes, all_users, all_ranking = [], [], []
    for name in args.datasets:
        print(f"Analyzing {name}", flush=True)
        probes, users, ranking = analyze_dataset(name, args, device)
        all_probes.extend(probes)
        all_users.extend(users)
        all_ranking.extend(ranking)
        save_csv(args.output_dir / "probe_results.csv", all_probes)
        save_csv(args.output_dir / "ranking_summary.csv", all_ranking)
        save_csv(args.output_dir / "per_user.csv", all_users)
        save_csv(args.output_dir / "user_group_summary.csv", summarize_user_groups(all_users))
        print(f"  users={len({row['user_id'] for row in users})} probe_items={probes[0]['items']}", flush=True)
    (args.output_dir / "settings.json").write_text(json.dumps({
        "datasets": args.datasets, "device": str(device), "max_users": args.max_users,
        "seed": args.seed, "k": args.k, "probe_epochs": args.probe_epochs,
        "penalty_definition": "preference_score - alpha * per_user_score_std * standardized_log1p_train_item_count",
        "probe_target": "log1p(target-domain training interaction count)",
    }, indent=2), encoding="utf-8")
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
