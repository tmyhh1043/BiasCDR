"""Analyze saved Proposed checkpoints without updating the recommender.

Popularity is an observational proxy. Probe splits only hold entities out of
the probe fit; the recommender itself saw their training interactions.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from recbole.utils import init_seed
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Incomplete checkpoint: {path}")
    config = checkpoint["config"]
    if config["model"].upper() != "PROPOSED":
        raise ValueError(f"Expected Proposed checkpoint, got {config['model']}")
    config["device"] = torch.device(device)
    config["use_gpu"] = device.startswith("cuda")
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data.dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.init_restore_e()
    del checkpoint
    model.eval()
    with torch.no_grad():
        source, target = model.forward()
    # Return CPU arrays so diagnostics do not keep four full embedding sets on GPU.
    source = tuple(x.detach().cpu().numpy() for x in source)
    target = tuple(x.detach().cpu().numpy() for x in target)
    return config, model, train_data, valid_data, test_data, source, target


def pearson(y, prediction):
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(prediction) < 1e-12:
        return None
    return float(np.corrcoef(y, prediction)[0, 1])


def probe_metrics(y, prediction):
    residual = np.mean((y - prediction) ** 2)
    variance = np.mean((y - np.mean(y)) ** 2)
    return {"r2": float(1 - residual / variance) if variance > 1e-12 else None,
            "mae": float(np.mean(np.abs(y - prediction))),
            "pearson": pearson(y, prediction)}


def ridge_probe(features, labels, eligible, seed):
    """Fit a linear probe on 60%, tune alpha on 20%, report held-out 20%."""
    entity_ids = np.flatnonzero(eligible)
    if len(entity_ids) < 10:
        return {"count": int(len(entity_ids)), "error": "fewer than 10 eligible entities"}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entity_ids))
    n_train = max(1, int(.6 * len(order)))
    n_valid = max(1, int(.2 * len(order)))
    train, valid, test = order[:n_train], order[n_train:n_train + n_valid], order[n_train + n_valid:]
    x = np.asarray(features[entity_ids], dtype=np.float64)
    y = np.asarray(labels[entity_ids], dtype=np.float64)
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-8] = 1
    x = np.clip((x - mean) / scale, -10, 10)
    y_mean = y[train].mean()
    # Dual ridge is wasteful for large catalogs; solve in feature dimension.
    xtx = x[train].T @ x[train]
    xty = x[train].T @ (y[train] - y_mean)
    best = None
    for alpha in (0.1, 1., 10., 100., 1000.):
        weights = np.linalg.solve(xtx + alpha * np.eye(x.shape[1]), xty)
        valid_mse = np.mean((y[valid] - (x[valid] @ weights + y_mean)) ** 2)
        if best is None or valid_mse < best[0]:
            best = valid_mse, alpha, weights
    prediction = x[test] @ best[2] + y_mean
    return {"count": int(len(entity_ids)), "train_count": len(train),
            "valid_count": len(valid), "test_count": len(test),
            "alpha": best[1], "probe": probe_metrics(y[test], prediction),
            "mean_baseline": probe_metrics(y[test], np.full(len(test), y_mean))}


def orthogonality(preference, bias, eligible):
    a = np.asarray(preference[eligible], dtype=np.float64)
    b = np.asarray(bias[eligible], dtype=np.float64)
    if len(a) == 0:
        return None
    cosine = (a * b).sum(axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return {"mean_abs_cosine": float(np.mean(np.abs(cosine))),
            "mean_squared_cosine": float(np.mean(cosine ** 2))}


def domain_probes(embeddings, interaction_matrix, seed):
    pref_user, pref_item, bias_user, bias_item = embeddings
    matrix = interaction_matrix.tocsr()
    degree = np.asarray(matrix.sum(axis=0)).ravel()
    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    log_pop = np.log1p(degree)
    mean_pop = np.asarray(matrix @ log_pop).ravel() / np.maximum(user_degree, 1)
    items = (degree > 0)
    items[0] = False
    users = user_degree > 0
    users[0] = False
    return {
        "item_popularity": {
            "preference": ridge_probe(pref_item, log_pop, items, seed),
            "bias": ridge_probe(bias_item, log_pop, items, seed),
            "preference_norm": ridge_probe(np.linalg.norm(pref_item, axis=1)[:, None],
                                           log_pop, items, seed),
            "bias_norm": ridge_probe(np.linalg.norm(bias_item, axis=1)[:, None],
                                     log_pop, items, seed),
        },
        "user_consumed_popularity": {
            "preference": ridge_probe(pref_user, mean_pop, users, seed + 1),
            "bias": ridge_probe(bias_user, mean_pop, users, seed + 1),
            "preference_norm": ridge_probe(np.linalg.norm(pref_user, axis=1)[:, None],
                                           mean_pop, users, seed + 1),
            "bias_norm": ridge_probe(np.linalg.norm(bias_user, axis=1)[:, None],
                                     mean_pop, users, seed + 1),
        },
        "item_orthogonality": orthogonality(pref_item, bias_item, items),
        "user_orthogonality": orthogonality(pref_user, bias_user, users),
    }


def evaluate_ranking(embeddings, loader, popularity, k, seed, max_users):
    pref_user, pref_item, bias_user, bias_item = embeddings
    num_items = len(popularity)
    k = min(k, num_items - 1)
    if k < 1:
        raise ValueError("Target catalog is empty")
    ids = loader.uid_list.detach().cpu().numpy()
    if max_users and len(ids) > max_users:
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_users, replace=False))
    # Popularity bins are formed from train interactions only. Zero-degree items
    # get their own bin; comparisons never cross popularity bins.
    bins = np.zeros(num_items, dtype=np.int8)
    warm = np.flatnonzero(popularity > 0)
    warm = warm[warm > 0]
    if len(warm):
        boundaries = np.quantile(popularity[warm], [.25, .5, .75])
        bins[warm] = 1 + np.searchsorted(boundaries, popularity[warm], side="right")
    members = [np.flatnonzero(bins == b) for b in range(5)]
    members[0] = members[0][members[0] > 0]
    rng = np.random.default_rng(seed)
    totals = {mode: {"recall_hits": 0, "ndcg_sum": 0., "matched_wins": 0,
                     "matched_pairs": 0} for mode in ("preference", "bias_dot", "popularity")}
    positive_count = user_count = 0
    for uid in ids:
        positives = loader.uid2positive_item[int(uid)]
        if positives is None or len(positives) == 0:
            continue
        pos = positives.detach().cpu().numpy()
        pos = pos[(pos > 0) & (pos < num_items)]
        if len(pos) == 0:
            continue
        history = loader.uid2history_item[int(uid)]
        excluded = set(pos.tolist())
        if history is not None:
            excluded.update(history.detach().cpu().numpy().tolist())
        user_count += 1
        positive_count += len(pos)
        scores = {
            "preference": pref_item[:num_items] @ pref_user[int(uid)],
            "bias_dot": bias_item[:num_items] @ bias_user[int(uid)],
            "popularity": np.log1p(popularity).astype(np.float64),
        }
        matched = []
        for item in pos:
            candidates = members[int(bins[item])]
            # No test-label or history item is used as a sampled negative.
            for _ in range(30):
                if len(candidates) == 0:
                    break
                negative = int(rng.choice(candidates))
                if negative not in excluded:
                    matched.append((item, negative))
                    break
        for mode, raw in scores.items():
            masked = raw.copy()
            masked[0] = -np.inf
            if history is not None:
                masked[history.detach().cpu().numpy()] = -np.inf
            candidates = np.argpartition(-masked, kth=k - 1)[:k]
            top = candidates[np.argsort(-masked[candidates], kind="stable")]
            hits = np.isin(top, pos)
            totals[mode]["recall_hits"] += int(hits.sum())
            ideal = np.sum(1 / np.log2(np.arange(2, min(len(pos), k) + 2)))
            totals[mode]["ndcg_sum"] += float(np.sum(hits / np.log2(np.arange(2, k + 2))) / ideal)
            totals[mode]["matched_wins"] += sum(float(raw[p] > raw[n]) +
                                                 .5 * float(raw[p] == raw[n]) for p, n in matched)
            totals[mode]["matched_pairs"] += len(matched)
    result = {}
    for mode, value in totals.items():
        result[mode] = {
            f"recall@{k}": value["recall_hits"] / positive_count if positive_count else None,
            f"ndcg@{k}": value["ndcg_sum"] / user_count if user_count else None,
            "popularity_matched_pair_accuracy": value["matched_wins"] / value["matched_pairs"]
            if value["matched_pairs"] else None,
            "matched_pairs": value["matched_pairs"],
        }
    return {"users": user_count, "positives": positive_count, "scores": result}


def flatten(record, prefix=""):
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/proposed_disentanglement"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=0,
                        help="0 evaluates all users; positive values sample users reproducibly")
    args = parser.parse_args()
    if args.topk < 1 or args.max_users < 0:
        parser.error("--topk must be positive and --max-users must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.checkpoints:
        print(f"Analyzing {path}", flush=True)
        config, model, train, valid, test, source, target = load_checkpoint(path, args.device)
        train_dataset = train.dataset
        source_mat = train_dataset.inter_matrix(form="csr", value_field=None, domain="source")
        target_mat = train_dataset.inter_matrix(form="csr", value_field=None, domain="target")
        target_pop = np.asarray(target_mat.sum(axis=0)).ravel()[:model.target_num_items]
        eval_loader = valid[-1] if args.split == "valid" and isinstance(valid, (tuple, list)) else (
            valid if args.split == "valid" else test)
        seed = int(config["seed"])
        report = {
            "checkpoint": str(path.resolve()),
            "variant": path.parent.name,
            "dataset": str(config["dataset"]),
            "seed": seed,
            "split": args.split,
            "loss_weights": {key: float(config[key]) for key in
                             ("lambda_cl", "lambda_pop", "lambda_orth", "lambda_multi")},
            "interpretation": "Popularity is an observed training statistic; matched unobserved items are not verified negatives.",
            "score_interpretation": "bias_dot is diagnostic only: its user-item dot product was not trained as a ranking score.",
            "source": domain_probes(source, source_mat, seed),
            "target": domain_probes(target, target_mat, seed),
            "ranking": evaluate_ranking(target, eval_loader, target_pop, args.topk, seed, args.max_users),
        }
        output = args.output_dir / f"{path.parent.name}__{path.stem}__{args.split}.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(dict(flatten(report)))
        print(f"Wrote {output}", flush=True)
        del model, source, target, train, valid, test, config, train_dataset
        del source_mat, target_mat, target_pop, eval_loader, report
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if rows:
        csv_path = args.output_dir / f"summary_{args.split}.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
