"""Inspect Selective-CL weights in a saved Proposed checkpoint.

Run with the same Python environment used for training. The checkpoint and
model parameters are read only; no training or model changes occur.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_proposed_disentanglement import load_checkpoint


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    points = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                    map(float, points))) | {"mean": float(values.mean()), "sd": float(values.std())}


def inspect(path, device, users_output=None):
    config, model, train, _valid, _test, source, _target = load_checkpoint(path, device)
    matrix = train.dataset.inter_matrix(form="csr", value_field=None, domain="source")
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    users = np.flatnonzero(counts[:model.overlapped_num_users] > 0)
    users = users[users > 0]
    if len(users) < 2:
        raise ValueError("Fewer than two active overlap users")

    risk_parts = []
    valid_parts = []
    with torch.no_grad():
        for part in np.array_split(users, max(1, (len(users) + 4095) // 4096)):
            ids = torch.as_tensor(part, device=device, dtype=torch.long)
            popularity = torch.sigmoid(model.src_user_pop[ids])
            if model.beta < 1:
                mm, valid = model._source_mm(ids)
                pref = torch.as_tensor(source[0][part], device=device)
                semantic = ((1 - F.cosine_similarity(pref, mm, dim=-1)) / 2).clamp(0, 1)
                semantic = torch.where(valid, semantic, torch.full_like(semantic, .5))
            else:
                valid = torch.zeros_like(popularity, dtype=torch.bool)
                semantic = torch.zeros_like(popularity)
            risk_parts.append(torch.stack((popularity, semantic), dim=1).cpu().numpy())
            valid_parts.append(valid.cpu().numpy())
    parts = np.concatenate(risk_parts)
    valid = np.concatenate(valid_parts)
    popularity, semantic = parts.T
    alpha = np.clip(model.beta * popularity + (1 - model.beta) * semantic, 0, 1)
    weights = np.exp(-model.gamma * alpha)
    relative = weights / weights.mean()  # Same scale as a uniform-CL batch mean.
    ess_ratio = float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum()))
    if users_output is not None:
        users_output.parent.mkdir(parents=True, exist_ok=True)
        with users_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("user_id", "source_train_count", "mm_valid", "popularity_risk",
                             "semantic_risk", "combined_risk", "raw_weight", "relative_weight"))
            for values in zip(users, counts[users], valid, popularity, semantic,
                              alpha, weights, relative):
                writer.writerow((int(values[0]), int(values[1]), int(values[2]),
                                 *(float(value) for value in values[3:])))
    result = {
        "checkpoint": str(path), "direction": config["source_domain"]["dataset"],
        "seed": int(config["seed"]), "gamma": float(model.gamma),
        "beta": float(model.beta), "lambda_cl": float(model.lambda_cl),
        "population": "Active source-train overlap users, counted once per user",
        "n_users": int(len(users)), "mm_valid_fraction": float(valid.mean()),
        "popularity_risk": describe(popularity), "semantic_risk": describe(semantic),
        "combined_risk": describe(alpha), "raw_weight": describe(weights),
        "relative_weight": describe(relative),
        "weight_cv": float(weights.std() / weights.mean()),
        "effective_sample_size_fraction": ess_ratio,
        "relative_weight_bins": {
            f"[{lo:g},{hi:g})": int(((relative >= lo) & (relative < hi)).sum())
            for lo, hi in zip([0, .25, .5, .75, 1, 1.25, 1.5, 2, 3],
                              [.25, .5, .75, 1, 1.25, 1.5, 2, 3, np.inf])
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--users-output", type=Path,
                        help="Write one row per active overlap user, with their CL weight")
    args = parser.parse_args()
    result = inspect(args.checkpoint, args.device, args.users_output)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
