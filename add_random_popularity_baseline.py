"""Compare saved popularity probes with a frozen random-vector control.

The random item vectors are sampled once per repetition and never optimized.
Only the same StandardScaler + LinearRegression probe used for the learned
streams is fitted. Each direction/seed shares one control across variants.
"""

import argparse
import csv
import gc
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_proposed_popularity_mae import (
    load_target_item_embeddings,
    target_train_popularity_labels,
)


EXTRA_FIELDS = ("random_repeats", "random_mae_mean", "random_mae_sd",
                "preference_minus_random", "bias_minus_random")


def random_probe_maes(labels, eligible, embedding_dim, repeats):
    """Use the exact item split and probe family of evaluate_linear_probe."""
    item_ids = np.flatnonzero(eligible)
    train_ids, test_ids = train_test_split(
        item_ids, test_size=0.2, random_state=2022, shuffle=True
    )
    scores = []
    for repetition in range(repeats):
        rng = np.random.default_rng(10000 + repetition)
        # Assign each eligible item a fixed vector for this probe fit.
        vectors = rng.standard_normal((len(item_ids), embedding_dim)).astype(np.float32)
        position = np.full(len(labels), -1, dtype=np.int64)
        position[item_ids] = np.arange(len(item_ids))
        probe = make_pipeline(StandardScaler(), LinearRegression())
        probe.fit(vectors[position[train_ids]], labels[train_ids])
        prediction = probe.predict(vectors[position[test_ids]])
        scores.append(float(mean_absolute_error(labels[test_ids], prediction)))
    return scores, train_ids, test_ids


def save_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/ablation_popularity_mae_5seeds.csv"))
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("saved/proposed_ablation"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/ablation_popularity_mae_with_random_5seeds.csv"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("results/random_popularity_baseline_by_direction.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to report variability")
    if len({args.source.resolve(), args.output.resolve(), args.summary_output.resolve()}) != 3:
        parser.error("--source, --output, and --summary-output must be distinct")

    with args.source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        source_fields = reader.fieldnames
        source_rows = list(reader)
    if not source_fields or not {"direction", "seed", "experiment_tag", "n_items",
                                  "baseline_mae", "preference_mae", "bias_mae"}.issubset(source_fields):
        raise ValueError("Source CSV lacks required popularity probe columns")
    if any(field in source_fields for field in EXTRA_FIELDS):
        raise ValueError("Source CSV already has random-baseline columns")

    groups = defaultdict(list)
    for row in source_rows:
        groups[(row["direction"], int(row["seed"]))].append(row)
    checkpoint_groups = set()
    pattern = re.compile(r"(.+)-PROPOSED-ADDITIVE-seed(\d+)\.pth$")
    for path in (args.checkpoint_root / "full").glob("*-PROPOSED-ADDITIVE-seed*.pth"):
        match = pattern.fullmatch(path.name)
        if match:
            checkpoint_groups.add((match.group(1), int(match.group(2))))
    if not checkpoint_groups:
        raise ValueError("No Full Model checkpoints found")
    if set(groups) != checkpoint_groups:
        missing = sorted(checkpoint_groups - set(groups))
        extra = sorted(set(groups) - checkpoint_groups)
        raise ValueError(f"Source CSV and Full Model checkpoints differ; missing={missing}, extra={extra}")
    completed = {}
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != source_fields + list(EXTRA_FIELDS):
                raise ValueError("Existing output has a different column layout")
            for row in reader:
                key = (row["direction"], int(row["seed"]), row["experiment_tag"])
                if int(row["random_repeats"]) != args.repeats:
                    raise ValueError("Existing output used a different --repeats value")
                completed[key] = row
        expected = {(row["direction"], int(row["seed"]), row["experiment_tag"]): row
                    for row in source_rows}
        for key, row in completed.items():
            if key not in expected or any(row[field] != expected[key][field]
                                          for field in source_fields):
                raise ValueError(f"Existing output differs from source CSV: {key}")

    for (direction, seed), rows in groups.items():
        keys = [(direction, seed, row["experiment_tag"]) for row in rows]
        if all(key in completed for key in keys):
            continue
        if not any(row["experiment_tag"] == "full" for row in rows):
            raise ValueError(f"Full-model row missing for {direction}, seed {seed}")
        path = args.checkpoint_root / "full" / f"{direction}-PROPOSED-ADDITIVE-seed{seed}.pth"
        config, dataset, preference, bias = load_target_item_embeddings(path, args.device)
        if int(config["seed"]) != seed or config["source_domain"]["dataset"] != direction:
            raise ValueError(f"Checkpoint identity differs: {path}")
        labels, _counts, eligible = target_train_popularity_labels(dataset, len(preference))
        scores, train_ids, test_ids = random_probe_maes(
            labels, eligible, preference.shape[1], args.repeats
        )
        random_mean = float(np.mean(scores))
        random_sd = float(np.std(scores, ddof=1))
        baseline = float(mean_absolute_error(
            labels[test_ids], np.full(len(test_ids), labels[train_ids].mean())
        ))
        for row, key in zip(rows, keys):
            if int(row["n_items"]) != int(eligible.sum()) or not np.isclose(
                float(row["baseline_mae"]), baseline, rtol=0, atol=1e-6
            ):
                raise ValueError(f"Item set or split differs from saved result: {key}")
            completed[key] = {**row,
                              "random_repeats": args.repeats,
                              "random_mae_mean": random_mean,
                              "random_mae_sd": random_sd,
                              "preference_minus_random": float(row["preference_mae"]) - random_mean,
                              "bias_minus_random": float(row["bias_mae"]) - random_mean}
        save_rows(args.output,
                  [completed[(row["direction"], int(row["seed"]), row["experiment_tag"])]
                   for row in source_rows if (row["direction"], int(row["seed"]), row["experiment_tag"]) in completed],
                  source_fields + list(EXTRA_FIELDS))
        print(f"{direction} seed={seed}: random MAE {random_mean:.6f} ± {random_sd:.6f}; "
              f"mean-prediction MAE {baseline:.6f}", flush=True)
        del config, dataset, preference, bias, labels, _counts, eligible
        gc.collect()

    summary_fields = ("direction", "n_seeds", "mean_prediction_mae",
                      "random_mae_mean", "random_mae_sd_across_seeds",
                      "preference_mae", "bias_mae")
    summary = []
    for direction in sorted({direction for direction, _seed in groups}):
        full = [completed[(direction, seed, "full")]
                for group_direction, seed in groups if group_direction == direction]
        random_values = [float(row["random_mae_mean"]) for row in full]
        summary.append({
            "direction": direction,
            "n_seeds": len(full),
            "mean_prediction_mae": float(np.mean([float(row["baseline_mae"]) for row in full])),
            "random_mae_mean": float(np.mean(random_values)),
            "random_mae_sd_across_seeds": float(np.std(random_values, ddof=1)) if len(full) > 1 else 0.0,
            "preference_mae": float(np.mean([float(row["preference_mae"]) for row in full])),
            "bias_mae": float(np.mean([float(row["bias_mae"]) for row in full])),
        })
    save_rows(args.summary_output, summary, summary_fields)
    print(f"Saved {len(completed)} rows across {len(summary)} directions to {args.output}", flush=True)
    print(f"Direction summary: {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
