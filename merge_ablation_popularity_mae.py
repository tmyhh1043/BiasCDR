"""Add saved-checkpoint popularity MAE to the five-seed ablation workbook."""

import argparse
import csv
import gc
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from recbole.utils import init_seed

from analyze_proposed_disentanglement import load_checkpoint
from evaluate_proposed_popularity_mae import (
    evaluate_linear_probe,
    target_train_popularity_labels,
)
from recbole_cdr.utils import get_model


CSV_FIELDS = (
    "direction", "seed", "experiment_tag", "n_items", "probe_train_items",
    "probe_test_items", "baseline_mae", "preference_mae", "bias_mae",
)
DATA_CONFIG_KEYS = (
    "source_domain", "target_domain", "eval_args", "source_split", "reproducibility",
)
SUMMARY_METRICS = (
    "recall@20", "ndcg@20", "arp@20", "coverage@20",
    "preference_mae", "bias_mae", "baseline_mae",
)
TRANSFER_LABELS = {
    "AmazonPhone_AmazonSport": "Phone→Sports",
    "AmazonSport_AmazonPhone": "Sports→Phone",
    "AmazonToy_AmazonSport": "Toy→Sports",
    "AmazonSport_AmazonToy": "Sports→Toy",
}


def checkpoint_path(root, direction, seed, variant):
    return root / variant / f"{direction}-PROPOSED-ADDITIVE-seed{seed}.pth"


def load_other_streams(path, device, train_dataset, reference_config, target_count):
    """Reuse the identical data split while restoring each variant's own weights."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    for key in DATA_CONFIG_KEYS:
        if config[key] != reference_config[key]:
            raise ValueError(f"Dataset configuration differs for {path}: {key}")
    if config["seed"] != reference_config["seed"] or config["model"].upper() != "PROPOSED":
        raise ValueError(f"Seed or model differs for {path}")
    config["device"] = torch.device(device)
    config["use_gpu"] = device.startswith("cuda")
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.init_restore_e()
    model.eval()
    del checkpoint
    with torch.no_grad():
        _source, target = model.forward()
    preference = target[1][:target_count].detach().cpu().numpy().copy()
    bias = target[3][:target_count].detach().cpu().numpy().copy()
    if preference.shape != bias.shape or not (np.isfinite(preference).all() and np.isfinite(bias).all()):
        raise ValueError(f"Invalid item streams in {path}")
    del model, target, _source
    return preference, bias


def save_progress(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_all(rows_by_key, checkpoint_root, progress_path, device):
    completed = {}
    if progress_path.exists():
        with progress_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = (row["direction"], int(row["seed"]), row["experiment_tag"])
                completed[key] = row
        if not set(completed).issubset(rows_by_key):
            raise ValueError("Progress CSV contains rows absent from the source workbook")
        print(f"Resuming {len(completed)} completed probes", flush=True)

    groups = defaultdict(list)
    for direction, seed, variant in rows_by_key:
        groups[(direction, seed)].append(variant)

    for group_index, ((direction, seed), variants) in enumerate(groups.items(), 1):
        if all((direction, seed, v) in completed for v in variants):
            continue
        full_path = checkpoint_path(checkpoint_root, direction, seed, "full")
        config, model, train, _valid, _test, _source, target = load_checkpoint(full_path, device)
        if config["seed"] != seed or config["source_domain"]["dataset"] != direction:
            raise ValueError(f"Checkpoint direction or seed mismatch: {full_path}")
        target_count = model.target_num_items
        labels, counts, eligible = target_train_popularity_labels(train.dataset, target_count)
        baseline_reference = None
        if (direction, seed, "full") not in completed:
            pref = target[1][:target_count].copy()
            bias = target[3][:target_count].copy()
            train_count, test_count, mae, baseline_reference = evaluate_linear_probe(
                pref, bias, labels, eligible
            )
            completed[(direction, seed, "full")] = {
                "direction": direction, "seed": seed, "experiment_tag": "full",
                "n_items": int(eligible.sum()), "probe_train_items": train_count,
                "probe_test_items": test_count, "baseline_mae": float(baseline_reference),
                "preference_mae": float(mae["preference"]), "bias_mae": float(mae["bias"]),
            }
            save_progress(progress_path, completed.values())
        else:
            baseline_reference = float(completed[(direction, seed, "full")]["baseline_mae"])
        del model, target, _source, _valid, _test, counts
        gc.collect()

        for variant in variants:
            key = (direction, seed, variant)
            if key in completed:
                continue
            path = checkpoint_path(checkpoint_root, direction, seed, variant)
            if not path.is_file():
                raise FileNotFoundError(path)
            pref, bias = load_other_streams(path, device, train.dataset, config, target_count)
            train_count, test_count, mae, baseline = evaluate_linear_probe(
                pref, bias, labels, eligible
            )
            if not np.isclose(baseline, baseline_reference, rtol=0, atol=1e-7):
                raise ValueError(f"Probe split or popularity labels changed for {path}")
            completed[key] = {
                "direction": direction, "seed": seed, "experiment_tag": variant,
                "n_items": int(eligible.sum()), "probe_train_items": train_count,
                "probe_test_items": test_count, "baseline_mae": float(baseline),
                "preference_mae": float(mae["preference"]), "bias_mae": float(mae["bias"]),
            }
            save_progress(progress_path, completed.values())
            del pref, bias
            gc.collect()
        print(f"[{group_index}/{len(groups)}] {direction} seed={seed}: {len(variants)} variants complete", flush=True)
        del train, labels, eligible, config
        gc.collect()
    if set(completed) != set(rows_by_key):
        raise ValueError(f"MAE results cover {len(completed)}/{len(rows_by_key)} workbook rows")
    return completed


def add_summary_sheet(workbook, original_rows, mae_by_key):
    summary = workbook.create_sheet("summary_5seeds")
    columns = ["transfer", "direction", "experiment_tag", "n_seeds"]
    for metric in SUMMARY_METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_sd"))
    summary.append(columns)

    groups = defaultdict(list)
    for row in original_rows:
        direction, seed, variant = row["dataset"].split("->")[0], int(row["seed"]), row["experiment_tag"]
        mae = mae_by_key[(direction, seed, variant)]
        record = {metric: float(row[metric]) for metric in SUMMARY_METRICS[:4]}
        record.update({metric: float(mae[metric]) for metric in SUMMARY_METRICS[4:]})
        groups[(direction, variant)].append(record)

    for (direction, variant), records in groups.items():
        if len(records) != 5:
            raise ValueError(f"Expected five seeds for {direction}/{variant}, got {len(records)}")
        output = [TRANSFER_LABELS[direction], direction, variant, len(records)]
        for metric in SUMMARY_METRICS:
            values = [record[metric] for record in records]
            output.extend((statistics.mean(values), statistics.stdev(values)))
        summary.append(output)
    summary.freeze_panes = "E2"
    summary.auto_filter.ref = summary.dimensions
    for cell in summary[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="234562")
    summary.column_dimensions["A"].width = 18
    summary.column_dimensions["B"].width = 27
    summary.column_dimensions["C"].width = 23
    summary.column_dimensions["D"].width = 10
    for column in range(5, len(columns) + 1):
        summary.column_dimensions[get_column_letter(column)].width = 18
    for row in summary.iter_rows(min_row=2, min_col=5):
        for cell in row:
            cell.number_format = "0.000000"

    notes = workbook.create_sheet("MAE_method")
    notes.append(["Item", "Definition"])
    for item, value in (
        ("Model seeds", "2022, 2023, 2024, 2025, 2026; each checkpoint is evaluated separately"),
        ("Probe split", "80% train / 20% test over eligible target train items; random_state=2022 for every model"),
        ("Popularity label", "Train-only target interaction count: standardized log1p(count), matching Proposed._popularity_targets"),
        ("Features", "Target item final LightGCN mean embeddings; Preference and Bias evaluated independently"),
        ("Probe", "StandardScaler fitted on probe-train items, then sklearn LinearRegression"),
        ("Baseline MAE", "Predict probe-train mean popularity on probe-test items"),
        ("MAE interpretation", "Lower MAE means popularity is more linearly predictable from that stream"),
        ("BPR-only Bias", "Bias stream receives no BPR gradient; its MAE is a control, not a learned-bias comparison"),
        ("Summary SD", "Sample standard deviation across five model seeds (ddof=1)"),
    ):
        notes.append([item, value])
    notes.column_dimensions["A"].width = 25
    notes.column_dimensions["B"].width = 105
    for cell in notes[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="234562")


def build_workbook(source_path, output_path, mae_by_key):
    workbook = load_workbook(source_path)
    sheet = workbook["test_metrics"]
    headers = [cell.value for cell in sheet[1]]
    if len(headers) != len(set(headers)) or not {"dataset", "seed", "experiment_tag"}.issubset(headers):
        raise ValueError("Unexpected source workbook header")
    original_rows = []
    extra = list(CSV_FIELDS[3:])
    for column, name in enumerate(extra, len(headers) + 1):
        sheet.cell(1, column, name)
        sheet.cell(1, column).font = Font(bold=True, color="FFFFFF")
        sheet.cell(1, column).fill = PatternFill("solid", fgColor="234562")
    for row_index in range(2, sheet.max_row + 1):
        row = {header: sheet.cell(row_index, column).value for column, header in enumerate(headers, 1)}
        original_rows.append(row)
        direction = row["dataset"].split("->")[0]
        key = (direction, int(row["seed"]), row["experiment_tag"])
        if key not in mae_by_key:
            raise ValueError(f"Missing MAE for {key}")
        for column, name in enumerate(extra, len(headers) + 1):
            cell = sheet.cell(row_index, column, float(mae_by_key[key][name]))
            cell.number_format = "0.000000" if name.endswith("mae") else "0"
    sheet.freeze_panes = "E2"
    sheet.auto_filter.ref = sheet.dimensions
    add_summary_sheet(workbook, original_rows, mae_by_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/student/Downloads/アブレーション実験.xlsx"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("saved/proposed_ablation"))
    parser.add_argument("--progress", type=Path, default=Path("results/ablation_popularity_mae_5seeds.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/ablation_with_popularity_mae_5seeds.xlsx"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    workbook = load_workbook(args.source, read_only=True, data_only=True)
    sheet = workbook["test_metrics"]
    values = sheet.values
    headers = next(values)
    rows_by_key = {}
    for values in values:
        row = dict(zip(headers, values))
        direction, target = row["dataset"].split("->")
        seed, variant = int(row["seed"]), row["experiment_tag"]
        key = (direction, seed, variant)
        if key in rows_by_key:
            raise ValueError(f"Duplicate workbook row: {key}")
        path = checkpoint_path(args.checkpoint_root, direction, seed, variant)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows_by_key[key] = target
    workbook.close()
    print(f"Workbook rows/checkpoints: {len(rows_by_key)}", flush=True)
    warnings.filterwarnings("ignore", category=FutureWarning, module="recbole")
    mae_by_key = evaluate_all(rows_by_key, args.checkpoint_root, args.progress, args.device)
    build_workbook(args.source, args.output, mae_by_key)
    print(f"Merged workbook: {args.output}", flush=True)


if __name__ == "__main__":
    main()
