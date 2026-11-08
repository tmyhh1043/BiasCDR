"""Run controlled Proposed ablations from saved Phase 1 checkpoints."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = (
    "AmazonSport_AmazonPhone",
    "AmazonPhone_AmazonSport",
    "AmazonSport_AmazonToy",
    "AmazonToy_AmazonSport",
)
VARIANTS = {
    "mixed_constant": {},
    "observation_only": {"lambda_preference_rank": 0.0},
    "uniform_preference": {"preference_negative_mode": "uniform"},
    "short_distillation": {
        "distill_schedule": "linear", "distill_epochs": 5
    },
}


def run(command, environment):
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stopping-step", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("results/additive_ablations"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("saved/additive_ablations"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.epochs < 4 or args.stopping_step < 1:
        parser.error("epochs must be at least 4 and stopping-step positive")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    rows = []
    for dataset in args.datasets:
        phase1 = Path("saved/phase1") / f"{dataset}.pth"
        if not phase1.is_file():
            raise FileNotFoundError(phase1)
        for variant in args.variants:
            checkpoint_dir = args.checkpoint_root / dataset / variant
            checkpoint = checkpoint_dir / f"{dataset}-PROPOSED-ADDITIVE.pth"
            output_dir = args.output_root / dataset / variant
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            validation_csv = output_dir / "validation_by_epoch.csv"
            report_json = output_dir / "alpha_selection.json"
            if not (args.skip_existing and checkpoint.is_file()):
                overrides = {
                    "pretrained_model_file": str(phase1),
                    "checkpoint_dir": str(checkpoint_dir),
                    "train_epochs": [f"BOTH:{args.epochs}"],
                    "stopping_step": args.stopping_step,
                    "valid_start_epoch": 4,
                    "record_valid_popularity_each_epoch": True,
                    "valid_epoch_metrics_csv_path": str(validation_csv),
                    "diagnostic_variant": variant,
                    "skip_test_evaluation": True,
                    "test_extra_metrics": False,
                    "show_progress": False,
                    **VARIANTS[variant],
                }
                command = [
                    sys.executable, "run_recbole_cdr.py", "--model", "Proposed",
                    "--dataset", dataset, "--gpu_id", str(args.gpu_id),
                ]
                for key, value in overrides.items():
                    command.extend(["--" + key, json.dumps(value)])
                run(command, environment)
            run([
                sys.executable, "select_additive_alpha.py",
                "--checkpoint", str(checkpoint),
                "--device", "cuda:0",
                "--batch-users", "64",
                "--output", str(report_json),
            ], environment)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            selected = report["selected"]
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "selected": selected,
                **{
                    "baseline_valid_" + key: value
                    for key, value in report["validation"]["baseline"].items()
                    if key != "users"
                },
                **{
                    "baseline_test_" + key: value
                    for key, value in report["test"]["baseline"].items()
                    if key != "users"
                },
                **{
                    "valid_" + key: value
                    for key, value in report["validation"][selected].items()
                    if key != "users"
                },
                **{
                    "test_" + key: value
                    for key, value in report["test"][selected].items()
                    if key != "users"
                },
                **report["diagnostics"],
            })
            summary = args.output_root / "summary.csv"
            summary.parent.mkdir(parents=True, exist_ok=True)
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print("Saved", summary, flush=True)


if __name__ == "__main__":
    main()
