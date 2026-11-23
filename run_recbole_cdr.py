"""Run a RecBole-CDR experiment with optional configuration overrides."""

import argparse
import os
import sys

import yaml


def _value(text):
    return None if text.lower() == "none" else yaml.safe_load(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="DGCDR")
    parser.add_argument("--dataset", "--datasets", "-d", default="AmazonSport_AmazonPhone")
    parser.add_argument("--config_files", help="comma-separated YAML files")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--popularity_only_cl", action="store_true",
                        help="Train Proposed with popularity-only Selective-CL weights "
                             "(beta=1, gamma=2 by default) in separate output paths")
    parser.add_argument("--config_overrides", action="append", default=[],
                        help="comma-separated KEY=VALUE settings")
    args, extra = parser.parse_known_args()

    overrides = {}
    for group in args.config_overrides:
        for entry in group.split(","):
            if entry.strip():
                key, separator, value = entry.partition("=")
                if not separator or not key.strip():
                    parser.error(f"Invalid override: {entry!r}; expected KEY=VALUE")
                overrides[key.strip()] = _value(value.strip())

    # Accept existing model-specific options without one argparse declaration each.
    while extra:
        option = extra.pop(0)
        if not option.startswith("--"):
            parser.error(f"Unexpected argument: {option}")
        if "=" in option:
            key, value = option[2:].split("=", 1)
        else:
            if not extra:
                parser.error(f"Missing value for {option}")
            key, value = option[2:], extra.pop(0)
        if not key:
            parser.error(f"Invalid option: {option}")
        overrides[key] = _value(value)

    if isinstance(overrides.get("metrics_csv_param_keys"), str):
        overrides["metrics_csv_param_keys"] = [
            key.strip() for key in overrides["metrics_csv_param_keys"].split(",")
            if key.strip()
        ]

    if args.popularity_only_cl:
        if args.model != "Proposed":
            parser.error("--popularity_only_cl requires --model Proposed")
        if "beta" in overrides and overrides["beta"] != 1:
            parser.error("--popularity_only_cl requires beta=1")
        if "lambda_cl" in overrides and (overrides["lambda_cl"] is None or
                                          overrides["lambda_cl"] <= 0):
            parser.error("--popularity_only_cl requires lambda_cl > 0")
        overrides["beta"] = 1.0
        overrides.setdefault("gamma", 2.0)
        overrides.setdefault("experiment_tag", "popularity_only_cl")
        overrides.setdefault("checkpoint_dir", "saved/proposed_popularity_only_cl")
        overrides.setdefault("metrics_csv_path", "results/proposed_popularity_only_cl/test_metrics.csv")
        overrides.setdefault("metrics_csv_include_params", True)
        os.makedirs(overrides["checkpoint_dir"], exist_ok=True)

    if args.config_files:
        files = [path.strip() for path in args.config_files.split(",") if path.strip()]
    else:
        dataset_file = f"recbole_cdr/properties/dataset/{args.dataset}.yaml"
        if not os.path.isfile(dataset_file):
            parser.error(f"Dataset configuration not found: {dataset_file}")
        files = [dataset_file]
        if args.model in {"Proposed", "PROPOSEDSIS"}:
            # External config files override internal model defaults. Load the
            # Proposed settings after the dataset so its learning rate and
            # evaluation batch size are not replaced by dataset defaults.
            files.append(
                "recbole_cdr/properties/model/Proposed.yaml"
                if args.model == "Proposed"
                else "recbole_cdr/properties/model/sis_model_settings.yaml"
            )

    if args.model == "P2M2CDR" and not args.config_files:
        # Dataset defaults override the internal model YAML. Restore the
        # P2M2CDR-specific settings while preserving explicit CLI overrides.
        model_file = "recbole_cdr/properties/model/P2M2CDR.yaml"
        with open(model_file, encoding="utf-8") as config_file:
            model_settings = yaml.safe_load(config_file) or {}
        for key in ("eval_batch_size", "learning_rate"):
            if key in model_settings:
                overrides.setdefault(key, model_settings[key])

    gpu_id = args.gpu_id if args.gpu_id is not None else overrides.get("gpu_id")
    if gpu_id is None:
        for path in files:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as config_file:
                    gpu_id = (yaml.safe_load(config_file) or {}).get("gpu_id", gpu_id)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        overrides["gpu_id"] = gpu_id

    from recbole_cdr.quick_start import run_recbole_cdr

    sys.argv = sys.argv[:1]  # RecBole receives the parsed settings via config_dict.

    run_recbole_cdr(model=args.model, config_file_list=files,
                    config_dict=overrides or None)


if __name__ == "__main__":
    main()
