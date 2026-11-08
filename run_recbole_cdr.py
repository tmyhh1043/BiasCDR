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
    if args.model in {"Proposed", "PROPOSEDSIS"}:
        from recbole_cdr.config import CDRConfig

        proposed_config = CDRConfig(
            model=args.model, config_file_list=files, config_dict=overrides or None
        )
        if not proposed_config['pretrained_model_file']:
            phase1_dir = os.path.join(proposed_config['checkpoint_dir'], 'phase1')
            source_dataset = os.path.basename(
                str(proposed_config['source_domain']['dataset']).strip()
            )
            phase1_checkpoint = os.path.join(phase1_dir, source_dataset + '.pth')
            if os.path.isfile(phase1_checkpoint):
                print(f'Reusing Phase 1 DGCDRBASE checkpoint: {phase1_checkpoint}', flush=True)
            else:
                os.makedirs(phase1_dir, exist_ok=True)
                phase1_overrides = dict(overrides)
                phase1_overrides.update({
                    'checkpoint_dir': phase1_dir,
                    'train_epochs': [f"BOTH:{int(proposed_config['gcn_pretrain_epochs'])}"],
                    'embedding_size': proposed_config['embedding_size'],
                    'n_layers': proposed_config['n_layers'],
                    'drop_rate': proposed_config['drop_rate'],
                    'connect_way': proposed_config['connect_way'],
                    'pretrained_model_file': None,
                    'skip_test_evaluation': True,
                    'metrics_csv_include_valid': False,
                })
                print(f'Training Phase 1 DGCDRBASE: {phase1_checkpoint}', flush=True)
                run_recbole_cdr(model='DGCDRBASE', config_file_list=files,
                                config_dict=phase1_overrides)
            overrides['pretrained_model_file'] = phase1_checkpoint

    run_recbole_cdr(model=args.model, config_file_list=files,
                    config_dict=overrides or None)


if __name__ == "__main__":
    main()
