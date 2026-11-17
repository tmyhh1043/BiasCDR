"""Evaluate target-item popularity probes for saved checkpoints."""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_proposed_disentanglement import load_checkpoint
from recbole_cdr.model.cross_domain_recommender.Proposed import _popularity_targets


VARIANTS = (
    ("full", "Full Model"),
    ("no_pop_loss", "w/o Popularity Loss"),
    ("no_orth", "w/o Orthogonality Loss"),
    ("bpr_only", "BPR Only"),
)
DIRECTIONS = (
    "AmazonPhone_AmazonSport",  # Phone -> Sports
    "AmazonSport_AmazonPhone",  # Sports -> Phone
    "AmazonToy_AmazonSport",    # Toy -> Sports
    "AmazonSport_AmazonToy",    # Sports -> Toy
)


def load_target_item_embeddings(checkpoint_path: Path, device: str = "cpu"):
    """Return checkpoint config, train dataset, and final target item streams.

    The model's forward() returns target (preference_user, preference_item,
    bias_user, bias_item). Its _gcn() has already averaged all LightGCN layers.
    Only IDs [0, target_num_items) belong to the target item catalog.
    """
    config, model, train, _valid, _test, _source, target = load_checkpoint(
        checkpoint_path, device
    )
    item_count = model.target_num_items
    preference = target[1][:item_count].copy()
    bias = target[3][:item_count].copy()
    if preference.shape != bias.shape or preference.shape != (item_count, model.embedding_size):
        raise ValueError(f"Unexpected target embedding shapes: {preference.shape}, {bias.shape}")
    if not (np.isfinite(preference).all() and np.isfinite(bias).all()):
        raise ValueError("Target item embeddings contain non-finite values")
    return config, train.dataset, preference, bias


def target_train_popularity_labels(dataset, target_item_count):
    """Use the same target train graph and transformation as Proposed.__init__."""
    matrix = dataset.inter_matrix(form="coo", value_field=None, domain="target").astype(np.float32)
    if matrix.shape[1] != dataset.num_total_item or target_item_count > matrix.shape[1]:
        raise ValueError("Training matrix and target embedding item IDs do not match")
    item_z, _user_z, _valid_users = _popularity_targets(
        matrix, dataset.num_total_user, dataset.num_total_item
    )
    counts = np.asarray(matrix.tocsr().sum(axis=0)).ravel()[:target_item_count]
    labels = item_z.numpy()[:target_item_count]
    eligible = counts > 0
    eligible[0] = False  # Padding ID; zero-interaction items are also excluded.
    if not np.isfinite(labels[eligible]).all() or not eligible.any():
        raise ValueError("No valid finite popularity labels for target train items")
    return labels, counts, eligible


def evaluate_linear_probe(preference, bias, labels, eligible):
    """Fit separate probes using one fixed item split for both streams."""
    item_ids = np.flatnonzero(eligible)
    train_ids, test_ids = train_test_split(
        item_ids, test_size=0.2, random_state=2022, shuffle=True
    )
    results = {}
    for name, features in (("preference", preference), ("bias", bias)):
        probe = make_pipeline(StandardScaler(), LinearRegression())
        probe.fit(features[train_ids], labels[train_ids])
        prediction = probe.predict(features[test_ids])
        results[name] = mean_absolute_error(labels[test_ids], prediction)
    baseline = mean_absolute_error(
        labels[test_ids], np.full(len(test_ids), labels[train_ids].mean())
    )
    return len(train_ids), len(test_ids), results, baseline


def evaluate_all(checkpoint_root: Path, device: str, output: Path):
    """Compare saved variants on identical target items within each direction."""
    rows = []
    for direction in DIRECTIONS:
        print(f"\n{direction}", flush=True)
        print(f"{'Model':27s} {'Preference MAE':>16s} {'Bias MAE':>12s}", flush=True)
        reference = None
        for variant, label in VARIANTS:
            path = checkpoint_root / variant / f"{direction}-PROPOSED-ADDITIVE-seed2022.pth"
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            config, dataset, preference, bias = load_target_item_embeddings(path, device)
            if config["seed"] != 2022 or config["source_domain"]["dataset"] != direction:
                raise ValueError(f"Checkpoint seed or direction mismatch: {path}")
            labels, counts, eligible = target_train_popularity_labels(dataset, len(preference))
            id_map = dict(dataset.target_item_ID_remap_dict)
            if reference is None:
                reference = (config["target_domain"]["dataset"], id_map, labels, counts, eligible)
                print(f"  target: {reference[0]}; eligible items: {int(eligible.sum())}", flush=True)
            else:
                target_name, ref_map, ref_labels, ref_counts, ref_eligible = reference
                if (config["target_domain"]["dataset"] != target_name or id_map != ref_map or
                        not np.array_equal(labels, ref_labels) or
                        not np.array_equal(counts, ref_counts) or
                        not np.array_equal(eligible, ref_eligible)):
                    raise ValueError(f"Item IDs, labels, or eligible items differ: {path}")
            _train_count, _test_count, mae, baseline = evaluate_linear_probe(
                preference, bias, labels, eligible
            )
            if variant == "full":
                print(f"  probe train/test items: {_train_count} / {_test_count}", flush=True)
                print(f"  training-mean baseline MAE: {baseline:.6f}", flush=True)
            print(f"{label:27s} {mae['preference']:16.6f} {mae['bias']:12.6f}", flush=True)
            rows.append({
                "direction": direction,
                "model": label,
                "seed": 2022,
                "n_items": int(eligible.sum()),
                "baseline_mae": float(baseline),
                "preference_mae": float(mae["preference"]),
                "bias_mae": float(mae["bias"]),
            })
            del config, dataset, preference, bias, labels, counts, eligible, id_map
            gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "direction", "model", "seed", "n_items", "baseline_mae", "preference_mae", "bias_mae"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all", action="store_true", help="evaluate all saved variants and four directions")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("saved/proposed_ablation"))
    parser.add_argument("--output", type=Path, default=Path("results/proposed_popularity_mae_seed2022.csv"))
    args = parser.parse_args()
    if args.all:
        if args.checkpoint is not None:
            parser.error("Do not pass a checkpoint with --all")
        evaluate_all(args.checkpoint_root, args.device, args.output)
        return
    if args.checkpoint is None:
        parser.error("Provide a checkpoint, or use --all")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    config, dataset, preference, bias = load_target_item_embeddings(
        args.checkpoint, args.device
    )
    print(f"checkpoint: {args.checkpoint}")
    print(f"direction: {config['source_domain']['dataset']} -> {config['target_domain']['dataset']}")
    print(f"seed: {config['seed']}")
    print(f"target item IDs: 0..{len(preference) - 1} (0 is padding)")
    print(f"all-domain item IDs: 0..{dataset.num_total_item - 1}")
    print(f"preference shape: {preference.shape}")
    print(f"bias shape: {bias.shape}")
    print("finite values: yes")
    labels, counts, eligible = target_train_popularity_labels(dataset, len(preference))
    print(f"eligible train items: {int(eligible.sum())}")
    print(f"excluded target IDs: {len(eligible) - int(eligible.sum())} (padding and zero-count)")
    print(f"target train interactions: {int(counts.sum())}")
    print(f"popularity label mean/std: {labels[eligible].mean():.6f} / {labels[eligible].std():.6f}")
    print(f"popularity label min/max: {labels[eligible].min():.6f} / {labels[eligible].max():.6f}")
    train_count, test_count, mae, baseline_mae = evaluate_linear_probe(
        preference, bias, labels, eligible
    )
    print(f"probe train/test items: {train_count} / {test_count}")
    print(f"preference MAE: {mae['preference']:.6f}")
    print(f"bias MAE: {mae['bias']:.6f}")
    print(f"training-mean baseline MAE: {baseline_mae:.6f}")


if __name__ == "__main__":
    main()
