#!/usr/bin/env python3

"""Cross-domain recommendation EDA for popularity bias and user diversity.

The script accepts two RecBole-style ``.inter`` files.  The first file is

treated as the source domain and the second as the target domain.  Only the

first two columns (user and item) are needed; type suffixes such as ``:token``

in the header are removed automatically.

"""

from __future__ import annotations

import argparse

import json

import math

import os

import tempfile

import warnings

from pathlib import Path

from typing import Any

# Avoid writing Matplotlib's cache into a possibly read-only home directory.

os.environ.setdefault(

    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cdr_eda_matplotlib")

)

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from sklearn.mixture import GaussianMixture

HEAD_FRACTION = 0.20

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(

            "Analyze popularity bias and user diversity in a pair of CDR "

            "interaction files."

        )

    )

    parser.add_argument("source_file", type=Path, help="Source-domain .inter file")

    parser.add_argument("target_file", type=Path, help="Target-domain .inter file")

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=Path("cdr_eda_output"),

        help="Directory in which plots and tables are saved",

    )

    parser.add_argument("--source-label", help="Label used for the source domain")

    parser.add_argument("--target-label", help="Label used for the target domain")

    parser.add_argument(

        "--head-fraction",

        type=float,

        default=HEAD_FRACTION,

        help="Top fraction of items classified as Head (default: 0.20)",

    )

    parser.add_argument(

        "--random-state",

        type=int,

        default=42,

        help="Random seed used by GaussianMixture",

    )

    parser.add_argument(

        "--show",

        action="store_true",

        help="Display figures interactively after saving them",

    )

    return parser.parse_args()

def infer_domain_label(path: Path) -> str:

    """Use the portion before the first underscore as the domain label."""

    return path.stem.split("_", maxsplit=1)[0]

def load_interactions(path: Path) -> pd.DataFrame:

    """Load user/item columns from a comma-separated RecBole .inter file."""

    if not path.is_file():

        raise FileNotFoundError(f"Interaction file not found: {path}")

    # Positional selection makes this work with both typed and plain headers.

    frame = pd.read_csv(path, usecols=[0, 1], dtype="string", low_memory=False)

    if frame.shape[1] != 2:

        raise ValueError(f"Expected at least two columns in {path}")

    frame.columns = [str(column).split(":", maxsplit=1)[0] for column in frame.columns]

    frame.columns = ["user_id", "item_id"]

    before = len(frame)

    frame = frame.dropna(subset=["user_id", "item_id"])

    dropped = before - len(frame)

    if dropped:

        warnings.warn(f"Dropped {dropped:,} rows with missing user/item IDs from {path}")

    if frame.empty:

        raise ValueError(f"No valid interactions found in {path}")

    return frame

def compute_domain_statistics(

    interactions: pd.DataFrame, head_fraction: float

) -> tuple[pd.DataFrame, pd.Series, set[str]]:

    """Compute item degree and user degree, Head Ratio, and API.

    API (average popularity of interacted items) is the arithmetic mean of the

    global item degrees attached to each of a user's interaction rows.

    """

    item_popularity = interactions["item_id"].value_counts().rename("item_degree")

    n_head = max(1, math.ceil(len(item_popularity) * head_fraction))

    head_items = set(item_popularity.nlargest(n_head).index.astype(str))

    enriched = interactions.assign(

        item_degree=interactions["item_id"].map(item_popularity).astype(float),

        is_head=interactions["item_id"].isin(head_items).astype(float),

    )

    user_metrics = (

        enriched.groupby("user_id", sort=False)

        .agg(

            user_degree=("item_id", "size"),

            head_ratio=("is_head", "mean"),

            api=("item_degree", "mean"),

        )

        .sort_index()

    )

    return user_metrics, item_popularity, head_items

def safe_distribution_metrics(values: pd.Series) -> dict[str, float]:

    clean = values.dropna().to_numpy(dtype=float)

    return {

        "n_users": float(clean.size),

        "mean": float(np.mean(clean)) if clean.size else np.nan,

        "sd": float(np.std(clean, ddof=1)) if clean.size > 1 else np.nan,

        # scipy.stats.kurtosis with fisher=True reports excess kurtosis.

        "kurtosis_excess": (

            float(stats.kurtosis(clean, fisher=True, bias=False))

            if clean.size >= 4 and np.ptp(clean) > 0

            else np.nan

        ),

        "iqr": float(stats.iqr(clean)) if clean.size else np.nan,

    }

def gmm_bic_comparison(values: pd.Series, random_state: int) -> dict[str, Any]:

    """Fit one- and two-component GMMs and return their BIC values."""

    clean = values.dropna().to_numpy(dtype=float).reshape(-1, 1)

    result: dict[str, Any] = {"bic_k1": np.nan, "bic_k2": np.nan, "preferred_k": None}

    if clean.shape[0] < 2:

        return result

    for k in (1, 2):

        if clean.shape[0] < k:

            continue

        try:

            model = GaussianMixture(

                n_components=k,

                covariance_type="full",

                reg_covar=1e-6,

                n_init=10,

                random_state=random_state,

            )

            model.fit(clean)

            result[f"bic_k{k}"] = float(model.bic(clean))

        except (ValueError, FloatingPointError) as exc:

            warnings.warn(f"GMM(k={k}) could not be fitted: {exc}")

    finite = {

        k: result[f"bic_k{k}"]

        for k in (1, 2)

        if np.isfinite(result[f"bic_k{k}"])

    }

    if finite:

        result["preferred_k"] = int(min(finite, key=finite.get))

    return result

def safe_correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:

    """Return Pearson/Spearman coefficients and p-values without constant warnings."""

    paired = pd.concat([x, y], axis=1).dropna()

    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:

        return {

            "n": float(len(paired)),

            "pearson_r": np.nan,

            "pearson_p": np.nan,

            "spearman_rho": np.nan,

            "spearman_p": np.nan,

        }

    pearson_r, pearson_p = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])

    spearman_rho, spearman_p = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])

    return {

        "n": float(len(paired)),

        "pearson_r": float(pearson_r),

        "pearson_p": float(pearson_p),

        "spearman_rho": float(spearman_rho),

        "spearman_p": float(spearman_p),

    }

def gini_coefficient(values: pd.Series | np.ndarray) -> float:

    """Calculate the Gini coefficient of non-negative item-degree values."""

    array = np.asarray(values, dtype=float).ravel()

    array = array[np.isfinite(array)]

    if array.size == 0:

        return np.nan

    if np.any(array < 0):

        raise ValueError("Gini coefficient is defined here only for non-negative values")

    total = array.sum()

    if total == 0:

        return 0.0

    sorted_values = np.sort(array)

    ranks = np.arange(1, sorted_values.size + 1, dtype=float)

    return float(

        (2.0 * np.sum(ranks * sorted_values))

        / (sorted_values.size * total)

        - (sorted_values.size + 1.0) / sorted_values.size

    )

def finish_figure(fig: plt.Figure, output_path: Path, show: bool) -> None:

    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:

        plt.show()

    plt.close(fig)

def plot_analysis_1(

    common: pd.DataFrame,

    source_label: str,

    target_label: str,

    output_path: Path,

    show: bool,

) -> None:

    """Analysis 1: common-user Head Ratio distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    bins = np.linspace(0.0, 1.0, 21)

    for axis, column, label, color in (

        (axes[0], "source_head_ratio", source_label, "#4472C4"),

        (axes[1], "target_head_ratio", target_label, "#ED7D31"),

    ):

        sns.histplot(common[column], bins=bins, ax=axis, color=color, edgecolor="white")

        axis.set(title=label, xlabel="Head Ratio", ylabel="Number of common users")

        axis.set_xlim(0.0, 1.0)

    fig.suptitle("Analysis 1: Distribution of common-user Head Ratio")

    finish_figure(fig, output_path, show)

def plot_analysis_2(

    common: pd.DataFrame,

    source_label: str,

    target_label: str,

    correlation: dict[str, float],

    output_path: Path,

    show: bool,

) -> None:

    """Analysis 2: cross-domain Head Ratio correlation."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))

    sns.scatterplot(

        data=common,

        x="source_head_ratio",

        y="target_head_ratio",

        s=18,

        alpha=0.35,

        linewidth=0,

        ax=axis,

        rasterized=True,

    )

    axis.set(

        xlim=(-0.02, 1.02),

        ylim=(-0.02, 1.02),

        xlabel=f"{source_label} Head Ratio",

        ylabel=f"{target_label} Head Ratio",

        title=(

            "Analysis 2: Cross-domain popularity preference\n"

            f"Pearson r={correlation['pearson_r']:.3f}, "

            f"Spearman rho={correlation['spearman_rho']:.3f}"

        ),

    )

    finish_figure(fig, output_path, show)

def plot_analysis_3(

    source_users: pd.DataFrame,

    target_users: pd.DataFrame,

    source_label: str,

    target_label: str,

    output_path: Path,

    show: bool,

) -> None:

    """Analysis 3: user activity versus Head Ratio in each domain."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for axis, frame, label, color in (

        (axes[0], source_users, source_label, "#4472C4"),

        (axes[1], target_users, target_label, "#ED7D31"),

    ):

        sns.scatterplot(

            data=frame,

            x="user_degree",

            y="head_ratio",

            s=14,

            alpha=0.25,

            linewidth=0,

            color=color,

            ax=axis,

            rasterized=True,

        )

        axis.set_xscale("log")

        axis.set(

            title=label,

            xlabel="User degree (log scale)",

            ylabel="Head Ratio",

            ylim=(-0.02, 1.02),

        )

    fig.suptitle("Analysis 3: User activity and popularity dependence")

    finish_figure(fig, output_path, show)

def plot_analysis_4(

    source_popularity: pd.Series,

    target_popularity: pd.Series,

    source_label: str,

    target_label: str,

    output_path: Path,

    show: bool,

) -> None:

    """Analysis 4: overlaid item-popularity long-tail curves."""

    fig, axis = plt.subplots(figsize=(7.5, 5.2))

    for popularity, label, color in (

        (source_popularity, source_label, "#4472C4"),

        (target_popularity, target_label, "#ED7D31"),

    ):

        counts = np.sort(popularity.to_numpy(dtype=float))[::-1]

        ranks = np.arange(1, counts.size + 1)

        axis.plot(ranks, counts, label=label, color=color, linewidth=1.5, alpha=0.9)

    axis.set_xscale("log")

    axis.set_yscale("log")

    axis.set(

        xlabel="Item popularity rank (log scale)",

        ylabel="Number of interactions (log scale)",

        title="Analysis 4: Item-popularity long-tail curves",

    )

    axis.legend()

    axis.grid(True, which="both", alpha=0.2)

    finish_figure(fig, output_path, show)

def json_safe(value: Any) -> Any:

    """Recursively replace numpy scalars and non-finite values for strict JSON."""

    if isinstance(value, dict):

        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):

        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating, float)):

        return float(value) if np.isfinite(value) else None

    return value

def flatten_metrics(metrics: dict[str, Any]) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    def walk(node: Any, path: list[str]) -> None:

        if isinstance(node, dict):

            for key, value in node.items():

                walk(value, [*path, str(key)])

        else:

            rows.append(

                {

                    "section": path[0] if path else "",

                    "scope": path[1] if len(path) > 1 else "",

                    "metric": ".".join(path[2:]) if len(path) > 2 else "value",

                    "value": node,

                }

            )

    walk(metrics, [])

    return pd.DataFrame(rows)

def print_metric_block(title: str, values: dict[str, Any]) -> None:

    print(f"\n[{title}]")

    for key, value in values.items():

        if isinstance(value, float):

            print(f"  {key}: {value:.6g}" if np.isfinite(value) else f"  {key}: NA")

        else:

            print(f"  {key}: {value}")

def main() -> None:

    args = parse_args()

    if not 0.0 < args.head_fraction <= 1.0:

        raise ValueError("--head-fraction must be in the interval (0, 1]")

    source_label = args.source_label or infer_domain_label(args.source_file)

    target_label = args.target_label or infer_domain_label(args.target_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    print(f"Loading source: {args.source_file}")

    source_interactions = load_interactions(args.source_file)

    print(f"Loading target: {args.target_file}")

    target_interactions = load_interactions(args.target_file)

    source_users, source_popularity, source_head = compute_domain_statistics(

        source_interactions, args.head_fraction

    )

    target_users, target_popularity, target_head = compute_domain_statistics(

        target_interactions, args.head_fraction

    )

    common_ids = source_users.index.intersection(target_users.index)

    if common_ids.empty:

        raise ValueError("The two input files have no common users")

    common = source_users.loc[common_ids].add_prefix("source_").join(

        target_users.loc[common_ids].add_prefix("target_"), how="inner"

    )

    common.index.name = "user_id"

    # ------------------------------------------------------------------

    # Analysis 1: diversity in user-level dependence on Head items.

    # SD uses the sample definition (ddof=1); kurtosis is excess kurtosis.

    # Lower GMM BIC is preferred.  k=2 support is suggestive, not by itself

    # proof that the distribution is meaningfully bimodal.

    # ------------------------------------------------------------------

    analysis_1: dict[str, Any] = {}

    for prefix, label in (("source", source_label), ("target", target_label)):

        values = common[f"{prefix}_head_ratio"]

        analysis_1[label] = {

            **safe_distribution_metrics(values),

            **gmm_bic_comparison(values, args.random_state),

        }

        print_metric_block(f"Analysis 1 - {label}", analysis_1[label])

    plot_analysis_1(

        common,

        source_label,

        target_label,

        args.output_dir / "analysis1_head_ratio_histograms.png",

        args.show,

    )

    # ------------------------------------------------------------------

    # Analysis 2: cross-domain correlation of common-user Head Ratios.

    # ------------------------------------------------------------------

    analysis_2 = safe_correlation(

        common["source_head_ratio"], common["target_head_ratio"]

    )

    print_metric_block("Analysis 2 - cross-domain Head Ratio", analysis_2)

    plot_analysis_2(

        common,

        source_label,

        target_label,

        analysis_2,

        args.output_dir / "analysis2_cross_domain_head_ratio.png",

        args.show,

    )

    # ------------------------------------------------------------------

    # Analysis 3: sparsity/activity versus popularity dependence.

    # A negative coefficient means lower-degree (lighter) users tend to have

    # a higher Head Ratio.  Both Pearson and rank-based Spearman are reported.

    # ------------------------------------------------------------------

    analysis_3 = {

        source_label: safe_correlation(

            source_users["user_degree"], source_users["head_ratio"]

        ),

        target_label: safe_correlation(

            target_users["user_degree"], target_users["head_ratio"]

        ),

    }

    for label, values in analysis_3.items():

        print_metric_block(f"Analysis 3 - {label}", values)

    plot_analysis_3(

        source_users,

        target_users,

        source_label,

        target_label,

        args.output_dir / "analysis3_user_degree_vs_head_ratio.png",

        args.show,

    )

    # ------------------------------------------------------------------

    # Analysis 4: item-popularity long tails and concentration (Gini).

    # Gini=0 indicates equal item degrees; values nearer 1 indicate stronger

    # concentration of interactions among a small set of items.

    # ------------------------------------------------------------------

    analysis_4 = {

        source_label: {

            "n_items": int(len(source_popularity)),

            "n_head_items": int(len(source_head)),

            "gini": gini_coefficient(source_popularity),

        },

        target_label: {

            "n_items": int(len(target_popularity)),

            "n_head_items": int(len(target_head)),

            "gini": gini_coefficient(target_popularity),

        },

    }

    for label, values in analysis_4.items():

        print_metric_block(f"Analysis 4 - {label}", values)

    plot_analysis_4(

        source_popularity,

        target_popularity,

        source_label,

        target_label,

        args.output_dir / "analysis4_long_tail.png",

        args.show,

    )

    # Save reusable numeric results in addition to the four figures.

    source_export = source_popularity.rename("item_degree").to_frame()

    source_export.index.name = "item_id"

    target_export = target_popularity.rename("item_degree").to_frame()

    target_export.index.name = "item_id"

    common.to_csv(args.output_dir / "common_user_metrics.csv")

    source_export.to_csv(args.output_dir / f"item_popularity_{source_label}.csv")

    target_export.to_csv(args.output_dir / f"item_popularity_{target_label}.csv")

    metrics = {

        "metadata": {

            "source_label": source_label,

            "target_label": target_label,

            "source_file": str(args.source_file.resolve()),

            "target_file": str(args.target_file.resolve()),

            "head_fraction": args.head_fraction,

            "source_interactions": int(len(source_interactions)),

            "target_interactions": int(len(target_interactions)),

            "source_users": int(len(source_users)),

            "target_users": int(len(target_users)),

            "common_users": int(len(common)),

        },

        "analysis_1": analysis_1,

        "analysis_2": analysis_2,

        "analysis_3": analysis_3,

        "analysis_4": analysis_4,

    }

    safe_metrics = json_safe(metrics)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:

        json.dump(safe_metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)

    flatten_metrics(safe_metrics).to_csv(

        args.output_dir / "summary_metrics.csv", index=False

    )

    print(f"\nSaved results to: {args.output_dir.resolve()}")

if __name__ == "__main__":

    main()
