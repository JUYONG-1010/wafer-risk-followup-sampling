from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS = Path("data") / "processed" / "sampling" / "sampling_results.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "03_sampling_risk"

SCHEME_ORDER = [
    "grid_5point",
    "grid_9point",
    "grid_25point",
    "interior_5point",
    "interior_9point",
    "interior_25point",
    "radial",
    "edge_biased",
    "random_25",
    "adaptive_9point",
]

PATTERN_ORDER = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sampling-induced risk metrics from simulation results."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional filename suffix, e.g. smoke for smoke-test plots.",
    )
    return parser.parse_args()


def suffix_name(base: str, suffix: str) -> str:
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["failureType", "scheme"])
        .agg(
            wafers=("row_index", "count"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
            mean_sampled_defect_ratio=("sampled_defect_ratio", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            underestimation_rate=("underestimated", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            hit_rate=("hit", "mean"),
        )
        .reset_index()
    )


def plot_heatmap(
    summary: pd.DataFrame,
    value: str,
    title: str,
    label: str,
    out_path: Path,
    cmap: str = "rocket_r",
) -> None:
    pivot = summary.pivot(index="failureType", columns="scheme", values=value)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=SCHEME_ORDER)

    plt.figure(figsize=(11, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": label},
    )
    plt.xlabel("Sampling scheme")
    plt.ylabel("Failure type")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk_tradeoff(summary: pd.DataFrame, out_path: Path) -> None:
    scheme_summary = (
        summary.groupby("scheme")
        .agg(
            mean_sampled_valid_count=("mean_sampled_valid_count", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
            underestimation_rate=("underestimation_rate", "mean"),
        )
        .reset_index()
    )
    scheme_summary["scheme"] = pd.Categorical(
        scheme_summary["scheme"], categories=SCHEME_ORDER, ordered=True
    )
    scheme_summary = scheme_summary.sort_values("scheme")

    plt.figure(figsize=(8.5, 5.5))
    ax = sns.scatterplot(
        data=scheme_summary,
        x="mean_sampled_valid_count",
        y="mean_absolute_error",
        size="severe_miss_rate",
        hue="underestimation_rate",
        sizes=(80, 450),
        palette="viridis",
        edgecolor="black",
        linewidth=0.6,
    )

    for _, row in scheme_summary.iterrows():
        ax.text(
            row["mean_sampled_valid_count"] + 0.8,
            row["mean_absolute_error"],
            str(row["scheme"]),
            fontsize=8,
            va="center",
        )

    plt.xlabel("Mean sampled valid die count (cost proxy)")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Cost-Risk Tradeoff by Sampling Scheme")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_scheme_bias(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(index="failureType", columns="scheme", values="mean_ratio_error")
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=SCHEME_ORDER)

    plt.figure(figsize=(11, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "sampled ratio - actual ratio"},
    )
    plt.xlabel("Sampling scheme")
    plt.ylabel("Failure type")
    plt.title("Mean Defect-Ratio Bias by Pattern")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results)
    summary = aggregate_results(results)

    summary_path = args.results.with_name(
        args.results.stem.replace("results", "summary_by_pattern") + ".csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"wrote summary: {summary_path}")

    plot_heatmap(
        summary,
        value="mean_absolute_error",
        title="Mean Absolute Defect-Ratio Error",
        label="absolute error",
        out_path=args.fig_dir / suffix_name("absolute_error_heatmap", args.suffix),
    )
    plot_heatmap(
        summary,
        value="severe_miss_rate",
        title="Severe Miss Rate: Actual Defects Exist, Sample Observes Zero",
        label="severe miss rate",
        out_path=args.fig_dir / suffix_name("severe_miss_heatmap", args.suffix),
        cmap="mako_r",
    )
    plot_heatmap(
        summary,
        value="underestimation_rate",
        title="Underestimation Rate by Pattern and Sampling Scheme",
        label="underestimation rate",
        out_path=args.fig_dir / suffix_name("underestimation_heatmap", args.suffix),
        cmap="crest",
    )
    plot_scheme_bias(
        summary,
        out_path=args.fig_dir / suffix_name("ratio_bias_heatmap", args.suffix),
    )
    plot_cost_risk_tradeoff(
        summary,
        out_path=args.fig_dir / suffix_name("cost_risk_tradeoff", args.suffix),
    )
    print(f"wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
