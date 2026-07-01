from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY = Path("data") / "processed" / "adaptive_v2" / "adaptive_v2_summary.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "09_adaptive_v2"

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
SCHEME_ORDER = [
    "grid9_first_pass",
    "grid9_plus_random16",
    "grid9_plus_edge16",
    "grid9_plus_radial32",
    "adaptive_v2_local_only",
    "adaptive_v2_density_aware",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adaptive v2 sampling results.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def prepare_categories(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "failureType" in data.columns:
        data["failureType"] = pd.Categorical(
            data["failureType"], PATTERN_ORDER, ordered=True
        )
    if "scheme" in data.columns:
        data["scheme"] = pd.Categorical(data["scheme"], SCHEME_ORDER, ordered=True)
    return data


def plot_heatmap(
    summary: pd.DataFrame,
    value: str,
    title: str,
    out_path: Path,
    cmap: str = "rocket_r",
    center: float | None = None,
) -> None:
    data = prepare_categories(summary)
    pivot = data.pivot(index="failureType", columns="scheme", values=value)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=SCHEME_ORDER)

    plt.figure(figsize=(11.5, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Sampling strategy")
    plt.ylabel("Failure type")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk(summary: pd.DataFrame, out_path: Path) -> None:
    scheme_summary = (
        summary.groupby("scheme", observed=False)
        .agg(
            mean_sampled_valid_count=("mean_sampled_valid_count", "mean"),
            mean_added_valid_count=("mean_added_valid_count", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
            underestimation_rate=("underestimation_rate", "mean"),
        )
        .reset_index()
    )
    scheme_summary = prepare_categories(scheme_summary)
    scheme_summary = scheme_summary.sort_values("scheme")

    plt.figure(figsize=(9, 5.8))
    ax = sns.scatterplot(
        data=scheme_summary,
        x="mean_sampled_valid_count",
        y="severe_miss_rate",
        hue="mean_absolute_error",
        size="mean_added_valid_count",
        sizes=(80, 450),
        palette="viridis",
        edgecolor="black",
        linewidth=0.6,
    )
    for _, row in scheme_summary.iterrows():
        ax.text(
            row["mean_sampled_valid_count"] + 0.9,
            row["severe_miss_rate"],
            str(row["scheme"]),
            fontsize=8,
            va="center",
        )
    plt.xlabel("Mean sampled valid die count")
    plt.ylabel("Severe miss rate")
    plt.title("Adaptive v2 Cost-Risk Tradeoff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_reduction_per_added_die(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["scheme"] != "grid9_first_pass"].copy()
    data = prepare_categories(data)
    pivot = data.pivot(
        index="failureType",
        columns="scheme",
        values="severe_miss_reduction_per_added_die",
    )
    pivot = pivot.reindex(
        index=PATTERN_ORDER, columns=[s for s in SCHEME_ORDER if s != "grid9_first_pass"]
    )

    plt.figure(figsize=(10.5, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="crest",
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Follow-up strategy")
    plt.ylabel("Failure type")
    plt.title("Severe-Miss Reduction per Added Sampled Die")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_bars(summary: pd.DataFrame, value: str, title: str, out_path: Path) -> None:
    data = prepare_categories(summary)
    plt.figure(figsize=(12, 6.2))
    sns.barplot(
        data=data,
        x="failureType",
        y=value,
        hue="scheme",
        hue_order=SCHEME_ORDER,
    )
    plt.xlabel("Failure type")
    plt.ylabel(value.replace("_", " "))
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Strategy", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)

    plot_heatmap(
        summary,
        value="severe_miss_rate",
        title="Adaptive v2 Severe Miss Rate",
        out_path=args.fig_dir / "adaptive_v2_severe_miss_heatmap.png",
        cmap="mako_r",
    )
    plot_heatmap(
        summary,
        value="mean_absolute_error",
        title="Adaptive v2 Mean Absolute Defect-Ratio Error",
        out_path=args.fig_dir / "adaptive_v2_absolute_error_heatmap.png",
        cmap="rocket_r",
    )
    plot_heatmap(
        summary,
        value="severe_miss_reduction_vs_grid9",
        title="Severe-Miss Reduction vs 9-Point First Pass",
        out_path=args.fig_dir / "adaptive_v2_severe_miss_reduction_heatmap.png",
        cmap="vlag",
        center=0.0,
    )
    plot_heatmap(
        summary,
        value="added_count_vs_grid9",
        title="Added Sample Count vs 9-Point First Pass",
        out_path=args.fig_dir / "adaptive_v2_added_count_heatmap.png",
        cmap="flare",
    )
    plot_reduction_per_added_die(
        summary,
        out_path=args.fig_dir / "adaptive_v2_reduction_per_added_die.png",
    )
    plot_cost_risk(
        summary,
        out_path=args.fig_dir / "adaptive_v2_cost_risk_tradeoff.png",
    )
    plot_pattern_bars(
        summary,
        value="severe_miss_rate",
        title="Adaptive v2 Severe Miss by Pattern",
        out_path=args.fig_dir / "adaptive_v2_severe_miss_bars.png",
    )
    print(f"wrote adaptive v2 figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
