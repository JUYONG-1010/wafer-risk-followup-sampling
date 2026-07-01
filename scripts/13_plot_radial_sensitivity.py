from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY = Path("data") / "processed" / "radial" / "radial_sensitivity_summary.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "07_radial_sensitivity"

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
    parser = argparse.ArgumentParser(description="Plot radial sensitivity analysis.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_stage_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plot_df = data.copy()
    plot_df["failureType"] = pd.Categorical(
        plot_df["failureType"], PATTERN_ORDER, ordered=True
    )
    plt.figure(figsize=(10, 5.8))
    sns.lineplot(
        data=plot_df,
        x=x,
        y=y,
        hue="failureType",
        hue_order=PATTERN_ORDER,
        marker="o",
    )
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_variant_heatmap(
    data: pd.DataFrame,
    value: str,
    title: str,
    out_path: Path,
    cmap: str = "rocket_r",
) -> None:
    pivot = data.pivot(index="failureType", columns="variant", values=value)
    pivot = pivot.reindex(index=PATTERN_ORDER)
    plt.figure(figsize=(12, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Radial variant")
    plt.ylabel("Failure type")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk(summary: pd.DataFrame, out_path: Path) -> None:
    variant_summary = (
        summary.groupby(["variant", "stage"])
        .agg(
            mean_sampled_valid_count=("mean_sampled_valid_count", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
            underestimation_rate=("underestimation_rate", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5.8))
    ax = sns.scatterplot(
        data=variant_summary,
        x="mean_sampled_valid_count",
        y="mean_absolute_error",
        hue="stage",
        size="severe_miss_rate",
        sizes=(80, 450),
        edgecolor="black",
        linewidth=0.6,
    )
    for _, row in variant_summary.iterrows():
        ax.text(
            row["mean_sampled_valid_count"] + 0.8,
            row["mean_absolute_error"],
            row["variant"],
            fontsize=7,
            va="center",
        )
    plt.xlabel("Mean sampled valid die count")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Radial Sensitivity Cost-Risk Tradeoff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)

    outer = summary[summary["stage"] == "outer_radius"].copy()
    plot_stage_line(
        outer,
        x="outer_radius",
        y="severe_miss_rate",
        title="Radial Outer Radius Sensitivity: Severe Miss Rate",
        ylabel="Severe miss rate",
        out_path=args.fig_dir / "radial_outer_radius_severe_miss.png",
    )
    plot_stage_line(
        outer,
        x="outer_radius",
        y="mean_absolute_error",
        title="Radial Outer Radius Sensitivity: Absolute Error",
        ylabel="Mean absolute defect-ratio error",
        out_path=args.fig_dir / "radial_outer_radius_absolute_error.png",
    )

    angle = summary[summary["stage"] == "angle_count"].copy()
    plot_stage_line(
        angle,
        x="angles",
        y="severe_miss_rate",
        title="Radial Angle Count Sensitivity: Severe Miss Rate",
        ylabel="Severe miss rate",
        out_path=args.fig_dir / "radial_angle_severe_miss.png",
    )
    plot_stage_line(
        angle,
        x="angles",
        y="mean_absolute_error",
        title="Radial Angle Count Sensitivity: Absolute Error",
        ylabel="Mean absolute defect-ratio error",
        out_path=args.fig_dir / "radial_angle_absolute_error.png",
    )

    ring = summary[summary["stage"] == "ring_count"].copy()
    plot_stage_line(
        ring,
        x="ring_count",
        y="severe_miss_rate",
        title="Radial Ring Count Sensitivity: Severe Miss Rate",
        ylabel="Severe miss rate",
        out_path=args.fig_dir / "radial_ring_count_severe_miss.png",
    )
    plot_stage_line(
        ring,
        x="ring_count",
        y="mean_absolute_error",
        title="Radial Ring Count Sensitivity: Absolute Error",
        ylabel="Mean absolute defect-ratio error",
        out_path=args.fig_dir / "radial_ring_count_absolute_error.png",
    )

    plot_variant_heatmap(
        summary,
        value="mean_absolute_error",
        title="Radial Variant Mean Absolute Error",
        out_path=args.fig_dir / "radial_pattern_absolute_error_heatmap.png",
    )
    plot_variant_heatmap(
        summary,
        value="severe_miss_rate",
        title="Radial Variant Severe Miss Rate",
        out_path=args.fig_dir / "radial_pattern_severe_miss_heatmap.png",
        cmap="mako_r",
    )
    plot_cost_risk(
        summary,
        out_path=args.fig_dir / "radial_cost_risk_tradeoff.png",
    )
    print(f"wrote radial sensitivity figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
