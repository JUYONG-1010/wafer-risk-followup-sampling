from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "first_pass_comparison_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "17_first_pass_comparison_v1"

STRATEGY_ORDER = [
    "grid9",
    "grid25",
    "center_disk_r5",
    "center_disk_r7",
    "grid9_coverage16",
    "grid9_coverage32",
    "grid25_coverage16",
    "grid25_edge16",
    "center_r5_edge16",
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
    parser = argparse.ArgumentParser(description="Plot first-pass strategy comparison.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def apply_strategy_order(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    return data


def plot_metric_bars(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = apply_strategy_order(summary[summary["cost_weight"] == 0.0])
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=data, x="strategy", y=metric, order=STRATEGY_ORDER, color="#4C78A8")
    plt.xlabel("Strategy")
    plt.ylabel(metric.replace("_", " "))
    plt.title(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_score(summary: pd.DataFrame, out_path: Path) -> None:
    data = apply_strategy_order(summary)
    plt.figure(figsize=(10, 5.8))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="mean_spatial_cost_score",
        hue="strategy",
        hue_order=STRATEGY_ORDER,
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Cost weight per sampled valid die")
    plt.ylabel("Mean spatial cost score")
    plt.title("First-Pass / Follow-Up Strategy Score vs Cost")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    data = apply_strategy_order(summary[summary["cost_weight"] == 0.003])
    plt.figure(figsize=(8.6, 6.2))
    sns.scatterplot(
        data=data,
        x="mean_sampled_valid_count",
        y="mean_spatial_error_score",
        hue="strategy",
        hue_order=STRATEGY_ORDER,
        s=110,
    )
    for _, row in data.iterrows():
        plt.text(
            row["mean_sampled_valid_count"] + 0.6,
            row["mean_spatial_error_score"],
            str(row["strategy"]),
            fontsize=8,
        )
    plt.xlabel("Mean sampled valid die count")
    plt.ylabel("Mean spatial error score")
    plt.title("Spatial Error vs Measurement Count at cost=0.003")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_heatmap(pattern: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = pattern[pattern["cost_weight"] == 0.003].copy()
    data["failureType"] = pd.Categorical(data["failureType"], PATTERN_ORDER, ordered=True)
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    pivot = data.pivot(index="failureType", columns="strategy", values=metric)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=STRATEGY_ORDER)
    plt.figure(figsize=(13, 5.9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Strategy")
    plt.ylabel("Failure type")
    plt.title(metric.replace("_", " ").title() + " at cost=0.003")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.data_dir / "first_pass_strategy_summary.csv")
    pattern = pd.read_csv(args.data_dir / "first_pass_strategy_pattern_summary.csv")

    plot_metric_bars(summary, "severe_miss_rate", args.fig_dir / "severe_miss_by_strategy.png")
    plot_metric_bars(summary, "mean_spatial_error_score", args.fig_dir / "spatial_error_by_strategy.png")
    plot_metric_bars(summary, "mean_sampled_valid_count", args.fig_dir / "sampled_count_by_strategy.png")
    plot_cost_score(summary, args.fig_dir / "spatial_cost_score_vs_cost.png")
    plot_cost_risk_scatter(summary, args.fig_dir / "spatial_error_vs_sampled_count.png")
    plot_pattern_heatmap(
        pattern,
        "mean_spatial_cost_score",
        args.fig_dir / "pattern_spatial_cost_heatmap.png",
    )
    print(f"wrote first-pass comparison figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
