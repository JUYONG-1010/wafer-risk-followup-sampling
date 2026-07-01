from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY = Path("data") / "processed" / "spatial_risk_v2" / "spatial_action_summary.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "14_spatial_risk_v2"

ACTION_ORDER = [
    "none",
    "coverage16",
    "coverage32",
    "edge16",
    "radial16",
    "radial32",
    "local_expand",
    "edge16_local",
    "radial32_local",
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
    parser = argparse.ArgumentParser(description="Plot v2 spatial risk metrics.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def apply_categories(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["action"] = pd.Categorical(data["action"], ACTION_ORDER, ordered=True)
    data["failureType"] = pd.Categorical(data["failureType"], PATTERN_ORDER, ordered=True)
    return data


def plot_action_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = (
        summary.groupby("action", observed=False)
        .agg(value=(metric, "mean"))
        .reset_index()
    )
    data["action"] = pd.Categorical(data["action"], ACTION_ORDER, ordered=True)
    data = data.sort_values("action")

    plt.figure(figsize=(9, 5.2))
    sns.barplot(data=data, x="action", y="value", color="#4C78A8")
    plt.xlabel("Action")
    plt.ylabel(metric.replace("_", " "))
    plt.title(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_heatmap(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = apply_categories(summary)
    pivot = data.pivot(index="failureType", columns="action", values=metric)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=ACTION_ORDER)

    plt.figure(figsize=(11.5, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Action")
    plt.ylabel("Failure type")
    plt.title(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_radial_zone_miss(summary: pd.DataFrame, out_path: Path) -> None:
    data = (
        summary.groupby("action", observed=False)
        .agg(
            center=("radial_center_severe_miss", "mean"),
            mid=("radial_mid_severe_miss", "mean"),
            edge=("radial_edge_severe_miss", "mean"),
        )
        .reset_index()
    )
    data["action"] = pd.Categorical(data["action"], ACTION_ORDER, ordered=True)
    long = data.melt(id_vars=["action"], var_name="zone", value_name="severe_miss_rate")

    plt.figure(figsize=(9.5, 5.5))
    sns.barplot(data=long, x="action", y="severe_miss_rate", hue="zone")
    plt.xlabel("Action")
    plt.ylabel("Region severe miss rate")
    plt.title("Radial-Zone Severe Miss by Action")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    plot_action_metric(
        summary,
        "spatial_error_score",
        args.fig_dir / "spatial_error_score_by_action.png",
    )
    plot_action_metric(
        summary,
        "radial_weighted_abs_error",
        args.fig_dir / "radial_weighted_error_by_action.png",
    )
    plot_action_metric(
        summary,
        "quadrant_weighted_abs_error",
        args.fig_dir / "quadrant_weighted_error_by_action.png",
    )
    plot_pattern_heatmap(
        summary,
        "spatial_error_score",
        args.fig_dir / "spatial_error_score_heatmap.png",
    )
    plot_pattern_heatmap(
        summary,
        "radial_edge_severe_miss",
        args.fig_dir / "edge_zone_severe_miss_heatmap.png",
    )
    plot_radial_zone_miss(
        summary,
        args.fig_dir / "radial_zone_severe_miss_by_action.png",
    )
    print(f"wrote spatial risk figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
