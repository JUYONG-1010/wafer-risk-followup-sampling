from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "point_ranking_bias_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "20_point_ranking_bias_v1"

STRATEGY_ORDER = [
    "first_only",
    "coverage16",
    "coverage32",
    "ml_rank16",
    "ml_rank32",
    "ml_diverse16",
    "ml_diverse32",
    "ml_biasaware16",
    "ml_biasaware32",
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
    parser = argparse.ArgumentParser(description="Plot point-ranking bias diagnostics.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cost-weight", type=float, default=0.003)
    return parser.parse_args()


def apply_strategy_order(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["strategy"] = pd.Categorical(out["strategy"], STRATEGY_ORDER, ordered=True)
    return out


def plot_actual_vs_sampled(summary: pd.DataFrame, cost_weight: float, out_path: Path) -> None:
    data = apply_strategy_order(summary[summary["cost_weight"] == cost_weight])
    long = data.melt(
        id_vars=["first_pass_type", "strategy"],
        value_vars=["mean_actual_defect_ratio", "mean_sampled_defect_ratio"],
        var_name="ratio_type",
        value_name="ratio",
    )
    grid = sns.catplot(
        data=long,
        x="strategy",
        y="ratio",
        hue="ratio_type",
        col="first_pass_type",
        kind="bar",
        order=STRATEGY_ORDER,
        height=4.1,
        aspect=1.35,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    grid.set_axis_labels("Strategy", "Mean defect ratio")
    grid.set_titles("{col_name}")
    grid.fig.suptitle(f"Actual vs Sampled Defect Ratio at cost={cost_weight}", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_metric(summary: pd.DataFrame, metric: str, cost_weight: float, out_path: Path) -> None:
    data = apply_strategy_order(summary[summary["cost_weight"] == cost_weight])
    grid = sns.catplot(
        data=data,
        x="strategy",
        y=metric,
        col="first_pass_type",
        kind="bar",
        order=STRATEGY_ORDER,
        color="#4C78A8",
        height=4.0,
        aspect=1.3,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0.0, color="black", linewidth=0.8)
    grid.set_axis_labels("Strategy", metric.replace("_", " "))
    grid.set_titles("{col_name}")
    grid.fig.suptitle(metric.replace("_", " ").title() + f" at cost={cost_weight}", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_pattern_heatmap(
    pattern: pd.DataFrame,
    metric: str,
    first_pass_type: str,
    cost_weight: float,
    out_path: Path,
) -> None:
    data = pattern[
        (pattern["cost_weight"] == cost_weight)
        & (pattern["first_pass_type"] == first_pass_type)
    ].copy()
    data["failureType"] = pd.Categorical(data["failureType"], PATTERN_ORDER, ordered=True)
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    pivot = data.pivot(index="failureType", columns="strategy", values=metric)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=STRATEGY_ORDER)
    plt.figure(figsize=(12.5, 5.9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0 if "error" in metric else None,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Strategy")
    plt.ylabel("Failure type")
    plt.title(f"{metric.replace('_', ' ').title()} ({first_pass_type}, cost={cost_weight})")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_delta(delta_summary: pd.DataFrame, cost_weight: float, out_path: Path) -> None:
    data = delta_summary[delta_summary["cost_weight"] == cost_weight].copy()
    data["comparison"] = data["candidate_strategy"] + " - " + data["baseline_strategy"]
    grid = sns.catplot(
        data=data,
        x="comparison",
        y="mean_delta_absolute_error",
        col="first_pass_type",
        kind="bar",
        color="#4C78A8",
        height=4.0,
        aspect=1.45,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0.0, color="black", linewidth=0.8)
    grid.set_axis_labels("Comparison", "Delta absolute error")
    grid.set_titles("{col_name}")
    grid.fig.suptitle(f"Absolute Error Delta vs Baseline at cost={cost_weight}", y=1.05)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    strategy = pd.read_csv(args.data_dir / "point_ranking_bias_strategy_summary.csv")
    pattern = pd.read_csv(args.data_dir / "point_ranking_bias_pattern_summary.csv")
    deltas = pd.read_csv(args.data_dir / "point_ranking_bias_delta_summary.csv")

    plot_actual_vs_sampled(
        strategy,
        args.cost_weight,
        args.fig_dir / "actual_vs_sampled_defect_ratio.png",
    )
    plot_metric(
        strategy,
        "mean_ratio_error",
        args.cost_weight,
        args.fig_dir / "mean_ratio_error_by_strategy.png",
    )
    plot_metric(
        strategy,
        "overestimation_rate",
        args.cost_weight,
        args.fig_dir / "overestimation_rate_by_strategy.png",
    )
    plot_metric(
        strategy,
        "underestimation_rate",
        args.cost_weight,
        args.fig_dir / "underestimation_rate_by_strategy.png",
    )
    plot_pattern_heatmap(
        pattern,
        "mean_ratio_error",
        "grid9",
        args.cost_weight,
        args.fig_dir / "grid9_pattern_ratio_error_heatmap.png",
    )
    plot_pattern_heatmap(
        pattern,
        "mean_ratio_error",
        "grid25",
        args.cost_weight,
        args.fig_dir / "grid25_pattern_ratio_error_heatmap.png",
    )
    plot_delta(
        deltas,
        args.cost_weight,
        args.fig_dir / "delta_absolute_error_vs_baseline.png",
    )
    print(f"wrote point-ranking bias figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
