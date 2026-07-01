from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "point_ranking_v0" / "model_training"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "18_point_ranking_v0"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot point-ranking model results.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_metric(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = summary[summary["cost_weight"] == 0.003].copy()
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    grid = sns.catplot(
        data=data,
        x="strategy",
        y=metric,
        col="first_pass_type",
        kind="bar",
        order=STRATEGY_ORDER,
        color="#4C78A8",
        height=4,
        aspect=1.25,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    grid.set_axis_labels("Strategy", metric.replace("_", " "))
    grid.set_titles("{col_name}")
    grid.fig.suptitle(metric.replace("_", " ").title() + " at cost=0.003", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_score_vs_cost(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary.copy()
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    grid = sns.relplot(
        data=data,
        x="cost_weight",
        y="mean_spatial_cost_proxy",
        hue="strategy",
        col="first_pass_type",
        kind="line",
        marker="o",
        hue_order=STRATEGY_ORDER,
        height=4,
        aspect=1.25,
    )
    for ax in grid.axes.flat:
        ax.set_xscale("symlog", linthresh=0.001)
    grid.set_axis_labels("Cost weight", "Mean cost proxy")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Point-Ranking Cost Proxy vs Cost", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_feature_importance(importances: pd.DataFrame, out_path: Path) -> None:
    top = importances.sort_values("importance", ascending=False).head(15).copy()
    plt.figure(figsize=(8.5, 6.2))
    sns.barplot(data=top, x="importance", y="feature", color="#4C78A8")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Point-Ranking RandomForest Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.data_dir / "point_ranking_eval_summary.csv")
    importances = pd.read_csv(args.data_dir / "point_ranking_feature_importance.csv")

    plot_metric(summary, "severe_miss_rate", args.fig_dir / "point_ranking_severe_miss.png")
    plot_metric(summary, "mean_absolute_error", args.fig_dir / "point_ranking_absolute_error.png")
    plot_metric(summary, "mean_defect_coverage", args.fig_dir / "point_ranking_defect_coverage.png")
    plot_metric(summary, "mean_sampled_valid_count", args.fig_dir / "point_ranking_sampled_count.png")
    plot_score_vs_cost(summary, args.fig_dir / "point_ranking_cost_proxy_vs_cost.png")
    plot_feature_importance(importances, args.fig_dir / "point_ranking_feature_importance.png")
    print(f"wrote point-ranking figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
