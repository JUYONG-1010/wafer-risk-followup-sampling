from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "policy_learning" / "model_training"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "12_policy_models"

KEY_STRATEGIES = [
    "oracle_best_by_score",
    "decision_tree",
    "random_forest",
    "fixed_random16",
    "fixed_edge16",
    "fixed_radial32",
    "fixed_none",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trained policy model results.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_score_by_cost(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["strategy"].isin(KEY_STRATEGIES)].copy()
    plt.figure(figsize=(9.5, 5.8))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="mean_score",
        hue="strategy",
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Added sample cost weight")
    plt.ylabel("Mean cost-risk score on test wafers")
    plt.title("Policy Models vs Fixed Baselines")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["strategy"].isin(KEY_STRATEGIES)].copy()
    grid = sns.relplot(
        data=data,
        x="mean_sampled_valid_count",
        y="severe_miss_rate",
        hue="strategy",
        col="cost_weight",
        col_wrap=2,
        kind="scatter",
        s=90,
        height=3.3,
        aspect=1.2,
    )
    grid.set_axis_labels("Mean sampled valid die count", "Severe miss rate")
    grid.set_titles("cost={col_name}")
    grid.fig.suptitle("Test Cost-Risk Position by Cost Regime", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_model_accuracy(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["strategy"].isin(["decision_tree", "random_forest"])].copy()
    long = data.melt(
        id_vars=["cost_weight", "strategy"],
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(
        data=long,
        x="cost_weight",
        y="value",
        hue="strategy",
        style="metric",
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Added sample cost weight")
    plt.ylabel("Classification metric")
    plt.title("Action-Label Prediction Quality")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_action_distribution(distribution: pd.DataFrame, out_path: Path) -> None:
    data = distribution[
        distribution["strategy"].isin(["oracle_label", "decision_tree", "random_forest"])
    ].copy()
    grid = sns.catplot(
        data=data,
        x="action",
        y="fraction",
        hue="strategy",
        col="cost_weight",
        col_wrap=2,
        kind="bar",
        height=3.2,
        aspect=1.35,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    grid.set_axis_labels("Action", "Fraction")
    grid.set_titles("cost={col_name}")
    grid.fig.suptitle("Predicted Action Distribution vs Oracle Labels", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_feature_importance(importances: pd.DataFrame, out_path: Path) -> None:
    top = (
        importances.sort_values(["cost_weight", "model", "importance"], ascending=[True, True, False])
        .groupby(["cost_weight", "model"], observed=False)
        .head(10)
        .copy()
    )
    grid = sns.catplot(
        data=top,
        x="importance",
        y="feature",
        hue="model",
        col="cost_weight",
        col_wrap=2,
        kind="bar",
        sharex=False,
        sharey=False,
        height=3.6,
        aspect=1.25,
    )
    grid.set_axis_labels("Importance", "Feature")
    grid.set_titles("cost={col_name}")
    grid.fig.suptitle("Top Feature Importances", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.data_dir / "policy_model_summary.csv")
    distribution = pd.read_csv(args.data_dir / "policy_model_action_distribution.csv")
    importances = pd.read_csv(args.data_dir / "policy_model_feature_importance.csv")

    plot_score_by_cost(summary, args.fig_dir / "policy_model_score_by_cost.png")
    plot_cost_risk(summary, args.fig_dir / "policy_model_cost_risk.png")
    plot_model_accuracy(summary, args.fig_dir / "policy_model_accuracy.png")
    plot_action_distribution(distribution, args.fig_dir / "policy_model_action_distribution.png")
    plot_feature_importance(importances, args.fig_dir / "policy_model_feature_importance.png")

    print(f"wrote policy model figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
