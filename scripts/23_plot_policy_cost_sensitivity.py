from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = (
    Path("data") / "processed" / "policy_learning" / "cost_sensitivity"
)
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "11_cost_sensitivity"

ACTION_ORDER = [
    "none",
    "random16",
    "edge16",
    "radial32",
    "local_expand",
    "edge16_local",
    "radial32_local",
]
PLOT_ACTIONS = ["none", "random16", "edge16", "radial32", "local_expand"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot policy cost sensitivity results.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_best_action_fraction(counts: pd.DataFrame, out_path: Path) -> None:
    data = counts[counts["action"].isin(PLOT_ACTIONS)].copy()
    data["action"] = pd.Categorical(data["action"], PLOT_ACTIONS, ordered=True)

    plt.figure(figsize=(9, 5.6))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="fraction",
        hue="action",
        hue_order=PLOT_ACTIONS,
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Added sample cost weight")
    plt.ylabel("Oracle best-action fraction")
    plt.title("Best Follow-Up Action Shifts as Added-Sample Cost Changes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_oracle_metrics(summary: pd.DataFrame, out_path: Path) -> None:
    long = summary.melt(
        id_vars=["cost_weight"],
        value_vars=[
            "mean_added_valid_count",
            "mean_absolute_error",
            "severe_miss_rate",
            "underestimation_rate",
        ],
        var_name="metric",
        value_name="value",
    )

    grid = sns.relplot(
        data=long,
        x="cost_weight",
        y="value",
        col="metric",
        col_wrap=2,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.1,
        aspect=1.25,
    )
    for ax in grid.axes.flat:
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlabel("Added sample cost weight")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Oracle Best-Action Metrics vs Cost Weight", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_fixed_vs_oracle(data: pd.DataFrame, out_path: Path) -> None:
    plot_df = data[
        (
            data["action"].isin(["random16", "edge16", "radial32", "none"])
        )
        | (data["action"] == "oracle_best_by_score")
    ].copy()

    plt.figure(figsize=(9, 5.8))
    sns.lineplot(
        data=plot_df,
        x="cost_weight",
        y="mean_score",
        hue="action",
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Added sample cost weight")
    plt.ylabel("Mean cost-risk score")
    plt.title("Fixed Baselines vs Oracle Best-Action Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk_scatter(data: pd.DataFrame, out_path: Path) -> None:
    selected_weights = sorted(data["cost_weight"].unique())
    if len(selected_weights) > 4:
        selected_weights = [
            selected_weights[0],
            selected_weights[1],
            selected_weights[len(selected_weights) // 2],
            selected_weights[-1],
        ]
    plot_df = data[
        data["cost_weight"].isin(selected_weights)
        & (
            data["action"].isin(["random16", "edge16", "radial32", "none"])
            | (data["action"] == "oracle_best_by_score")
        )
    ].copy()

    grid = sns.relplot(
        data=plot_df,
        x="mean_sampled_valid_count",
        y="severe_miss_rate",
        hue="action",
        col="cost_weight",
        kind="scatter",
        s=90,
        height=3.2,
        aspect=1.05,
    )
    grid.set_axis_labels("Mean sampled valid die count", "Severe miss rate")
    grid.set_titles("cost={col_name}")
    grid.fig.suptitle("Cost-Risk Position at Selected Cost Weights", y=1.05)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(args.data_dir / "cost_sensitivity_best_action_counts.csv")
    oracle_summary = pd.read_csv(args.data_dir / "cost_sensitivity_oracle_summary.csv")
    fixed_vs_oracle = pd.read_csv(args.data_dir / "cost_sensitivity_fixed_vs_oracle.csv")

    plot_best_action_fraction(
        counts,
        args.fig_dir / "best_action_fraction_vs_cost.png",
    )
    plot_oracle_metrics(
        oracle_summary,
        args.fig_dir / "oracle_metrics_vs_cost.png",
    )
    plot_fixed_vs_oracle(
        fixed_vs_oracle,
        args.fig_dir / "fixed_vs_oracle_score_vs_cost.png",
    )
    plot_cost_risk_scatter(
        fixed_vs_oracle,
        args.fig_dir / "cost_risk_selected_weights.png",
    )

    print(f"wrote cost sensitivity figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
