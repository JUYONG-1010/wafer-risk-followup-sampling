from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "cost_aware_spatial_policy_v2"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "15_cost_aware_spatial_policy_v2"

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
KEY_ACTIONS = [
    "none",
    "coverage16",
    "coverage32",
    "edge16",
    "radial32",
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
    parser = argparse.ArgumentParser(
        description="Plot cost-aware spatial policy analysis."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_fixed_score_vs_cost(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["action"].isin(KEY_ACTIONS)].copy()
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)

    plt.figure(figsize=(9.5, 5.6))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="mean_spatial_cost_score",
        hue="action",
        hue_order=KEY_ACTIONS,
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Added valid die cost weight")
    plt.ylabel("Mean spatial cost score")
    plt.title("Cost-Aware Spatial Score by Fixed Follow-Up Action")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_risk_vs_added_count(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["action"].isin(KEY_ACTIONS)].copy()
    selected_weights = sorted(data["cost_weight"].unique())
    if len(selected_weights) > 4:
        selected_weights = [
            selected_weights[0],
            selected_weights[1],
            selected_weights[len(selected_weights) // 2],
            selected_weights[-1],
        ]
    data = data[data["cost_weight"].isin(selected_weights)].copy()

    grid = sns.relplot(
        data=data,
        x="mean_added_valid_count",
        y="mean_spatial_error_score",
        hue="action",
        col="cost_weight",
        col_wrap=2,
        kind="scatter",
        s=90,
        height=3.3,
        aspect=1.2,
    )
    grid.set_axis_labels("Mean added valid die count", "Mean spatial error score")
    grid.set_titles("cost={col_name}")
    grid.fig.suptitle("Spatial Risk vs Added Measurement Count", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_oracle_action_fraction(counts: pd.DataFrame, out_path: Path) -> None:
    data = counts[counts["action"].isin(KEY_ACTIONS)].copy()
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)

    plt.figure(figsize=(9.5, 5.6))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="fraction",
        hue="action",
        hue_order=KEY_ACTIONS,
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Added valid die cost weight")
    plt.ylabel("Oracle best-action fraction")
    plt.title("Spatial Oracle Best Action Shifts With Measurement Cost")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_heatmap(pattern_summary: pd.DataFrame, out_path: Path) -> None:
    selected_cost = 0.003
    if selected_cost not in set(pattern_summary["cost_weight"]):
        selected_cost = float(sorted(pattern_summary["cost_weight"].unique())[0])

    data = pattern_summary[
        (pattern_summary["cost_weight"] == selected_cost)
        & (pattern_summary["action"].isin(KEY_ACTIONS))
    ].copy()
    data["failureType"] = pd.Categorical(data["failureType"], PATTERN_ORDER, ordered=True)
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)

    pivot = data.pivot(
        index="failureType",
        columns="action",
        values="mean_spatial_cost_score",
    ).reindex(index=PATTERN_ORDER, columns=KEY_ACTIONS)

    plt.figure(figsize=(10.5, 5.8))
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
    plt.title(f"Pattern-Wise Spatial Cost Score at cost={selected_cost}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_reduction_per_cost(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[
        summary["action"].isin([a for a in KEY_ACTIONS if a != "none"])
    ].copy()
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)

    selected_cost = 0.003
    if selected_cost not in set(data["cost_weight"]):
        selected_cost = float(sorted(data["cost_weight"].unique())[0])
    data = data[data["cost_weight"] == selected_cost].copy()

    plt.figure(figsize=(9, 5.4))
    sns.barplot(
        data=data,
        x="action",
        y="mean_risk_reduction_per_added_die",
        order=[a for a in KEY_ACTIONS if a != "none"],
        color="#4C78A8",
    )
    plt.xlabel("Action")
    plt.ylabel("Mean spatial risk reduction per added die")
    plt.title(f"Spatial Risk Reduction Efficiency at cost={selected_cost}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fixed_summary = pd.read_csv(args.data_dir / "cost_aware_spatial_fixed_summary.csv")
    oracle_counts = pd.read_csv(args.data_dir / "cost_aware_spatial_oracle_counts.csv")
    pattern_summary = pd.read_csv(args.data_dir / "cost_aware_spatial_pattern_summary.csv")

    plot_fixed_score_vs_cost(
        fixed_summary,
        args.fig_dir / "fixed_spatial_cost_score_vs_cost.png",
    )
    plot_risk_vs_added_count(
        fixed_summary,
        args.fig_dir / "spatial_risk_vs_added_count.png",
    )
    plot_oracle_action_fraction(
        oracle_counts,
        args.fig_dir / "oracle_action_fraction_vs_cost.png",
    )
    plot_pattern_heatmap(
        pattern_summary,
        args.fig_dir / "pattern_spatial_cost_score_heatmap.png",
    )
    plot_reduction_per_cost(
        fixed_summary,
        args.fig_dir / "risk_reduction_per_added_die.png",
    )

    print(f"wrote cost-aware spatial policy figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
