from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "policy_learning"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "10_policy_learning"

ACTION_ORDER = [
    "none",
    "random16",
    "edge16",
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
    parser = argparse.ArgumentParser(
        description="Plot QA figures for the policy-learning dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def apply_categories(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "failureType" in data.columns:
        data["failureType"] = pd.Categorical(
            data["failureType"], categories=PATTERN_ORDER, ordered=True
        )
    if "action" in data.columns:
        data["action"] = pd.Categorical(
            data["action"], categories=ACTION_ORDER, ordered=True
        )
    if "best_action" in data.columns:
        data["best_action"] = pd.Categorical(
            data["best_action"], categories=ACTION_ORDER, ordered=True
        )
    return data


def plot_best_action_counts(policy_dataset: pd.DataFrame, out_path: Path) -> None:
    data = apply_categories(policy_dataset)
    counts = (
        data["best_action"]
        .value_counts()
        .rename_axis("best_action")
        .reset_index(name="wafers")
    )
    counts = apply_categories(counts).sort_values("best_action")

    plt.figure(figsize=(9, 5.2))
    ax = sns.barplot(data=counts, x="best_action", y="wafers", color="#4C78A8")
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + counts["wafers"].max() * 0.01,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xlabel("Best action by dense-reference score")
    plt.ylabel("Wafers")
    plt.title("Policy-Learning Label Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_best_action_by_pattern(policy_dataset: pd.DataFrame, out_path: Path) -> None:
    data = apply_categories(policy_dataset)
    pivot = pd.crosstab(data["failureType"], data["best_action"], normalize="index")
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=ACTION_ORDER)

    plt.figure(figsize=(10, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="crest",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "within-pattern fraction"},
    )
    plt.xlabel("Best action")
    plt.ylabel("Failure type, used only for post-hoc evaluation")
    plt.title("Best Follow-Up Action Distribution by Pattern")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def fixed_and_oracle_summary(
    action_outcomes: pd.DataFrame, policy_dataset: pd.DataFrame
) -> pd.DataFrame:
    fixed = (
        action_outcomes.groupby("action", observed=False)
        .agg(
            mean_sampled_count=("sampled_valid_count", "mean"),
            mean_added_count=("added_valid_count", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    oracle = pd.DataFrame(
        [
            {
                "action": "oracle_best_by_score",
                "mean_sampled_count": policy_dataset[
                    "best_action_sampled_valid_count"
                ].mean(),
                "mean_added_count": policy_dataset[
                    "best_action_added_valid_count"
                ].mean(),
                "mean_absolute_error": policy_dataset[
                    "best_action_absolute_error"
                ].mean(),
                "severe_miss_rate": policy_dataset[
                    "best_action_severe_miss"
                ].mean(),
                "underestimation_rate": policy_dataset[
                    "best_action_underestimated"
                ].mean(),
                "mean_score": policy_dataset["best_action_score"].mean(),
            }
        ]
    )
    return pd.concat([fixed, oracle], ignore_index=True)


def plot_cost_risk_tradeoff(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5.8))
    ax = sns.scatterplot(
        data=summary,
        x="mean_sampled_count",
        y="severe_miss_rate",
        hue="mean_absolute_error",
        size="mean_score",
        sizes=(90, 430),
        palette="viridis",
        edgecolor="black",
        linewidth=0.6,
    )
    for _, row in summary.iterrows():
        ax.text(
            row["mean_sampled_count"] + 0.9,
            row["severe_miss_rate"],
            str(row["action"]),
            fontsize=8,
            va="center",
        )
    plt.xlabel("Mean sampled valid die count")
    plt.ylabel("Severe miss rate")
    plt.title("Fixed Actions vs Oracle Best-Action Cost-Risk")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_first_pass_feature_box(policy_dataset: pd.DataFrame, out_path: Path) -> None:
    data = apply_categories(policy_dataset)
    plt.figure(figsize=(10, 5.5))
    sns.boxplot(
        data=data,
        x="best_action",
        y="first_sampling_density",
        order=ACTION_ORDER,
        color="#72B7B2",
        showfliers=False,
    )
    plt.xlabel("Best action")
    plt.ylabel("First-pass sampling density")
    plt.title("First-Pass Density Distribution by Best Action")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    policy_dataset = pd.read_csv(args.data_dir / "policy_learning_dataset.csv")
    action_outcomes = pd.read_csv(args.data_dir / "action_outcomes.csv")
    summary = fixed_and_oracle_summary(action_outcomes, policy_dataset)
    summary.to_csv(args.data_dir / "fixed_vs_oracle_summary.csv", index=False)

    plot_best_action_counts(
        policy_dataset,
        args.fig_dir / "best_action_distribution.png",
    )
    plot_best_action_by_pattern(
        policy_dataset,
        args.fig_dir / "best_action_by_pattern_heatmap.png",
    )
    plot_cost_risk_tradeoff(
        summary,
        args.fig_dir / "fixed_vs_oracle_cost_risk.png",
    )
    plot_first_pass_feature_box(
        policy_dataset,
        args.fig_dir / "first_pass_density_by_best_action.png",
    )

    print(f"wrote policy-learning QA figures to {args.fig_dir}")
    print(f"wrote fixed-vs-oracle summary: {args.data_dir / 'fixed_vs_oracle_summary.csv'}")


if __name__ == "__main__":
    main()
