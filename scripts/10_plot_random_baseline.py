from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RANDOM = Path("data") / "processed" / "random" / "random_baseline_by_seed.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "05_random_baseline"

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
    parser = argparse.ArgumentParser(description="Plot repeated random baseline summaries.")
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--suffix", type=str, default="")
    return parser.parse_args()


def suffix_name(base: str, suffix: str) -> str:
    if suffix:
        return f"{base}_{suffix}.png"
    return f"{base}.png"


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.random)
    df["failureType"] = pd.Categorical(df["failureType"], PATTERN_ORDER, ordered=True)

    plt.figure(figsize=(10, 5.8))
    sns.lineplot(
        data=df,
        x="budget",
        y="mean_absolute_error",
        hue="failureType",
        hue_order=PATTERN_ORDER,
        marker="o",
        errorbar="sd",
    )
    plt.xlabel("Random sampling budget")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Repeated Random Baseline: Absolute Error")
    plt.tight_layout()
    plt.savefig(args.fig_dir / suffix_name("random_baseline_absolute_error", args.suffix), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.8))
    sns.lineplot(
        data=df,
        x="budget",
        y="severe_miss_rate",
        hue="failureType",
        hue_order=PATTERN_ORDER,
        marker="o",
        errorbar="sd",
    )
    plt.xlabel("Random sampling budget")
    plt.ylabel("Severe miss rate")
    plt.title("Repeated Random Baseline: Severe Miss Rate")
    plt.tight_layout()
    plt.savefig(args.fig_dir / suffix_name("random_baseline_severe_miss", args.suffix), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.8))
    sns.boxplot(
        data=df,
        x="failureType",
        y="mean_absolute_error",
        hue="budget",
        order=PATTERN_ORDER,
    )
    plt.xlabel("Failure type")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Random Baseline Variance by Pattern")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.fig_dir / suffix_name("random_variance_by_pattern", args.suffix), dpi=180)
    plt.close()

    print(f"wrote random baseline figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
