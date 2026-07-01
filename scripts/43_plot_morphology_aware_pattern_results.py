from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = (
    Path("data")
    / "processed"
    / "morphology_aware_policy_v1"
    / "morphology_aware_pattern_summary.csv"
)
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "26_morphology_aware_pattern_v1"

STRATEGY_ORDER = ["coverage32", "ml_rank32", "ml_biasaware32", "morphrisk32"]
WEAK_PATTERNS = ["Loc", "Random", "Scratch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pattern-level morphology-aware policy results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cost-weight", type=float, default=0.003)
    return parser.parse_args()


def plot_metric(data: pd.DataFrame, metric: str, out_path: Path) -> None:
    grid = sns.catplot(
        data=data,
        x="strategy",
        y=metric,
        hue="failureType",
        col="first_pass_type",
        kind="bar",
        order=STRATEGY_ORDER,
        height=4.2,
        aspect=1.35,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    grid.set_axis_labels("Strategy", metric.replace("_", " "))
    grid.set_titles("{col_name}")
    grid.fig.suptitle(metric.replace("_", " ").title() + " on Weak Patterns", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.input)
    data = summary[
        (summary["cost_weight"] == args.cost_weight)
        & (summary["failureType"].isin(WEAK_PATTERNS))
        & (summary["strategy"].isin(STRATEGY_ORDER))
    ].copy()
    data["strategy"] = pd.Categorical(data["strategy"], STRATEGY_ORDER, ordered=True)
    data["failureType"] = pd.Categorical(data["failureType"], WEAK_PATTERNS, ordered=True)

    plot_metric(data, "mean_defect_coverage", args.fig_dir / "weak_pattern_defect_coverage.png")
    plot_metric(data, "severe_miss_rate", args.fig_dir / "weak_pattern_severe_miss.png")
    plot_metric(data, "mean_absolute_error", args.fig_dir / "weak_pattern_absolute_error_guardrail.png")

    print(f"wrote morphology-aware pattern figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
