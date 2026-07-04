from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY = Path("data") / "processed" / "radial" / "radial_2d_sweep_summary.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "07_radial_sensitivity"

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
    parser = argparse.ArgumentParser(description="Plot radial 2D sweep results.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_pattern_grid(
    summary: pd.DataFrame,
    failure_type: str,
    metric: str,
    title: str,
    out_path: Path,
    cmap: str = "rocket_r",
) -> None:
    data = summary[summary["failureType"] == failure_type]
    pivot = data.pivot(index="outer_radius", columns="angles", values=metric)
    pivot = pivot.sort_index(ascending=False)

    plt.figure(figsize=(6.5, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Angle count")
    plt.ylabel("Outer radius")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_all_pattern_heatmap(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = summary.copy()
    data["config"] = data.apply(
        lambda r: f"r{r['outer_radius']:.2f}_a{int(r['angles'])}", axis=1
    )
    config_order = [
        f"r{r:.2f}_a{a}" for r in [0.75, 0.85, 0.95] for a in [8, 16, 32]
    ]
    pivot = data.pivot(index="failureType", columns="config", values=metric)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=config_order)

    plt.figure(figsize=(13, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="mako_r" if metric == "severe_miss_rate" else "rocket_r",
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Radial config")
    plt.ylabel("Failure type")
    plt.title(f"Radial 2D Sweep: {metric}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk(summary: pd.DataFrame, out_path: Path) -> None:
    config_summary = (
        summary.groupby(["outer_radius", "angles", "variant"])
        .agg(
            mean_sampled_valid_count=("mean_sampled_valid_count", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
            underestimation_rate=("underestimation_rate", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5.8))
    ax = sns.scatterplot(
        data=config_summary,
        x="mean_sampled_valid_count",
        y="mean_absolute_error",
        hue="outer_radius",
        style="angles",
        size="severe_miss_rate",
        sizes=(80, 450),
        edgecolor="black",
        linewidth=0.6,
        palette="viridis",
    )
    for _, row in config_summary.iterrows():
        ax.text(
            row["mean_sampled_valid_count"] + 0.8,
            row["mean_absolute_error"],
            f"r{row['outer_radius']:.2f}/a{int(row['angles'])}",
            fontsize=8,
            va="center",
        )
    plt.xlabel("Mean sampled valid die count")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Radial 2D Sweep Cost-Risk Tradeoff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)

    for failure_type in ["Edge-Ring", "Edge-Loc", "Scratch", "Donut", "Loc"]:
        safe_name = failure_type.lower().replace("-", "_")
        plot_pattern_grid(
            summary,
            failure_type=failure_type,
            metric="severe_miss_rate",
            title=f"{failure_type}: Severe Miss by Outer Radius and Angle Count",
            out_path=args.fig_dir / f"radial_2d_{safe_name}_severe_miss.png",
            cmap="mako_r",
        )
        plot_pattern_grid(
            summary,
            failure_type=failure_type,
            metric="mean_absolute_error",
            title=f"{failure_type}: Absolute Error by Outer Radius and Angle Count",
            out_path=args.fig_dir / f"radial_2d_{safe_name}_absolute_error.png",
        )

    plot_all_pattern_heatmap(
        summary,
        metric="severe_miss_rate",
        out_path=args.fig_dir / "radial_2d_all_patterns_severe_miss.png",
    )
    plot_all_pattern_heatmap(
        summary,
        metric="mean_absolute_error",
        out_path=args.fig_dir / "radial_2d_all_patterns_absolute_error.png",
    )
    plot_cost_risk(
        summary,
        out_path=args.fig_dir / "radial_2d_cost_risk_tradeoff.png",
    )
    print(f"wrote radial 2D sweep figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
