from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_DIR = Path("data") / "processed" / "size_density"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "08_size_density"

SIZE_BIN_ORDER = ["Q1_small", "Q2_mid_small", "Q3_mid_large", "Q4_large"]
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
BASELINE_SCHEMES = [
    "grid_5point",
    "grid_9point",
    "grid_25point",
    "interior_25point",
    "radial",
    "edge_biased",
    "random_25",
    "adaptive_9point",
]
HEATMAP_SCHEMES = ["grid_25point", "radial", "edge_biased", "random_25", "adaptive_9point"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot wafer-size sampling-density analysis.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def prepare_categories(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "size_bin" in data.columns:
        data["size_bin"] = pd.Categorical(
            data["size_bin"], SIZE_BIN_ORDER, ordered=True
        )
    if "failureType" in data.columns:
        data["failureType"] = pd.Categorical(
            data["failureType"], PATTERN_ORDER, ordered=True
        )
    if "scheme" in data.columns:
        data["scheme"] = pd.Categorical(
            data["scheme"], BASELINE_SCHEMES, ordered=True
        )
    return data


def plot_wafer_size_bins(wafer_bins: pd.DataFrame, out_path: Path) -> None:
    wafer_bins = prepare_categories(wafer_bins)

    plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(
        data=wafer_bins,
        x="size_bin",
        y="wafers",
        color="#4C78A8",
    )
    for _, row in wafer_bins.iterrows():
        ax.text(
            row.name,
            row["wafers"] + wafer_bins["wafers"].max() * 0.015,
            f"{int(row['min_valid_die_count'])}-{int(row['max_valid_die_count'])}",
            ha="center",
            fontsize=9,
        )
    plt.xlabel("Wafer size bin by valid die count")
    plt.ylabel("Number of wafers")
    plt.title("Patterned Wafer Size Bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_metric_lines(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plot_df = summary[
        (summary["source"] == "sampling_schemes")
        & (summary["scheme"].isin(BASELINE_SCHEMES))
    ].copy()
    plot_df = prepare_categories(plot_df)

    plt.figure(figsize=(10.5, 5.8))
    sns.lineplot(
        data=plot_df,
        x="size_bin",
        y=metric,
        hue="scheme",
        hue_order=BASELINE_SCHEMES,
        marker="o",
    )
    plt.xlabel("Wafer size bin")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Scheme", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cost_risk_by_size(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary[
        (summary["source"] == "sampling_schemes")
        & (summary["scheme"].isin(BASELINE_SCHEMES))
    ].copy()
    plot_df = prepare_categories(plot_df)

    plt.figure(figsize=(9, 5.8))
    ax = sns.scatterplot(
        data=plot_df,
        x="mean_sampling_density",
        y="mean_absolute_error",
        hue="size_bin",
        style="scheme",
        size="severe_miss_rate",
        sizes=(60, 360),
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Mean sampling density")
    plt.ylabel("Mean absolute defect-ratio error")
    plt.title("Sampling Density vs Error by Wafer Size Bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_heatmap(
    pattern_summary: pd.DataFrame,
    scheme: str,
    metric: str,
    title: str,
    out_path: Path,
    cmap: str,
) -> None:
    plot_df = pattern_summary[
        (pattern_summary["source"] == "sampling_schemes")
        & (pattern_summary["scheme"] == scheme)
    ].copy()
    plot_df = prepare_categories(plot_df)
    pivot = plot_df.pivot(index="failureType", columns="size_bin", values=metric)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=SIZE_BIN_ORDER)

    plt.figure(figsize=(7.5, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Wafer size bin")
    plt.ylabel("Failure type")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_largest_trends(trends: pd.DataFrame, out_path: Path) -> None:
    plot_df = trends[
        (trends["source"] == "sampling_schemes")
        & (trends["scheme"].isin(BASELINE_SCHEMES))
    ].copy()
    plot_df["label"] = plot_df["scheme"].astype(str) + " / " + plot_df["failureType"].astype(str)
    plot_df = plot_df.sort_values(
        "severe_miss_delta_large_minus_small", ascending=False
    ).head(15)

    plt.figure(figsize=(10, 5.8))
    sns.barplot(
        data=plot_df,
        y="label",
        x="severe_miss_delta_large_minus_small",
        color="#E45756",
    )
    plt.xlabel("Severe miss delta: Q4 large - Q1 small")
    plt.ylabel("Scheme / failure type")
    plt.title("Largest Severe-Miss Increase from Small to Large Wafers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    wafer_bins = pd.read_csv(args.data_dir / "wafer_size_bin_summary.csv")
    scheme_summary = pd.read_csv(args.data_dir / "scheme_size_summary.csv")
    pattern_summary = pd.read_csv(args.data_dir / "pattern_size_summary.csv")
    trends = pd.read_csv(args.data_dir / "size_risk_trends.csv")

    plot_wafer_size_bins(
        wafer_bins,
        args.fig_dir / "wafer_size_bin_distribution.png",
    )
    plot_metric_lines(
        scheme_summary,
        metric="mean_sampling_density",
        ylabel="Mean sampling density",
        title="Effective Sampling Density Shrinks on Larger Wafers",
        out_path=args.fig_dir / "sampling_density_by_size_bin.png",
    )
    plot_metric_lines(
        scheme_summary,
        metric="mean_absolute_error",
        ylabel="Mean absolute defect-ratio error",
        title="Defect-Ratio Error by Wafer Size Bin",
        out_path=args.fig_dir / "absolute_error_by_size_bin.png",
    )
    plot_metric_lines(
        scheme_summary,
        metric="severe_miss_rate",
        ylabel="Severe miss rate",
        title="Severe Miss Rate by Wafer Size Bin",
        out_path=args.fig_dir / "severe_miss_by_size_bin.png",
    )
    plot_cost_risk_by_size(
        scheme_summary,
        args.fig_dir / "density_vs_error_by_size_bin.png",
    )
    plot_largest_trends(
        trends,
        args.fig_dir / "largest_severe_miss_size_trends.png",
    )

    for scheme in HEATMAP_SCHEMES:
        safe_scheme = scheme.replace("_", "-")
        plot_pattern_heatmap(
            pattern_summary,
            scheme=scheme,
            metric="severe_miss_rate",
            title=f"{scheme}: Severe Miss by Pattern and Wafer Size",
            out_path=args.fig_dir / f"{safe_scheme}_pattern_size_severe_miss.png",
            cmap="mako_r",
        )
        plot_pattern_heatmap(
            pattern_summary,
            scheme=scheme,
            metric="mean_absolute_error",
            title=f"{scheme}: Absolute Error by Pattern and Wafer Size",
            out_path=args.fig_dir / f"{safe_scheme}_pattern_size_absolute_error.png",
            cmap="rocket_r",
        )

    print(f"wrote size-density figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
