from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY = Path("data") / "processed" / "sampling" / "sampling_summary_by_pattern_with_interior.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "04_edge_exclusion"

PAIR_ORDER = [
    ("grid_5point", "interior_5point", "5-point"),
    ("grid_9point", "interior_9point", "9-point"),
    ("grid_25point", "interior_25point", "25-point"),
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
        description="Plot edge-including grid vs edge-excluding interior grid."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def build_pair_delta(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for grid_scheme, interior_scheme, label in PAIR_ORDER:
        grid = summary[summary["scheme"] == grid_scheme].set_index("failureType")
        interior = summary[summary["scheme"] == interior_scheme].set_index("failureType")
        for failure_type in PATTERN_ORDER:
            if failure_type not in grid.index or failure_type not in interior.index:
                continue
            records.append(
                {
                    "failureType": failure_type,
                    "budget": label,
                    "grid": float(grid.loc[failure_type, metric]),
                    "interior": float(interior.loc[failure_type, metric]),
                    "delta_interior_minus_grid": float(
                        interior.loc[failure_type, metric] - grid.loc[failure_type, metric]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_delta_heatmap(delta: pd.DataFrame, metric_label: str, out_path: Path) -> None:
    pivot = delta.pivot(
        index="failureType", columns="budget", values="delta_interior_minus_grid"
    ).reindex(index=PATTERN_ORDER, columns=["5-point", "9-point", "25-point"])

    plt.figure(figsize=(7.2, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "interior - edge-including grid"},
    )
    plt.xlabel("Sampling budget")
    plt.ylabel("Failure type")
    plt.title(f"Edge Exclusion Effect on {metric_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_grid_vs_interior(summary: pd.DataFrame, out_path: Path) -> None:
    keep_schemes = [
        "grid_5point",
        "interior_5point",
        "grid_9point",
        "interior_9point",
        "grid_25point",
        "interior_25point",
    ]
    plot_df = summary[summary["scheme"].isin(keep_schemes)].copy()
    plot_df["scheme"] = pd.Categorical(plot_df["scheme"], keep_schemes, ordered=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="failureType",
        y="mean_absolute_error",
        hue="scheme",
        order=PATTERN_ORDER,
        hue_order=keep_schemes,
    )
    plt.ylabel("Mean absolute defect-ratio error")
    plt.xlabel("Failure type")
    plt.title("Edge-Including Grid vs Interior Grid")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Scheme", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    abs_delta = build_pair_delta(summary, "mean_absolute_error")
    severe_delta = build_pair_delta(summary, "severe_miss_rate")

    abs_delta_path = args.summary.with_name("edge_exclusion_absolute_error_delta.csv")
    severe_delta_path = args.summary.with_name("edge_exclusion_severe_miss_delta.csv")
    abs_delta.to_csv(abs_delta_path, index=False)
    severe_delta.to_csv(severe_delta_path, index=False)
    print(f"wrote {abs_delta_path}")
    print(f"wrote {severe_delta_path}")

    plot_delta_heatmap(
        abs_delta,
        metric_label="Mean Absolute Error",
        out_path=args.fig_dir / "edge_exclusion_absolute_error_delta.png",
    )
    plot_delta_heatmap(
        severe_delta,
        metric_label="Severe Miss Rate",
        out_path=args.fig_dir / "edge_exclusion_severe_miss_delta.png",
    )
    plot_grid_vs_interior(
        summary,
        out_path=args.fig_dir / "edge_including_vs_interior_grid.png",
    )
    print(f"wrote edge exclusion figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
