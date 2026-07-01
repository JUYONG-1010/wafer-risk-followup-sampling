from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DOMAIN = Path("data") / "processed" / "sampling" / "sampling_summary_by_pattern_with_interior.csv"
DEFAULT_RANDOM = Path("data") / "processed" / "random" / "random_baseline_summary.csv"
DEFAULT_OUT = Path("data") / "processed" / "comparisons" / "domain_vs_random_comparison.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "06_domain_vs_random"

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

SCHEME_BUDGET_MATCH = {
    "grid_5point": 5,
    "interior_5point": 5,
    "grid_9point": 9,
    "interior_9point": 9,
    "grid_25point": 25,
    "interior_25point": 25,
    "edge_biased": 25,
}

SCHEME_ORDER = list(SCHEME_BUDGET_MATCH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare domain-aware sampling summaries against random baselines."
    )
    parser.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def safe_z(delta: float, std: float) -> float:
    if pd.isna(std) or std == 0:
        if delta == 0:
            return 0.0
        return float("inf") if delta > 0 else float("-inf")
    return delta / std


def build_comparison(domain: pd.DataFrame, random_summary: pd.DataFrame) -> pd.DataFrame:
    random_idx = random_summary.set_index(["failureType", "budget"])
    records: list[dict[str, object]] = []

    for scheme, budget in SCHEME_BUDGET_MATCH.items():
        scheme_df = domain[domain["scheme"] == scheme]
        for row in scheme_df.itertuples(index=False):
            key = (row.failureType, budget)
            if key not in random_idx.index:
                continue
            random_row = random_idx.loc[key]

            abs_delta = row.mean_absolute_error - random_row.mean_absolute_error
            severe_delta = row.severe_miss_rate - random_row.mean_severe_miss_rate
            under_delta = row.underestimation_rate - random_row.mean_underestimation_rate

            records.append(
                {
                    "failureType": row.failureType,
                    "scheme": scheme,
                    "matched_random_budget": budget,
                    "domain_mean_sampled_valid_count": row.mean_sampled_valid_count,
                    "domain_absolute_error": row.mean_absolute_error,
                    "random_absolute_error_mean": random_row.mean_absolute_error,
                    "random_absolute_error_std": random_row.std_absolute_error,
                    "absolute_error_delta": abs_delta,
                    "absolute_error_zlike": safe_z(
                        abs_delta, random_row.std_absolute_error
                    ),
                    "domain_severe_miss_rate": row.severe_miss_rate,
                    "random_severe_miss_rate_mean": random_row.mean_severe_miss_rate,
                    "random_severe_miss_rate_std": random_row.std_severe_miss_rate,
                    "severe_miss_delta": severe_delta,
                    "severe_miss_zlike": safe_z(
                        severe_delta, random_row.std_severe_miss_rate
                    ),
                    "domain_underestimation_rate": row.underestimation_rate,
                    "random_underestimation_rate_mean": random_row.mean_underestimation_rate,
                    "random_underestimation_rate_std": random_row.std_underestimation_rate,
                    "underestimation_delta": under_delta,
                    "underestimation_zlike": safe_z(
                        under_delta, random_row.std_underestimation_rate
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


def plot_delta_heatmap(
    comparison: pd.DataFrame,
    value: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = ".2f",
) -> None:
    pivot = comparison.pivot(index="failureType", columns="scheme", values=value)
    pivot = pivot.reindex(index=PATTERN_ORDER, columns=SCHEME_ORDER)

    plt.figure(figsize=(11.5, 5.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
    )
    plt.xlabel("Domain-aware sampling scheme")
    plt.ylabel("Failure type")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_absolute_error_bars(comparison: pd.DataFrame, out_path: Path) -> None:
    plot_df = comparison.copy()
    plot_df["scheme"] = pd.Categorical(plot_df["scheme"], SCHEME_ORDER, ordered=True)
    plot_df["failureType"] = pd.Categorical(
        plot_df["failureType"], PATTERN_ORDER, ordered=True
    )

    plt.figure(figsize=(13, 6))
    sns.barplot(
        data=plot_df,
        x="failureType",
        y="absolute_error_delta",
        hue="scheme",
        order=PATTERN_ORDER,
        hue_order=SCHEME_ORDER,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Failure type")
    plt.ylabel("Absolute error delta vs matched random")
    plt.title("Domain-Aware Sampling vs Random Baseline")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Scheme", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    domain = pd.read_csv(args.domain)
    random_summary = pd.read_csv(args.random)

    comparison = build_comparison(domain, random_summary)
    comparison.to_csv(args.out, index=False)
    print(f"wrote comparison: {args.out}")

    plot_delta_heatmap(
        comparison,
        value="absolute_error_delta",
        title="Absolute Error Delta: Domain-Aware minus Matched Random",
        cbar_label="domain - random",
        out_path=args.fig_dir / "domain_random_absolute_error_delta.png",
    )
    plot_delta_heatmap(
        comparison,
        value="severe_miss_delta",
        title="Severe Miss Delta: Domain-Aware minus Matched Random",
        cbar_label="domain - random",
        out_path=args.fig_dir / "domain_random_severe_miss_delta.png",
    )
    plot_delta_heatmap(
        comparison,
        value="absolute_error_zlike",
        title="Absolute Error Z-like Score vs Random Seed Variability",
        cbar_label="delta / random std",
        out_path=args.fig_dir / "domain_random_absolute_error_zlike.png",
        fmt=".1f",
    )
    plot_delta_heatmap(
        comparison,
        value="severe_miss_zlike",
        title="Severe Miss Z-like Score vs Random Seed Variability",
        cbar_label="delta / random std",
        out_path=args.fig_dir / "domain_random_severe_miss_zlike.png",
        fmt=".1f",
    )
    plot_absolute_error_bars(
        comparison,
        out_path=args.fig_dir / "domain_random_absolute_error_delta_bar.png",
    )
    print(f"wrote domain-vs-random figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
