from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = (
    Path("data")
    / "processed"
    / "point_ranking_bias_v2"
    / "point_ranking_bias_strategy_summary.csv"
)
DEFAULT_OUT_DIR = Path("reports") / "final_policy_improvement_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "23_final_policy_improvement_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final baseline-vs-proposed improvement summary."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cost-weight", type=float, default=0.003)
    parser.add_argument("--proposed", default="ml_biasaware32")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["first_only", "coverage32"],
        help="Baseline strategies to compare against the proposed strategy.",
    )
    return parser.parse_args()


def pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return float("nan")
    return (new_value - old_value) / old_value * 100.0


def pct_reduction(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return float("nan")
    return (old_value - new_value) / old_value * 100.0


def build_improvement_table(
    summary: pd.DataFrame,
    cost_weight: float,
    proposed_strategy: str,
    baseline_strategies: list[str],
) -> pd.DataFrame:
    data = summary[summary["cost_weight"] == cost_weight].copy()
    records: list[dict[str, object]] = []

    for first_pass_type in sorted(data["first_pass_type"].unique()):
        proposed_rows = data[
            (data["first_pass_type"] == first_pass_type)
            & (data["strategy"] == proposed_strategy)
        ]
        if proposed_rows.empty:
            continue
        proposed = proposed_rows.iloc[0]

        for baseline_strategy in baseline_strategies:
            baseline_rows = data[
                (data["first_pass_type"] == first_pass_type)
                & (data["strategy"] == baseline_strategy)
            ]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]

            records.append(
                {
                    "cost_weight": cost_weight,
                    "first_pass_type": first_pass_type,
                    "baseline_strategy": baseline_strategy,
                    "proposed_strategy": proposed_strategy,
                    "baseline_severe_miss_rate": baseline["severe_miss_rate"],
                    "proposed_severe_miss_rate": proposed["severe_miss_rate"],
                    "severe_miss_relative_reduction_pct": pct_reduction(
                        proposed["severe_miss_rate"], baseline["severe_miss_rate"]
                    ),
                    "baseline_defect_coverage": baseline["mean_defect_coverage"],
                    "proposed_defect_coverage": proposed["mean_defect_coverage"],
                    "defect_coverage_relative_improvement_pct": pct_change(
                        proposed["mean_defect_coverage"],
                        baseline["mean_defect_coverage"],
                    ),
                    "baseline_absolute_error": baseline["mean_absolute_error"],
                    "proposed_absolute_error": proposed["mean_absolute_error"],
                    "absolute_error_delta": proposed["mean_absolute_error"]
                    - baseline["mean_absolute_error"],
                    "baseline_mean_ratio_error": baseline["mean_ratio_error"],
                    "proposed_mean_ratio_error": proposed["mean_ratio_error"],
                    "mean_ratio_error_delta": proposed["mean_ratio_error"]
                    - baseline["mean_ratio_error"],
                    "baseline_overestimation_rate": baseline["overestimation_rate"],
                    "proposed_overestimation_rate": proposed["overestimation_rate"],
                    "overestimation_rate_delta": proposed["overestimation_rate"]
                    - baseline["overestimation_rate"],
                }
            )
    return pd.DataFrame.from_records(records)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    rows = data.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(table: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Final Policy Improvement Summary",
        "",
        "Final objective: recommend follow-up sampling points from first-pass sparse observations to capture high-risk defect regions better than fixed baselines.",
        "",
        "Primary metrics:",
        "",
        "- severe miss relative reduction",
        "- defect coverage relative improvement",
        "",
        "Guardrail metrics:",
        "",
        "- absolute error",
        "- mean ratio error",
        "- overestimation rate",
        "",
        "## Improvement Table",
        "",
    ]

    display_cols = [
        "first_pass_type",
        "baseline_strategy",
        "proposed_strategy",
        "baseline_severe_miss_rate",
        "proposed_severe_miss_rate",
        "severe_miss_relative_reduction_pct",
        "baseline_defect_coverage",
        "proposed_defect_coverage",
        "defect_coverage_relative_improvement_pct",
        "absolute_error_delta",
        "mean_ratio_error_delta",
    ]
    rounded = table[display_cols].copy()
    for col in rounded.columns:
        if col.endswith("_pct"):
            rounded[col] = rounded[col].map(format_pct)
        elif pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")
    lines.append(dataframe_to_markdown(rounded))
    lines.extend(
        [
            "",
            "## Portfolio-Safe Claim",
            "",
            "Use the claim only for high-risk defect discovery, not for wafer-level defect-ratio estimation.",
            "",
            "Example:",
            "",
            "> Under the same follow-up budget, the proposed bias-aware ML sampling policy reduced severe miss rate and improved defect coverage compared with fixed coverage sampling, while ratio bias was monitored as a guardrail.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_pairs(table: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_records: list[dict[str, object]] = []
    for _, row in table.iterrows():
        label = f"{row['first_pass_type']} vs {row['baseline_strategy']}"
        long_records.extend(
            [
                {
                    "comparison": label,
                    "strategy": "baseline",
                    "metric": "severe_miss_rate",
                    "value": row["baseline_severe_miss_rate"],
                },
                {
                    "comparison": label,
                    "strategy": "proposed",
                    "metric": "severe_miss_rate",
                    "value": row["proposed_severe_miss_rate"],
                },
                {
                    "comparison": label,
                    "strategy": "baseline",
                    "metric": "defect_coverage",
                    "value": row["baseline_defect_coverage"],
                },
                {
                    "comparison": label,
                    "strategy": "proposed",
                    "metric": "defect_coverage",
                    "value": row["proposed_defect_coverage"],
                },
            ]
        )
    long = pd.DataFrame.from_records(long_records)

    for metric, filename, ylabel in [
        ("severe_miss_rate", "baseline_vs_proposed_severe_miss.png", "Severe miss rate"),
        ("defect_coverage", "baseline_vs_proposed_defect_coverage.png", "Defect coverage"),
    ]:
        data = long[long["metric"] == metric]
        plt.figure(figsize=(10.8, 5.2))
        sns.barplot(data=data, x="comparison", y="value", hue="strategy")
        plt.xlabel("Comparison")
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    improvement = table.melt(
        id_vars=["first_pass_type", "baseline_strategy", "proposed_strategy"],
        value_vars=[
            "severe_miss_relative_reduction_pct",
            "defect_coverage_relative_improvement_pct",
        ],
        var_name="metric",
        value_name="improvement_pct",
    )
    improvement["comparison"] = (
        improvement["first_pass_type"] + " vs " + improvement["baseline_strategy"]
    )
    plt.figure(figsize=(11.2, 5.5))
    sns.barplot(data=improvement, x="comparison", y="improvement_pct", hue="metric")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Comparison")
    plt.ylabel("Relative improvement / reduction (%)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "final_relative_improvement_summary.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.input)
    table = build_improvement_table(
        summary=summary,
        cost_weight=args.cost_weight,
        proposed_strategy=args.proposed,
        baseline_strategies=args.baselines,
    )
    table = table.sort_values(["first_pass_type", "baseline_strategy"])

    table.to_csv(args.out_dir / "final_improvement_summary.csv", index=False)
    write_markdown(table, args.out_dir / "final_improvement_summary.md")
    plot_metric_pairs(table, args.fig_dir)

    print(f"wrote final improvement table to {args.out_dir}")
    print(f"wrote final improvement figures to {args.fig_dir}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
