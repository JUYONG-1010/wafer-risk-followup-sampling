from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_INPUT_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "density_followup_tradeoff_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "38_density_followup_tradeoff_v1"
CLUSTERED_PATTERNS = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze defect-discovery vs ratio-error trade-offs by pattern."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--baseline", type=str, default="coverage32")
    parser.add_argument("--cost-weight", type=float, default=0.003)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "hybrid_guarded1",
            "hybrid_guarded2",
            "hybrid_guarded3",
            "hybrid_guarded4",
            "morphrisk_guarded32",
            "morphrisk32",
            "ml_rank32",
        ],
    )
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(input_dir / "density_followup_eval_summary.csv")
    pattern = pd.read_csv(input_dir / "density_followup_pattern_summary.csv")
    return summary, pattern


def add_tradeoff_metrics(
    data: pd.DataFrame,
    group_cols: list[str],
    baseline: str,
    strategies: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [
        "mean_absolute_error",
        "mean_ratio_error",
        "severe_miss_rate",
        "mean_defect_coverage",
    ]
    for keys, group in data.groupby(group_cols, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = group[group["strategy"] == baseline]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for strategy in strategies:
            cand = group[group["strategy"] == strategy]
            if cand.empty:
                continue
            row = cand.iloc[0]
            record = {col: value for col, value in zip(group_cols, keys, strict=True)}
            record.update(
                {
                    "baseline_strategy": baseline,
                    "strategy": strategy,
                    "wafers": int(row.get("wafers", 0)),
                }
            )
            for metric in metric_cols:
                record[f"baseline_{metric}"] = float(base_row[metric])
                record[f"strategy_{metric}"] = float(row[metric])
                record[f"{metric}_delta"] = float(row[metric] - base_row[metric])

            base_cov = float(base_row["mean_defect_coverage"])
            cov_delta = float(record["mean_defect_coverage_delta"])
            abs_delta = float(record["mean_absolute_error_delta"])
            severe_delta = float(record["severe_miss_rate_delta"])
            record["defect_coverage_relative_improvement_pct"] = (
                cov_delta / base_cov * 100.0 if base_cov else np.nan
            )
            record["coverage_gain_per_abs_error_delta"] = (
                cov_delta / abs_delta if abs_delta > 0 else np.inf if cov_delta > 0 else np.nan
            )
            record["severe_miss_reduction_per_abs_error_delta"] = (
                -severe_delta / abs_delta if abs_delta > 0 else np.inf if severe_delta < 0 else np.nan
            )
            record["strict_abs_error_guardrail_pass"] = bool(abs_delta <= 0)
            record["small_abs_error_delta_pass_0p01"] = bool(abs_delta <= 0.01)
            record["discovery_gain"] = bool(cov_delta > 0 or severe_delta < 0)
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def classify_pattern_tradeoff(tradeoff: pd.DataFrame) -> pd.DataFrame:
    data = tradeoff.copy()
    data["pattern_type"] = data["failureType"].map(
        lambda value: "clustered_structured" if value in CLUSTERED_PATTERNS else "random_like"
    )
    conditions = [
        data["strict_abs_error_guardrail_pass"] & data["discovery_gain"],
        data["small_abs_error_delta_pass_0p01"] & data["discovery_gain"],
        (data["mean_defect_coverage_delta"] > 0)
        & (data["coverage_gain_per_abs_error_delta"] >= 1.0),
        data["mean_defect_coverage_delta"] > 0,
    ]
    labels = [
        "dominant_or_free_gain",
        "useful_low_cost_bias",
        "efficient_tradeoff",
        "costly_discovery_bias",
    ]
    data["tradeoff_class"] = np.select(conditions, labels, default="harmful_or_no_gain")
    return data


def best_by_pattern(pattern_tradeoff: pd.DataFrame) -> pd.DataFrame:
    data = pattern_tradeoff.copy()
    data["rank_abs_penalty"] = data["mean_absolute_error_delta"].clip(lower=0.0)
    data["rank_score"] = (
        5.0 * data["mean_defect_coverage_delta"]
        - 1.0 * data["rank_abs_penalty"]
        - 2.0 * data["severe_miss_rate_delta"]
    )
    return (
        data.sort_values(
            [
                "target_density",
                "failureType",
                "rank_score",
                "coverage_gain_per_abs_error_delta",
            ],
            ascending=[True, True, False, False],
        )
        .groupby(["target_density", "failureType"], observed=False)
        .head(1)
        .reset_index(drop=True)
    )


def plot_global_tradeoff(global_tradeoff: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = global_tradeoff.copy()
    data["density_pct"] = data["target_density"] * 100.0
    plt.figure(figsize=(9.5, 5.4))
    sns.scatterplot(
        data=data,
        x="mean_absolute_error_delta",
        y="defect_coverage_relative_improvement_pct",
        hue="strategy",
        style="density_pct",
        s=90,
    )
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.01, color="#777777", linewidth=0.8, linestyle="--")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Mean absolute-error delta vs coverage32")
    plt.ylabel("Defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "global_discovery_vs_abs_error_tradeoff.png", dpi=180)
    plt.close()


def plot_pattern_heatmaps(pattern_tradeoff: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = pattern_tradeoff[
        pattern_tradeoff["strategy"].isin(["hybrid_guarded1", "hybrid_guarded2", "morphrisk32"])
    ].copy()
    focus["density_pct"] = (focus["target_density"] * 100.0).map(lambda v: f"{v:g}%")
    for metric, filename, title in [
        (
            "defect_coverage_relative_improvement_pct",
            "pattern_coverage_gain_heatmap.png",
            "Pattern-wise defect coverage improvement vs coverage32 (%)",
        ),
        (
            "mean_absolute_error_delta",
            "pattern_absolute_error_delta_heatmap.png",
            "Pattern-wise absolute-error delta vs coverage32",
        ),
        (
            "coverage_gain_per_abs_error_delta",
            "pattern_tradeoff_efficiency_heatmap.png",
            "Pattern-wise coverage gain per absolute-error delta",
        ),
    ]:
        grid = sns.FacetGrid(focus, col="strategy", height=4.0, aspect=0.9)

        def draw_heatmap(data: pd.DataFrame, **_: object) -> None:
            pivot = data.pivot_table(
                index="failureType",
                columns="density_pct",
                values=metric,
                aggfunc="mean",
            )
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="vlag", center=0.0, cbar=True)

        grid.map_dataframe(draw_heatmap)
        grid.fig.suptitle(title, y=1.04)
        grid.fig.tight_layout()
        grid.fig.savefig(fig_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(grid.fig)


def write_markdown_report(
    global_tradeoff: pd.DataFrame,
    pattern_tradeoff: pd.DataFrame,
    best_patterns: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Density Follow-Up Trade-Off Analysis")
    lines.append("")
    lines.append("Baseline: coverage32")
    lines.append("")
    lines.append("## Global Hybrid Guarded1")
    lines.append("")
    focus = global_tradeoff[global_tradeoff["strategy"] == "hybrid_guarded1"].copy()
    for row in focus.sort_values("target_density").itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%} initial: "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"absolute-error delta {row.mean_absolute_error_delta:.4f}, "
            f"small-delta pass={row.small_abs_error_delta_pass_0p01}"
        )
    lines.append("")
    lines.append("## Pattern Takeaways")
    lines.append("")
    pattern_focus = pattern_tradeoff[pattern_tradeoff["strategy"] == "hybrid_guarded1"].copy()
    summary = (
        pattern_focus.groupby("failureType", observed=False)
        .agg(
            mean_coverage_gain_pct=("defect_coverage_relative_improvement_pct", "mean"),
            mean_abs_error_delta=("mean_absolute_error_delta", "mean"),
            low_cost_bias_count=("small_abs_error_delta_pass_0p01", "sum"),
        )
        .reset_index()
        .sort_values("mean_coverage_gain_pct", ascending=False)
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.failureType}: mean coverage gain {row.mean_coverage_gain_pct:.2f}%, "
            f"mean abs-error delta {row.mean_abs_error_delta:.4f}, "
            f"low-cost cases {int(row.low_cost_bias_count)}/4"
        )
    lines.append("")
    lines.append("## Best Strategy By Pattern And Density")
    lines.append("")
    for row in best_patterns.sort_values(["target_density", "failureType"]).itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%} {row.failureType}: {row.strategy}, "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.mean_absolute_error_delta:.4f}, "
            f"class {row.tradeoff_class}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary, pattern = load_inputs(args.input_dir)
    summary = summary[np.isclose(summary["cost_weight"], args.cost_weight)].copy()
    pattern = pattern[np.isclose(pattern["cost_weight"], args.cost_weight)].copy()

    global_tradeoff = add_tradeoff_metrics(
        summary,
        group_cols=["cost_weight", "target_density", "first_pass_type"],
        baseline=args.baseline,
        strategies=args.strategies,
    )
    pattern_tradeoff = add_tradeoff_metrics(
        pattern,
        group_cols=["cost_weight", "target_density", "failureType", "first_pass_type"],
        baseline=args.baseline,
        strategies=args.strategies,
    )
    pattern_tradeoff = classify_pattern_tradeoff(pattern_tradeoff)
    best_patterns = best_by_pattern(pattern_tradeoff)

    global_tradeoff.to_csv(args.out_dir / "density_followup_global_tradeoff.csv", index=False)
    pattern_tradeoff.to_csv(args.out_dir / "density_followup_pattern_tradeoff.csv", index=False)
    best_patterns.to_csv(args.out_dir / "density_followup_best_strategy_by_pattern.csv", index=False)
    write_markdown_report(
        global_tradeoff,
        pattern_tradeoff,
        best_patterns,
        args.out_dir / "density_followup_tradeoff_report.md",
    )

    plot_global_tradeoff(global_tradeoff, args.fig_dir)
    plot_pattern_heatmaps(pattern_tradeoff, args.fig_dir)

    print(f"wrote trade-off outputs to {args.out_dir}")
    print(f"wrote trade-off figures to {args.fig_dir}")
    print(
        global_tradeoff[
            [
                "target_density",
                "strategy",
                "mean_absolute_error_delta",
                "defect_coverage_relative_improvement_pct",
                "coverage_gain_per_abs_error_delta",
                "small_abs_error_delta_pass_0p01",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
