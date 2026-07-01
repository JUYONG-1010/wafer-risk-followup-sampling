from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_AUG_DIR = Path("data") / "processed" / "augmented_morphology_policy_aligned_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "augmented_gated_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "51_augmented_gated_policy_v1"

AVAILABLE_STRATEGIES = [
    "coverage32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
    "ml_rank32",
    "ml_biasaware32",
    "morphrisk_guarded32",
]
DISCOVERY_PATTERNS = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"}
LOW_BIAS_PATTERNS = {"Random"}
COVERAGE_PATTERNS = {"Near-full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence-gated policy using augmented morphology predictions."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--aug-dir", type=Path, default=DEFAULT_AUG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--low-bias-delta", type=float, default=0.01)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.7, 0.8])
    return parser.parse_args()


def pattern_mode(pattern: str) -> str:
    if pattern in DISCOVERY_PATTERNS:
        return "discovery_first"
    if pattern in LOW_BIAS_PATTERNS:
        return "low_bias_default"
    if pattern in COVERAGE_PATTERNS:
        return "coverage32_or_uncertain"
    return "low_bias_default"


def add_pattern_baseline(data: pd.DataFrame) -> pd.DataFrame:
    baseline = data[data["strategy"] == "coverage32"][
        ["target_density", "failureType", "mean_absolute_error", "mean_defect_coverage"]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
        }
    )
    out = data.merge(baseline, on=["target_density", "failureType"], how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["defect_coverage_relative_improvement_pct"] = (
        (out["mean_defect_coverage"] - out["baseline_defect_coverage"])
        / out["baseline_defect_coverage"]
        * 100.0
    )
    return out


def build_strategy_map(pattern_summary: pd.DataFrame, low_bias_delta: float) -> pd.DataFrame:
    data = pattern_summary[pattern_summary["strategy"].isin(AVAILABLE_STRATEGIES)].copy()
    data = add_pattern_baseline(data)
    rows: list[dict[str, object]] = []
    for (density, pattern), group in data.groupby(["target_density", "failureType"], observed=False):
        positive = group[
            (group["strategy"] != "coverage32")
            & (group["defect_coverage_relative_improvement_pct"] > 0)
        ].copy()
        if positive.empty:
            discovery_strategy = "coverage32"
            low_bias_strategy = "coverage32"
        else:
            discovery_strategy = positive.sort_values(
                ["defect_coverage_relative_improvement_pct", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0]["strategy"]
            low_bias_pool = positive[positive["absolute_error_delta"] <= low_bias_delta]
            if low_bias_pool.empty:
                low_bias_strategy = "coverage32"
            else:
                low_bias_strategy = low_bias_pool.sort_values(
                    ["defect_coverage_relative_improvement_pct", "absolute_error_delta"],
                    ascending=[False, True],
                ).iloc[0]["strategy"]
        rows.append(
            {
                "target_density": density,
                "pattern": pattern,
                "pattern_mode": pattern_mode(pattern),
                "discovery_strategy": discovery_strategy,
                "low_bias_strategy": low_bias_strategy,
            }
        )
    return pd.DataFrame(rows)


def choose_strategy(
    predicted_pattern: str,
    confidence: float,
    threshold: float,
    density: float,
    lookup: dict[tuple[float, str], dict[str, str]],
) -> tuple[str, str]:
    entry = lookup.get((float(density), str(predicted_pattern)))
    if entry is None:
        return "coverage32", "fallback_unknown_pattern"
    mode = pattern_mode(str(predicted_pattern))
    if mode == "coverage32_or_uncertain":
        return "coverage32", "coverage_uncertain"
    if mode == "low_bias_default":
        return entry["low_bias_strategy"], "predicted_low_bias"
    if float(confidence) >= threshold:
        return entry["discovery_strategy"], "gated_discovery"
    return entry["low_bias_strategy"], "gated_low_bias_fallback"


def build_views(
    eval_results: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy_map: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    lookup = {
        (float(row.target_density), str(row.pattern)): {
            "discovery_strategy": str(row.discovery_strategy),
            "low_bias_strategy": str(row.low_bias_strategy),
        }
        for row in strategy_map.itertuples(index=False)
    }
    rows: list[pd.DataFrame] = []
    base = eval_results[eval_results["strategy"] == "coverage32"].copy()
    base["variant"] = "coverage32"
    base["confidence_threshold"] = -1.0
    base["selected_strategy"] = "coverage32"
    base["selection_reason"] = "baseline"
    rows.append(base)

    pred = predictions.rename(
        columns={
            "pred_label": "predicted_pattern",
            "top1_confidence": "predicted_confidence",
        }
    )
    pred = pred[["row_index", "target_density", "variant", "predicted_pattern", "predicted_confidence"]]
    for threshold in thresholds:
        selected = pred.copy()
        choices = selected.apply(
            lambda row: choose_strategy(
                row["predicted_pattern"],
                row["predicted_confidence"],
                threshold,
                row["target_density"],
                lookup,
            ),
            axis=1,
            result_type="expand",
        )
        selected["selected_strategy"] = choices[0]
        selected["selection_reason"] = choices[1]
        merged = eval_results.merge(
            selected[
                [
                    "row_index",
                    "target_density",
                    "variant",
                    "predicted_pattern",
                    "predicted_confidence",
                    "selected_strategy",
                    "selection_reason",
                ]
            ],
            on=["row_index", "target_density"],
            how="inner",
        )
        merged = merged[merged["strategy"] == merged["selected_strategy"]].copy()
        merged["confidence_threshold"] = float(threshold)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def summarize(views: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        views.groupby(["variant", "confidence_threshold", "target_density"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            discovery_selection_rate=("selection_reason", lambda values: (values == "gated_discovery").mean()),
            low_bias_fallback_rate=("selection_reason", lambda values: (values == "gated_low_bias_fallback").mean()),
        )
        .reset_index()
    )
    baseline = summary[summary["variant"] == "coverage32"][
        ["target_density", "mean_absolute_error", "mean_defect_coverage", "mean_sampled_defects"]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
            "mean_sampled_defects": "baseline_sampled_defects",
        }
    )
    summary = summary.merge(baseline, on="target_density", how="left")
    summary["absolute_error_delta"] = summary["mean_absolute_error"] - summary["baseline_absolute_error"]
    summary["defect_coverage_relative_improvement_pct"] = (
        (summary["mean_defect_coverage"] - summary["baseline_defect_coverage"])
        / summary["baseline_defect_coverage"]
        * 100.0
    )
    summary["sampled_defects_delta"] = summary["mean_sampled_defects"] - summary["baseline_sampled_defects"]

    comparison = summary[summary["variant"].isin(["baseline", "augmented"])].copy()
    pivot_cols = [
        "defect_coverage_relative_improvement_pct",
        "absolute_error_delta",
        "sampled_defects_delta",
        "discovery_selection_rate",
    ]
    wide = comparison.pivot_table(
        index=["confidence_threshold", "target_density"],
        columns="variant",
        values=pivot_cols,
        aggfunc="mean",
    )
    records: list[dict[str, object]] = []
    for idx in wide.index:
        record = {
            "confidence_threshold": idx[0],
            "target_density": idx[1],
        }
        for metric in pivot_cols:
            base_val = wide.get((metric, "baseline"), pd.Series(dtype=float)).get(idx, pd.NA)
            aug_val = wide.get((metric, "augmented"), pd.Series(dtype=float)).get(idx, pd.NA)
            record[f"baseline_{metric}"] = base_val
            record[f"augmented_{metric}"] = aug_val
            record[f"delta_{metric}"] = aug_val - base_val
        records.append(record)
    return summary, pd.DataFrame.from_records(records)


def plot_outputs(summary: pd.DataFrame, comparison: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary[summary["variant"].isin(["baseline", "augmented"])].copy()
    data["density_pct"] = data["target_density"] * 100.0
    for metric, filename, ylabel in [
        ("defect_coverage_relative_improvement_pct", "augmented_gated_coverage_gain.png", "Coverage gain vs coverage32 (%)"),
        ("absolute_error_delta", "augmented_gated_abs_error_delta.png", "Absolute-error delta vs coverage32"),
        ("sampled_defects_delta", "augmented_gated_sampled_defects_delta.png", "Sampled defects delta vs coverage32"),
    ]:
        plt.figure(figsize=(9.2, 5.4))
        sns.lineplot(
            data=data,
            x="confidence_threshold",
            y=metric,
            hue="density_pct",
            style="variant",
            marker="o",
            palette="viridis",
        )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Confidence threshold")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    comp = comparison.copy()
    comp["density_pct"] = comp["target_density"] * 100.0
    plt.figure(figsize=(8.6, 5.2))
    sns.lineplot(
        data=comp,
        x="confidence_threshold",
        y="delta_defect_coverage_relative_improvement_pct",
        hue="density_pct",
        marker="o",
        palette="viridis",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Augmented - baseline coverage gain (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "augmentation_delta_coverage_gain.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Augmented Confidence-Gated Policy v1",
        "",
        "Purpose: test whether train-only geometric augmentation improves the deployable confidence-gated policy on held-out wafers.",
        "",
        "Important: this is split-based evaluation, not all-data model evaluation.",
        "",
        "## Threshold 0.60 Comparison",
        "",
    ]
    focus = comparison[comparison["confidence_threshold"] == 0.60]
    for row in focus.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: "
            f"baseline gain {row.baseline_defect_coverage_relative_improvement_pct:.2f}%, "
            f"augmented gain {row.augmented_defect_coverage_relative_improvement_pct:.2f}%, "
            f"delta {row.delta_defect_coverage_relative_improvement_pct:.2f}%, "
            f"baseline abs-delta {row.baseline_absolute_error_delta:.4f}, "
            f"augmented abs-delta {row.augmented_absolute_error_delta:.4f}"
        )
    lines.extend(["", "## All Thresholds", ""])
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, threshold {row.confidence_threshold:.2f}: "
            f"coverage-gain delta {row.delta_defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error-delta change {row.delta_absolute_error_delta:.4f}, "
            f"sampled-defects delta change {row.delta_sampled_defects_delta:.2f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If augmentation improves mode prediction but not gated-policy discovery, the bottleneck is downstream strategy scoring rather than morphology classification.",
            "If augmentation improves both, integrate train-only augmentation into final morphology training and then consider point-risk augmentation.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    eval_results = pd.read_csv(args.eval_dir / "density_followup_eval_results.csv")
    eval_results = eval_results[
        (eval_results["cost_weight"] == eval_results["cost_weight"].min())
        & (eval_results["strategy"].isin(AVAILABLE_STRATEGIES))
    ].copy()
    pattern_summary = pd.read_csv(args.eval_dir / "density_followup_pattern_summary.csv")
    pattern_summary = pattern_summary[
        (pattern_summary["cost_weight"] == pattern_summary["cost_weight"].min())
        & (pattern_summary["strategy"].isin(AVAILABLE_STRATEGIES))
    ].copy()
    predictions = pd.read_csv(args.aug_dir / "augmented_morphology_predictions.csv")

    strategy_map = build_strategy_map(pattern_summary, args.low_bias_delta)
    views = build_views(eval_results, predictions, strategy_map, args.thresholds)
    summary, comparison = summarize(views)

    strategy_map.to_csv(args.out_dir / "augmented_gated_strategy_map.csv", index=False)
    views.to_csv(args.out_dir / "augmented_gated_policy_results.csv", index=False)
    summary.to_csv(args.out_dir / "augmented_gated_policy_summary.csv", index=False)
    comparison.to_csv(args.out_dir / "augmented_gated_policy_comparison.csv", index=False)
    plot_outputs(summary, comparison, args.fig_dir)
    write_report(summary, comparison, args.out_dir / "augmented_gated_policy_report.md")

    print(f"wrote augmented gated policy outputs to {args.out_dir}")
    print(f"wrote augmented gated policy figures to {args.fig_dir}")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
