from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "confidence_gated_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "48_confidence_gated_policy_v1"

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
        description="Evaluate confidence-gated predicted-pattern follow-up policy."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
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
    out["defect_coverage_delta"] = (
        out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    )
    out["defect_coverage_relative_improvement_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
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


def selected_strategy_for_prediction(
    predicted_pattern: str,
    confidence: float,
    threshold: float,
    strategy_lookup: dict[tuple[float, str], dict[str, str]],
    density: float,
) -> tuple[str, str]:
    entry = strategy_lookup.get((density, predicted_pattern))
    if entry is None:
        return "coverage32", "fallback_unknown_pattern"

    mode = pattern_mode(predicted_pattern)
    if mode == "coverage32_or_uncertain":
        return "coverage32", "coverage_uncertain"
    if mode == "low_bias_default":
        return entry["low_bias_strategy"], "predicted_low_bias"
    if confidence >= threshold:
        return entry["discovery_strategy"], "gated_discovery"
    return entry["low_bias_strategy"], "gated_low_bias_fallback"


def build_gated_views(
    eval_results: pd.DataFrame,
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
    base["confidence_threshold"] = -1.0
    base["policy_view"] = "coverage32"
    base["selected_strategy"] = "coverage32"
    base["selection_reason"] = "baseline"
    rows.append(base)

    key_cols = ["row_index", "target_density", "morph_top1", "morph_confidence"]
    wafer_predictions = eval_results[eval_results["strategy"] == "coverage32"][key_cols].drop_duplicates()
    for threshold in thresholds:
        selected = wafer_predictions.copy()
        choices = selected.apply(
            lambda row: selected_strategy_for_prediction(
                str(row["morph_top1"]),
                float(row["morph_confidence"]),
                float(threshold),
                lookup,
                float(row["target_density"]),
            ),
            axis=1,
            result_type="expand",
        )
        selected["selected_strategy"] = choices[0]
        selected["selection_reason"] = choices[1]
        merged = eval_results.merge(
            selected[["row_index", "target_density", "selected_strategy", "selection_reason"]],
            on=["row_index", "target_density"],
            how="inner",
        )
        merged = merged[merged["strategy"] == merged["selected_strategy"]].copy()
        merged["confidence_threshold"] = float(threshold)
        merged["policy_view"] = "confidence_gated"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def summarize(views: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        views.groupby(["confidence_threshold", "policy_view", "target_density"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            gated_discovery_rate=("selection_reason", lambda values: (values == "gated_discovery").mean()),
            low_bias_fallback_rate=("selection_reason", lambda values: (values == "gated_low_bias_fallback").mean()),
            predicted_low_bias_rate=("selection_reason", lambda values: (values == "predicted_low_bias").mean()),
            coverage_uncertain_rate=("selection_reason", lambda values: (values == "coverage_uncertain").mean()),
        )
        .reset_index()
    )
    baseline = summary[summary["policy_view"] == "coverage32"][
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
    summary["sampled_defects_delta"] = (
        summary["mean_sampled_defects"] - summary["baseline_sampled_defects"]
    )

    pattern = (
        views.groupby(
            ["confidence_threshold", "policy_view", "target_density", "failureType"],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
        )
        .reset_index()
    )
    return summary, pattern


def plot_outputs(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    gated = summary[summary["policy_view"] == "confidence_gated"].copy()
    gated["density_pct"] = gated["target_density"] * 100.0
    for metric, filename, ylabel in [
        ("defect_coverage_relative_improvement_pct", "gated_coverage_gain.png", "Coverage gain vs coverage32 (%)"),
        ("absolute_error_delta", "gated_abs_error_delta.png", "Absolute-error delta vs coverage32"),
        ("sampled_defects_delta", "gated_sampled_defects_delta.png", "Mean sampled defects delta vs coverage32"),
        ("gated_discovery_rate", "gated_discovery_rate.png", "Discovery-first selection rate"),
    ]:
        plt.figure(figsize=(8.8, 5.2))
        sns.lineplot(
            data=gated,
            x="confidence_threshold",
            y=metric,
            hue="density_pct",
            marker="o",
            palette="viridis",
        )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Confidence threshold")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    gated = summary[summary["policy_view"] == "confidence_gated"].copy()
    lines = [
        "# Confidence-Gated Predicted Pattern Policy v1",
        "",
        "Policy logic:",
        "",
        "```text",
        "1. predict morphology from first-pass sparse data",
        "2. if predicted pattern is discovery-first and confidence >= threshold:",
        "     use discovery-first strategy",
        "3. otherwise:",
        "     use low-bias strategy or coverage32 fallback",
        "```",
        "",
        "## Summary",
        "",
    ]
    for row in gated.itertuples(index=False):
        lines.append(
            f"- density {row.target_density:.0%}, threshold {row.confidence_threshold:.2f}: "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"sampled defects delta {row.sampled_defects_delta:.2f}, "
            f"abs-error delta {row.absolute_error_delta:.4f}, "
            f"discovery selection rate {row.gated_discovery_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Higher threshold reduces aggressive discovery selections and bias, but it also lowers defect discovery gain.",
            "This gives an explicit operating knob between discovery-first and low-bias behavior.",
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

    strategy_map = build_strategy_map(pattern_summary, args.low_bias_delta)
    views = build_gated_views(eval_results, strategy_map, args.thresholds)
    summary, pattern = summarize(views)

    strategy_map.to_csv(args.out_dir / "confidence_gated_strategy_map.csv", index=False)
    views.to_csv(args.out_dir / "confidence_gated_policy_results.csv", index=False)
    summary.to_csv(args.out_dir / "confidence_gated_policy_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "confidence_gated_policy_pattern_summary.csv", index=False)
    plot_outputs(summary, args.fig_dir)
    write_report(summary, args.out_dir / "confidence_gated_policy_report.md")

    print(f"wrote confidence-gated policy outputs to {args.out_dir}")
    print(f"wrote confidence-gated policy figures to {args.fig_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
