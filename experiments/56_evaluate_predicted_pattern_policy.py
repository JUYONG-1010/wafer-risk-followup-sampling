from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "predicted_pattern_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "46_predicted_pattern_policy_v1"

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
        description="Evaluate predicted-pattern operating policy selection."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--low-bias-delta", type=float, default=0.01)
    return parser.parse_args()


def add_baseline_deltas(data: pd.DataFrame) -> pd.DataFrame:
    baseline = data[data["strategy"] == "coverage32"][
        ["target_density", "failureType", "strategy", "mean_absolute_error", "mean_defect_coverage"]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
        }
    )
    baseline = baseline.drop(columns=["strategy"])
    out = data.merge(baseline, on=["target_density", "failureType"], how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["defect_coverage_delta"] = (
        out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    )
    out["defect_coverage_relative_improvement_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    return out


def build_strategy_map(pattern_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = pattern_summary[pattern_summary["strategy"].isin(AVAILABLE_STRATEGIES)].copy()
    data = add_baseline_deltas(data)
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
            low_bias_pool = positive[positive["absolute_error_delta"] <= args.low_bias_delta]
            if low_bias_pool.empty:
                low_bias_strategy = "coverage32"
            else:
                low_bias_strategy = low_bias_pool.sort_values(
                    ["defect_coverage_relative_improvement_pct", "absolute_error_delta"],
                    ascending=[False, True],
                ).iloc[0]["strategy"]

        if pattern in DISCOVERY_PATTERNS:
            pattern_aware_strategy = discovery_strategy
            pattern_mode = "discovery_first"
        elif pattern in LOW_BIAS_PATTERNS:
            pattern_aware_strategy = low_bias_strategy
            pattern_mode = "low_bias_default"
        elif pattern in COVERAGE_PATTERNS:
            pattern_aware_strategy = "coverage32"
            pattern_mode = "coverage32_or_uncertain"
        else:
            pattern_aware_strategy = low_bias_strategy
            pattern_mode = "unknown_low_bias_default"

        rows.append(
            {
                "target_density": density,
                "pattern": pattern,
                "pattern_mode": pattern_mode,
                "discovery_strategy": discovery_strategy,
                "low_bias_strategy": low_bias_strategy,
                "pattern_aware_strategy": pattern_aware_strategy,
            }
        )
    return pd.DataFrame(rows)


def select_row_metrics(
    eval_results: pd.DataFrame,
    selector: pd.DataFrame,
    view_name: str,
    pattern_column: str,
    strategy_column: str,
) -> pd.DataFrame:
    chosen = eval_results.merge(
        selector[["target_density", "pattern", strategy_column]],
        left_on=["target_density", pattern_column],
        right_on=["target_density", "pattern"],
        how="left",
    )
    chosen["selected_strategy"] = chosen[strategy_column].fillna("coverage32")
    chosen = chosen[chosen["strategy"] == chosen["selected_strategy"]].copy()
    chosen["policy_view"] = view_name
    return chosen.drop(columns=["pattern", strategy_column])


def build_policy_views(eval_results: pd.DataFrame, strategy_map: pd.DataFrame) -> pd.DataFrame:
    base = eval_results[eval_results["strategy"] == "coverage32"].copy()
    base["selected_strategy"] = "coverage32"
    base["policy_view"] = "coverage32"

    oracle = select_row_metrics(
        eval_results,
        strategy_map,
        view_name="oracle_true_pattern_mode",
        pattern_column="failureType",
        strategy_column="pattern_aware_strategy",
    )
    predicted = select_row_metrics(
        eval_results,
        strategy_map,
        view_name="predicted_pattern_mode",
        pattern_column="morph_top1",
        strategy_column="pattern_aware_strategy",
    )
    predicted_low_bias = select_row_metrics(
        eval_results,
        strategy_map,
        view_name="predicted_low_bias_mode",
        pattern_column="morph_top1",
        strategy_column="low_bias_strategy",
    )
    predicted_discovery = select_row_metrics(
        eval_results,
        strategy_map,
        view_name="predicted_discovery_only_mode",
        pattern_column="morph_top1",
        strategy_column="discovery_strategy",
    )
    return pd.concat([base, oracle, predicted, predicted_low_bias, predicted_discovery], ignore_index=True)


def summarize(views: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        views.groupby(["target_density", "policy_view"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_morph_confidence=("morph_confidence", "mean"),
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
        views.groupby(["target_density", "failureType", "policy_view"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
        )
        .reset_index()
    )
    return summary, pattern


def morphology_accuracy(eval_results: pd.DataFrame) -> pd.DataFrame:
    base = eval_results[eval_results["strategy"] == "coverage32"].copy()
    base["morph_top1_correct"] = base["morph_top1"] == base["failureType"]
    return (
        base.groupby("target_density", observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            morph_top1_accuracy=("morph_top1_correct", "mean"),
            mean_morph_confidence=("morph_confidence", "mean"),
        )
        .reset_index()
    )


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    for metric, filename, ylabel in [
        ("defect_coverage_relative_improvement_pct", "predicted_policy_coverage_gain.png", "Coverage gain vs coverage32 (%)"),
        ("absolute_error_delta", "predicted_policy_abs_error_delta.png", "Absolute-error delta vs coverage32"),
        ("sampled_defects_delta", "predicted_policy_sampled_defects_delta.png", "Mean sampled defects delta vs coverage32"),
    ]:
        plt.figure(figsize=(9.2, 5.2))
        sns.lineplot(data=data, x="density_pct", y=metric, hue="policy_view", marker="o")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Initial probe density (%)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()


def write_report(summary: pd.DataFrame, accuracy: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Predicted Pattern Policy Evaluation v1",
        "",
        "Purpose: evaluate the deployable policy layer:",
        "",
        "```text",
        "first-pass sparse data -> predicted morphology -> operating mode -> follow-up strategy",
        "```",
        "",
        "Compared views:",
        "",
        "```text",
        "coverage32: non-ML spatial baseline",
        "oracle_true_pattern_mode: upper-bound mode using true pattern labels",
        "predicted_pattern_mode: deployable pattern-aware mode using morph_top1",
        "predicted_low_bias_mode: conservative deployable mode",
        "predicted_discovery_only_mode: aggressive deployable discovery mode",
        "```",
        "",
        "## Morphology Prediction Accuracy",
        "",
    ]
    for row in accuracy.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: top1 accuracy {row.morph_top1_accuracy:.3f}, "
            f"mean confidence {row.mean_morph_confidence:.3f}"
        )
    lines.extend(["", "## Policy Summary", ""])
    for row in summary.itertuples(index=False):
        if row.policy_view == "coverage32":
            continue
        lines.append(
            f"- {row.target_density:.0%}, {row.policy_view}: "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"sampled defects delta {row.sampled_defects_delta:.2f}, "
            f"abs-error delta {row.absolute_error_delta:.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The gap between oracle_true_pattern_mode and predicted_pattern_mode is the cost of relying on first-pass morphology prediction.",
            "If the predicted mode remains close to oracle, the pattern-aware policy layer is usable. If not, confidence gating or fallback to coverage32 is needed.",
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

    strategy_map = build_strategy_map(pattern_summary, args)
    views = build_policy_views(eval_results, strategy_map)
    summary, pattern = summarize(views)
    accuracy = morphology_accuracy(eval_results)

    strategy_map.to_csv(args.out_dir / "predicted_pattern_strategy_map.csv", index=False)
    views.to_csv(args.out_dir / "predicted_pattern_policy_results.csv", index=False)
    summary.to_csv(args.out_dir / "predicted_pattern_policy_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "predicted_pattern_policy_pattern_summary.csv", index=False)
    accuracy.to_csv(args.out_dir / "morphology_prediction_accuracy.csv", index=False)
    plot_summary(summary, args.fig_dir)
    write_report(summary, accuracy, args.out_dir / "predicted_pattern_policy_report.md")

    print(f"wrote predicted pattern policy outputs to {args.out_dir}")
    print(f"wrote predicted pattern policy figures to {args.fig_dir}")
    print(summary.round(4).to_string(index=False))
    print(accuracy.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
