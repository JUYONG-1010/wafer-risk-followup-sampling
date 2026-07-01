from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OBJECTIVE_DIR = Path("data") / "processed" / "pure_ml_discovery_objective_v1"
DEFAULT_PRED_DIR = Path("data") / "processed" / "augmented_morphology_policy_aligned_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "objective_mode_operating_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "53_objective_mode_operating_policy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a final-style objective-mode operating policy: "
            "first-pass morphology confidence selects discovery-first or low-bias follow-up."
        )
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--objective-dir", type=Path, default=DEFAULT_OBJECTIVE_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.7, 0.8])
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_results = pd.read_csv(args.eval_dir / "density_followup_eval_results.csv")
    min_cost = eval_results["cost_weight"].min()
    eval_results = eval_results[eval_results["cost_weight"] == min_cost].copy()

    objective_map = pd.read_csv(args.objective_dir / "pattern_density_objective_view.csv")
    predictions = pd.read_csv(args.pred_dir / "augmented_morphology_predictions.csv")
    return eval_results, objective_map, predictions


def build_lookup(objective_map: pd.DataFrame) -> dict[tuple[float, str], dict[str, str]]:
    lookup: dict[tuple[float, str], dict[str, str]] = {}
    for row in objective_map.itertuples(index=False):
        low_bias_strategy = str(row.low_bias_strategy)
        if low_bias_strategy == "nan" or not low_bias_strategy:
            low_bias_strategy = "coverage32"
        lookup[(float(row.target_density), str(row.failureType))] = {
            "recommended_objective": str(row.recommended_objective),
            "best_discovery_strategy": str(row.best_discovery_strategy),
            "low_bias_strategy": low_bias_strategy,
        }
    return lookup


def choose_strategy(
    pattern: str,
    confidence: float,
    threshold: float,
    density: float,
    lookup: dict[tuple[float, str], dict[str, str]],
    oracle: bool = False,
) -> tuple[str, str, str]:
    entry = lookup.get((float(density), str(pattern)))
    if entry is None:
        return "coverage32", "fallback_unknown_pattern", "coverage_or_unknown"

    objective = entry["recommended_objective"]
    if objective == "low_bias_or_coverage":
        return entry["low_bias_strategy"], "low_bias_or_coverage", objective

    if objective == "discovery_first_allow_ml_bias":
        if oracle or float(confidence) >= float(threshold):
            return entry["best_discovery_strategy"], "discovery_first", objective
        return entry["low_bias_strategy"], "low_confidence_low_bias_fallback", objective

    return entry["low_bias_strategy"], "fallback_low_bias", objective


def build_oracle_view(
    eval_results: pd.DataFrame,
    thresholds: list[float],
    lookup: dict[tuple[float, str], dict[str, str]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selector_base = eval_results[["row_index", "target_density", "failureType"]].drop_duplicates()
    for threshold in thresholds:
        selector = selector_base.copy()
        choices = selector.apply(
            lambda row: choose_strategy(
                row["failureType"],
                1.0,
                threshold,
                row["target_density"],
                lookup,
                oracle=True,
            ),
            axis=1,
            result_type="expand",
        )
        selector["selected_strategy"] = choices[0]
        selector["selection_reason"] = choices[1]
        selector["selected_objective"] = choices[2]
        selected = eval_results.merge(
            selector,
            on=["row_index", "target_density", "failureType"],
            how="inner",
        )
        selected = selected[selected["strategy"] == selected["selected_strategy"]].copy()
        selected["policy_view"] = "oracle_true_pattern_objective"
        selected["prediction_variant"] = "oracle"
        selected["confidence_threshold"] = float(threshold)
        selected["predicted_pattern"] = selected["failureType"]
        selected["predicted_confidence"] = 1.0
        selected["top1_correct"] = True
        selected["mode_correct"] = True
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def build_predicted_view(
    eval_results: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: list[float],
    lookup: dict[tuple[float, str], dict[str, str]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pred = predictions.rename(
        columns={
            "pred_label": "predicted_pattern",
            "top1_confidence": "predicted_confidence",
        }
    )
    pred = pred[
        [
            "row_index",
            "target_density",
            "variant",
            "predicted_pattern",
            "predicted_confidence",
            "top1_correct",
            "mode_correct",
        ]
    ].copy()

    for threshold in thresholds:
        selector = pred.copy()
        choices = selector.apply(
            lambda row: choose_strategy(
                row["predicted_pattern"],
                row["predicted_confidence"],
                threshold,
                row["target_density"],
                lookup,
                oracle=False,
            ),
            axis=1,
            result_type="expand",
        )
        selector["selected_strategy"] = choices[0]
        selector["selection_reason"] = choices[1]
        selector["selected_objective"] = choices[2]
        selected = eval_results.merge(
            selector,
            on=["row_index", "target_density"],
            how="inner",
        )
        selected = selected[selected["strategy"] == selected["selected_strategy"]].copy()
        selected["policy_view"] = "predicted_objective_mode"
        selected["prediction_variant"] = selected["variant"]
        selected["confidence_threshold"] = float(threshold)
        rows.append(selected.drop(columns=["variant"]))
    return pd.concat(rows, ignore_index=True)


def build_coverage_view(eval_results: pd.DataFrame) -> pd.DataFrame:
    base = eval_results[eval_results["strategy"] == "coverage32"].copy()
    base["policy_view"] = "coverage32"
    base["prediction_variant"] = "coverage32"
    base["confidence_threshold"] = -1.0
    base["selected_strategy"] = "coverage32"
    base["selection_reason"] = "baseline"
    base["selected_objective"] = "representative_baseline"
    base["predicted_pattern"] = base["failureType"]
    base["predicted_confidence"] = 1.0
    base["top1_correct"] = True
    base["mode_correct"] = True
    return base


def build_policy_views(
    eval_results: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: list[float],
    lookup: dict[tuple[float, str], dict[str, str]],
) -> pd.DataFrame:
    return pd.concat(
        [
            build_coverage_view(eval_results),
            build_oracle_view(eval_results, thresholds, lookup),
            build_predicted_view(eval_results, predictions, thresholds, lookup),
        ],
        ignore_index=True,
    )


def summarize(views: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        views.groupby(
            ["policy_view", "prediction_variant", "confidence_threshold", "target_density"],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            discovery_first_rate=("selection_reason", lambda values: (values == "discovery_first").mean()),
            low_bias_fallback_rate=(
                "selection_reason",
                lambda values: (values == "low_confidence_low_bias_fallback").mean(),
            ),
            low_bias_or_coverage_rate=(
                "selection_reason",
                lambda values: (values == "low_bias_or_coverage").mean(),
            ),
            mean_prediction_confidence=("predicted_confidence", "mean"),
            top1_accuracy=("top1_correct", "mean"),
            mode_accuracy=("mode_correct", "mean"),
        )
        .reset_index()
    )
    baseline = summary[summary["policy_view"] == "coverage32"][
        [
            "target_density",
            "mean_absolute_error",
            "mean_ratio_error",
            "mean_defect_coverage",
            "mean_sampled_defects",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_ratio_error": "baseline_ratio_error",
            "mean_defect_coverage": "baseline_defect_coverage",
            "mean_sampled_defects": "baseline_sampled_defects",
        }
    )
    summary = summary.merge(baseline, on="target_density", how="left")
    summary["absolute_error_delta"] = summary["mean_absolute_error"] - summary["baseline_absolute_error"]
    summary["ratio_error_delta"] = summary["mean_ratio_error"] - summary["baseline_ratio_error"]
    summary["defect_coverage_gain_pct"] = (
        (summary["mean_defect_coverage"] - summary["baseline_defect_coverage"])
        / summary["baseline_defect_coverage"]
        * 100.0
    )
    summary["sampled_defects_delta"] = (
        summary["mean_sampled_defects"] - summary["baseline_sampled_defects"]
    )

    pattern = (
        views.groupby(
            [
                "policy_view",
                "prediction_variant",
                "confidence_threshold",
                "target_density",
                "failureType",
            ],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            discovery_first_rate=("selection_reason", lambda values: (values == "discovery_first").mean()),
        )
        .reset_index()
    )
    return summary, pattern


def plot_outputs(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary[
        (summary["policy_view"] != "coverage32")
        & (summary["prediction_variant"].isin(["oracle", "baseline", "augmented"]))
    ].copy()
    data["density_pct"] = data["target_density"] * 100.0

    for metric, filename, ylabel in [
        ("defect_coverage_gain_pct", "objective_policy_coverage_gain.png", "Coverage gain vs coverage32 (%)"),
        ("absolute_error_delta", "objective_policy_abs_error_delta.png", "Absolute-error delta vs coverage32"),
        ("sampled_defects_delta", "objective_policy_sampled_defects_delta.png", "Sampled defects delta vs coverage32"),
    ]:
        plt.figure(figsize=(9.4, 5.4))
        sns.lineplot(
            data=data,
            x="confidence_threshold",
            y=metric,
            hue="density_pct",
            style="prediction_variant",
            marker="o",
            palette="viridis",
        )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Morphology confidence threshold")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    focus = data[data["confidence_threshold"] == 0.60].copy()
    plt.figure(figsize=(8.8, 5.0))
    sns.scatterplot(
        data=focus,
        x="absolute_error_delta",
        y="defect_coverage_gain_pct",
        hue="density_pct",
        style="prediction_variant",
        s=90,
        palette="viridis",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.xlabel("Absolute-error delta vs coverage32")
    plt.ylabel("Coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_0p60_gain_vs_bias_warning.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        "# Objective-Mode Operating Policy v1",
        "",
        "Policy:",
        "",
        "```text",
        "first-pass morphology prediction + confidence",
        "-> choose discovery-first or low-bias/coverage follow-up objective",
        "```",
        "",
        "Discovery-first patterns can use aggressive ML-risk policies such as pure ML top32.",
        "Random/Near-full or low-confidence predictions fall back to low-bias/coverage strategies.",
        "",
        "## Threshold 0.60 Summary",
        "",
    ]
    focus = summary[
        (summary["confidence_threshold"] == 0.60)
        & (summary["prediction_variant"].isin(["oracle", "baseline", "augmented"]))
    ].copy()
    for row in focus.itertuples(index=False):
        lines.append(
            f"- {row.prediction_variant}, {row.target_density:.0%}: "
            f"coverage gain {row.defect_coverage_gain_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f}, "
            f"sampled-defects delta {row.sampled_defects_delta:.2f}, "
            f"discovery-first rate {row.discovery_first_rate:.3f}, "
            f"mode accuracy {row.mode_accuracy:.3f}"
        )
    lines.extend(["", "## All Thresholds", ""])
    for row in summary[
        (summary["policy_view"] != "coverage32")
        & (summary["prediction_variant"].isin(["oracle", "baseline", "augmented"]))
    ].itertuples(index=False):
        lines.append(
            f"- {row.prediction_variant}, {row.target_density:.0%}, threshold {row.confidence_threshold:.2f}: "
            f"gain {row.defect_coverage_gain_pct:.2f}%, "
            f"abs-delta {row.absolute_error_delta:.4f}, "
            f"discovery-rate {row.discovery_first_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This policy explicitly separates the follow-up objective:",
            "",
            "```text",
            "discovery-first: allow ML concentration and report representativeness warning",
            "estimation-sensitive: use low-bias/coverage strategy",
            "```",
            "",
            "The threshold controls how confidently the model must identify a discovery-first pattern before allowing aggressive ML follow-up.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    eval_results, objective_map, predictions = load_inputs(args)
    lookup = build_lookup(objective_map)
    views = build_policy_views(eval_results, predictions, args.thresholds, lookup)
    summary, pattern = summarize(views)

    views.to_csv(args.out_dir / "objective_mode_policy_results.csv", index=False)
    summary.to_csv(args.out_dir / "objective_mode_policy_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "objective_mode_policy_pattern_summary.csv", index=False)
    plot_outputs(summary, args.fig_dir)
    write_report(summary, args.out_dir / "objective_mode_operating_policy_report.md")

    print(f"wrote objective-mode policy outputs to {args.out_dir}")
    print(f"wrote objective-mode policy figures to {args.fig_dir}")
    print(
        summary[
            (summary["confidence_threshold"] == 0.60)
            & (summary["prediction_variant"].isin(["oracle", "baseline", "augmented"]))
        ][
            [
                "prediction_variant",
                "target_density",
                "defect_coverage_gain_pct",
                "absolute_error_delta",
                "sampled_defects_delta",
                "discovery_first_rate",
                "mode_accuracy",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
