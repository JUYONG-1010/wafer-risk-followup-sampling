from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_FOLLOWUP_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_TUNING_DIR = Path("data") / "processed" / "density_hybrid_scoring_tuning_v1"
DEFAULT_RISK_DIR = Path("data") / "processed" / "density_risk_maps_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "discovery_objective_view_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "44_discovery_objective_view_v1"

BASIC_POLICIES = [
    "coverage32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
    "morphrisk_guarded32",
]
POLICY_LABELS = {
    "coverage32": "coverage32",
    "hybrid_guarded1": "hybrid N=1",
    "hybrid_guarded2": "hybrid N=2",
    "hybrid_guarded4": "hybrid N=4",
    "morphrisk_guarded32": "risk-only top32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reframe policy results for defect-discovery objective."
    )
    parser.add_argument("--followup-dir", type=Path, default=DEFAULT_FOLLOWUP_DIR)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--risk-dir", type=Path, default=DEFAULT_RISK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--low-bias-delta", type=float, default=0.01)
    parser.add_argument("--moderate-bias-delta", type=float, default=0.03)
    return parser.parse_args()


def add_baseline_deltas(data: pd.DataFrame) -> pd.DataFrame:
    baseline = data[data["strategy"] == "coverage32"][
        [
            "target_density",
            "mean_absolute_error",
            "mean_defect_coverage",
            "mean_sampled_valid_count",
            "mean_sampling_density",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
            "mean_sampled_valid_count": "baseline_sampled_valid_count",
            "mean_sampling_density": "baseline_sampling_density",
        }
    )
    out = data.merge(baseline, on="target_density", how="left")
    out["absolute_error_delta"] = (
        out["mean_absolute_error"] - out["baseline_absolute_error"]
    )
    out["defect_coverage_delta"] = (
        out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    )
    out["defect_coverage_relative_improvement_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    return out


def bias_band(delta: float, low: float, moderate: float) -> str:
    if delta <= 0:
        return "improves_estimation"
    if delta <= low:
        return "low_bias_warning"
    if delta <= moderate:
        return "moderate_bias_warning"
    return "high_bias_warning"


def discovery_status(row: pd.Series) -> str:
    coverage_delta = float(row["defect_coverage_delta"])
    error_delta = float(row["absolute_error_delta"])
    if row["strategy"] == "coverage32":
        return "baseline"
    if coverage_delta <= 0 and error_delta >= 0:
        return "dominated_reject"
    if coverage_delta > 0 and error_delta <= 0:
        return "strong_improves_both"
    if coverage_delta > 0:
        return "discovery_gain_with_bias_warning"
    return "estimation_gain_no_discovery_gain"


def load_basic_policy_view(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.followup_dir / "density_followup_eval_summary.csv")
    data = data[data["strategy"].isin(BASIC_POLICIES)].copy()
    out = add_baseline_deltas(data)
    out["policy_family"] = "fixed_policy"
    out["policy_label"] = out["strategy"].map(POLICY_LABELS)
    return out


def load_tuned_policy_view(args: argparse.Namespace) -> pd.DataFrame:
    tuned = pd.read_csv(args.tuning_dir / "density_hybrid_scoring_tuning_summary.csv")
    tuned = tuned.rename(
        columns={
            "wafers": "wafers",
            "mean_sampled_valid_count": "mean_sampled_valid_count",
        }
    ).copy()
    tuned["mean_sampling_density"] = pd.NA
    tuned["mean_ratio_error"] = pd.NA
    tuned["underestimation_rate"] = pd.NA
    tuned["mean_spatial_cost_proxy"] = pd.NA
    tuned["mean_morph_confidence"] = pd.NA
    tuned["mean_morph_uncertainty"] = pd.NA
    tuned["mean_weak_pattern_risk"] = pd.NA
    tuned["mean_group_irregular_prob"] = pd.NA
    out = tuned.copy()
    out["policy_family"] = "tuned_hybrid_grid"
    out["policy_label"] = out["strategy"]
    return out


def attach_risk_metrics(data: pd.DataFrame, risk_dir: Path) -> pd.DataFrame:
    risk = pd.read_csv(risk_dir / "density_risk_map_summary.csv")
    risk = risk[risk["risk_map"] == "risk_guarded"][
        [
            "target_density",
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top10pct_iou",
            "mean_top32_defect_coverage",
        ]
    ]
    return data.merge(risk, on="target_density", how="left")


def build_views(args: argparse.Namespace) -> pd.DataFrame:
    basic = load_basic_policy_view(args)
    tuned = load_tuned_policy_view(args)

    common_cols = sorted(set(basic.columns) | set(tuned.columns))
    data = pd.concat(
        [basic.reindex(columns=common_cols), tuned.reindex(columns=common_cols)],
        ignore_index=True,
    )
    data["bias_band"] = data["absolute_error_delta"].map(
        lambda value: bias_band(float(value), args.low_bias_delta, args.moderate_bias_delta)
    )
    data["discovery_status"] = data.apply(discovery_status, axis=1)
    data["sampled_defect_ratio_lift_vs_actual"] = (
        data["mean_defect_coverage"] / data["mean_sampling_density"]
        if "mean_sampling_density" in data.columns
        else pd.NA
    )
    data = attach_risk_metrics(data, args.risk_dir)
    return data


def select_recommendations(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = data[
        (data["strategy"] != "coverage32")
        & (data["defect_coverage_delta"] > 0)
    ].copy()
    discovery_rows: list[pd.Series] = []
    low_bias_rows: list[pd.Series] = []
    for density, group in candidates.groupby("target_density", observed=False):
        discovery_rows.append(
            group.sort_values(
                ["defect_coverage_delta", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0]
        )
        low_bias = group[group["absolute_error_delta"] <= args.low_bias_delta]
        if low_bias.empty:
            low_bias_rows.append(
                group.sort_values(
                    ["absolute_error_delta", "defect_coverage_delta"],
                    ascending=[True, False],
                ).iloc[0]
            )
        else:
            low_bias_rows.append(
                low_bias.sort_values(
                    ["defect_coverage_delta", "absolute_error_delta"],
                    ascending=[False, True],
                ).iloc[0]
            )
    discovery = pd.DataFrame(discovery_rows).reset_index(drop=True)
    discovery["recommendation_view"] = "discovery_first"
    low_bias = pd.DataFrame(low_bias_rows).reset_index(drop=True)
    low_bias["recommendation_view"] = "low_bias_discovery"
    return discovery, low_bias


def plot_views(data: pd.DataFrame, discovery: pd.DataFrame, low_bias: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_data = data[data["strategy"] != "coverage32"].copy()
    plot_data["density_pct"] = plot_data["target_density"] * 100.0

    plt.figure(figsize=(9.2, 5.6))
    sns.scatterplot(
        data=plot_data,
        x="absolute_error_delta",
        y="defect_coverage_relative_improvement_pct",
        hue="density_pct",
        style="policy_family",
        alpha=0.72,
        palette="viridis",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.axvline(0.03, color="#aaaaaa", linestyle=":", linewidth=0.9)
    plt.xlabel("Absolute-error delta vs coverage32 (bias warning)")
    plt.ylabel("Defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "discovery_gain_vs_bias_warning.png", dpi=180)
    plt.close()

    top = pd.concat([discovery, low_bias], ignore_index=True)
    top["density_view"] = (
        (top["target_density"] * 100.0).map(lambda value: f"{value:g}%")
        + " "
        + top["recommendation_view"]
    )
    plt.figure(figsize=(10.8, 5.4))
    sns.barplot(
        data=top,
        x="density_view",
        y="defect_coverage_relative_improvement_pct",
        hue="bias_band",
    )
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("")
    plt.ylabel("Selected policy coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "selected_discovery_recommendations.png", dpi=180)
    plt.close()


def write_report(
    data: pd.DataFrame,
    discovery: pd.DataFrame,
    low_bias: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = [
        "# Discovery Objective View v1",
        "",
        "This report reframes the project around defect-rich-region discovery rather than unbiased wafer defect-ratio estimation.",
        "",
        "## Metric Roles",
        "",
        "Primary discovery metrics:",
        "",
        "```text",
        "defect coverage gain vs coverage32",
        "risk-map ROC-AUC / AP",
        "top-k or top-fraction localization metrics",
        "```",
        "",
        "Secondary warning metric:",
        "",
        "```text",
        "absolute-error delta vs coverage32",
        "```",
        "",
        "Absolute-error delta is no longer treated as a hard failure for discovery-first use. It is a warning that the recommended points are biased toward defect-rich regions and should not be used alone as an unbiased defect-ratio estimator.",
        "",
        "## Discovery-First Best Policies",
        "",
    ]
    for row in discovery.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: {row.strategy}, "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f} ({row.bias_band}), "
            f"AP {row.mean_average_precision:.3f}, Top10IoU {row.mean_top10pct_iou:.3f}"
        )
    lines.extend(["", "## Low-Bias Discovery Policies", ""])
    for row in low_bias.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: {row.strategy}, "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f} ({row.bias_band})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the final product is follow-up defect discovery, then a higher sampled defect ratio is expected and can be desirable.",
            "If the final product also needs wafer-level defect-rate estimation, use a separate representative subset for estimation and do not use ML-risk follow-up points alone as an unbiased estimator.",
            "",
            "## Decision Rule",
            "",
            "```text",
            "Discovery-first mode:",
            "  choose policies by defect coverage gain and report bias warning separately",
            "",
            "Estimation-sensitive mode:",
            "  require low absolute-error delta or use coverage32",
            "```",
            "",
        ]
    )
    (args.out_dir / "discovery_objective_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    data = build_views(args)
    discovery, low_bias = select_recommendations(data, args)

    data.to_csv(args.out_dir / "discovery_objective_policy_view.csv", index=False)
    discovery.to_csv(args.out_dir / "discovery_first_recommendations.csv", index=False)
    low_bias.to_csv(args.out_dir / "low_bias_discovery_recommendations.csv", index=False)
    plot_views(data, discovery, low_bias, args.fig_dir)
    write_report(data, discovery, low_bias, args)

    print(f"wrote discovery objective outputs to {args.out_dir}")
    print(f"wrote discovery objective figures to {args.fig_dir}")
    print("Discovery-first recommendations:")
    print(
        discovery[
            [
                "target_density",
                "strategy",
                "defect_coverage_relative_improvement_pct",
                "absolute_error_delta",
                "bias_band",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("Low-bias discovery recommendations:")
    print(
        low_bias[
            [
                "target_density",
                "strategy",
                "defect_coverage_relative_improvement_pct",
                "absolute_error_delta",
                "bias_band",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
