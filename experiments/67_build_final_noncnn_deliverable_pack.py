from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OBJECTIVE_DIR = Path("data") / "processed" / "objective_mode_operating_policy_v1"
ADEQUACY_DIR = Path("data") / "processed" / "model_adequacy_baseline_audit_v1"
DISCOVERY_DIR = Path("data") / "processed" / "pure_ml_discovery_objective_v1"
RISK_MAP_DIR = Path("data") / "processed" / "density_risk_maps_v1"
STABILITY_DIR = Path("data") / "processed" / "repeated_split_stability_v1"
EXAMPLE_REPORT_DIR = Path("reports") / "density_risk_map_examples_v1"
EXAMPLE_FIG_DIR = Path("outputs") / "figures" / "41_density_risk_map_examples_v1"

OUT_DIR = Path("data") / "processed" / "final_noncnn_deliverable_pack_v1"
REPORT_DIR = Path("reports") / "final_noncnn_deliverable_pack_v1"
FIG_DIR = Path("outputs") / "figures" / "57_final_noncnn_deliverable_pack_v1"

FOCUS_POLICIES = [
    "coverage32",
    "matched_random32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
    "morphrisk_guarded32",
    "objective_augmented_t0.60",
    "ml_rank32",
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, REPORT_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_inputs() -> dict[str, pd.DataFrame]:
    return {
        "policy_summary": pd.read_csv(ADEQUACY_DIR / "model_adequacy_policy_summary.csv"),
        "bootstrap_ci": pd.read_csv(ADEQUACY_DIR / "model_adequacy_bootstrap_ci.csv"),
        "risk_map": pd.read_csv(RISK_MAP_DIR / "density_risk_map_summary.csv"),
        "stability": pd.read_csv(STABILITY_DIR / "repeated_split_stability_summary.csv"),
        "objective": pd.read_csv(OBJECTIVE_DIR / "objective_mode_policy_summary.csv"),
        "discovery": pd.read_csv(DISCOVERY_DIR / "global_policy_objective_view.csv"),
    }


def build_final_policy_table(policy_summary: pd.DataFrame, bootstrap_ci: pd.DataFrame) -> pd.DataFrame:
    focus = policy_summary[policy_summary["policy_name"].isin(FOCUS_POLICIES)].copy()
    focus = focus[
        [
            "target_density",
            "policy_name",
            "wafers",
            "mean_defect_coverage",
            "defect_coverage_gain_pct",
            "mean_sampled_defects",
            "sampled_defects_delta",
            "mean_absolute_error",
            "absolute_error_delta",
            "severe_miss_rate",
            "severe_miss_delta",
        ]
    ]
    ci = bootstrap_ci[
        [
            "target_density",
            "policy_name",
            "coverage_gain_ci_low",
            "coverage_gain_ci_high",
            "coverage_gain_prob_positive",
            "absolute_error_delta_ci_low",
            "absolute_error_delta_ci_high",
        ]
    ].copy()
    table = focus.merge(ci, on=["target_density", "policy_name"], how="left")
    table["density_label"] = table["target_density"].map(lambda value: f"{value:.0%}")
    table = table.sort_values(["target_density", "policy_name"]).reset_index(drop=True)
    return table


def build_claim_table(final_policy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in ["matched_random32", "hybrid_guarded1", "objective_augmented_t0.60", "ml_rank32"]:
        subset = final_policy[final_policy["policy_name"] == policy].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "claim_area": policy,
                "density_range": "1%, 3%, 5%, 10%",
                "coverage_gain_min_pct": subset["defect_coverage_gain_pct"].min(),
                "coverage_gain_max_pct": subset["defect_coverage_gain_pct"].max(),
                "abs_error_delta_min": subset["absolute_error_delta"].min(),
                "abs_error_delta_max": subset["absolute_error_delta"].max(),
                "interpretation": interpretation_for_policy(policy),
            }
        )
    return pd.DataFrame(rows)


def interpretation_for_policy(policy: str) -> str:
    return {
        "matched_random32": "negative/near-zero control; confirms coverage32 is a nontrivial geometry baseline",
        "hybrid_guarded1": "limited one-point ML replacement; small positive discovery gain with limited extra bias",
        "objective_augmented_t0.60": "objective-mode policy; strong discovery gain but representativeness warning remains",
        "ml_rank32": "pure discovery-first model; highest defect discovery and highest sampled-ratio bias",
    }[policy]


def build_risk_map_table(risk_map: pd.DataFrame) -> pd.DataFrame:
    focus = risk_map[risk_map["risk_map"].isin(["risk_point", "risk_morphrisk", "risk_guarded"])].copy()
    focus["density_label"] = focus["target_density"].map(lambda value: f"{value:.0%}")
    keep = [
        "density_label",
        "target_density",
        "risk_map",
        "wafers",
        "mean_roc_auc",
        "mean_average_precision",
        "mean_top10pct_iou",
        "mean_top32_defect_coverage",
        "mean_candidate_defects",
    ]
    return focus[keep].sort_values(["target_density", "risk_map"]).reset_index(drop=True)


def build_stability_table(stability: pd.DataFrame) -> pd.DataFrame:
    focus = stability[stability["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1"])].copy()
    focus["density_label"] = focus["target_density"].map(lambda value: f"{value:.0%}")
    keep = [
        "density_label",
        "target_density",
        "strategy",
        "seeds",
        "gain_mean",
        "gain_std",
        "gain_min",
        "gain_max",
        "gain_positive_rate",
        "abs_delta_mean",
    ]
    return focus[keep].sort_values(["target_density", "strategy"]).reset_index(drop=True)


def build_leakage_audit_table() -> pd.DataFrame:
    rows = [
        {
            "check": "Allowed input: first-pass sparse observations",
            "status": "pass",
            "reason": "Policies are trained/evaluated from sparse first-pass masks and candidate geometry.",
        },
        {
            "check": "Allowed input: wafer geometry and candidate coordinates",
            "status": "pass",
            "reason": "Point-ranking features use candidate position, radius/angle, coverage distance, and first-pass-derived features.",
        },
        {
            "check": "Allowed input: model scores from allowed features",
            "status": "pass",
            "reason": "Risk scores are derived from fitted models using inference-safe feature columns.",
        },
        {
            "check": "Forbidden input: dense hidden defect labels at inference time",
            "status": "pass-by-design",
            "reason": "Dense maps are used after selection for offline evaluation metrics and visualization overlays.",
        },
        {
            "check": "Forbidden input: true failureType at inference time",
            "status": "needs-care",
            "reason": "Oracle/pattern analyses are diagnostic only; deployable policy must use predicted morphology and confidence gating.",
        },
        {
            "check": "Forbidden input: actual defect ratio / total defect count",
            "status": "pass-by-design",
            "reason": "These fields are metrics, not candidate-ranking inputs.",
        },
        {
            "check": "External fab generalization",
            "status": "not-proven",
            "reason": "WM-811K is a proxy dataset; no process recipe, tool, lot-history, or external fab validation is available.",
        },
    ]
    return pd.DataFrame(rows)


def plot_gain_vs_bias(final_policy: pd.DataFrame) -> None:
    plot_df = final_policy[
        final_policy["policy_name"].isin(
            ["matched_random32", "hybrid_guarded1", "morphrisk_guarded32", "objective_augmented_t0.60", "ml_rank32"]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for policy, group in plot_df.groupby("policy_name"):
        ax.plot(
            group["absolute_error_delta"],
            group["defect_coverage_gain_pct"],
            marker="o",
            linewidth=1.8,
            label=policy,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"{row.target_density:.0%}",
                (row.absolute_error_delta, row.defect_coverage_gain_pct),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Absolute error delta vs coverage32")
    ax.set_ylabel("Defect coverage gain vs coverage32 (%)")
    ax.set_title("Non-CNN follow-up policy tradeoff")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "final_noncnn_gain_vs_bias.png", dpi=180)
    plt.close(fig)


def plot_risk_map_metrics(risk_map_table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    metrics = [
        ("mean_roc_auc", "ROC-AUC"),
        ("mean_average_precision", "Average Precision"),
        ("mean_top10pct_iou", "Top-10% IoU"),
    ]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for risk_map, group in risk_map_table.groupby("risk_map"):
            ax.plot(group["target_density"], group[metric], marker="o", linewidth=1.8, label=risk_map)
        ax.set_title(title)
        ax.set_xlabel("Initial probe density")
        ax.set_xticks(sorted(risk_map_table["target_density"].unique()))
        ax.set_xticklabels([f"{value:.0%}" for value in sorted(risk_map_table["target_density"].unique())])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(fontsize=8)
    fig.savefig(FIG_DIR / "final_risk_map_metrics.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], precision: int = 3) -> str:
    view = df[columns].copy()
    for col in view.select_dtypes(include="number").columns:
        if col == "wafers" or col == "seeds":
            view[col] = view[col].astype(int)
        else:
            view[col] = view[col].map(lambda value: f"{value:.{precision}f}" if pd.notna(value) else "")
    return frame_to_markdown(view)


def frame_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *[fmt_row(row) for row in rows]])


def existing_example_figures() -> list[Path]:
    if not EXAMPLE_FIG_DIR.exists():
        return []
    return sorted(EXAMPLE_FIG_DIR.glob("*.png"))


def write_report(
    final_policy: pd.DataFrame,
    claim_table: pd.DataFrame,
    risk_map_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    leakage_table: pd.DataFrame,
) -> None:
    example_figs = existing_example_figures()
    lines = [
        "# Final Non-CNN Deliverable Pack",
        "",
        "This pack freezes the current non-CNN baseline before any CNN branch is attempted.",
        "",
        "Frozen goal:",
        "",
        "> Given first-pass sparse wafer observations, generate a follow-up inspection recommendation and optional 2D defect-risk map, then evaluate discovery gain and representativeness tradeoff using WM-811K dense maps as offline reference.",
        "",
        "## Main Claim Boundaries",
        "",
        markdown_table(
            claim_table,
            [
                "claim_area",
                "density_range",
                "coverage_gain_min_pct",
                "coverage_gain_max_pct",
                "abs_error_delta_min",
                "abs_error_delta_max",
                "interpretation",
            ],
        ),
        "",
        "## Follow-Up Policy Metrics",
        "",
        markdown_table(
            final_policy,
            [
                "density_label",
                "policy_name",
                "mean_defect_coverage",
                "defect_coverage_gain_pct",
                "coverage_gain_ci_low",
                "coverage_gain_ci_high",
                "absolute_error_delta",
                "mean_sampled_defects",
                "severe_miss_rate",
            ],
        ),
        "",
        "## 2D Risk-Map Metrics",
        "",
        "These metrics score the predicted risk ranking over unmeasured candidate dies. They are not full dense-map reconstruction accuracy.",
        "",
        markdown_table(
            risk_map_table,
            [
                "density_label",
                "risk_map",
                "mean_roc_auc",
                "mean_average_precision",
                "mean_top10pct_iou",
                "mean_top32_defect_coverage",
            ],
        ),
        "",
        "## Split-Stability Snapshot",
        "",
        markdown_table(
            stability_table,
            [
                "density_label",
                "strategy",
                "seeds",
                "gain_mean",
                "gain_std",
                "gain_min",
                "gain_positive_rate",
                "abs_delta_mean",
            ],
        ),
        "",
        "## Leakage Audit",
        "",
        frame_to_markdown(leakage_table),
        "",
        "## Figures",
        "",
        f"- Tradeoff figure: `{FIG_DIR / 'final_noncnn_gain_vs_bias.png'}`",
        f"- Risk-map metrics figure: `{FIG_DIR / 'final_risk_map_metrics.png'}`",
    ]
    if example_figs:
        lines.append("- Existing representative wafer examples:")
        lines.extend([f"  - `{path}`" for path in example_figs[:12]])
    if (EXAMPLE_REPORT_DIR / "density_risk_map_examples.md").exists():
        lines.append(f"- Existing example report: `{EXAMPLE_REPORT_DIR / 'density_risk_map_examples.md'}`")
    lines.extend(
        [
            "",
            "## Decision Before CNN",
            "",
            "A CNN branch should be accepted only if it beats this pack under the same sparse-input, leakage-safe setup. If it only improves full-map classification, it is outside the frozen goal.",
            "",
        ]
    )
    (REPORT_DIR / "final_noncnn_deliverable_pack.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme_draft(claim_table: pd.DataFrame) -> None:
    objective = claim_table[claim_table["claim_area"] == "objective_augmented_t0.60"]
    if not objective.empty:
        gain_min = float(objective["coverage_gain_min_pct"].iloc[0])
        gain_max = float(objective["coverage_gain_max_pct"].iloc[0])
        obj_sentence = (
            f"The confidence-gated objective policy improved defect coverage by {gain_min:.1f}% to {gain_max:.1f}% "
            "over coverage32 across 1%, 3%, 5%, and 10% first-pass densities, while increasing sampled-ratio bias."
        )
    else:
        obj_sentence = "The confidence-gated objective policy improved discovery but requires a representativeness warning."

    lines = [
        "# Wafer Defect Follow-Up Sampling",
        "",
        "Risk- and representativeness-aware follow-up sampling for sparse wafer defect observation.",
        "",
        "This project uses WM-811K dense wafer maps as offline reference data to evaluate follow-up sampling policies after first-pass sparse observation. The goal is not to classify a fully observed wafer image, but to recommend which unmeasured dies should be inspected next.",
        "",
        "## Key Idea",
        "",
        "- Start with a sparse first-pass wafer observation.",
        "- Rank unmeasured valid dies by defect risk using only inference-safe features.",
        "- Compare discovery gain against geometry-only representative sampling and report the bias tradeoff.",
        "",
        "## Not Ordinary WM-811K Classification",
        "",
        "Most public WM-811K examples use the full wafer map to predict a defect pattern class. This project masks the unmeasured dies and treats the dense map as an offline reference for evaluation only.",
        "",
        "## Main Result Snapshot",
        "",
        obj_sentence,
        "",
        "Pure ML ranking gives the largest discovery gain, but it is intentionally reported with a high representativeness warning. The guarded and objective-mode policies are more defensible when follow-up inspection must balance defect discovery against spatial sampling bias.",
        "",
        "## Policy Terminology",
        "",
        "| Policy | Selection logic | Main strength | Main risk |",
        "| --- | --- | --- | --- |",
        "| coverage32 | Geometry-only space filling | Spatial representativeness | May miss localized high-risk defects |",
        "| ml_rank32 | Highest predicted defect risk | High defect discovery | Over-concentration and sampled-ratio bias |",
        "| morphrisk32 | Morphology-aware risk ranking | Better high-risk targeting | Still biased if unconstrained |",
        "| guarded hybrid | Coverage backbone plus limited ML replacement | Balances discovery and representativeness | Needs guardrail validation |",
        "| objective-mode policy | Uses morphology confidence to select discovery-first or low-bias mode | Aligns policy with pattern objective | Depends on confidence calibration |",
        "",
        "## Leakage Prevention",
        "",
        "Allowed inference-time inputs: first-pass sparse observations, wafer geometry, candidate coordinates, risk scores derived from allowed features, morphology-risk probabilities, uncertainty scores, and coverage distance.",
        "",
        "Forbidden inference-time inputs: dense defect map, true failureType, actual defect ratio, total defect count, hidden defect coordinates, future wafer/lot information, and labels unavailable at follow-up decision time.",
        "",
        "## What This Project Is Not",
        "",
        "- Not a production fab recipe.",
        "- Not a production yield-improvement claim.",
        "- Not SEM image classification.",
        "- Not root-cause analysis.",
        "- Not complete wafer defect map reconstruction.",
        "- Not generic WM-811K full-map classification.",
        "",
        "## Current Finalization Status",
        "",
        "The current non-CNN baseline is frozen in `reports/final_noncnn_deliverable_pack_v1/final_noncnn_deliverable_pack.md`. A CNN branch should only be added if it improves the same sparse-input follow-up objective under the same leakage rules.",
        "",
    ]
    (REPORT_DIR / "README_rewrite_draft.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    inputs = read_inputs()
    final_policy = build_final_policy_table(inputs["policy_summary"], inputs["bootstrap_ci"])
    claim_table = build_claim_table(final_policy)
    risk_map_table = build_risk_map_table(inputs["risk_map"])
    stability_table = build_stability_table(inputs["stability"])
    leakage_table = build_leakage_audit_table()

    final_policy.to_csv(OUT_DIR / "final_policy_metric_table.csv", index=False)
    claim_table.to_csv(OUT_DIR / "final_claim_boundary_table.csv", index=False)
    risk_map_table.to_csv(OUT_DIR / "final_risk_map_metric_table.csv", index=False)
    stability_table.to_csv(OUT_DIR / "final_split_stability_table.csv", index=False)
    leakage_table.to_csv(OUT_DIR / "final_leakage_audit_table.csv", index=False)

    plot_gain_vs_bias(final_policy)
    plot_risk_map_metrics(risk_map_table)
    write_report(final_policy, claim_table, risk_map_table, stability_table, leakage_table)
    write_readme_draft(claim_table)

    print(f"wrote tables to {OUT_DIR}")
    print(f"wrote report to {REPORT_DIR / 'final_noncnn_deliverable_pack.md'}")
    print(f"wrote README draft to {REPORT_DIR / 'README_rewrite_draft.md'}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
