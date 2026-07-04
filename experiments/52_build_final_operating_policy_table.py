from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_FOLLOWUP_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_SWEEP_DIR = Path("data") / "processed" / "density_followup_replacement_sweep_v1"
DEFAULT_RISK_DIR = Path("data") / "processed" / "density_risk_maps_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "final_operating_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "42_final_operating_policy_v1"

POLICY_ORDER = [
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
RISK_POLICY_MAP = {
    "hybrid_guarded1": "risk_guarded",
    "hybrid_guarded2": "risk_guarded",
    "hybrid_guarded4": "risk_guarded",
    "morphrisk_guarded32": "risk_guarded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final operating policy comparison table."
    )
    parser.add_argument("--followup-dir", type=Path, default=DEFAULT_FOLLOWUP_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--risk-dir", type=Path, default=DEFAULT_RISK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--abs-error-tolerance", type=float, default=0.01)
    return parser.parse_args()


def policy_rank(strategy: str) -> int:
    return POLICY_ORDER.index(strategy) if strategy in POLICY_ORDER else len(POLICY_ORDER)


def load_policy_summary(followup_dir: Path, risk_dir: Path) -> pd.DataFrame:
    followup = pd.read_csv(followup_dir / "density_followup_eval_summary.csv")
    followup = followup[followup["strategy"].isin(POLICY_ORDER)].copy()

    baseline = followup[followup["strategy"] == "coverage32"][
        [
            "target_density",
            "mean_absolute_error",
            "mean_defect_coverage",
            "severe_miss_rate",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
            "severe_miss_rate": "baseline_severe_miss_rate",
        }
    )
    out = followup.merge(baseline, on="target_density", how="left")
    out["absolute_error_delta"] = (
        out["mean_absolute_error"] - out["baseline_absolute_error"]
    )
    out["defect_coverage_delta"] = (
        out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    )
    out["defect_coverage_relative_improvement_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    out["severe_miss_delta"] = out["severe_miss_rate"] - out["baseline_severe_miss_rate"]
    out["policy_label"] = out["strategy"].map(POLICY_LABELS)
    out["risk_map"] = out["strategy"].map(RISK_POLICY_MAP)

    risk = pd.read_csv(risk_dir / "density_risk_map_summary.csv")
    risk = risk[risk["risk_map"] == "risk_guarded"][
        [
            "target_density",
            "risk_map",
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top10pct_iou",
            "mean_top32_defect_coverage",
        ]
    ]
    out = out.merge(risk, on=["target_density", "risk_map"], how="left")
    out["policy_rank"] = out["strategy"].map(policy_rank)
    return out.sort_values(["target_density", "policy_rank"]).reset_index(drop=True)


def load_pattern_summary(followup_dir: Path, risk_dir: Path) -> pd.DataFrame:
    pattern = pd.read_csv(followup_dir / "density_followup_pattern_summary.csv")
    pattern = pattern[pattern["strategy"].isin(POLICY_ORDER)].copy()
    baseline = pattern[pattern["strategy"] == "coverage32"][
        [
            "target_density",
            "failureType",
            "mean_absolute_error",
            "mean_defect_coverage",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
        }
    )
    out = pattern.merge(baseline, on=["target_density", "failureType"], how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["defect_coverage_delta"] = (
        out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    )
    out["defect_coverage_relative_improvement_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    out["policy_label"] = out["strategy"].map(POLICY_LABELS)
    out["risk_map"] = out["strategy"].map(RISK_POLICY_MAP)

    risk_pattern = pd.read_csv(risk_dir / "density_risk_map_pattern_summary.csv")
    risk_pattern = risk_pattern[risk_pattern["risk_map"] == "risk_guarded"][
        [
            "target_density",
            "failureType",
            "risk_map",
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top10pct_iou",
            "mean_top32_defect_coverage",
        ]
    ]
    out = out.merge(
        risk_pattern,
        on=["target_density", "failureType", "risk_map"],
        how="left",
    )
    out["policy_rank"] = out["strategy"].map(policy_rank)
    return out.sort_values(["target_density", "failureType", "policy_rank"]).reset_index(
        drop=True
    )


def choose_operating_rule(
    policy_summary: pd.DataFrame,
    abs_error_tolerance: float,
) -> pd.DataFrame:
    candidates = policy_summary[
        policy_summary["strategy"].isin(["hybrid_guarded1", "hybrid_guarded2", "hybrid_guarded4"])
    ].copy()
    rows: list[pd.Series] = []
    for density, group in candidates.groupby("target_density", observed=False):
        allowed = group[
            (group["absolute_error_delta"] <= abs_error_tolerance)
            & (group["defect_coverage_delta"] > 0.0)
        ]
        if allowed.empty:
            chosen = policy_summary[
                (policy_summary["target_density"] == density)
                & (policy_summary["strategy"] == "coverage32")
            ].iloc[0].copy()
            best_hybrid = group.sort_values(
                ["absolute_error_delta", "defect_coverage_delta"],
                ascending=[True, False],
            ).iloc[0]
            chosen["rule_reason"] = (
                "keep coverage32; no hybrid candidate is inside the absolute-error tolerance"
            )
            chosen["best_hybrid_candidate"] = best_hybrid["policy_label"]
            chosen["best_hybrid_gain_pct"] = best_hybrid[
                "defect_coverage_relative_improvement_pct"
            ]
            chosen["best_hybrid_abs_error_delta"] = best_hybrid["absolute_error_delta"]
            chosen["hybrid_guardrail_pass"] = False
        else:
            chosen = allowed.sort_values(
                ["defect_coverage_delta", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0].copy()
            chosen["rule_reason"] = "highest coverage gain inside the absolute-error tolerance"
            chosen["best_hybrid_candidate"] = chosen["policy_label"]
            chosen["best_hybrid_gain_pct"] = chosen[
                "defect_coverage_relative_improvement_pct"
            ]
            chosen["best_hybrid_abs_error_delta"] = chosen["absolute_error_delta"]
            chosen["hybrid_guardrail_pass"] = True
        rows.append(chosen)
    rule = pd.DataFrame(rows).reset_index(drop=True)
    rule["abs_error_tolerance"] = abs_error_tolerance
    return rule


def plot_policy_summary(policy_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = policy_summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    data["policy_label"] = pd.Categorical(
        data["policy_label"],
        categories=[POLICY_LABELS[s] for s in POLICY_ORDER],
        ordered=True,
    )

    plt.figure(figsize=(9.2, 5.2))
    sns.lineplot(
        data=data,
        x="density_pct",
        y="mean_defect_coverage",
        hue="policy_label",
        marker="o",
    )
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Mean defect coverage")
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_defect_coverage_vs_density.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.2, 5.2))
    sns.lineplot(
        data=data,
        x="density_pct",
        y="mean_absolute_error",
        hue="policy_label",
        marker="o",
    )
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Mean absolute error")
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_absolute_error_vs_density.png", dpi=180)
    plt.close()

    trade = data[data["strategy"] != "coverage32"].copy()
    plt.figure(figsize=(8.4, 5.6))
    sns.scatterplot(
        data=trade,
        x="absolute_error_delta",
        y="defect_coverage_relative_improvement_pct",
        hue="density_pct",
        style="policy_label",
        s=95,
        palette="viridis",
    )
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Absolute-error delta vs coverage32")
    plt.ylabel("Defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "policy_gain_vs_error_tradeoff.png", dpi=180)
    plt.close()


def plot_pattern_summary(pattern_summary: pd.DataFrame, fig_dir: Path) -> None:
    focus = pattern_summary[
        pattern_summary["strategy"].isin(["hybrid_guarded1", "hybrid_guarded2", "hybrid_guarded4"])
    ].copy()
    focus["density_policy"] = (
        (focus["target_density"] * 100.0).map(lambda v: f"{v:g}%")
        + " "
        + focus["policy_label"].astype(str)
    )
    pivot = focus.pivot_table(
        index="failureType",
        columns="density_policy",
        values="defect_coverage_relative_improvement_pct",
        aggfunc="mean",
    )
    plt.figure(figsize=(13.5, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0.0)
    plt.title("Pattern-wise defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_policy_coverage_gain_heatmap.png", dpi=180)
    plt.close()


def write_report(
    policy_summary: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    operating_rule: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = [
        "# Final Operating Policy Comparison v1",
        "",
        "Purpose: choose the follow-up operating rule before further model tuning.",
        "",
        "Primary metric: defect coverage among limited follow-up points.",
        "Guardrail metric: absolute-error delta vs coverage32.",
        "Diagnostic metrics: ROC-AUC, average precision, Top-10% IoU of the underlying risk map.",
        "",
        "## Global Policy Table",
        "",
    ]
    cols = [
        "target_density",
        "policy_label",
        "mean_defect_coverage",
        "defect_coverage_relative_improvement_pct",
        "mean_absolute_error",
        "absolute_error_delta",
        "mean_roc_auc",
        "mean_average_precision",
        "mean_top10pct_iou",
    ]
    for row in policy_summary[cols].itertuples(index=False):
        auc = "n/a" if pd.isna(row.mean_roc_auc) else f"{row.mean_roc_auc:.3f}"
        ap = "n/a" if pd.isna(row.mean_average_precision) else f"{row.mean_average_precision:.3f}"
        iou = "n/a" if pd.isna(row.mean_top10pct_iou) else f"{row.mean_top10pct_iou:.3f}"
        lines.append(
            f"- {row.target_density:.0%}, {row.policy_label}: "
            f"coverage={row.mean_defect_coverage:.4f}, "
            f"gain={row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs_error={row.mean_absolute_error:.4f}, "
            f"abs_delta={row.absolute_error_delta:.4f}, "
            f"AUC={auc}, AP={ap}, Top10IoU={iou}"
        )
    lines.extend(["", "## Suggested Operating Rule", ""])
    for row in operating_rule.itertuples(index=False):
        if row.strategy == "coverage32":
            lines.append(
                f"- {row.target_density:.0%}: keep {row.policy_label}; "
                f"best hybrid candidate is {row.best_hybrid_candidate} "
                f"(gain {row.best_hybrid_gain_pct:.2f}%, "
                f"abs-error delta {row.best_hybrid_abs_error_delta:.4f})"
            )
        else:
            lines.append(
                f"- {row.target_density:.0%}: {row.policy_label}; "
                f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
                f"absolute-error delta {row.absolute_error_delta:.4f}"
            )
    lines.extend(
        [
            "",
            "## Pattern-Level Caution",
            "",
            "Pattern-wise gains are not uniform. The operating rule should be presented as a conservative default, not a universal optimum.",
            "",
        ]
    )
    focus = pattern_summary[
        (pattern_summary["strategy"] == "hybrid_guarded1")
        & (pattern_summary["target_density"].isin([0.03, 0.10]))
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, {row.failureType}: "
            f"hybrid N=1 gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f}, "
            f"risk Top10IoU {row.mean_top10pct_iou:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The next model-improvement work should optimize the constrained follow-up objective, not IoU alone.",
            "IoU remains a diagnostic for spatial localization quality.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    policy_summary = load_policy_summary(args.followup_dir, args.risk_dir)
    pattern_summary = load_pattern_summary(args.followup_dir, args.risk_dir)
    operating_rule = choose_operating_rule(policy_summary, args.abs_error_tolerance)

    policy_summary.to_csv(args.out_dir / "final_operating_policy_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "final_operating_policy_pattern_summary.csv", index=False)
    operating_rule.to_csv(args.out_dir / "final_operating_policy_recommendation.csv", index=False)
    plot_policy_summary(policy_summary, args.fig_dir)
    plot_pattern_summary(pattern_summary, args.fig_dir)
    write_report(
        policy_summary,
        pattern_summary,
        operating_rule,
        args.out_dir / "final_operating_policy_report.md",
    )

    print(f"wrote final operating policy outputs to {args.out_dir}")
    print(f"wrote final operating policy figures to {args.fig_dir}")
    print(
        operating_rule[
            [
                "target_density",
                "policy_label",
                "mean_defect_coverage",
                "defect_coverage_relative_improvement_pct",
                "mean_absolute_error",
                "absolute_error_delta",
                "rule_reason",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
