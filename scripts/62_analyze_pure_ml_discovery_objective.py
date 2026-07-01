from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_RISK_DIR = Path("data") / "processed" / "density_risk_maps_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "pure_ml_discovery_objective_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "52_pure_ml_discovery_objective_v1"

STRATEGIES = [
    "coverage32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
    "ml_rank32",
    "ml_biasaware32",
    "morphrisk32",
    "morphrisk_guarded32",
]

DISCOVERY_PATTERNS = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"}
LOW_BIAS_PATTERNS = {"Random", "Near-full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate pure ML defect-discovery value from representativeness/bias "
            "warnings using existing held-out follow-up results."
        )
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--risk-dir", type=Path, default=DEFAULT_RISK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--low-bias-delta", type=float, default=0.01)
    parser.add_argument("--moderate-bias-delta", type=float, default=0.03)
    parser.add_argument("--large-extra-gain-pct", type=float, default=10.0)
    return parser.parse_args()


def bias_band(delta: float, low: float, moderate: float) -> str:
    if delta <= 0:
        return "no_extra_bias"
    if delta <= low:
        return "low_bias_warning"
    if delta <= moderate:
        return "moderate_bias_warning"
    return "high_bias_warning"


def add_global_baseline(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    baseline = data[data["strategy"] == "coverage32"][
        [
            "target_density",
            "mean_absolute_error",
            "mean_ratio_error",
            "severe_miss_rate",
            "mean_defect_coverage",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_ratio_error": "baseline_ratio_error",
            "severe_miss_rate": "baseline_severe_miss_rate",
            "mean_defect_coverage": "baseline_defect_coverage",
        }
    )
    out = data.merge(baseline, on="target_density", how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["ratio_error_delta"] = out["mean_ratio_error"] - out["baseline_ratio_error"]
    out["severe_miss_delta"] = out["severe_miss_rate"] - out["baseline_severe_miss_rate"]
    out["defect_coverage_delta"] = out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    out["defect_coverage_gain_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    out["bias_band"] = out["absolute_error_delta"].map(
        lambda value: bias_band(float(value), args.low_bias_delta, args.moderate_bias_delta)
    )
    return out


def add_pattern_baseline(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    baseline = data[data["strategy"] == "coverage32"][
        [
            "target_density",
            "failureType",
            "mean_absolute_error",
            "mean_ratio_error",
            "severe_miss_rate",
            "mean_defect_coverage",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_ratio_error": "baseline_ratio_error",
            "severe_miss_rate": "baseline_severe_miss_rate",
            "mean_defect_coverage": "baseline_defect_coverage",
        }
    )
    out = data.merge(baseline, on=["target_density", "failureType"], how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["ratio_error_delta"] = out["mean_ratio_error"] - out["baseline_ratio_error"]
    out["severe_miss_delta"] = out["severe_miss_rate"] - out["baseline_severe_miss_rate"]
    out["defect_coverage_delta"] = out["mean_defect_coverage"] - out["baseline_defect_coverage"]
    out["defect_coverage_gain_pct"] = (
        out["defect_coverage_delta"] / out["baseline_defect_coverage"] * 100.0
    )
    out["bias_band"] = out["absolute_error_delta"].map(
        lambda value: bias_band(float(value), args.low_bias_delta, args.moderate_bias_delta)
    )
    return out


def load_views(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(args.eval_dir / "density_followup_eval_summary.csv")
    pattern = pd.read_csv(args.eval_dir / "density_followup_pattern_summary.csv")
    summary = summary[summary["strategy"].isin(STRATEGIES)].copy()
    pattern = pattern[pattern["strategy"].isin(STRATEGIES)].copy()
    return add_global_baseline(summary, args), add_pattern_baseline(pattern, args)


def attach_risk_map_metrics(global_view: pd.DataFrame, risk_dir: Path) -> pd.DataFrame:
    risk_path = risk_dir / "density_risk_map_summary.csv"
    if not risk_path.exists():
        return global_view
    risk = pd.read_csv(risk_path)
    risk = risk[risk["risk_map"].isin(["risk_point", "risk_morphrisk", "risk_guarded"])].copy()
    risk = risk[
        [
            "target_density",
            "risk_map",
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top10pct_iou",
            "mean_top32_defect_coverage",
        ]
    ]
    risk_wide = risk.pivot_table(
        index="target_density",
        columns="risk_map",
        values=[
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top10pct_iou",
            "mean_top32_defect_coverage",
        ],
        aggfunc="mean",
    )
    risk_wide.columns = [f"{metric}_{risk_map}" for metric, risk_map in risk_wide.columns]
    return global_view.merge(risk_wide.reset_index(), on="target_density", how="left")


def classify_objective(row: pd.Series, args: argparse.Namespace) -> str:
    if row["strategy"] == "coverage32":
        return "representative_baseline"
    if float(row["defect_coverage_delta"]) <= 0:
        return "no_discovery_gain"
    if float(row["absolute_error_delta"]) <= args.low_bias_delta:
        return "low_bias_discovery"
    return "discovery_first_with_bias_warning"


def build_global_policy_view(global_view: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = attach_risk_map_metrics(global_view, args.risk_dir)
    out["objective_interpretation"] = out.apply(lambda row: classify_objective(row, args), axis=1)
    cols = [
        "target_density",
        "strategy",
        "wafers",
        "mean_absolute_error",
        "absolute_error_delta",
        "mean_ratio_error",
        "ratio_error_delta",
        "severe_miss_rate",
        "severe_miss_delta",
        "mean_defect_coverage",
        "defect_coverage_gain_pct",
        "bias_band",
        "objective_interpretation",
    ]
    risk_cols = [col for col in out.columns if col.startswith("mean_")]
    cols = cols + [col for col in risk_cols if col not in cols]
    return out[cols].sort_values(["target_density", "strategy"]).reset_index(drop=True)


def build_pattern_objective_view(pattern_view: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = pattern_view.copy()
    data["objective_interpretation"] = data.apply(lambda row: classify_objective(row, args), axis=1)

    rows: list[dict[str, object]] = []
    for (density, pattern), group in data.groupby(["target_density", "failureType"], observed=False):
        positive = group[
            (group["strategy"] != "coverage32") & (group["defect_coverage_gain_pct"] > 0)
        ].copy()
        if positive.empty:
            best_discovery = group[group["strategy"] == "coverage32"].iloc[0]
            low_bias = best_discovery
        else:
            best_discovery = positive.sort_values(
                ["defect_coverage_gain_pct", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0]
            low_bias_pool = positive[positive["absolute_error_delta"] <= args.low_bias_delta]
            if low_bias_pool.empty:
                low_bias = group[group["strategy"] == "coverage32"].iloc[0]
            else:
                low_bias = low_bias_pool.sort_values(
                    ["defect_coverage_gain_pct", "absolute_error_delta"],
                    ascending=[False, True],
                ).iloc[0]

        ml = group[group["strategy"] == "ml_rank32"]
        ml_row = ml.iloc[0] if not ml.empty else best_discovery
        pure_ml_extra_over_low_bias = (
            float(ml_row["defect_coverage_gain_pct"]) - float(low_bias["defect_coverage_gain_pct"])
        )

        if pattern in LOW_BIAS_PATTERNS:
            recommended_objective = "low_bias_or_coverage"
        elif pure_ml_extra_over_low_bias >= args.large_extra_gain_pct:
            recommended_objective = "discovery_first_allow_ml_bias"
        elif float(low_bias["defect_coverage_gain_pct"]) > 0:
            recommended_objective = "low_bias_discovery"
        elif pattern in DISCOVERY_PATTERNS and float(best_discovery["defect_coverage_gain_pct"]) > 0:
            recommended_objective = "discovery_first_allow_ml_bias"
        else:
            recommended_objective = "coverage_or_uncertain"

        rows.append(
            {
                "target_density": density,
                "failureType": pattern,
                "wafers": int(best_discovery["wafers"]),
                "recommended_objective": recommended_objective,
                "best_discovery_strategy": best_discovery["strategy"],
                "best_discovery_gain_pct": float(best_discovery["defect_coverage_gain_pct"]),
                "best_discovery_abs_error_delta": float(best_discovery["absolute_error_delta"]),
                "best_discovery_bias_band": best_discovery["bias_band"],
                "low_bias_strategy": low_bias["strategy"],
                "low_bias_gain_pct": float(low_bias["defect_coverage_gain_pct"]),
                "low_bias_abs_error_delta": float(low_bias["absolute_error_delta"]),
                "pure_ml_gain_pct": float(ml_row["defect_coverage_gain_pct"]),
                "pure_ml_abs_error_delta": float(ml_row["absolute_error_delta"]),
                "pure_ml_extra_gain_over_low_bias_pct": pure_ml_extra_over_low_bias,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_density", "failureType"]).reset_index(drop=True)


def aggregate_pattern_modes(pattern_objective: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pattern, group in pattern_objective.groupby("failureType", observed=False):
        counts = group["recommended_objective"].value_counts()
        rows.append(
            {
                "failureType": pattern,
                "dominant_objective": str(counts.index[0]),
                "density_count": int(len(group)),
                "avg_best_discovery_gain_pct": float(group["best_discovery_gain_pct"].mean()),
                "avg_low_bias_gain_pct": float(group["low_bias_gain_pct"].mean()),
                "avg_pure_ml_gain_pct": float(group["pure_ml_gain_pct"].mean()),
                "avg_pure_ml_extra_gain_over_low_bias_pct": float(
                    group["pure_ml_extra_gain_over_low_bias_pct"].mean()
                ),
                "avg_pure_ml_abs_error_delta": float(group["pure_ml_abs_error_delta"].mean()),
                **{f"count_{name}": int(counts.get(name, 0)) for name in sorted(counts.index)},
            }
        )
    return pd.DataFrame(rows).sort_values("failureType").reset_index(drop=True)


def plot_outputs(
    global_policy: pd.DataFrame,
    pattern_objective: pd.DataFrame,
    aggregate: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot = global_policy[global_policy["strategy"] != "coverage32"].copy()
    plot["density_pct"] = plot["target_density"] * 100.0
    plt.figure(figsize=(9.0, 5.4))
    sns.scatterplot(
        data=plot,
        x="absolute_error_delta",
        y="defect_coverage_gain_pct",
        hue="density_pct",
        style="strategy",
        s=85,
        palette="viridis",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.axvline(0.03, color="#aaaaaa", linestyle=":", linewidth=0.9)
    plt.xlabel("Absolute-error delta vs coverage32 (representativeness warning)")
    plt.ylabel("Defect coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "global_discovery_gain_vs_representativeness_warning.png", dpi=180)
    plt.close()

    pure_ml = pattern_objective.copy()
    pure_ml["density_pct"] = (pure_ml["target_density"] * 100.0).map(lambda value: f"{value:g}%")
    pivot = pure_ml.pivot_table(
        index="failureType",
        columns="density_pct",
        values="pure_ml_gain_pct",
        aggfunc="mean",
    )
    plt.figure(figsize=(7.4, 5.4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Pure ML top32 discovery gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pure_ml_pattern_gain_heatmap.png", dpi=180)
    plt.close()

    pivot_delta = pure_ml.pivot_table(
        index="failureType",
        columns="density_pct",
        values="pure_ml_abs_error_delta",
        aggfunc="mean",
    )
    plt.figure(figsize=(7.4, 5.4))
    sns.heatmap(pivot_delta, annot=True, fmt=".3f", cmap="Reds")
    plt.title("Pure ML absolute-error delta vs coverage32")
    plt.tight_layout()
    plt.savefig(fig_dir / "pure_ml_pattern_bias_warning_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.4, 4.8))
    sns.barplot(
        data=aggregate,
        x="failureType",
        y="avg_pure_ml_extra_gain_over_low_bias_pct",
        hue="dominant_objective",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Avg pure ML extra gain over low-bias policy (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pure_ml_extra_gain_over_low_bias_by_pattern.png", dpi=180)
    plt.close()


def write_report(
    global_policy: pd.DataFrame,
    pattern_objective: pd.DataFrame,
    aggregate: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: list[str] = [
        "# Pure ML Discovery Objective v1",
        "",
        "Purpose: answer whether a biased ML policy can be useful when the objective is defect-rich-region discovery.",
        "",
        "Core interpretation:",
        "",
        "```text",
        "Higher sampled-ratio / absolute-error delta is not automatically bad.",
        "For discovery-first follow-up, it can mean the model concentrated points in a real defect-rich region.",
        "For wafer-level defect-ratio estimation, it remains a representativeness warning.",
        "```",
        "",
        "## Global Pure ML Result",
        "",
    ]
    focus = global_policy[global_policy["strategy"].isin(["coverage32", "ml_rank32"])].copy()
    for row in focus.itertuples(index=False):
        if row.strategy == "coverage32":
            lines.append(
                f"- {row.target_density:.0%} coverage32: defect coverage "
                f"{row.mean_defect_coverage:.4f}, abs error {row.mean_absolute_error:.4f}"
            )
        else:
            lines.append(
                f"- {row.target_density:.0%} pure ML top32: coverage gain "
                f"{row.defect_coverage_gain_pct:.2f}%, abs-error delta "
                f"{row.absolute_error_delta:.4f}, ratio-error delta "
                f"{row.ratio_error_delta:.4f}, bias band {row.bias_band}"
            )
    lines.extend(["", "## Pattern Objective Summary", ""])
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"- {row.failureType}: dominant objective {row.dominant_objective}; "
            f"avg pure ML gain {row.avg_pure_ml_gain_pct:.2f}%, "
            f"extra over low-bias {row.avg_pure_ml_extra_gain_over_low_bias_pct:.2f}%, "
            f"pure ML abs-error delta {row.avg_pure_ml_abs_error_delta:.4f}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use pure ML top32 as a required discovery-first baseline, not as a forbidden policy.",
            "Keep coverage32 as the representative baseline so the project can quantify what ML gains and what representativeness it gives up.",
            "For final claims, report both objective views:",
            "",
            "```text",
            "Discovery-first: maximize defect coverage, report bias as an expected concentration warning.",
            "Estimation-sensitive: require low absolute-error delta or use coverage32 / guarded replacement.",
            "```",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    global_view, pattern_view = load_views(args)
    global_policy = build_global_policy_view(global_view, args)
    pattern_objective = build_pattern_objective_view(pattern_view, args)
    aggregate = aggregate_pattern_modes(pattern_objective)

    global_policy.to_csv(args.out_dir / "global_policy_objective_view.csv", index=False)
    pattern_objective.to_csv(args.out_dir / "pattern_density_objective_view.csv", index=False)
    aggregate.to_csv(args.out_dir / "pattern_objective_summary.csv", index=False)
    plot_outputs(global_policy, pattern_objective, aggregate, args.fig_dir)
    write_report(
        global_policy,
        pattern_objective,
        aggregate,
        args.out_dir / "pure_ml_discovery_objective_report.md",
    )

    print(f"wrote pure ML discovery objective outputs to {args.out_dir}")
    print(f"wrote pure ML discovery objective figures to {args.fig_dir}")
    print(
        global_policy[global_policy["strategy"].isin(["coverage32", "ml_rank32"])][
            [
                "target_density",
                "strategy",
                "mean_defect_coverage",
                "defect_coverage_gain_pct",
                "absolute_error_delta",
                "bias_band",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
