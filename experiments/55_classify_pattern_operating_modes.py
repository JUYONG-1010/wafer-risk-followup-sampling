from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_TUNING_DIR = Path("data") / "processed" / "density_hybrid_scoring_tuning_v1"
DEFAULT_FIXED_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "pattern_operating_modes_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "45_pattern_operating_modes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify pattern-specific operating modes for defect discovery."
    )
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--low-bias-delta", type=float, default=0.01)
    parser.add_argument("--moderate-bias-delta", type=float, default=0.03)
    parser.add_argument("--minimum-useful-gain-pct", type=float, default=1.0)
    parser.add_argument("--large-extra-gain-pct", type=float, default=5.0)
    parser.add_argument("--low-bias-capture-ratio", type=float, default=0.60)
    return parser.parse_args()


def bias_band(delta: float, low: float, moderate: float) -> str:
    if delta <= 0:
        return "improves_estimation"
    if delta <= low:
        return "low_bias_warning"
    if delta <= moderate:
        return "moderate_bias_warning"
    return "high_bias_warning"


def strategy_family(strategy: str) -> str:
    if strategy == "coverage32":
        return "coverage"
    if strategy.startswith("morphrisk") or strategy in {"ml_rank32", "ml_biasaware32"}:
        return "risk_only_or_ml32"
    if re.search(r"_N\d+$", strategy) or strategy.startswith("hybrid_guarded"):
        return "hybrid_replacement"
    return "other"


def add_baseline_deltas(data: pd.DataFrame) -> pd.DataFrame:
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


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    tuned = pd.read_csv(args.tuning_dir / "density_hybrid_scoring_tuning_pattern_summary.csv")
    tuned["source"] = "tuned_grid"

    fixed = pd.read_csv(args.fixed_dir / "density_followup_pattern_summary.csv")
    fixed = fixed[
        fixed["strategy"].isin(
            [
                "coverage32",
                "hybrid_guarded1",
                "hybrid_guarded2",
                "hybrid_guarded4",
                "morphrisk_guarded32",
                "ml_rank32",
                "ml_biasaware32",
            ]
        )
    ].copy()
    fixed = fixed[
        ["target_density", "failureType", "strategy", "wafers", "mean_absolute_error", "mean_defect_coverage"]
    ]
    fixed["source"] = "fixed_policy"

    data = pd.concat([tuned, fixed], ignore_index=True)
    data = data.drop_duplicates(["target_density", "failureType", "strategy"], keep="first")
    data = add_baseline_deltas(data)
    data["bias_band"] = data["absolute_error_delta"].map(
        lambda value: bias_band(float(value), args.low_bias_delta, args.moderate_bias_delta)
    )
    data["strategy_family"] = data["strategy"].map(strategy_family)
    return data


def classify_row(row: pd.Series, args: argparse.Namespace) -> str:
    best_gain = float(row["discovery_gain_pct"])
    low_gain = row["low_bias_gain_pct"]
    extra_gain = row["extra_discovery_gain_pct"]
    best_bias = str(row["discovery_bias_band"])

    if str(row["failureType"]) == "Near-full":
        return "coverage32_or_uncertain"
    if best_gain < args.minimum_useful_gain_pct:
        return "coverage32_or_uncertain"
    if pd.isna(low_gain) or float(low_gain) < args.minimum_useful_gain_pct:
        if best_bias == "high_bias_warning":
            return "discovery_first_high_bias"
        return "discovery_first"
    if best_gain > 0 and float(low_gain) / best_gain >= args.low_bias_capture_ratio:
        return "low_bias_discovery"
    if float(extra_gain) >= args.large_extra_gain_pct:
        return "discovery_first"
    return "low_bias_default_discovery_optional"


def choose_modes(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = candidates[
        (candidates["strategy"] != "coverage32")
        & (candidates["defect_coverage_relative_improvement_pct"].notna())
    ].copy()
    for (density, pattern), group in usable.groupby(["target_density", "failureType"], observed=False):
        positive = group[group["defect_coverage_relative_improvement_pct"] > 0].copy()
        if positive.empty:
            baseline = candidates[
                (candidates["target_density"] == density)
                & (candidates["failureType"] == pattern)
                & (candidates["strategy"] == "coverage32")
            ].iloc[0]
            rows.append(
                {
                    "target_density": density,
                    "failureType": pattern,
                    "recommended_mode": "coverage32_or_uncertain",
                    "discovery_strategy": "coverage32",
                    "discovery_gain_pct": 0.0,
                    "discovery_abs_error_delta": 0.0,
                    "discovery_bias_band": "baseline",
                    "low_bias_strategy": "coverage32",
                    "low_bias_gain_pct": 0.0,
                    "low_bias_abs_error_delta": 0.0,
                    "extra_discovery_gain_pct": 0.0,
                    "baseline_defect_coverage": baseline["baseline_defect_coverage"],
                    "baseline_absolute_error": baseline["baseline_absolute_error"],
                }
            )
            continue

        discovery = positive.sort_values(
            ["defect_coverage_relative_improvement_pct", "absolute_error_delta"],
            ascending=[False, True],
        ).iloc[0]
        low_bias_pool = positive[positive["absolute_error_delta"] <= args.low_bias_delta]
        if low_bias_pool.empty:
            low_bias_strategy = pd.NA
            low_bias_gain = pd.NA
            low_bias_delta = pd.NA
        else:
            low_bias = low_bias_pool.sort_values(
                ["defect_coverage_relative_improvement_pct", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0]
            low_bias_strategy = low_bias["strategy"]
            low_bias_gain = float(low_bias["defect_coverage_relative_improvement_pct"])
            low_bias_delta = float(low_bias["absolute_error_delta"])

        extra_gain = (
            float(discovery["defect_coverage_relative_improvement_pct"]) - float(low_bias_gain)
            if not pd.isna(low_bias_gain)
            else float(discovery["defect_coverage_relative_improvement_pct"])
        )
        record = {
            "target_density": density,
            "failureType": pattern,
            "discovery_strategy": discovery["strategy"],
            "discovery_family": discovery["strategy_family"],
            "discovery_gain_pct": float(discovery["defect_coverage_relative_improvement_pct"]),
            "discovery_abs_error_delta": float(discovery["absolute_error_delta"]),
            "discovery_bias_band": discovery["bias_band"],
            "low_bias_strategy": low_bias_strategy,
            "low_bias_gain_pct": low_bias_gain,
            "low_bias_abs_error_delta": low_bias_delta,
            "extra_discovery_gain_pct": extra_gain,
            "baseline_defect_coverage": float(discovery["baseline_defect_coverage"]),
            "baseline_absolute_error": float(discovery["baseline_absolute_error"]),
        }
        rows.append(record)

    modes = pd.DataFrame.from_records(rows)
    modes["recommended_mode"] = modes.apply(lambda row: classify_row(row, args), axis=1)
    return modes.sort_values(["target_density", "failureType"]).reset_index(drop=True)


def aggregate_pattern_modes(modes: pd.DataFrame) -> pd.DataFrame:
    mode_priority = {
        "coverage32_or_uncertain": 0,
        "low_bias_discovery": 1,
        "low_bias_default_discovery_optional": 2,
        "discovery_first": 3,
        "discovery_first_high_bias": 4,
    }
    rows: list[dict[str, object]] = []
    for pattern, group in modes.groupby("failureType", observed=False):
        counts = group["recommended_mode"].value_counts()
        dominant = counts.index[0]
        avg_discovery_gain = float(group["discovery_gain_pct"].mean())
        avg_low_bias_gain = float(group["low_bias_gain_pct"].dropna().mean()) if group["low_bias_gain_pct"].notna().any() else pd.NA
        avg_extra = float(group["extra_discovery_gain_pct"].mean())
        avg_bias = float(group["discovery_abs_error_delta"].mean())
        max_priority_mode = max(group["recommended_mode"], key=lambda mode: mode_priority.get(str(mode), -1))
        rows.append(
            {
                "failureType": pattern,
                "dominant_mode": dominant,
                "highest_intensity_mode": max_priority_mode,
                "density_count": int(len(group)),
                "avg_discovery_gain_pct": avg_discovery_gain,
                "avg_low_bias_gain_pct": avg_low_bias_gain,
                "avg_extra_discovery_gain_pct": avg_extra,
                "avg_discovery_abs_error_delta": avg_bias,
                **{f"count_{mode}": int(counts.get(mode, 0)) for mode in mode_priority},
            }
        )
    return pd.DataFrame(rows).sort_values("failureType").reset_index(drop=True)


def plot_modes(modes: pd.DataFrame, aggregate: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot = modes.copy()
    plot["density_pct"] = (plot["target_density"] * 100.0).map(lambda value: f"{value:g}%")
    pivot = plot.pivot_table(
        index="failureType",
        columns="density_pct",
        values="discovery_gain_pct",
        aggfunc="mean",
    )
    plt.figure(figsize=(7.4, 5.4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Best discovery-first coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_discovery_gain_heatmap.png", dpi=180)
    plt.close()

    mode_codes = {
        "coverage32_or_uncertain": 0,
        "low_bias_discovery": 1,
        "low_bias_default_discovery_optional": 2,
        "discovery_first": 3,
        "discovery_first_high_bias": 4,
    }
    coded = plot.copy()
    coded["mode_code"] = coded["recommended_mode"].map(mode_codes)
    mode_pivot = coded.pivot_table(
        index="failureType",
        columns="density_pct",
        values="mode_code",
        aggfunc="mean",
    )
    plt.figure(figsize=(7.4, 5.4))
    sns.heatmap(mode_pivot, annot=True, fmt=".0f", cmap="viridis", cbar_kws={"label": "mode code"})
    plt.title("Pattern operating mode code")
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_operating_mode_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    sns.barplot(
        data=aggregate,
        x="failureType",
        y="avg_extra_discovery_gain_pct",
        hue="dominant_mode",
    )
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Avg extra gain of discovery-first over low-bias (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_extra_discovery_gain.png", dpi=180)
    plt.close()


def write_report(modes: pd.DataFrame, aggregate: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Pattern Operating Modes v1",
        "",
        "Purpose: classify which wafer defect patterns should use defect-rich discovery mode and which should use low-bias discovery mode.",
        "",
        "## Classification Logic",
        "",
        "For each pattern and first-pass density:",
        "",
        "```text",
        "1. find the policy with the largest defect coverage gain vs coverage32",
        "2. find the best policy with absolute-error delta <= 0.01",
        "3. compare the extra discovery gain from allowing more bias",
        "```",
        "",
        "Mode meanings:",
        "",
        "```text",
        "discovery_first: larger ML-risk replacement is justified by much higher defect coverage",
        "discovery_first_high_bias: discovery gain is high, but bias warning is high",
        "low_bias_discovery: low-bias policy captures most useful gain",
        "low_bias_default_discovery_optional: low-bias default, discovery-first optional",
        "coverage32_or_uncertain: no reliable positive discovery gain",
        "```",
        "",
        "Domain override:",
        "",
        "```text",
        "Near-full is not treated as a defect-rich localization target because defects are already widespread.",
        "For Near-full, use coverage32 or a low-bias mode rather than aggressive discovery-first replacement.",
        "```",
        "",
        "## Pattern Summary",
        "",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"- {row.failureType}: dominant={row.dominant_mode}, "
            f"highest_intensity={row.highest_intensity_mode}, "
            f"avg discovery gain={row.avg_discovery_gain_pct:.2f}%, "
            f"avg low-bias gain={row.avg_low_bias_gain_pct:.2f}%, "
            f"avg extra gain={row.avg_extra_discovery_gain_pct:.2f}%, "
            f"avg discovery abs-error delta={row.avg_discovery_abs_error_delta:.4f}"
        )
    lines.extend(["", "## Density-Level Decisions", ""])
    for row in modes.itertuples(index=False):
        low_bias = (
            "none"
            if pd.isna(row.low_bias_strategy)
            else f"{row.low_bias_strategy} ({row.low_bias_gain_pct:.2f}%, delta {row.low_bias_abs_error_delta:.4f})"
        )
        lines.append(
            f"- {row.target_density:.0%}, {row.failureType}: {row.recommended_mode}; "
            f"discovery={row.discovery_strategy} ({row.discovery_gain_pct:.2f}%, "
            f"delta {row.discovery_abs_error_delta:.4f}, {row.discovery_bias_band}); "
            f"low_bias={low_bias}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args)
    modes = choose_modes(candidates, args)
    aggregate = aggregate_pattern_modes(modes)

    candidates.to_csv(args.out_dir / "pattern_policy_candidates.csv", index=False)
    modes.to_csv(args.out_dir / "pattern_density_operating_modes.csv", index=False)
    aggregate.to_csv(args.out_dir / "pattern_operating_mode_summary.csv", index=False)
    plot_modes(modes, aggregate, args.fig_dir)
    write_report(modes, aggregate, args.out_dir / "pattern_operating_modes_report.md")

    print(f"wrote pattern operating-mode outputs to {args.out_dir}")
    print(f"wrote pattern operating-mode figures to {args.fig_dir}")
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
