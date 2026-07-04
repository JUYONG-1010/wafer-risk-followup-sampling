from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = (
    Path("data")
    / "processed"
    / "point_ranking_v1_diverse_medium"
    / "model_training"
    / "point_ranking_eval_results.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "point_ranking_bias_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze over/underestimation bias in point-ranking policies."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def add_bias_flags(data: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    out = data.copy()
    out["overestimated"] = (out["ratio_error"] > epsilon).astype(int)
    out["near_unbiased"] = (out["ratio_error"].abs() <= epsilon).astype(int)
    out["overestimation_magnitude"] = out["ratio_error"].clip(lower=0.0)
    out["underestimation_magnitude"] = (-out["ratio_error"]).clip(lower=0.0)
    out["sampled_to_actual_ratio"] = out["sampled_defect_ratio"] / out[
        "actual_defect_ratio"
    ].clip(lower=epsilon)
    return out


def summarize(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_cols, observed=False)
        .agg(
            rows=("row_index", "count"),
            wafers=("row_index", "nunique"),
            mean_valid_die_count=("valid_die_count", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
            mean_sampled_defect_ratio=("sampled_defect_ratio", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            median_ratio_error=("ratio_error", "median"),
            mean_absolute_error=("absolute_error", "mean"),
            overestimation_rate=("overestimated", "mean"),
            underestimation_rate=("underestimated", "mean"),
            near_unbiased_rate=("near_unbiased", "mean"),
            mean_overestimation_magnitude=("overestimation_magnitude", "mean"),
            mean_underestimation_magnitude=("underestimation_magnitude", "mean"),
            mean_sampled_to_actual_ratio=("sampled_to_actual_ratio", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            hit_rate=("hit", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
        )
        .reset_index()
    )


def paired_strategy_delta(data: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["row_index", "failureType", "first_pass_type", "cost_weight"]
    metric_cols = [
        "sampled_defect_ratio",
        "ratio_error",
        "absolute_error",
        "defect_coverage",
        "severe_miss",
        "underestimated",
        "overestimated",
    ]
    pivot = data.pivot_table(
        index=key_cols,
        columns="strategy",
        values=metric_cols,
        aggfunc="first",
    )
    records: list[dict[str, object]] = []
    comparisons = [
        ("ml_rank16", "coverage16"),
        ("ml_rank32", "coverage32"),
        ("ml_diverse16", "coverage16"),
        ("ml_diverse32", "coverage32"),
        ("ml_biasaware16", "coverage16"),
        ("ml_biasaware32", "coverage32"),
        ("ml_diverse16", "ml_rank16"),
        ("ml_diverse32", "ml_rank32"),
        ("ml_biasaware16", "ml_rank16"),
        ("ml_biasaware32", "ml_rank32"),
        ("ml_biasaware16", "ml_diverse16"),
        ("ml_biasaware32", "ml_diverse32"),
    ]
    for candidate, baseline in comparisons:
        candidate_df = data[data["strategy"] == candidate][
            key_cols + metric_cols
        ].copy()
        baseline_df = data[data["strategy"] == baseline][
            key_cols + metric_cols
        ].copy()
        if candidate_df.empty or baseline_df.empty:
            continue
        merged = candidate_df.merge(
            baseline_df,
            on=key_cols,
            how="inner",
            suffixes=("_candidate", "_baseline"),
        )
        if merged.empty:
            continue
        frame = merged[key_cols].copy()
        frame["candidate_strategy"] = candidate
        frame["baseline_strategy"] = baseline
        for metric in metric_cols:
            frame[f"delta_{metric}"] = (
                merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
            )
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    data = add_bias_flags(raw, args.epsilon)

    strategy_summary = summarize(data, ["cost_weight", "first_pass_type", "strategy"])
    pattern_summary = summarize(
        data, ["cost_weight", "failureType", "first_pass_type", "strategy"]
    )
    deltas = paired_strategy_delta(data)
    delta_summary = (
        deltas.groupby(
            ["cost_weight", "first_pass_type", "candidate_strategy", "baseline_strategy"],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_delta_sampled_defect_ratio=("delta_sampled_defect_ratio", "mean"),
            mean_delta_ratio_error=("delta_ratio_error", "mean"),
            mean_delta_absolute_error=("delta_absolute_error", "mean"),
            mean_delta_defect_coverage=("delta_defect_coverage", "mean"),
            mean_delta_severe_miss=("delta_severe_miss", "mean"),
            mean_delta_underestimated=("delta_underestimated", "mean"),
            mean_delta_overestimated=("delta_overestimated", "mean"),
        )
        .reset_index()
        if not deltas.empty
        else pd.DataFrame()
    )

    data.to_csv(args.out_dir / "point_ranking_bias_rows.csv", index=False)
    strategy_summary.to_csv(args.out_dir / "point_ranking_bias_strategy_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "point_ranking_bias_pattern_summary.csv", index=False)
    deltas.to_csv(args.out_dir / "point_ranking_bias_strategy_deltas.csv", index=False)
    delta_summary.to_csv(args.out_dir / "point_ranking_bias_delta_summary.csv", index=False)

    print(f"wrote bias diagnostics to {args.out_dir}")
    print(f"rows: {len(data):,}")


if __name__ == "__main__":
    main()
