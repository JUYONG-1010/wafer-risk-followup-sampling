from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SPATIAL = Path("data") / "processed" / "spatial_risk_v2" / "spatial_action_metrics.csv"
DEFAULT_ACTION_OUTCOMES = (
    Path("data") / "processed" / "policy_learning_v2" / "action_outcomes.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "cost_aware_spatial_policy_v2"

ACTION_ORDER = [
    "none",
    "coverage16",
    "coverage32",
    "edge16",
    "radial16",
    "radial32",
    "local_expand",
    "edge16_local",
    "radial32_local",
]
DEFAULT_COST_WEIGHTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine spatial risk metrics with added sampling cost to evaluate "
            "cost-aware fixed follow-up actions."
        )
    )
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--action-outcomes", type=Path, default=DEFAULT_ACTION_OUTCOMES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cost-weights",
        type=float,
        nargs="+",
        default=DEFAULT_COST_WEIGHTS,
        help="Penalty per added valid die for spatial cost score.",
    )
    return parser.parse_args()


def load_joined(spatial_path: Path, action_outcomes_path: Path) -> pd.DataFrame:
    spatial_cols = [
        "row_index",
        "failureType",
        "action",
        "radial_weighted_abs_error",
        "quadrant_weighted_abs_error",
        "defect_centroid_error_norm",
        "spatial_error_score",
        "radial_center_severe_miss",
        "radial_mid_severe_miss",
        "radial_edge_severe_miss",
    ]
    action_cols = [
        "row_index",
        "action",
        "added_valid_count",
        "sampled_valid_count",
        "absolute_error",
        "severe_miss",
        "underestimated",
        "hit",
    ]
    spatial = pd.read_csv(spatial_path, usecols=spatial_cols)
    outcomes = pd.read_csv(action_outcomes_path, usecols=action_cols)
    joined = spatial.merge(outcomes, on=["row_index", "action"], how="inner")
    if len(joined) != len(spatial):
        raise ValueError(
            f"Join lost rows: spatial={len(spatial):,}, joined={len(joined):,}"
        )
    return joined


def score_for_cost(data: pd.DataFrame, cost_weight: float) -> pd.DataFrame:
    scored = data.copy()
    scored["cost_weight"] = cost_weight
    scored["spatial_cost_score"] = (
        scored["spatial_error_score"]
        + cost_weight * scored["added_valid_count"].astype(float)
    )
    first_pass = (
        scored[scored["action"] == "none"][
            ["row_index", "spatial_error_score", "severe_miss", "absolute_error"]
        ]
        .rename(
            columns={
                "spatial_error_score": "first_pass_spatial_error_score",
                "severe_miss": "first_pass_severe_miss",
                "absolute_error": "first_pass_absolute_error",
            }
        )
        .copy()
    )
    scored = scored.merge(first_pass, on="row_index", how="left")
    scored["spatial_error_reduction"] = (
        scored["first_pass_spatial_error_score"] - scored["spatial_error_score"]
    )
    scored["risk_reduction_per_added_die"] = scored["spatial_error_reduction"] / scored[
        "added_valid_count"
    ].clip(lower=1)
    return scored


def select_oracle_best(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.copy()
    data["action"] = pd.Categorical(data["action"], ACTION_ORDER, ordered=True)
    ordered = data.sort_values(
        ["row_index", "spatial_cost_score", "added_valid_count", "action"]
    )
    return ordered.groupby("row_index", observed=False).head(1).copy()


def summarize_fixed(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["cost_weight", "action"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_spatial_error_score=("spatial_error_score", "mean"),
            mean_spatial_cost_score=("spatial_cost_score", "mean"),
            mean_spatial_error_reduction=("spatial_error_reduction", "mean"),
            mean_risk_reduction_per_added_die=(
                "risk_reduction_per_added_die",
                "mean",
            ),
            severe_miss_rate=("severe_miss", "mean"),
            radial_edge_severe_miss_rate=("radial_edge_severe_miss", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .reset_index()
    )


def summarize_by_pattern(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["cost_weight", "failureType", "action"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_spatial_error_score=("spatial_error_score", "mean"),
            mean_spatial_cost_score=("spatial_cost_score", "mean"),
            mean_spatial_error_reduction=("spatial_error_reduction", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            radial_edge_severe_miss_rate=("radial_edge_severe_miss", "mean"),
        )
        .reset_index()
    )


def summarize_oracle(best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle_summary = (
        best.groupby("cost_weight", observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_spatial_error_score=("spatial_error_score", "mean"),
            mean_spatial_cost_score=("spatial_cost_score", "mean"),
            mean_spatial_error_reduction=("spatial_error_reduction", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            radial_edge_severe_miss_rate=("radial_edge_severe_miss", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .reset_index()
    )

    action_counts = (
        best.groupby(["cost_weight", "action"], observed=False)
        .agg(wafers=("row_index", "count"))
        .reset_index()
    )
    totals = action_counts.groupby("cost_weight", observed=False)["wafers"].transform("sum")
    action_counts["fraction"] = action_counts["wafers"] / totals
    return oracle_summary, action_counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    joined = load_joined(args.spatial, args.action_outcomes)

    scored_frames: list[pd.DataFrame] = []
    best_frames: list[pd.DataFrame] = []
    fixed_frames: list[pd.DataFrame] = []
    pattern_frames: list[pd.DataFrame] = []

    for cost_weight in args.cost_weights:
        scored = score_for_cost(joined, cost_weight)
        best = select_oracle_best(scored)
        scored_frames.append(scored)
        best_frames.append(best)
        fixed_frames.append(summarize_fixed(scored))
        pattern_frames.append(summarize_by_pattern(scored))

    scored_all = pd.concat(scored_frames, ignore_index=True)
    best_all = pd.concat(best_frames, ignore_index=True)
    fixed_summary = pd.concat(fixed_frames, ignore_index=True)
    pattern_summary = pd.concat(pattern_frames, ignore_index=True)
    oracle_summary, oracle_counts = summarize_oracle(best_all)

    scored_all.to_csv(args.out_dir / "cost_aware_spatial_action_scores.csv", index=False)
    best_all.to_csv(args.out_dir / "cost_aware_spatial_oracle_best.csv", index=False)
    fixed_summary.to_csv(args.out_dir / "cost_aware_spatial_fixed_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "cost_aware_spatial_pattern_summary.csv", index=False)
    oracle_summary.to_csv(args.out_dir / "cost_aware_spatial_oracle_summary.csv", index=False)
    oracle_counts.to_csv(args.out_dir / "cost_aware_spatial_oracle_counts.csv", index=False)

    print(f"wrote cost-aware spatial policy data to {args.out_dir}")
    print("cost weights:", ", ".join(str(v) for v in args.cost_weights))


if __name__ == "__main__":
    main()
