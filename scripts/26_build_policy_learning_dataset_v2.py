from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import (
    defect_mask,
    expand_points,
    make_9point_mask,
    make_coverage_sampling_mask,
    make_edge_biased_sampling_mask,
    make_radial_sampling_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "policy_learning_v2"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build policy-learning v2 dataset with deterministic coverage "
            "follow-up actions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--severe-miss-weight", type=float, default=10.0)
    parser.add_argument("--absolute-error-weight", type=float, default=1.0)
    parser.add_argument("--underestimation-weight", type=float, default=0.25)
    parser.add_argument("--added-cost-weight", type=float, default=0.003)
    return parser.parse_args()


def normalized_radii_for_mask(wafer_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(valid)
    target_ys, target_xs = np.nonzero(mask)
    if len(xs) == 0 or len(target_xs) == 0:
        return np.array([], dtype=float)

    cy, cx = wafer_center(valid)
    max_radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    if max_radius == 0:
        return np.zeros(len(target_xs), dtype=float)

    return np.sqrt((target_ys - cy) ** 2 + (target_xs - cx) ** 2) / max_radius


def first_pass_features(row_index: int, failure_type: str, wafer_map: np.ndarray) -> dict[str, object]:
    first = make_9point_mask(wafer_map)
    metrics = sampling_metrics(wafer_map, first)
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    hit_mask = first & defects
    hit_radii = normalized_radii_for_mask(wafer_map, hit_mask)
    first_radii = normalized_radii_for_mask(wafer_map, first & valid)

    hit_count = int(hit_mask.sum())
    edge_hit_count = int((hit_radii >= 0.72).sum()) if len(hit_radii) else 0
    center_hit_count = int((hit_radii <= 0.35).sum()) if len(hit_radii) else 0
    mid_hit_count = int(hit_count - edge_hit_count - center_hit_count)

    return {
        "row_index": row_index,
        "failureType": failure_type,
        "map_height": int(np.asarray(wafer_map).shape[0]),
        "map_width": int(np.asarray(wafer_map).shape[1]),
        "valid_die_count": int(metrics["valid_die_count"]),
        "first_sampled_valid_count": int(metrics["sampled_valid_count"]),
        "first_sampling_density": float(metrics["sampling_density"]),
        "first_sampled_defects": int(metrics["sampled_defects"]),
        "first_sampled_defect_ratio": float(metrics["sampled_defect_ratio"]),
        "first_no_hit": int(metrics["sampled_defects"] == 0),
        "first_hit_count": hit_count,
        "first_edge_hit_count": edge_hit_count,
        "first_center_hit_count": center_hit_count,
        "first_mid_hit_count": mid_hit_count,
        "first_has_edge_hit": int(edge_hit_count > 0),
        "first_has_center_hit": int(center_hit_count > 0),
        "first_hit_radius_mean": float(hit_radii.mean()) if len(hit_radii) else 0.0,
        "first_hit_radius_max": float(hit_radii.max()) if len(hit_radii) else 0.0,
        "first_hit_radius_min": float(hit_radii.min()) if len(hit_radii) else 0.0,
        "first_sample_radius_mean": float(first_radii.mean()) if len(first_radii) else 0.0,
        "first_sample_radius_max": float(first_radii.max()) if len(first_radii) else 0.0,
    }


def local_expand_mask(wafer_map: np.ndarray, first: np.ndarray) -> np.ndarray:
    hit_points = list(zip(*np.nonzero(first & defect_mask(wafer_map))))
    if not hit_points:
        return np.zeros_like(first, dtype=bool)
    return expand_points(valid_die_mask(wafer_map), hit_points, radius=2)


def action_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    first = make_9point_mask(wafer_map)
    local = local_expand_mask(wafer_map, first)
    coverage16 = make_coverage_sampling_mask(
        wafer_map, n_points=16, existing_mask=first
    )
    coverage32 = make_coverage_sampling_mask(
        wafer_map, n_points=32, existing_mask=first
    )
    edge16 = make_edge_biased_sampling_mask(
        wafer_map, edge_points=16, inner_points=0
    )
    radial16 = make_radial_sampling_mask(
        wafer_map, rings=(0.0, 0.95), angles=16
    )
    radial32 = make_radial_sampling_mask(
        wafer_map, rings=(0.0, 0.5, 0.95), angles=16
    )

    return {
        "none": first,
        "coverage16": first | coverage16,
        "coverage32": first | coverage32,
        "edge16": first | edge16,
        "radial16": first | radial16,
        "radial32": first | radial32,
        "local_expand": first | local,
        "edge16_local": first | edge16 | local,
        "radial32_local": first | radial32 | local,
    }


def score_action(
    metrics: dict[str, float | int],
    added_valid_count: int,
    severe_miss_weight: float,
    absolute_error_weight: float,
    underestimation_weight: float,
    added_cost_weight: float,
) -> float:
    return float(
        severe_miss_weight * int(metrics["severe_miss"])
        + absolute_error_weight * float(metrics["absolute_error"])
        + underestimation_weight * int(metrics["underestimated"])
        + added_cost_weight * added_valid_count
    )


def evaluate_actions(
    row_index: int,
    failure_type: str,
    wafer_map: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    first = make_9point_mask(wafer_map)
    first_metrics = sampling_metrics(wafer_map, first)
    first_count = int(first_metrics["sampled_valid_count"])

    records: list[dict[str, object]] = []
    for action, mask in action_masks(wafer_map).items():
        metrics = sampling_metrics(wafer_map, mask)
        sampled_count = int(metrics["sampled_valid_count"])
        added_valid_count = sampled_count - first_count
        action_score = score_action(
            metrics,
            added_valid_count=added_valid_count,
            severe_miss_weight=args.severe_miss_weight,
            absolute_error_weight=args.absolute_error_weight,
            underestimation_weight=args.underestimation_weight,
            added_cost_weight=args.added_cost_weight,
        )
        records.append(
            {
                "row_index": row_index,
                "failureType": failure_type,
                "action": action,
                "score": action_score,
                "added_valid_count": added_valid_count,
                **metrics,
            }
        )
    return records


def select_best_actions(action_outcomes: pd.DataFrame) -> pd.DataFrame:
    action_outcomes = action_outcomes.copy()
    action_outcomes["action"] = pd.Categorical(
        action_outcomes["action"], categories=ACTION_ORDER, ordered=True
    )
    ordered = action_outcomes.sort_values(
        ["row_index", "score", "added_valid_count", "action"]
    )
    best = ordered.groupby("row_index", observed=False).head(1).copy()
    return best[
        [
            "row_index",
            "action",
            "score",
            "added_valid_count",
            "sampled_valid_count",
            "absolute_error",
            "severe_miss",
            "underestimated",
        ]
    ].rename(
        columns={
            "action": "best_action",
            "score": "best_action_score",
            "added_valid_count": "best_action_added_valid_count",
            "sampled_valid_count": "best_action_sampled_valid_count",
            "absolute_error": "best_action_absolute_error",
            "severe_miss": "best_action_severe_miss",
            "underestimated": "best_action_underestimated",
        }
    )


def summarize_actions(action_outcomes: pd.DataFrame) -> pd.DataFrame:
    return (
        action_outcomes.groupby(["failureType", "action"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_score=("score", "mean"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            hit_rate=("hit", "mean"),
        )
        .reset_index()
    )


def summarize_best_actions(policy_dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        policy_dataset.groupby(["failureType", "best_action"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_first_sampling_density=("first_sampling_density", "mean"),
            mean_first_sampled_defects=("first_sampled_defects", "mean"),
            mean_best_added_valid_count=("best_action_added_valid_count", "mean"),
            mean_best_score=("best_action_score", "mean"),
        )
        .reset_index()
        .query("wafers > 0")
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    feature_records: list[dict[str, object]] = []
    outcome_records: list[dict[str, object]] = []

    total = len(df)
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType

        row_index = int(row.Index)
        wafer_map = np.asarray(row.waferMap)

        feature_records.append(
            first_pass_features(row_index, failure_type, wafer_map)
        )
        outcome_records.extend(evaluate_actions(row_index, failure_type, wafer_map, args))

        if pos % 10000 == 0 or pos == total:
            print(f"policy v2 rows processed: {pos:,}/{total:,}")

    features = pd.DataFrame.from_records(feature_records)
    action_outcomes = pd.DataFrame.from_records(outcome_records)
    best_actions = select_best_actions(action_outcomes)
    policy_dataset = features.merge(
        best_actions, on="row_index", how="left", validate="one_to_one"
    )

    action_summary = summarize_actions(action_outcomes)
    best_action_summary = summarize_best_actions(policy_dataset)

    features.to_csv(args.out_dir / "first_pass_features.csv", index=False)
    action_outcomes.to_csv(args.out_dir / "action_outcomes.csv", index=False)
    policy_dataset.to_csv(args.out_dir / "policy_learning_dataset.csv", index=False)
    action_summary.to_csv(args.out_dir / "action_outcome_summary.csv", index=False)
    best_action_summary.to_csv(args.out_dir / "best_action_summary.csv", index=False)

    print(f"wrote first-pass features: {args.out_dir / 'first_pass_features.csv'}")
    print(f"wrote action outcomes: {args.out_dir / 'action_outcomes.csv'}")
    print(f"wrote policy dataset: {args.out_dir / 'policy_learning_dataset.csv'}")
    print(f"wrote action summary: {args.out_dir / 'action_outcome_summary.csv'}")
    print(f"wrote best-action summary: {args.out_dir / 'best_action_summary.csv'}")


if __name__ == "__main__":
    main()
