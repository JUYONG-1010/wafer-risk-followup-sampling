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
    make_edge_biased_sampling_mask,
    make_radial_sampling_mask,
    make_random_sampling_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "adaptive_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate risk-aware adaptive follow-up sampling strategies from a "
            "9-point first pass."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Process only first N patterned rows for quick iteration.",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=0.01,
        help=(
            "If first-stage sampling density is below this and no defect was "
            "observed, adaptive_v2 adds a geometry coverage guard."
        ),
    )
    return parser.parse_args()


def normalized_hit_radii(wafer_map: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(valid)
    hit_ys, hit_xs = np.nonzero(hit_mask)
    if len(hit_xs) == 0 or len(xs) == 0:
        return np.array([], dtype=float)

    cy, cx = wafer_center(valid)
    max_radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    if max_radius == 0:
        return np.zeros(len(hit_xs), dtype=float)

    return np.sqrt((hit_ys - cy) ** 2 + (hit_xs - cx) ** 2) / max_radius


def make_grid9_plus_random16(wafer_map: np.ndarray, seed: int) -> np.ndarray:
    first = make_9point_mask(wafer_map)
    random_followup = make_random_sampling_mask(wafer_map, n_points=16, seed=seed)
    return first | random_followup


def make_grid9_plus_edge16(wafer_map: np.ndarray) -> np.ndarray:
    first = make_9point_mask(wafer_map)
    edge_followup = make_edge_biased_sampling_mask(
        wafer_map, edge_points=16, inner_points=0
    )
    return first | edge_followup


def make_grid9_plus_radial32(wafer_map: np.ndarray) -> np.ndarray:
    first = make_9point_mask(wafer_map)
    radial_followup = make_radial_sampling_mask(
        wafer_map, rings=(0.0, 0.5, 0.95), angles=16
    )
    return first | radial_followup


def make_adaptive_v2_local_only(wafer_map: np.ndarray) -> np.ndarray:
    wafer = np.asarray(wafer_map)
    first = make_9point_mask(wafer)
    hit_points = list(zip(*np.nonzero(defect_mask(wafer) & first)))
    if not hit_points:
        return first
    local_followup = expand_points(valid_die_mask(wafer), hit_points, radius=2)
    return first | local_followup


def make_adaptive_v2_density_aware(
    wafer_map: np.ndarray,
    density_threshold: float,
) -> np.ndarray:
    """Risk-aware follow-up that uses only first-pass observations and geometry.

    Rules:
    - Start with 9-point grid sampling.
    - If first-pass defects are observed, add a local neighborhood around hits.
    - Edge hits trigger an edge ring follow-up.
    - Center/mid hits trigger a radial follow-up.
    - If no defect is observed but the first-pass density is very low, add a
      broad radial coverage guard. This is meant to reduce silent severe misses
      on large wafers without looking at the hidden dense defect map.
    """
    wafer = np.asarray(wafer_map)
    valid = valid_die_mask(wafer)
    first = make_9point_mask(wafer)
    first_metrics = sampling_metrics(wafer, first)
    hit_mask = defect_mask(wafer) & first
    hit_points = list(zip(*np.nonzero(hit_mask)))

    followup = np.zeros(valid.shape, dtype=bool)
    if hit_points:
        followup |= expand_points(valid, hit_points, radius=2)
        radii = normalized_hit_radii(wafer, hit_mask)
        if len(radii) and float(radii.max()) >= 0.72:
            followup |= make_edge_biased_sampling_mask(
                wafer, edge_points=16, inner_points=0
            )
        elif len(radii) and float(radii.min()) <= 0.35:
            followup |= make_radial_sampling_mask(
                wafer, rings=(0.25, 0.5, 0.75), angles=12
            )
        else:
            followup |= make_radial_sampling_mask(
                wafer, rings=(0.45, 0.7, 0.9), angles=12
            )
    elif first_metrics["sampling_density"] < density_threshold:
        followup |= make_radial_sampling_mask(
            wafer, rings=(0.35, 0.65, 0.95), angles=16
        )

    return first | followup


def add_stage_metrics(
    wafer_map: np.ndarray,
    final_mask: np.ndarray,
    density_threshold: float,
) -> dict[str, float | int]:
    first = make_9point_mask(wafer_map)
    first_metrics = sampling_metrics(wafer_map, first)
    final_metrics = sampling_metrics(wafer_map, final_mask)
    added_mask = final_mask & ~first
    added_valid_count = int((valid_die_mask(wafer_map) & added_mask).sum())
    added_defects = int((defect_mask(wafer_map) & added_mask).sum())
    first_density = float(first_metrics["sampling_density"])

    return {
        "first_stage_sampled_valid_count": int(first_metrics["sampled_valid_count"]),
        "first_stage_sampled_defects": int(first_metrics["sampled_defects"]),
        "first_stage_sampling_density": first_density,
        "first_stage_severe_miss": int(first_metrics["severe_miss"]),
        "added_valid_count": added_valid_count,
        "added_defects": added_defects,
        "low_density_guard_eligible": int(
            first_metrics["sampled_defects"] == 0
            and first_density < density_threshold
        ),
        "final_minus_first_sampled_valid_count": int(
            final_metrics["sampled_valid_count"]
            - first_metrics["sampled_valid_count"]
        ),
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["failureType", "scheme"])
        .agg(
            wafers=("row_index", "count"),
            mean_valid_die_count=("valid_die_count", "mean"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
            mean_sampled_defect_ratio=("sampled_defect_ratio", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            hit_rate=("hit", "mean"),
            first_stage_severe_miss_rate=("first_stage_severe_miss", "mean"),
            low_density_guard_eligible_rate=("low_density_guard_eligible", "mean"),
        )
        .reset_index()
    )


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["scheme"] == "grid9_first_pass"][
        [
            "failureType",
            "mean_sampled_valid_count",
            "mean_absolute_error",
            "severe_miss_rate",
            "underestimation_rate",
        ]
    ].rename(
        columns={
            "mean_sampled_valid_count": "baseline_sampled_valid_count",
            "mean_absolute_error": "baseline_absolute_error",
            "severe_miss_rate": "baseline_severe_miss_rate",
            "underestimation_rate": "baseline_underestimation_rate",
        }
    )
    merged = summary.merge(baseline, on="failureType", how="left")
    merged["added_count_vs_grid9"] = (
        merged["mean_sampled_valid_count"] - merged["baseline_sampled_valid_count"]
    )
    merged["absolute_error_reduction_vs_grid9"] = (
        merged["baseline_absolute_error"] - merged["mean_absolute_error"]
    )
    merged["severe_miss_reduction_vs_grid9"] = (
        merged["baseline_severe_miss_rate"] - merged["severe_miss_rate"]
    )
    merged["underestimation_reduction_vs_grid9"] = (
        merged["baseline_underestimation_rate"] - merged["underestimation_rate"]
    )
    merged["severe_miss_reduction_per_added_die"] = np.where(
        merged["added_count_vs_grid9"] > 0,
        merged["severe_miss_reduction_vs_grid9"] / merged["added_count_vs_grid9"],
        0.0,
    )
    return merged


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    schemes = [
        ("grid9_first_pass", lambda wafer, seed: make_9point_mask(wafer)),
        ("grid9_plus_random16", make_grid9_plus_random16),
        ("grid9_plus_edge16", lambda wafer, seed: make_grid9_plus_edge16(wafer)),
        ("grid9_plus_radial32", lambda wafer, seed: make_grid9_plus_radial32(wafer)),
        (
            "adaptive_v2_local_only",
            lambda wafer, seed: make_adaptive_v2_local_only(wafer),
        ),
        (
            "adaptive_v2_density_aware",
            lambda wafer, seed: make_adaptive_v2_density_aware(
                wafer, args.density_threshold
            ),
        ),
    ]

    records: list[dict[str, object]] = []
    total = len(df)
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        row_seed = int(row.Index) + 2026

        for scheme_name, maker in schemes:
            sample_mask = maker(row.waferMap, row_seed)
            metrics = sampling_metrics(row.waferMap, sample_mask)
            stage_metrics = add_stage_metrics(
                row.waferMap, sample_mask, args.density_threshold
            )
            records.append(
                {
                    "row_index": row.Index,
                    "failureType": failure_type,
                    "scheme": scheme_name,
                    "density_threshold": args.density_threshold,
                    **metrics,
                    **stage_metrics,
                }
            )

        if pos % 10000 == 0 or pos == total:
            print(f"adaptive v2 rows processed: {pos:,}/{total:,}")

    results = pd.DataFrame.from_records(records)
    results_path = args.out_dir / "adaptive_v2_results.csv"
    results.to_csv(results_path, index=False)
    print(f"wrote adaptive v2 results: {results_path}")

    summary = add_baseline_deltas(aggregate_results(results))
    summary_path = args.out_dir / "adaptive_v2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote adaptive v2 summary: {summary_path}")


if __name__ == "__main__":
    main()
