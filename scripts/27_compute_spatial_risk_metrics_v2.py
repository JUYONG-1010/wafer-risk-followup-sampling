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
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "spatial_risk_v2"

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
        description="Compute spatial risk metrics for policy-learning v2 actions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


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


def radial_zone_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(valid)
    zones = {
        "center": np.zeros_like(valid, dtype=bool),
        "mid": np.zeros_like(valid, dtype=bool),
        "edge": np.zeros_like(valid, dtype=bool),
    }
    if len(xs) == 0:
        return zones

    cy, cx = wafer_center(valid)
    radius = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_radius = float(radius.max())
    if max_radius == 0:
        zones["center"] = valid.copy()
        return zones

    norm_radius = radius / max_radius
    coords = (ys, xs)
    zones["center"][coords] = norm_radius <= 0.35
    zones["mid"][coords] = (norm_radius > 0.35) & (norm_radius <= 0.72)
    zones["edge"][coords] = norm_radius > 0.72
    return zones


def quadrant_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(valid)
    quadrants = {name: np.zeros_like(valid, dtype=bool) for name in ["q1", "q2", "q3", "q4"]}
    if len(xs) == 0:
        return quadrants

    cy, cx = wafer_center(valid)
    quadrants["q1"] = valid & (np.indices(valid.shape)[0] < cy) & (np.indices(valid.shape)[1] >= cx)
    quadrants["q2"] = valid & (np.indices(valid.shape)[0] < cy) & (np.indices(valid.shape)[1] < cx)
    quadrants["q3"] = valid & (np.indices(valid.shape)[0] >= cy) & (np.indices(valid.shape)[1] < cx)
    quadrants["q4"] = valid & (np.indices(valid.shape)[0] >= cy) & (np.indices(valid.shape)[1] >= cx)
    return quadrants


def region_ratios(
    wafer_map: np.ndarray,
    sample_mask: np.ndarray,
    regions: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float | int]:
    defects = defect_mask(wafer_map)
    records: dict[str, float | int] = {}
    weighted_abs_error = 0.0
    max_abs_error = 0.0
    valid_total = int(valid_die_mask(wafer_map).sum())

    for name, region in regions.items():
        region_valid = valid_die_mask(wafer_map) & region
        region_sample = sample_mask & region_valid
        valid_count = int(region_valid.sum())
        sampled_count = int(region_sample.sum())
        defect_count = int((defects & region_valid).sum())
        sampled_defects = int((defects & region_sample).sum())

        actual_ratio = defect_count / valid_count if valid_count else 0.0
        sampled_ratio = sampled_defects / sampled_count if sampled_count else 0.0
        abs_error = abs(sampled_ratio - actual_ratio)
        weight = valid_count / valid_total if valid_total else 0.0
        weighted_abs_error += weight * abs_error
        max_abs_error = max(max_abs_error, abs_error)

        records[f"{prefix}_{name}_valid_count"] = valid_count
        records[f"{prefix}_{name}_sampled_count"] = sampled_count
        records[f"{prefix}_{name}_actual_ratio"] = float(actual_ratio)
        records[f"{prefix}_{name}_sampled_ratio"] = float(sampled_ratio)
        records[f"{prefix}_{name}_abs_error"] = float(abs_error)
        records[f"{prefix}_{name}_severe_miss"] = int(defect_count > 0 and sampled_defects == 0)

    records[f"{prefix}_weighted_abs_error"] = float(weighted_abs_error)
    records[f"{prefix}_max_abs_error"] = float(max_abs_error)
    return records


def centroid_metrics(wafer_map: np.ndarray, sample_mask: np.ndarray) -> dict[str, float | int]:
    defects = defect_mask(wafer_map)
    valid = valid_die_mask(wafer_map)
    all_ys, all_xs = np.nonzero(defects)
    sampled_ys, sampled_xs = np.nonzero(defects & sample_mask)
    valid_ys, valid_xs = np.nonzero(valid)
    if len(all_xs) == 0 or len(valid_xs) == 0:
        return {
            "defect_centroid_error_norm": 0.0,
            "sampled_defect_centroid_available": 0,
        }
    if len(sampled_xs) == 0:
        return {
            "defect_centroid_error_norm": 1.0,
            "sampled_defect_centroid_available": 0,
        }

    cy_actual = float(all_ys.mean())
    cx_actual = float(all_xs.mean())
    cy_sampled = float(sampled_ys.mean())
    cx_sampled = float(sampled_xs.mean())
    wafer_scale = float(
        np.sqrt((valid_ys.max() - valid_ys.min()) ** 2 + (valid_xs.max() - valid_xs.min()) ** 2)
    )
    error = np.sqrt((cy_actual - cy_sampled) ** 2 + (cx_actual - cx_sampled) ** 2)
    return {
        "defect_centroid_error_norm": float(error / wafer_scale) if wafer_scale else 0.0,
        "sampled_defect_centroid_available": 1,
    }


def compute_records(row_index: int, failure_type: str, wafer_map: np.ndarray) -> list[dict[str, object]]:
    radial_regions = radial_zone_masks(wafer_map)
    quadrants = quadrant_masks(wafer_map)
    records: list[dict[str, object]] = []
    for action, mask in action_masks(wafer_map).items():
        record: dict[str, object] = {
            "row_index": row_index,
            "failureType": failure_type,
            "action": action,
        }
        record.update(region_ratios(wafer_map, mask, radial_regions, "radial"))
        record.update(region_ratios(wafer_map, mask, quadrants, "quadrant"))
        record.update(centroid_metrics(wafer_map, mask))
        record["spatial_error_score"] = float(
            0.5 * record["radial_weighted_abs_error"]
            + 0.3 * record["quadrant_weighted_abs_error"]
            + 0.2 * record["defect_centroid_error_norm"]
        )
        records.append(record)
    return records


def summarize(spatial: pd.DataFrame) -> pd.DataFrame:
    return (
        spatial.groupby(["failureType", "action"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            radial_weighted_abs_error=("radial_weighted_abs_error", "mean"),
            radial_max_abs_error=("radial_max_abs_error", "mean"),
            quadrant_weighted_abs_error=("quadrant_weighted_abs_error", "mean"),
            quadrant_max_abs_error=("quadrant_max_abs_error", "mean"),
            centroid_error=("defect_centroid_error_norm", "mean"),
            centroid_available_rate=("sampled_defect_centroid_available", "mean"),
            spatial_error_score=("spatial_error_score", "mean"),
            radial_center_severe_miss=("radial_center_severe_miss", "mean"),
            radial_mid_severe_miss=("radial_mid_severe_miss", "mean"),
            radial_edge_severe_miss=("radial_edge_severe_miss", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    records: list[dict[str, object]] = []
    total = len(df)
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        records.extend(compute_records(int(row.Index), failure_type, np.asarray(row.waferMap)))
        if pos % 10000 == 0 or pos == total:
            print(f"spatial metrics rows processed: {pos:,}/{total:,}")

    spatial = pd.DataFrame.from_records(records)
    spatial_summary = summarize(spatial)
    spatial.to_csv(args.out_dir / "spatial_action_metrics.csv", index=False)
    spatial_summary.to_csv(args.out_dir / "spatial_action_summary.csv", index=False)
    print(f"wrote spatial metrics: {args.out_dir / 'spatial_action_metrics.csv'}")
    print(f"wrote spatial summary: {args.out_dir / 'spatial_action_summary.csv'}")


if __name__ == "__main__":
    main()
