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
    make_9point_mask,
    make_25point_mask,
    make_center_disk_sampling_mask,
    make_coverage_sampling_mask,
    make_edge_biased_sampling_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "first_pass_comparison_v1"
DEFAULT_COST_WEIGHTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]

STRATEGY_ORDER = [
    "grid9",
    "grid25",
    "center_disk_r5",
    "center_disk_r7",
    "grid9_coverage16",
    "grid9_coverage32",
    "grid25_coverage16",
    "grid25_edge16",
    "center_r5_edge16",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare grid9, grid25, center-disk first-pass, and follow-up variants."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cost-weights", type=float, nargs="+", default=DEFAULT_COST_WEIGHTS)
    return parser.parse_args()


def radial_zone_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    zones = {
        "center": np.zeros_like(valid, dtype=bool),
        "mid": np.zeros_like(valid, dtype=bool),
        "edge": np.zeros_like(valid, dtype=bool),
    }
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return zones
    cy, cx = wafer_center(valid)
    radius = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_radius = float(radius.max())
    if max_radius == 0:
        zones["center"] = valid.copy()
        return zones
    norm_radius = radius / max_radius
    zones["center"][ys, xs] = norm_radius <= 0.35
    zones["mid"][ys, xs] = (norm_radius > 0.35) & (norm_radius <= 0.72)
    zones["edge"][ys, xs] = norm_radius > 0.72
    return zones


def quadrant_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    yy, xx = np.indices(valid.shape)
    cy, cx = wafer_center(valid)
    return {
        "q1": valid & (yy < cy) & (xx >= cx),
        "q2": valid & (yy < cy) & (xx < cx),
        "q3": valid & (yy >= cy) & (xx < cx),
        "q4": valid & (yy >= cy) & (xx >= cx),
    }


def region_ratios(
    wafer_map: np.ndarray,
    sample_mask: np.ndarray,
    regions: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float | int]:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    valid_total = int(valid.sum())
    weighted_abs_error = 0.0
    max_abs_error = 0.0
    records: dict[str, float | int] = {}
    for name, region in regions.items():
        region_valid = valid & region
        region_sample = sample_mask & region_valid
        valid_count = int(region_valid.sum())
        sampled_count = int(region_sample.sum())
        defect_count = int((defects & region_valid).sum())
        sampled_defects = int((defects & region_sample).sum())
        actual_ratio = defect_count / valid_count if valid_count else 0.0
        sampled_ratio = sampled_defects / sampled_count if sampled_count else 0.0
        abs_error = abs(sampled_ratio - actual_ratio)
        weighted_abs_error += (valid_count / valid_total if valid_total else 0.0) * abs_error
        max_abs_error = max(max_abs_error, abs_error)
        records[f"{prefix}_{name}_severe_miss"] = int(
            defect_count > 0 and sampled_defects == 0
        )
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
        return {"defect_centroid_error_norm": 0.0, "sampled_defect_centroid_available": 0}
    if len(sampled_xs) == 0:
        return {"defect_centroid_error_norm": 1.0, "sampled_defect_centroid_available": 0}
    actual_y, actual_x = float(all_ys.mean()), float(all_xs.mean())
    sampled_y, sampled_x = float(sampled_ys.mean()), float(sampled_xs.mean())
    wafer_scale = float(
        np.sqrt((valid_ys.max() - valid_ys.min()) ** 2 + (valid_xs.max() - valid_xs.min()) ** 2)
    )
    error = np.sqrt((actual_y - sampled_y) ** 2 + (actual_x - sampled_x) ** 2)
    return {
        "defect_centroid_error_norm": float(error / wafer_scale) if wafer_scale else 0.0,
        "sampled_defect_centroid_available": 1,
    }


def spatial_metrics(wafer_map: np.ndarray, sample_mask: np.ndarray) -> dict[str, float | int]:
    record: dict[str, float | int] = {}
    record.update(region_ratios(wafer_map, sample_mask, radial_zone_masks(wafer_map), "radial"))
    record.update(region_ratios(wafer_map, sample_mask, quadrant_masks(wafer_map), "quadrant"))
    record.update(centroid_metrics(wafer_map, sample_mask))
    record["spatial_error_score"] = float(
        0.5 * record["radial_weighted_abs_error"]
        + 0.3 * record["quadrant_weighted_abs_error"]
        + 0.2 * record["defect_centroid_error_norm"]
    )
    return record


def strategy_masks(wafer_map: np.ndarray) -> dict[str, np.ndarray]:
    grid9 = make_9point_mask(wafer_map)
    grid25 = make_25point_mask(wafer_map)
    center5 = make_center_disk_sampling_mask(wafer_map, radius_cells=5)
    center7 = make_center_disk_sampling_mask(wafer_map, radius_cells=7)
    edge16 = make_edge_biased_sampling_mask(wafer_map, edge_points=16, inner_points=0)
    return {
        "grid9": grid9,
        "grid25": grid25,
        "center_disk_r5": center5,
        "center_disk_r7": center7,
        "grid9_coverage16": grid9 | make_coverage_sampling_mask(
            wafer_map, n_points=16, existing_mask=grid9
        ),
        "grid9_coverage32": grid9 | make_coverage_sampling_mask(
            wafer_map, n_points=32, existing_mask=grid9
        ),
        "grid25_coverage16": grid25 | make_coverage_sampling_mask(
            wafer_map, n_points=16, existing_mask=grid25
        ),
        "grid25_edge16": grid25 | edge16,
        "center_r5_edge16": center5 | edge16,
    }


def evaluate_wafer(row_index: int, failure_type: str, wafer_map: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for strategy, mask in strategy_masks(wafer_map).items():
        sample = sampling_metrics(wafer_map, mask)
        spatial = spatial_metrics(wafer_map, mask)
        records.append(
            {
                "row_index": row_index,
                "failureType": failure_type,
                "strategy": strategy,
                **sample,
                **spatial,
            }
        )
    return records


def add_cost_scores(data: pd.DataFrame, cost_weights: list[float]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cost_weight in cost_weights:
        scored = data.copy()
        scored["cost_weight"] = cost_weight
        scored["spatial_cost_score"] = (
            scored["spatial_error_score"]
            + cost_weight * scored["sampled_valid_count"].astype(float)
        )
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def summarize(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = (
        scored.groupby(["cost_weight", "strategy"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_spatial_error_score=("spatial_error_score", "mean"),
            mean_spatial_cost_score=("spatial_cost_score", "mean"),
            radial_edge_severe_miss_rate=("radial_edge_severe_miss", "mean"),
        )
        .reset_index()
    )
    pattern = (
        scored.groupby(["cost_weight", "failureType", "strategy"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_spatial_error_score=("spatial_error_score", "mean"),
            mean_spatial_cost_score=("spatial_cost_score", "mean"),
        )
        .reset_index()
    )
    return fixed, pattern


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
        records.extend(evaluate_wafer(int(row.Index), failure_type, np.asarray(row.waferMap)))
        if pos % 10000 == 0 or pos == total:
            print(f"first-pass comparison rows processed: {pos:,}/{total:,}")

    metrics = pd.DataFrame.from_records(records)
    scored = add_cost_scores(metrics, args.cost_weights)
    fixed, pattern = summarize(scored)

    metrics.to_csv(args.out_dir / "first_pass_strategy_metrics.csv", index=False)
    scored.to_csv(args.out_dir / "first_pass_strategy_cost_scores.csv", index=False)
    fixed.to_csv(args.out_dir / "first_pass_strategy_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "first_pass_strategy_pattern_summary.csv", index=False)

    print(f"wrote first-pass comparison data to {args.out_dir}")


if __name__ == "__main__":
    main()
