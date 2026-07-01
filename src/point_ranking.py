from __future__ import annotations

import numpy as np
import pandas as pd

from src.sampling import (
    defect_mask,
    make_9point_mask,
    make_25point_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


FIRST_PASS_TYPES = ("grid9", "grid25")

FEATURE_COLUMNS = [
    "first_pass_is_grid25",
    "map_height",
    "map_width",
    "valid_die_count",
    "first_sampled_valid_count",
    "first_sampling_density",
    "first_hit_count",
    "first_defect_ratio",
    "first_no_hit",
    "first_center_hit_count",
    "first_mid_hit_count",
    "first_edge_hit_count",
    "candidate_y_norm",
    "candidate_x_norm",
    "candidate_radius_norm",
    "candidate_angle_sin",
    "candidate_angle_cos",
    "candidate_is_center",
    "candidate_is_mid",
    "candidate_is_edge",
    "candidate_quadrant_q1",
    "candidate_quadrant_q2",
    "candidate_quadrant_q3",
    "candidate_quadrant_q4",
    "distance_to_nearest_first_sample_norm",
    "distance_to_nearest_first_hit_norm",
    "distance_to_first_hit_centroid_norm",
    "same_quadrant_as_any_hit",
    "same_radial_zone_as_any_hit",
    "angle_diff_to_nearest_hit",
]


def make_first_pass_mask(wafer_map: np.ndarray, first_pass_type: str) -> np.ndarray:
    if first_pass_type == "grid9":
        return make_9point_mask(wafer_map)
    if first_pass_type == "grid25":
        return make_25point_mask(wafer_map)
    raise ValueError(f"Unsupported first_pass_type: {first_pass_type}")


def normalized_geometry(wafer_map: np.ndarray) -> dict[str, object]:
    valid = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return {
            "valid": valid,
            "cy": wafer_map.shape[0] / 2.0,
            "cx": wafer_map.shape[1] / 2.0,
            "max_radius": 1.0,
            "y_min": 0.0,
            "y_max": max(wafer_map.shape[0] - 1, 1),
            "x_min": 0.0,
            "x_max": max(wafer_map.shape[1] - 1, 1),
        }
    cy, cx = wafer_center(valid)
    max_radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    return {
        "valid": valid,
        "cy": cy,
        "cx": cx,
        "max_radius": max(max_radius, 1.0),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
    }


def radial_zone(radius_norm: np.ndarray) -> np.ndarray:
    zone = np.full(radius_norm.shape, 1, dtype=int)
    zone[radius_norm <= 0.35] = 0
    zone[radius_norm > 0.72] = 2
    return zone


def quadrant_ids(y: np.ndarray, x: np.ndarray, cy: float, cx: float) -> np.ndarray:
    quadrant = np.zeros(y.shape, dtype=int)
    quadrant[(y < cy) & (x >= cx)] = 1
    quadrant[(y < cy) & (x < cx)] = 2
    quadrant[(y >= cy) & (x < cx)] = 3
    quadrant[(y >= cy) & (x >= cx)] = 4
    return quadrant


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def first_pass_summary(wafer_map: np.ndarray, first_mask: np.ndarray) -> dict[str, float | int]:
    metrics = sampling_metrics(wafer_map, first_mask)
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    geometry = normalized_geometry(wafer_map)
    cy = float(geometry["cy"])
    cx = float(geometry["cx"])
    max_radius = float(geometry["max_radius"])

    hit_y, hit_x = np.nonzero(first_mask & defects)
    hit_count = int(len(hit_x))
    if hit_count:
        hit_radius = np.sqrt((hit_y - cy) ** 2 + (hit_x - cx) ** 2) / max_radius
        hit_zones = radial_zone(hit_radius)
        center_hits = int((hit_zones == 0).sum())
        edge_hits = int((hit_zones == 2).sum())
        mid_hits = int(hit_count - center_hits - edge_hits)
    else:
        center_hits = 0
        mid_hits = 0
        edge_hits = 0

    return {
        "valid_die_count": int(valid.sum()),
        "first_sampled_valid_count": int(metrics["sampled_valid_count"]),
        "first_sampling_density": float(metrics["sampling_density"]),
        "first_hit_count": hit_count,
        "first_defect_ratio": float(metrics["sampled_defect_ratio"]),
        "first_no_hit": int(hit_count == 0),
        "first_center_hit_count": center_hits,
        "first_mid_hit_count": mid_hits,
        "first_edge_hit_count": edge_hits,
    }


def remaining_candidate_coords(
    wafer_map: np.ndarray, first_mask: np.ndarray
) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    return np.column_stack(np.nonzero(valid & ~first_mask))


def candidate_feature_frame(
    wafer_map: np.ndarray,
    first_pass_type: str,
    first_mask: np.ndarray | None = None,
    candidate_coords: np.ndarray | None = None,
    row_index: int | None = None,
    failure_type: str | None = None,
    include_label: bool = True,
) -> pd.DataFrame:
    wafer = np.asarray(wafer_map)
    if first_mask is None:
        first_mask = make_first_pass_mask(wafer, first_pass_type)
    else:
        first_mask = np.asarray(first_mask, dtype=bool) & valid_die_mask(wafer)
    if candidate_coords is None:
        candidate_coords = remaining_candidate_coords(wafer, first_mask)
    if len(candidate_coords) == 0:
        return pd.DataFrame()

    geometry = normalized_geometry(wafer)
    valid = geometry["valid"]
    defects = defect_mask(wafer)
    cy = float(geometry["cy"])
    cx = float(geometry["cx"])
    max_radius = float(geometry["max_radius"])
    y_min = float(geometry["y_min"])
    y_max = float(geometry["y_max"])
    x_min = float(geometry["x_min"])
    x_max = float(geometry["x_max"])

    cand = np.asarray(candidate_coords, dtype=float)
    cand_y = cand[:, 0]
    cand_x = cand[:, 1]
    cand_dy = cand_y - cy
    cand_dx = cand_x - cx
    cand_radius = np.sqrt(cand_dy**2 + cand_dx**2) / max_radius
    cand_angle = np.arctan2(cand_dy, cand_dx)
    cand_zone = radial_zone(cand_radius)
    cand_quadrant = quadrant_ids(cand_y, cand_x, cy, cx)

    first_y, first_x = np.nonzero(first_mask & valid)
    first_coords = np.column_stack([first_y, first_x]).astype(float)
    if len(first_coords):
        dist_first = np.sqrt(
            ((cand[:, None, :] - first_coords[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_first_sample = dist_first.min(axis=1) / max_radius
    else:
        nearest_first_sample = np.ones(len(cand), dtype=float)

    hit_y, hit_x = np.nonzero(first_mask & defects)
    hit_coords = np.column_stack([hit_y, hit_x]).astype(float)
    if len(hit_coords):
        dist_hit = np.sqrt(
            ((cand[:, None, :] - hit_coords[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_hit_idx = dist_hit.argmin(axis=1)
        nearest_hit_distance = dist_hit.min(axis=1) / max_radius
        hit_centroid = hit_coords.mean(axis=0)
        distance_to_hit_centroid = (
            np.sqrt(((cand - hit_centroid[None, :]) ** 2).sum(axis=1)) / max_radius
        )

        hit_dy = hit_coords[:, 0] - cy
        hit_dx = hit_coords[:, 1] - cx
        hit_radius = np.sqrt(hit_dy**2 + hit_dx**2) / max_radius
        hit_angle = np.arctan2(hit_dy, hit_dx)
        hit_zone = radial_zone(hit_radius)
        hit_quadrant = quadrant_ids(hit_coords[:, 0], hit_coords[:, 1], cy, cx)
        nearest_hit_angle_diff = angle_distance(cand_angle, hit_angle[nearest_hit_idx])
        same_quadrant = np.array(
            [int(q in set(hit_quadrant)) for q in cand_quadrant], dtype=int
        )
        same_zone = np.array([int(z in set(hit_zone)) for z in cand_zone], dtype=int)
    else:
        nearest_hit_distance = np.full(len(cand), 2.0, dtype=float)
        distance_to_hit_centroid = np.full(len(cand), 2.0, dtype=float)
        nearest_hit_angle_diff = np.full(len(cand), np.pi, dtype=float)
        same_quadrant = np.zeros(len(cand), dtype=int)
        same_zone = np.zeros(len(cand), dtype=int)

    y_span = max(y_max - y_min, 1.0)
    x_span = max(x_max - x_min, 1.0)
    first_summary = first_pass_summary(wafer, first_mask)

    data: dict[str, object] = {
        "row_index": row_index if row_index is not None else -1,
        "failureType": failure_type if failure_type is not None else "",
        "first_pass_type": first_pass_type,
        "candidate_y": candidate_coords[:, 0].astype(int),
        "candidate_x": candidate_coords[:, 1].astype(int),
        "first_pass_is_grid25": int(first_pass_type == "grid25"),
        "map_height": int(wafer.shape[0]),
        "map_width": int(wafer.shape[1]),
        **first_summary,
        "candidate_y_norm": (cand_y - y_min) / y_span,
        "candidate_x_norm": (cand_x - x_min) / x_span,
        "candidate_radius_norm": cand_radius,
        "candidate_angle_sin": np.sin(cand_angle),
        "candidate_angle_cos": np.cos(cand_angle),
        "candidate_is_center": (cand_zone == 0).astype(int),
        "candidate_is_mid": (cand_zone == 1).astype(int),
        "candidate_is_edge": (cand_zone == 2).astype(int),
        "candidate_quadrant_q1": (cand_quadrant == 1).astype(int),
        "candidate_quadrant_q2": (cand_quadrant == 2).astype(int),
        "candidate_quadrant_q3": (cand_quadrant == 3).astype(int),
        "candidate_quadrant_q4": (cand_quadrant == 4).astype(int),
        "distance_to_nearest_first_sample_norm": nearest_first_sample,
        "distance_to_nearest_first_hit_norm": nearest_hit_distance,
        "distance_to_first_hit_centroid_norm": distance_to_hit_centroid,
        "same_quadrant_as_any_hit": same_quadrant,
        "same_radial_zone_as_any_hit": same_zone,
        "angle_diff_to_nearest_hit": nearest_hit_angle_diff,
    }
    if include_label:
        yy = candidate_coords[:, 0].astype(int)
        xx = candidate_coords[:, 1].astype(int)
        data["label_candidate_is_defect"] = defects[yy, xx].astype(int)
    return pd.DataFrame(data)


def sample_training_candidates(
    wafer_map: np.ndarray,
    first_pass_type: str,
    max_defect_candidates: int,
    max_normal_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    wafer = np.asarray(wafer_map)
    first_mask = make_first_pass_mask(wafer, first_pass_type)
    remaining = remaining_candidate_coords(wafer, first_mask)
    if len(remaining) == 0:
        return remaining
    defects = defect_mask(wafer)
    yy = remaining[:, 0].astype(int)
    xx = remaining[:, 1].astype(int)
    is_defect = defects[yy, xx]
    defect_coords = remaining[is_defect]
    normal_coords = remaining[~is_defect]

    if len(defect_coords) > max_defect_candidates:
        defect_coords = defect_coords[
            rng.choice(len(defect_coords), size=max_defect_candidates, replace=False)
        ]
    if len(normal_coords) > max_normal_candidates:
        normal_coords = normal_coords[
            rng.choice(len(normal_coords), size=max_normal_candidates, replace=False)
        ]
    if len(defect_coords) == 0:
        return normal_coords
    if len(normal_coords) == 0:
        return defect_coords
    return np.vstack([defect_coords, normal_coords])
