from __future__ import annotations

import numpy as np


def valid_die_mask(wafer_map: np.ndarray) -> np.ndarray:
    """Return true for cells that represent real die locations."""
    return np.asarray(wafer_map) > 0


def defect_mask(wafer_map: np.ndarray) -> np.ndarray:
    """Return true for defective die cells."""
    return np.asarray(wafer_map) == 2


def wafer_center(mask: np.ndarray) -> tuple[float, float]:
    """Estimate the wafer center from valid die coordinates."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    return (float(ys.mean()), float(xs.mean()))


def nearest_valid_cell(mask: np.ndarray, y: float, x: float) -> tuple[int, int] | None:
    """Return the valid die cell nearest to a target coordinate."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    distances = (ys - y) ** 2 + (xs - x) ** 2
    idx = int(np.argmin(distances))
    return int(ys[idx]), int(xs[idx])


def expand_points(
    mask: np.ndarray, points: list[tuple[int, int]], radius: int = 0
) -> np.ndarray:
    """Create a sampling mask by expanding point sites by a square radius."""
    sample = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y, x in points:
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        sample[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return sample


def make_grid_sampling_mask(
    wafer_map: np.ndarray, grid_size: int, radius: int = 0
) -> np.ndarray:
    """Create an N-by-N sparse sampling mask over valid wafer area."""
    return make_fractional_grid_sampling_mask(
        wafer_map,
        y_fractions=np.linspace(0.0, 1.0, grid_size),
        x_fractions=np.linspace(0.0, 1.0, grid_size),
        radius=radius,
    )


def make_fractional_grid_sampling_mask(
    wafer_map: np.ndarray,
    y_fractions: np.ndarray,
    x_fractions: np.ndarray,
    radius: int = 0,
) -> np.ndarray:
    """Create grid sampling sites from fractional valid-area coordinates."""
    points = fractional_grid_points(wafer_map, y_fractions, x_fractions)
    return expand_points(valid_die_mask(wafer_map), points, radius=radius)


def fractional_grid_points(
    wafer_map: np.ndarray,
    y_fractions: np.ndarray,
    x_fractions: np.ndarray,
) -> list[tuple[int, int]]:
    """Return valid die coordinates for fractional grid sampling sites."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []

    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    y_targets = y_min + np.asarray(y_fractions) * (y_max - y_min)
    x_targets = x_min + np.asarray(x_fractions) * (x_max - x_min)

    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y in y_targets:
        for x in x_targets:
            cell = nearest_valid_cell(mask, y, x)
            if cell is not None and cell not in seen:
                points.append(cell)
                seen.add(cell)
    return points


def make_interior_grid_sampling_mask(
    wafer_map: np.ndarray,
    grid_size: int,
    margin: float | None = None,
    radius: int = 0,
) -> np.ndarray:
    """Create an N-by-N grid that avoids the outer wafer bounding-box edge.

    The margin is specified in normalized valid-area coordinates. With
    ``grid_size=3`` and ``margin=0.2``, the target fractions are
    ``[0.2, 0.5, 0.8]`` instead of ``[0.0, 0.5, 1.0]``.
    """
    if grid_size < 2:
        fractions = np.array([0.5])
    else:
        if margin is None:
            margin = 0.2 if grid_size <= 3 else 0.1
        if not 0.0 <= margin < 0.5:
            raise ValueError("margin must satisfy 0 <= margin < 0.5")
        fractions = np.linspace(margin, 1.0 - margin, grid_size)
    return make_fractional_grid_sampling_mask(
        wafer_map, y_fractions=fractions, x_fractions=fractions, radius=radius
    )


def make_5point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create center plus cardinal-direction sparse sampling sites."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    cy, cx = wafer_center(mask)
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    targets = [
        (cy, cx),
        (y_min, cx),
        (y_max, cx),
        (cy, x_min),
        (cy, x_max),
    ]

    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y, x in targets:
        cell = nearest_valid_cell(mask, y, x)
        if cell is not None and cell not in seen:
            points.append(cell)
            seen.add(cell)
    return points


def make_9point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create a 3-by-3 sparse sampling mask."""
    return make_grid_sampling_mask(wafer_map, grid_size=3, radius=radius)


def make_9point_sites(wafer_map: np.ndarray) -> list[dict[str, int | str]]:
    """Return named 3-by-3 sampling sites used by the first-pass grid."""
    labels = [
        "top_left",
        "top_center",
        "top_right",
        "mid_left",
        "center",
        "mid_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]
    fractions = np.array([0.0, 0.5, 1.0])
    points = fractional_grid_points(wafer_map, fractions, fractions)
    records: list[dict[str, int | str]] = []
    for label, point in zip(labels, points, strict=False):
        y, x = point
        records.append({"site": label, "y": int(y), "x": int(x)})
    return records


def make_25point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create a 5-by-5 sparse sampling mask."""
    return make_grid_sampling_mask(wafer_map, grid_size=5, radius=radius)


def make_center_disk_sampling_mask(
    wafer_map: np.ndarray,
    radius_cells: float = 5.0,
) -> np.ndarray:
    """Sample all valid die inside an absolute-radius disk around wafer center."""
    mask = valid_die_mask(wafer_map)
    sample = np.zeros_like(mask, dtype=bool)
    if radius_cells < 0 or not mask.any():
        return sample

    cy, cx = wafer_center(mask)
    yy, xx = np.indices(mask.shape)
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return mask & (distance <= radius_cells)


def make_interior_5point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create an edge-excluding 5-point-style interior sampling mask."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    cy, cx = wafer_center(mask)
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    margin = 0.2
    targets = [
        (cy, cx),
        (y_min + margin * (y_max - y_min), cx),
        (y_max - margin * (y_max - y_min), cx),
        (cy, x_min + margin * (x_max - x_min)),
        (cy, x_max - margin * (x_max - x_min)),
    ]

    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y, x in targets:
        cell = nearest_valid_cell(mask, y, x)
        if cell is not None and cell not in seen:
            points.append(cell)
            seen.add(cell)
    return expand_points(mask, points, radius=radius)


def make_interior_9point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create a 3-by-3 interior grid sampling mask."""
    return make_interior_grid_sampling_mask(
        wafer_map, grid_size=3, margin=0.2, radius=radius
    )


def make_interior_25point_mask(wafer_map: np.ndarray, radius: int = 0) -> np.ndarray:
    """Create a 5-by-5 interior grid sampling mask."""
    return make_interior_grid_sampling_mask(
        wafer_map, grid_size=5, margin=0.1, radius=radius
    )


def make_radial_sampling_mask(
    wafer_map: np.ndarray,
    rings: tuple[float, ...] = (0.0, 0.5, 0.85),
    angles: int = 8,
    radius: int = 0,
) -> np.ndarray:
    """Create center/ring/edge sampling sites in wafer polar coordinates."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    cy, cx = wafer_center(mask)
    max_radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    theta_values = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)

    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for ring in rings:
        if ring == 0.0:
            targets = [(cy, cx)]
        else:
            targets = [
                (
                    cy + ring * max_radius * np.sin(theta),
                    cx + ring * max_radius * np.cos(theta),
                )
                for theta in theta_values
            ]
        for y, x in targets:
            cell = nearest_valid_cell(mask, y, x)
            if cell is not None and cell not in seen:
                points.append(cell)
                seen.add(cell)
    return expand_points(mask, points, radius=radius)


def make_edge_biased_sampling_mask(
    wafer_map: np.ndarray, edge_points: int = 16, inner_points: int = 8, radius: int = 0
) -> np.ndarray:
    """Create a sampling mask that allocates more sites near the wafer edge."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    cy, cx = wafer_center(mask)
    max_radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    edge_angles = np.linspace(0.0, 2.0 * np.pi, edge_points, endpoint=False)
    inner_angles = np.linspace(0.0, 2.0 * np.pi, inner_points, endpoint=False)

    targets = [(cy, cx)]
    targets.extend(
        (
            cy + 0.92 * max_radius * np.sin(theta),
            cx + 0.92 * max_radius * np.cos(theta),
        )
        for theta in edge_angles
    )
    targets.extend(
        (
            cy + 0.55 * max_radius * np.sin(theta),
            cx + 0.55 * max_radius * np.cos(theta),
        )
        for theta in inner_angles
    )

    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y, x in targets:
        cell = nearest_valid_cell(mask, y, x)
        if cell is not None and cell not in seen:
            points.append(cell)
            seen.add(cell)
    return expand_points(mask, points, radius=radius)


def make_random_sampling_mask(
    wafer_map: np.ndarray, n_points: int = 25, radius: int = 0, seed: int = 42
) -> np.ndarray:
    """Create a reproducible random sparse sampling mask over valid die cells."""
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    rng = np.random.default_rng(seed)
    count = min(n_points, len(xs))
    chosen = rng.choice(len(xs), size=count, replace=False)
    points = [(int(ys[i]), int(xs[i])) for i in chosen]
    return expand_points(mask, points, radius=radius)


def make_coverage_sampling_mask(
    wafer_map: np.ndarray,
    n_points: int = 16,
    existing_mask: np.ndarray | None = None,
    radius: int = 0,
) -> np.ndarray:
    """Create deterministic space-filling follow-up sites.

    The algorithm selects valid die locations that are far from already sampled
    sites. It uses only wafer geometry and the existing sample mask, not defect
    labels or dense defect coordinates.
    """
    mask = valid_die_mask(wafer_map)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or n_points <= 0:
        return np.zeros_like(mask, dtype=bool)

    if existing_mask is None:
        existing = np.zeros_like(mask, dtype=bool)
    else:
        existing = np.asarray(existing_mask, dtype=bool) & mask

    selected = existing.copy()
    candidate_coords = np.column_stack(np.nonzero(mask & ~selected))
    if len(candidate_coords) == 0:
        return np.zeros_like(mask, dtype=bool)

    seed_coords = np.column_stack(np.nonzero(selected))
    if len(seed_coords) == 0:
        cy, cx = wafer_center(mask)
        first = nearest_valid_cell(mask, cy, cx)
        if first is not None:
            selected[first] = True
            seed_coords = np.array([[first[0], first[1]]], dtype=float)
            candidate_coords = np.column_stack(np.nonzero(mask & ~selected))

    if len(seed_coords) == 0 or len(candidate_coords) == 0:
        return np.zeros_like(mask, dtype=bool)

    candidate_coords = candidate_coords.astype(float)
    seed_coords = seed_coords.astype(float)
    min_dist_sq = np.min(
        ((candidate_coords[:, None, :] - seed_coords[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )

    points: list[tuple[int, int]] = []
    for _ in range(min(n_points, len(candidate_coords))):
        best_idx = int(np.argmax(min_dist_sq))
        y, x = candidate_coords[best_idx].astype(int)
        points.append((int(y), int(x)))

        new_dist_sq = ((candidate_coords - candidate_coords[best_idx]) ** 2).sum(axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)
        min_dist_sq[best_idx] = -1.0

    return expand_points(mask, points, radius=radius)


def make_hit_pattern_followup_mask(
    wafer_map: np.ndarray,
    n_points: int = 16,
    existing_mask: np.ndarray | None = None,
    radius: int = 0,
) -> np.ndarray:
    """Generate follow-up sites from first-pass 9-point hit geometry.

    The score uses only first-pass defect hits and wafer geometry. Dense defect
    coordinates are only touched at sampled first-pass sites, matching the
    information that would be available after a sparse first measurement.
    """
    wafer = np.asarray(wafer_map)
    valid = valid_die_mask(wafer)
    defects = defect_mask(wafer)
    if n_points <= 0 or not valid.any():
        return np.zeros_like(valid, dtype=bool)

    if existing_mask is None:
        first = make_9point_mask(wafer)
    else:
        first = np.asarray(existing_mask, dtype=bool) & valid

    hit_mask = first & defects
    hit_coords = np.column_stack(np.nonzero(hit_mask))
    if len(hit_coords) == 0:
        return make_coverage_sampling_mask(
            wafer, n_points=n_points, existing_mask=first, radius=radius
        )

    candidate_coords = np.column_stack(np.nonzero(valid & ~first))
    if len(candidate_coords) == 0:
        return np.zeros_like(valid, dtype=bool)

    cy, cx = wafer_center(valid)
    valid_ys, valid_xs = np.nonzero(valid)
    max_radius = float(np.sqrt(((valid_ys - cy) ** 2 + (valid_xs - cx) ** 2).max()))
    if max_radius == 0:
        return np.zeros_like(valid, dtype=bool)

    cand = candidate_coords.astype(float)
    hits = hit_coords.astype(float)
    cand_y = cand[:, 0]
    cand_x = cand[:, 1]
    cand_dy = cand_y - cy
    cand_dx = cand_x - cx
    cand_r = np.sqrt(cand_dy**2 + cand_dx**2) / max_radius
    cand_theta = np.arctan2(cand_dy, cand_dx)

    hit_dy = hits[:, 0] - cy
    hit_dx = hits[:, 1] - cx
    hit_r = np.sqrt(hit_dy**2 + hit_dx**2) / max_radius
    hit_theta = np.arctan2(hit_dy, hit_dx)

    distance_sq = ((cand[:, None, :] - hits[None, :, :]) ** 2).sum(axis=2)
    local_sigma = 0.13 * max_radius
    local_score = np.exp(-distance_sq / (2.0 * local_sigma**2)).max(axis=1)

    angle_diff = np.abs(
        np.angle(np.exp(1j * (cand_theta[:, None] - hit_theta[None, :])))
    )
    radial_score = np.exp(-(angle_diff**2) / (2.0 * 0.28**2)).max(axis=1)
    same_side = (cand_r[:, None] >= np.maximum(hit_r[None, :] - 0.08, 0.0)).max(axis=1)
    radial_score = radial_score * same_side

    edge_hit_count = int((hit_r >= 0.70).sum())
    center_hit_count = int((hit_r <= 0.35).sum())
    edge_score = (cand_r >= 0.72).astype(float) if edge_hit_count else np.zeros(len(cand))
    center_score = (cand_r <= 0.45).astype(float) if center_hit_count else np.zeros(len(cand))
    transition_score = (
        ((cand_r > 0.35) & (cand_r <= 0.72)).astype(float)
        if center_hit_count
        else np.zeros(len(cand))
    )

    quadrant_score = np.zeros(len(cand), dtype=float)
    for hy, hx in hits:
        same_y = np.sign(cand_y - cy) == np.sign(hy - cy)
        same_x = np.sign(cand_x - cx) == np.sign(hx - cx)
        quadrant_score = np.maximum(quadrant_score, (same_y & same_x).astype(float))

    line_score = np.zeros(len(cand), dtype=float)
    if len(hits) >= 2:
        centered_hits = hits - hits.mean(axis=0)
        _, _, vh = np.linalg.svd(centered_hits, full_matrices=False)
        direction = vh[0]
        rel = cand - hits.mean(axis=0)
        projection = rel @ direction
        closest = np.outer(projection, direction)
        line_distance = np.sqrt(((rel - closest) ** 2).sum(axis=1))
        line_sigma = 0.08 * max_radius
        line_score = np.exp(-(line_distance**2) / (2.0 * line_sigma**2))

    base_score = (
        2.4 * local_score
        + 1.2 * radial_score
        + 0.9 * edge_score
        + 0.5 * center_score
        + 0.6 * transition_score
        + 0.7 * quadrant_score
        + 1.0 * line_score
    )

    selected = first.copy()
    selected_coords = np.column_stack(np.nonzero(selected)).astype(float)
    if len(selected_coords) == 0:
        selected_coords = np.array([[cy, cx]], dtype=float)
    min_dist_sq = np.min(
        ((cand[:, None, :] - selected_coords[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )

    chosen: list[tuple[int, int]] = []
    available = np.ones(len(cand), dtype=bool)
    for _ in range(min(n_points, len(cand))):
        max_dist = float(min_dist_sq[available].max()) if available.any() else 1.0
        diversity_score = min_dist_sq / max(max_dist, 1.0)
        total_score = base_score + 0.35 * diversity_score
        total_score[~available] = -np.inf
        best_idx = int(np.argmax(total_score))
        y, x = candidate_coords[best_idx]
        chosen.append((int(y), int(x)))
        available[best_idx] = False
        new_dist_sq = ((cand - cand[best_idx]) ** 2).sum(axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

    return expand_points(valid, chosen, radius=radius)


def make_adaptive_sampling_mask(
    wafer_map: np.ndarray,
    first_stage_radius: int = 0,
    followup_radius: int = 2,
) -> np.ndarray:
    """Two-stage adaptive sampling from a 9-point first pass.

    The first stage uses 9-point grid sampling. If any defect die is observed
    in that first pass, the mask expands around those hit sites as a local
    follow-up measurement. This intentionally cannot rescue defects that the
    first-stage sampling never touches.
    """
    wafer = np.asarray(wafer_map)
    first_stage = make_9point_mask(wafer, radius=first_stage_radius)
    hit_points = list(zip(*np.nonzero((wafer == 2) & first_stage)))
    if not hit_points:
        return first_stage
    followup = expand_points(valid_die_mask(wafer), hit_points, radius=followup_radius)
    return first_stage | followup


def sampling_metrics(wafer_map: np.ndarray, sample_mask: np.ndarray) -> dict[str, float | int]:
    """Measure dense-to-sparse information loss for one sampling mask."""
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)

    valid_die_count = int(valid.sum())
    total_defects = int(defects.sum())
    sampled_valid_count = int((valid & sample_mask).sum())
    sampled_defects = int((defects & sample_mask).sum())

    actual_defect_ratio = total_defects / valid_die_count if valid_die_count else 0.0
    sampling_density = sampled_valid_count / valid_die_count if valid_die_count else 0.0
    sampled_defect_ratio = (
        sampled_defects / sampled_valid_count if sampled_valid_count else 0.0
    )
    ratio_error = sampled_defect_ratio - actual_defect_ratio
    absolute_error = abs(ratio_error)
    estimated_defect_count = sampled_defect_ratio * valid_die_count

    if total_defects == 0:
        coverage = 1.0
        relative_count_error = 0.0
    else:
        coverage = sampled_defects / total_defects
        relative_count_error = (estimated_defect_count - total_defects) / total_defects

    return {
        "valid_die_count": valid_die_count,
        "total_defects": total_defects,
        "sampled_valid_count": sampled_valid_count,
        "sampled_defects": sampled_defects,
        "sampling_density": float(sampling_density),
        "actual_defect_ratio": float(actual_defect_ratio),
        "sampled_defect_ratio": float(sampled_defect_ratio),
        "ratio_error": float(ratio_error),
        "absolute_error": float(absolute_error),
        "estimated_defect_count": float(estimated_defect_count),
        "relative_count_error": float(relative_count_error),
        "defect_coverage": float(coverage),
        "miss_rate": float(1.0 - coverage),
        "hit": int(sampled_defects > 0),
        "underestimated": int(sampled_defect_ratio < actual_defect_ratio),
        "severe_miss": int(total_defects > 0 and sampled_defects == 0),
    }


def sampling_coverage(wafer_map: np.ndarray, sample_mask: np.ndarray) -> dict[str, float | int]:
    """Backward-compatible alias for the expanded sampling metrics."""
    return sampling_metrics(wafer_map, sample_mask)
