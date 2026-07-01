from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_CNN_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("data") / "processed" / "low_evidence_gate_refinement_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "77_low_evidence_gate_refinement_v1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", Path("scripts") / "47_evaluate_density_followup_policy.py")
sparse_cnn = load_module("sparse_cnn68", Path("scripts") / "68_train_sparse_cnn_risk_map.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine low-evidence discovery gates before ensemble follow-up ranking."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=2500)
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--ensemble-noncnn-weight", type=float, default=0.3)
    parser.add_argument("--hit-thresholds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--discovery-counts", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument(
        "--discovery-geometries",
        nargs="+",
        default=["line", "coverage", "radial_line", "random_line"],
        choices=["line", "coverage", "radial_line", "random_line"],
    )
    return parser.parse_args()


def load_cnn_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden = int(checkpoint.get("args", {}).get("hidden_channels", 32))
    model = sparse_cnn.SparseRiskCNN(in_channels=7, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def tensor_from_observed_mask(
    wafer_map: np.ndarray,
    observed_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    observed = np.asarray(observed_mask, dtype=bool) & valid
    observed_defect = observed & defects
    observed_normal = observed & ~defects
    unknown_valid = valid & ~observed
    y_norm, x_norm, radius = sparse_cnn.coordinate_channels(wafer_map)
    channels = np.stack(
        [
            valid.astype(np.float32),
            observed_normal.astype(np.float32),
            observed_defect.astype(np.float32),
            unknown_valid.astype(np.float32),
            y_norm.astype(np.float32),
            x_norm.astype(np.float32),
            radius.astype(np.float32),
        ],
        axis=0,
    )
    return torch.from_numpy(channels[None, ...]).to(device)


def cnn_score_map_for_observed(
    model,
    wafer_map: np.ndarray,
    observed_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = tensor_from_observed_mask(wafer_map, observed_mask, device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    candidates = valid_die_mask(wafer_map) & ~np.asarray(observed_mask, dtype=bool)
    scores = scores.astype(float)
    scores[~candidates] = np.nan
    return scores


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= 1:
        return np.ones_like(scores)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores), dtype=float)
    return ranks / float(len(scores) - 1)


def topk_from_candidate_scores(
    wafer_map: np.ndarray,
    candidates: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    selected = np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    if candidates.empty or len(scores) == 0 or top_k <= 0:
        return selected
    finite = np.isfinite(scores)
    if not finite.any():
        return selected
    candidate_array = candidates[["candidate_y", "candidate_x"]].astype(int).to_numpy()
    candidate_array = candidate_array[finite]
    values = np.asarray(scores, dtype=float)[finite]
    chosen = candidate_array[np.argsort(values)[::-1][: min(int(top_k), len(values))]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def ensemble_candidate_scores(
    point_model,
    cnn_model,
    wafer_map: np.ndarray,
    observed_mask: np.ndarray,
    density: float,
    noncnn_weight: float,
    device: torch.device,
    row_index: int,
    failure_type: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    candidates = candidate_feature_frame(
        wafer_map,
        first_pass_type=density_policy.density_key(float(density)),
        first_mask=observed_mask,
        row_index=row_index,
        failure_type=failure_type,
        include_label=True,
    )
    if candidates.empty:
        empty = np.asarray([], dtype=float)
        return candidates, empty, empty, empty
    features = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    noncnn_scores = point_model.predict_proba(features)[:, 1].astype(float)
    cnn_map = cnn_score_map_for_observed(cnn_model, wafer_map, observed_mask, device)
    yy = candidates["candidate_y"].astype(int).to_numpy()
    xx = candidates["candidate_x"].astype(int).to_numpy()
    cnn_scores = cnn_map[yy, xx].astype(float)
    ensemble_scores = noncnn_weight * noncnn_scores + (1.0 - noncnn_weight) * cnn_scores
    return candidates, ensemble_scores, noncnn_scores, cnn_scores


def diverse_select_by_score(
    wafer_map: np.ndarray,
    existing_mask: np.ndarray,
    base_score: np.ndarray,
    n_points: int,
    diversity_weight: float = 1.0,
) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    selected = np.zeros_like(valid, dtype=bool)
    coords = np.column_stack(np.nonzero(valid & ~np.asarray(existing_mask, dtype=bool)))
    if n_points <= 0 or len(coords) == 0:
        return selected

    existing_coords = np.column_stack(np.nonzero(np.asarray(existing_mask, dtype=bool) & valid)).astype(float)
    if len(existing_coords) == 0:
        cy, cx = wafer_center(valid)
        existing_coords = np.asarray([[cy, cx]], dtype=float)

    cand = coords.astype(float)
    min_dist_sq = np.min(((cand[:, None, :] - existing_coords[None, :, :]) ** 2).sum(axis=2), axis=1)
    score = np.asarray(base_score, dtype=float).copy()
    score = score - np.nanmin(score)
    max_score = float(np.nanmax(score)) if np.isfinite(score).any() else 0.0
    if max_score > 0:
        score = score / max_score
    available = np.ones(len(coords), dtype=bool)
    for _ in range(min(int(n_points), len(coords))):
        max_dist = float(min_dist_sq[available].max()) if available.any() else 1.0
        diversity = min_dist_sq / max(max_dist, 1.0)
        combined = score + diversity_weight * diversity
        combined[~available] = -np.inf
        best = int(np.argmax(combined))
        y, x = coords[best]
        selected[int(y), int(x)] = True
        available[best] = False
        new_dist_sq = ((cand - cand[best]) ** 2).sum(axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)
    return selected


def line_scores(wafer_map: np.ndarray, existing_mask: np.ndarray, angles_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    coords = np.column_stack(np.nonzero(valid & ~np.asarray(existing_mask, dtype=bool)))
    if len(coords) == 0:
        return coords, np.asarray([], dtype=float)
    cy, cx = wafer_center(valid)
    dy = coords[:, 0].astype(float) - cy
    dx = coords[:, 1].astype(float) - cx
    scale = max(float(max(valid.shape)), 1.0)
    distances = []
    for angle in angles_deg:
        theta = np.deg2rad(float(angle))
        distances.append(np.abs(dx * np.sin(theta) - dy * np.cos(theta)) / scale)
    line_distance = np.min(np.vstack(distances), axis=0)
    scores = np.exp(-(line_distance**2) / (2.0 * 0.035**2))
    return coords, scores


def line_discovery_mask(wafer_map: np.ndarray, existing_mask: np.ndarray, n_points: int) -> np.ndarray:
    coords, scores = line_scores(wafer_map, existing_mask, [0.0, 45.0, 90.0, 135.0])
    if len(coords) == 0:
        return np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    return diverse_select_by_score(wafer_map, existing_mask, scores, n_points, diversity_weight=0.8)


def random_line_discovery_mask(
    wafer_map: np.ndarray,
    existing_mask: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 180.0, size=4).tolist()
    coords, scores = line_scores(wafer_map, existing_mask, angles)
    if len(coords) == 0:
        return np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    return diverse_select_by_score(wafer_map, existing_mask, scores, n_points, diversity_weight=0.8)


def radial_line_discovery_mask(wafer_map: np.ndarray, existing_mask: np.ndarray, n_points: int) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    coords = np.column_stack(np.nonzero(valid & ~np.asarray(existing_mask, dtype=bool)))
    if len(coords) == 0:
        return np.zeros_like(valid, dtype=bool)
    cy, cx = wafer_center(valid)
    yy = coords[:, 0].astype(float)
    xx = coords[:, 1].astype(float)
    dy = yy - cy
    dx = xx - cx
    valid_ys, valid_xs = np.nonzero(valid)
    max_radius = float(np.sqrt(((valid_ys - cy) ** 2 + (valid_xs - cx) ** 2).max()))
    max_radius = max(max_radius, 1.0)
    radius = np.sqrt(dy**2 + dx**2) / max_radius

    line_angles = [0.0, 45.0, 90.0, 135.0]
    scale = max(float(max(valid.shape)), 1.0)
    line_distances = []
    for angle in line_angles:
        theta = np.deg2rad(float(angle))
        line_distances.append(np.abs(dx * np.sin(theta) - dy * np.cos(theta)) / scale)
    line_score = np.exp(-(np.min(np.vstack(line_distances), axis=0) ** 2) / (2.0 * 0.045**2))

    rings = np.asarray([0.35, 0.60, 0.82], dtype=float)
    ring_distance = np.min(np.abs(radius[:, None] - rings[None, :]), axis=1)
    ring_score = np.exp(-(ring_distance**2) / (2.0 * 0.055**2))
    score = 0.65 * line_score + 0.35 * ring_score
    return diverse_select_by_score(wafer_map, existing_mask, score, n_points, diversity_weight=0.8)


def discovery_mask(
    geometry: str,
    wafer_map: np.ndarray,
    existing_mask: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    if geometry == "coverage":
        return make_coverage_sampling_mask(wafer_map, n_points=n_points, existing_mask=existing_mask)
    if geometry == "line":
        return line_discovery_mask(wafer_map, existing_mask, n_points)
    if geometry == "radial_line":
        return radial_line_discovery_mask(wafer_map, existing_mask, n_points)
    if geometry == "random_line":
        return random_line_discovery_mask(wafer_map, existing_mask, n_points, seed=seed)
    raise ValueError(f"Unknown discovery geometry: {geometry}")


def policy_record(
    split: str,
    row_index: int,
    failure_type: str,
    density: float,
    top_k: int,
    policy_name: str,
    gate_name: str,
    gate_triggered: bool,
    discovery_geometry: str,
    discovery_count: int,
    discovery_hits: int,
    first_hits: int,
    risk_top_mean: float,
    risk_gap: float,
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    sample_mask: np.ndarray,
    coverage_metrics: dict[str, object],
) -> dict[str, object]:
    metrics = sampling_metrics(wafer_map, sample_mask)
    followup_mask = np.asarray(sample_mask, dtype=bool) & ~np.asarray(first_mask, dtype=bool)
    followup_metrics = sampling_metrics(wafer_map, followup_mask)
    followup_valid_count = int(followup_metrics["sampled_valid_count"])
    followup_defects = int(followup_metrics["sampled_defects"])
    precision = followup_defects / followup_valid_count if followup_valid_count else float("nan")
    coverage = float(metrics["defect_coverage"])
    base_coverage = float(coverage_metrics["defect_coverage"])
    gain = 100.0 * (coverage - base_coverage) / base_coverage if base_coverage > 0 else float("nan")
    return {
        "split": split,
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "top_k": int(top_k),
        "policy_name": policy_name,
        "gate_name": gate_name,
        "gate_triggered": int(gate_triggered),
        "discovery_geometry": discovery_geometry,
        "discovery_count": int(discovery_count),
        "discovery_hits": int(discovery_hits),
        "first_hit_count": int(first_hits),
        "risk_top32_mean": float(risk_top_mean),
        "risk_top32_gap": float(risk_gap),
        "sampled_valid_count": int(metrics["sampled_valid_count"]),
        "sampled_defects": int(metrics["sampled_defects"]),
        "followup_valid_count": followup_valid_count,
        "followup_defects": followup_defects,
        "followup_precision_at_k": precision,
        "defect_coverage": coverage,
        "coverage_defect_coverage": base_coverage,
        "defect_coverage_gain_pct": gain,
        "absolute_error": float(metrics["absolute_error"]),
        "coverage_absolute_error": float(coverage_metrics["absolute_error"]),
        "absolute_error_delta": float(metrics["absolute_error"]) - float(coverage_metrics["absolute_error"]),
        "severe_miss": int(metrics["severe_miss"]),
    }


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(
            [
                "split",
                "target_density",
                "top_k",
                "policy_name",
                "gate_name",
                "discovery_geometry",
                "discovery_count",
            ],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            trigger_rate=("gate_triggered", "mean"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    pattern = (
        rows.groupby(
            [
                "split",
                "target_density",
                "failureType",
                "top_k",
                "policy_name",
                "gate_name",
                "discovery_geometry",
                "discovery_count",
            ],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            trigger_rate=("gate_triggered", "mean"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    low_rows = rows[rows["gate_triggered"] == 1].copy()
    low_summary = (
        low_rows.groupby(
            [
                "split",
                "target_density",
                "failureType",
                "policy_name",
                "gate_name",
                "discovery_geometry",
                "discovery_count",
            ],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    return summary, pattern, low_summary


def select_val_best(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = summary[(summary["split"] == "val") & ~summary["policy_name"].isin(["coverage32", "ensemble_top32"])].copy()
    global_scores = (
        val.groupby(["policy_name", "gate_name", "discovery_geometry", "discovery_count"], dropna=False)
        .agg(
            mean_precision=("mean_followup_precision_at_k", "mean"),
            mean_coverage=("mean_defect_coverage", "mean"),
            mean_abs_delta=("mean_absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
            mean_trigger_rate=("trigger_rate", "mean"),
        )
        .reset_index()
    )
    global_best = (
        global_scores[global_scores["mean_trigger_rate"] > 0.0]
        .sort_values(["mean_precision", "mean_coverage"], ascending=[False, False])
        .head(1)
    )

    density_candidates = val[val["trigger_rate"] > 0.0].copy()
    density_best = (
        density_candidates.sort_values(
            ["target_density", "mean_followup_precision_at_k", "mean_defect_coverage"],
            ascending=[True, False, False],
        )
        .groupby("target_density", as_index=False)
        .head(1)
    )
    return global_best, density_best


def selected_test_summary(summary: pd.DataFrame, global_best: pd.DataFrame, density_best: pd.DataFrame) -> pd.DataFrame:
    test = summary[summary["split"] == "test"].copy()
    base = test[test["policy_name"].isin(["coverage32", "ensemble_top32"])].copy()
    base["selection"] = "baseline"
    parts = [base]
    if not global_best.empty:
        row = global_best.iloc[0]
        selected = test[
            (test["policy_name"] == row["policy_name"])
            & (test["gate_name"] == row["gate_name"])
            & (test["discovery_geometry"] == row["discovery_geometry"])
            & (test["discovery_count"] == int(row["discovery_count"]))
        ].copy()
        selected["selection"] = "val_global_best_gate"
        parts.append(selected)
    density_parts = []
    for row in density_best.itertuples(index=False):
        selected = test[
            np.isclose(test["target_density"], float(row.target_density))
            & (test["policy_name"] == row.policy_name)
            & (test["gate_name"] == row.gate_name)
            & (test["discovery_geometry"] == row.discovery_geometry)
            & (test["discovery_count"] == int(row.discovery_count))
        ].copy()
        selected["selection"] = "val_density_best_gate"
        density_parts.append(selected)
    if density_parts:
        parts.append(pd.concat(density_parts, ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def risk_summary(scores: np.ndarray, top_k: int) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("nan"), float("nan")
    top = np.sort(scores)[::-1][: min(int(top_k), len(scores))]
    return float(top.mean()), float(top.mean() - scores.mean())


def evaluate_split(
    split_name: str,
    patterned: pd.DataFrame,
    ids: np.ndarray,
    densities: list[float],
    args: argparse.Namespace,
    point_model,
    cnn_model,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    df = patterned[patterned.index.isin(set(int(v) for v in ids))]
    defects_cache: dict[int, np.ndarray] = {}
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        valid = valid_die_mask(wafer_map)
        defects = defects_cache.setdefault(row_index, defect_mask(wafer_map))
        for density in densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(density))
            first_hits = int((first_mask & defects).sum())
            coverage_follow = make_coverage_sampling_mask(wafer_map, n_points=args.top_k, existing_mask=first_mask)
            coverage_mask = first_mask | coverage_follow
            coverage_metrics = sampling_metrics(wafer_map, coverage_mask)

            candidates, ensemble_scores, _, _ = ensemble_candidate_scores(
                point_model,
                cnn_model,
                wafer_map,
                first_mask,
                float(density),
                float(args.ensemble_noncnn_weight),
                device,
                row_index,
                failure_type,
            )
            risk_top_mean, risk_gap = risk_summary(ensemble_scores, args.top_k)
            ensemble_follow = topk_from_candidate_scores(wafer_map, candidates, ensemble_scores, args.top_k)
            ensemble_mask = first_mask | ensemble_follow

            records.append(
                policy_record(
                    split_name,
                    row_index,
                    failure_type,
                    float(density),
                    args.top_k,
                    "coverage32",
                    "none",
                    False,
                    "none",
                    0,
                    0,
                    first_hits,
                    risk_top_mean,
                    risk_gap,
                    wafer_map,
                    first_mask,
                    coverage_mask,
                    coverage_metrics,
                )
            )
            records.append(
                policy_record(
                    split_name,
                    row_index,
                    failure_type,
                    float(density),
                    args.top_k,
                    "ensemble_top32",
                    "none",
                    False,
                    "none",
                    0,
                    0,
                    first_hits,
                    risk_top_mean,
                    risk_gap,
                    wafer_map,
                    first_mask,
                    ensemble_mask,
                    coverage_metrics,
                )
            )

            gate_cache: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
            for threshold in args.hit_thresholds:
                gate_triggered = first_hits <= int(threshold)
                gate_name = f"first_hits_le_{int(threshold)}"
                for geometry in args.discovery_geometries:
                    for discovery_count in args.discovery_counts:
                        discovery_count = int(discovery_count)
                        remaining = max(0, int(args.top_k) - discovery_count)
                        policy_name = f"gate_{geometry}{discovery_count}_ens{remaining}_hit{int(threshold)}"
                        if not gate_triggered:
                            sample_mask = ensemble_mask
                            discovery_hits = 0
                        else:
                            cache_key = (str(geometry), discovery_count)
                            if cache_key not in gate_cache:
                                seed = int(args.seed + row_index * 1009 + round(float(density) * 10000) + discovery_count)
                                discovery = discovery_mask(str(geometry), wafer_map, first_mask, discovery_count, seed)
                                observed = first_mask | discovery
                                rescored_candidates, rescored_scores, _, _ = ensemble_candidate_scores(
                                    point_model,
                                    cnn_model,
                                    wafer_map,
                                    observed,
                                    float(density),
                                    float(args.ensemble_noncnn_weight),
                                    device,
                                    row_index,
                                    failure_type,
                                )
                                exploit = topk_from_candidate_scores(
                                    wafer_map,
                                    rescored_candidates,
                                    rescored_scores,
                                    remaining,
                                )
                                gate_cache[cache_key] = (observed | exploit, int((discovery & defects).sum()))
                            sample_mask, discovery_hits = gate_cache[cache_key]
                        records.append(
                            policy_record(
                                split_name,
                                row_index,
                                failure_type,
                                float(density),
                                args.top_k,
                                policy_name,
                                gate_name,
                                gate_triggered,
                                str(geometry),
                                discovery_count,
                                int(discovery_hits),
                                first_hits,
                                risk_top_mean,
                                risk_gap,
                                wafer_map,
                                first_mask,
                                sample_mask,
                                coverage_metrics,
                            )
                        )
        if pos % 25 == 0 or pos == len(df):
            print(f"{split_name} low-evidence gate wafers evaluated: {pos:,}/{len(df):,}")
    return pd.DataFrame.from_records(records)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: f"{value:.5f}")
        else:
            text[col] = text[col].astype(str)
    columns = list(text.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in text.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def plot_outputs(summary: pd.DataFrame, selected: pd.DataFrame, low_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    test_selected = selected[selected["split"] == "test"].copy()
    if not test_selected.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=test_selected,
            x="target_density",
            y="mean_followup_precision_at_k",
            hue="selection",
        )
        plt.title("Low-evidence gate refinement: selected test precision@32")
        plt.xlabel("First-pass density")
        plt.ylabel("Follow-up precision@32")
        plt.tight_layout()
        plt.savefig(fig_dir / "selected_test_precision_at32.png", dpi=180)
        plt.close()

    val = summary[(summary["split"] == "val") & ~summary["policy_name"].isin(["coverage32", "ensemble_top32"])].copy()
    if not val.empty:
        top_val = (
            val.groupby(["policy_name", "discovery_geometry", "discovery_count"], dropna=False)
            .agg(mean_precision=("mean_followup_precision_at_k", "mean"))
            .reset_index()
            .sort_values("mean_precision", ascending=False)
            .head(15)
        )
        plt.figure(figsize=(12, 5))
        sns.barplot(data=top_val, x="policy_name", y="mean_precision", hue="discovery_geometry")
        plt.title("Top validation gate candidates by precision@32")
        plt.xlabel("")
        plt.ylabel("Mean precision@32")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "top_validation_gate_candidates.png", dpi=180)
        plt.close()

    scratch = low_summary[(low_summary["split"] == "test") & (low_summary["failureType"] == "Scratch")].copy()
    if not scratch.empty:
        scratch = scratch.sort_values("mean_followup_precision_at_k", ascending=False).head(20)
        plt.figure(figsize=(12, 5))
        sns.barplot(data=scratch, x="policy_name", y="mean_followup_precision_at_k", hue="target_density")
        plt.title("Scratch triggered subset: candidate precision@32")
        plt.xlabel("")
        plt.ylabel("Mean precision@32")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "scratch_triggered_candidate_precision.png", dpi=180)
        plt.close()


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    pattern: pd.DataFrame,
    low_summary: pd.DataFrame,
    global_best: pd.DataFrame,
    density_best: pd.DataFrame,
    selected: pd.DataFrame,
    train_count: int,
    val_count: int,
    test_count: int,
    checkpoint: dict,
) -> None:
    test_selected = selected[selected["split"] == "test"].copy()
    scratch_loc = pattern[
        (pattern["split"] == "test")
        & (pattern["failureType"].isin(["Scratch", "Loc"]))
        & (
            pattern["policy_name"].isin(["coverage32", "ensemble_top32"])
            | pattern["policy_name"].isin(set(test_selected["policy_name"]))
        )
    ].copy()
    triggered = low_summary[
        (low_summary["split"] == "test")
        & (low_summary["failureType"].isin(["Scratch", "Loc"]))
    ].copy()
    triggered = triggered.sort_values("mean_followup_precision_at_k", ascending=False).head(30)
    lines = [
        "# Low-Evidence Gate Refinement v1",
        "",
        "Purpose: test whether low-evidence wafers should spend part of the limited follow-up budget on geometry-based discovery before ensemble risk exploitation.",
        "",
        "Default exploitation score:",
        "",
        f"- `score = {args.ensemble_noncnn_weight:.2f} * nonCNN_probability + {1.0 - args.ensemble_noncnn_weight:.2f} * CNN_probability`",
        "",
        "Setup:",
        "",
        f"- train wafers after validation split: {train_count}",
        f"- validation wafers: {val_count}",
        f"- test wafers: {test_count}",
        f"- top-k follow-up budget: {args.top_k}",
        f"- hit thresholds: {', '.join(str(v) for v in args.hit_thresholds)}",
        f"- discovery counts: {', '.join(str(v) for v in args.discovery_counts)}",
        f"- discovery geometries: {', '.join(args.discovery_geometries)}",
        f"- CNN checkpoint best epoch: {checkpoint.get('best_epoch', 'unknown')}",
        "",
        "Dense maps are used only for offline evaluation and to simulate labels observed after the discovery substep.",
        "",
        "## Validation-Selected Global Best Gate",
        "",
        dataframe_to_markdown(global_best.round(5)),
        "",
        "## Validation-Selected Density Best Gates",
        "",
        dataframe_to_markdown(density_best.round(5)),
        "",
        "## Test Summary: Baselines And Selected Gates",
        "",
        dataframe_to_markdown(test_selected.round(5)),
        "",
        "## Test Pattern Focus: Scratch And Loc",
        "",
        dataframe_to_markdown(scratch_loc.round(5)),
        "",
        "## Triggered Scratch/Loc Candidate Detail",
        "",
        dataframe_to_markdown(triggered.round(5)),
        "",
        "Interpretation rule: a gate is useful only if validation selects it and its fixed test performance improves precision@32 or Scratch/Loc behavior without a major global penalty.",
        "",
    ]
    (args.out_dir / "low_evidence_gate_refinement_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patterned = pd.read_pickle(args.patterned)
    rng = np.random.default_rng(args.seed)
    train_ids, test_ids = density_policy.split_wafers(patterned, args.test_size, args.seed)
    train_ids = sparse_cnn.limit_ids(patterned, train_ids, args.max_train_wafers, rng, args.seed)
    test_ids = sparse_cnn.limit_ids(patterned, test_ids, args.max_test_wafers, rng, args.seed + 1)
    train_ids, val_ids = sparse_cnn.validation_split(patterned, train_ids, args.val_size, args.seed)

    cnn_model, checkpoint = load_cnn_model(args.cnn_model, device)
    point_args = argparse.Namespace(
        seed=args.seed,
        max_train_wafers=0,
        max_defect_candidates=args.max_defect_candidates,
        max_normal_candidates=args.max_normal_candidates,
        point_estimators=args.point_estimators,
        n_jobs=args.n_jobs,
    )
    print(
        "low-evidence refinement "
        f"train wafers={len(train_ids):,}, val wafers={len(val_ids):,}, test wafers={len(test_ids):,}, "
        f"device={device}"
    )
    point_train = density_policy.build_point_training_data(patterned, train_ids, args.densities, point_args)
    point_model = density_policy.train_point_model(point_train, point_args)

    val_rows = evaluate_split("val", patterned, val_ids, args.densities, args, point_model, cnn_model, device)
    test_rows = evaluate_split("test", patterned, test_ids, args.densities, args, point_model, cnn_model, device)
    rows = pd.concat([val_rows, test_rows], ignore_index=True)
    summary, pattern, low_summary = summarize(rows)
    global_best, density_best = select_val_best(summary)
    selected = selected_test_summary(summary, global_best, density_best)

    rows.to_csv(args.out_dir / "low_evidence_gate_rows.csv", index=False)
    summary.to_csv(args.out_dir / "low_evidence_gate_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "low_evidence_gate_pattern_summary.csv", index=False)
    low_summary.to_csv(args.out_dir / "low_evidence_gate_triggered_pattern_summary.csv", index=False)
    global_best.to_csv(args.out_dir / "validation_global_best_gate.csv", index=False)
    density_best.to_csv(args.out_dir / "validation_density_best_gates.csv", index=False)
    selected.to_csv(args.out_dir / "test_selected_gate_summary.csv", index=False)
    pd.DataFrame({"train_ids": pd.Series(train_ids), "val_ids": pd.Series(val_ids), "test_ids": pd.Series(test_ids)}).to_csv(
        args.out_dir / "low_evidence_gate_split_ids.csv",
        index=False,
    )
    plot_outputs(summary, selected, low_summary, args.fig_dir)
    write_report(
        args,
        summary,
        pattern,
        low_summary,
        global_best,
        density_best,
        selected,
        len(train_ids),
        len(val_ids),
        len(test_ids),
        checkpoint,
    )

    print(f"wrote low-evidence gate outputs to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")
    print("Validation global best gate:")
    print(global_best.round(5).to_string(index=False))
    print("Validation density best gates:")
    print(density_best.round(5).to_string(index=False))
    print("Test selected gates:")
    print(selected.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
