from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame, make_first_pass_mask
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    sampling_metrics,
    valid_die_mask,
)


DEFAULT_POINT_DATASET = (
    Path("data")
    / "processed"
    / "point_ranking_v0_medium"
    / "point_ranking_dataset.csv"
)
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "morphology_risk_v1"
    / "morphology_risk_dataset.csv"
)
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "morphology_aware_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "25_morphology_aware_policy_v1"
DEFAULT_COST_WEIGHTS = [0.0, 0.003, 0.01]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate morphology-risk-aware follow-up sampling policy."
    )
    parser.add_argument("--point-dataset", type=Path, default=DEFAULT_POINT_DATASET)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test-wafers", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--point-weight", type=float, default=0.55)
    parser.add_argument("--morph-weight", type=float, default=0.35)
    parser.add_argument("--weak-rescue-weight", type=float, default=0.0)
    parser.add_argument("--diversity-weight", type=float, default=0.45)
    parser.add_argument("--uncertainty-diversity-weight", type=float, default=0.35)
    parser.add_argument("--bias-weight", type=float, default=1.0)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cost-weights", type=float, nargs="+", default=DEFAULT_COST_WEIGHTS)
    parser.add_argument("--point-estimators", type=int, default=120)
    parser.add_argument("--morph-estimators", type=int, default=160)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def split_wafers(dataset: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    wafers = np.array(sorted(dataset["row_index"].unique()))
    train_wafers, test_wafers = train_test_split(
        wafers,
        test_size=test_size,
        random_state=seed,
    )
    return np.asarray(train_wafers), np.asarray(test_wafers)


def train_point_model(
    train_df: pd.DataFrame,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label_candidate_is_defect"])
    return model


def point_model_metrics(model: RandomForestClassifier, test_df: pd.DataFrame) -> dict[str, float]:
    y_true = test_df["label_candidate_is_defect"].astype(int).to_numpy()
    y_score = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    return {
        "point_roc_auc": float(roc_auc_score(y_true, y_score)),
        "point_average_precision": float(average_precision_score(y_true, y_score)),
    }


def morph_feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {"row_index", "failureType", "first_pass_type"}
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def train_morph_models(
    morph_data: pd.DataFrame,
    train_wafers: np.ndarray,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, RandomForestClassifier], dict[str, list[str]], dict[tuple[int, str], pd.Series]]:
    train_set = set(int(v) for v in train_wafers)
    models: dict[str, RandomForestClassifier] = {}
    columns: dict[str, list[str]] = {}
    lookup: dict[tuple[int, str], pd.Series] = {}

    for row in morph_data.itertuples(index=False):
        lookup[(int(row.row_index), str(row.first_pass_type))] = pd.Series(row._asdict())

    for first_pass_type in sorted(morph_data["first_pass_type"].unique()):
        subset = morph_data[morph_data["first_pass_type"] == first_pass_type].copy()
        train = subset[subset["row_index"].isin(train_set)].copy()
        cols = morph_feature_columns(subset)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(train[cols], train["failureType"])
        models[first_pass_type] = model
        columns[first_pass_type] = cols
    return models, columns, lookup


def mean_actual_defect_ratio(patterned: pd.DataFrame, train_wafers: np.ndarray) -> float:
    train_set = set(int(v) for v in train_wafers)
    ratios: list[float] = []
    for row in patterned[patterned.index.isin(train_set)].itertuples(index=False):
        wafer_map = np.asarray(row.waferMap)
        valid = valid_die_mask(wafer_map)
        if valid.any():
            ratios.append(float(defect_mask(wafer_map).sum() / valid.sum()))
    return float(np.mean(ratios)) if ratios else 0.0


def make_mask_from_points(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    selected: pd.DataFrame,
) -> np.ndarray:
    mask = first_mask.copy()
    valid = valid_die_mask(wafer_map)
    for row in selected.itertuples(index=False):
        y = int(row.candidate_y)
        x = int(row.candidate_x)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and valid[y, x]:
            mask[y, x] = True
    return mask


def pattern_prior_scores(candidates: pd.DataFrame, probs: dict[str, float]) -> np.ndarray:
    radius = candidates["candidate_radius_norm"].to_numpy(dtype=float)
    center = candidates["candidate_is_center"].to_numpy(dtype=float)
    mid = candidates["candidate_is_mid"].to_numpy(dtype=float)
    edge = candidates["candidate_is_edge"].to_numpy(dtype=float)
    same_quad = candidates["same_quadrant_as_any_hit"].to_numpy(dtype=float)
    same_zone = candidates["same_radial_zone_as_any_hit"].to_numpy(dtype=float)
    dist_hit = candidates["distance_to_nearest_first_hit_norm"].to_numpy(dtype=float)
    angle_diff = candidates["angle_diff_to_nearest_hit"].to_numpy(dtype=float)
    dist_first = candidates["distance_to_nearest_first_sample_norm"].to_numpy(dtype=float)

    local_near_hit = np.clip(1.0 - dist_hit, 0.0, 1.0)
    directional = np.clip(1.0 - angle_diff / np.pi, 0.0, 1.0)
    coverage = np.clip(dist_first, 0.0, 1.0)
    donut_ring = np.clip(1.0 - np.abs(radius - 0.55) / 0.55, 0.0, 1.0)
    near_full = 0.5 + 0.5 * np.clip(radius, 0.0, 1.0)

    components = {
        "Center": 0.75 * center + 0.25 * np.clip(1.0 - radius, 0.0, 1.0),
        "Donut": 0.65 * mid + 0.35 * donut_ring,
        "Edge-Ring": 0.70 * edge + 0.30 * np.clip(radius, 0.0, 1.0),
        "Edge-Loc": 0.55 * edge + 0.25 * same_quad + 0.20 * local_near_hit,
        "Loc": 0.40 * same_quad + 0.35 * local_near_hit + 0.25 * same_zone,
        "Scratch": 0.40 * directional + 0.35 * coverage + 0.25 * same_quad,
        "Random": coverage,
        "Near-full": near_full,
    }

    score = np.zeros(len(candidates), dtype=float)
    for label, component in components.items():
        score += float(probs.get(label, 0.0)) * component
    return np.clip(score, 0.0, 1.0)


def morphology_uncertainty(probs: dict[str, float]) -> float:
    values = np.array(list(probs.values()), dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    entropy = float(-(values * np.log2(values)).sum())
    return entropy / np.log2(len(values))


def weak_pattern_risk(probs: dict[str, float]) -> float:
    return float(probs.get("Scratch", 0.0) + probs.get("Loc", 0.0) + probs.get("Random", 0.0))


def weak_pattern_rescue_scores(candidates: pd.DataFrame, probs: dict[str, float]) -> np.ndarray:
    dist_hit = candidates["distance_to_nearest_first_hit_norm"].to_numpy(dtype=float)
    angle_diff = candidates["angle_diff_to_nearest_hit"].to_numpy(dtype=float)
    dist_first = candidates["distance_to_nearest_first_sample_norm"].to_numpy(dtype=float)
    same_quad = candidates["same_quadrant_as_any_hit"].to_numpy(dtype=float)
    same_zone = candidates["same_radial_zone_as_any_hit"].to_numpy(dtype=float)

    local_near_hit = np.clip(1.0 - dist_hit, 0.0, 1.0)
    directional = np.clip(1.0 - angle_diff / np.pi, 0.0, 1.0)
    coverage = np.clip(dist_first, 0.0, 1.0)

    loc_component = 0.45 * local_near_hit + 0.35 * same_quad + 0.20 * same_zone
    scratch_component = 0.45 * directional + 0.35 * coverage + 0.20 * same_quad
    random_component = coverage

    score = (
        float(probs.get("Loc", 0.0)) * loc_component
        + float(probs.get("Scratch", 0.0)) * scratch_component
        + float(probs.get("Random", 0.0)) * random_component
    )
    return np.clip(score, 0.0, 1.0)


def greedy_select_by_score(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidates: pd.DataFrame,
    base_score: np.ndarray,
    top_k: int,
    diversity_weight: float,
) -> pd.DataFrame:
    if candidates.empty or top_k <= 0:
        return candidates.head(0)

    valid = valid_die_mask(wafer_map)
    valid_y, valid_x = np.nonzero(valid)
    max_scale = float(
        np.sqrt((valid_y.max() - valid_y.min()) ** 2 + (valid_x.max() - valid_x.min()) ** 2)
    )
    max_scale = max(max_scale, 1.0)

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=float)
    selected_seed = np.column_stack(np.nonzero(first_mask & valid)).astype(float)
    if len(selected_seed) == 0:
        selected_seed = np.array([[valid_y.mean(), valid_x.mean()]], dtype=float)

    min_dist = np.sqrt(
        ((coords[:, None, :] - selected_seed[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1) / max_scale
    available = np.ones(len(candidates), dtype=bool)
    selected_indices: list[int] = []

    for _ in range(min(top_k, len(candidates))):
        current_max = float(min_dist[available].max()) if available.any() else 1.0
        diversity = min_dist / max(current_max, 1e-9)
        hybrid = base_score + diversity_weight * diversity
        hybrid[~available] = -np.inf
        best_idx = int(np.argmax(hybrid))
        selected_indices.append(best_idx)
        available[best_idx] = False
        new_dist = np.sqrt(((coords - coords[best_idx]) ** 2).sum(axis=1)) / max_scale
        min_dist = np.minimum(min_dist, new_dist)
    return candidates.iloc[selected_indices].copy()


def select_bias_aware_candidates(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidates: pd.DataFrame,
    top_k: int,
    diversity_weight: float,
    bias_weight: float,
    first_ratio_weight: float,
    global_target_ratio: float,
) -> pd.DataFrame:
    valid = valid_die_mask(wafer_map)
    valid_y, valid_x = np.nonzero(valid)
    max_scale = float(
        np.sqrt((valid_y.max() - valid_y.min()) ** 2 + (valid_x.max() - valid_x.min()) ** 2)
    )
    max_scale = max(max_scale, 1.0)

    first_metrics = sampling_metrics(wafer_map, first_mask)
    first_count = int(first_metrics["sampled_valid_count"])
    first_defects = int(first_metrics["sampled_defects"])
    first_ratio = float(first_metrics["sampled_defect_ratio"])
    target_ratio = first_ratio_weight * first_ratio + (1.0 - first_ratio_weight) * global_target_ratio

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=float)
    probabilities = candidates["point_score"].to_numpy(dtype=float)
    selected_seed = np.column_stack(np.nonzero(first_mask & valid)).astype(float)
    if len(selected_seed) == 0:
        selected_seed = np.array([[valid_y.mean(), valid_x.mean()]], dtype=float)

    min_dist = np.sqrt(
        ((coords[:, None, :] - selected_seed[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1) / max_scale
    available = np.ones(len(candidates), dtype=bool)
    selected_indices: list[int] = []
    expected_selected_defects = 0.0

    for step in range(min(top_k, len(candidates))):
        current_max = float(min_dist[available].max()) if available.any() else 1.0
        diversity = min_dist / max(current_max, 1e-9)
        expected_ratio_if_selected = (
            first_defects + expected_selected_defects + probabilities
        ) / max(first_count + step + 1, 1)
        over_bias = np.clip(expected_ratio_if_selected - target_ratio, 0.0, None)
        hybrid = probabilities + diversity_weight * diversity - bias_weight * over_bias
        hybrid[~available] = -np.inf
        best_idx = int(np.argmax(hybrid))
        selected_indices.append(best_idx)
        available[best_idx] = False
        expected_selected_defects += float(probabilities[best_idx])
        new_dist = np.sqrt(((coords - coords[best_idx]) ** 2).sum(axis=1)) / max_scale
        min_dist = np.minimum(min_dist, new_dist)
    return candidates.iloc[selected_indices].copy()


def evaluate_mask(
    row_index: int,
    failure_type: str,
    first_pass_type: str,
    strategy: str,
    wafer_map: np.ndarray,
    mask: np.ndarray,
    cost_weights: list[float],
    morph_top1: str | None = None,
    morph_confidence: float | None = None,
    morph_uncertainty_value: float | None = None,
    weak_risk_value: float | None = None,
) -> list[dict[str, object]]:
    metrics = sampling_metrics(wafer_map, mask)
    records: list[dict[str, object]] = []
    for cost_weight in cost_weights:
        records.append(
            {
                "row_index": row_index,
                "failureType": failure_type,
                "first_pass_type": first_pass_type,
                "strategy": strategy,
                "cost_weight": cost_weight,
                "morph_top1": morph_top1 or "",
                "morph_confidence": morph_confidence if morph_confidence is not None else np.nan,
                "morph_uncertainty": morph_uncertainty_value if morph_uncertainty_value is not None else np.nan,
                "weak_pattern_risk": weak_risk_value if weak_risk_value is not None else np.nan,
                "spatial_cost_proxy": float(metrics["absolute_error"])
                + cost_weight * int(metrics["sampled_valid_count"]),
                **metrics,
            }
        )
    return records


def evaluate_policy(
    point_model: RandomForestClassifier,
    morph_models: dict[str, RandomForestClassifier],
    morph_columns: dict[str, list[str]],
    morph_lookup: dict[tuple[int, str], pd.Series],
    patterned: pd.DataFrame,
    test_wafers: np.ndarray,
    args: argparse.Namespace,
    global_target_ratio: float,
) -> pd.DataFrame:
    eval_df = patterned[patterned.index.isin(set(int(v) for v in test_wafers))].copy()
    if args.max_test_wafers:
        eval_df = eval_df.head(args.max_test_wafers).copy()

    records: list[dict[str, object]] = []
    for pos, row in enumerate(eval_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        failure_type = str(failure_type)
        wafer_map = np.asarray(row.waferMap)

        for first_pass_type in ["grid9", "grid25"]:
            first_mask = make_first_pass_mask(wafer_map, first_pass_type)
            morph_row = morph_lookup[(row_index, first_pass_type)]
            morph_model = morph_models[first_pass_type]
            cols = morph_columns[first_pass_type]
            morph_proba = morph_model.predict_proba(pd.DataFrame([morph_row[cols].to_dict()]))[0]
            morph_probs = {
                str(label): float(prob)
                for label, prob in zip(morph_model.classes_, morph_proba, strict=True)
            }
            morph_top1 = max(morph_probs, key=morph_probs.get)
            morph_conf = float(morph_probs[morph_top1])
            uncertainty = morphology_uncertainty(morph_probs)
            weak_risk = weak_pattern_risk(morph_probs)

            records.extend(
                evaluate_mask(
                    row_index,
                    failure_type,
                    first_pass_type,
                    "first_only",
                    wafer_map,
                    first_mask,
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                )
            )

            coverage = make_coverage_sampling_mask(
                wafer_map,
                n_points=args.top_k,
                existing_mask=first_mask,
            )
            records.extend(
                evaluate_mask(
                    row_index,
                    failure_type,
                    first_pass_type,
                    f"coverage{args.top_k}",
                    wafer_map,
                    first_mask | coverage,
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                )
            )

            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=first_pass_type,
                row_index=row_index,
                failure_type=failure_type,
                include_label=True,
            )
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["point_score"] = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]

            ml_rank = candidates.sort_values("point_score", ascending=False).head(args.top_k)
            records.extend(
                evaluate_mask(
                    row_index,
                    failure_type,
                    first_pass_type,
                    f"ml_rank{args.top_k}",
                    wafer_map,
                    make_mask_from_points(wafer_map, first_mask, ml_rank),
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                )
            )

            bias_selected = select_bias_aware_candidates(
                wafer_map,
                first_mask,
                candidates,
                top_k=args.top_k,
                diversity_weight=args.diversity_weight,
                bias_weight=args.bias_weight,
                first_ratio_weight=args.first_ratio_weight,
                global_target_ratio=global_target_ratio,
            )
            records.extend(
                evaluate_mask(
                    row_index,
                    failure_type,
                    first_pass_type,
                    f"ml_biasaware{args.top_k}",
                    wafer_map,
                    make_mask_from_points(wafer_map, first_mask, bias_selected),
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                )
            )

            morph_prior = pattern_prior_scores(candidates, morph_probs)
            weak_rescue = weak_pattern_rescue_scores(candidates, morph_probs)
            uncertain_diversity_weight = (
                args.diversity_weight
                + args.uncertainty_diversity_weight * max(uncertainty, weak_risk)
            )
            base_score = (
                args.point_weight * candidates["point_score"].to_numpy(dtype=float)
                + args.morph_weight * morph_prior
                + args.weak_rescue_weight * weak_rescue
            )
            morph_selected = greedy_select_by_score(
                wafer_map,
                first_mask,
                candidates,
                base_score=base_score,
                top_k=args.top_k,
                diversity_weight=uncertain_diversity_weight,
            )
            records.extend(
                evaluate_mask(
                    row_index,
                    failure_type,
                    first_pass_type,
                    f"morphrisk{args.top_k}",
                    wafer_map,
                    make_mask_from_points(wafer_map, first_mask, morph_selected),
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                )
            )

        if pos % 100 == 0 or pos == len(eval_df):
            print(f"morphology-aware policy wafers evaluated: {pos:,}/{len(eval_df):,}")
    return pd.DataFrame.from_records(records)


def summarize_eval(eval_results: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_results.groupby(["cost_weight", "first_pass_type", "strategy"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_spatial_cost_proxy=("spatial_cost_proxy", "mean"),
            mean_morph_confidence=("morph_confidence", "mean"),
            mean_morph_uncertainty=("morph_uncertainty", "mean"),
            mean_weak_pattern_risk=("weak_pattern_risk", "mean"),
        )
        .reset_index()
    )


def summarize_pattern(eval_results: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_results.groupby(
            ["cost_weight", "failureType", "first_pass_type", "strategy"],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
        )
        .reset_index()
    )


def improvement_vs_coverage(summary: pd.DataFrame, top_k: int, cost_weight: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    data = summary[summary["cost_weight"] == cost_weight]
    for first_pass_type in sorted(data["first_pass_type"].unique()):
        base = data[
            (data["first_pass_type"] == first_pass_type)
            & (data["strategy"] == f"coverage{top_k}")
        ]
        proposed = data[
            (data["first_pass_type"] == first_pass_type)
            & (data["strategy"] == f"morphrisk{top_k}")
        ]
        if base.empty or proposed.empty:
            continue
        base_row = base.iloc[0]
        prop_row = proposed.iloc[0]
        records.append(
            {
                "cost_weight": cost_weight,
                "first_pass_type": first_pass_type,
                "baseline_strategy": f"coverage{top_k}",
                "proposed_strategy": f"morphrisk{top_k}",
                "severe_miss_relative_reduction_pct": (
                    (base_row["severe_miss_rate"] - prop_row["severe_miss_rate"])
                    / base_row["severe_miss_rate"]
                    * 100.0
                    if base_row["severe_miss_rate"]
                    else np.nan
                ),
                "defect_coverage_relative_improvement_pct": (
                    (prop_row["mean_defect_coverage"] - base_row["mean_defect_coverage"])
                    / base_row["mean_defect_coverage"]
                    * 100.0
                    if base_row["mean_defect_coverage"]
                    else np.nan
                ),
                "absolute_error_delta": prop_row["mean_absolute_error"] - base_row["mean_absolute_error"],
                "ratio_error_delta": prop_row["mean_ratio_error"] - base_row["mean_ratio_error"],
            }
        )
    return pd.DataFrame.from_records(records)


def plot_summary(summary: pd.DataFrame, improvement: pd.DataFrame, fig_dir: Path, cost_weight: float) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary[summary["cost_weight"] == cost_weight].copy()
    order = ["first_only", "coverage32", "ml_rank32", "ml_biasaware32", "morphrisk32"]

    for metric, filename, ylabel in [
        ("mean_defect_coverage", "policy_defect_coverage.png", "Mean defect coverage"),
        ("severe_miss_rate", "policy_severe_miss_rate.png", "Severe miss rate"),
        ("mean_absolute_error", "policy_absolute_error_guardrail.png", "Mean absolute error"),
    ]:
        grid = sns.catplot(
            data=data,
            x="strategy",
            y=metric,
            col="first_pass_type",
            kind="bar",
            order=order,
            color="#4C78A8",
            height=4.0,
            aspect=1.35,
        )
        for ax in grid.axes.flat:
            ax.tick_params(axis="x", rotation=30)
        grid.set_axis_labels("Strategy", ylabel)
        grid.set_titles("{col_name}")
        grid.fig.suptitle(f"{ylabel} at cost={cost_weight}", y=1.04)
        grid.fig.tight_layout()
        grid.fig.savefig(fig_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(grid.fig)

    if not improvement.empty:
        long = improvement.melt(
            id_vars=["first_pass_type"],
            value_vars=[
                "severe_miss_relative_reduction_pct",
                "defect_coverage_relative_improvement_pct",
            ],
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(8.4, 4.8))
        sns.barplot(data=long, x="first_pass_type", y="value", hue="metric")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("First-pass type")
        plt.ylabel("Improvement vs coverage32 (%)")
        plt.tight_layout()
        plt.savefig(fig_dir / "morphrisk_improvement_vs_coverage.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    point_data = pd.read_csv(args.point_dataset)
    morph_data = pd.read_csv(args.morph_dataset)
    patterned = pd.read_pickle(args.patterned)

    train_wafers, test_wafers = split_wafers(point_data, args.test_size, args.seed)
    train_point = point_data[point_data["row_index"].isin(train_wafers)].copy()
    test_point = point_data[point_data["row_index"].isin(test_wafers)].copy()

    point_model = train_point_model(
        train_point,
        args.seed,
        args.point_estimators,
        args.n_jobs,
    )
    morph_models, morph_columns, morph_lookup = train_morph_models(
        morph_data,
        train_wafers=train_wafers,
        seed=args.seed,
        n_estimators=args.morph_estimators,
        n_jobs=args.n_jobs,
    )
    global_target_ratio = mean_actual_defect_ratio(patterned, train_wafers)

    eval_results = evaluate_policy(
        point_model,
        morph_models,
        morph_columns,
        morph_lookup,
        patterned,
        test_wafers,
        args,
        global_target_ratio,
    )
    summary = summarize_eval(eval_results)
    pattern_summary = summarize_pattern(eval_results)
    improvement = improvement_vs_coverage(summary, args.top_k, cost_weight=0.003)

    model_summary = pd.DataFrame(
        [
            {
                "train_wafers": len(train_wafers),
                "test_wafers": len(test_wafers),
                "train_point_rows": len(train_point),
                "test_point_rows": len(test_point),
                "global_target_ratio": global_target_ratio,
                "top_k": args.top_k,
                "point_weight": args.point_weight,
                "morph_weight": args.morph_weight,
                "weak_rescue_weight": args.weak_rescue_weight,
                "diversity_weight": args.diversity_weight,
                "uncertainty_diversity_weight": args.uncertainty_diversity_weight,
                "point_estimators": args.point_estimators,
                "morph_estimators": args.morph_estimators,
                "n_jobs": args.n_jobs,
                **point_model_metrics(point_model, test_point),
            }
        ]
    )

    eval_results.to_csv(args.out_dir / "morphology_aware_eval_results.csv", index=False)
    summary.to_csv(args.out_dir / "morphology_aware_eval_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "morphology_aware_pattern_summary.csv", index=False)
    improvement.to_csv(args.out_dir / "morphrisk_improvement_vs_coverage.csv", index=False)
    model_summary.to_csv(args.out_dir / "morphology_aware_model_summary.csv", index=False)

    plot_summary(summary, improvement, args.fig_dir, cost_weight=0.003)

    print(f"wrote morphology-aware policy outputs to {args.out_dir}")
    print(f"wrote morphology-aware policy figures to {args.fig_dir}")
    print(model_summary.round(4).to_string(index=False))
    print(summary[summary["cost_weight"] == 0.003].round(4).to_string(index=False))
    if not improvement.empty:
        print(improvement.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
