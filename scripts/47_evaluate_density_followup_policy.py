from __future__ import annotations

import argparse
import importlib.util
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

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    nearest_valid_cell,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "density_followup_policy_v1_smoke"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "33_density_followup_policy_v1_smoke"
DEFAULT_COST_WEIGHTS = [0.003]


def load_policy_helpers():
    module_path = PROJECT_ROOT / "scripts" / "42_evaluate_morphology_aware_policy.py"
    spec = importlib.util.spec_from_file_location("policy42", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_policy_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate follow-up policies after density-based initial probes."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=1200)
    parser.add_argument("--max-test-wafers", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=80)
    parser.add_argument("--morph-estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--point-weight", type=float, default=0.60)
    parser.add_argument("--morph-weight", type=float, default=0.30)
    parser.add_argument("--weak-rescue-weight", type=float, default=0.25)
    parser.add_argument("--diversity-weight", type=float, default=0.40)
    parser.add_argument("--uncertainty-diversity-weight", type=float, default=0.35)
    parser.add_argument("--bias-weight", type=float, default=1.0)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cost-weights", type=float, nargs="+", default=DEFAULT_COST_WEIGHTS)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--hybrid-replacements", type=int, nargs="+", default=[4, 8, 12])
    return parser.parse_args()


def failure_type(row) -> str:
    value = getattr(row, "failureType_clean", None)
    if value is None:
        value = row.failureType
    return str(value)


def density_key(density: float) -> str:
    return f"density_{density:.3f}".replace(".", "p")


def center_seed_mask(wafer_map: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    seed = np.zeros_like(valid, dtype=bool)
    if not valid.any():
        return seed
    cy, cx = wafer_center(valid)
    cell = nearest_valid_cell(valid, cy, cx)
    if cell is not None:
        seed[cell] = True
    return seed


def make_initial_coverage_mask(wafer_map: np.ndarray, density: float) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    if not valid.any():
        return np.zeros_like(valid, dtype=bool)
    target_count = int(np.ceil(int(valid.sum()) * density))
    target_count = max(1, min(target_count, int(valid.sum())))
    first = center_seed_mask(wafer_map)
    if target_count <= int(first.sum()):
        return first & valid
    follow = make_coverage_sampling_mask(
        wafer_map,
        n_points=target_count - int(first.sum()),
        existing_mask=first,
    )
    return (first | follow) & valid


def split_wafers(patterned: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = patterned.apply(failure_type, axis=1)
    train_idx, test_idx = train_test_split(
        patterned.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def sample_candidate_coords(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    max_defect: int,
    max_normal: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    coords = np.column_stack(np.nonzero(valid & ~first_mask))
    if len(coords) == 0:
        return coords
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    defect_coords = coords[defects[yy, xx]]
    normal_coords = coords[~defects[yy, xx]]
    if len(defect_coords) > max_defect:
        defect_coords = defect_coords[rng.choice(len(defect_coords), size=max_defect, replace=False)]
    if len(normal_coords) > max_normal:
        normal_coords = normal_coords[rng.choice(len(normal_coords), size=max_normal, replace=False)]
    if len(defect_coords) == 0:
        return normal_coords
    if len(normal_coords) == 0:
        return defect_coords
    return np.vstack([defect_coords, normal_coords])


def build_point_training_data(
    patterned: pd.DataFrame,
    train_wafers: np.ndarray,
    densities: list[float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    train_ids = np.asarray(train_wafers)
    if args.max_train_wafers and len(train_ids) > args.max_train_wafers:
        train_ids = rng.choice(train_ids, size=args.max_train_wafers, replace=False)

    frames: list[pd.DataFrame] = []
    train_df = patterned[patterned.index.isin(set(int(v) for v in train_ids))]
    for pos, row in enumerate(train_df.itertuples(index=True), start=1):
        wafer_map = np.asarray(row.waferMap)
        label = failure_type(row)
        for density in densities:
            first_mask = make_initial_coverage_mask(wafer_map, density)
            coords = sample_candidate_coords(
                wafer_map,
                first_mask,
                args.max_defect_candidates,
                args.max_normal_candidates,
                rng,
            )
            if len(coords) == 0:
                continue
            frames.append(
                candidate_feature_frame(
                    wafer_map,
                    first_pass_type=density_key(density),
                    first_mask=first_mask,
                    candidate_coords=coords,
                    row_index=int(row.Index),
                    failure_type=label,
                    include_label=True,
                )
            )
        if pos % 200 == 0 or pos == len(train_df):
            print(f"density point-training wafers processed: {pos:,}/{len(train_df):,}")
    if not frames:
        raise RuntimeError("No point-ranking training rows were generated.")
    data = pd.concat(frames, ignore_index=True)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def train_point_model(train_data: pd.DataFrame, args: argparse.Namespace) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=args.point_estimators,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data["label_candidate_is_defect"])
    return model


def point_model_metrics(model: RandomForestClassifier, train_data: pd.DataFrame) -> dict[str, float]:
    y_true = train_data["label_candidate_is_defect"].astype(int).to_numpy()
    y_score = model.predict_proba(train_data[FEATURE_COLUMNS])[:, 1]
    return {
        "train_point_roc_auc": float(roc_auc_score(y_true, y_score)),
        "train_point_average_precision": float(average_precision_score(y_true, y_score)),
    }


def morph_feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {
        "row_index",
        "failureType",
        "failure_group",
        "target_density",
        "initial_probe_type",
    }
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def train_morph_models(
    morph_data: pd.DataFrame,
    train_wafers: np.ndarray,
    densities: list[float],
    args: argparse.Namespace,
) -> tuple[
    dict[tuple[float, str], RandomForestClassifier],
    dict[float, list[str]],
    dict[tuple[int, float], pd.Series],
]:
    train_set = set(int(v) for v in train_wafers)
    models: dict[tuple[float, str], RandomForestClassifier] = {}
    columns: dict[float, list[str]] = {}
    lookup: dict[tuple[int, float], pd.Series] = {}
    for row in morph_data.itertuples(index=False):
        lookup[(int(row.row_index), float(row.target_density))] = pd.Series(row._asdict())

    for density in densities:
        subset = morph_data[np.isclose(morph_data["target_density"], density)].copy()
        train = subset[subset["row_index"].isin(train_set)].copy()
        cols = morph_feature_columns(subset)
        columns[density] = cols
        for target_name, target_col in [("exact", "failureType"), ("group", "failure_group")]:
            model = RandomForestClassifier(
                n_estimators=args.morph_estimators,
                max_depth=16,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=args.seed,
                n_jobs=args.n_jobs,
            )
            model.fit(train[cols], train[target_col])
            models[(density, target_name)] = model
    return models, columns, lookup


def mean_actual_defect_ratio(patterned: pd.DataFrame, train_wafers: np.ndarray) -> float:
    ratios: list[float] = []
    for row in patterned[patterned.index.isin(set(int(v) for v in train_wafers))].itertuples(index=False):
        wafer_map = np.asarray(row.waferMap)
        valid = valid_die_mask(wafer_map)
        if valid.any():
            ratios.append(float(defect_mask(wafer_map).sum() / valid.sum()))
    return float(np.mean(ratios)) if ratios else 0.0


def make_mask_from_points(wafer_map: np.ndarray, first_mask: np.ndarray, selected: pd.DataFrame) -> np.ndarray:
    mask = first_mask.copy()
    valid = valid_die_mask(wafer_map)
    for row in selected.itertuples(index=False):
        y = int(row.candidate_y)
        x = int(row.candidate_x)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and valid[y, x]:
            mask[y, x] = True
    return mask


def morphrisk_configs(args: argparse.Namespace) -> list[dict[str, float | str]]:
    configs: list[dict[str, float | str]] = [
        {
            "strategy": f"morphrisk{args.top_k}",
            "point_weight": args.point_weight,
            "morph_weight": args.morph_weight,
            "weak_rescue_weight": args.weak_rescue_weight,
            "diversity_weight": args.diversity_weight,
            "uncertainty_diversity_weight": args.uncertainty_diversity_weight,
            "bias_penalty_weight": 0.0,
        }
    ]
    if args.sweep:
        configs.extend(
            [
                {
                    "strategy": f"morphrisk_cautious{args.top_k}",
                    "point_weight": 0.35,
                    "morph_weight": 0.20,
                    "weak_rescue_weight": 0.10,
                    "diversity_weight": 0.75,
                    "uncertainty_diversity_weight": 0.45,
                    "bias_penalty_weight": 1.5,
                },
                {
                    "strategy": f"morphrisk_guarded{args.top_k}",
                    "point_weight": 0.25,
                    "morph_weight": 0.15,
                    "weak_rescue_weight": 0.05,
                    "diversity_weight": 1.00,
                    "uncertainty_diversity_weight": 0.50,
                    "bias_penalty_weight": 3.0,
                },
                {
                    "strategy": f"morphrisk_weakrescue{args.top_k}",
                    "point_weight": 0.45,
                    "morph_weight": 0.25,
                    "weak_rescue_weight": 0.35,
                    "diversity_weight": 0.55,
                    "uncertainty_diversity_weight": 0.45,
                    "bias_penalty_weight": 1.0,
                },
            ]
        )
    return configs


def select_guarded_morphrisk_candidates(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidates: pd.DataFrame,
    base_score: np.ndarray,
    top_k: int,
    diversity_weight: float,
    bias_penalty_weight: float,
    first_ratio_weight: float,
    global_target_ratio: float,
) -> pd.DataFrame:
    if candidates.empty or top_k <= 0:
        return candidates.head(0)

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
    probabilities = candidates["point_score"].to_numpy(dtype=float)

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=float)
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
        expected_ratio = (
            first_defects + expected_selected_defects + probabilities
        ) / max(first_count + step + 1, 1)
        over_bias = np.clip(expected_ratio - target_ratio, 0.0, None)
        hybrid = base_score + diversity_weight * diversity - bias_penalty_weight * over_bias
        hybrid[~available] = -np.inf
        best_idx = int(np.argmax(hybrid))
        selected_indices.append(best_idx)
        available[best_idx] = False
        expected_selected_defects += float(probabilities[best_idx])
        new_dist = np.sqrt(((coords - coords[best_idx]) ** 2).sum(axis=1)) / max_scale
        min_dist = np.minimum(min_dist, new_dist)
    return candidates.iloc[selected_indices].copy()


def select_hybrid_coverage_morphrisk_candidates(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    coverage_mask: np.ndarray,
    candidates: pd.DataFrame,
    base_score: np.ndarray,
    replacement_count: int,
    bias_penalty_weight: float,
    first_ratio_weight: float,
    global_target_ratio: float,
) -> pd.DataFrame:
    if candidates.empty or replacement_count <= 0:
        return candidates.head(0)

    first_metrics = sampling_metrics(wafer_map, first_mask)
    coverage_metrics = sampling_metrics(wafer_map, first_mask | coverage_mask)
    first_ratio = float(first_metrics["sampled_defect_ratio"])
    target_ratio = first_ratio_weight * first_ratio + (1.0 - first_ratio_weight) * global_target_ratio

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=int)
    coord_to_idx = {(int(y), int(x)): idx for idx, (y, x) in enumerate(coords)}
    coverage_coords = np.column_stack(np.nonzero(coverage_mask & ~first_mask))
    coverage_indices = [
        coord_to_idx[(int(y), int(x))]
        for y, x in coverage_coords
        if (int(y), int(x)) in coord_to_idx
    ]
    coverage_set = set(coverage_indices)
    noncoverage_indices = [idx for idx in range(len(candidates)) if idx not in coverage_set]
    if not coverage_indices or not noncoverage_indices:
        return candidates.iloc[coverage_indices].copy()

    replace_n = min(replacement_count, len(coverage_indices), len(noncoverage_indices))
    remove_indices = sorted(coverage_indices, key=lambda idx: float(base_score[idx]))[:replace_n]

    current_count = int(coverage_metrics["sampled_valid_count"]) - replace_n
    current_expected_defects = (
        float(coverage_metrics["sampled_defect_ratio"]) * int(coverage_metrics["sampled_valid_count"])
        - candidates.iloc[remove_indices]["point_score"].sum()
    )
    probabilities = candidates["point_score"].to_numpy(dtype=float)
    available = np.zeros(len(candidates), dtype=bool)
    available[noncoverage_indices] = True
    selected_additions: list[int] = []

    for step in range(replace_n):
        expected_ratio = (current_expected_defects + probabilities) / max(current_count + step + 1, 1)
        over_bias = np.clip(expected_ratio - target_ratio, 0.0, None)
        hybrid = base_score - bias_penalty_weight * over_bias
        hybrid[~available] = -np.inf
        best_idx = int(np.argmax(hybrid))
        selected_additions.append(best_idx)
        available[best_idx] = False
        current_expected_defects += float(probabilities[best_idx])

    final_indices = [idx for idx in coverage_indices if idx not in set(remove_indices)]
    final_indices.extend(selected_additions)
    return candidates.iloc[final_indices].copy()


def evaluate_mask(
    row_index: int,
    failure_type_value: str,
    density: float,
    strategy: str,
    wafer_map: np.ndarray,
    mask: np.ndarray,
    cost_weights: list[float],
    morph_top1: str,
    morph_confidence: float,
    morph_uncertainty_value: float,
    weak_risk_value: float,
    group_irregular_prob: float,
) -> list[dict[str, object]]:
    metrics = sampling_metrics(wafer_map, mask)
    records: list[dict[str, object]] = []
    for cost_weight in cost_weights:
        records.append(
            {
                "row_index": row_index,
                "failureType": failure_type_value,
                "target_density": density,
                "first_pass_type": density_key(density),
                "strategy": strategy,
                "cost_weight": cost_weight,
                "morph_top1": morph_top1,
                "morph_confidence": morph_confidence,
                "morph_uncertainty": morph_uncertainty_value,
                "weak_pattern_risk": weak_risk_value,
                "group_irregular_prob": group_irregular_prob,
                "spatial_cost_proxy": float(metrics["absolute_error"])
                + cost_weight * int(metrics["sampled_valid_count"]),
                **metrics,
            }
        )
    return records


def evaluate_policy(
    patterned: pd.DataFrame,
    test_wafers: np.ndarray,
    densities: list[float],
    point_model: RandomForestClassifier,
    morph_models: dict[tuple[float, str], RandomForestClassifier],
    morph_columns: dict[float, list[str]],
    morph_lookup: dict[tuple[int, float], pd.Series],
    global_target_ratio: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    test_ids = np.asarray(test_wafers)
    if args.max_test_wafers and len(test_ids) > args.max_test_wafers:
        rng = np.random.default_rng(args.seed)
        test_ids = rng.choice(test_ids, size=args.max_test_wafers, replace=False)
    eval_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]

    records: list[dict[str, object]] = []
    for pos, row in enumerate(eval_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        label = failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in densities:
            first_mask = make_initial_coverage_mask(wafer_map, density)
            morph_row = morph_lookup[(row_index, density)]
            cols = morph_columns[density]
            exact_model = morph_models[(density, "exact")]
            group_model = morph_models[(density, "group")]
            x_morph = pd.DataFrame([morph_row[cols].to_dict()])
            exact_proba = exact_model.predict_proba(x_morph)[0]
            group_proba = group_model.predict_proba(x_morph)[0]
            morph_probs = {
                str(label_name): float(prob)
                for label_name, prob in zip(exact_model.classes_, exact_proba, strict=True)
            }
            group_probs = {
                str(label_name): float(prob)
                for label_name, prob in zip(group_model.classes_, group_proba, strict=True)
            }
            morph_top1 = max(morph_probs, key=morph_probs.get)
            morph_conf = float(morph_probs[morph_top1])
            uncertainty = policy.morphology_uncertainty(morph_probs)
            exact_weak_risk = policy.weak_pattern_risk(morph_probs)
            group_irregular = float(group_probs.get("irregular_local", 0.0))
            weak_risk = max(exact_weak_risk, group_irregular)

            records.extend(
                evaluate_mask(
                    row_index,
                    label,
                    density,
                    "first_only",
                    wafer_map,
                    first_mask,
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                    group_irregular,
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
                    label,
                    density,
                    f"coverage{args.top_k}",
                    wafer_map,
                    first_mask | coverage,
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                    group_irregular,
                )
            )

            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=density_key(density),
                first_mask=first_mask,
                row_index=row_index,
                failure_type=label,
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
                    label,
                    density,
                    f"ml_rank{args.top_k}",
                    wafer_map,
                    make_mask_from_points(wafer_map, first_mask, ml_rank),
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                    group_irregular,
                )
            )

            bias_selected = policy.select_bias_aware_candidates(
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
                    label,
                    density,
                    f"ml_biasaware{args.top_k}",
                    wafer_map,
                    make_mask_from_points(wafer_map, first_mask, bias_selected),
                    args.cost_weights,
                    morph_top1,
                    morph_conf,
                    uncertainty,
                    weak_risk,
                    group_irregular,
                )
            )

            morph_prior = policy.pattern_prior_scores(candidates, morph_probs)
            weak_rescue = policy.weak_pattern_rescue_scores(candidates, morph_probs)
            for config in morphrisk_configs(args):
                diversity_weight = float(config["diversity_weight"]) + float(
                    config["uncertainty_diversity_weight"]
                ) * max(uncertainty, weak_risk)
                base_score = (
                    float(config["point_weight"]) * candidates["point_score"].to_numpy(dtype=float)
                    + float(config["morph_weight"]) * morph_prior
                    + float(config["weak_rescue_weight"]) * weak_rescue * (1.0 + group_irregular)
                )
                bias_penalty_weight = float(config["bias_penalty_weight"])
                if bias_penalty_weight > 0:
                    morph_selected = select_guarded_morphrisk_candidates(
                        wafer_map,
                        first_mask,
                        candidates,
                        base_score=base_score,
                        top_k=args.top_k,
                        diversity_weight=diversity_weight,
                        bias_penalty_weight=bias_penalty_weight,
                        first_ratio_weight=args.first_ratio_weight,
                        global_target_ratio=global_target_ratio,
                    )
                else:
                    morph_selected = policy.greedy_select_by_score(
                        wafer_map,
                        first_mask,
                        candidates,
                        base_score=base_score,
                        top_k=args.top_k,
                        diversity_weight=diversity_weight,
                    )
                records.extend(
                    evaluate_mask(
                        row_index,
                        label,
                        density,
                        str(config["strategy"]),
                        wafer_map,
                        make_mask_from_points(wafer_map, first_mask, morph_selected),
                        args.cost_weights,
                        morph_top1,
                        morph_conf,
                        uncertainty,
                        weak_risk,
                        group_irregular,
                    )
                )

            if args.sweep:
                hybrid_base_score = (
                    0.30 * candidates["point_score"].to_numpy(dtype=float)
                    + 0.15 * morph_prior
                    + 0.10 * weak_rescue * (1.0 + group_irregular)
                )
                for replacement_count in args.hybrid_replacements:
                    hybrid_selected = select_hybrid_coverage_morphrisk_candidates(
                        wafer_map,
                        first_mask,
                        coverage,
                        candidates,
                        base_score=hybrid_base_score,
                        replacement_count=int(replacement_count),
                        bias_penalty_weight=3.0,
                        first_ratio_weight=args.first_ratio_weight,
                        global_target_ratio=global_target_ratio,
                    )
                    records.extend(
                        evaluate_mask(
                            row_index,
                            label,
                            density,
                            f"hybrid_guarded{replacement_count}",
                            wafer_map,
                            make_mask_from_points(wafer_map, first_mask, hybrid_selected),
                            args.cost_weights,
                            morph_top1,
                            morph_conf,
                            uncertainty,
                            weak_risk,
                            group_irregular,
                        )
                    )
        if pos % 50 == 0 or pos == len(eval_df):
            print(f"density follow-up wafers evaluated: {pos:,}/{len(eval_df):,}")
    return pd.DataFrame.from_records(records)


def summarize_eval(eval_results: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_results.groupby(["cost_weight", "target_density", "first_pass_type", "strategy"], observed=False)
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
            mean_group_irregular_prob=("group_irregular_prob", "mean"),
        )
        .reset_index()
    )


def summarize_pattern(eval_results: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_results.groupby(
            ["cost_weight", "target_density", "failureType", "first_pass_type", "strategy"],
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


def guardrail_vs_coverage(summary: pd.DataFrame, top_k: int, cost_weight: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    data = summary[summary["cost_weight"] == cost_weight]
    for density in sorted(data["target_density"].unique()):
        subset = data[np.isclose(data["target_density"], density)]
        base = subset[subset["strategy"] == f"coverage{top_k}"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for row in subset.itertuples(index=False):
            if row.strategy == f"coverage{top_k}":
                continue
            records.append(
                {
                    "cost_weight": cost_weight,
                    "target_density": density,
                    "baseline_strategy": f"coverage{top_k}",
                    "strategy": row.strategy,
                    "baseline_absolute_error": float(base_row["mean_absolute_error"]),
                    "strategy_absolute_error": float(row.mean_absolute_error),
                    "absolute_error_delta": float(row.mean_absolute_error - base_row["mean_absolute_error"]),
                    "absolute_error_guardrail_pass": bool(
                        row.mean_absolute_error <= base_row["mean_absolute_error"]
                    ),
                    "baseline_severe_miss_rate": float(base_row["severe_miss_rate"]),
                    "strategy_severe_miss_rate": float(row.severe_miss_rate),
                    "severe_miss_relative_reduction_pct": (
                        float((base_row["severe_miss_rate"] - row.severe_miss_rate) / base_row["severe_miss_rate"] * 100.0)
                        if base_row["severe_miss_rate"]
                        else np.nan
                    ),
                    "baseline_defect_coverage": float(base_row["mean_defect_coverage"]),
                    "strategy_defect_coverage": float(row.mean_defect_coverage),
                    "defect_coverage_relative_improvement_pct": (
                        float((row.mean_defect_coverage - base_row["mean_defect_coverage"]) / base_row["mean_defect_coverage"] * 100.0)
                        if base_row["mean_defect_coverage"]
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_summary(summary: pd.DataFrame, guardrail: pd.DataFrame, fig_dir: Path, cost_weight: float) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary[summary["cost_weight"] == cost_weight].copy()
    data["density_pct"] = data["target_density"] * 100.0
    order = [
        "first_only",
        "coverage32",
        "ml_rank32",
        "ml_biasaware32",
        "morphrisk32",
        "morphrisk_cautious32",
        "morphrisk_guarded32",
        "morphrisk_weakrescue32",
        "hybrid_guarded4",
        "hybrid_guarded8",
        "hybrid_guarded12",
    ]
    for metric, filename, ylabel in [
        ("mean_defect_coverage", "density_followup_defect_coverage.png", "Mean defect coverage"),
        ("severe_miss_rate", "density_followup_severe_miss.png", "Severe miss rate"),
        ("mean_absolute_error", "density_followup_absolute_error.png", "Mean absolute error"),
    ]:
        plt.figure(figsize=(9.2, 5.2))
        sns.lineplot(
            data=data,
            x="density_pct",
            y=metric,
            hue="strategy",
            hue_order=order,
            marker="o",
        )
        plt.xlabel("Initial probe density (%)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    if not guardrail.empty:
        view = guardrail[guardrail["strategy"] == "morphrisk32"].copy()
        view["density_pct"] = view["target_density"] * 100.0
        long = view.melt(
            id_vars=["density_pct"],
            value_vars=[
                "severe_miss_relative_reduction_pct",
                "defect_coverage_relative_improvement_pct",
                "absolute_error_delta",
            ],
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(9.2, 5.2))
        sns.barplot(data=long, x="density_pct", y="value", hue="metric")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Initial probe density (%)")
        plt.ylabel("Value vs coverage32")
        plt.tight_layout()
        plt.savefig(fig_dir / "morphrisk_vs_coverage_guardrail.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    densities = [float(v) for v in args.densities]

    train_wafers, test_wafers = split_wafers(patterned, args.test_size, args.seed)
    point_train = build_point_training_data(patterned, train_wafers, densities, args)
    point_model = train_point_model(point_train, args)
    morph_models, morph_columns, morph_lookup = train_morph_models(
        morph_data,
        train_wafers,
        densities,
        args,
    )
    global_target_ratio = mean_actual_defect_ratio(patterned, train_wafers)

    eval_results = evaluate_policy(
        patterned,
        test_wafers,
        densities,
        point_model,
        morph_models,
        morph_columns,
        morph_lookup,
        global_target_ratio,
        args,
    )
    summary = summarize_eval(eval_results)
    pattern_summary = summarize_pattern(eval_results)
    guardrail = guardrail_vs_coverage(summary, args.top_k, cost_weight=args.cost_weights[0])
    model_summary = pd.DataFrame(
        [
            {
                "train_wafers": len(train_wafers),
                "test_wafers": len(test_wafers),
                "point_train_wafers_used": min(args.max_train_wafers, len(train_wafers))
                if args.max_train_wafers
                else len(train_wafers),
                "eval_test_wafers_used": min(args.max_test_wafers, len(test_wafers))
                if args.max_test_wafers
                else len(test_wafers),
                "point_train_rows": len(point_train),
                "global_target_ratio": global_target_ratio,
                "top_k": args.top_k,
                "point_weight": args.point_weight,
                "morph_weight": args.morph_weight,
                "weak_rescue_weight": args.weak_rescue_weight,
                "diversity_weight": args.diversity_weight,
                "uncertainty_diversity_weight": args.uncertainty_diversity_weight,
                "bias_weight": args.bias_weight,
                "first_ratio_weight": args.first_ratio_weight,
                "point_estimators": args.point_estimators,
                "morph_estimators": args.morph_estimators,
                "n_jobs": args.n_jobs,
                **point_model_metrics(point_model, point_train),
            }
        ]
    )

    eval_results.to_csv(args.out_dir / "density_followup_eval_results.csv", index=False)
    summary.to_csv(args.out_dir / "density_followup_eval_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "density_followup_pattern_summary.csv", index=False)
    guardrail.to_csv(args.out_dir / "density_followup_guardrail_vs_coverage.csv", index=False)
    model_summary.to_csv(args.out_dir / "density_followup_model_summary.csv", index=False)
    plot_summary(summary, guardrail, args.fig_dir, cost_weight=args.cost_weights[0])

    print(f"wrote density follow-up outputs to {args.out_dir}")
    print(f"wrote density follow-up figures to {args.fig_dir}")
    print(model_summary.round(4).to_string(index=False))
    print(summary[summary["cost_weight"] == args.cost_weights[0]].round(4).to_string(index=False))
    if not guardrail.empty:
        print(guardrail.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
