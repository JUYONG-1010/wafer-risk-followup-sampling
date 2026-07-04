from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import (
    FEATURE_COLUMNS,
    candidate_feature_frame,
    make_first_pass_mask,
)
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    sampling_metrics,
    valid_die_mask,
)


DEFAULT_DATASET = (
    Path("data") / "processed" / "point_ranking_v0" / "point_ranking_dataset.csv"
)
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "point_ranking_v0" / "model_training"
DEFAULT_COST_WEIGHTS = [0.0, 0.003, 0.01]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first-pass-only ML point-ranking model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test-wafers", type=int, default=300)
    parser.add_argument("--top-k", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--bias-weight", type=float, default=1.0)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cost-weights", type=float, nargs="+", default=DEFAULT_COST_WEIGHTS)
    return parser.parse_args()


def split_wafers(dataset: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    wafers = np.array(sorted(dataset["row_index"].unique()))
    train_wafers, test_wafers = train_test_split(
        wafers,
        test_size=test_size,
        random_state=seed,
    )
    return np.asarray(train_wafers), np.asarray(test_wafers)


def train_model(train_df: pd.DataFrame, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label_candidate_is_defect"])
    return model


def classification_metrics(model: RandomForestClassifier, test_df: pd.DataFrame) -> dict[str, float]:
    if test_df.empty:
        return {"roc_auc": np.nan, "average_precision": np.nan}
    y_true = test_df["label_candidate_is_defect"].astype(int).to_numpy()
    y_score = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))
    return {
        "roc_auc": roc_auc,
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


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


def select_diverse_candidates(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidates: pd.DataFrame,
    top_k: int,
    diversity_weight: float,
) -> pd.DataFrame:
    """Greedily select high-score candidates while preserving spatial spread."""
    if candidates.empty or top_k <= 0:
        return candidates.head(0)

    valid = valid_die_mask(wafer_map)
    valid_y, valid_x = np.nonzero(valid)
    if len(valid_x) == 0:
        return candidates.head(0)
    max_scale = float(
        np.sqrt((valid_y.max() - valid_y.min()) ** 2 + (valid_x.max() - valid_x.min()) ** 2)
    )
    max_scale = max(max_scale, 1.0)

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=float)
    scores = candidates["score"].to_numpy(dtype=float)
    selected_seed = np.column_stack(np.nonzero(first_mask & valid)).astype(float)
    if len(selected_seed) == 0:
        selected_seed = np.array([[valid_y.mean(), valid_x.mean()]], dtype=float)

    min_dist = np.sqrt(
        ((coords[:, None, :] - selected_seed[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1) / max_scale
    available = np.ones(len(candidates), dtype=bool)
    selected_indices: list[int] = []

    for _ in range(min(top_k, len(candidates))):
        if not available.any():
            break
        current_max = float(min_dist[available].max()) if available.any() else 1.0
        diversity = min_dist / max(current_max, 1e-9)
        hybrid = scores + diversity_weight * diversity
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
    """Select candidates using defect probability, diversity, and ratio-bias penalty."""
    if candidates.empty or top_k <= 0:
        return candidates.head(0)

    valid = valid_die_mask(wafer_map)
    valid_y, valid_x = np.nonzero(valid)
    if len(valid_x) == 0:
        return candidates.head(0)
    max_scale = float(
        np.sqrt((valid_y.max() - valid_y.min()) ** 2 + (valid_x.max() - valid_x.min()) ** 2)
    )
    max_scale = max(max_scale, 1.0)

    first_metrics = sampling_metrics(wafer_map, first_mask)
    first_count = int(first_metrics["sampled_valid_count"])
    first_defects = int(first_metrics["sampled_defects"])
    first_ratio = float(first_metrics["sampled_defect_ratio"])
    target_ratio = (
        first_ratio_weight * first_ratio
        + (1.0 - first_ratio_weight) * global_target_ratio
    )

    coords = candidates[["candidate_y", "candidate_x"]].to_numpy(dtype=float)
    probabilities = candidates["score"].to_numpy(dtype=float)
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
        if not available.any():
            break
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
                "spatial_cost_proxy": float(metrics["absolute_error"])
                + cost_weight * int(metrics["sampled_valid_count"]),
                **metrics,
            }
        )
    return records


def evaluate_test_wafers(
    model: RandomForestClassifier,
    patterned: pd.DataFrame,
    test_wafers: np.ndarray,
    top_k_values: list[int],
    cost_weights: list[float],
    max_test_wafers: int,
    diversity_weight: float,
    bias_weight: float,
    first_ratio_weight: float,
    global_target_ratio: float,
) -> pd.DataFrame:
    test_set = set(int(v) for v in test_wafers)
    eval_df = patterned[patterned.index.isin(test_set)].copy()
    if max_test_wafers:
        eval_df = eval_df.head(max_test_wafers).copy()

    records: list[dict[str, object]] = []
    total = len(eval_df)
    for pos, row in enumerate(eval_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        wafer_map = np.asarray(row.waferMap)

        for first_pass_type in ["grid9", "grid25"]:
            first_mask = make_first_pass_mask(wafer_map, first_pass_type)
            records.extend(
                evaluate_mask(
                    row_index,
                    str(failure_type),
                    first_pass_type,
                    "first_only",
                    wafer_map,
                    first_mask,
                    cost_weights,
                )
            )

            for n_points in [16, 32]:
                coverage = make_coverage_sampling_mask(
                    wafer_map,
                    n_points=n_points,
                    existing_mask=first_mask,
                )
                records.extend(
                    evaluate_mask(
                        row_index,
                        str(failure_type),
                        first_pass_type,
                        f"coverage{n_points}",
                        wafer_map,
                        first_mask | coverage,
                        cost_weights,
                    )
                )

            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=first_pass_type,
                row_index=row_index,
                failure_type=str(failure_type),
                include_label=True,
            )
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["score"] = model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]
            candidates = candidates.sort_values("score", ascending=False)

            for top_k in top_k_values:
                selected = candidates.head(top_k)
                ml_mask = make_mask_from_points(wafer_map, first_mask, selected)
                records.extend(
                    evaluate_mask(
                        row_index,
                        str(failure_type),
                        first_pass_type,
                        f"ml_rank{top_k}",
                        wafer_map,
                        ml_mask,
                        cost_weights,
                    )
                )

                diverse_selected = select_diverse_candidates(
                    wafer_map,
                    first_mask,
                    candidates,
                    top_k=top_k,
                    diversity_weight=diversity_weight,
                )
                diverse_mask = make_mask_from_points(
                    wafer_map, first_mask, diverse_selected
                )
                records.extend(
                    evaluate_mask(
                        row_index,
                        str(failure_type),
                        first_pass_type,
                        f"ml_diverse{top_k}",
                        wafer_map,
                        diverse_mask,
                        cost_weights,
                    )
                )

                bias_selected = select_bias_aware_candidates(
                    wafer_map,
                    first_mask,
                    candidates,
                    top_k=top_k,
                    diversity_weight=diversity_weight,
                    bias_weight=bias_weight,
                    first_ratio_weight=first_ratio_weight,
                    global_target_ratio=global_target_ratio,
                )
                bias_mask = make_mask_from_points(
                    wafer_map, first_mask, bias_selected
                )
                records.extend(
                    evaluate_mask(
                        row_index,
                        str(failure_type),
                        first_pass_type,
                        f"ml_biasaware{top_k}",
                        wafer_map,
                        bias_mask,
                        cost_weights,
                    )
                )

        if pos % 100 == 0 or pos == total:
            print(f"point-ranking test wafers evaluated: {pos:,}/{total:,}")

    return pd.DataFrame.from_records(records)


def summarize_eval(eval_results: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_results.groupby(["cost_weight", "first_pass_type", "strategy"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_spatial_cost_proxy=("spatial_cost_proxy", "mean"),
        )
        .reset_index()
    )


def mean_actual_defect_ratio(patterned: pd.DataFrame, train_wafers: np.ndarray) -> float:
    ratios: list[float] = []
    train_set = set(int(v) for v in train_wafers)
    train_df = patterned[patterned.index.isin(train_set)]
    for row in train_df.itertuples(index=False):
        wafer_map = np.asarray(row.waferMap)
        valid = valid_die_mask(wafer_map)
        valid_count = int(valid.sum())
        if valid_count:
            ratios.append(float(defect_mask(wafer_map).sum() / valid_count))
    return float(np.mean(ratios)) if ratios else 0.0


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset)
    patterned = pd.read_pickle(args.patterned)
    train_wafers, test_wafers = split_wafers(dataset, args.test_size, args.seed)
    train_df = dataset[dataset["row_index"].isin(train_wafers)].copy()
    test_df = dataset[dataset["row_index"].isin(test_wafers)].copy()
    global_target_ratio = mean_actual_defect_ratio(patterned, train_wafers)

    model = train_model(train_df, args.seed)
    class_metrics = classification_metrics(model, test_df)
    eval_results = evaluate_test_wafers(
        model,
        patterned,
        test_wafers=test_wafers,
        top_k_values=args.top_k,
        cost_weights=args.cost_weights,
        max_test_wafers=args.max_test_wafers,
        diversity_weight=args.diversity_weight,
        bias_weight=args.bias_weight,
        first_ratio_weight=args.first_ratio_weight,
        global_target_ratio=global_target_ratio,
    )
    summary = summarize_eval(eval_results)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    model_summary = pd.DataFrame(
        [
            {
                "train_wafers": len(train_wafers),
                "test_wafers": len(test_wafers),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "diversity_weight": args.diversity_weight,
                "bias_weight": args.bias_weight,
                "first_ratio_weight": args.first_ratio_weight,
                "global_target_ratio": global_target_ratio,
                **class_metrics,
            }
        ]
    )

    eval_results.to_csv(args.out_dir / "point_ranking_eval_results.csv", index=False)
    summary.to_csv(args.out_dir / "point_ranking_eval_summary.csv", index=False)
    feature_importance.to_csv(args.out_dir / "point_ranking_feature_importance.csv", index=False)
    model_summary.to_csv(args.out_dir / "point_ranking_model_summary.csv", index=False)

    print(f"wrote point-ranking model outputs to {args.out_dir}")
    print(model_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
