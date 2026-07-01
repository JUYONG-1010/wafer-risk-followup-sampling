from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import defect_mask, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "density_risk_maps_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "40_density_risk_maps_v1"
POLICY_SCRIPT = Path("scripts") / "42_evaluate_morphology_aware_policy.py"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_module("policy42", POLICY_SCRIPT)
density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full-wafer unmeasured defect-risk maps from density first probes."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=350)
    parser.add_argument("--max-test-wafers", type=int, default=300)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=40)
    parser.add_argument("--morph-estimators", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--point-weight", type=float, default=0.60)
    parser.add_argument("--morph-weight", type=float, default=0.30)
    parser.add_argument("--weak-rescue-weight", type=float, default=0.25)
    parser.add_argument("--guarded-point-weight", type=float, default=0.30)
    parser.add_argument("--guarded-morph-weight", type=float, default=0.15)
    parser.add_argument("--guarded-weak-rescue-weight", type=float, default=0.10)
    return parser.parse_args()


def top_fraction_iou(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    if len(labels) == 0 or labels.sum() == 0:
        return np.nan
    k = max(1, int(np.ceil(len(labels) * fraction)))
    top_idx = np.argsort(scores)[-k:]
    selected = np.zeros(len(labels), dtype=bool)
    selected[top_idx] = True
    truth = labels.astype(bool)
    union = selected | truth
    if not union.any():
        return np.nan
    return float((selected & truth).sum() / union.sum())


def top_k_defect_coverage(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    total = int(labels.sum())
    if len(labels) == 0 or total == 0:
        return np.nan
    chosen = min(k, len(labels))
    top_idx = np.argsort(scores)[-chosen:]
    return float(labels[top_idx].sum() / total)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return np.nan
    return float(average_precision_score(labels, scores))


def risk_scores_for_candidates(
    candidates: pd.DataFrame,
    morph_probs: dict[str, float],
    group_irregular_prob: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = candidates.copy()
    morph_prior = policy.pattern_prior_scores(out, morph_probs)
    weak_rescue = policy.weak_pattern_rescue_scores(out, morph_probs)
    out["morph_prior_score"] = morph_prior
    out["weak_rescue_score"] = weak_rescue
    out["risk_point"] = out["point_score"].to_numpy(dtype=float)
    out["risk_morphrisk"] = (
        args.point_weight * out["point_score"].to_numpy(dtype=float)
        + args.morph_weight * morph_prior
        + args.weak_rescue_weight * weak_rescue
    )
    out["risk_guarded"] = (
        args.guarded_point_weight * out["point_score"].to_numpy(dtype=float)
        + args.guarded_morph_weight * morph_prior
        + args.guarded_weak_rescue_weight * weak_rescue * (1.0 + group_irregular_prob)
    )
    return out


def evaluate_one_score(
    row_index: int,
    failure_type: str,
    density: float,
    risk_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    first_hit_count: int,
    sampled_valid_count: int,
    morph_top1: str,
    morph_confidence: float,
    group_irregular_prob: float,
) -> dict[str, object]:
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "risk_map": risk_name,
        "candidate_count": int(len(labels)),
        "candidate_defects": int(labels.sum()),
        "first_hit_count": first_hit_count,
        "sampled_valid_count": sampled_valid_count,
        "morph_top1": morph_top1,
        "morph_confidence": morph_confidence,
        "group_irregular_prob": group_irregular_prob,
        "roc_auc": safe_auc(labels, scores),
        "average_precision": safe_ap(labels, scores),
        "top5pct_iou": top_fraction_iou(labels, scores, 0.05),
        "top10pct_iou": top_fraction_iou(labels, scores, 0.10),
        "top20pct_iou": top_fraction_iou(labels, scores, 0.20),
        "top32_defect_coverage": top_k_defect_coverage(labels, scores, 32),
        "mean_defect_score": float(np.mean(scores[labels.astype(bool)])) if labels.sum() else np.nan,
        "mean_normal_score": float(np.mean(scores[~labels.astype(bool)])) if (labels == 0).sum() else np.nan,
    }


def evaluate_risk_maps(args: argparse.Namespace) -> pd.DataFrame:
    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    densities = [float(v) for v in args.densities]

    train_wafers, test_wafers = density_policy.split_wafers(
        patterned,
        test_size=args.test_size,
        seed=args.seed,
    )
    point_train = density_policy.build_point_training_data(patterned, train_wafers, densities, args)
    point_model = density_policy.train_point_model(point_train, args)
    morph_models, morph_columns, morph_lookup = density_policy.train_morph_models(
        morph_data,
        train_wafers,
        densities,
        args,
    )

    test_ids = np.asarray(test_wafers)
    if args.max_test_wafers and len(test_ids) > args.max_test_wafers:
        rng = np.random.default_rng(args.seed)
        test_ids = rng.choice(test_ids, size=args.max_test_wafers, replace=False)
    eval_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]

    records: list[dict[str, object]] = []
    for pos, row in enumerate(eval_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        defects = defect_mask(wafer_map)
        for density in densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, density)
            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=density_policy.density_key(density),
                first_mask=first_mask,
                row_index=row_index,
                failure_type=failure_type,
                include_label=True,
            )
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["point_score"] = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]

            morph_row = morph_lookup[(row_index, density)]
            cols = morph_columns[density]
            exact_model = morph_models[(density, "exact")]
            group_model = morph_models[(density, "group")]
            x_morph = pd.DataFrame([morph_row[cols].to_dict()])
            exact_proba = exact_model.predict_proba(x_morph)[0]
            group_proba = group_model.predict_proba(x_morph)[0]
            morph_probs = {
                str(label): float(prob)
                for label, prob in zip(exact_model.classes_, exact_proba, strict=True)
            }
            group_probs = {
                str(label): float(prob)
                for label, prob in zip(group_model.classes_, group_proba, strict=True)
            }
            morph_top1 = max(morph_probs, key=morph_probs.get)
            morph_confidence = float(morph_probs[morph_top1])
            group_irregular_prob = float(group_probs.get("irregular_local", 0.0))

            scored = risk_scores_for_candidates(candidates, morph_probs, group_irregular_prob, args)
            labels = scored["label_candidate_is_defect"].astype(int).to_numpy()
            sampled_valid_count = int((first_mask & valid_die_mask(wafer_map)).sum())
            first_hit_count = int((first_mask & defects).sum())
            for risk_col in ["risk_point", "risk_morphrisk", "risk_guarded"]:
                records.append(
                    evaluate_one_score(
                        row_index,
                        failure_type,
                        density,
                        risk_col,
                        labels,
                        scored[risk_col].to_numpy(dtype=float),
                        first_hit_count,
                        sampled_valid_count,
                        morph_top1,
                        morph_confidence,
                        group_irregular_prob,
                    )
                )
        if pos % 50 == 0 or pos == len(eval_df):
            print(f"density risk-map wafers evaluated: {pos:,}/{len(eval_df):,}")
    return pd.DataFrame.from_records(records)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["target_density", "risk_map"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_candidate_defects=("candidate_defects", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top5pct_iou=("top5pct_iou", "mean"),
            mean_top10pct_iou=("top10pct_iou", "mean"),
            mean_top20pct_iou=("top20pct_iou", "mean"),
            mean_top32_defect_coverage=("top32_defect_coverage", "mean"),
            mean_defect_score=("mean_defect_score", "mean"),
            mean_normal_score=("mean_normal_score", "mean"),
            mean_first_hit_count=("first_hit_count", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
        )
        .reset_index()
    )
    pattern = (
        results.groupby(["target_density", "failureType", "risk_map"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top10pct_iou=("top10pct_iou", "mean"),
            mean_top32_defect_coverage=("top32_defect_coverage", "mean"),
        )
        .reset_index()
    )
    return summary, pattern


def plot_summary(summary: pd.DataFrame, pattern: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    for metric, filename, ylabel in [
        ("mean_top10pct_iou", "risk_map_top10_iou_vs_density.png", "Mean Top-10% IoU"),
        ("mean_top32_defect_coverage", "risk_map_top32_coverage_vs_density.png", "Mean Top-32 defect coverage"),
        ("mean_average_precision", "risk_map_average_precision_vs_density.png", "Mean average precision"),
    ]:
        plt.figure(figsize=(8.6, 5.0))
        sns.lineplot(data=data, x="density_pct", y=metric, hue="risk_map", marker="o")
        plt.xlabel("Initial probe density (%)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    focus = pattern.copy()
    focus["density_pct"] = (focus["target_density"] * 100.0).map(lambda v: f"{v:g}%")
    for risk_map in sorted(focus["risk_map"].unique()):
        subset = focus[focus["risk_map"] == risk_map]
        pivot = subset.pivot_table(
            index="failureType",
            columns="density_pct",
            values="mean_top10pct_iou",
            aggfunc="mean",
        )
        plt.figure(figsize=(6.8, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Top-10% IoU by pattern: {risk_map}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"pattern_top10_iou_{risk_map}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_risk_maps(args)
    summary, pattern = summarize(results)

    results.to_csv(args.out_dir / "density_risk_map_results.csv", index=False)
    summary.to_csv(args.out_dir / "density_risk_map_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "density_risk_map_pattern_summary.csv", index=False)
    plot_summary(summary, pattern, args.fig_dir)

    print(f"wrote risk-map outputs to {args.out_dir}")
    print(f"wrote risk-map figures to {args.fig_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
