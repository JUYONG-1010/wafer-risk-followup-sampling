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
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import defect_mask, make_coverage_sampling_mask, sampling_metrics, valid_die_mask, wafer_center


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_CNN_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_MORPH_DATASET = (
    Path("data") / "processed" / "initial_probe_density_v1" / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "scratch_guard_policy_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "78_scratch_guard_policy_v1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", Path("experiments") / "47_evaluate_density_followup_policy.py")
sparse_cnn = load_module("sparse_cnn68", Path("experiments") / "68_train_sparse_cnn_risk_map.py")
gate_refinement = load_module("gate_refinement77", Path("experiments") / "77_evaluate_low_evidence_gate_refinement.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scratch-aware fixed guard allocation before ensemble exploitation."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--guard-counts", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=2500)
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=100)
    parser.add_argument("--morph-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--ensemble-noncnn-weight", type=float, default=0.3)
    return parser.parse_args()


def morphology_feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {
        "row_index",
        "failureType",
        "failure_group",
        "target_density",
        "target_sample_count",
        "initial_probe_type",
    }
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def train_morphology_models(
    morph_data: pd.DataFrame,
    train_ids: np.ndarray,
    densities: list[float],
    args: argparse.Namespace,
) -> tuple[dict[float, RandomForestClassifier], dict[float, list[str]], dict[tuple[int, float], dict[str, object]]]:
    train_set = set(int(v) for v in train_ids)
    models: dict[float, RandomForestClassifier] = {}
    columns: dict[float, list[str]] = {}
    lookup: dict[tuple[int, float], dict[str, object]] = {}
    numeric_cols = morph_data.select_dtypes(include=[np.number]).columns
    morph_data[numeric_cols] = morph_data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for row in morph_data.itertuples(index=False):
        lookup[(int(row.row_index), float(row.target_density))] = row._asdict()

    for density in densities:
        subset = morph_data[np.isclose(morph_data["target_density"], float(density))].copy()
        train = subset[subset["row_index"].isin(train_set)].copy()
        cols = morphology_feature_columns(subset)
        columns[float(density)] = cols
        model = RandomForestClassifier(
            n_estimators=args.morph_estimators,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(train[cols], train["failureType"].astype(str))
        models[float(density)] = model
    return models, columns, lookup


def predict_morphology(
    row_index: int,
    density: float,
    models: dict[float, RandomForestClassifier],
    columns: dict[float, list[str]],
    lookup: dict[tuple[int, float], dict[str, object]],
) -> tuple[str, float, float]:
    density = float(density)
    features = lookup.get((int(row_index), density))
    if features is None:
        return "", 0.0, 0.0
    frame = pd.DataFrame([features])
    cols = columns[density]
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = models[density]
    proba = model.predict_proba(x)[0]
    labels = np.asarray(model.classes_, dtype=str)
    best_idx = int(np.argmax(proba))
    scratch_idx = np.where(labels == "Scratch")[0]
    scratch_prob = float(proba[int(scratch_idx[0])]) if len(scratch_idx) else 0.0
    return str(labels[best_idx]), float(proba[best_idx]), scratch_prob


def scratch_chord_guard_mask(wafer_map: np.ndarray, existing_mask: np.ndarray, n_points: int) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    coords = np.column_stack(np.nonzero(valid & ~np.asarray(existing_mask, dtype=bool)))
    if n_points <= 0 or len(coords) == 0:
        return np.zeros_like(valid, dtype=bool)

    cy, cx = wafer_center(valid)
    yy = coords[:, 0].astype(float)
    xx = coords[:, 1].astype(float)
    dy = yy - cy
    dx = xx - cx
    valid_y, valid_x = np.nonzero(valid)
    max_radius = float(np.sqrt(((valid_y - cy) ** 2 + (valid_x - cx) ** 2).max()))
    max_radius = max(max_radius, 1.0)

    angles = np.deg2rad([0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0])
    offsets = np.asarray([-0.55, -0.32, -0.12, 0.12, 0.32, 0.55], dtype=float)
    distances = []
    for theta in angles:
        signed_distance = (dx * np.sin(theta) - dy * np.cos(theta)) / max_radius
        distances.append(np.min(np.abs(signed_distance[:, None] - offsets[None, :]), axis=1))
    min_distance = np.min(np.vstack(distances), axis=0)
    line_score = np.exp(-(min_distance**2) / (2.0 * 0.028**2))
    edge_bonus = 0.15 * (np.sqrt(dx**2 + dy**2) / max_radius > 0.72).astype(float)
    return gate_refinement.diverse_select_by_score(
        wafer_map,
        existing_mask,
        line_score + edge_bonus,
        n_points,
        diversity_weight=0.9,
    )


def select_ensemble_followup(
    point_model,
    cnn_model,
    wafer_map: np.ndarray,
    observed_mask: np.ndarray,
    density: float,
    top_k: int,
    noncnn_weight: float,
    device: torch.device,
    row_index: int,
    failure_type: str,
) -> np.ndarray:
    candidates, scores, _, _ = gate_refinement.ensemble_candidate_scores(
        point_model,
        cnn_model,
        wafer_map,
        observed_mask,
        float(density),
        float(noncnn_weight),
        device,
        int(row_index),
        str(failure_type),
    )
    return gate_refinement.topk_from_candidate_scores(wafer_map, candidates, scores, int(top_k))


def sample_with_scratch_guard(
    point_model,
    cnn_model,
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    density: float,
    top_k: int,
    guard_count: int,
    noncnn_weight: float,
    device: torch.device,
    row_index: int,
    failure_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    guard = scratch_chord_guard_mask(wafer_map, first_mask, int(guard_count))
    observed = first_mask | guard
    exploit = select_ensemble_followup(
        point_model,
        cnn_model,
        wafer_map,
        observed,
        density,
        max(0, int(top_k) - int(guard_count)),
        noncnn_weight,
        device,
        row_index,
        failure_type,
    )
    return observed | exploit, guard


def policy_record(
    row_index: int,
    failure_type: str,
    density: float,
    policy_name: str,
    route_source: str,
    guard_count: int,
    routed_to_guard: bool,
    morph_top1: str,
    morph_confidence: float,
    morph_scratch_prob: float,
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    sample_mask: np.ndarray,
    guard_mask: np.ndarray,
    coverage_metrics: dict[str, object],
) -> dict[str, object]:
    metrics = sampling_metrics(wafer_map, sample_mask)
    followup_mask = np.asarray(sample_mask, dtype=bool) & ~np.asarray(first_mask, dtype=bool)
    followup_metrics = sampling_metrics(wafer_map, followup_mask)
    guard_metrics = sampling_metrics(wafer_map, guard_mask)
    followup_valid = int(followup_metrics["sampled_valid_count"])
    followup_defects = int(followup_metrics["sampled_defects"])
    precision = followup_defects / followup_valid if followup_valid else float("nan")
    coverage = float(metrics["defect_coverage"])
    base_coverage = float(coverage_metrics["defect_coverage"])
    gain = 100.0 * (coverage - base_coverage) / base_coverage if base_coverage > 0 else float("nan")
    return {
        "row_index": int(row_index),
        "failureType": str(failure_type),
        "target_density": float(density),
        "top_k": int(followup_valid),
        "policy_name": policy_name,
        "route_source": route_source,
        "guard_count": int(guard_count),
        "routed_to_guard": int(routed_to_guard),
        "morph_top1": morph_top1,
        "morph_confidence": float(morph_confidence),
        "morph_scratch_prob": float(morph_scratch_prob),
        "morph_top1_correct": int(str(morph_top1) == str(failure_type)),
        "scratch_predicted": int(str(morph_top1) == "Scratch"),
        "scratch_actual": int(str(failure_type) == "Scratch"),
        "guard_valid_count": int(guard_metrics["sampled_valid_count"]),
        "guard_defects": int(guard_metrics["sampled_defects"]),
        "sampled_valid_count": int(metrics["sampled_valid_count"]),
        "sampled_defects": int(metrics["sampled_defects"]),
        "followup_valid_count": followup_valid,
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
        rows.groupby(["target_density", "policy_name", "route_source", "guard_count"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            route_rate=("routed_to_guard", "mean"),
            mean_guard_defects=("guard_defects", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            morph_top1_accuracy=("morph_top1_correct", "mean"),
        )
        .reset_index()
    )
    pattern = (
        rows.groupby(["target_density", "failureType", "policy_name", "route_source", "guard_count"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            route_rate=("routed_to_guard", "mean"),
            mean_guard_defects=("guard_defects", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    morph = (
        rows[rows["policy_name"] == "ensemble_top32"]
        .groupby("target_density", dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            morph_top1_accuracy=("morph_top1_correct", "mean"),
            scratch_actual_rate=("scratch_actual", "mean"),
            scratch_predicted_rate=("scratch_predicted", "mean"),
            scratch_precision=("scratch_actual", lambda s: float("nan")),
        )
        .reset_index()
    )
    morph_rows = []
    base = rows[rows["policy_name"] == "ensemble_top32"].copy()
    for density, group in base.groupby("target_density", dropna=False):
        predicted = group["scratch_predicted"].astype(bool)
        actual = group["scratch_actual"].astype(bool)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        morph_rows.append(
            {
                "target_density": float(density),
                "wafers": int(group["row_index"].nunique()),
                "morph_top1_accuracy": float(group["morph_top1_correct"].mean()),
                "scratch_actual_wafers": int(actual.sum()),
                "scratch_predicted_wafers": int(predicted.sum()),
                "scratch_precision": tp / max(tp + fp, 1),
                "scratch_recall": tp / max(tp + fn, 1),
            }
        )
    morph = pd.DataFrame(morph_rows)
    return summary, pattern, morph


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


def plot_outputs(summary: pd.DataFrame, pattern: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    scratch = pattern[pattern["failureType"] == "Scratch"].copy()
    if not scratch.empty:
        plt.figure(figsize=(13.5, 5.5))
        sns.lineplot(
            data=scratch,
            x="guard_count",
            y="mean_defect_coverage",
            hue="policy_name",
            style="target_density",
            markers=True,
            dashes=False,
        )
        plt.title("Scratch defect coverage by scratch-guard allocation")
        plt.xlabel("Scratch guard points within Top32")
        plt.ylabel("Mean defect coverage")
        plt.grid(alpha=0.25)
        plt.legend(
            title="policy_name / target_density",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=8,
        )
        plt.tight_layout()
        plt.savefig(fig_dir / "scratch_coverage_by_guard_count.png", dpi=180, bbox_inches="tight")
        plt.close()

    focus = summary[summary["policy_name"].isin(["ensemble_top32", "scratch_guard_all_8", "oracle_scratch_route_8", "predicted_scratch_top1_route_8"])].copy()
    if not focus.empty:
        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=focus, x="target_density", y="mean_followup_precision_at_k", hue="policy_name")
        plt.title("Global precision@32: scratch guard routing views")
        plt.xlabel("First-pass density")
        plt.ylabel("Follow-up precision@32")
        plt.tight_layout()
        plt.savefig(fig_dir / "global_precision_guard8.png", dpi=180)
        plt.close()


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    pattern: pd.DataFrame,
    morph: pd.DataFrame,
    out_path: Path,
) -> None:
    selected = summary[
        summary["policy_name"].isin(
            ["coverage32", "ensemble_top32", "scratch_guard_all_8", "oracle_scratch_route_8", "predicted_scratch_top1_route_8"]
        )
    ].copy()
    scratch = pattern[
        (pattern["failureType"] == "Scratch")
        & (
            pattern["policy_name"].isin(["coverage32", "ensemble_top32"])
            | pattern["policy_name"].str.contains("scratch", case=False, na=False)
        )
    ].copy()
    scratch = scratch.sort_values(["target_density", "guard_count", "policy_name"])
    lines = [
        "# Scratch Guard Policy v1",
        "",
        "Purpose: test whether reserving part of the Top32 follow-up budget for multi-angle Scratch discovery improves thin-line defect capture.",
        "",
        "Policies:",
        "",
        "- `ensemble_top32`: no fixed Scratch guard, all 32 points by ensemble risk.",
        "- `scratch_guard_all_N`: use N Scratch guard points on every wafer, then ensemble for 32-N points.",
        "- `oracle_scratch_route_N`: use Scratch guard only when the true label is Scratch. This is an upper bound, not deployable.",
        "- `predicted_scratch_top1_route_N`: use Scratch guard only when first-pass morphology classifier predicts Scratch.",
        "",
        "Why non-Scratch patterns are still reported:",
        "",
        "Pattern-specific routing is only safe if the pattern classifier is correct. Non-Scratch cost measures the penalty when Scratch guard is applied too broadly or when predicted routing is wrong.",
        "",
        "## Morphology Routing Quality",
        "",
        dataframe_to_markdown(morph.round(5)),
        "",
        "## Global Selected Summary",
        "",
        dataframe_to_markdown(selected.round(5)),
        "",
        "## Scratch Pattern Summary",
        "",
        dataframe_to_markdown(scratch.round(5)),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    patterned = pd.read_pickle(args.patterned)
    train_ids, test_ids = density_policy.split_wafers(patterned, args.test_size, args.seed)
    train_ids = sparse_cnn.limit_ids(patterned, train_ids, args.max_train_wafers, rng, args.seed)
    test_ids = sparse_cnn.limit_ids(patterned, test_ids, args.max_test_wafers, rng, args.seed + 1)
    train_ids, val_ids = sparse_cnn.validation_split(patterned, train_ids, args.val_size, args.seed)
    _ = val_ids

    cnn_model, checkpoint = gate_refinement.load_cnn_model(args.cnn_model, device)
    point_args = argparse.Namespace(
        seed=args.seed,
        max_train_wafers=0,
        max_defect_candidates=args.max_defect_candidates,
        max_normal_candidates=args.max_normal_candidates,
        point_estimators=args.point_estimators,
        n_jobs=args.n_jobs,
    )
    print(
        f"scratch guard train wafers={len(train_ids):,}, test wafers={len(test_ids):,}, "
        f"device={device}, CNN best_epoch={checkpoint.get('best_epoch', 'unknown')}"
    )
    point_train = density_policy.build_point_training_data(patterned, train_ids, args.densities, point_args)
    point_model = density_policy.train_point_model(point_train, point_args)

    morph_data = pd.read_csv(args.morph_dataset)
    morph_models, morph_columns, morph_lookup = train_morphology_models(morph_data, train_ids, args.densities, args)

    records: list[dict[str, object]] = []
    test_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]
    for pos, row in enumerate(test_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in args.densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(density))
            coverage_follow = make_coverage_sampling_mask(wafer_map, n_points=args.top_k, existing_mask=first_mask)
            coverage_mask = first_mask | coverage_follow
            coverage_metrics = sampling_metrics(wafer_map, coverage_mask)
            morph_top1, morph_confidence, scratch_prob = predict_morphology(
                row_index, float(density), morph_models, morph_columns, morph_lookup
            )
            ensemble_follow = select_ensemble_followup(
                point_model,
                cnn_model,
                wafer_map,
                first_mask,
                float(density),
                args.top_k,
                args.ensemble_noncnn_weight,
                device,
                row_index,
                failure_type,
            )
            ensemble_mask = first_mask | ensemble_follow
            zero_guard = np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
            records.append(
                policy_record(
                    row_index,
                    failure_type,
                    float(density),
                    "coverage32",
                    "baseline",
                    0,
                    False,
                    morph_top1,
                    morph_confidence,
                    scratch_prob,
                    wafer_map,
                    first_mask,
                    coverage_mask,
                    zero_guard,
                    coverage_metrics,
                )
            )
            records.append(
                policy_record(
                    row_index,
                    failure_type,
                    float(density),
                    "ensemble_top32",
                    "baseline",
                    0,
                    False,
                    morph_top1,
                    morph_confidence,
                    scratch_prob,
                    wafer_map,
                    first_mask,
                    ensemble_mask,
                    zero_guard,
                    coverage_metrics,
                )
            )
            for guard_count in args.guard_counts:
                guard_count = int(guard_count)
                guard_mask, guard_only = sample_with_scratch_guard(
                    point_model,
                    cnn_model,
                    wafer_map,
                    first_mask,
                    float(density),
                    args.top_k,
                    guard_count,
                    args.ensemble_noncnn_weight,
                    device,
                    row_index,
                    failure_type,
                )
                policy_specs = [
                    (f"scratch_guard_all_{guard_count}", "all", True, guard_mask, guard_only),
                    (
                        f"oracle_scratch_route_{guard_count}",
                        "oracle_true_scratch",
                        failure_type == "Scratch",
                        guard_mask if failure_type == "Scratch" else ensemble_mask,
                        guard_only if failure_type == "Scratch" else zero_guard,
                    ),
                    (
                        f"predicted_scratch_top1_route_{guard_count}",
                        "predicted_top1_scratch",
                        morph_top1 == "Scratch",
                        guard_mask if morph_top1 == "Scratch" else ensemble_mask,
                        guard_only if morph_top1 == "Scratch" else zero_guard,
                    ),
                ]
                for policy_name, route_source, routed, sample_mask, used_guard in policy_specs:
                    records.append(
                        policy_record(
                            row_index,
                            failure_type,
                            float(density),
                            policy_name,
                            route_source,
                            guard_count,
                            bool(routed),
                            morph_top1,
                            morph_confidence,
                            scratch_prob,
                            wafer_map,
                            first_mask,
                            sample_mask,
                            used_guard,
                            coverage_metrics,
                        )
                    )
        if pos % 25 == 0 or pos == len(test_df):
            print(f"scratch guard wafers evaluated: {pos:,}/{len(test_df):,}")

    rows = pd.DataFrame.from_records(records)
    summary, pattern, morph = summarize(rows)
    rows.to_csv(args.out_dir / "scratch_guard_rows.csv", index=False)
    summary.to_csv(args.out_dir / "scratch_guard_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "scratch_guard_pattern_summary.csv", index=False)
    morph.to_csv(args.out_dir / "scratch_guard_morphology_routing.csv", index=False)
    plot_outputs(summary, pattern, args.fig_dir)
    write_report(args, summary, pattern, morph, args.out_dir / "scratch_guard_policy_report.md")

    print(f"wrote scratch guard outputs to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")
    print("Morphology routing:")
    print(morph.round(5).to_string(index=False))
    print("Selected global summary:")
    selected = summary[
        summary["policy_name"].isin(
            ["coverage32", "ensemble_top32", "scratch_guard_all_8", "oracle_scratch_route_8", "predicted_scratch_top1_route_8"]
        )
    ]
    print(selected.round(5).to_string(index=False))
    print("Scratch summary:")
    scratch = pattern[pattern["failureType"] == "Scratch"].sort_values(["target_density", "guard_count", "policy_name"])
    print(scratch.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
