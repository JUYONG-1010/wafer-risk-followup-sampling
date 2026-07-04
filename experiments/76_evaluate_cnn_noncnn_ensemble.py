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
from src.sampling import make_coverage_sampling_mask, sampling_metrics, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_CNN_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("data") / "processed" / "cnn_noncnn_ensemble_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "76_cnn_noncnn_ensemble_v1"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune CNN/non-CNN ensemble weights on validation split and evaluate on test split."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Weight on non-CNN score. 0 means CNN-only; 1 means non-CNN-only.",
    )
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
    return parser.parse_args()


def load_cnn_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden = int(checkpoint.get("args", {}).get("hidden_channels", 32))
    model = sparse_cnn.SparseRiskCNN(in_channels=7, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def cnn_scores_for_candidates(
    model,
    wafer_map: np.ndarray,
    density: float,
    candidates: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    x, _, _, _ = sparse_cnn.make_tensors(wafer_map, density, device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    yy = candidates["candidate_y"].astype(int).to_numpy()
    xx = candidates["candidate_x"].astype(int).to_numpy()
    return scores[yy, xx].astype(float)


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    if len(scores) == 1:
        return np.ones_like(scores)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores), dtype=float)
    return ranks / float(len(scores) - 1)


def selected_mask_from_scores(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidate_y: np.ndarray,
    candidate_x: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    selected = np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    if len(scores) == 0 or top_k <= 0:
        return selected
    order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    yy = candidate_y[order].astype(int)
    xx = candidate_x[order].astype(int)
    selected[yy, xx] = True
    return selected & ~first_mask & valid_die_mask(wafer_map)


def policy_record(
    split: str,
    row_index: int,
    failure_type: str,
    density: float,
    top_k: int,
    policy_name: str,
    ensemble_mode: str,
    noncnn_weight: float,
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
    precision_at_k = followup_defects / followup_valid_count if followup_valid_count else float("nan")
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
        "ensemble_mode": ensemble_mode,
        "noncnn_weight": float(noncnn_weight),
        "sampled_valid_count": int(metrics["sampled_valid_count"]),
        "sampled_defects": int(metrics["sampled_defects"]),
        "followup_valid_count": followup_valid_count,
        "followup_defects": followup_defects,
        "followup_precision_at_k": precision_at_k,
        "defect_coverage": coverage,
        "coverage_defect_coverage": base_coverage,
        "defect_coverage_gain_pct": gain,
        "absolute_error": float(metrics["absolute_error"]),
        "coverage_absolute_error": float(coverage_metrics["absolute_error"]),
        "absolute_error_delta": float(metrics["absolute_error"]) - float(coverage_metrics["absolute_error"]),
        "severe_miss": int(metrics["severe_miss"]),
    }


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["split", "target_density", "top_k", "policy_name", "ensemble_mode", "noncnn_weight"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
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
            ["split", "target_density", "failureType", "top_k", "policy_name", "ensemble_mode", "noncnn_weight"],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    return summary, pattern


def select_weights(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = summary[
        (summary["split"] == "val")
        & (summary["ensemble_mode"].isin(["raw", "rank"]))
    ].copy()
    global_scores = (
        candidates.groupby(["ensemble_mode", "noncnn_weight"], dropna=False)
        .agg(
            mean_precision=("mean_followup_precision_at_k", "mean"),
            mean_coverage=("mean_defect_coverage", "mean"),
            mean_abs_delta=("mean_absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
        )
        .reset_index()
        .sort_values(["mean_precision", "mean_coverage"], ascending=[False, False])
    )
    density_scores = (
        candidates.groupby(["target_density", "ensemble_mode", "noncnn_weight"], dropna=False)
        .agg(
            mean_precision=("mean_followup_precision_at_k", "mean"),
            mean_coverage=("mean_defect_coverage", "mean"),
            mean_abs_delta=("mean_absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss_rate", "mean"),
        )
        .reset_index()
        .sort_values(["target_density", "mean_precision", "mean_coverage"], ascending=[True, False, False])
    )
    global_best = global_scores.head(1).copy()
    density_best = density_scores.groupby("target_density", as_index=False).head(1).copy()
    return global_best, density_best


def filter_selected_test(summary: pd.DataFrame, global_best: pd.DataFrame, density_best: pd.DataFrame) -> pd.DataFrame:
    test = summary[summary["split"] == "test"].copy()
    base = test[test["policy_name"].isin(["coverage", "noncnn", "cnn"])].copy()
    base["selection"] = "baseline"

    selected_parts = [base]
    if not global_best.empty:
        row = global_best.iloc[0]
        global_sel = test[
            (test["ensemble_mode"] == row["ensemble_mode"])
            & np.isclose(test["noncnn_weight"], float(row["noncnn_weight"]))
        ].copy()
        global_sel["selection"] = "val_global_best"
        selected_parts.append(global_sel)

    density_rows = []
    for row in density_best.itertuples(index=False):
        selected = test[
            (np.isclose(test["target_density"], float(row.target_density)))
            & (test["ensemble_mode"] == row.ensemble_mode)
            & np.isclose(test["noncnn_weight"], float(row.noncnn_weight))
        ].copy()
        selected["selection"] = "val_density_best"
        density_rows.append(selected)
    if density_rows:
        selected_parts.append(pd.concat(density_rows, ignore_index=True))
    return pd.concat(selected_parts, ignore_index=True)


def evaluate_split(
    split_name: str,
    patterned: pd.DataFrame,
    ids: np.ndarray,
    densities: list[float],
    weights: list[float],
    top_k: int,
    point_model,
    cnn_model,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    df = patterned[patterned.index.isin(set(int(v) for v in ids))]
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(density))
            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=density_policy.density_key(float(density)),
                first_mask=first_mask,
                row_index=row_index,
                failure_type=failure_type,
                include_label=True,
            )
            if candidates.empty:
                continue
            yy = candidates["candidate_y"].astype(int).to_numpy()
            xx = candidates["candidate_x"].astype(int).to_numpy()
            noncnn_scores = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1].astype(float)
            cnn_scores = cnn_scores_for_candidates(cnn_model, wafer_map, float(density), candidates, device)
            noncnn_rank = percentile_scores(noncnn_scores)
            cnn_rank = percentile_scores(cnn_scores)

            coverage_follow = make_coverage_sampling_mask(wafer_map, n_points=top_k, existing_mask=first_mask)
            coverage_mask = first_mask | coverage_follow
            coverage_metrics = sampling_metrics(wafer_map, coverage_mask)
            base_scores = [
                ("coverage", "baseline", np.nan, None),
                ("noncnn", "baseline", 1.0, noncnn_scores),
                ("cnn", "baseline", 0.0, cnn_scores),
            ]
            for policy_name, mode, weight, scores in base_scores:
                if scores is None:
                    sample_mask = coverage_mask
                else:
                    selected = selected_mask_from_scores(wafer_map, first_mask, yy, xx, scores, top_k)
                    sample_mask = first_mask | selected
                records.append(
                    policy_record(
                        split_name,
                        row_index,
                        failure_type,
                        float(density),
                        top_k,
                        policy_name,
                        mode,
                        weight,
                        wafer_map,
                        first_mask,
                        sample_mask,
                        coverage_metrics,
                    )
                )

            for weight in weights:
                raw_scores = weight * noncnn_scores + (1.0 - weight) * cnn_scores
                rank_scores = weight * noncnn_rank + (1.0 - weight) * cnn_rank
                for mode, scores in [("raw", raw_scores), ("rank", rank_scores)]:
                    selected = selected_mask_from_scores(wafer_map, first_mask, yy, xx, scores, top_k)
                    records.append(
                        policy_record(
                            split_name,
                            row_index,
                            failure_type,
                            float(density),
                            top_k,
                            f"ensemble_{mode}_w{weight:.2f}",
                            mode,
                            weight,
                            wafer_map,
                            first_mask,
                            first_mask | selected,
                            coverage_metrics,
                        )
                    )
        if pos % 50 == 0 or pos == len(df):
            print(f"{split_name} ensemble wafers evaluated: {pos:,}/{len(df):,}")
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


def plot_outputs(summary: pd.DataFrame, selected: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    val_ens = summary[(summary["split"] == "val") & (summary["ensemble_mode"].isin(["raw", "rank"]))].copy()
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=val_ens,
        x="noncnn_weight",
        y="mean_followup_precision_at_k",
        hue="ensemble_mode",
        style="target_density",
        markers=True,
        dashes=False,
    )
    plt.title("Validation precision@K by ensemble weight")
    plt.xlabel("non-CNN weight")
    plt.ylabel("Follow-up precision@K")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "validation_precision_by_weight.png", dpi=180)
    plt.close()

    test_selected = selected[selected["split"] == "test"].copy()
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=test_selected,
        x="target_density",
        y="mean_followup_precision_at_k",
        hue="selection",
    )
    plt.title("Test precision@K for selected policies")
    plt.xlabel("First-pass density")
    plt.ylabel("Follow-up precision@K")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_selected_precision.png", dpi=180)
    plt.close()


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    pattern: pd.DataFrame,
    global_best: pd.DataFrame,
    density_best: pd.DataFrame,
    selected: pd.DataFrame,
    train_count: int,
    val_count: int,
    test_count: int,
    checkpoint: dict,
) -> None:
    test_focus = selected[
        selected["policy_name"].isin(["coverage", "noncnn", "cnn"])
        | selected["selection"].isin(["val_global_best", "val_density_best"])
    ].copy()
    scratch = pattern[
        (pattern["split"] == "test")
        & (pattern["failureType"].isin(["Scratch", "Loc", "Edge-Ring", "Center"]))
        & (
            pattern["policy_name"].isin(["coverage", "noncnn", "cnn"])
            | pattern["policy_name"].isin(set(test_focus["policy_name"]))
        )
    ].copy()
    lines = [
        "# CNN + Non-CNN Ensemble Weight Sweep v1",
        "",
        "Purpose: tune ensemble weights on validation wafers and evaluate the selected weights on held-out test wafers.",
        "",
        "Score definitions:",
        "",
        "- raw: `w * nonCNN_probability + (1 - w) * CNN_probability`",
        "- rank: `w * nonCNN_percentile_rank + (1 - w) * CNN_percentile_rank`",
        "",
        "Setup:",
        "",
        f"- train wafers after validation split: {train_count}",
        f"- validation wafers: {val_count}",
        f"- test wafers: {test_count}",
        f"- top-k: {args.top_k}",
        f"- weights: {', '.join(f'{value:.1f}' for value in args.weights)}",
        f"- CNN checkpoint best epoch: {checkpoint.get('best_epoch', 'unknown')}",
        "",
        "## Validation-Selected Global Best",
        "",
        dataframe_to_markdown(global_best.round(5)),
        "",
        "## Validation-Selected Density-Specific Best",
        "",
        dataframe_to_markdown(density_best.round(5)),
        "",
        "## Test Summary For Baselines And Selected Ensembles",
        "",
        dataframe_to_markdown(test_focus.round(5)),
        "",
        "## Test Pattern Focus",
        "",
        dataframe_to_markdown(scratch.round(5)),
        "",
    ]
    (args.out_dir / "cnn_noncnn_ensemble_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    print(f"ensemble train wafers={len(train_ids):,}, val wafers={len(val_ids):,}, test wafers={len(test_ids):,}")
    point_train = density_policy.build_point_training_data(patterned, train_ids, args.densities, point_args)
    point_model = density_policy.train_point_model(point_train, point_args)

    weights = sorted(set(float(value) for value in args.weights))
    val_rows = evaluate_split(
        "val", patterned, val_ids, args.densities, weights, args.top_k, point_model, cnn_model, device
    )
    test_rows = evaluate_split(
        "test", patterned, test_ids, args.densities, weights, args.top_k, point_model, cnn_model, device
    )
    rows = pd.concat([val_rows, test_rows], ignore_index=True)
    summary, pattern = summarize(rows)
    global_best, density_best = select_weights(summary)
    selected = filter_selected_test(summary, global_best, density_best)

    rows.to_csv(args.out_dir / "cnn_noncnn_ensemble_rows.csv", index=False)
    summary.to_csv(args.out_dir / "cnn_noncnn_ensemble_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "cnn_noncnn_ensemble_pattern_summary.csv", index=False)
    global_best.to_csv(args.out_dir / "validation_global_best_weight.csv", index=False)
    density_best.to_csv(args.out_dir / "validation_density_best_weights.csv", index=False)
    selected.to_csv(args.out_dir / "test_selected_ensemble_summary.csv", index=False)
    pd.DataFrame({"train_ids": pd.Series(train_ids), "val_ids": pd.Series(val_ids), "test_ids": pd.Series(test_ids)}).to_csv(
        args.out_dir / "cnn_noncnn_ensemble_split_ids.csv",
        index=False,
    )
    plot_outputs(summary, selected, args.fig_dir)
    write_report(
        args,
        summary,
        pattern,
        global_best,
        density_best,
        selected,
        len(train_ids),
        len(val_ids),
        len(test_ids),
        checkpoint,
    )
    print(f"wrote ensemble outputs to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")
    print("Validation global best:")
    print(global_best.round(5).to_string(index=False))
    print("Validation density best:")
    print(density_best.round(5).to_string(index=False))
    print("Test selected:")
    print(selected.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
