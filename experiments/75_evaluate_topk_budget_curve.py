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
DEFAULT_OUT_DIR = Path("data") / "processed" / "topk_budget_curve_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "75_topk_budget_curve_v1"


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
        description="Evaluate follow-up sampling budget curves for coverage, non-CNN, and CNN policies."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[16, 32, 64, 128])
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
    row_index: int,
    failure_type: str,
    density: float,
    top_k: int,
    policy_name: str,
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
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "top_k": int(top_k),
        "policy_name": policy_name,
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
        rows.groupby(["target_density", "top_k", "policy_name"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            mean_followup_valid_count=("followup_valid_count", "mean"),
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
        rows.groupby(["target_density", "failureType", "top_k", "policy_name"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_sampled_defects=("sampled_defects", "mean"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_followup_precision_at_k=("followup_precision_at_k", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    return summary, pattern


def plot_outputs(summary: pd.DataFrame, pattern: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = summary.copy()
    plt.figure(figsize=(10, 5.5))
    sns.lineplot(
        data=focus,
        x="top_k",
        y="mean_defect_coverage",
        hue="policy_name",
        style="target_density",
        markers=True,
        dashes=False,
    )
    plt.title("Defect coverage vs follow-up budget K")
    plt.ylabel("Mean defect coverage")
    plt.xlabel("Follow-up budget K")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "topk_defect_coverage_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    sns.lineplot(
        data=focus,
        x="top_k",
        y="mean_followup_precision_at_k",
        hue="policy_name",
        style="target_density",
        markers=True,
        dashes=False,
    )
    plt.title("Follow-up precision@K")
    plt.ylabel("Actual defect ratio among selected follow-up points")
    plt.xlabel("Follow-up budget K")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "topk_followup_precision_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    sns.lineplot(
        data=focus[focus["policy_name"] != "coverage"],
        x="top_k",
        y="mean_defect_coverage_gain_pct",
        hue="policy_name",
        style="target_density",
        markers=True,
        dashes=False,
    )
    plt.axhline(0.0, color="#666666", linewidth=0.8)
    plt.title("Defect coverage gain vs geometry-only coverageK")
    plt.ylabel("Gain vs coverageK (%)")
    plt.xlabel("Follow-up budget K")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "topk_gain_vs_coverage.png", dpi=180)
    plt.close()

    scratch = pattern[pattern["failureType"].isin(["Scratch", "Loc", "Edge-Ring", "Center"])].copy()
    plt.figure(figsize=(11, 6))
    sns.relplot(
        data=scratch,
        x="top_k",
        y="mean_defect_coverage",
        hue="policy_name",
        col="failureType",
        row="target_density",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=2.4,
        aspect=1.25,
    )
    plt.savefig(fig_dir / "topk_pattern_focus_curve.png", dpi=180)
    plt.close("all")


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


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    pattern: pd.DataFrame,
    train_count: int,
    test_count: int,
    checkpoint: dict,
) -> None:
    lines = [
        "# Top-K Budget Curve v1",
        "",
        "Purpose: evaluate how follow-up recommendation quality changes when the limited sampling budget K changes.",
        "",
        "Policies:",
        "",
        "- `coverage`: geometry-only space-filling follow-up with the same K",
        "- `noncnn_ml_rank`: tabular point-risk model, top K by predicted defect probability",
        "- `sparse_cnn`: sparse CNN risk map, top K by predicted risk score",
        "",
        "Setup:",
        "",
        f"- train wafers after validation split: {train_count}",
        f"- test wafers: {test_count}",
        f"- densities: {', '.join(f'{value:.0%}' for value in args.densities)}",
        f"- K values: {', '.join(str(value) for value in args.top_ks)}",
        f"- CNN model: `{args.cnn_model}`",
        f"- CNN checkpoint best epoch: {checkpoint.get('best_epoch', 'unknown')}",
        "",
        "## Global Summary",
        "",
        dataframe_to_markdown(summary.round(5)),
        "",
        "## Scratch / Loc / Edge-Ring / Center Focus",
        "",
        dataframe_to_markdown(
            pattern[
                pattern["failureType"].isin(["Scratch", "Loc", "Edge-Ring", "Center"])
            ].round(5)
        ),
        "",
    ]
    (args.out_dir / "topk_budget_curve_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    print(f"top-k curve train wafers={len(train_ids):,}, val wafers={len(val_ids):,}, test wafers={len(test_ids):,}")
    point_train = density_policy.build_point_training_data(patterned, train_ids, args.densities, point_args)
    point_model = density_policy.train_point_model(point_train, point_args)

    records: list[dict[str, object]] = []
    test_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]
    top_ks = sorted(set(int(value) for value in args.top_ks))
    for pos, row in enumerate(test_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in args.densities:
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
            candidates = candidates.copy()
            yy = candidates["candidate_y"].astype(int).to_numpy()
            xx = candidates["candidate_x"].astype(int).to_numpy()
            point_scores = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1].astype(float)
            cnn_scores = cnn_scores_for_candidates(cnn_model, wafer_map, float(density), candidates, device)

            for top_k in top_ks:
                coverage_follow = make_coverage_sampling_mask(
                    wafer_map,
                    n_points=top_k,
                    existing_mask=first_mask,
                )
                coverage_mask = first_mask | coverage_follow
                coverage_metrics = sampling_metrics(wafer_map, coverage_mask)
                point_selected = selected_mask_from_scores(
                    wafer_map, first_mask, yy, xx, point_scores, top_k
                )
                cnn_selected = selected_mask_from_scores(
                    wafer_map, first_mask, yy, xx, cnn_scores, top_k
                )
                records.append(
                    policy_record(
                        row_index,
                        failure_type,
                        float(density),
                        top_k,
                        "coverage",
                        wafer_map,
                        first_mask,
                        coverage_mask,
                        coverage_metrics,
                    )
                )
                records.append(
                    policy_record(
                        row_index,
                        failure_type,
                        float(density),
                        top_k,
                        "noncnn_ml_rank",
                        wafer_map,
                        first_mask,
                        first_mask | point_selected,
                        coverage_metrics,
                    )
                )
                records.append(
                    policy_record(
                        row_index,
                        failure_type,
                        float(density),
                        top_k,
                        "sparse_cnn",
                        wafer_map,
                        first_mask,
                        first_mask | cnn_selected,
                        coverage_metrics,
                    )
                )
        if pos % 50 == 0 or pos == len(test_df):
            print(f"top-k curve wafers evaluated: {pos:,}/{len(test_df):,}")

    rows = pd.DataFrame.from_records(records)
    summary, pattern = summarize(rows)
    rows.to_csv(args.out_dir / "topk_budget_curve_rows.csv", index=False)
    summary.to_csv(args.out_dir / "topk_budget_curve_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "topk_budget_curve_pattern_summary.csv", index=False)
    pd.DataFrame({"train_ids": pd.Series(train_ids), "val_ids": pd.Series(val_ids), "test_ids": pd.Series(test_ids)}).to_csv(
        args.out_dir / "topk_budget_curve_split_ids.csv",
        index=False,
    )
    plot_outputs(summary, pattern, args.fig_dir)
    write_report(args, summary, pattern, len(train_ids), len(test_ids), checkpoint)
    print(f"wrote top-k budget curve outputs to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
