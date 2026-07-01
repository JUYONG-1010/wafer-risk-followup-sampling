from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import defect_mask, make_coverage_sampling_mask, sampling_metrics, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_CNN_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("data") / "processed" / "sparse_cnn_matched_split_comparison_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "60_sparse_cnn_matched_split_comparison_v1"


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
        description="Matched-split comparison between sparse CNN and non-CNN point-risk model."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=1600)
    parser.add_argument("--max-test-wafers", type=int, default=300)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=10)
    return parser.parse_args()


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if int(labels.sum()) == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def top_fraction_iou(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    if len(labels) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(labels) * fraction)))
    selected = np.zeros(len(labels), dtype=bool)
    selected[np.argsort(scores)[-k:]] = True
    truth = labels.astype(bool)
    union = selected | truth
    if not union.any():
        return float("nan")
    return float((selected & truth).sum() / union.sum())


def selected_mask_from_scores(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidate_y: np.ndarray,
    candidate_x: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    selected = np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    if len(scores) == 0:
        return selected
    k = min(top_k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    yy = candidate_y[order].astype(int)
    xx = candidate_x[order].astype(int)
    selected[yy, xx] = True
    return selected & ~first_mask & valid_die_mask(wafer_map)


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


def policy_record(
    row_index: int,
    failure_type: str,
    density: float,
    policy_name: str,
    wafer_map: np.ndarray,
    mask: np.ndarray,
    coverage_metrics: dict[str, object],
) -> dict[str, object]:
    metrics = sampling_metrics(wafer_map, mask)
    coverage = float(metrics["defect_coverage"])
    base_coverage = float(coverage_metrics["defect_coverage"])
    if base_coverage > 0:
        gain_pct = 100.0 * (coverage - base_coverage) / base_coverage
    else:
        gain_pct = float("nan")
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "policy_name": policy_name,
        "sampled_valid_count": int(metrics["sampled_valid_count"]),
        "sampled_defects": int(metrics["sampled_defects"]),
        "actual_defect_ratio": float(metrics["actual_defect_ratio"]),
        "sampled_defect_ratio": float(metrics["sampled_defect_ratio"]),
        "absolute_error": float(metrics["absolute_error"]),
        "defect_coverage": coverage,
        "severe_miss": int(metrics["severe_miss"]),
        "coverage32_defect_coverage": base_coverage,
        "defect_coverage_gain_pct": gain_pct,
        "absolute_error_delta": float(metrics["absolute_error"]) - float(coverage_metrics["absolute_error"]),
        "sampled_defects_delta": int(metrics["sampled_defects"]) - int(coverage_metrics["sampled_defects"]),
    }


def risk_record(
    row_index: int,
    failure_type: str,
    density: float,
    risk_map: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "risk_map": risk_map,
        "candidate_count": int(len(labels)),
        "candidate_defects": int(labels.sum()),
        "roc_auc": safe_auc(labels, scores),
        "average_precision": safe_ap(labels, scores),
        "top10pct_iou": top_fraction_iou(labels, scores, 0.10),
    }


def summarize_policy(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["target_density", "policy_name"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            mean_sampled_defects_delta=("sampled_defects_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    return summary


def summarize_risk(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["target_density", "risk_map"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top10pct_iou=("top10pct_iou", "mean"),
            mean_candidate_defects=("candidate_defects", "mean"),
        )
        .reset_index()
    )
    return summary


def summarize_pattern(rows: pd.DataFrame, value_cols: list[str], group_col: str) -> pd.DataFrame:
    return (
        rows.groupby(["target_density", "failureType", group_col], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            **{f"mean_{col}": (col, "mean") for col in value_cols},
        )
        .reset_index()
        .sort_values(["target_density", "failureType", group_col])
    )


def plot_policy_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for policy, group in summary.groupby("policy_name"):
        axes[0].plot(
            group["target_density"],
            group["mean_defect_coverage_gain_pct"],
            marker="o",
            label=policy,
        )
        axes[1].plot(
            group["target_density"],
            group["mean_absolute_error_delta"],
            marker="o",
            label=policy,
        )
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_title("Top32 coverage gain vs coverage32")
    axes[0].set_ylabel("Gain (%)")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_title("Absolute error delta vs coverage32")
    axes[1].set_ylabel("Delta")
    for ax in axes:
        ax.set_xlabel("Initial probe density")
        ax.set_xticks(sorted(summary["target_density"].unique()))
        ax.set_xticklabels([f"{value:.0%}" for value in sorted(summary["target_density"].unique())])
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(fig_dir / "matched_policy_gain_vs_bias.png", dpi=180)
    plt.close(fig)


def plot_risk_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    metrics = [
        ("mean_roc_auc", "ROC-AUC"),
        ("mean_average_precision", "Average Precision"),
        ("mean_top10pct_iou", "Top-10% IoU"),
    ]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for risk_map, group in summary.groupby("risk_map"):
            ax.plot(group["target_density"], group[metric], marker="o", label=risk_map)
        ax.set_title(title)
        ax.set_xlabel("Initial probe density")
        ax.set_xticks(sorted(summary["target_density"].unique()))
        ax.set_xticklabels([f"{value:.0%}" for value in sorted(summary["target_density"].unique())])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric")
    axes[-1].legend(fontsize=8)
    fig.savefig(fig_dir / "matched_risk_map_metrics.png", dpi=180)
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    policy_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    train_count: int,
    test_count: int,
    cnn_checkpoint: dict,
) -> None:
    lines = [
        "# Sparse CNN Matched-Split Comparison",
        "",
        "This report compares the saved sparse CNN against a newly trained non-CNN point-risk model on the same wafer split.",
        "",
        "## Split And Training Setup",
        "",
        f"- seed: {args.seed}",
        f"- train wafer cap before validation split: {args.max_train_wafers}",
        f"- validation size removed from CNN/non-CNN training ids: {args.val_size:.2f}",
        f"- final train wafers used for non-CNN point model: {train_count}",
        f"- test wafers: {test_count}",
        f"- densities: {', '.join(f'{v:.0%}' for v in args.densities)}",
        f"- top-k follow-up: {args.top_k}",
        f"- loaded CNN model: `{args.cnn_model}`",
        f"- CNN checkpoint best epoch: {cnn_checkpoint.get('best_epoch', 'unknown')}",
        "",
        "## Policy Summary",
        "",
        policy_summary.round(5).to_string(index=False),
        "",
        "## Risk-Map Ranking Summary",
        "",
        risk_summary.round(5).to_string(index=False),
        "",
        "## Interpretation Notes",
        "",
        "- `coverage32` is the same geometry-only representative baseline.",
        "- `noncnn_ml_rank32` ranks unmeasured dies using the tabular point-risk RandomForest model.",
        "- `sparse_cnn_top32` ranks unmeasured dies using the sparse CNN risk map.",
        "- `absolute_error_delta` is a representativeness warning, not the main discovery objective.",
        "- If CNN beats non-CNN on AP/ROC-AUC/Top-10% IoU and top32 coverage gain on this matched split, CNN can be treated as the stronger risk-map branch, subject to minority-pattern checks.",
        "",
    ]
    (args.out_dir / "sparse_cnn_matched_split_comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


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

    cnn_model, cnn_checkpoint = load_cnn_model(args.cnn_model, device)
    point_args = argparse.Namespace(
        seed=args.seed,
        max_train_wafers=0,
        max_defect_candidates=args.max_defect_candidates,
        max_normal_candidates=args.max_normal_candidates,
        point_estimators=args.point_estimators,
        n_jobs=args.n_jobs,
    )
    print(f"matched split train wafers={len(train_ids):,}, val wafers={len(val_ids):,}, test wafers={len(test_ids):,}")
    point_train = density_policy.build_point_training_data(patterned, train_ids, args.densities, point_args)
    point_model = density_policy.train_point_model(point_train, point_args)

    policy_records: list[dict[str, object]] = []
    risk_records: list[dict[str, object]] = []
    test_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]
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
            labels = candidates["label_candidate_is_defect"].astype(int).to_numpy()
            yy = candidates["candidate_y"].astype(int).to_numpy()
            xx = candidates["candidate_x"].astype(int).to_numpy()

            point_scores = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1].astype(float)
            cnn_scores = cnn_scores_for_candidates(cnn_model, wafer_map, float(density), candidates, device)

            coverage_follow = make_coverage_sampling_mask(
                wafer_map,
                n_points=args.top_k,
                existing_mask=first_mask,
            )
            coverage_mask = first_mask | coverage_follow
            coverage_metrics = sampling_metrics(wafer_map, coverage_mask)
            point_selected = selected_mask_from_scores(
                wafer_map, first_mask, yy, xx, point_scores, args.top_k
            )
            cnn_selected = selected_mask_from_scores(
                wafer_map, first_mask, yy, xx, cnn_scores, args.top_k
            )

            policy_records.append(
                policy_record(
                    row_index,
                    failure_type,
                    float(density),
                    "coverage32",
                    wafer_map,
                    coverage_mask,
                    coverage_metrics,
                )
            )
            policy_records.append(
                policy_record(
                    row_index,
                    failure_type,
                    float(density),
                    "noncnn_ml_rank32",
                    wafer_map,
                    first_mask | point_selected,
                    coverage_metrics,
                )
            )
            policy_records.append(
                policy_record(
                    row_index,
                    failure_type,
                    float(density),
                    "sparse_cnn_top32",
                    wafer_map,
                    first_mask | cnn_selected,
                    coverage_metrics,
                )
            )

            risk_records.append(risk_record(row_index, failure_type, float(density), "noncnn_point", labels, point_scores))
            risk_records.append(risk_record(row_index, failure_type, float(density), "sparse_cnn", labels, cnn_scores))
        if pos % 50 == 0 or pos == len(test_df):
            print(f"matched split wafers evaluated: {pos:,}/{len(test_df):,}")

    policy_rows = pd.DataFrame.from_records(policy_records)
    risk_rows = pd.DataFrame.from_records(risk_records)
    policy_summary = summarize_policy(policy_rows)
    risk_summary = summarize_risk(risk_rows)
    policy_pattern = summarize_pattern(
        policy_rows,
        ["defect_coverage", "defect_coverage_gain_pct", "absolute_error_delta", "sampled_defects_delta"],
        "policy_name",
    )
    risk_pattern = summarize_pattern(
        risk_rows,
        ["roc_auc", "average_precision", "top10pct_iou", "candidate_defects"],
        "risk_map",
    )

    policy_rows.to_csv(args.out_dir / "matched_policy_rows.csv", index=False)
    risk_rows.to_csv(args.out_dir / "matched_risk_rows.csv", index=False)
    policy_summary.to_csv(args.out_dir / "matched_policy_summary.csv", index=False)
    risk_summary.to_csv(args.out_dir / "matched_risk_summary.csv", index=False)
    policy_pattern.to_csv(args.out_dir / "matched_policy_pattern_summary.csv", index=False)
    risk_pattern.to_csv(args.out_dir / "matched_risk_pattern_summary.csv", index=False)
    pd.DataFrame({"train_ids": pd.Series(train_ids), "val_ids": pd.Series(val_ids), "test_ids": pd.Series(test_ids)}).to_csv(
        args.out_dir / "matched_split_ids.csv",
        index=False,
    )

    plot_policy_summary(policy_summary, args.fig_dir)
    plot_risk_summary(risk_summary, args.fig_dir)
    write_report(args, policy_summary, risk_summary, len(train_ids), len(test_ids), cnn_checkpoint)

    print(f"wrote matched comparison to {args.out_dir}")
    print(f"wrote matched figures to {args.fig_dir}")
    print(policy_summary.round(5).to_string(index=False))
    print(risk_summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
