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

from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    sampling_metrics,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("data") / "processed" / "low_evidence_discovery_probe_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "74_low_evidence_discovery_probe_v1"


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
        description="Evaluate low-evidence discovery probes before CNN follow-up exploitation."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hit-thresholds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--discovery-counts", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def load_cnn_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden = int(checkpoint.get("args", {}).get("hidden_channels", 32))
    model = sparse_cnn.SparseRiskCNN(in_channels=7, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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


def cnn_score_map(
    model,
    wafer_map: np.ndarray,
    observed_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = tensor_from_observed_mask(wafer_map, observed_mask, device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    candidates = valid_die_mask(wafer_map) & ~observed_mask
    scores = scores.copy()
    scores[~candidates] = np.nan
    return scores


def topk_from_scores(scores: np.ndarray, candidate_mask: np.ndarray, top_k: int) -> np.ndarray:
    selected = np.zeros_like(candidate_mask, dtype=bool)
    coords = np.column_stack(np.nonzero(candidate_mask))
    if len(coords) == 0 or top_k <= 0:
        return selected
    values = scores[coords[:, 0], coords[:, 1]]
    finite = np.isfinite(values)
    coords = coords[finite]
    values = values[finite]
    if len(coords) == 0:
        return selected
    chosen = coords[np.argsort(values)[::-1][: min(top_k, len(values))]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def line_sweep_discovery_mask(
    wafer_map: np.ndarray,
    existing_mask: np.ndarray,
    n_points: int,
) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    if n_points <= 0 or not valid.any():
        return np.zeros_like(valid, dtype=bool)

    existing = np.asarray(existing_mask, dtype=bool) & valid
    coords = np.column_stack(np.nonzero(valid & ~existing))
    if len(coords) == 0:
        return np.zeros_like(valid, dtype=bool)

    cy, cx = wafer_center(valid)
    yy = coords[:, 0].astype(float)
    xx = coords[:, 1].astype(float)
    dy = yy - cy
    dx = xx - cx
    scale = max(float(max(valid.shape)), 1.0)

    angles = np.deg2rad([0.0, 45.0, 90.0, 135.0])
    distances = []
    for theta in angles:
        # Distance from point to infinite line through wafer center.
        distances.append(np.abs(dx * np.sin(theta) - dy * np.cos(theta)) / scale)
    line_distance = np.min(np.vstack(distances), axis=0)
    line_score = np.exp(-(line_distance**2) / (2.0 * 0.035**2))

    selected_coords = np.column_stack(np.nonzero(existing)).astype(float)
    if len(selected_coords) == 0:
        selected_coords = np.array([[cy, cx]], dtype=float)
    cand = coords.astype(float)
    min_dist_sq = np.min(
        ((cand[:, None, :] - selected_coords[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )

    selected = np.zeros_like(valid, dtype=bool)
    available = np.ones(len(coords), dtype=bool)
    for _ in range(min(n_points, len(coords))):
        if not available.any():
            break
        max_dist = float(min_dist_sq[available].max())
        diversity = min_dist_sq / max(max_dist, 1.0)
        score = 1.6 * line_score + 1.0 * diversity
        score[~available] = -np.inf
        best = int(np.argmax(score))
        y, x = coords[best]
        selected[int(y), int(x)] = True
        available[best] = False
        new_dist_sq = ((cand - cand[best]) ** 2).sum(axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)
    return selected


def policy_row(
    row_index: int,
    failure_type: str,
    density: float,
    policy_name: str,
    first_hits: int,
    discovery_hits: int,
    discovery_count: int,
    threshold: int,
    wafer_map: np.ndarray,
    sample_mask: np.ndarray,
    coverage_metrics: dict[str, object],
) -> dict[str, object]:
    metrics = sampling_metrics(wafer_map, sample_mask)
    coverage = float(metrics["defect_coverage"])
    base_coverage = float(coverage_metrics["defect_coverage"])
    gain = 100.0 * (coverage - base_coverage) / base_coverage if base_coverage > 0 else float("nan")
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "policy_name": policy_name,
        "first_hit_count": first_hits,
        "low_evidence_triggered": int(first_hits <= threshold),
        "hit_threshold": threshold,
        "discovery_count": discovery_count,
        "discovery_hits": discovery_hits,
        "sampled_valid_count": int(metrics["sampled_valid_count"]),
        "sampled_defects": int(metrics["sampled_defects"]),
        "defect_coverage": coverage,
        "coverage32_defect_coverage": base_coverage,
        "defect_coverage_gain_pct": gain,
        "absolute_error": float(metrics["absolute_error"]),
        "absolute_error_delta": float(metrics["absolute_error"]) - float(coverage_metrics["absolute_error"]),
        "severe_miss": int(metrics["severe_miss"]),
    }


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["target_density", "policy_name", "hit_threshold", "discovery_count"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            trigger_rate=("low_evidence_triggered", "mean"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    pattern = (
        rows.groupby(
            ["target_density", "failureType", "policy_name", "hit_threshold", "discovery_count"],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            trigger_rate=("low_evidence_triggered", "mean"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    low_evidence = rows[rows["first_hit_count"] <= rows["hit_threshold"]].copy()
    low_summary = (
        low_evidence.groupby(
            ["target_density", "failureType", "policy_name", "hit_threshold", "discovery_count"],
            dropna=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_discovery_hits=("discovery_hits", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_defect_coverage_gain_pct=("defect_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    return summary, pattern, low_summary


def plot_outputs(summary: pd.DataFrame, low_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = summary[summary["policy_name"].isin(["coverage32", "cnn_top32", "gated_line8_cnn24", "gated_line16_cnn16"])].copy()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=focus, x="target_density", y="mean_defect_coverage", hue="policy_name", marker="o")
    plt.title("Low-evidence discovery gate: global defect coverage")
    plt.ylabel("Mean defect coverage")
    plt.tight_layout()
    plt.savefig(fig_dir / "global_defect_coverage.png", dpi=180)
    plt.close()

    scratch = low_summary[
        (low_summary["failureType"] == "Scratch")
        & (low_summary["policy_name"].isin(["coverage32", "cnn_top32", "gated_line8_cnn24", "gated_line16_cnn16"]))
    ].copy()
    if not scratch.empty:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=scratch, x="target_density", y="mean_defect_coverage", hue="policy_name", marker="o")
        plt.title("Scratch low-evidence subset defect coverage")
        plt.ylabel("Mean defect coverage")
        plt.tight_layout()
        plt.savefig(fig_dir / "scratch_low_evidence_defect_coverage.png", dpi=180)
        plt.close()


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


def write_report(args: argparse.Namespace, summary: pd.DataFrame, pattern: pd.DataFrame, low_summary: pd.DataFrame) -> None:
    lines = [
        "# Low-Evidence Discovery Probe v1",
        "",
        "Purpose: test whether wafers with too few first-pass defect hits should spend part of the limited follow-up budget on line-sensitive discovery before CNN risk exploitation.",
        "",
        "Policy definitions:",
        "",
        "- `coverage32`: first-pass + geometry-only 32-point follow-up",
        "- `cnn_top32`: first-pass + CNN top32 immediately",
        "- `gated_lineN_cnnM`: if first-pass hits are below the threshold, spend N points on line-sweep discovery, observe those points, then rescore CNN and use M remaining points",
        "",
        "Dense maps are used only for offline evaluation and for simulating observed labels after the discovery substep.",
        "",
    ]
    focus = summary[summary["policy_name"].isin(["coverage32", "cnn_top32", "gated_line8_cnn24", "gated_line16_cnn16"])].copy()
    lines.extend(["## Global Summary", "", dataframe_to_markdown(focus.round(5)), ""])

    scratch = low_summary[
        (low_summary["failureType"] == "Scratch")
        & (low_summary["policy_name"].isin(["coverage32", "cnn_top32", "gated_line4_cnn28", "gated_line8_cnn24", "gated_line12_cnn20", "gated_line16_cnn16"]))
    ].copy()
    lines.extend(["## Scratch Low-Evidence Subset", ""])
    if scratch.empty:
        lines.append("No Scratch rows met the low-evidence threshold.")
    else:
        lines.append(dataframe_to_markdown(scratch.round(5)))
    lines.append("")

    major = pattern[
        (pattern["target_density"].isin([0.03, 0.10]))
        & (pattern["policy_name"].isin(["coverage32", "cnn_top32", "gated_line8_cnn24", "gated_line16_cnn16"]))
    ].copy()
    lines.extend(["## Pattern Summary At 3% And 10%", "", dataframe_to_markdown(major.round(5)), ""])
    (args.out_dir / "low_evidence_discovery_probe_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    patterned = pd.read_pickle(args.patterned)
    _, test_ids = density_policy.split_wafers(patterned, args.test_size, args.seed)
    test_ids = sparse_cnn.limit_ids(patterned, test_ids, args.max_test_wafers, rng, args.seed + 1)
    test_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]
    model = load_cnn_model(args.model, device)

    records: list[dict[str, object]] = []
    for pos, row in enumerate(test_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        defects = defect_mask(wafer_map)
        for density in args.densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(density))
            first_hits = int((first_mask & defects).sum())
            coverage_follow = make_coverage_sampling_mask(wafer_map, n_points=args.top_k, existing_mask=first_mask)
            coverage_mask = first_mask | coverage_follow
            coverage_metrics = sampling_metrics(wafer_map, coverage_mask)

            first_scores = cnn_score_map(model, wafer_map, first_mask, device)
            cnn_follow = topk_from_scores(first_scores, valid_die_mask(wafer_map) & ~first_mask, args.top_k)
            records.append(
                policy_row(
                    row_index,
                    failure_type,
                    float(density),
                    "coverage32",
                    first_hits,
                    0,
                    0,
                    -1,
                    wafer_map,
                    coverage_mask,
                    coverage_metrics,
                )
            )
            records.append(
                policy_row(
                    row_index,
                    failure_type,
                    float(density),
                    "cnn_top32",
                    first_hits,
                    0,
                    0,
                    -1,
                    wafer_map,
                    first_mask | cnn_follow,
                    coverage_metrics,
                )
            )

            for threshold in args.hit_thresholds:
                for discovery_count in args.discovery_counts:
                    remaining = max(0, args.top_k - int(discovery_count))
                    if first_hits <= int(threshold):
                        discovery = line_sweep_discovery_mask(wafer_map, first_mask, int(discovery_count))
                        observed = first_mask | discovery
                        discovery_hits = int((discovery & defects).sum())
                        rescored = cnn_score_map(model, wafer_map, observed, device)
                        exploit = topk_from_scores(rescored, valid_die_mask(wafer_map) & ~observed, remaining)
                        sample = observed | exploit
                    else:
                        discovery = np.zeros_like(first_mask, dtype=bool)
                        discovery_hits = 0
                        sample = first_mask | cnn_follow
                    policy_name = f"gated_line{int(discovery_count)}_cnn{remaining}"
                    records.append(
                        policy_row(
                            row_index,
                            failure_type,
                            float(density),
                            policy_name,
                            first_hits,
                            discovery_hits,
                            int(discovery_count),
                            int(threshold),
                            wafer_map,
                            sample,
                            coverage_metrics,
                        )
                    )
        if pos % 50 == 0 or pos == len(test_df):
            print(f"low-evidence discovery wafers evaluated: {pos:,}/{len(test_df):,}")

    rows = pd.DataFrame.from_records(records)
    summary, pattern, low_summary = summarize(rows)
    rows.to_csv(args.out_dir / "low_evidence_discovery_rows.csv", index=False)
    summary.to_csv(args.out_dir / "low_evidence_discovery_summary.csv", index=False)
    pattern.to_csv(args.out_dir / "low_evidence_discovery_pattern_summary.csv", index=False)
    low_summary.to_csv(args.out_dir / "low_evidence_discovery_lowhit_pattern_summary.csv", index=False)
    plot_outputs(summary, low_summary, args.fig_dir)
    write_report(args, summary, pattern, low_summary)

    print(f"wrote low-evidence discovery outputs to {args.out_dir}")
    print(f"wrote figures to {args.fig_dir}")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
