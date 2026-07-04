from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import defect_mask, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_EVAL_ROWS = (
    Path("data") / "processed" / "sparse_cnn_risk_map_v1_large" / "sparse_cnn_eval_rows.csv"
)
DEFAULT_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("outputs") / "figures" / "59_sparse_cnn_visual_examples_v1"
DEFAULT_REPORT_DIR = Path("reports") / "sparse_cnn_visual_examples_v1"
DEFAULT_PATTERNS = ["Edge-Ring", "Loc", "Scratch", "Donut", "Random"]


def load_script68():
    module_path = PROJECT_ROOT / "experiments" / "68_train_sparse_cnn_risk_map.py"
    spec = importlib.util.spec_from_file_location("sparse_cnn68", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sparse_cnn = load_script68()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse CNN visual examples as JPG files.")
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--eval-rows", type=Path, default=DEFAULT_EVAL_ROWS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    parser.add_argument("--density", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--examples-per-pattern", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def choose_examples(eval_rows: pd.DataFrame, patterns: list[str], density: float, per_pattern: int) -> pd.DataFrame:
    focus = eval_rows[np.isclose(eval_rows["target_density"], density)].copy()
    selected: list[pd.DataFrame] = []
    for pattern in patterns:
        subset = focus[focus["failureType"].astype(str) == pattern].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["top32_coverage_gain_pct", "average_precision"],
            ascending=[False, False],
        ).head(per_pattern)
        selected.append(subset)
    if not selected:
        return focus.sort_values("average_precision", ascending=False).head(len(patterns))
    return pd.concat(selected, ignore_index=True)


def wafer_display(wafer_map: np.ndarray) -> np.ndarray:
    arr = np.zeros_like(wafer_map, dtype=int)
    arr[valid_die_mask(wafer_map)] = 1
    arr[defect_mask(wafer_map)] = 2
    return arr


def first_pass_display(wafer_map: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    arr = np.zeros_like(wafer_map, dtype=int)
    arr[valid] = 1
    arr[first_mask & valid] = 2
    arr[first_mask & defects] = 3
    return arr


def score_wafer(model, wafer_map: np.ndarray, density: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x, _, _, first_mask = sparse_cnn.make_tensors(wafer_map, density, device)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    candidates = valid_die_mask(wafer_map) & ~first_mask
    scores = scores.copy()
    scores[~candidates] = np.nan
    return scores, first_mask


def topk_from_scores(scores: np.ndarray, candidate_mask: np.ndarray, top_k: int) -> np.ndarray:
    selected = np.zeros_like(candidate_mask, dtype=bool)
    coords = np.column_stack(np.nonzero(candidate_mask))
    if len(coords) == 0:
        return selected
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    values = scores[yy, xx]
    k = min(top_k, len(values))
    chosen = coords[np.argsort(values)[::-1][:k]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def overlay_points(ax, mask: np.ndarray, wafer_map: np.ndarray, marker: str, normal_color: str, defect_color: str, size: int) -> None:
    defects = defect_mask(wafer_map)
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return
    hit = defects[yy, xx]
    ax.scatter(xx[~hit], yy[~hit], s=size, marker=marker, facecolors="none", edgecolors=normal_color, linewidths=0.8)
    ax.scatter(xx[hit], yy[hit], s=size + 12, marker=marker, facecolors="none", edgecolors=defect_color, linewidths=1.2)


def plot_example(
    wafer_map: np.ndarray,
    scores: np.ndarray,
    first_mask: np.ndarray,
    selected: np.ndarray,
    record,
    out_path: Path,
) -> dict[str, object]:
    wafer_cmap = ListedColormap(["#2f2f2f", "#e8e8e8", "#d62728"])
    first_cmap = ListedColormap(["#2f2f2f", "#e8e8e8", "#1f77b4", "#d62728"])
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    selected_hits = int((selected & defects).sum())
    selected_count = int(selected.sum())
    first_hits = int((first_mask & defects).sum())

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1), constrained_layout=True)
    axes[0].imshow(wafer_display(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    axes[0].set_title(f"Actual dense map\n{record.failureType}, row {int(record.row_index)}")

    axes[1].imshow(first_pass_display(wafer_map, first_mask), cmap=first_cmap, interpolation="nearest")
    axes[1].set_title(f"First-pass input\nhits={first_hits}, density={float(record.target_density):.0%}")

    im = axes[2].imshow(scores, cmap="viridis", interpolation="nearest")
    axes[2].set_title("CNN predicted risk\nunmeasured dies only")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(wafer_display(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    overlay_points(axes[3], first_mask & valid, wafer_map, "s", "white", "#00ffff", 20)
    overlay_points(axes[3], selected, wafer_map, "o", "#ffcc00", "#00ffff", 28)
    axes[3].set_title(f"Top{selected_count} follow-up overlay\nhits={selected_hits}, coverage={float(record.top32_defect_coverage):.3f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=180, format="jpg")
    plt.close(fig)

    return {
        "row_index": int(record.row_index),
        "failureType": str(record.failureType),
        "target_density": float(record.target_density),
        "figure": str(out_path),
        "top32_hits": selected_hits,
        "top32_selected": selected_count,
        "cnn_defect_coverage": float(record.top32_defect_coverage),
        "coverage32_defect_coverage": float(record.coverage32_defect_coverage),
        "average_precision": float(record.average_precision),
        "roc_auc": float(record.roc_auc),
        "first_pass_hits": first_hits,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    hidden = int(checkpoint.get("args", {}).get("hidden_channels", 32))
    model = sparse_cnn.SparseRiskCNN(in_channels=7, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    patterned = pd.read_pickle(args.patterned)
    eval_rows = pd.read_csv(args.eval_rows)
    examples = choose_examples(eval_rows, args.patterns, args.density, args.examples_per_pattern)

    records: list[dict[str, object]] = []
    for record in examples.itertuples(index=False):
        wafer_map = np.asarray(patterned.at[int(record.row_index), "waferMap"])
        scores, first_mask = score_wafer(model, wafer_map, float(record.target_density), device)
        candidates = valid_die_mask(wafer_map) & ~first_mask
        selected = topk_from_scores(scores, candidates, args.top_k)
        out_name = f"row_{int(record.row_index)}_{record.failureType}_density_{float(record.target_density):.2f}.jpg"
        records.append(plot_example(wafer_map, scores, first_mask, selected, record, args.out_dir / out_name))

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.report_dir / "sparse_cnn_visual_examples.csv", index=False)
    lines = [
        "# Sparse CNN Visual Examples",
        "",
        "Each JPG shows actual dense defects, first-pass sparse input, CNN risk scores over unmeasured dies, and top32 follow-up overlay.",
        "",
        "Dense maps are shown only for offline evaluation/visualization.",
        "",
    ]
    for row in records:
        lines.extend(
            [
                f"## {row['failureType']} row {row['row_index']} density {row['target_density']:.0%}",
                "",
                f"- Figure: `{row['figure']}`",
                f"- CNN top32 hits: {row['top32_hits']} / {row['top32_selected']}",
                f"- CNN defect coverage: {row['cnn_defect_coverage']:.3f}",
                f"- Coverage32 defect coverage: {row['coverage32_defect_coverage']:.3f}",
                f"- AP: {row['average_precision']:.3f}",
                f"- ROC-AUC: {row['roc_auc']:.3f}",
                "",
            ]
        )
    (args.report_dir / "sparse_cnn_visual_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote JPG examples to {args.out_dir}")
    print(f"wrote report to {args.report_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
