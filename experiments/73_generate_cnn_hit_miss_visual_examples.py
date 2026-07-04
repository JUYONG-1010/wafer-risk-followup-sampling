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
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import defect_mask, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_EVAL_ROWS = (
    Path("data") / "processed" / "sparse_cnn_risk_map_v1_large" / "sparse_cnn_eval_rows.csv"
)
DEFAULT_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"
DEFAULT_OUT_DIR = Path("outputs") / "figures" / "73_cnn_hit_miss_visual_examples_v1"
DEFAULT_REPORT_DIR = Path("reports") / "cnn_hit_miss_visual_examples_v1"


def load_sparse_cnn_module():
    module_path = PROJECT_ROOT / "experiments" / "68_train_sparse_cnn_risk_map.py"
    spec = importlib.util.spec_from_file_location("sparse_cnn68", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sparse_cnn = load_sparse_cnn_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clearer CNN hit/miss visual examples for random wafers."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--eval-rows", type=Path, default=DEFAULT_EVAL_ROWS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--density", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--min-candidate-defects", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


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
    arr[first_mask & valid & ~defects] = 2
    arr[first_mask & defects] = 3
    return arr


def outcome_display(wafer_map: np.ndarray, first_mask: np.ndarray, selected: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    arr = np.zeros_like(wafer_map, dtype=int)
    arr[valid] = 1
    arr[defects & ~first_mask] = 2
    arr[first_mask & valid] = 3
    arr[selected & valid & ~defects] = 4
    arr[selected & defects] = 5
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
    values = scores[coords[:, 0], coords[:, 1]]
    k = min(top_k, len(values))
    chosen = coords[np.argsort(values)[::-1][:k]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def overlay_selected(ax, selected: np.ndarray, defects: np.ndarray) -> None:
    false_pos = selected & ~defects
    true_pos = selected & defects
    yy, xx = np.nonzero(false_pos)
    ax.scatter(xx, yy, s=36, marker="x", c="#ffb000", linewidths=1.4)
    yy, xx = np.nonzero(true_pos)
    ax.scatter(xx, yy, s=40, marker="o", facecolors="none", edgecolors="#00d5ff", linewidths=1.5)


def plot_example(
    wafer_map: np.ndarray,
    scores: np.ndarray,
    first_mask: np.ndarray,
    selected: np.ndarray,
    record,
    out_path: Path,
) -> dict[str, object]:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    unmeasured_defects = defects & ~first_mask
    true_pos = selected & defects
    false_pos = selected & ~defects
    false_neg = unmeasured_defects & ~selected
    first_hits = first_mask & defects

    tp = int(true_pos.sum())
    fp = int(false_pos.sum())
    fn = int(false_neg.sum())
    first_hit_count = int(first_hits.sum())
    total_defects = int(defects.sum())
    unmeasured_defect_count = int(unmeasured_defects.sum())
    selected_count = int(selected.sum())
    hit_rate_top32 = tp / selected_count if selected_count else float("nan")
    recall_unmeasured = tp / unmeasured_defect_count if unmeasured_defect_count else float("nan")
    recall_total = (tp + first_hit_count) / total_defects if total_defects else float("nan")

    wafer_cmap = ListedColormap(["#2f2f2f", "#e8e8e8", "#d62728"])
    first_cmap = ListedColormap(["#2f2f2f", "#e8e8e8", "#1f77b4", "#d62728"])
    outcome_cmap = ListedColormap(["#2f2f2f", "#eeeeee", "#d62728", "#1f77b4", "#ffb000", "#00d5ff"])

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2), constrained_layout=True)
    axes = axes.ravel()

    axes[0].imshow(wafer_display(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    axes[0].set_title(f"Actual dense defect map\n{record.failureType}, row {int(record.row_index)}")

    axes[1].imshow(first_pass_display(wafer_map, first_mask), cmap=first_cmap, interpolation="nearest")
    axes[1].set_title(f"First-pass input only\nfirst hits={first_hit_count}, density={float(record.target_density):.0%}")

    im = axes[2].imshow(scores, cmap="viridis", interpolation="nearest")
    overlay_selected(axes[2], selected, defects)
    axes[2].set_title(f"CNN risk map + Top{selected_count}\ncyan circle=hit, orange x=miss")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(outcome_display(wafer_map, first_mask, selected), cmap=outcome_cmap, interpolation="nearest")
    axes[3].set_title(
        "Top32 hit/miss outcome\n"
        f"TP={tp}, FP={fp}, FN(unmeasured)={fn}, hit-rate={hit_rate_top32:.1%}"
    )
    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="first-pass observed", markerfacecolor="#1f77b4", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="missed defect (FN)", markerfacecolor="#d62728", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="selected normal (FP)", markerfacecolor="#ffb000", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="selected defect (TP)", markerfacecolor="#00d5ff", markersize=8),
    ]
    axes[3].legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=2, fontsize=8)

    fig.suptitle(
        f"CNN follow-up result: Top{selected_count} selected | "
        f"TP {tp}, FP {fp}, first-pass hits {first_hit_count}, total recall {recall_total:.1%}",
        fontsize=14,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(out_path, dpi=170, format="jpg")
    plt.close(fig)

    return {
        "row_index": int(record.row_index),
        "failureType": str(record.failureType),
        "target_density": float(record.target_density),
        "figure": str(out_path),
        "topk": selected_count,
        "tp_selected_defects": tp,
        "fp_selected_normals": fp,
        "fn_unmeasured_defects": fn,
        "first_pass_hits": first_hit_count,
        "total_defects": total_defects,
        "unmeasured_defects": unmeasured_defect_count,
        "top32_hit_rate": hit_rate_top32,
        "unmeasured_recall": recall_unmeasured,
        "total_recall_after_followup": recall_total,
        "average_precision": float(record.average_precision),
        "roc_auc": float(record.roc_auc),
        "top32_defect_coverage": float(record.top32_defect_coverage),
    }


def choose_random_examples(eval_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    focus = eval_rows[np.isclose(eval_rows["target_density"], args.density)].copy()
    focus = focus[focus["candidate_defects"] >= args.min_candidate_defects].copy()
    if len(focus) < args.n_examples:
        focus = eval_rows[np.isclose(eval_rows["target_density"], args.density)].copy()
    return focus.sample(n=min(args.n_examples, len(focus)), random_state=args.seed)


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
    examples = choose_random_examples(eval_rows, args)

    records: list[dict[str, object]] = []
    for record in examples.itertuples(index=False):
        wafer_map = np.asarray(patterned.at[int(record.row_index), "waferMap"])
        scores, first_mask = score_wafer(model, wafer_map, float(record.target_density), device)
        candidates = valid_die_mask(wafer_map) & ~first_mask
        selected = topk_from_scores(scores, candidates, args.top_k)
        safe_pattern = str(record.failureType).replace("/", "-")
        out_path = args.out_dir / f"row_{int(record.row_index)}_{safe_pattern}_density_{float(record.target_density):.2f}_hitmiss.jpg"
        records.append(plot_example(wafer_map, scores, first_mask, selected, record, out_path))

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.report_dir / "cnn_hit_miss_visual_examples.csv", index=False)

    lines = [
        "# CNN Hit/Miss Visual Examples",
        "",
        "Random examples from the saved sparse CNN evaluation rows.",
        "",
        "Color meaning in the outcome panel:",
        "",
        "- cyan: selected by CNN and actually defective (TP)",
        "- orange: selected by CNN but normal (FP)",
        "- red: unmeasured defect not selected in Top32 (FN)",
        "- blue: first-pass observed die",
        "",
    ]
    for row in records:
        lines.extend(
            [
                f"## {row['failureType']} row {row['row_index']}",
                "",
                f"- Figure: `{row['figure']}`",
                f"- Top{row['topk']} TP/FP/FN: {row['tp_selected_defects']} / {row['fp_selected_normals']} / {row['fn_unmeasured_defects']}",
                f"- First-pass hits: {row['first_pass_hits']}",
                f"- Total recall after first-pass + follow-up: {row['total_recall_after_followup']:.3f}",
                f"- AP: {row['average_precision']:.3f}",
                f"- ROC-AUC: {row['roc_auc']:.3f}",
                "",
            ]
        )
    (args.report_dir / "cnn_hit_miss_visual_examples.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote hit/miss JPG examples to {args.out_dir}")
    print(f"wrote report to {args.report_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
