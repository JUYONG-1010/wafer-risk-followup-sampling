from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import defect_mask, make_coverage_sampling_mask, sampling_metrics, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("outputs") / "figures" / "41_density_risk_map_examples_v1"
DEFAULT_REPORT_DIR = Path("reports") / "density_risk_map_examples_v1"
POLICY_SCRIPT = Path("scripts") / "42_evaluate_morphology_aware_policy.py"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"
RISK_MAP_SCRIPT = Path("scripts") / "50_evaluate_density_risk_maps.py"
DEFAULT_PATTERNS = ["Loc", "Scratch", "Edge-Loc", "Edge-Ring", "Donut", "Center", "Random"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_module("policy42", POLICY_SCRIPT)
density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)
risk_eval = load_module("risk_eval50", RISK_MAP_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final-style density risk map and follow-up examples."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.03, 0.10])
    parser.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    parser.add_argument("--examples-per-pattern", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=350)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=40)
    parser.add_argument("--morph-estimators", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--replacement-count", type=int, default=1)
    parser.add_argument("--point-weight", type=float, default=0.60)
    parser.add_argument("--morph-weight", type=float, default=0.30)
    parser.add_argument("--weak-rescue-weight", type=float, default=0.25)
    parser.add_argument("--guarded-point-weight", type=float, default=0.30)
    parser.add_argument("--guarded-morph-weight", type=float, default=0.15)
    parser.add_argument("--guarded-weak-rescue-weight", type=float, default=0.10)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    return parser.parse_args()


def representative_rows(
    patterned: pd.DataFrame,
    test_wafers: np.ndarray,
    patterns: list[str],
    examples_per_pattern: int,
) -> pd.DataFrame:
    test_set = set(int(v) for v in test_wafers)
    data = patterned[patterned.index.isin(test_set)].copy()
    records: list[pd.DataFrame] = []
    for pattern in patterns:
        subset = data[data["failureType_clean"].astype(str) == pattern].head(examples_per_pattern)
        if not subset.empty:
            records.append(subset)
    return pd.concat(records) if records else pd.DataFrame()


def normalize_map(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        out = values.copy()
        out[finite] = 0.0
        return out
    return (values - lo) / (hi - lo)


def risk_array(wafer_map: np.ndarray, candidates: pd.DataFrame, score_col: str) -> np.ndarray:
    risk = np.full(wafer_map.shape, np.nan, dtype=float)
    yy = candidates["candidate_y"].astype(int).to_numpy()
    xx = candidates["candidate_x"].astype(int).to_numpy()
    risk[yy, xx] = candidates[score_col].to_numpy(dtype=float)
    return normalize_map(risk)


def wafer_array(wafer_map: np.ndarray) -> np.ndarray:
    arr = np.zeros_like(wafer_map, dtype=int)
    arr[wafer_map == 1] = 1
    arr[wafer_map == 2] = 2
    return arr


def first_pass_array(wafer_map: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
    arr = np.zeros_like(wafer_map, dtype=int)
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    arr[valid] = 1
    arr[first_mask & valid] = 2
    arr[first_mask & defects] = 3
    return arr


def overlay_followup(ax, selected: pd.DataFrame, wafer_map: np.ndarray, color: str = "#ffcc00") -> None:
    if selected.empty:
        return
    defects = defect_mask(wafer_map)
    yy = selected["candidate_y"].astype(int).to_numpy()
    xx = selected["candidate_x"].astype(int).to_numpy()
    hit = defects[yy, xx]
    ax.scatter(xx[~hit], yy[~hit], s=24, marker="o", facecolors="none", edgecolors=color, linewidths=0.9)
    ax.scatter(xx[hit], yy[hit], s=34, marker="o", facecolors="none", edgecolors="#00ffff", linewidths=1.2)


def overlay_first(ax, first_mask: np.ndarray, wafer_map: np.ndarray) -> None:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    yy, xx = np.nonzero(first_mask & valid)
    if len(xx):
        ax.scatter(xx, yy, s=16, marker="s", facecolors="none", edgecolors="white", linewidths=0.7)
    hy, hx = np.nonzero(first_mask & defects)
    if len(hx):
        ax.scatter(hx, hy, s=24, marker="s", facecolors="none", edgecolors="#00ffff", linewidths=1.0)


def make_models(args: argparse.Namespace, patterned: pd.DataFrame, densities: list[float]):
    morph_data = pd.read_csv(args.morph_dataset)
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
    global_target_ratio = density_policy.mean_actual_defect_ratio(patterned, train_wafers)
    return point_model, morph_models, morph_columns, morph_lookup, global_target_ratio, test_wafers


def score_wafer(
    args: argparse.Namespace,
    wafer_map: np.ndarray,
    row_index: int,
    failure_type: str,
    density: float,
    point_model,
    morph_models,
    morph_columns,
    morph_lookup,
    global_target_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float], dict[str, float]]:
    first_mask = density_policy.make_initial_coverage_mask(wafer_map, density)
    candidates = candidate_feature_frame(
        wafer_map,
        first_pass_type=density_policy.density_key(density),
        first_mask=first_mask,
        row_index=row_index,
        failure_type=failure_type,
        include_label=True,
    )
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
    group_irregular_prob = float(group_probs.get("irregular_local", 0.0))
    candidates = risk_eval.risk_scores_for_candidates(candidates, morph_probs, group_irregular_prob, args)

    coverage_mask = make_coverage_sampling_mask(
        wafer_map,
        n_points=args.top_k,
        existing_mask=first_mask,
    )
    coverage_coords = np.column_stack(np.nonzero(coverage_mask & ~first_mask))
    coverage_selected = pd.DataFrame(
        {
            "candidate_y": coverage_coords[:, 0].astype(int) if len(coverage_coords) else [],
            "candidate_x": coverage_coords[:, 1].astype(int) if len(coverage_coords) else [],
        }
    )
    hybrid_selected = density_policy.select_hybrid_coverage_morphrisk_candidates(
        wafer_map,
        first_mask,
        coverage_mask,
        candidates,
        base_score=candidates["risk_guarded"].to_numpy(dtype=float),
        replacement_count=args.replacement_count,
        bias_penalty_weight=3.0,
        first_ratio_weight=args.first_ratio_weight,
        global_target_ratio=global_target_ratio,
    )
    return candidates, coverage_selected, hybrid_selected, first_mask, morph_probs, group_probs


def plot_example(
    args: argparse.Namespace,
    wafer_map: np.ndarray,
    candidates: pd.DataFrame,
    coverage_selected: pd.DataFrame,
    hybrid_selected: pd.DataFrame,
    first_mask: np.ndarray,
    morph_probs: dict[str, float],
    row_index: int,
    failure_type: str,
    density: float,
    out_path: Path,
) -> dict[str, object]:
    wafer_cmap = ListedColormap(["#2f2f2f", "#e7e7e7", "#d62728"])
    first_cmap = ListedColormap(["#2f2f2f", "#d8d8d8", "#1f77b4", "#d62728"])
    risk = risk_array(wafer_map, candidates, "risk_guarded")
    top_probs = sorted(morph_probs.items(), key=lambda item: item[1], reverse=True)[:3]

    coverage_mask = density_policy.make_mask_from_points(wafer_map, first_mask, coverage_selected)
    hybrid_mask = density_policy.make_mask_from_points(wafer_map, first_mask, hybrid_selected)
    coverage_metrics = sampling_metrics(wafer_map, coverage_mask)
    hybrid_metrics = sampling_metrics(wafer_map, hybrid_mask)

    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.9), constrained_layout=True)
    axes[0].imshow(wafer_array(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    axes[0].set_title(f"Actual dense map\n{failure_type}, row {row_index}")

    axes[1].imshow(first_pass_array(wafer_map, first_mask), cmap=first_cmap, interpolation="nearest")
    axes[1].set_title(f"First pass\n{density:.0%} coverage")

    im = axes[2].imshow(risk, cmap="viridis", interpolation="nearest")
    axes[2].set_title("Predicted risk map\n(first-pass only)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(risk, cmap="viridis", interpolation="nearest")
    overlay_first(axes[3], first_mask, wafer_map)
    overlay_followup(axes[3], hybrid_selected, wafer_map)
    prob_text = "\n".join([f"{label}: {prob:.2f}" for label, prob in top_probs])
    axes[3].set_title(f"Hybrid follow-up\n{prob_text}")

    axes[4].imshow(wafer_array(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    overlay_first(axes[4], first_mask, wafer_map)
    overlay_followup(axes[4], hybrid_selected, wafer_map)
    axes[4].set_title(
        "Evaluation overlay\n"
        f"coverage={hybrid_metrics['defect_coverage']:.3f}, "
        f"abs err={hybrid_metrics['absolute_error']:.3f}"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "figure": str(out_path),
        "top1_morphology": top_probs[0][0],
        "top1_probability": top_probs[0][1],
        "coverage32_defect_coverage": float(coverage_metrics["defect_coverage"]),
        "hybrid_defect_coverage": float(hybrid_metrics["defect_coverage"]),
        "coverage32_absolute_error": float(coverage_metrics["absolute_error"]),
        "hybrid_absolute_error": float(hybrid_metrics["absolute_error"]),
        "hybrid_sampled_defects": int(hybrid_metrics["sampled_defects"]),
        "actual_defects": int(hybrid_metrics["total_defects"]),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    densities = [float(v) for v in args.densities]
    patterned = pd.read_pickle(args.patterned)
    point_model, morph_models, morph_columns, morph_lookup, global_target_ratio, test_wafers = make_models(
        args,
        patterned,
        densities,
    )

    examples = representative_rows(
        patterned,
        test_wafers=test_wafers,
        patterns=args.patterns,
        examples_per_pattern=args.examples_per_pattern,
    )
    records: list[dict[str, object]] = []
    for row in examples.itertuples(index=True):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in densities:
            candidates, coverage_selected, hybrid_selected, first_mask, morph_probs, _ = score_wafer(
                args,
                wafer_map,
                row_index,
                failure_type,
                density,
                point_model,
                morph_models,
                morph_columns,
                morph_lookup,
                global_target_ratio,
            )
            out_path = args.out_dir / f"row_{row_index}_{failure_type}_density_{density:.2f}.png"
            records.append(
                plot_example(
                    args,
                    wafer_map,
                    candidates,
                    coverage_selected,
                    hybrid_selected,
                    first_mask,
                    morph_probs,
                    row_index,
                    failure_type,
                    density,
                    out_path,
                )
            )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.report_dir / "density_risk_map_examples.csv", index=False)
    lines = [
        "# Density Risk Map Examples",
        "",
        "Each figure shows actual dense defects, first-pass observations, predicted risk map, recommended hybrid follow-up, and offline evaluation overlay.",
        "",
        "Dense maps are used only for evaluation and visualization, not for prediction.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['failureType']} row {record['row_index']} density {record['target_density']:.0%}",
                "",
                f"- Top predicted morphology: {record['top1_morphology']} ({record['top1_probability']:.2f})",
                f"- Coverage32 defect coverage: {record['coverage32_defect_coverage']:.3f}",
                f"- Hybrid defect coverage: {record['hybrid_defect_coverage']:.3f}",
                f"- Coverage32 absolute error: {record['coverage32_absolute_error']:.3f}",
                f"- Hybrid absolute error: {record['hybrid_absolute_error']:.3f}",
                f"- Figure: `{record['figure']}`",
                "",
            ]
        )
    (args.report_dir / "density_risk_map_examples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"wrote density risk map examples to {args.out_dir}")
    print(f"wrote density risk map example report to {args.report_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
