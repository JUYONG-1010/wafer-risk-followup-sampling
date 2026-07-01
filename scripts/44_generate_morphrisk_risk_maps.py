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

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame, make_first_pass_mask
from src.sampling import defect_mask, valid_die_mask


DEFAULT_POINT_DATASET = (
    Path("data")
    / "processed"
    / "point_ranking_v0_medium"
    / "point_ranking_dataset.csv"
)
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "morphology_risk_v1"
    / "morphology_risk_dataset.csv"
)
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("outputs") / "figures" / "30_morphrisk_risk_maps_v1"
DEFAULT_REPORT_DIR = Path("reports") / "morphrisk_risk_maps_v1"
POLICY_SCRIPT = Path("scripts") / "42_evaluate_morphology_aware_policy.py"
PATTERN_ORDER = ["Loc", "Scratch", "Random", "Edge-Ring", "Center", "Donut"]


def load_policy_module():
    spec = importlib.util.spec_from_file_location("morph_policy", PROJECT_ROOT / POLICY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load policy module: {POLICY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D morphrisk score maps for representative wafers."
    )
    parser.add_argument("--point-dataset", type=Path, default=DEFAULT_POINT_DATASET)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--first-pass-types", nargs="+", default=["grid9", "grid25"])
    parser.add_argument("--patterns", nargs="+", default=PATTERN_ORDER)
    parser.add_argument("--examples-per-pattern", type=int, default=1)
    parser.add_argument("--point-estimators", type=int, default=50)
    parser.add_argument("--morph-estimators", type=int, default=70)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--point-weight", type=float, default=0.60)
    parser.add_argument("--morph-weight", type=float, default=0.30)
    parser.add_argument("--weak-rescue-weight", type=float, default=0.25)
    parser.add_argument("--diversity-weight", type=float, default=0.40)
    parser.add_argument("--uncertainty-diversity-weight", type=float, default=0.35)
    return parser.parse_args()


def clean_failure_type(row) -> str:
    value = getattr(row, "failureType_clean", None)
    if value is None:
        value = row.failureType
    return str(value)


def representative_rows(
    patterned: pd.DataFrame,
    test_wafers: np.ndarray,
    patterns: list[str],
    examples_per_pattern: int,
) -> pd.DataFrame:
    test_set = set(int(v) for v in test_wafers)
    data = patterned[patterned.index.isin(test_set)].copy()
    records = []
    for pattern in patterns:
        subset = data[data["failureType_clean"].astype(str) == pattern].head(examples_per_pattern)
        records.append(subset)
    return pd.concat(records) if records else pd.DataFrame()


def normalized_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def score_candidates(
    policy,
    wafer_map: np.ndarray,
    first_pass_type: str,
    row_index: int,
    failure_type: str,
    point_model,
    morph_models: dict,
    morph_columns: dict,
    morph_lookup: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, np.ndarray]:
    first_mask = make_first_pass_mask(wafer_map, first_pass_type)
    candidates = candidate_feature_frame(
        wafer_map,
        first_pass_type=first_pass_type,
        row_index=row_index,
        failure_type=failure_type,
        include_label=True,
    )
    candidates = candidates.copy()
    candidates["point_score"] = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]

    morph_row = morph_lookup[(row_index, first_pass_type)]
    morph_model = morph_models[first_pass_type]
    cols = morph_columns[first_pass_type]
    morph_proba = morph_model.predict_proba(pd.DataFrame([morph_row[cols].to_dict()]))[0]
    morph_probs = {
        str(label): float(prob)
        for label, prob in zip(morph_model.classes_, morph_proba, strict=True)
    }

    morph_prior = policy.pattern_prior_scores(candidates, morph_probs)
    weak_rescue = policy.weak_pattern_rescue_scores(candidates, morph_probs)
    uncertainty = policy.morphology_uncertainty(morph_probs)
    weak_risk = policy.weak_pattern_risk(morph_probs)
    diversity_weight = (
        args.diversity_weight
        + args.uncertainty_diversity_weight * max(uncertainty, weak_risk)
    )
    base_score = (
        args.point_weight * candidates["point_score"].to_numpy(dtype=float)
        + args.morph_weight * morph_prior
        + args.weak_rescue_weight * weak_rescue
    )
    candidates["morph_prior_score"] = morph_prior
    candidates["weak_rescue_score"] = weak_rescue
    candidates["final_score"] = base_score
    candidates["final_score_norm"] = normalized_scores(base_score)

    selected = policy.greedy_select_by_score(
        wafer_map,
        first_mask,
        candidates,
        base_score=base_score,
        top_k=args.top_k,
        diversity_weight=diversity_weight,
    )
    return candidates, morph_probs, selected, first_mask


def wafer_display_array(wafer_map: np.ndarray) -> np.ndarray:
    arr = np.zeros_like(wafer_map, dtype=int)
    arr[wafer_map == 0] = 0
    arr[wafer_map == 1] = 1
    arr[wafer_map == 2] = 2
    return arr


def risk_map_array(wafer_map: np.ndarray, candidates: pd.DataFrame) -> np.ndarray:
    risk = np.full(wafer_map.shape, np.nan, dtype=float)
    yy = candidates["candidate_y"].astype(int).to_numpy()
    xx = candidates["candidate_x"].astype(int).to_numpy()
    risk[yy, xx] = candidates["final_score_norm"].to_numpy(dtype=float)
    return risk


def observed_first_pass_array(wafer_map: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(wafer_map, dtype=int)
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    out[valid] = 1
    out[first_mask & valid] = 2
    out[first_mask & defects] = 3
    return out


def overlay_points(ax, selected: pd.DataFrame, first_mask: np.ndarray, wafer_map: np.ndarray) -> None:
    fy, fx = np.nonzero(first_mask & valid_die_mask(wafer_map))
    if len(fx):
        ax.scatter(fx, fy, s=18, marker="s", facecolors="none", edgecolors="white", linewidths=0.8)
    if not selected.empty:
        ax.scatter(
            selected["candidate_x"],
            selected["candidate_y"],
            s=20,
            marker="o",
            facecolors="none",
            edgecolors="#ffcc00",
            linewidths=0.9,
        )


def plot_example(
    wafer_map: np.ndarray,
    candidates: pd.DataFrame,
    morph_probs: dict[str, float],
    selected: pd.DataFrame,
    first_mask: np.ndarray,
    row_index: int,
    failure_type: str,
    first_pass_type: str,
    out_path: Path,
) -> dict[str, object]:
    wafer_cmap = ListedColormap(["#2f2f2f", "#e8e8e8", "#d62728"])
    observed_cmap = ListedColormap(["#2f2f2f", "#d8d8d8", "#1f77b4", "#d62728"])
    risk = risk_map_array(wafer_map, candidates)
    top_probs = sorted(morph_probs.items(), key=lambda item: item[1], reverse=True)[:3]

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    axes[0].imshow(wafer_display_array(wafer_map), cmap=wafer_cmap, interpolation="nearest")
    axes[0].set_title(f"Actual map\n{failure_type}, row {row_index}")

    axes[1].imshow(observed_first_pass_array(wafer_map, first_mask), cmap=observed_cmap, interpolation="nearest")
    axes[1].set_title(f"First-pass\n{first_pass_type}")

    im = axes[2].imshow(risk, cmap="viridis", interpolation="nearest")
    axes[2].set_title("Morphrisk score map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(risk, cmap="viridis", interpolation="nearest")
    overlay_points(axes[3], selected, first_mask, wafer_map)
    prob_text = "\n".join([f"{label}: {prob:.2f}" for label, prob in top_probs])
    axes[3].set_title(f"Selected follow-up\n{prob_text}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    actual_defects = int(defect_mask(wafer_map).sum())
    selected_hits = 0
    if not selected.empty:
        yy = selected["candidate_y"].astype(int).to_numpy()
        xx = selected["candidate_x"].astype(int).to_numpy()
        selected_hits = int(defect_mask(wafer_map)[yy, xx].sum())
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "first_pass_type": first_pass_type,
        "figure": str(out_path),
        "actual_defect_count": actual_defects,
        "selected_followup_hits": selected_hits,
        "top1_morphology": top_probs[0][0],
        "top1_probability": top_probs[0][1],
        "top2_morphology": top_probs[1][0] if len(top_probs) > 1 else "",
        "top2_probability": top_probs[1][1] if len(top_probs) > 1 else np.nan,
        "top3_morphology": top_probs[2][0] if len(top_probs) > 2 else "",
        "top3_probability": top_probs[2][1] if len(top_probs) > 2 else np.nan,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy_module()

    point_data = pd.read_csv(args.point_dataset)
    morph_data = pd.read_csv(args.morph_dataset)
    patterned = pd.read_pickle(args.patterned)

    train_wafers, test_wafers = policy.split_wafers(point_data, args.test_size, args.seed)
    train_point = point_data[point_data["row_index"].isin(train_wafers)].copy()
    point_model = policy.train_point_model(
        train_point,
        args.seed,
        args.point_estimators,
        args.n_jobs,
    )
    morph_models, morph_columns, morph_lookup = policy.train_morph_models(
        morph_data,
        train_wafers=train_wafers,
        seed=args.seed,
        n_estimators=args.morph_estimators,
        n_jobs=args.n_jobs,
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
        failure_type = clean_failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for first_pass_type in args.first_pass_types:
            candidates, morph_probs, selected, first_mask = score_candidates(
                policy,
                wafer_map,
                first_pass_type,
                row_index,
                failure_type,
                point_model,
                morph_models,
                morph_columns,
                morph_lookup,
                args,
            )
            safe_pattern = failure_type.replace("/", "_")
            out_path = args.out_dir / f"row_{row_index}_{safe_pattern}_{first_pass_type}.png"
            records.append(
                plot_example(
                    wafer_map,
                    candidates,
                    morph_probs,
                    selected,
                    first_mask,
                    row_index,
                    failure_type,
                    first_pass_type,
                    out_path,
                )
            )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.report_dir / "morphrisk_risk_map_examples.csv", index=False)
    lines = [
        "# Morphrisk Risk Map Examples",
        "",
        "These figures visualize the internal morphrisk score map used to choose follow-up points.",
        "",
        "Important: the dense defect map is shown only for offline evaluation/interpretation. It is not used to select follow-up points.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['failureType']} row {record['row_index']} {record['first_pass_type']}",
                "",
                f"- Top morphology risks: {record['top1_morphology']} {record['top1_probability']:.2f}, "
                f"{record['top2_morphology']} {record['top2_probability']:.2f}, "
                f"{record['top3_morphology']} {record['top3_probability']:.2f}",
                f"- Selected follow-up hits: {record['selected_followup_hits']} / {record['actual_defect_count']} actual defect dies",
                f"- Figure: `{record['figure']}`",
                "",
            ]
        )
    (args.report_dir / "morphrisk_risk_map_examples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"wrote morphrisk maps to {args.out_dir}")
    print(f"wrote morphrisk map summary to {args.report_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
