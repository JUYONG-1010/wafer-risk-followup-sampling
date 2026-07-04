from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    sampling_metrics,
    valid_die_mask,
)


DEFAULT_PATTERNED = PROJECT_ROOT / "data" / "processed" / "subsets" / "patterned_subset.pkl"
REPORT_DIR = PROJECT_ROOT / "reports" / "final_demo"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_demo"


def load_density_policy_module():
    module_path = PROJECT_ROOT / "experiments" / "47_evaluate_density_followup_policy.py"
    spec = importlib.util.spec_from_file_location("density_policy47_final_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


density_policy = load_density_policy_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight public demo for coverage32 vs final RandomForest "
            "point-risk ranking."
        )
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--max-test-wafers", type=int, default=100)
    parser.add_argument("--max-train-wafers", type=int, default=1200)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--first-pass-density", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--point-estimators", type=int, default=100)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def require_processed_data(patterned_path: Path) -> bool:
    if patterned_path.exists():
        return True

    print("\nRequired processed data was not found.")
    print(f"Missing: {patterned_path}")
    print("\nTo create it, place the raw WM-811K pickle at:")
    print("  LSWMD.pkl/LSWMD.pkl")
    print("\nThen run:")
    print("  python experiments/01_extract_labeled_subset.py")
    print("\nThis demo exits without fabricating results.\n")
    return False


def limit_ids(ids: np.ndarray, limit: int, seed: int) -> np.ndarray:
    ids = np.asarray(ids)
    if limit and limit > 0 and len(ids) > limit:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=limit, replace=False)
    return ids


def mask_from_top_scores(
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    candidates: pd.DataFrame,
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    followup = np.zeros_like(valid_die_mask(wafer_map), dtype=bool)
    if candidates.empty or len(scores) == 0 or k <= 0:
        return first_mask.copy()
    order = np.argsort(scores)[::-1][: min(k, len(scores))]
    yy = candidates.iloc[order]["candidate_y"].astype(int).to_numpy()
    xx = candidates.iloc[order]["candidate_x"].astype(int).to_numpy()
    followup[yy, xx] = True
    followup &= valid_die_mask(wafer_map) & ~first_mask
    return first_mask | followup


def policy_record(
    policy_name: str,
    row_index: int,
    failure_type: str,
    density: float,
    k: int,
    wafer_map: np.ndarray,
    first_mask: np.ndarray,
    selected_mask: np.ndarray,
) -> dict[str, object]:
    followup_mask = np.asarray(selected_mask, dtype=bool) & ~np.asarray(first_mask, dtype=bool)
    followup_metrics = sampling_metrics(wafer_map, followup_mask)
    all_metrics = sampling_metrics(wafer_map, selected_mask)
    followup_valid = int(followup_metrics["sampled_valid_count"])
    followup_defects = int(followup_metrics["sampled_defects"])
    hit_rate = followup_defects / followup_valid if followup_valid else float("nan")
    total_hidden_defects = int((defect_mask(wafer_map) & ~first_mask).sum())
    hidden_coverage = (
        followup_defects / total_hidden_defects if total_hidden_defects > 0 else float("nan")
    )
    return {
        "row_index": row_index,
        "failureType": failure_type,
        "target_density": density,
        "top_k": k,
        "policy_name": policy_name,
        "followup_valid_count": followup_valid,
        "followup_defects": followup_defects,
        "followup_hit_rate": hit_rate,
        "hidden_defect_coverage": hidden_coverage,
        "absolute_error": float(all_metrics["absolute_error"]),
        "severe_miss": int(followup_defects == 0 and total_hidden_defects > 0),
    }


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby("policy_name", dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_followup_defects=("followup_defects", "mean"),
            mean_hit_rate=("followup_hit_rate", "mean"),
            mean_hidden_defect_coverage=("hidden_defect_coverage", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .reset_index()
    )


def markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.select_dtypes(include="number").columns:
        if col == "wafers":
            view[col] = view[col].astype(int).astype(str)
        else:
            view[col] = view[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    headers = list(view.columns)
    rows = view.astype(str).values.tolist()
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "final_demo_hit_rate.png"
    plot = summary.copy()
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(plot["policy_name"], plot["mean_hit_rate"], color=["#4c78a8", "#59a14f"])
    plt.ylabel("Hit rate among follow-up points")
    plt.xlabel("Policy")
    plt.title("Final demo: coverage32 vs RandomForest point-risk ranker")
    plt.ylim(0.0, max(0.05, min(1.0, float(plot["mean_hit_rate"].max()) * 1.25)))
    for idx, value in enumerate(plot["mean_hit_rate"]):
        plt.text(idx, value + 0.01, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return fig_path


def write_report(args: argparse.Namespace, summary: pd.DataFrame, fig_path: Path) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "final_demo_summary.md"
    try:
        fig_display = fig_path.relative_to(PROJECT_ROOT)
    except ValueError:
        fig_display = fig_path
    lines = [
        "# Final Demo Summary",
        "",
        "This is a lightweight local demo, not the headline 3-seed robustness result.",
        "",
        "The dense WM-811K map is used for supervised training labels and offline evaluation.",
        "At recommendation time, the RF ranker scores candidates from sparse first-pass evidence and geometry-derived features only.",
        "",
        "## Settings",
        "",
        f"- first-pass density: {args.first_pass_density:.0%}",
        f"- follow-up budget K: {args.k}",
        f"- seed: {args.seed}",
        f"- max train wafers: {args.max_train_wafers}",
        f"- max test wafers: {args.max_test_wafers}",
        f"- RandomForest estimators: {args.point_estimators}",
        "",
        "## Result",
        "",
        markdown_table(summary),
        "",
        "## Generated Figure",
        "",
        f"- `{fig_display}`",
        "",
        "## Headline Result Provenance",
        "",
        "The README headline result uses a 3-seed repeated wafer-level split:",
        "",
        "- coverage32 hit rate: 20.1%",
        "- final RandomForest hit rate: 69.1%",
        "- coverage32 true defects found out of 32: 6.43",
        "- final RF true defects found out of 32: 22.10",
        "- relative hit-rate improvement: +244.0%",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    args.patterned = (PROJECT_ROOT / args.patterned).resolve() if not args.patterned.is_absolute() else args.patterned
    if not require_processed_data(args.patterned):
        return 2

    patterned = pd.read_pickle(args.patterned)
    train_ids, test_ids = density_policy.split_wafers(patterned, args.test_size, args.seed)
    train_ids = limit_ids(train_ids, args.max_train_wafers, args.seed)
    test_ids = limit_ids(test_ids, args.max_test_wafers, args.seed + 1)

    point_args = argparse.Namespace(
        seed=args.seed,
        max_train_wafers=0,
        max_defect_candidates=args.max_defect_candidates,
        max_normal_candidates=args.max_normal_candidates,
        point_estimators=args.point_estimators,
        n_jobs=args.n_jobs,
    )
    print(f"training wafers: {len(train_ids):,}")
    print(f"test wafers: {len(test_ids):,}")
    print("building RandomForest point-risk training rows...")
    train_data = density_policy.build_point_training_data(
        patterned, train_ids, [float(args.first_pass_density)], point_args
    )
    point_model = density_policy.train_point_model(train_data, point_args)

    records: list[dict[str, object]] = []
    test_df = patterned[patterned.index.isin(set(int(value) for value in test_ids))]
    for pos, row in enumerate(test_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        failure_type = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(args.first_pass_density))

        coverage_followup = make_coverage_sampling_mask(
            wafer_map,
            n_points=int(args.k),
            existing_mask=first_mask,
        )
        coverage_selected = first_mask | coverage_followup
        records.append(
            policy_record(
                "coverage32",
                row_index,
                failure_type,
                float(args.first_pass_density),
                int(args.k),
                wafer_map,
                first_mask,
                coverage_selected,
            )
        )

        candidates = candidate_feature_frame(
            wafer_map,
            first_pass_type=density_policy.density_key(float(args.first_pass_density)),
            first_mask=first_mask,
            row_index=row_index,
            failure_type=failure_type,
            include_label=False,
        )
        if not candidates.empty:
            features = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scores = point_model.predict_proba(features)[:, 1].astype(float)
            rf_selected = mask_from_top_scores(wafer_map, first_mask, candidates, scores, int(args.k))
        else:
            rf_selected = first_mask.copy()
        records.append(
            policy_record(
                "noncnn_top32",
                row_index,
                failure_type,
                float(args.first_pass_density),
                int(args.k),
                wafer_map,
                first_mask,
                rf_selected,
            )
        )

        if pos % 25 == 0 or pos == len(test_df):
            print(f"evaluated wafers: {pos:,}/{len(test_df):,}")

    rows = pd.DataFrame.from_records(records)
    summary = summarize(rows)
    fig_path = plot_summary(summary, FIG_DIR)
    report_path = write_report(args, summary, fig_path)

    print("\nFinal demo result")
    print(markdown_table(summary))
    print(f"\nwrote report: {report_path}")
    print(f"wrote figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
