from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS, candidate_feature_frame
from src.sampling import make_coverage_sampling_mask, sampling_metrics


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "density_hybrid_scoring_tuning_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "43_density_hybrid_scoring_tuning_v1"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)
policy = density_policy.policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune guarded hybrid scoring weights for density-based follow-up."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--replacement-counts", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=350)
    parser.add_argument("--max-test-wafers", type=int, default=150)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=40)
    parser.add_argument("--morph-estimators", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    parser.add_argument("--abs-error-tolerance", type=float, default=0.01)
    parser.add_argument("--full-grid", action="store_true")
    return parser.parse_args()


def score_configs(full_grid: bool) -> pd.DataFrame:
    if not full_grid:
        rows = [
            ("baseline_guarded", 0.30, 0.15, 0.10, 3.0),
            ("point_heavy", 0.45, 0.10, 0.05, 3.0),
            ("point_very_heavy", 0.60, 0.05, 0.02, 3.0),
            ("morph_balanced", 0.35, 0.25, 0.10, 3.0),
            ("morph_heavy", 0.25, 0.35, 0.10, 3.0),
            ("weak_rescue", 0.35, 0.15, 0.25, 3.0),
            ("conservative_point", 0.40, 0.08, 0.04, 4.0),
            ("conservative_morph", 0.25, 0.20, 0.06, 4.0),
            ("low_bias_penalty", 0.30, 0.15, 0.10, 2.0),
            ("high_bias_penalty", 0.30, 0.15, 0.10, 5.0),
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "config_name",
                "point_weight",
                "morph_weight",
                "weak_rescue_weight",
                "bias_penalty_weight",
            ],
        )

    records: list[dict[str, object]] = []
    for point_weight in [0.20, 0.30, 0.40, 0.50]:
        for morph_weight in [0.05, 0.15, 0.25, 0.35]:
            for weak_weight in [0.02, 0.10, 0.20]:
                for bias_penalty in [2.0, 3.0, 4.0, 5.0]:
                    records.append(
                        {
                            "config_name": (
                                f"p{point_weight:.2f}_m{morph_weight:.2f}_"
                                f"w{weak_weight:.2f}_b{bias_penalty:.1f}"
                            ),
                            "point_weight": point_weight,
                            "morph_weight": morph_weight,
                            "weak_rescue_weight": weak_weight,
                            "bias_penalty_weight": bias_penalty,
                        }
                    )
    return pd.DataFrame.from_records(records)


def morph_probs_for_row(
    row_index: int,
    density: float,
    morph_models,
    morph_columns,
    morph_lookup,
) -> tuple[dict[str, float], float]:
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
    return morph_probs, float(group_probs.get("irregular_local", 0.0))


def evaluate_tuning(args: argparse.Namespace, configs: pd.DataFrame) -> pd.DataFrame:
    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    densities = [float(v) for v in args.densities]

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

    test_ids = np.asarray(test_wafers)
    if args.max_test_wafers and len(test_ids) > args.max_test_wafers:
        rng = np.random.default_rng(args.seed)
        test_ids = rng.choice(test_ids, size=args.max_test_wafers, replace=False)
    eval_df = patterned[patterned.index.isin(set(int(v) for v in test_ids))]

    records: list[dict[str, object]] = []
    for pos, row in enumerate(eval_df.itertuples(index=True), start=1):
        row_index = int(row.Index)
        label = density_policy.failure_type(row)
        wafer_map = np.asarray(row.waferMap)
        for density in densities:
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, density)
            coverage = make_coverage_sampling_mask(
                wafer_map,
                n_points=args.top_k,
                existing_mask=first_mask,
            )
            coverage_metrics = sampling_metrics(wafer_map, first_mask | coverage)
            records.append(
                {
                    "row_index": row_index,
                    "failureType": label,
                    "target_density": density,
                    "config_name": "coverage32",
                    "strategy": "coverage32",
                    "replacement_count": 0,
                    "point_weight": 0.0,
                    "morph_weight": 0.0,
                    "weak_rescue_weight": 0.0,
                    "bias_penalty_weight": 0.0,
                    **coverage_metrics,
                }
            )

            candidates = candidate_feature_frame(
                wafer_map,
                first_pass_type=density_policy.density_key(density),
                first_mask=first_mask,
                row_index=row_index,
                failure_type=label,
                include_label=True,
            )
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["point_score"] = point_model.predict_proba(candidates[FEATURE_COLUMNS])[:, 1]
            morph_probs, group_irregular = morph_probs_for_row(
                row_index,
                density,
                morph_models,
                morph_columns,
                morph_lookup,
            )
            morph_prior = policy.pattern_prior_scores(candidates, morph_probs)
            weak_rescue = policy.weak_pattern_rescue_scores(candidates, morph_probs)

            for config in configs.itertuples(index=False):
                base_score = (
                    float(config.point_weight)
                    * candidates["point_score"].to_numpy(dtype=float)
                    + float(config.morph_weight) * morph_prior
                    + float(config.weak_rescue_weight) * weak_rescue * (1.0 + group_irregular)
                )
                for replacement_count in args.replacement_counts:
                    selected = density_policy.select_hybrid_coverage_morphrisk_candidates(
                        wafer_map,
                        first_mask,
                        coverage,
                        candidates,
                        base_score=base_score,
                        replacement_count=int(replacement_count),
                        bias_penalty_weight=float(config.bias_penalty_weight),
                        first_ratio_weight=args.first_ratio_weight,
                        global_target_ratio=global_target_ratio,
                    )
                    mask = density_policy.make_mask_from_points(wafer_map, first_mask, selected)
                    metrics = sampling_metrics(wafer_map, mask)
                    records.append(
                        {
                            "row_index": row_index,
                            "failureType": label,
                            "target_density": density,
                            "config_name": str(config.config_name),
                            "strategy": f"{config.config_name}_N{replacement_count}",
                            "replacement_count": int(replacement_count),
                            "point_weight": float(config.point_weight),
                            "morph_weight": float(config.morph_weight),
                            "weak_rescue_weight": float(config.weak_rescue_weight),
                            "bias_penalty_weight": float(config.bias_penalty_weight),
                            **metrics,
                        }
                    )
        if pos % 25 == 0 or pos == len(eval_df):
            print(f"hybrid tuning wafers evaluated: {pos:,}/{len(eval_df):,}")
    return pd.DataFrame.from_records(records)


def summarize(results: pd.DataFrame, abs_error_tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(
            [
                "target_density",
                "config_name",
                "strategy",
                "replacement_count",
                "point_weight",
                "morph_weight",
                "weak_rescue_weight",
                "bias_penalty_weight",
            ],
            observed=False,
        )
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
        )
        .reset_index()
    )
    baseline = summary[summary["strategy"] == "coverage32"][
        ["target_density", "mean_absolute_error", "mean_defect_coverage", "severe_miss_rate"]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_defect_coverage": "baseline_defect_coverage",
            "severe_miss_rate": "baseline_severe_miss_rate",
        }
    )
    summary = summary.merge(baseline, on="target_density", how="left")
    summary["absolute_error_delta"] = summary["mean_absolute_error"] - summary["baseline_absolute_error"]
    summary["defect_coverage_delta"] = (
        summary["mean_defect_coverage"] - summary["baseline_defect_coverage"]
    )
    summary["defect_coverage_relative_improvement_pct"] = (
        summary["defect_coverage_delta"] / summary["baseline_defect_coverage"] * 100.0
    )
    summary["guardrail_pass"] = (
        (summary["strategy"] != "coverage32")
        & (summary["absolute_error_delta"] <= abs_error_tolerance)
        & (summary["defect_coverage_delta"] > 0.0)
    )

    tuned = summary[summary["strategy"] != "coverage32"].copy()
    best_rows: list[pd.Series] = []
    for density, group in tuned.groupby("target_density", observed=False):
        allowed = group[group["guardrail_pass"]].copy()
        if allowed.empty:
            chosen = group.sort_values(
                ["absolute_error_delta", "defect_coverage_delta"],
                ascending=[True, False],
            ).iloc[0].copy()
            chosen["selection_status"] = "no_guardrail_pass"
        else:
            chosen = allowed.sort_values(
                ["defect_coverage_delta", "absolute_error_delta"],
                ascending=[False, True],
            ).iloc[0].copy()
            chosen["selection_status"] = "best_inside_guardrail"
        best_rows.append(chosen)
    best = pd.DataFrame(best_rows).reset_index(drop=True)

    pattern = (
        results.groupby(["target_density", "failureType", "strategy"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
        )
        .reset_index()
    )
    return summary, best, pattern


def plot_outputs(summary: pd.DataFrame, best: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    tuned = summary[summary["strategy"] != "coverage32"].copy()
    tuned["density_pct"] = tuned["target_density"] * 100.0
    best_keys = set(zip(best["target_density"], best["strategy"]))
    tuned["is_best"] = [
        (density, strategy) in best_keys
        for density, strategy in zip(tuned["target_density"], tuned["strategy"], strict=True)
    ]

    plt.figure(figsize=(9.0, 5.5))
    sns.scatterplot(
        data=tuned,
        x="absolute_error_delta",
        y="defect_coverage_relative_improvement_pct",
        hue="density_pct",
        style="guardrail_pass",
        alpha=0.72,
        palette="viridis",
    )
    best_points = tuned[tuned["is_best"]]
    plt.scatter(
        best_points["absolute_error_delta"],
        best_points["defect_coverage_relative_improvement_pct"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
    )
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Absolute-error delta vs coverage32")
    plt.ylabel("Defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "scoring_grid_gain_vs_error.png", dpi=180)
    plt.close()

    view = tuned[tuned["guardrail_pass"]].copy()
    if not view.empty:
        top = (
            view.sort_values(
                ["target_density", "defect_coverage_relative_improvement_pct"],
                ascending=[True, False],
            )
            .groupby("target_density", observed=False)
            .head(8)
        )
        top["density_config"] = (
            (top["target_density"] * 100.0).map(lambda value: f"{value:g}%")
            + " "
            + top["strategy"]
        )
        plt.figure(figsize=(11.5, 5.2))
        sns.barplot(
            data=top,
            x="density_config",
            y="defect_coverage_relative_improvement_pct",
            hue="replacement_count",
            dodge=False,
        )
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("Coverage improvement inside guardrail (%)")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(fig_dir / "top_guardrail_configs.png", dpi=180)
        plt.close()


def write_report(best: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Density Hybrid Scoring Tuning v1",
        "",
        "Objective:",
        "",
        "```text",
        "maximize follow-up defect coverage",
        "subject to absolute-error delta <= 0.01 vs coverage32",
        "```",
        "",
        "## Best Config by Density",
        "",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: {row.strategy}, "
            f"status={row.selection_status}, "
            f"coverage gain={row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta={row.absolute_error_delta:.4f}, "
            f"weights=(point {row.point_weight:.2f}, morph {row.morph_weight:.2f}, "
            f"weak {row.weak_rescue_weight:.2f}, bias {row.bias_penalty_weight:.1f})"
        )
    lines.extend(["", "## Guardrail Pass Count", ""])
    counts = (
        summary[summary["strategy"] != "coverage32"]
        .groupby("target_density", observed=False)["guardrail_pass"]
        .sum()
        .reset_index()
    )
    for row in counts.itertuples(index=False):
        lines.append(f"- {row.target_density:.0%}: {int(row.guardrail_pass)} candidate configs")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the first constrained tuning sweep. If a density has many guardrail-passing configs, the next step is a narrower local grid around the best weights.",
            "If 1% still fails or barely passes, keep coverage32 for strict operation and treat hybrid N=1 as a discovery-priority option.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    configs = score_configs(args.full_grid)
    configs.to_csv(args.out_dir / "scoring_config_grid.csv", index=False)
    results = evaluate_tuning(args, configs)
    summary, best, pattern = summarize(results, args.abs_error_tolerance)

    results.to_csv(args.out_dir / "density_hybrid_scoring_tuning_results.csv", index=False)
    summary.to_csv(args.out_dir / "density_hybrid_scoring_tuning_summary.csv", index=False)
    best.to_csv(args.out_dir / "density_hybrid_scoring_tuning_best.csv", index=False)
    pattern.to_csv(args.out_dir / "density_hybrid_scoring_tuning_pattern_summary.csv", index=False)
    plot_outputs(summary, best, args.fig_dir)
    write_report(best, summary, args.out_dir / "density_hybrid_scoring_tuning_report.md")

    print(f"wrote tuning outputs to {args.out_dir}")
    print(f"wrote tuning figures to {args.fig_dir}")
    print(
        best[
            [
                "target_density",
                "strategy",
                "selection_status",
                "defect_coverage_relative_improvement_pct",
                "absolute_error_delta",
                "point_weight",
                "morph_weight",
                "weak_rescue_weight",
                "bias_penalty_weight",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
