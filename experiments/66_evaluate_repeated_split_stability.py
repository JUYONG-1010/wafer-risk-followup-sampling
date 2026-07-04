from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "repeated_split_stability_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "56_repeated_split_stability_v1"
DENSITY_POLICY_SCRIPT = Path("experiments") / "47_evaluate_density_followup_policy.py"

FOCUS_STRATEGIES = [
    "coverage32",
    "ml_rank32",
    "morphrisk32",
    "morphrisk_guarded32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated split stability audit for follow-up policy metrics."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-wafers", type=int, default=250)
    parser.add_argument("--max-test-wafers", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-defect-candidates", type=int, default=50)
    parser.add_argument("--max-normal-candidates", type=int, default=80)
    parser.add_argument("--point-estimators", type=int, default=30)
    parser.add_argument("--morph-estimators", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    parser.add_argument("--summarize-partial-only", action="store_true")
    return parser.parse_args()


def policy_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        patterned=args.patterned,
        morph_dataset=args.morph_dataset,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        densities=[float(v) for v in args.densities],
        test_size=float(args.test_size),
        seed=int(seed),
        max_train_wafers=int(args.max_train_wafers),
        max_test_wafers=int(args.max_test_wafers),
        top_k=int(args.top_k),
        max_defect_candidates=int(args.max_defect_candidates),
        max_normal_candidates=int(args.max_normal_candidates),
        point_estimators=int(args.point_estimators),
        morph_estimators=int(args.morph_estimators),
        n_jobs=int(args.n_jobs),
        point_weight=0.60,
        morph_weight=0.30,
        weak_rescue_weight=0.25,
        diversity_weight=0.40,
        uncertainty_diversity_weight=0.35,
        bias_weight=1.0,
        first_ratio_weight=float(args.first_ratio_weight),
        cost_weights=[0.003],
        sweep=True,
        hybrid_replacements=[1, 2, 4],
    )


def run_one_seed(
    patterned: pd.DataFrame,
    morph_data: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_args = policy_args(args, seed)
    densities = [float(v) for v in run_args.densities]
    train_wafers, test_wafers = density_policy.split_wafers(
        patterned,
        test_size=run_args.test_size,
        seed=seed,
    )
    point_train = density_policy.build_point_training_data(patterned, train_wafers, densities, run_args)
    point_model = density_policy.train_point_model(point_train, run_args)
    morph_models, morph_columns, morph_lookup = density_policy.train_morph_models(
        morph_data,
        train_wafers,
        densities,
        run_args,
    )
    global_target_ratio = density_policy.mean_actual_defect_ratio(patterned, train_wafers)
    eval_results = density_policy.evaluate_policy(
        patterned,
        test_wafers,
        densities,
        point_model,
        morph_models,
        morph_columns,
        morph_lookup,
        global_target_ratio,
        run_args,
    )
    eval_results["split_seed"] = int(seed)
    summary = density_policy.summarize_eval(eval_results)
    summary["split_seed"] = int(seed)
    model_summary = pd.DataFrame(
        [
            {
                "split_seed": int(seed),
                "train_wafers_total": int(len(train_wafers)),
                "test_wafers_total": int(len(test_wafers)),
                "max_train_wafers": int(run_args.max_train_wafers),
                "max_test_wafers": int(run_args.max_test_wafers),
                "point_train_rows": int(len(point_train)),
                "global_target_ratio": float(global_target_ratio),
                **density_policy.point_model_metrics(point_model, point_train),
            }
        ]
    )
    return eval_results, summary, model_summary


def add_coverage_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary[summary["strategy"].isin(FOCUS_STRATEGIES)].copy()
    if "mean_sampled_defects" not in data.columns:
        data["mean_sampled_defects"] = np.nan
    baseline = data[data["strategy"] == "coverage32"][
        [
            "split_seed",
            "target_density",
            "mean_absolute_error",
            "mean_ratio_error",
            "severe_miss_rate",
            "mean_defect_coverage",
            "mean_sampled_defects",
        ]
    ].rename(
        columns={
            "mean_absolute_error": "baseline_absolute_error",
            "mean_ratio_error": "baseline_ratio_error",
            "severe_miss_rate": "baseline_severe_miss_rate",
            "mean_defect_coverage": "baseline_defect_coverage",
            "mean_sampled_defects": "baseline_sampled_defects",
        }
    )
    out = data.merge(baseline, on=["split_seed", "target_density"], how="left")
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    out["ratio_error_delta"] = out["mean_ratio_error"] - out["baseline_ratio_error"]
    out["severe_miss_delta"] = out["severe_miss_rate"] - out["baseline_severe_miss_rate"]
    out["defect_coverage_gain_pct"] = (
        (out["mean_defect_coverage"] - out["baseline_defect_coverage"])
        / out["baseline_defect_coverage"]
        * 100.0
    )
    out["sampled_defects_delta"] = out["mean_sampled_defects"] - out["baseline_sampled_defects"]
    return out


def stability_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    data = deltas[deltas["strategy"] != "coverage32"].copy()
    return (
        data.groupby(["target_density", "strategy"], observed=False)
        .agg(
            seeds=("split_seed", "nunique"),
            gain_mean=("defect_coverage_gain_pct", "mean"),
            gain_std=("defect_coverage_gain_pct", "std"),
            gain_min=("defect_coverage_gain_pct", "min"),
            gain_max=("defect_coverage_gain_pct", "max"),
            gain_positive_rate=("defect_coverage_gain_pct", lambda values: float((values > 0).mean())),
            abs_delta_mean=("absolute_error_delta", "mean"),
            abs_delta_std=("absolute_error_delta", "std"),
            severe_miss_delta_mean=("severe_miss_delta", "mean"),
            sampled_defects_delta_mean=("sampled_defects_delta", "mean"),
        )
        .reset_index()
    )


def plot_outputs(deltas: pd.DataFrame, stability: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = deltas[
        deltas["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1", "hybrid_guarded4"])
    ].copy()
    focus["density_pct"] = focus["target_density"] * 100.0
    plt.figure(figsize=(9.4, 5.4))
    sns.lineplot(
        data=focus,
        x="density_pct",
        y="defect_coverage_gain_pct",
        hue="strategy",
        style="split_seed",
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "repeated_split_coverage_gain_by_seed.png", dpi=180)
    plt.close()

    stab = stability[
        stability["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1", "hybrid_guarded4"])
    ].copy()
    stab["density_pct"] = stab["target_density"] * 100.0
    plt.figure(figsize=(9.4, 5.4))
    for strategy, group in stab.groupby("strategy", observed=False):
        group = group.sort_values("density_pct")
        yerr = group["gain_std"].fillna(0.0)
        plt.errorbar(
            group["density_pct"],
            group["gain_mean"],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=strategy,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Mean coverage gain vs coverage32 (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "repeated_split_gain_mean_std.png", dpi=180)
    plt.close()


def write_report(stability: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        "# Repeated Split Stability v1",
        "",
        "Purpose: check whether coverage gain is stable across multiple train/test split seeds.",
        "",
        "This is a compact audit, not the final large-scale benchmark.",
        "",
        "## Focus Policies",
        "",
    ]
    focus = stability[
        stability["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1", "hybrid_guarded4"])
    ].copy()
    for row in focus.sort_values(["target_density", "strategy"]).itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, {row.strategy}: "
            f"gain mean {row.gain_mean:.2f}%, std {row.gain_std:.2f}, "
            f"min {row.gain_min:.2f}%, max {row.gain_max:.2f}%, "
            f"P(gain>0) {row.gain_positive_rate:.3f}, "
            f"abs-delta mean {row.abs_delta_mean:.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stable positive gain across seeds supports that the model is not only benefiting from one lucky split.",
            "If a policy has high gain but also high absolute-error delta, treat it as discovery-first rather than unbiased ratio estimation.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_partial_only:
        partial_path = args.out_dir / "repeated_split_summary_partial.csv"
        if not partial_path.exists():
            raise FileNotFoundError(partial_path)
        summary_all = pd.read_csv(partial_path)
        deltas = add_coverage_deltas(summary_all)
        stability = stability_summary(deltas)
        summary_all.to_csv(args.out_dir / "repeated_split_eval_summary.csv", index=False)
        deltas.to_csv(args.out_dir / "repeated_split_vs_coverage32.csv", index=False)
        stability.to_csv(args.out_dir / "repeated_split_stability_summary.csv", index=False)
        plot_outputs(deltas, stability, args.fig_dir)
        write_report(stability, args.out_dir / "repeated_split_stability_report.md")
        print(f"summarized partial repeated split outputs from {partial_path}")
        print(
            stability[
                stability["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1", "hybrid_guarded4"])
            ]
            .round(4)
            .to_string(index=False)
        )
        return

    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    eval_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        eval_results, summary, model_summary = run_one_seed(patterned, morph_data, args, int(seed))
        eval_frames.append(eval_results)
        summary_frames.append(summary)
        model_frames.append(model_summary)
        pd.concat(summary_frames, ignore_index=True).to_csv(
            args.out_dir / "repeated_split_summary_partial.csv",
            index=False,
        )
        print(f"completed repeated split seed: {seed}")

    eval_all = pd.concat(eval_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    model_all = pd.concat(model_frames, ignore_index=True)
    deltas = add_coverage_deltas(summary_all)
    stability = stability_summary(deltas)

    eval_all.to_csv(args.out_dir / "repeated_split_eval_results.csv", index=False)
    summary_all.to_csv(args.out_dir / "repeated_split_eval_summary.csv", index=False)
    model_all.to_csv(args.out_dir / "repeated_split_model_summary.csv", index=False)
    deltas.to_csv(args.out_dir / "repeated_split_vs_coverage32.csv", index=False)
    stability.to_csv(args.out_dir / "repeated_split_stability_summary.csv", index=False)
    plot_outputs(deltas, stability, args.fig_dir)
    write_report(stability, args.out_dir / "repeated_split_stability_report.md")

    print(f"wrote repeated split outputs to {args.out_dir}")
    print(f"wrote repeated split figures to {args.fig_dir}")
    print(
        stability[
            stability["strategy"].isin(["ml_rank32", "morphrisk_guarded32", "hybrid_guarded1", "hybrid_guarded4"])
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
