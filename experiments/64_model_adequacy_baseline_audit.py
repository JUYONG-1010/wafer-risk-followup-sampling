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

from src.sampling import sampling_metrics, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_EVAL_DIR = Path("data") / "processed" / "density_followup_hybrid_small_v1"
DEFAULT_OBJECTIVE_DIR = Path("data") / "processed" / "objective_mode_operating_policy_v1"
DEFAULT_OUT_DIR = Path("data") / "processed" / "model_adequacy_baseline_audit_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "54_model_adequacy_baseline_audit_v1"

FIXED_POLICIES = [
    "coverage32",
    "ml_rank32",
    "morphrisk32",
    "morphrisk_guarded32",
    "hybrid_guarded1",
    "hybrid_guarded2",
    "hybrid_guarded4",
]


def load_density_policy_module():
    module_path = PROJECT_ROOT / "experiments" / "47_evaluate_density_followup_policy.py"
    spec = importlib.util.spec_from_file_location("density_policy47", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_density_policy_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether current follow-up models are useful against matched baselines."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--objective-dir", type=Path, default=DEFAULT_OBJECTIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--random-seeds", type=int, default=30)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--objective-threshold", type=float, default=0.60)
    return parser.parse_args()


def metric_columns() -> list[str]:
    return [
        "valid_die_count",
        "total_defects",
        "sampled_valid_count",
        "sampled_defects",
        "sampling_density",
        "actual_defect_ratio",
        "sampled_defect_ratio",
        "ratio_error",
        "absolute_error",
        "defect_coverage",
        "miss_rate",
        "hit",
        "underestimated",
        "severe_miss",
    ]


def load_fixed_policy_rows(eval_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(eval_dir / "density_followup_eval_results.csv")
    min_cost = data["cost_weight"].min()
    data = data[(data["cost_weight"] == min_cost) & (data["strategy"].isin(FIXED_POLICIES))].copy()
    data["policy_name"] = data["strategy"]
    keep = ["row_index", "failureType", "target_density", "policy_name", *metric_columns()]
    return data[keep].copy()


def load_objective_policy_rows(objective_dir: Path, threshold: float) -> pd.DataFrame:
    data = pd.read_csv(objective_dir / "objective_mode_policy_results.csv")
    focus = data[
        np.isclose(data["confidence_threshold"], threshold)
        & (data["prediction_variant"].isin(["baseline", "augmented", "oracle"]))
    ].copy()
    focus["policy_name"] = focus["prediction_variant"].map(
        {
            "baseline": f"objective_baseline_t{threshold:.2f}",
            "augmented": f"objective_augmented_t{threshold:.2f}",
            "oracle": f"objective_oracle_t{threshold:.2f}",
        }
    )
    keep = ["row_index", "failureType", "target_density", "policy_name", *metric_columns()]
    return focus[keep].copy()


def matched_random_rows(
    patterned: pd.DataFrame,
    eval_rows: pd.DataFrame,
    top_k: int,
    n_seeds: int,
) -> pd.DataFrame:
    base = eval_rows[eval_rows["policy_name"] == "coverage32"][
        ["row_index", "failureType", "target_density"]
    ].drop_duplicates()
    records: list[dict[str, object]] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        for row in base.itertuples(index=False):
            wafer_map = np.asarray(patterned.loc[int(row.row_index), "waferMap"])
            first_mask = density_policy.make_initial_coverage_mask(wafer_map, float(row.target_density))
            valid = valid_die_mask(wafer_map)
            coords = np.column_stack(np.nonzero(valid & ~first_mask))
            selected_mask = first_mask.copy()
            if len(coords):
                take = min(top_k, len(coords))
                selected = coords[rng.choice(len(coords), size=take, replace=False)]
                selected_mask[selected[:, 0], selected[:, 1]] = True
            metrics = sampling_metrics(wafer_map, selected_mask)
            records.append(
                {
                    "row_index": int(row.row_index),
                    "failureType": str(row.failureType),
                    "target_density": float(row.target_density),
                    "policy_name": "matched_random32",
                    "random_seed": seed,
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def averaged_random_rows(random_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        random_rows.groupby(["row_index", "failureType", "target_density", "policy_name"], observed=False)[
            metric_columns()
        ]
        .mean()
        .reset_index()
    )
    return grouped


def summarize_policy_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["target_density", "policy_name"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            mean_sampled_defects=("sampled_defects", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
        )
        .reset_index()
    )


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["policy_name"] == "coverage32"][
        [
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
    out = summary.merge(baseline, on="target_density", how="left")
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


def paired_bootstrap(rows: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for density, density_rows in rows.groupby("target_density", observed=False):
        base = density_rows[density_rows["policy_name"] == "coverage32"][
            ["row_index", "defect_coverage", "absolute_error", "sampled_defects", "severe_miss"]
        ].rename(
            columns={
                "defect_coverage": "base_defect_coverage",
                "absolute_error": "base_absolute_error",
                "sampled_defects": "base_sampled_defects",
                "severe_miss": "base_severe_miss",
            }
        )
        for policy_name, policy_rows in density_rows.groupby("policy_name", observed=False):
            if policy_name == "coverage32":
                continue
            paired = policy_rows[
                ["row_index", "defect_coverage", "absolute_error", "sampled_defects", "severe_miss"]
            ].merge(base, on="row_index", how="inner")
            if paired.empty:
                continue
            values = paired.to_dict("records")
            n = len(values)
            coverage_gains: list[float] = []
            abs_deltas: list[float] = []
            sampled_defect_deltas: list[float] = []
            severe_miss_deltas: list[float] = []
            for _ in range(iterations):
                idx = rng.integers(0, n, size=n)
                sample = [values[int(i)] for i in idx]
                mean_policy_cov = np.mean([float(v["defect_coverage"]) for v in sample])
                mean_base_cov = np.mean([float(v["base_defect_coverage"]) for v in sample])
                coverage_gains.append((mean_policy_cov - mean_base_cov) / mean_base_cov * 100.0)
                abs_deltas.append(
                    np.mean([float(v["absolute_error"]) - float(v["base_absolute_error"]) for v in sample])
                )
                sampled_defect_deltas.append(
                    np.mean([float(v["sampled_defects"]) - float(v["base_sampled_defects"]) for v in sample])
                )
                severe_miss_deltas.append(
                    np.mean([float(v["severe_miss"]) - float(v["base_severe_miss"]) for v in sample])
                )
            records.append(
                {
                    "target_density": float(density),
                    "policy_name": str(policy_name),
                    "paired_wafers": n,
                    "coverage_gain_mean": float(np.mean(coverage_gains)),
                    "coverage_gain_ci_low": float(np.percentile(coverage_gains, 2.5)),
                    "coverage_gain_ci_high": float(np.percentile(coverage_gains, 97.5)),
                    "coverage_gain_prob_positive": float(np.mean(np.asarray(coverage_gains) > 0)),
                    "absolute_error_delta_mean": float(np.mean(abs_deltas)),
                    "absolute_error_delta_ci_low": float(np.percentile(abs_deltas, 2.5)),
                    "absolute_error_delta_ci_high": float(np.percentile(abs_deltas, 97.5)),
                    "sampled_defects_delta_mean": float(np.mean(sampled_defect_deltas)),
                    "sampled_defects_delta_ci_low": float(np.percentile(sampled_defect_deltas, 2.5)),
                    "sampled_defects_delta_ci_high": float(np.percentile(sampled_defect_deltas, 97.5)),
                    "severe_miss_delta_mean": float(np.mean(severe_miss_deltas)),
                    "severe_miss_delta_ci_low": float(np.percentile(severe_miss_deltas, 2.5)),
                    "severe_miss_delta_ci_high": float(np.percentile(severe_miss_deltas, 97.5)),
                }
            )
    return pd.DataFrame.from_records(records)


def pattern_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["target_density", "failureType", "policy_name"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_defect_coverage=("defect_coverage", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
        )
        .reset_index()
    )
    baseline = summary[summary["policy_name"] == "coverage32"][
        ["target_density", "failureType", "mean_defect_coverage", "mean_absolute_error"]
    ].rename(
        columns={
            "mean_defect_coverage": "baseline_defect_coverage",
            "mean_absolute_error": "baseline_absolute_error",
        }
    )
    out = summary.merge(baseline, on=["target_density", "failureType"], how="left")
    out["defect_coverage_gain_pct"] = (
        (out["mean_defect_coverage"] - out["baseline_defect_coverage"])
        / out["baseline_defect_coverage"]
        * 100.0
    )
    out["absolute_error_delta"] = out["mean_absolute_error"] - out["baseline_absolute_error"]
    return out


def plot_outputs(summary: pd.DataFrame, bootstrap: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = summary[
        summary["policy_name"].isin(
            [
                "matched_random32",
                "ml_rank32",
                "objective_augmented_t0.60",
                "morphrisk_guarded32",
                "hybrid_guarded1",
            ]
        )
    ].copy()
    focus["density_pct"] = focus["target_density"] * 100.0

    plt.figure(figsize=(9.4, 5.4))
    sns.lineplot(
        data=focus,
        x="density_pct",
        y="defect_coverage_gain_pct",
        hue="policy_name",
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Defect coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "adequacy_policy_coverage_gain.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.8, 5.2))
    sns.scatterplot(
        data=focus[focus["policy_name"] != "coverage32"],
        x="absolute_error_delta",
        y="defect_coverage_gain_pct",
        hue="policy_name",
        style="density_pct",
        s=90,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.xlabel("Absolute-error delta vs coverage32")
    plt.ylabel("Defect coverage gain vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "adequacy_gain_vs_warning.png", dpi=180)
    plt.close()

    boot_focus = bootstrap[
        bootstrap["policy_name"].isin(["matched_random32", "ml_rank32", "objective_augmented_t0.60"])
    ].copy()
    boot_focus["density_pct"] = boot_focus["target_density"] * 100.0
    plt.figure(figsize=(9.4, 5.4))
    for policy_name, group in boot_focus.groupby("policy_name", observed=False):
        group = group.sort_values("density_pct")
        plt.errorbar(
            group["density_pct"],
            group["coverage_gain_mean"],
            yerr=[
                group["coverage_gain_mean"] - group["coverage_gain_ci_low"],
                group["coverage_gain_ci_high"] - group["coverage_gain_mean"],
            ],
            marker="o",
            capsize=4,
            label=policy_name,
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Bootstrap coverage gain vs coverage32 (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bootstrap_coverage_gain_ci.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        "# Model Adequacy Baseline Audit v1",
        "",
        "Purpose: check whether the current follow-up models are useful against matched baselines on the same held-out wafer subset.",
        "",
        "Compared policies:",
        "",
        "```text",
        "coverage32: geometry-only representative baseline",
        "matched_random32: first-pass + 32 random remaining dies, averaged over seeds",
        "ml_rank32: pure point-risk top32",
        "objective_augmented_t0.60: deployable objective-mode policy using augmented morphology, threshold 0.60",
        "```",
        "",
        "## Main Summary",
        "",
    ]
    focus_names = ["matched_random32", "ml_rank32", "objective_augmented_t0.60"]
    focus = summary[summary["policy_name"].isin(focus_names)].copy()
    for row in focus.sort_values(["target_density", "policy_name"]).itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, {row.policy_name}: "
            f"coverage gain {row.defect_coverage_gain_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f}, "
            f"sampled-defects delta {row.sampled_defects_delta:.2f}"
        )
    lines.extend(["", "## Bootstrap 95% CI vs coverage32", ""])
    boot = bootstrap[bootstrap["policy_name"].isin(focus_names)].copy()
    for row in boot.sort_values(["target_density", "policy_name"]).itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, {row.policy_name}: "
            f"coverage gain {row.coverage_gain_mean:.2f}% "
            f"[{row.coverage_gain_ci_low:.2f}, {row.coverage_gain_ci_high:.2f}], "
            f"P(gain>0)={row.coverage_gain_prob_positive:.3f}, "
            f"abs-delta {row.absolute_error_delta_mean:.4f} "
            f"[{row.absolute_error_delta_ci_low:.4f}, {row.absolute_error_delta_ci_high:.4f}]"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the lower CI bound for coverage gain stays above zero, the policy is not just a one-split visual artifact on this held-out subset.",
            "This is still not a repeated train/test split audit; it is a paired bootstrap adequacy check on the current held-out evaluation.",
            "The next audit step should test learning curves and repeated split stability.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = load_fixed_policy_rows(args.eval_dir)
    objective_rows = load_objective_policy_rows(args.objective_dir, args.objective_threshold)
    patterned = pd.read_pickle(args.patterned)
    random_seed_rows = matched_random_rows(patterned, fixed_rows, args.top_k, args.random_seeds)
    random_rows = averaged_random_rows(random_seed_rows)
    all_rows = pd.concat([fixed_rows, objective_rows, random_rows], ignore_index=True)

    summary = add_baseline_deltas(summarize_policy_rows(all_rows))
    bootstrap = paired_bootstrap(all_rows, args.bootstrap_iterations, args.bootstrap_seed)
    patterns = pattern_summary(all_rows)

    random_seed_rows.to_csv(args.out_dir / "matched_random32_by_seed.csv", index=False)
    all_rows.to_csv(args.out_dir / "model_adequacy_policy_rows.csv", index=False)
    summary.to_csv(args.out_dir / "model_adequacy_policy_summary.csv", index=False)
    bootstrap.to_csv(args.out_dir / "model_adequacy_bootstrap_ci.csv", index=False)
    patterns.to_csv(args.out_dir / "model_adequacy_pattern_summary.csv", index=False)
    plot_outputs(summary, bootstrap, args.fig_dir)
    write_report(summary, bootstrap, args.out_dir / "model_adequacy_baseline_audit_report.md")

    print(f"wrote model adequacy audit outputs to {args.out_dir}")
    print(f"wrote model adequacy audit figures to {args.fig_dir}")
    print(
        summary[
            summary["policy_name"].isin(["matched_random32", "ml_rank32", "objective_augmented_t0.60"])
        ][
            [
                "target_density",
                "policy_name",
                "defect_coverage_gain_pct",
                "absolute_error_delta",
                "sampled_defects_delta",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        bootstrap[
            bootstrap["policy_name"].isin(["matched_random32", "ml_rank32", "objective_augmented_t0.60"])
        ][
            [
                "target_density",
                "policy_name",
                "coverage_gain_mean",
                "coverage_gain_ci_low",
                "coverage_gain_ci_high",
                "coverage_gain_prob_positive",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
