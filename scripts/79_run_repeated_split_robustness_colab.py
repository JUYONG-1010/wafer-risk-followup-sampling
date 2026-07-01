from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe repeated split robustness with per-seed CNN retraining."
    )
    parser.add_argument("--patterned", type=Path, default=Path("data/processed/subsets/patterned_subset.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/repeated_split_robustness_colab_v1"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/79_repeated_split_robustness_colab_v1"))
    parser.add_argument("--model-root", type=Path, default=Path("models/repeated_split_robustness_colab_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-wafers", type=int, default=2500)
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--point-estimators", type=int, default=100)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def seed_paths(args: argparse.Namespace, seed: int) -> dict[str, Path]:
    seed_name = f"seed_{seed}"
    return {
        "cnn_out": args.output_root / seed_name / "cnn",
        "cnn_fig": args.figure_root / seed_name / "cnn",
        "model": args.model_root / f"sparse_cnn_seed_{seed}.pt",
        "ensemble_out": args.output_root / seed_name / "ensemble",
        "ensemble_fig": args.figure_root / seed_name / "ensemble",
    }


def run_seed(args: argparse.Namespace, seed: int) -> None:
    paths = seed_paths(args, seed)
    ensemble_summary = paths["ensemble_out"] / "cnn_noncnn_ensemble_summary.csv"
    if args.skip_existing and ensemble_summary.exists():
        print(f"seed {seed}: existing summary found, skipping", flush=True)
        return

    density_args = [str(value) for value in args.densities]
    cnn_command = [
        sys.executable,
        "scripts/71_train_sparse_cnn_risk_map_batched.py",
        "--patterned",
        str(args.patterned),
        "--out-dir",
        str(paths["cnn_out"]),
        "--fig-dir",
        str(paths["cnn_fig"]),
        "--model-path",
        str(paths["model"]),
        "--densities",
        *density_args,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--max-train-wafers",
        str(args.max_train_wafers),
        "--max-test-wafers",
        str(args.max_test_wafers),
        "--hidden-channels",
        str(args.hidden_channels),
        "--batch-size",
        str(args.batch_size),
        "--top-k",
        str(args.top_k),
        "--threads",
        str(args.threads),
    ]
    if args.use_amp:
        cnn_command.append("--use-amp")
    run_command(cnn_command)

    ensemble_command = [
        sys.executable,
        "scripts/76_evaluate_cnn_noncnn_ensemble.py",
        "--patterned",
        str(args.patterned),
        "--cnn-model",
        str(paths["model"]),
        "--out-dir",
        str(paths["ensemble_out"]),
        "--fig-dir",
        str(paths["ensemble_fig"]),
        "--densities",
        *density_args,
        "--top-k",
        str(args.top_k),
        "--weights",
        "0.3",
        "--seed",
        str(seed),
        "--max-train-wafers",
        str(args.max_train_wafers),
        "--max-test-wafers",
        str(args.max_test_wafers),
        "--max-defect-candidates",
        str(args.max_defect_candidates),
        "--max-normal-candidates",
        str(args.max_normal_candidates),
        "--point-estimators",
        str(args.point_estimators),
        "--n-jobs",
        str(args.n_jobs),
        "--threads",
        str(args.threads),
    ]
    run_command(ensemble_command)


def collect_policy_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts: list[pd.DataFrame] = []
    pattern_parts: list[pd.DataFrame] = []
    for seed in args.seeds:
        paths = seed_paths(args, int(seed))
        summary_path = paths["ensemble_out"] / "cnn_noncnn_ensemble_summary.csv"
        pattern_path = paths["ensemble_out"] / "cnn_noncnn_ensemble_pattern_summary.csv"
        if not summary_path.exists():
            print(f"missing summary for seed {seed}: {summary_path}", flush=True)
            continue
        summary = pd.read_csv(summary_path)
        summary["seed"] = int(seed)
        pattern = pd.read_csv(pattern_path)
        pattern["seed"] = int(seed)
        summary_parts.append(summary)
        pattern_parts.append(pattern)
    if not summary_parts:
        raise RuntimeError("No seed summaries were found.")
    return pd.concat(summary_parts, ignore_index=True), pd.concat(pattern_parts, ignore_index=True)


def policy_label(row: pd.Series) -> str:
    policy = str(row["policy_name"])
    mode = str(row.get("ensemble_mode", ""))
    weight = row.get("noncnn_weight")
    if policy == "coverage":
        return "coverage32"
    if policy == "noncnn":
        return "noncnn_top32"
    if policy == "cnn":
        return "cnn_top32"
    if mode == "raw" and pd.notna(weight) and abs(float(weight) - 0.3) < 1e-9:
        return "ensemble_raw_w0.30"
    return policy


def aggregate(args: argparse.Namespace) -> None:
    summary, pattern = collect_policy_rows(args)
    summary["final_policy"] = summary.apply(policy_label, axis=1)
    pattern["final_policy"] = pattern.apply(policy_label, axis=1)
    keep = ["coverage32", "noncnn_top32", "cnn_top32", "ensemble_raw_w0.30"]
    summary = summary[(summary["split"] == "test") & summary["final_policy"].isin(keep)].copy()
    pattern = pattern[(pattern["split"] == "test") & pattern["final_policy"].isin(keep)].copy()

    metric_cols = [
        "mean_followup_defects",
        "mean_followup_precision_at_k",
        "mean_defect_coverage",
        "mean_defect_coverage_gain_pct",
        "mean_absolute_error_delta",
        "severe_miss_rate",
    ]
    robust = (
        summary.groupby(["target_density", "top_k", "final_policy"], dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    robust.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in robust.columns
    ]

    focus_patterns = ["Scratch", "Loc", "Edge-Ring", "Donut", "Center"]
    pattern_focus = pattern[pattern["failureType"].isin(focus_patterns)].copy()
    pattern_metrics = [
        "mean_followup_defects",
        "mean_followup_precision_at_k",
        "mean_defect_coverage",
        "mean_absolute_error_delta",
        "severe_miss_rate",
    ]
    robust_pattern = (
        pattern_focus.groupby(["target_density", "failureType", "top_k", "final_policy"], dropna=False)[pattern_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    robust_pattern.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in robust_pattern.columns
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "repeated_split_policy_seed_summaries.csv", index=False)
    pattern.to_csv(args.output_root / "repeated_split_pattern_seed_summaries.csv", index=False)
    robust.to_csv(args.output_root / "repeated_split_policy_robustness_summary.csv", index=False)
    robust_pattern.to_csv(args.output_root / "repeated_split_pattern_robustness_summary.csv", index=False)
    write_report(args, robust, robust_pattern)
    print(f"wrote repeated split robustness outputs to {args.output_root}", flush=True)
    print(robust.round(5).to_string(index=False), flush=True)


def frame_to_markdown(frame: pd.DataFrame) -> str:
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


def write_report(args: argparse.Namespace, robust: pd.DataFrame, robust_pattern: pd.DataFrame) -> None:
    selected_cols = [
        "target_density",
        "top_k",
        "final_policy",
        "mean_followup_precision_at_k_mean",
        "mean_followup_precision_at_k_std",
        "mean_defect_coverage_mean",
        "mean_defect_coverage_std",
        "mean_followup_defects_mean",
        "mean_followup_defects_std",
        "severe_miss_rate_mean",
    ]
    pattern_cols = [
        "target_density",
        "failureType",
        "top_k",
        "final_policy",
        "mean_followup_precision_at_k_mean",
        "mean_followup_precision_at_k_std",
        "mean_defect_coverage_mean",
        "mean_defect_coverage_std",
    ]
    lines = [
        "# Repeated Split Robustness v1",
        "",
        "Purpose: test whether the final candidate policies are stable across repeated wafer-level train/validation/test splits.",
        "",
        "Protocol:",
        "",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- CNN is retrained separately for each seed.",
        f"- non-CNN point-risk model is retrained separately for each seed.",
        f"- top-k: {args.top_k}",
        f"- max train wafers: {args.max_train_wafers}",
        f"- max test wafers: {args.max_test_wafers}",
        "",
        "Final policy candidates:",
        "",
        "- `coverage32`",
        "- `noncnn_top32`",
        "- `cnn_top32`",
        "- `ensemble_raw_w0.30`",
        "",
        "## Global Robustness Summary",
        "",
        frame_to_markdown(robust[selected_cols].round(5)),
        "",
        "## Pattern Focus Robustness Summary",
        "",
        frame_to_markdown(robust_pattern[pattern_cols].round(5)),
        "",
    ]
    (args.output_root / "repeated_split_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in [args.output_root, args.figure_root, args.model_root]:
        path.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(args, int(seed))
    aggregate(args)


if __name__ == "__main__":
    main()
