from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_71 = PROJECT_ROOT / "experiments" / "71_train_sparse_cnn_risk_map_batched.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small tuning grid for the batched sparse CNN pipeline."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-wafers", type=int, default=2500)
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--root-out-dir", type=Path, default=Path("data/processed/batched_cnn_tuning_grid"))
    parser.add_argument("--root-fig-dir", type=Path, default=Path("outputs/figures/72_batched_cnn_tuning_grid"))
    parser.add_argument("--root-model-dir", type=Path, default=Path("models/batched_cnn_tuning_grid"))
    parser.add_argument("--root-log-dir", type=Path, default=Path("reports/batched_cnn_tuning_grid"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--include-coordinate-ablation",
        action="store_true",
        help="Also run the best-prior baseline without y/x/radius coordinate channels.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small CPU-friendly smoke grid for checking the runner itself.",
    )
    return parser.parse_args()


def grid_configs(include_coordinate_ablation: bool, quick: bool) -> list[dict[str, object]]:
    if quick:
        return [
            {
                "name": "quick_b8_lr1e3_amp_off",
                "batch_size": 8,
                "learning_rate": 1e-3,
                "use_amp": False,
                "drop_coordinate_channels": False,
            }
        ]

    configs: list[dict[str, object]] = [
        {
            "name": "b8_lr1e3_amp_off",
            "batch_size": 8,
            "learning_rate": 1e-3,
            "use_amp": False,
            "drop_coordinate_channels": False,
        },
        {
            "name": "b16_lr1e3_amp_off",
            "batch_size": 16,
            "learning_rate": 1e-3,
            "use_amp": False,
            "drop_coordinate_channels": False,
        },
        {
            "name": "b16_lr5e4_amp_on",
            "batch_size": 16,
            "learning_rate": 5e-4,
            "use_amp": True,
            "drop_coordinate_channels": False,
        },
        {
            "name": "b32_lr5e4_amp_on",
            "batch_size": 32,
            "learning_rate": 5e-4,
            "use_amp": True,
            "drop_coordinate_channels": False,
        },
    ]
    if include_coordinate_ablation:
        configs.append(
            {
                "name": "coord_ablation_b16_lr1e3_amp_off",
                "batch_size": 16,
                "learning_rate": 1e-3,
                "use_amp": False,
                "drop_coordinate_channels": True,
            }
        )
    return configs


def run_one(args: argparse.Namespace, config: dict[str, object]) -> Path:
    run_name = str(config["name"])
    out_dir = args.root_out_dir / run_name
    fig_dir = args.root_fig_dir / run_name
    model_path = args.root_model_dir / f"{run_name}.pt"
    log_path = args.root_log_dir / f"{run_name}.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_71),
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
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--num-workers",
        str(args.num_workers),
        "--threads",
        str(args.threads),
        "--out-dir",
        str(out_dir),
        "--fig-dir",
        str(fig_dir),
        "--model-path",
        str(model_path),
        "--densities",
        *[str(value) for value in args.densities],
    ]
    if args.cpu_only:
        cmd.append("--cpu-only")
    if bool(config["use_amp"]):
        cmd.append("--use-amp")
    if bool(config["drop_coordinate_channels"]):
        cmd.append("--drop-coordinate-channels")

    print(f"\n=== Running {run_name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(process.stdout, end="")
        log_file.write(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"{run_name} failed with exit code {process.returncode}; see {log_path}")
    return out_dir


def collect_summary(args: argparse.Namespace, run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        summary_path = run_dir / "sparse_cnn_eval_summary.csv"
        history_path = run_dir / "sparse_cnn_training_history.csv"
        report_path = run_dir / "sparse_cnn_batched_report.md"
        summary = pd.read_csv(summary_path)
        history = pd.read_csv(history_path)
        best_metric = "val_mean_average_precision"
        best_row = history.sort_values(best_metric, ascending=False).head(1)
        summary.insert(0, "run_name", run_dir.name)
        summary.insert(1, "best_epoch", int(best_row["epoch"].iloc[0]) if not best_row.empty else -1)
        summary.insert(
            2,
            "best_val_average_precision",
            float(best_row[best_metric].iloc[0]) if not best_row.empty else float("nan"),
        )
        summary.insert(3, "report_path", str(report_path))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.max_train_wafers = min(args.max_train_wafers, 40)
        args.max_test_wafers = min(args.max_test_wafers, 20)
        args.hidden_channels = min(args.hidden_channels, 16)
        args.densities = [0.03]

    configs = grid_configs(args.include_coordinate_ablation, args.quick)
    run_dirs = [run_one(args, config) for config in configs]
    combined = collect_summary(args, run_dirs)

    args.root_log_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.root_log_dir / "batched_cnn_tuning_summary.csv"
    combined.to_csv(combined_path, index=False)

    print("\n=== Combined tuning summary ===")
    cols = [
        "run_name",
        "target_density",
        "mean_roc_auc",
        "mean_average_precision",
        "mean_top32_defect_coverage",
        "mean_coverage32_defect_coverage",
        "mean_top32_coverage_gain_pct",
        "mean_absolute_error_delta",
    ]
    print(combined[cols].round(5).to_string(index=False))
    print(f"\nSaved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
