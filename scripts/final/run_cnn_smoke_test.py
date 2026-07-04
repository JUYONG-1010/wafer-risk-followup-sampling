from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small local sparse-CNN smoke test.")
    parser.add_argument("--max-train-wafers", type=int, default=80)
    parser.add_argument("--max-test-wafers", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--density", type=float, default=0.03)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (PROJECT_ROOT / DEFAULT_PATTERNED).exists():
        print("Missing processed data. Run:")
        print("  python scripts/final/prepare_data.py")
        return 2
    command = [
        sys.executable,
        "experiments/71_train_sparse_cnn_risk_map_batched.py",
        "--patterned",
        str(DEFAULT_PATTERNED),
        "--densities",
        str(args.density),
        "--epochs",
        str(args.epochs),
        "--max-train-wafers",
        str(args.max_train_wafers),
        "--max-test-wafers",
        str(args.max_test_wafers),
        "--out-dir",
        "data/processed/final_cnn_smoke_test",
        "--fig-dir",
        "outputs/figures/final_cnn_smoke_test",
        "--model-path",
        "models/final_cnn_smoke_test.pt",
    ]
    if args.cpu_only:
        command.append("--cpu-only")
    print("Running CNN smoke test:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
