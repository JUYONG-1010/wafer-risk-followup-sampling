from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_CNN_MODEL = Path("models") / "sparse_cnn_risk_map_v1_large.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public wrapper for the Top-K budget curve experiment.")
    parser.add_argument("--execute", action="store_true", help="Run the command instead of only printing it.")
    parser.add_argument("--max-test-wafers", type=int, default=500)
    parser.add_argument("--max-train-wafers", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        "experiments/75_evaluate_topk_budget_curve.py",
        "--patterned",
        str(DEFAULT_PATTERNED),
        "--cnn-model",
        str(DEFAULT_CNN_MODEL),
        "--max-train-wafers",
        str(args.max_train_wafers),
        "--max-test-wafers",
        str(args.max_test_wafers),
        "--seed",
        str(args.seed),
    ]
    display_command = ["python", *command[1:]]
    print("Top-K budget curve command:")
    print(" ".join(display_command))
    if not (PROJECT_ROOT / DEFAULT_PATTERNED).exists():
        print("\nMissing processed data. Run:")
        print("  python experiments/01_extract_labeled_subset.py")
        return 2
    if not (PROJECT_ROOT / DEFAULT_CNN_MODEL).exists():
        print("\nMissing CNN checkpoint. This wrapper will not fabricate results.")
        print("Run a CNN smoke test or use the Colab notebook before executing this curve.")
        return 2
    if args.execute:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        print("\nUse --execute to run it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
