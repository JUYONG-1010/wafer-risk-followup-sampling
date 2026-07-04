from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    raw_pickle = PROJECT_ROOT / "LSWMD.pkl" / "LSWMD.pkl"
    if not raw_pickle.exists():
        print("Missing raw WM-811K pickle:")
        print("  LSWMD.pkl/LSWMD.pkl")
        print()
        print("Place the dataset there, then rerun:")
        print("  python scripts/final/prepare_data.py")
        return 2
    command = [sys.executable, "experiments/01_extract_labeled_subset.py"]
    print("Running data preparation:")
    print(" ".join(["python", *command[1:]]))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
