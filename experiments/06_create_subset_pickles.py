from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = Path("LSWMD.pkl") / "LSWMD.pkl"
DEFAULT_METADATA = Path("data") / "processed" / "metadata" / "labeled_metadata.csv"
DEFAULT_OUT_DIR = Path("data") / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create labeled and patterned subset pickle files using existing "
            "metadata. This avoids recomputing per-wafer metadata."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--patterned-only",
        action="store_true",
        help="Only write patterned_subset.pkl, not labeled_subset.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    required = {"row_index", "failureType", "trianTestLabel", "is_labeled", "is_patterned"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    print(f"loading source pickle: {args.input}")
    df = pd.read_pickle(args.input)
    print(f"loaded DataFrame shape: {df.shape}")

    df = df.copy()
    df["failureType_clean"] = metadata["failureType"].fillna("").to_numpy()
    df["trianTestLabel_clean"] = metadata["trianTestLabel"].fillna("").to_numpy()

    if not args.patterned_only:
        subset_dir = args.out_dir / "subsets"
        subset_dir.mkdir(parents=True, exist_ok=True)
        labeled_path = subset_dir / "labeled_subset.pkl"
        labeled_mask = metadata["is_labeled"].astype(bool).to_numpy()
        df.loc[labeled_mask].to_pickle(labeled_path)
        print(f"wrote labeled subset: {labeled_path} ({labeled_mask.sum():,} rows)")

    subset_dir = args.out_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    patterned_path = subset_dir / "patterned_subset.pkl"
    patterned_mask = metadata["is_patterned"].astype(bool).to_numpy()
    df.loc[patterned_mask].to_pickle(patterned_path)
    print(f"wrote patterned subset: {patterned_path} ({patterned_mask.sum():,} rows)")

    del df
    gc.collect()
    print("done")


if __name__ == "__main__":
    main()
