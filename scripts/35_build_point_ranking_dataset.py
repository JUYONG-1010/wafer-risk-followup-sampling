from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import (
    FIRST_PASS_TYPES,
    candidate_feature_frame,
    sample_training_candidates,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "point_ranking_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-point dataset for ML point-ranking."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-wafers", type=int, default=0)
    parser.add_argument("--max-defect-candidates", type=int, default=50)
    parser.add_argument("--max-normal-candidates", type=int, default=200)
    parser.add_argument(
        "--first-pass-types",
        nargs="+",
        default=list(FIRST_PASS_TYPES),
        choices=list(FIRST_PASS_TYPES),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.input)
    if args.max_wafers:
        df = df.head(args.max_wafers).copy()

    rng = np.random.default_rng(args.seed)
    frames: list[pd.DataFrame] = []
    total = len(df)
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        wafer_map = np.asarray(row.waferMap)

        for first_pass_type in args.first_pass_types:
            coords = sample_training_candidates(
                wafer_map,
                first_pass_type=first_pass_type,
                max_defect_candidates=args.max_defect_candidates,
                max_normal_candidates=args.max_normal_candidates,
                rng=rng,
            )
            if len(coords) == 0:
                continue
            features = candidate_feature_frame(
                wafer_map,
                first_pass_type=first_pass_type,
                candidate_coords=coords,
                row_index=int(row.Index),
                failure_type=str(failure_type),
                include_label=True,
            )
            frames.append(features)

        if pos % 1000 == 0 or pos == total:
            print(f"point-ranking dataset rows processed: {pos:,}/{total:,}")

    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = (
        dataset.groupby(["first_pass_type", "failureType"], observed=False)
        .agg(
            rows=("row_index", "count"),
            wafers=("row_index", "nunique"),
            positive_rate=("label_candidate_is_defect", "mean"),
            positives=("label_candidate_is_defect", "sum"),
        )
        .reset_index()
        if not dataset.empty
        else pd.DataFrame()
    )
    first_summary = (
        dataset.groupby("first_pass_type", observed=False)
        .agg(
            rows=("row_index", "count"),
            wafers=("row_index", "nunique"),
            positive_rate=("label_candidate_is_defect", "mean"),
            positives=("label_candidate_is_defect", "sum"),
            mean_first_hit_count=("first_hit_count", "mean"),
            first_no_hit_rate=("first_no_hit", "mean"),
        )
        .reset_index()
        if not dataset.empty
        else pd.DataFrame()
    )

    dataset.to_csv(args.out_dir / "point_ranking_dataset.csv", index=False)
    summary.to_csv(args.out_dir / "point_ranking_dataset_by_pattern.csv", index=False)
    first_summary.to_csv(args.out_dir / "point_ranking_dataset_summary.csv", index=False)

    print(f"wrote point-ranking dataset: {args.out_dir / 'point_ranking_dataset.csv'}")
    print(f"rows: {len(dataset):,}")
    if not dataset.empty:
        print(
            "positive rate:",
            f"{float(dataset['label_candidate_is_defect'].mean()):.4f}",
        )


if __name__ == "__main__":
    main()
