from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import (
    make_adaptive_sampling_mask,
    make_edge_biased_sampling_mask,
    make_5point_mask,
    make_interior_5point_mask,
    make_interior_9point_mask,
    make_interior_25point_mask,
    make_9point_mask,
    make_25point_mask,
    make_radial_sampling_mask,
    make_random_sampling_mask,
    sampling_metrics,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT = Path("data") / "processed" / "sampling" / "sampling_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify sparse sampling blind spots for patterned wafers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Process only first N patterned rows for quick iteration.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=0,
        help="Expand each sparse sampling site by this cell radius.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    schemes = [
        ("grid_5point", lambda wafer: make_5point_mask(wafer, radius=args.radius)),
        ("grid_9point", lambda wafer: make_9point_mask(wafer, radius=args.radius)),
        ("grid_25point", lambda wafer: make_25point_mask(wafer, radius=args.radius)),
        (
            "interior_5point",
            lambda wafer: make_interior_5point_mask(wafer, radius=args.radius),
        ),
        (
            "interior_9point",
            lambda wafer: make_interior_9point_mask(wafer, radius=args.radius),
        ),
        (
            "interior_25point",
            lambda wafer: make_interior_25point_mask(wafer, radius=args.radius),
        ),
        ("radial", lambda wafer: make_radial_sampling_mask(wafer, radius=args.radius)),
        (
            "edge_biased",
            lambda wafer: make_edge_biased_sampling_mask(wafer, radius=args.radius),
        ),
        (
            "random_25",
            lambda wafer, row_seed=0: make_random_sampling_mask(
                wafer, n_points=25, radius=args.radius, seed=row_seed
            ),
        ),
        ("adaptive_9point", lambda wafer: make_adaptive_sampling_mask(wafer)),
    ]

    records: list[dict[str, object]] = []
    total = len(df)
    for pos, row in enumerate(df.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType

        for scheme_name, maker in schemes:
            if scheme_name == "random_25":
                sample_mask = maker(row.waferMap, row_seed=int(row.Index) + 42)
            else:
                sample_mask = maker(row.waferMap)
            metrics = sampling_metrics(row.waferMap, sample_mask)
            records.append(
                {
                    "row_index": row.Index,
                    "failureType": failure_type,
                    "scheme": scheme_name,
                    "radius": args.radius,
                    **metrics,
                }
            )

        if pos % 10000 == 0 or pos == total:
            print(f"sampling rows processed: {pos:,}/{total:,}")

    result = pd.DataFrame.from_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote sampling results: {args.out}")

    summary = (
        result.groupby(["failureType", "scheme"])
        .agg(
            wafers=("row_index", "count"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
            mean_sampled_defect_ratio=("sampled_defect_ratio", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            underestimation_rate=("underestimated", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_miss_rate=("miss_rate", "mean"),
            median_miss_rate=("miss_rate", "median"),
            hit_rate=("hit", "mean"),
        )
        .reset_index()
        .sort_values(["scheme", "mean_miss_rate"], ascending=[True, False])
    )
    summary_path = args.out.with_name("sampling_summary_by_pattern.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote sampling summary: {summary_path}")


if __name__ == "__main__":
    main()
