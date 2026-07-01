from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import make_radial_sampling_mask, sampling_metrics


DEFAULT_INPUT = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_RESULTS = Path("data") / "processed" / "radial" / "radial_2d_sweep_results.csv"


OUTER_RADII = [0.75, 0.85, 0.95]
ANGLE_COUNTS = [8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2D radial sweep over outer radius and angle count."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--radius", type=int, default=0)
    return parser.parse_args()


def clean_failure_type(row: object) -> str:
    value = getattr(row, "failureType_clean", None)
    if isinstance(value, str) and value:
        return value
    return str(getattr(row, "failureType"))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading patterned subset: {args.input}")
    df = pd.read_pickle(args.input)
    total = len(df)
    print(f"loaded rows: {total:,}")

    records: list[dict[str, object]] = []
    for outer_radius in OUTER_RADII:
        for angles in ANGLE_COUNTS:
            rings = (0.0, 0.5, outer_radius)
            variant = f"outer_{outer_radius:.2f}_a{angles}"
            print(f"running variant={variant}, rings={rings}, angles={angles}")

            for pos, row in enumerate(df.itertuples(index=True), start=1):
                sample_mask = make_radial_sampling_mask(
                    row.waferMap,
                    rings=rings,
                    angles=angles,
                    radius=args.radius,
                )
                metrics = sampling_metrics(row.waferMap, sample_mask)
                records.append(
                    {
                        "row_index": row.Index,
                        "failureType": clean_failure_type(row),
                        "variant": variant,
                        "rings": "|".join(str(x) for x in rings),
                        "angles": angles,
                        "outer_radius": outer_radius,
                        **metrics,
                    }
                )

                if pos % 10000 == 0 or pos == total:
                    print(f"  processed {pos:,}/{total:,}")

    results = pd.DataFrame.from_records(records)
    results.to_csv(args.out, index=False)
    print(f"wrote radial 2D sweep results: {args.out}")

    summary = (
        results.groupby(["failureType", "variant", "rings", "angles", "outer_radius"])
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
            hit_rate=("hit", "mean"),
        )
        .reset_index()
    )
    summary_path = args.out.with_name("radial_2d_sweep_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote radial 2D sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
