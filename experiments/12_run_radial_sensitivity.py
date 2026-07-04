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
DEFAULT_RESULTS = Path("data") / "processed" / "radial" / "radial_sensitivity_results.csv"


RADIAL_VARIANTS = [
    {
        "variant": "outer_0.75_a8",
        "stage": "outer_radius",
        "rings": (0.0, 0.5, 0.75),
        "angles": 8,
        "outer_radius": 0.75,
        "ring_count": 3,
    },
    {
        "variant": "outer_0.85_a8",
        "stage": "outer_radius",
        "rings": (0.0, 0.5, 0.85),
        "angles": 8,
        "outer_radius": 0.85,
        "ring_count": 3,
    },
    {
        "variant": "outer_0.95_a8",
        "stage": "outer_radius",
        "rings": (0.0, 0.5, 0.95),
        "angles": 8,
        "outer_radius": 0.95,
        "ring_count": 3,
    },
    {
        "variant": "outer_0.95_a8_angle",
        "stage": "angle_count",
        "rings": (0.0, 0.5, 0.95),
        "angles": 8,
        "outer_radius": 0.95,
        "ring_count": 3,
    },
    {
        "variant": "outer_0.95_a16",
        "stage": "angle_count",
        "rings": (0.0, 0.5, 0.95),
        "angles": 16,
        "outer_radius": 0.95,
        "ring_count": 3,
    },
    {
        "variant": "outer_0.95_a32",
        "stage": "angle_count",
        "rings": (0.0, 0.5, 0.95),
        "angles": 32,
        "outer_radius": 0.95,
        "ring_count": 3,
    },
    {
        "variant": "rings_simple_a16",
        "stage": "ring_count",
        "rings": (0.0, 0.5, 0.95),
        "angles": 16,
        "outer_radius": 0.95,
        "ring_count": 3,
    },
    {
        "variant": "rings_multi_a16",
        "stage": "ring_count",
        "rings": (0.0, 0.33, 0.66, 0.95),
        "angles": 16,
        "outer_radius": 0.95,
        "ring_count": 4,
    },
    {
        "variant": "rings_dense_a16",
        "stage": "ring_count",
        "rings": (0.0, 0.25, 0.5, 0.75, 0.95),
        "angles": 16,
        "outer_radius": 0.95,
        "ring_count": 5,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run radial sampling sensitivity analysis.")
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
    for variant_config in RADIAL_VARIANTS:
        variant = variant_config["variant"]
        rings = variant_config["rings"]
        angles = int(variant_config["angles"])
        stage = variant_config["stage"]
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
                    "stage": stage,
                    "rings": "|".join(str(x) for x in rings),
                    "angles": angles,
                    "outer_radius": variant_config["outer_radius"],
                    "ring_count": variant_config["ring_count"],
                    **metrics,
                }
            )

            if pos % 10000 == 0 or pos == total:
                print(f"  processed {pos:,}/{total:,}")

    results = pd.DataFrame.from_records(records)
    results.to_csv(args.out, index=False)
    print(f"wrote radial sensitivity results: {args.out}")

    summary = (
        results.groupby(
            [
                "failureType",
                "variant",
                "stage",
                "rings",
                "angles",
                "outer_radius",
                "ring_count",
            ]
        )
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
    summary_path = args.out.with_name("radial_sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote radial sensitivity summary: {summary_path}")


if __name__ == "__main__":
    main()
