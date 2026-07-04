from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import make_random_sampling_mask, sampling_metrics


DEFAULT_INPUT = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT = Path("data") / "processed" / "random" / "random_baseline_by_seed_smoke.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated random sampling baselines and save pattern-level "
            "summaries per seed, without writing huge row-level outputs."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[5, 9, 25, 49],
        help="Random sampling point counts to test.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--radius", type=int, default=0)
    return parser.parse_args()


def clean_failure_type(row: object) -> str:
    value = getattr(row, "failureType_clean", None)
    if isinstance(value, str) and value:
        return value
    return str(getattr(row, "failureType"))


def summarize_records(records: list[dict[str, object]]) -> pd.DataFrame:
    result = pd.DataFrame.from_records(records)
    return (
        result.groupby(["failureType", "budget", "seed"])
        .agg(
            wafers=("row_index", "count"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            underestimation_rate=("underestimated", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading patterned subset: {args.input}")
    df = pd.read_pickle(args.input)
    total = len(df)
    print(f"loaded rows: {total:,}")

    summaries = []
    for budget in args.budgets:
        for seed in range(args.seeds):
            records: list[dict[str, object]] = []
            for pos, row in enumerate(df.itertuples(index=True), start=1):
                row_seed = args.seed_offset + seed * 1_000_003 + int(row.Index)
                sample_mask = make_random_sampling_mask(
                    row.waferMap,
                    n_points=budget,
                    radius=args.radius,
                    seed=row_seed,
                )
                metrics = sampling_metrics(row.waferMap, sample_mask)
                records.append(
                    {
                        "row_index": row.Index,
                        "failureType": clean_failure_type(row),
                        "budget": budget,
                        "seed": seed,
                        **metrics,
                    }
                )

            summary = summarize_records(records)
            summaries.append(summary)
            print(
                f"completed random budget={budget}, seed={seed} "
                f"({total:,} wafers)"
            )

    by_seed = pd.concat(summaries, ignore_index=True)
    by_seed.to_csv(args.out, index=False)
    print(f"wrote by-seed summary: {args.out}")

    aggregate = (
        by_seed.groupby(["failureType", "budget"])
        .agg(
            seeds=("seed", "count"),
            wafers_per_seed=("wafers", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            std_absolute_error=("mean_absolute_error", "std"),
            mean_severe_miss_rate=("severe_miss_rate", "mean"),
            std_severe_miss_rate=("severe_miss_rate", "std"),
            mean_underestimation_rate=("underestimation_rate", "mean"),
            std_underestimation_rate=("underestimation_rate", "std"),
            mean_sampled_valid_count=("mean_sampled_valid_count", "mean"),
            mean_sampling_density=("mean_sampling_density", "mean"),
        )
        .reset_index()
    )
    aggregate_path = args.out.with_name(args.out.stem.replace("by_seed", "summary") + ".csv")
    aggregate.to_csv(aggregate_path, index=False)
    print(f"wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
