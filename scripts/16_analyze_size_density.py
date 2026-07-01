from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SAMPLING_RESULTS = (
    Path("data") / "processed" / "sampling" / "sampling_results_with_interior.csv"
)
DEFAULT_RADIAL_2D_RESULTS = (
    Path("data") / "processed" / "radial" / "radial_2d_sweep_results.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "size_density"

SIZE_BIN_LABELS = ["Q1_small", "Q2_mid_small", "Q3_mid_large", "Q4_large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how fixed sampling schemes change with wafer valid-die count "
            "and effective sampling density."
        )
    )
    parser.add_argument("--sampling-results", type=Path, default=DEFAULT_SAMPLING_RESULTS)
    parser.add_argument("--radial-2d-results", type=Path, default=DEFAULT_RADIAL_2D_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-radial-2d",
        action="store_true",
        help="Also aggregate radial 2D sweep variants in the same output tables.",
    )
    return parser.parse_args()


def standardize_results(path: Path, source: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "scheme" not in data.columns:
        data = data.rename(columns={"variant": "scheme"})
    data["source"] = source
    return data


def assign_size_bins(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wafer_sizes = (
        data[["row_index", "failureType", "valid_die_count", "total_defects"]]
        .drop_duplicates("row_index")
        .copy()
    )
    wafer_sizes["size_bin"] = pd.qcut(
        wafer_sizes["valid_die_count"],
        q=4,
        labels=SIZE_BIN_LABELS,
        duplicates="drop",
    )
    wafer_sizes["actual_defect_ratio"] = (
        wafer_sizes["total_defects"] / wafer_sizes["valid_die_count"]
    )

    data = data.merge(
        wafer_sizes[["row_index", "size_bin"]],
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    data["size_bin"] = pd.Categorical(
        data["size_bin"], categories=SIZE_BIN_LABELS, ordered=True
    )
    return data, wafer_sizes


def aggregate(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_cols, observed=False)
        .agg(
            rows=("row_index", "count"),
            wafers=("row_index", "nunique"),
            mean_valid_die_count=("valid_die_count", "mean"),
            median_valid_die_count=("valid_die_count", "median"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_sampling_density=("sampling_density", "mean"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
            mean_sampled_defect_ratio=("sampled_defect_ratio", "mean"),
            mean_ratio_error=("ratio_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            underestimation_rate=("underestimated", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            hit_rate=("hit", "mean"),
        )
        .reset_index()
    )


def summarize_wafer_bins(wafer_sizes: pd.DataFrame) -> pd.DataFrame:
    return (
        wafer_sizes.groupby(["size_bin"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            min_valid_die_count=("valid_die_count", "min"),
            max_valid_die_count=("valid_die_count", "max"),
            mean_valid_die_count=("valid_die_count", "mean"),
            median_valid_die_count=("valid_die_count", "median"),
            mean_total_defects=("total_defects", "mean"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
        )
        .reset_index()
    )


def summarize_wafer_bins_by_pattern(wafer_sizes: pd.DataFrame) -> pd.DataFrame:
    return (
        wafer_sizes.groupby(["failureType", "size_bin"], observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_valid_die_count=("valid_die_count", "mean"),
            median_valid_die_count=("valid_die_count", "median"),
            mean_total_defects=("total_defects", "mean"),
            mean_actual_defect_ratio=("actual_defect_ratio", "mean"),
        )
        .reset_index()
    )


def calculate_size_trends(summary: pd.DataFrame) -> pd.DataFrame:
    small = summary[summary["size_bin"] == "Q1_small"].copy()
    large = summary[summary["size_bin"] == "Q4_large"].copy()
    key_cols = ["source", "scheme", "failureType"]
    merged = small.merge(
        large,
        on=key_cols,
        how="inner",
        suffixes=("_small", "_large"),
    )
    merged["sampling_density_delta_large_minus_small"] = (
        merged["mean_sampling_density_large"] - merged["mean_sampling_density_small"]
    )
    merged["absolute_error_delta_large_minus_small"] = (
        merged["mean_absolute_error_large"] - merged["mean_absolute_error_small"]
    )
    merged["severe_miss_delta_large_minus_small"] = (
        merged["severe_miss_rate_large"] - merged["severe_miss_rate_small"]
    )
    merged["underestimation_delta_large_minus_small"] = (
        merged["underestimation_rate_large"] - merged["underestimation_rate_small"]
    )
    return merged[
        key_cols
        + [
            "mean_sampling_density_small",
            "mean_sampling_density_large",
            "sampling_density_delta_large_minus_small",
            "mean_absolute_error_small",
            "mean_absolute_error_large",
            "absolute_error_delta_large_minus_small",
            "severe_miss_rate_small",
            "severe_miss_rate_large",
            "severe_miss_delta_large_minus_small",
            "underestimation_rate_small",
            "underestimation_rate_large",
            "underestimation_delta_large_minus_small",
        ]
    ]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [standardize_results(args.sampling_results, "sampling_schemes")]
    if args.include_radial_2d:
        frames.append(standardize_results(args.radial_2d_results, "radial_2d"))

    results = pd.concat(frames, ignore_index=True)
    results, wafer_sizes = assign_size_bins(results)

    wafer_bin_summary = summarize_wafer_bins(wafer_sizes)
    wafer_pattern_summary = summarize_wafer_bins_by_pattern(wafer_sizes)
    scheme_size_summary = aggregate(results, ["source", "scheme", "size_bin"])
    pattern_size_summary = aggregate(
        results, ["source", "scheme", "failureType", "size_bin"]
    )
    size_trends = calculate_size_trends(pattern_size_summary)

    wafer_bin_summary.to_csv(args.out_dir / "wafer_size_bin_summary.csv", index=False)
    wafer_pattern_summary.to_csv(
        args.out_dir / "wafer_size_bin_by_pattern.csv", index=False
    )
    scheme_size_summary.to_csv(args.out_dir / "scheme_size_summary.csv", index=False)
    pattern_size_summary.to_csv(args.out_dir / "pattern_size_summary.csv", index=False)
    size_trends.to_csv(args.out_dir / "size_risk_trends.csv", index=False)

    print(f"wrote wafer bin summary: {args.out_dir / 'wafer_size_bin_summary.csv'}")
    print(f"wrote scheme-size summary: {args.out_dir / 'scheme_size_summary.csv'}")
    print(f"wrote pattern-size summary: {args.out_dir / 'pattern_size_summary.csv'}")
    print(f"wrote size-risk trends: {args.out_dir / 'size_risk_trends.csv'}")


if __name__ == "__main__":
    main()
