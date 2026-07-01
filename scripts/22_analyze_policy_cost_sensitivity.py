from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data") / "processed" / "policy_learning" / "action_outcomes.csv"
DEFAULT_OUT_DIR = Path("data") / "processed" / "policy_learning" / "cost_sensitivity"

DEFAULT_ACTION_ORDER = [
    "none",
    "random16",
    "coverage16",
    "coverage32",
    "edge16",
    "radial16",
    "radial32",
    "local_expand",
    "edge16_local",
    "radial32_local",
]
DEFAULT_COST_WEIGHTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how policy-learning oracle labels shift with cost weight."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cost-weights",
        type=float,
        nargs="+",
        default=DEFAULT_COST_WEIGHTS,
        help="Added-valid-die cost weights to evaluate.",
    )
    parser.add_argument("--severe-miss-weight", type=float, default=10.0)
    parser.add_argument("--absolute-error-weight", type=float, default=1.0)
    parser.add_argument("--underestimation-weight", type=float, default=0.25)
    return parser.parse_args()


def compute_score(
    data: pd.DataFrame,
    cost_weight: float,
    severe_miss_weight: float,
    absolute_error_weight: float,
    underestimation_weight: float,
) -> pd.Series:
    return (
        severe_miss_weight * data["severe_miss"].astype(float)
        + absolute_error_weight * data["absolute_error"].astype(float)
        + underestimation_weight * data["underestimated"].astype(float)
        + cost_weight * data["added_valid_count"].astype(float)
    )


def select_best(data: pd.DataFrame, cost_weight: float, args: argparse.Namespace) -> pd.DataFrame:
    scored = data.copy()
    scored["cost_weight"] = cost_weight
    scored["sensitivity_score"] = compute_score(
        scored,
        cost_weight=cost_weight,
        severe_miss_weight=args.severe_miss_weight,
        absolute_error_weight=args.absolute_error_weight,
        underestimation_weight=args.underestimation_weight,
    )
    action_order = [a for a in DEFAULT_ACTION_ORDER if a in set(scored["action"])]
    action_order += [a for a in sorted(scored["action"].unique()) if a not in action_order]
    scored["action"] = pd.Categorical(scored["action"], action_order, ordered=True)
    ordered = scored.sort_values(
        ["row_index", "sensitivity_score", "added_valid_count", "action"]
    )
    best = ordered.groupby("row_index", observed=False).head(1).copy()
    return best


def summarize_best(best: pd.DataFrame) -> dict[str, float | int]:
    return {
        "wafers": int(best["row_index"].nunique()),
        "mean_sampled_valid_count": float(best["sampled_valid_count"].mean()),
        "mean_added_valid_count": float(best["added_valid_count"].mean()),
        "mean_absolute_error": float(best["absolute_error"].mean()),
        "severe_miss_rate": float(best["severe_miss"].mean()),
        "underestimation_rate": float(best["underestimated"].mean()),
        "mean_score": float(best["sensitivity_score"].mean()),
    }


def summarize_fixed(data: pd.DataFrame, cost_weight: float, args: argparse.Namespace) -> pd.DataFrame:
    scored = data.copy()
    scored["cost_weight"] = cost_weight
    scored["sensitivity_score"] = compute_score(
        scored,
        cost_weight=cost_weight,
        severe_miss_weight=args.severe_miss_weight,
        absolute_error_weight=args.absolute_error_weight,
        underestimation_weight=args.underestimation_weight,
    )
    return (
        scored.groupby("action", observed=False)
        .agg(
            wafers=("row_index", "count"),
            mean_sampled_valid_count=("sampled_valid_count", "mean"),
            mean_added_valid_count=("added_valid_count", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            underestimation_rate=("underestimated", "mean"),
            mean_score=("sensitivity_score", "mean"),
        )
        .reset_index()
        .assign(cost_weight=cost_weight)
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    action_outcomes = pd.read_csv(args.input)

    best_frames: list[pd.DataFrame] = []
    oracle_summary_records: list[dict[str, float | int]] = []
    best_count_records: list[dict[str, float | int | str]] = []
    fixed_summary_frames: list[pd.DataFrame] = []

    for cost_weight in args.cost_weights:
        best = select_best(action_outcomes, cost_weight, args)
        best_frames.append(best)

        oracle_summary = summarize_best(best)
        oracle_summary["cost_weight"] = cost_weight
        oracle_summary_records.append(oracle_summary)

        action_order = [a for a in DEFAULT_ACTION_ORDER if a in set(action_outcomes["action"])]
        action_order += [
            a for a in sorted(action_outcomes["action"].unique()) if a not in action_order
        ]
        counts = best["action"].value_counts().reindex(action_order, fill_value=0)
        for action, count in counts.items():
            best_count_records.append(
                {
                    "cost_weight": cost_weight,
                    "action": action,
                    "wafers": int(count),
                    "fraction": float(count / len(best)),
                }
            )

        fixed_summary_frames.append(summarize_fixed(action_outcomes, cost_weight, args))

    best_actions = pd.concat(best_frames, ignore_index=True)
    oracle_summary_df = pd.DataFrame.from_records(oracle_summary_records)
    best_counts = pd.DataFrame.from_records(best_count_records)
    fixed_summary = pd.concat(fixed_summary_frames, ignore_index=True)
    fixed_vs_oracle = pd.concat(
        [
            fixed_summary.assign(strategy_type="fixed"),
            oracle_summary_df.assign(
                action="oracle_best_by_score",
                strategy_type="oracle",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    best_actions.to_csv(args.out_dir / "cost_sensitivity_best_actions.csv", index=False)
    oracle_summary_df.to_csv(args.out_dir / "cost_sensitivity_oracle_summary.csv", index=False)
    best_counts.to_csv(args.out_dir / "cost_sensitivity_best_action_counts.csv", index=False)
    fixed_summary.to_csv(args.out_dir / "cost_sensitivity_fixed_summary.csv", index=False)
    fixed_vs_oracle.to_csv(args.out_dir / "cost_sensitivity_fixed_vs_oracle.csv", index=False)

    print(f"wrote best actions: {args.out_dir / 'cost_sensitivity_best_actions.csv'}")
    print(f"wrote oracle summary: {args.out_dir / 'cost_sensitivity_oracle_summary.csv'}")
    print(f"wrote best action counts: {args.out_dir / 'cost_sensitivity_best_action_counts.csv'}")
    print(f"wrote fixed summary: {args.out_dir / 'cost_sensitivity_fixed_summary.csv'}")
    print(f"wrote fixed-vs-oracle summary: {args.out_dir / 'cost_sensitivity_fixed_vs_oracle.csv'}")
    print("cost weights:", ", ".join(str(v) for v in args.cost_weights))


if __name__ == "__main__":
    main()
