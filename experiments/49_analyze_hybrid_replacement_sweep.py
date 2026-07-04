from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT_DIRS = [
    Path("data") / "processed" / "density_followup_hybrid_small_v1",
    Path("data") / "processed" / "density_followup_hybrid_v1",
]
DEFAULT_OUT_DIR = Path("data") / "processed" / "density_followup_replacement_sweep_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "39_density_followup_replacement_sweep_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize hybrid_guardedN replacement-count trade-off."
    )
    parser.add_argument("--input-dirs", type=Path, nargs="+", default=DEFAULT_INPUT_DIRS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def replacement_count(strategy: str) -> int | None:
    match = re.fullmatch(r"hybrid_guarded(\d+)", strategy)
    return int(match.group(1)) if match else None


def load_sweep(input_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for input_dir in input_dirs:
        path = input_dir / "density_followup_guardrail_vs_coverage.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path)
        data["source_dir"] = str(input_dir)
        frames.append(data)
    if not frames:
        raise FileNotFoundError("No density_followup_guardrail_vs_coverage.csv files found.")
    data = pd.concat(frames, ignore_index=True)
    data["replacement_count"] = data["strategy"].map(replacement_count)
    data = data[data["replacement_count"].notna()].copy()
    data["replacement_count"] = data["replacement_count"].astype(int)
    data = data.drop_duplicates(["target_density", "replacement_count"], keep="first")
    data["small_abs_error_delta_pass_0p01"] = data["absolute_error_delta"] <= 0.01
    data["coverage_gain_per_abs_error_delta"] = (
        (data["strategy_defect_coverage"] - data["baseline_defect_coverage"])
        / data["absolute_error_delta"].where(data["absolute_error_delta"] > 0)
    )
    return data.sort_values(["target_density", "replacement_count"]).reset_index(drop=True)


def choose_knee_points(sweep: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for density, group in sweep.groupby("target_density", observed=False):
        low_cost = group[group["absolute_error_delta"] <= 0.01]
        if not low_cost.empty:
            chosen = low_cost.sort_values(
                ["defect_coverage_relative_improvement_pct", "replacement_count"],
                ascending=[False, True],
            ).iloc[0]
        else:
            chosen = group.sort_values(
                ["absolute_error_delta", "defect_coverage_relative_improvement_pct"],
                ascending=[True, False],
            ).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_sweep(sweep: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = sweep.copy()
    data["density_pct"] = data["target_density"] * 100.0

    for metric, filename, ylabel in [
        (
            "defect_coverage_relative_improvement_pct",
            "replacement_count_coverage_gain.png",
            "Defect coverage improvement vs coverage32 (%)",
        ),
        (
            "absolute_error_delta",
            "replacement_count_absolute_error_delta.png",
            "Absolute-error delta vs coverage32",
        ),
        (
            "coverage_gain_per_abs_error_delta",
            "replacement_count_tradeoff_efficiency.png",
            "Coverage gain per absolute-error delta",
        ),
    ]:
        plt.figure(figsize=(8.8, 5.2))
        sns.lineplot(
            data=data,
            x="replacement_count",
            y=metric,
            hue="density_pct",
            marker="o",
            palette="viridis",
        )
        if metric == "absolute_error_delta":
            plt.axhline(0.01, color="#777777", linestyle="--", linewidth=0.9)
        plt.xlabel("Number of coverage32 follow-up points replaced")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(8.8, 5.2))
    sns.scatterplot(
        data=data,
        x="absolute_error_delta",
        y="defect_coverage_relative_improvement_pct",
        hue="density_pct",
        size="replacement_count",
        sizes=(50, 180),
        palette="viridis",
    )
    plt.axvline(0.01, color="#777777", linestyle="--", linewidth=0.9)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Absolute-error delta vs coverage32")
    plt.ylabel("Defect coverage improvement vs coverage32 (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "replacement_count_gain_vs_error.png", dpi=180)
    plt.close()


def write_report(sweep: pd.DataFrame, knee: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Hybrid Replacement Sweep")
    lines.append("")
    lines.append("Question: how many coverage32 points should be replaced by guarded morphrisk points?")
    lines.append("")
    lines.append("## Global Sweep")
    lines.append("")
    for row in sweep.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, N={row.replacement_count}: "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f}, "
            f"small-delta pass={row.small_abs_error_delta_pass_0p01}"
        )
    lines.append("")
    lines.append("## Suggested Replacement Count Under abs-error delta <= 0.01")
    lines.append("")
    for row in knee.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: N={row.replacement_count}, "
            f"coverage gain {row.defect_coverage_relative_improvement_pct:.2f}%, "
            f"abs-error delta {row.absolute_error_delta:.4f}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Increasing N gives more defect coverage, but the absolute-error cost grows almost monotonically."
    )
    lines.append(
        "N=1 is the conservative choice. N=2 becomes acceptable mainly at denser initial probing, especially 10%."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    sweep = load_sweep(args.input_dirs)
    knee = choose_knee_points(sweep)

    sweep.to_csv(args.out_dir / "hybrid_replacement_sweep.csv", index=False)
    knee.to_csv(args.out_dir / "hybrid_replacement_sweep_recommended.csv", index=False)
    write_report(sweep, knee, args.out_dir / "hybrid_replacement_sweep_report.md")
    plot_sweep(sweep, args.fig_dir)

    print(f"wrote replacement sweep outputs to {args.out_dir}")
    print(f"wrote replacement sweep figures to {args.fig_dir}")
    print(
        sweep[
            [
                "target_density",
                "replacement_count",
                "defect_coverage_relative_improvement_pct",
                "absolute_error_delta",
                "small_abs_error_delta_pass_0p01",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
