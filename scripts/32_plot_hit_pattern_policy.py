from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import (
    defect_mask,
    make_9point_mask,
    make_hit_pattern_followup_mask,
    valid_die_mask,
)


DEFAULT_DATA_DIR = Path("data") / "processed" / "hit_pattern_policy_v1"
DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "16_hit_pattern_policy_v1"

ACTION_ORDER = [
    "none",
    "coverage16",
    "coverage32",
    "edge16",
    "radial32",
    "hit_pattern16",
    "hit_pattern32",
]
KEY_ACTIONS = [
    "none",
    "coverage16",
    "coverage32",
    "edge16",
    "radial32",
    "hit_pattern16",
    "hit_pattern32",
]
PATTERN_ORDER = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 9-point hit-pattern policy results.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def plot_score_vs_cost(summary: pd.DataFrame, out_path: Path) -> None:
    data = summary[summary["action"].isin(KEY_ACTIONS)].copy()
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)
    plt.figure(figsize=(9.8, 5.7))
    sns.lineplot(
        data=data,
        x="cost_weight",
        y="mean_spatial_cost_score",
        hue="action",
        hue_order=KEY_ACTIONS,
        marker="o",
    )
    plt.xscale("symlog", linthresh=0.001)
    plt.xlabel("Added valid die cost weight")
    plt.ylabel("Mean spatial cost score")
    plt.title("Hit-Pattern Policy vs Fixed Follow-Up Baselines")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_risk_efficiency(summary: pd.DataFrame, out_path: Path) -> None:
    selected_cost = 0.003
    if selected_cost not in set(summary["cost_weight"]):
        selected_cost = float(sorted(summary["cost_weight"].unique())[0])
    data = summary[
        (summary["cost_weight"] == selected_cost)
        & (summary["action"].isin([a for a in KEY_ACTIONS if a != "none"]))
    ].copy()
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)
    plt.figure(figsize=(9.2, 5.4))
    sns.barplot(
        data=data,
        x="action",
        y="mean_risk_reduction_per_added_die",
        order=[a for a in KEY_ACTIONS if a != "none"],
        color="#4C78A8",
    )
    plt.xlabel("Action")
    plt.ylabel("Spatial risk reduction per added die")
    plt.title(f"Risk-Reduction Efficiency at cost={selected_cost}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_hit_group(summary: pd.DataFrame, out_path: Path) -> None:
    selected_cost = 0.003
    if selected_cost not in set(summary["cost_weight"]):
        selected_cost = float(sorted(summary["cost_weight"].unique())[0])
    data = summary[
        (summary["cost_weight"] == selected_cost)
        & (summary["action"].isin(KEY_ACTIONS))
    ].copy()
    data["first_pass"] = np.where(data["first_no_hit"] == 1, "no hit", "hit")
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)

    plt.figure(figsize=(10.2, 5.7))
    sns.barplot(
        data=data,
        x="action",
        y="mean_spatial_cost_score",
        hue="first_pass",
        order=KEY_ACTIONS,
    )
    plt.xlabel("Action")
    plt.ylabel("Mean spatial cost score")
    plt.title(f"Policy Performance Split by First-Pass Hit Result at cost={selected_cost}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pattern_heatmap(pattern_summary: pd.DataFrame, out_path: Path) -> None:
    selected_cost = 0.003
    if selected_cost not in set(pattern_summary["cost_weight"]):
        selected_cost = float(sorted(pattern_summary["cost_weight"].unique())[0])
    data = pattern_summary[
        (pattern_summary["cost_weight"] == selected_cost)
        & (pattern_summary["action"].isin(KEY_ACTIONS))
    ].copy()
    data["failureType"] = pd.Categorical(data["failureType"], PATTERN_ORDER, ordered=True)
    data["action"] = pd.Categorical(data["action"], KEY_ACTIONS, ordered=True)
    pivot = data.pivot(
        index="failureType",
        columns="action",
        values="mean_spatial_cost_score",
    ).reindex(index=PATTERN_ORDER, columns=KEY_ACTIONS)

    plt.figure(figsize=(11.5, 5.9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Action")
    plt.ylabel("Failure type")
    plt.title(f"Pattern-Wise Spatial Cost Score at cost={selected_cost}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_policy_examples(patterned_path: Path, out_path: Path) -> None:
    df = pd.read_pickle(patterned_path)
    preferred = ["Center", "Edge-Loc", "Edge-Ring", "Scratch", "Loc", "Random"]
    rows = []
    for pattern in preferred:
        sub = df[df["failureType_clean"].eq(pattern)] if "failureType_clean" in df else df[df["failureType"].eq(pattern)]
        for row in sub.itertuples(index=True):
            wafer_map = np.asarray(row.waferMap)
            first = make_9point_mask(wafer_map)
            if int((first & defect_mask(wafer_map)).sum()) > 0:
                rows.append((pattern, int(row.Index), wafer_map))
                break
        if len(rows) >= 4:
            break
    if len(rows) < 4:
        for row in df.head(4).itertuples(index=True):
            rows.append((str(row.failureType_clean), int(row.Index), np.asarray(row.waferMap)))
            if len(rows) >= 4:
                break

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.4))
    for ax, (pattern, row_index, wafer_map) in zip(axes.flat, rows, strict=False):
        valid = valid_die_mask(wafer_map)
        defects = defect_mask(wafer_map)
        first = make_9point_mask(wafer_map)
        follow = make_hit_pattern_followup_mask(wafer_map, n_points=16, existing_mask=first)

        image = np.zeros((*wafer_map.shape, 3), dtype=float)
        image[valid] = (0.85, 0.88, 0.90)
        image[defects] = (0.72, 0.08, 0.08)
        image[follow] = (0.95, 0.78, 0.15)
        image[first] = (0.05, 0.42, 0.86)
        image[first & defects] = (0.05, 0.85, 0.95)
        ax.imshow(image, interpolation="nearest")
        ax.set_title(f"{pattern} row={row_index}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Hit-Pattern16 Follow-Up Examples: blue=first, yellow=follow-up")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.data_dir / "hit_pattern_fixed_summary.csv")
    pattern = pd.read_csv(args.data_dir / "hit_pattern_pattern_summary.csv")
    hit_group = pd.read_csv(args.data_dir / "hit_pattern_first_hit_summary.csv")

    plot_score_vs_cost(fixed, args.fig_dir / "hit_pattern_score_vs_cost.png")
    plot_risk_efficiency(fixed, args.fig_dir / "hit_pattern_risk_efficiency.png")
    plot_hit_group(hit_group, args.fig_dir / "hit_pattern_first_hit_split.png")
    plot_pattern_heatmap(pattern, args.fig_dir / "hit_pattern_pattern_heatmap.png")
    plot_policy_examples(args.patterned, args.fig_dir / "hit_pattern_policy_examples.png")

    print(f"wrote hit-pattern policy figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
