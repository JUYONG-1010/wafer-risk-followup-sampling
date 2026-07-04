from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_METADATA = Path("data") / "processed" / "metadata" / "labeled_metadata.csv"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "01_eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create first EDA figures.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def save_class_distribution(metadata: pd.DataFrame, fig_dir: Path) -> None:
    labeled = metadata[metadata["is_labeled"]]
    counts = labeled["failureType"].value_counts()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#4C78A8")
    plt.xlabel("Wafer count")
    plt.ylabel("Failure type")
    plt.title("WM-811K Labeled Class Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=180)
    plt.close()


def save_wafer_size_distribution(metadata: pd.DataFrame, fig_dir: Path) -> None:
    top_shapes = metadata["map_shape"].value_counts().head(20)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_shapes.index, y=top_shapes.values, color="#59A14F")
    plt.xlabel("Wafer map shape")
    plt.ylabel("Count")
    plt.title("Top Wafer Map Shapes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "wafer_size_distribution.png", dpi=180)
    plt.close()


def save_defect_ratio_by_class(metadata: pd.DataFrame, fig_dir: Path) -> None:
    patterned = metadata[metadata["is_patterned"]]
    order = (
        patterned.groupby("failureType")["defect_ratio_valid"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=patterned,
        x="defect_ratio_valid",
        y="failureType",
        order=order,
        color="#F28E2B",
        showfliers=False,
    )
    plt.xlabel("Defect die ratio among valid dies")
    plt.ylabel("Failure type")
    plt.title("Defect Ratio by Pattern")
    plt.tight_layout()
    plt.savefig(fig_dir / "defect_ratio_by_class.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)

    save_class_distribution(metadata, args.fig_dir)
    save_wafer_size_distribution(metadata, args.fig_dir)
    save_defect_ratio_by_class(metadata, args.fig_dir)
    print(f"wrote EDA figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
