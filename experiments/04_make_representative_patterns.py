from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wm811k_utils import normalize_nested_label


DEFAULT_INPUT = Path("LSWMD.pkl") / "LSWMD.pkl"
DEFAULT_METADATA = Path("data") / "processed" / "metadata" / "labeled_metadata.csv"
DEFAULT_OUT = Path("outputs") / "figures" / "01_eda" / "representative_patterns.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a gallery of representative wafer maps by failure type."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="Include the `none` class in the representative gallery.",
    )
    return parser.parse_args()


def choose_representative_rows(metadata: pd.DataFrame, include_none: bool) -> pd.DataFrame:
    """Pick one typical row per class using median-nearest defect ratio."""
    labeled = metadata[metadata["is_labeled"]].copy()
    if not include_none:
        labeled = labeled[labeled["failureType"].str.lower() != "none"].copy()

    selected_rows = []
    class_order = labeled["failureType"].value_counts().index.tolist()
    for failure_type in class_order:
        group = labeled[labeled["failureType"] == failure_type].copy()
        median_ratio = group["defect_ratio_valid"].median()
        group["distance_to_median"] = (
            group["defect_ratio_valid"] - median_ratio
        ).abs()
        selected = group.sort_values(
            ["distance_to_median", "defect_die_count"], ascending=[True, False]
        ).iloc[0]
        selected_rows.append(selected)

    return pd.DataFrame(selected_rows)


def draw_wafer(ax: plt.Axes, wafer_map: np.ndarray, title: str) -> None:
    """Draw one wafer map with a fixed 0/1/2 color meaning."""
    cmap = ListedColormap(["#F1F3F4", "#4C78A8", "#E45756"])
    ax.imshow(wafer_map, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    representatives = choose_representative_rows(metadata, args.include_none)
    row_indices = representatives["row_index"].tolist()

    print("selected representative rows:")
    print(
        representatives[
            ["row_index", "failureType", "map_shape", "defect_die_count", "defect_ratio_valid"]
        ].to_string(index=False)
    )

    print(f"loading pickle to fetch {len(row_indices)} wafer maps: {args.input}")
    df = pd.read_pickle(args.input)
    selected_maps = df.loc[row_indices, "waferMap"]

    n = len(representatives)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, (_, meta_row), wafer_map in zip(
        axes_arr, representatives.iterrows(), selected_maps
    ):
        failure_type = meta_row["failureType"]
        if not isinstance(failure_type, str) or not failure_type:
            failure_type = normalize_nested_label(failure_type)
        title = (
            f"{failure_type}\n"
            f"shape {meta_row['map_shape']}, "
            f"defect {meta_row['defect_ratio_valid']:.2%}"
        )
        draw_wafer(ax, np.asarray(wafer_map), title)

    for ax in axes_arr[n:]:
        ax.axis("off")

    fig.suptitle("Representative WM-811K Defect Patterns", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote representative gallery: {args.out}")


if __name__ == "__main__":
    main()
