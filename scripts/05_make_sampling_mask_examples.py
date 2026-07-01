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

from src.sampling import make_5point_mask, make_9point_mask, make_25point_mask
from src.sampling import (
    make_adaptive_sampling_mask,
    make_edge_biased_sampling_mask,
    make_radial_sampling_mask,
    make_random_sampling_mask,
)


DEFAULT_INPUT = Path("LSWMD.pkl") / "LSWMD.pkl"
DEFAULT_METADATA = Path("data") / "processed" / "metadata" / "labeled_metadata.csv"
DEFAULT_OUT = Path("outputs") / "figures" / "02_sampling_baselines" / "sampling_mask_examples.png"
DEFAULT_CLASSES = ["Edge-Ring", "Edge-Loc", "Center", "Scratch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sparse sampling masks on representative wafer maps."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Failure types to visualize.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=0,
        help="Expand each sampling site by this cell radius.",
    )
    return parser.parse_args()


def choose_representative_rows(
    metadata: pd.DataFrame, failure_types: list[str]
) -> pd.DataFrame:
    """Pick one median-like representative row for each requested class."""
    selected_rows = []
    for failure_type in failure_types:
        group = metadata[metadata["failureType"] == failure_type].copy()
        if group.empty:
            raise ValueError(f"No rows found for failure type: {failure_type}")
        median_ratio = group["defect_ratio_valid"].median()
        group["distance_to_median"] = (
            group["defect_ratio_valid"] - median_ratio
        ).abs()
        selected = group.sort_values(
            ["distance_to_median", "defect_die_count"], ascending=[True, False]
        ).iloc[0]
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows)


def overlay_sampling_view(wafer_map: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    """Encode wafer, defect, and sampled cells into one image.

    Values:
    0 = outside wafer
    1 = normal die
    2 = defect die
    3 = sampled normal die
    4 = sampled defect die
    """
    wafer = np.asarray(wafer_map)
    view = np.zeros(wafer.shape, dtype=int)
    view[wafer == 1] = 1
    view[wafer == 2] = 2
    view[(wafer == 1) & sample_mask] = 3
    view[(wafer == 2) & sample_mask] = 4
    return view


def draw_panel(
    ax: plt.Axes,
    wafer_map: np.ndarray,
    sample_mask: np.ndarray,
    title: str,
) -> None:
    cmap = ListedColormap(
        [
            "#F1F3F4",  # outside
            "#4C78A8",  # normal die
            "#E45756",  # defect die
            "#111111",  # sampled normal die
            "#FFD166",  # sampled defect die
        ]
    )
    view = overlay_sampling_view(wafer_map, sample_mask)
    ax.imshow(view, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    representatives = choose_representative_rows(metadata, args.classes)
    row_indices = representatives["row_index"].tolist()

    print("selected rows for sampling-mask examples:")
    print(
        representatives[
            ["row_index", "failureType", "map_shape", "defect_die_count", "defect_ratio_valid"]
        ].to_string(index=False)
    )

    print(f"loading pickle to fetch {len(row_indices)} wafer maps: {args.input}")
    df = pd.read_pickle(args.input)
    selected_maps = df.loc[row_indices, "waferMap"]

    schemes = [
        ("Original", None),
        ("5-point", lambda wafer: make_5point_mask(wafer, radius=args.radius)),
        ("9-point", lambda wafer: make_9point_mask(wafer, radius=args.radius)),
        ("25-point", lambda wafer: make_25point_mask(wafer, radius=args.radius)),
        ("Radial", lambda wafer: make_radial_sampling_mask(wafer, radius=args.radius)),
        (
            "Edge-biased",
            lambda wafer: make_edge_biased_sampling_mask(wafer, radius=args.radius),
        ),
        (
            "Random-25",
            lambda wafer: make_random_sampling_mask(wafer, radius=args.radius, seed=7),
        ),
        ("Adaptive", lambda wafer: make_adaptive_sampling_mask(wafer)),
    ]

    rows = len(representatives)
    cols = len(schemes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.asarray(axes)

    for row_idx, ((_, meta_row), wafer_map) in enumerate(
        zip(representatives.iterrows(), selected_maps)
    ):
        wafer = np.asarray(wafer_map)
        for col_idx, (scheme_name, maker) in enumerate(schemes):
            ax = axes_arr[row_idx, col_idx]
            if maker is None:
                sample_mask = np.zeros(wafer.shape, dtype=bool)
            else:
                sample_mask = maker(wafer)

            if col_idx == 0:
                title = f"{meta_row['failureType']}\nOriginal"
            else:
                title = f"{scheme_name}"
            draw_panel(ax, wafer, sample_mask, title)

    fig.suptitle(
        "Sparse Sampling Mask Examples\n"
        "black = sampled normal die, yellow = sampled defect die",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote sampling mask examples: {args.out}")


if __name__ == "__main__":
    main()
