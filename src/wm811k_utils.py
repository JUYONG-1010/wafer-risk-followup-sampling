from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


EMPTY_LABEL = ""


def normalize_nested_label(value: Any) -> str:
    """Convert WM-811K nested label cells into a clean string.

    The original LSWMD pickle often stores labels as nested arrays like
    ``array([['Center']], dtype=object)`` or as empty arrays. Downstream code
    should not need to care about those storage details.
    """
    if value is None:
        return EMPTY_LABEL

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return EMPTY_LABEL
        return normalize_nested_label(value.ravel()[0])

    if isinstance(value, Iterable):
        items = list(value)
        if not items:
            return EMPTY_LABEL
        return normalize_nested_label(items[0])

    if isinstance(value, float) and np.isnan(value):
        return EMPTY_LABEL

    return str(value).strip()


def wafer_stats(wafer_map: Any) -> dict[str, float | int | str]:
    """Return shape and defect-density statistics for one wafer map."""
    arr = np.asarray(wafer_map)
    if arr.ndim != 2:
        return {
            "map_height": 0,
            "map_width": 0,
            "map_shape": "",
            "total_cells": 0,
            "valid_die_count": 0,
            "defect_die_count": 0,
            "defect_ratio_valid": 0.0,
            "defect_ratio_cells": 0.0,
        }

    height, width = arr.shape
    total_cells = int(arr.size)
    valid_die_count = int(np.count_nonzero(arr > 0))
    defect_die_count = int(np.count_nonzero(arr == 2))
    defect_ratio_valid = (
        defect_die_count / valid_die_count if valid_die_count else 0.0
    )
    defect_ratio_cells = defect_die_count / total_cells if total_cells else 0.0

    return {
        "map_height": int(height),
        "map_width": int(width),
        "map_shape": f"{height}x{width}",
        "total_cells": total_cells,
        "valid_die_count": valid_die_count,
        "defect_die_count": defect_die_count,
        "defect_ratio_valid": float(defect_ratio_valid),
        "defect_ratio_cells": float(defect_ratio_cells),
    }


def is_labeled_failure_type(label: str) -> bool:
    """Return whether a normalized failure type is an explicit label."""
    return bool(label)


def is_patterned_failure_type(label: str) -> bool:
    """Return whether a normalized failure type is a defect pattern."""
    return bool(label) and label.lower() != "none"
