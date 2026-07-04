from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import normalized_geometry, quadrant_ids, radial_zone
from src.sampling import (
    defect_mask,
    make_coverage_sampling_mask,
    nearest_valid_cell,
    valid_die_mask,
    wafer_center,
)


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "initial_probe_density_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "32_initial_probe_density_v1"
GROUP_MAP = {
    "Center": "center_global",
    "Donut": "center_global",
    "Near-full": "center_global",
    "Edge-Ring": "edge",
    "Edge-Loc": "edge",
    "Loc": "irregular_local",
    "Scratch": "irregular_local",
    "Random": "irregular_local",
}
WEAK_PATTERNS = {"Loc", "Random", "Scratch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate morphology prediction vs initial probe density."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--max-wafers", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def safe_entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values[values > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def center_seed_mask(wafer_map: np.ndarray) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    seed = np.zeros_like(valid, dtype=bool)
    if not valid.any():
        return seed
    cy, cx = wafer_center(valid)
    cell = nearest_valid_cell(valid, cy, cx)
    if cell is not None:
        seed[cell] = True
    return seed


def make_initial_coverage_mask(wafer_map: np.ndarray, target_count: int) -> np.ndarray:
    valid = valid_die_mask(wafer_map)
    if target_count <= 0 or not valid.any():
        return np.zeros_like(valid, dtype=bool)
    target_count = min(target_count, int(valid.sum()))
    first = center_seed_mask(wafer_map)
    if target_count == 1:
        return first
    follow = make_coverage_sampling_mask(
        wafer_map,
        n_points=target_count - int(first.sum()),
        existing_mask=first,
    )
    return (first | follow) & valid


def density_label(density: float) -> str:
    return f"density_{density:.3f}".replace(".", "p")


def sample_geometry_features(wafer_map: np.ndarray, sample_mask: np.ndarray) -> dict[str, float | int]:
    wafer = np.asarray(wafer_map)
    valid = valid_die_mask(wafer)
    defects = defect_mask(wafer)
    geometry = normalized_geometry(wafer)
    cy = float(geometry["cy"])
    cx = float(geometry["cx"])
    max_radius = float(geometry["max_radius"])
    sampled_valid = sample_mask & valid
    sampled_defects = sampled_valid & defects

    sample_y, sample_x = np.nonzero(sampled_valid)
    hit_y, hit_x = np.nonzero(sampled_defects)
    valid_count = int(valid.sum())
    sampled_count = int(sampled_valid.sum())
    hit_count = int(sampled_defects.sum())

    if hit_count:
        hit_dy = hit_y.astype(float) - cy
        hit_dx = hit_x.astype(float) - cx
        hit_radius = np.sqrt(hit_dy**2 + hit_dx**2) / max_radius
        hit_angle = np.arctan2(hit_dy, hit_dx)
        zones = radial_zone(hit_radius)
        quadrants = quadrant_ids(hit_y.astype(float), hit_x.astype(float), cy, cx)
        zone_counts = np.array([(zones == idx).sum() for idx in [0, 1, 2]], dtype=float)
        quadrant_counts = np.array([(quadrants == idx).sum() for idx in [1, 2, 3, 4]], dtype=float)
        centroid_y_norm = float((hit_y.mean() - cy) / max_radius)
        centroid_x_norm = float((hit_x.mean() - cx) / max_radius)
        centroid_radius_norm = float(
            np.sqrt((hit_y.mean() - cy) ** 2 + (hit_x.mean() - cx) ** 2) / max_radius
        )
        angular_concentration = float(np.abs(np.mean(np.exp(1j * hit_angle))))
        radius_mean = float(hit_radius.mean())
        radius_std = float(hit_radius.std())
        radius_min = float(hit_radius.min())
        radius_max = float(hit_radius.max())
    else:
        zone_counts = np.zeros(3, dtype=float)
        quadrant_counts = np.zeros(4, dtype=float)
        centroid_y_norm = 0.0
        centroid_x_norm = 0.0
        centroid_radius_norm = 0.0
        angular_concentration = 0.0
        radius_mean = 0.0
        radius_std = 0.0
        radius_min = 0.0
        radius_max = 0.0

    if sampled_count:
        sample_radius = np.sqrt((sample_y - cy) ** 2 + (sample_x - cx) ** 2) / max_radius
        sample_zones = radial_zone(sample_radius)
        sample_zone_counts = np.array(
            [(sample_zones == idx).sum() for idx in [0, 1, 2]], dtype=float
        )
    else:
        sample_zone_counts = np.zeros(3, dtype=float)

    features: dict[str, float | int] = {
        "map_height": int(wafer.shape[0]),
        "map_width": int(wafer.shape[1]),
        "valid_die_count": valid_count,
        "log_valid_die_count": float(np.log1p(valid_count)),
        "sampled_valid_count": sampled_count,
        "sampling_density": sampled_count / valid_count if valid_count else 0.0,
        "hit_count": hit_count,
        "hit_ratio": hit_count / sampled_count if sampled_count else 0.0,
        "no_hit": int(hit_count == 0),
        "hit_centroid_y_norm": centroid_y_norm,
        "hit_centroid_x_norm": centroid_x_norm,
        "hit_centroid_radius_norm": centroid_radius_norm,
        "hit_radius_mean": radius_mean,
        "hit_radius_std": radius_std,
        "hit_radius_min": radius_min,
        "hit_radius_max": radius_max,
        "hit_radius_range": radius_max - radius_min,
        "hit_radius_cv": radius_std / radius_mean if radius_mean else 0.0,
        "hit_angular_concentration": angular_concentration,
        "hit_zone_entropy": safe_entropy(zone_counts),
        "hit_quadrant_entropy": safe_entropy(quadrant_counts),
        "hit_zone_imbalance": float(zone_counts.max() / max(hit_count, 1)),
        "hit_quadrant_imbalance": float(quadrant_counts.max() / max(hit_count, 1)),
        "hit_active_quadrants": int((quadrant_counts > 0).sum()),
    }

    for idx, name in enumerate(["center", "mid", "edge"]):
        features[f"hit_zone_{name}_count"] = int(zone_counts[idx])
        features[f"hit_zone_{name}_fraction"] = float(zone_counts[idx] / max(hit_count, 1))
        features[f"sample_zone_{name}_fraction"] = float(sample_zone_counts[idx] / max(sampled_count, 1))

    for idx in range(4):
        features[f"hit_quadrant_q{idx + 1}_count"] = int(quadrant_counts[idx])
        features[f"hit_quadrant_q{idx + 1}_fraction"] = float(quadrant_counts[idx] / max(hit_count, 1))

    return features


def clean_failure_type(row) -> str:
    value = getattr(row, "failureType_clean", None)
    if value is None:
        value = row.failureType
    return str(value)


def build_density_dataset(patterned: pd.DataFrame, densities: list[float]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total = len(patterned)
    for pos, row in enumerate(patterned.itertuples(index=True), start=1):
        wafer_map = np.asarray(row.waferMap)
        valid_count = int(valid_die_mask(wafer_map).sum())
        failure_type = clean_failure_type(row)
        for density in densities:
            target_count = int(np.ceil(valid_count * density))
            sample_mask = make_initial_coverage_mask(wafer_map, target_count)
            records.append(
                {
                    "row_index": int(row.Index),
                    "failureType": failure_type,
                    "failure_group": GROUP_MAP.get(failure_type, "other"),
                    "target_density": density,
                    "target_sample_count": target_count,
                    "initial_probe_type": "coverage_initial",
                    **sample_geometry_features(wafer_map, sample_mask),
                }
            )
        if pos % 1000 == 0 or pos == total:
            print(f"initial-density feature rows processed: {pos:,}/{total:,}")
    data = pd.DataFrame.from_records(records)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {
        "row_index",
        "failureType",
        "failure_group",
        "target_density",
        "initial_probe_type",
    }
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def top_k_metric(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray, k: int) -> float:
    if proba.shape[1] < k:
        return float("nan")
    return float(top_k_accuracy_score(y_true, proba, k=k, labels=labels))


def evaluate_density_models(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    reports: list[pd.DataFrame] = []
    confusions: list[pd.DataFrame] = []
    cols = feature_columns(data)

    for density in sorted(data["target_density"].unique()):
        subset = data[data["target_density"] == density].copy()
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            random_state=seed,
            stratify=subset["failureType"],
        )
        x_train = train_df[cols]
        x_test = test_df[cols]
        for target_type, target_col, topk in [
            ("exact", "failureType", 3),
            ("group", "failure_group", 2),
        ]:
            y_train = train_df[target_col]
            y_test = test_df[target_col]
            labels = np.array(sorted(subset[target_col].unique()))
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=18,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            proba = model.predict_proba(x_test)
            proba_labels = np.asarray(model.classes_)

            if target_type == "exact":
                weak_mask = y_test.isin(WEAK_PATTERNS).to_numpy()
                weak_recall = (
                    float((pred[weak_mask] == y_test.to_numpy()[weak_mask]).mean())
                    if weak_mask.any()
                    else np.nan
                )
            else:
                weak_mask = y_test.eq("irregular_local").to_numpy()
                weak_recall = (
                    float((pred[weak_mask] == y_test.to_numpy()[weak_mask]).mean())
                    if weak_mask.any()
                    else np.nan
                )

            metrics.append(
                {
                    "target_density": density,
                    "target_type": target_type,
                    "model": "random_forest",
                    "train_wafers": len(train_df),
                    "test_wafers": len(test_df),
                    "features": len(cols),
                    "mean_sampled_valid_count": float(test_df["sampled_valid_count"].mean()),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro")),
                    "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
                    "topk_accuracy": top_k_metric(y_test.to_numpy(), proba, proba_labels, topk),
                    "weak_or_irregular_recall": weak_recall,
                    "no_hit_rate": float(test_df["no_hit"].mean()),
                    "mean_hit_count": float(test_df["hit_count"].mean()),
                }
            )

            report = pd.DataFrame(
                classification_report(
                    y_test,
                    pred,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
            ).T.reset_index(names="label")
            report.insert(0, "target_type", target_type)
            report.insert(0, "target_density", density)
            reports.append(report)

            confusion = pd.DataFrame(
                confusion_matrix(y_test, pred, labels=labels, normalize="true"),
                index=labels,
                columns=labels,
            )
            confusion.insert(0, "actual_label", confusion.index)
            confusion = confusion.reset_index(drop=True)
            confusion.insert(0, "target_type", target_type)
            confusion.insert(0, "target_density", density)
            confusions.append(confusion)

    return (
        pd.DataFrame.from_records(metrics),
        pd.concat(reports, ignore_index=True),
        pd.concat(confusions, ignore_index=True),
    )


def plot_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = metrics.copy()
    data["density_pct"] = data["target_density"] * 100.0
    long = data.melt(
        id_vars=["density_pct", "target_type"],
        value_vars=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "topk_accuracy",
            "weak_or_irregular_recall",
            "no_hit_rate",
        ],
        var_name="metric",
        value_name="value",
    )
    grid = sns.relplot(
        data=long,
        x="density_pct",
        y="value",
        hue="metric",
        col="target_type",
        kind="line",
        marker="o",
        height=4.4,
        aspect=1.35,
    )
    for ax in grid.axes.flat:
        ax.set_ylim(0.0, 1.05)
    grid.set_axis_labels("Initial probe density (%)", "Score")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Morphology Prediction vs Initial Probe Density", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(fig_dir / "morphology_accuracy_vs_initial_density.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)

    exact = data[data["target_type"] == "exact"]
    plt.figure(figsize=(7.8, 4.8))
    sns.lineplot(data=exact, x="density_pct", y="mean_hit_count", marker="o", label="mean hit count")
    sns.lineplot(data=exact, x="density_pct", y="mean_sampled_valid_count", marker="o", label="mean sampled count")
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "sampled_count_and_hit_count_vs_density.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    patterned = pd.read_pickle(args.input)
    if args.max_wafers:
        patterned = patterned.head(args.max_wafers).copy()

    dataset = build_density_dataset(patterned, args.densities)
    metrics, reports, confusions = evaluate_density_models(
        dataset,
        test_size=args.test_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )

    dataset.to_csv(args.out_dir / "initial_probe_density_dataset.csv", index=False)
    metrics.to_csv(args.out_dir / "initial_probe_density_model_metrics.csv", index=False)
    reports.to_csv(args.out_dir / "initial_probe_density_pattern_report.csv", index=False)
    confusions.to_csv(args.out_dir / "initial_probe_density_confusion_matrix.csv", index=False)
    plot_metrics(metrics, args.fig_dir)

    print(f"wrote initial probe density outputs to {args.out_dir}")
    print(f"wrote initial probe density figures to {args.fig_dir}")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
