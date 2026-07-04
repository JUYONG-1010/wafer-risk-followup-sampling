from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import (
    first_pass_summary,
    make_first_pass_mask,
    normalized_geometry,
    quadrant_ids,
    radial_zone,
)
from src.sampling import defect_mask, fractional_grid_points, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "morphology_risk_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "24_morphology_risk_v1"
FIRST_PASS_TYPES = ("grid9", "grid25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train morphology-risk models from first-pass sparse observations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--max-wafers", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values[values > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def grid_size_for_first_pass(first_pass_type: str) -> int:
    if first_pass_type == "grid9":
        return 3
    if first_pass_type == "grid25":
        return 5
    raise ValueError(f"Unsupported first_pass_type: {first_pass_type}")


def make_site_hit_features(wafer_map: np.ndarray, first_pass_type: str) -> dict[str, int]:
    grid_size = grid_size_for_first_pass(first_pass_type)
    fractions = np.linspace(0.0, 1.0, grid_size)
    points = fractional_grid_points(wafer_map, fractions, fractions)
    defects = defect_mask(wafer_map)
    features: dict[str, int] = {}

    idx = 0
    for row_pos in range(grid_size):
        for col_pos in range(grid_size):
            key = f"site_hit_r{row_pos}_c{col_pos}"
            if idx >= len(points):
                features[key] = 0
            else:
                y, x = points[idx]
                features[key] = int(defects[y, x])
            idx += 1
    return features


def wafer_morphology_features(
    wafer_map: np.ndarray,
    first_pass_type: str,
    row_index: int,
    failure_type: str,
) -> dict[str, float | int | str]:
    wafer = np.asarray(wafer_map)
    first_mask = make_first_pass_mask(wafer, first_pass_type)
    valid = valid_die_mask(wafer)
    defects = defect_mask(wafer)
    geometry = normalized_geometry(wafer)
    cy = float(geometry["cy"])
    cx = float(geometry["cx"])
    max_radius = float(geometry["max_radius"])
    summary = first_pass_summary(wafer, first_mask)

    hit_y, hit_x = np.nonzero(first_mask & defects)
    hit_count = int(len(hit_x))
    if hit_count:
        hit_dy = hit_y.astype(float) - cy
        hit_dx = hit_x.astype(float) - cx
        hit_radius = np.sqrt(hit_dy**2 + hit_dx**2) / max_radius
        hit_angle = np.arctan2(hit_dy, hit_dx)
        hit_quadrants = quadrant_ids(hit_y.astype(float), hit_x.astype(float), cy, cx)
        quadrant_counts = np.array(
            [(hit_quadrants == q).sum() for q in [1, 2, 3, 4]], dtype=float
        )
        zone_counts = np.array(
            [
                (radial_zone(hit_radius) == zone).sum()
                for zone in [0, 1, 2]
            ],
            dtype=float,
        )
        centroid_y_norm = float((hit_y.mean() - cy) / max_radius)
        centroid_x_norm = float((hit_x.mean() - cx) / max_radius)
        centroid_radius_norm = float(
            np.sqrt((hit_y.mean() - cy) ** 2 + (hit_x.mean() - cx) ** 2) / max_radius
        )
        resultant = np.abs(np.mean(np.exp(1j * hit_angle)))
        radius_mean = float(hit_radius.mean())
        radius_std = float(hit_radius.std())
        radius_max = float(hit_radius.max())
        radius_min = float(hit_radius.min())
    else:
        quadrant_counts = np.zeros(4, dtype=float)
        zone_counts = np.zeros(3, dtype=float)
        centroid_y_norm = 0.0
        centroid_x_norm = 0.0
        centroid_radius_norm = 0.0
        resultant = 0.0
        radius_mean = 0.0
        radius_std = 0.0
        radius_max = 0.0
        radius_min = 0.0

    first_valid_count = max(int(summary["first_sampled_valid_count"]), 1)
    features: dict[str, float | int | str] = {
        "row_index": row_index,
        "failureType": failure_type,
        "first_pass_type": first_pass_type,
        "first_pass_is_grid25": int(first_pass_type == "grid25"),
        "map_height": int(wafer.shape[0]),
        "map_width": int(wafer.shape[1]),
        "valid_die_count": int(valid.sum()),
        **summary,
        "first_hit_fraction": hit_count / first_valid_count,
        "hit_centroid_y_norm": centroid_y_norm,
        "hit_centroid_x_norm": centroid_x_norm,
        "hit_centroid_radius_norm": centroid_radius_norm,
        "hit_radius_mean": radius_mean,
        "hit_radius_std": radius_std,
        "hit_radius_min": radius_min,
        "hit_radius_max": radius_max,
        "hit_angular_concentration": float(resultant),
        "hit_quadrant_entropy": safe_entropy(quadrant_counts),
        "hit_zone_entropy": safe_entropy(zone_counts),
        "hit_quadrant_imbalance": float(quadrant_counts.max() / max(hit_count, 1)),
        "hit_zone_imbalance": float(zone_counts.max() / max(hit_count, 1)),
    }

    for idx, value in enumerate(quadrant_counts, start=1):
        features[f"hit_quadrant_q{idx}_count"] = int(value)
        features[f"hit_quadrant_q{idx}_fraction"] = float(value / max(hit_count, 1))
    for zone, name in enumerate(["center", "mid", "edge"]):
        value = zone_counts[zone]
        features[f"hit_zone_{name}_count"] = int(value)
        features[f"hit_zone_{name}_fraction"] = float(value / max(hit_count, 1))

    features.update(make_site_hit_features(wafer, first_pass_type))
    return features


def build_dataset(patterned: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    total = len(patterned)
    for pos, row in enumerate(patterned.itertuples(index=True), start=1):
        failure_type = getattr(row, "failureType_clean", None)
        if failure_type is None:
            failure_type = row.failureType
        wafer_map = np.asarray(row.waferMap)
        for first_pass_type in FIRST_PASS_TYPES:
            records.append(
                wafer_morphology_features(
                    wafer_map=wafer_map,
                    first_pass_type=first_pass_type,
                    row_index=int(row.Index),
                    failure_type=str(failure_type),
                )
            )
        if pos % 2500 == 0 or pos == total:
            print(f"morphology feature rows processed: {pos:,}/{total:,}")
    data = pd.DataFrame.from_records(records)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {"row_index", "failureType", "first_pass_type"}
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def top_k_metric(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray, k: int) -> float:
    if proba.shape[1] < k:
        return float("nan")
    return float(top_k_accuracy_score(y_true, proba, k=k, labels=labels))


def evaluate_model(
    name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        proba_labels = np.asarray(model.classes_)
    else:
        proba = np.zeros((len(x_test), len(labels)), dtype=float)
        proba_labels = labels

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "top2_accuracy": top_k_metric(y_test.to_numpy(), proba, proba_labels, 2),
        "top3_accuracy": top_k_metric(y_test.to_numpy(), proba, proba_labels, 3),
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="failureType")
    report.insert(0, "model", name)

    confusion = pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels, normalize="true"),
        index=labels,
        columns=labels,
    )
    confusion.insert(0, "actual_failureType", confusion.index)
    confusion = confusion.reset_index(drop=True)
    confusion.insert(0, "model", name)
    return metrics, report, confusion


def train_and_evaluate(data: pd.DataFrame, seed: int, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_records: list[dict[str, object]] = []
    report_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for first_pass_type in FIRST_PASS_TYPES:
        subset = data[data["first_pass_type"] == first_pass_type].copy()
        cols = feature_columns(subset)
        labels = np.array(sorted(subset["failureType"].unique()))
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            random_state=seed,
            stratify=subset["failureType"],
        )
        x_train = train_df[cols]
        y_train = train_df["failureType"]
        x_test = test_df[cols]
        y_test = test_df["failureType"]

        models = {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_balanced": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
        }

        for model_name, model in models.items():
            metrics, report, confusion = evaluate_model(
                model_name,
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                labels,
            )
            metrics.update(
                {
                    "first_pass_type": first_pass_type,
                    "train_wafers": len(train_df),
                    "test_wafers": len(test_df),
                    "features": len(cols),
                }
            )
            metrics_records.append(metrics)
            report.insert(0, "first_pass_type", first_pass_type)
            confusion.insert(0, "first_pass_type", first_pass_type)
            report_frames.append(report)
            confusion_frames.append(confusion)

            if model_name == "random_forest":
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "first_pass_type": first_pass_type,
                            "feature": cols,
                            "importance": model.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)
                )

    return (
        pd.DataFrame.from_records(metrics_records),
        pd.concat(report_frames, ignore_index=True),
        pd.concat(confusion_frames, ignore_index=True),
        pd.concat(importance_frames, ignore_index=True),
    )


def plot_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "top2_accuracy", "top3_accuracy"]
    long = metrics.melt(
        id_vars=["first_pass_type", "model"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    grid = sns.catplot(
        data=long,
        x="model",
        y="value",
        hue="metric",
        col="first_pass_type",
        kind="bar",
        height=4.4,
        aspect=1.35,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0.0, 1.0)
    grid.set_axis_labels("Model", "Score")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Morphology Risk Prediction from First-Pass Sparse Observations", y=1.04)
    grid.fig.tight_layout()
    grid.fig.savefig(fig_dir / "morphology_model_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_confusion(confusion: pd.DataFrame, fig_dir: Path) -> None:
    for first_pass_type in FIRST_PASS_TYPES:
        subset = confusion[
            (confusion["first_pass_type"] == first_pass_type)
            & (confusion["model"] == "random_forest")
        ].copy()
        if subset.empty:
            continue
        labels = subset["actual_failureType"].tolist()
        matrix = subset.drop(columns=["first_pass_type", "model", "actual_failureType"])
        matrix = matrix[labels]
        plt.figure(figsize=(8.4, 6.7))
        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="white",
        )
        plt.xlabel("Predicted pattern")
        plt.ylabel("Actual pattern")
        plt.title(f"Random Forest Normalized Confusion Matrix ({first_pass_type})")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{first_pass_type}_random_forest_confusion.png", dpi=180)
        plt.close()


def plot_importance(importance: pd.DataFrame, fig_dir: Path) -> None:
    for first_pass_type in FIRST_PASS_TYPES:
        subset = importance[importance["first_pass_type"] == first_pass_type].head(18)
        if subset.empty:
            continue
        plt.figure(figsize=(8.8, 5.2))
        sns.barplot(data=subset, y="feature", x="importance", color="#4C78A8")
        plt.xlabel("Random Forest importance")
        plt.ylabel("Feature")
        plt.title(f"Top Morphology-Risk Features ({first_pass_type})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{first_pass_type}_feature_importance.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    patterned = pd.read_pickle(args.input)
    if args.max_wafers:
        patterned = patterned.head(args.max_wafers).copy()

    dataset = build_dataset(patterned)
    metrics, pattern_report, confusion, importance = train_and_evaluate(
        dataset,
        seed=args.seed,
        test_size=args.test_size,
    )

    dataset.to_csv(args.out_dir / "morphology_risk_dataset.csv", index=False)
    metrics.to_csv(args.out_dir / "morphology_risk_model_metrics.csv", index=False)
    pattern_report.to_csv(args.out_dir / "morphology_risk_pattern_report.csv", index=False)
    confusion.to_csv(args.out_dir / "morphology_risk_confusion_matrix.csv", index=False)
    importance.to_csv(args.out_dir / "morphology_risk_feature_importance.csv", index=False)

    plot_metrics(metrics, args.fig_dir)
    plot_confusion(confusion, args.fig_dir)
    plot_importance(importance, args.fig_dir)

    print(f"wrote morphology-risk data to {args.out_dir}")
    print(f"wrote morphology-risk figures to {args.fig_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
