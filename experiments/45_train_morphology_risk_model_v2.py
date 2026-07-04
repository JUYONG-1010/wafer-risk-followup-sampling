from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
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


DEFAULT_INPUT = (
    Path("data")
    / "processed"
    / "morphology_risk_v1"
    / "morphology_risk_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "morphology_risk_v2"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "31_morphology_risk_v2"

WEAK_PATTERNS = {"Loc", "Random", "Scratch"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve and diagnose first-pass morphology-risk prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy_most_frequent", "logistic_balanced", "random_forest", "extra_trees"],
        choices=[
            "dummy_most_frequent",
            "logistic_balanced",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
        ],
    )
    return parser.parse_args()


def add_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["failure_group"] = out["failureType"].map(GROUP_MAP).fillna("other")

    out["aspect_ratio"] = out["map_width"] / out["map_height"].replace(0, np.nan)
    out["log_valid_die_count"] = np.log1p(out["valid_die_count"])
    out["edge_hit_ratio"] = out["first_edge_hit_count"] / out["first_hit_count"].clip(lower=1)
    out["mid_hit_ratio"] = out["first_mid_hit_count"] / out["first_hit_count"].clip(lower=1)
    out["center_hit_ratio"] = out["first_center_hit_count"] / out["first_hit_count"].clip(lower=1)
    out["edge_minus_center_hits"] = out["first_edge_hit_count"] - out["first_center_hit_count"]
    out["edge_plus_mid_hit_ratio"] = (
        out["first_edge_hit_count"] + out["first_mid_hit_count"]
    ) / out["first_hit_count"].clip(lower=1)
    out["hit_radius_range"] = out["hit_radius_max"] - out["hit_radius_min"]
    out["hit_radius_cv"] = out["hit_radius_std"] / out["hit_radius_mean"].replace(0, np.nan)

    quadrant_count_cols = [f"hit_quadrant_q{i}_count" for i in range(1, 5)]
    if all(col in out.columns for col in quadrant_count_cols):
        q_counts = out[quadrant_count_cols]
        out["hit_quadrant_count_range"] = q_counts.max(axis=1) - q_counts.min(axis=1)
        out["hit_active_quadrants"] = (q_counts > 0).sum(axis=1)

    site_cols = sorted([col for col in out.columns if col.startswith("site_hit_")])
    if site_cols:
        out["site_hit_bitcount"] = out[site_cols].sum(axis=1)
        weights = np.arange(1, len(site_cols) + 1)
        out["site_hit_weighted_code"] = (out[site_cols] * weights).sum(axis=1)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {"row_index", "failureType", "failure_group", "first_pass_type"}
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def top_k_metric(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray, k: int) -> float:
    if proba.shape[1] < k:
        return float("nan")
    return float(top_k_accuracy_score(y_true, proba, k=k, labels=labels))


def make_models(seed: int, n_jobs: int, selected: list[str]) -> dict[str, object]:
    all_models = {
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
            max_depth=18,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=24,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
    }
    return {name: all_models[name] for name in selected}


def evaluate_model(
    first_pass_type: str,
    target_type: str,
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else np.zeros((len(x_test), len(labels)))
    proba_labels = np.asarray(model.classes_) if hasattr(model, "classes_") else labels

    wrong = pred != y_test.to_numpy()
    if target_type == "exact":
        actual_groups = y_test.map(GROUP_MAP).to_numpy()
        pred_groups = pd.Series(pred).map(GROUP_MAP).fillna("other").to_numpy()
        cross_group_error_rate = float(((actual_groups != pred_groups) & wrong).mean())
        weak_recall_subset = y_test.isin(WEAK_PATTERNS).to_numpy()
    else:
        cross_group_error_rate = np.nan
        weak_recall_subset = y_test.eq("irregular_local").to_numpy()

    weak_recall = np.nan
    if weak_recall_subset.any():
        weak_recall = float((pred[weak_recall_subset] == y_test.to_numpy()[weak_recall_subset]).mean())

    metrics = {
        "first_pass_type": first_pass_type,
        "target_type": target_type,
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "top2_accuracy": top_k_metric(y_test.to_numpy(), proba, proba_labels, 2),
        "top3_accuracy": top_k_metric(y_test.to_numpy(), proba, proba_labels, 3),
        "cross_group_error_rate": cross_group_error_rate,
        "weak_or_irregular_recall": weak_recall,
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="label")
    report.insert(0, "model", model_name)
    report.insert(0, "target_type", target_type)
    report.insert(0, "first_pass_type", first_pass_type)

    confusion = pd.DataFrame(
        confusion_matrix(y_test, pred, labels=labels, normalize="true"),
        index=labels,
        columns=labels,
    )
    confusion.insert(0, "actual_label", confusion.index)
    confusion = confusion.reset_index(drop=True)
    confusion.insert(0, "model", model_name)
    confusion.insert(0, "target_type", target_type)
    confusion.insert(0, "first_pass_type", first_pass_type)
    return metrics, report, confusion


def run_experiments(
    data: pd.DataFrame,
    seed: int,
    test_size: float,
    n_jobs: int,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, object]] = []
    report_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []

    for first_pass_type in ["grid9", "grid25"]:
        subset = data[data["first_pass_type"] == first_pass_type].copy()
        cols = feature_columns(subset)
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            random_state=seed,
            stratify=subset["failureType"],
        )
        x_train = train_df[cols]
        x_test = test_df[cols]

        for target_type, target_col in [("exact", "failureType"), ("group", "failure_group")]:
            labels = np.array(sorted(subset[target_col].unique()))
            y_train = train_df[target_col]
            y_test = test_df[target_col]
            for model_name, model in make_models(seed, n_jobs, models).items():
                metrics, report, confusion = evaluate_model(
                    first_pass_type,
                    target_type,
                    model_name,
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    labels,
                )
                metrics["train_wafers"] = len(train_df)
                metrics["test_wafers"] = len(test_df)
                metrics["features"] = len(cols)
                metric_records.append(metrics)
                report_frames.append(report)
                confusion_frames.append(confusion)

    return (
        pd.DataFrame.from_records(metric_records),
        pd.concat(report_frames, ignore_index=True),
        pd.concat(confusion_frames, ignore_index=True),
    )


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (first_pass_type, target_type), subset in metrics.groupby(["first_pass_type", "target_type"]):
        ranked = subset.sort_values(
            ["balanced_accuracy", "macro_f1", "top3_accuracy"],
            ascending=False,
        )
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "weak_or_irregular_recall"]
    long = metrics.melt(
        id_vars=["first_pass_type", "target_type", "model"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    grid = sns.catplot(
        data=long,
        x="model",
        y="value",
        hue="metric",
        row="target_type",
        col="first_pass_type",
        kind="bar",
        height=3.8,
        aspect=1.45,
    )
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.0, 1.0)
    grid.set_axis_labels("Model", "Score")
    grid.set_titles("{row_name} / {col_name}")
    grid.fig.suptitle("First-Pass Morphology Prediction v2", y=1.02)
    grid.fig.tight_layout()
    grid.fig.savefig(fig_dir / "morphology_v2_model_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_best_confusions(confusion: pd.DataFrame, best: pd.DataFrame, fig_dir: Path) -> None:
    for row in best.itertuples(index=False):
        subset = confusion[
            (confusion["first_pass_type"] == row.first_pass_type)
            & (confusion["target_type"] == row.target_type)
            & (confusion["model"] == row.model)
        ]
        if subset.empty:
            continue
        labels = subset["actual_label"].tolist()
        matrix = subset.drop(columns=["first_pass_type", "target_type", "model", "actual_label"])
        matrix = matrix[labels]
        plt.figure(figsize=(8.0, 6.6))
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
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{row.model} confusion ({row.first_pass_type}, {row.target_type})")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{row.first_pass_type}_{row.target_type}_{row.model}_confusion.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    data = add_enhanced_features(raw)
    metrics, report, confusion = run_experiments(
        data,
        seed=args.seed,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        models=args.models,
    )
    best = best_models(metrics)

    data.to_csv(args.out_dir / "morphology_risk_v2_dataset.csv", index=False)
    metrics.to_csv(args.out_dir / "morphology_risk_v2_model_metrics.csv", index=False)
    report.to_csv(args.out_dir / "morphology_risk_v2_pattern_report.csv", index=False)
    confusion.to_csv(args.out_dir / "morphology_risk_v2_confusion_matrix.csv", index=False)
    best.to_csv(args.out_dir / "morphology_risk_v2_best_models.csv", index=False)

    plot_metrics(metrics, args.fig_dir)
    plot_best_confusions(confusion, best, args.fig_dir)

    print(f"wrote morphology v2 outputs to {args.out_dir}")
    print(f"wrote morphology v2 figures to {args.fig_dir}")
    print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
