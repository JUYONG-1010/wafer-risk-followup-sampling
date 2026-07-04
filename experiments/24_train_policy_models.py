from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


DEFAULT_FEATURES = Path("data") / "processed" / "policy_learning" / "first_pass_features.csv"
DEFAULT_ACTION_OUTCOMES = Path("data") / "processed" / "policy_learning" / "action_outcomes.csv"
DEFAULT_BEST_ACTIONS = (
    Path("data")
    / "processed"
    / "policy_learning"
    / "cost_sensitivity"
    / "cost_sensitivity_best_actions.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "policy_learning" / "model_training"

FEATURE_COLUMNS = [
    "map_height",
    "map_width",
    "valid_die_count",
    "first_sampled_valid_count",
    "first_sampling_density",
    "first_sampled_defects",
    "first_sampled_defect_ratio",
    "first_no_hit",
    "first_hit_count",
    "first_edge_hit_count",
    "first_center_hit_count",
    "first_mid_hit_count",
    "first_has_edge_hit",
    "first_has_center_hit",
    "first_hit_radius_mean",
    "first_hit_radius_max",
    "first_hit_radius_min",
    "first_sample_radius_mean",
    "first_sample_radius_max",
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train interpretable policy models that predict follow-up action "
            "from first-pass sampling features."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--action-outcomes", type=Path, default=DEFAULT_ACTION_OUTCOMES)
    parser.add_argument("--best-actions", type=Path, default=DEFAULT_BEST_ACTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cost-weights",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03],
        help="Cost regimes to train and evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--severe-miss-weight", type=float, default=10.0)
    parser.add_argument("--absolute-error-weight", type=float, default=1.0)
    parser.add_argument("--underestimation-weight", type=float, default=0.25)
    return parser.parse_args()


def sensitivity_score(
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


def can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return bool((counts >= 2).all())


def train_models(random_state: int) -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=80,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=40,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
    }


def action_order_for(values: pd.Series) -> list[str]:
    observed = set(values.astype(str))
    order = [action for action in DEFAULT_ACTION_ORDER if action in observed]
    order += [action for action in sorted(observed) if action not in order]
    return order


def prepare_target(
    features: pd.DataFrame,
    best_actions: pd.DataFrame,
    cost_weight: float,
) -> pd.DataFrame:
    target = best_actions[np.isclose(best_actions["cost_weight"], cost_weight)].copy()
    target = target[["row_index", "action"]].rename(columns={"action": "best_action"})
    data = features.merge(target, on="row_index", how="inner", validate="one_to_one")
    return data


def lookup_predicted_outcomes(
    action_outcomes: pd.DataFrame,
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    cost_weight: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    chosen = pd.DataFrame(
        {
            "row_index": test_rows["row_index"].to_numpy(),
            "predicted_action": predictions,
        }
    )
    outcomes = action_outcomes.merge(
        chosen,
        left_on=["row_index", "action"],
        right_on=["row_index", "predicted_action"],
        how="inner",
        validate="one_to_one",
    )
    outcomes["cost_weight"] = cost_weight
    outcomes["evaluation_score"] = sensitivity_score(
        outcomes,
        cost_weight=cost_weight,
        severe_miss_weight=args.severe_miss_weight,
        absolute_error_weight=args.absolute_error_weight,
        underestimation_weight=args.underestimation_weight,
    )
    return outcomes


def summarize_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    cost_weight: float,
    accuracy: float | None = None,
    macro_f1: float | None = None,
) -> dict[str, float | int | str | None]:
    return {
        "cost_weight": cost_weight,
        "strategy": strategy_name,
        "wafers": int(data["row_index"].nunique()),
        "mean_sampled_valid_count": float(data["sampled_valid_count"].mean()),
        "mean_added_valid_count": float(data["added_valid_count"].mean()),
        "mean_absolute_error": float(data["absolute_error"].mean()),
        "severe_miss_rate": float(data["severe_miss"].mean()),
        "underestimation_rate": float(data["underestimated"].mean()),
        "mean_score": float(data["evaluation_score"].mean()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def fixed_baseline_summaries(
    action_outcomes: pd.DataFrame,
    test_row_indices: pd.Series,
    cost_weight: float,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str | None]]:
    subset = action_outcomes[action_outcomes["row_index"].isin(test_row_indices)].copy()
    subset["evaluation_score"] = sensitivity_score(
        subset,
        cost_weight=cost_weight,
        severe_miss_weight=args.severe_miss_weight,
        absolute_error_weight=args.absolute_error_weight,
        underestimation_weight=args.underestimation_weight,
    )
    records = []
    for action in action_order_for(subset["action"]):
        action_data = subset[subset["action"] == action]
        if len(action_data):
            records.append(
                summarize_strategy(action_data, f"fixed_{action}", cost_weight)
            )
    return records


def oracle_summary(
    action_outcomes: pd.DataFrame,
    test_row_indices: pd.Series,
    cost_weight: float,
    args: argparse.Namespace,
) -> dict[str, float | int | str | None]:
    subset = action_outcomes[action_outcomes["row_index"].isin(test_row_indices)].copy()
    subset["evaluation_score"] = sensitivity_score(
        subset,
        cost_weight=cost_weight,
        severe_miss_weight=args.severe_miss_weight,
        absolute_error_weight=args.absolute_error_weight,
        underestimation_weight=args.underestimation_weight,
    )
    subset["action"] = pd.Categorical(
        subset["action"], action_order_for(subset["action"]), ordered=True
    )
    best = (
        subset.sort_values(
            ["row_index", "evaluation_score", "added_valid_count", "action"]
        )
        .groupby("row_index", observed=False)
        .head(1)
    )
    return summarize_strategy(best, "oracle_best_by_score", cost_weight)


def action_distribution(
    row_indices: pd.Series,
    actions: pd.Series | np.ndarray,
    cost_weight: float,
    strategy: str,
) -> pd.DataFrame:
    data = pd.DataFrame({"row_index": row_indices.to_numpy(), "action": actions})
    action_order = action_order_for(pd.Series(actions))
    counts = data["action"].value_counts().reindex(action_order, fill_value=0)
    total = len(data)
    return pd.DataFrame(
        {
            "cost_weight": cost_weight,
            "strategy": strategy,
            "action": counts.index.astype(str),
            "wafers": counts.values,
            "fraction": counts.values / total,
        }
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    action_outcomes = pd.read_csv(args.action_outcomes)
    best_actions = pd.read_csv(args.best_actions)

    missing = [col for col in FEATURE_COLUMNS if col not in features.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    summary_records: list[dict[str, float | int | str | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    tree_texts: dict[str, str] = {}

    for cost_weight in args.cost_weights:
        dataset = prepare_target(features, best_actions, cost_weight)
        x = dataset[FEATURE_COLUMNS]
        y = dataset["best_action"].astype(str)
        stratify = y if can_stratify(y) else None
        train_idx, test_idx = train_test_split(
            dataset.index,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        train = dataset.loc[train_idx].copy()
        test = dataset.loc[test_idx].copy()
        x_train = train[FEATURE_COLUMNS]
        y_train = train["best_action"].astype(str)
        x_test = test[FEATURE_COLUMNS]
        y_test = test["best_action"].astype(str)

        summary_records.extend(
            fixed_baseline_summaries(
                action_outcomes,
                test["row_index"],
                cost_weight,
                args,
            )
        )
        summary_records.append(
            oracle_summary(action_outcomes, test["row_index"], cost_weight, args)
        )
        distribution_frames.append(
            action_distribution(test["row_index"], y_test, cost_weight, "oracle_label")
        )

        for model_name, model in train_models(args.random_state).items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, pred))
            macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
            outcomes = lookup_predicted_outcomes(
                action_outcomes,
                test,
                pred,
                cost_weight,
                args,
            )
            summary_records.append(
                summarize_strategy(
                    outcomes,
                    model_name,
                    cost_weight,
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "cost_weight": cost_weight,
                        "model": model_name,
                        "row_index": test["row_index"].to_numpy(),
                        "true_best_action": y_test.to_numpy(),
                        "predicted_action": pred,
                    }
                )
            )
            distribution_frames.append(
                action_distribution(
                    test["row_index"], pred, cost_weight, model_name
                )
            )
            if hasattr(model, "feature_importances_"):
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "cost_weight": cost_weight,
                            "model": model_name,
                            "feature": FEATURE_COLUMNS,
                            "importance": model.feature_importances_,
                        }
                    )
                )
            if model_name == "decision_tree":
                tree_texts[f"cost_{cost_weight:g}"] = export_text(
                    model,
                    feature_names=FEATURE_COLUMNS,
                    max_depth=5,
                )

    summary = pd.DataFrame.from_records(summary_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    distributions = pd.concat(distribution_frames, ignore_index=True)
    importances = pd.concat(importance_frames, ignore_index=True)

    summary.to_csv(args.out_dir / "policy_model_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "policy_model_predictions.csv", index=False)
    distributions.to_csv(args.out_dir / "policy_model_action_distribution.csv", index=False)
    importances.to_csv(args.out_dir / "policy_model_feature_importance.csv", index=False)

    with (args.out_dir / "decision_tree_rules.json").open("w", encoding="utf-8") as fp:
        json.dump(tree_texts, fp, indent=2)
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as fp:
        json.dump(FEATURE_COLUMNS, fp, indent=2)

    print(f"wrote model summary: {args.out_dir / 'policy_model_summary.csv'}")
    print(f"wrote predictions: {args.out_dir / 'policy_model_predictions.csv'}")
    print(f"wrote action distributions: {args.out_dir / 'policy_model_action_distribution.csv'}")
    print(f"wrote feature importances: {args.out_dir / 'policy_model_feature_importance.csv'}")
    print(f"wrote tree rules: {args.out_dir / 'decision_tree_rules.json'}")


if __name__ == "__main__":
    main()
