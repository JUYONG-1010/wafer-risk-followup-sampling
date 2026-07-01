from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "augmented_morphology_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "49_augmented_morphology_v1"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"
INITIAL_PROBE_SCRIPT = Path("scripts") / "46_evaluate_initial_probe_density.py"

DISCOVERY_PATTERNS = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"}
LOW_BIAS_PATTERNS = {"Random"}
COVERAGE_PATTERNS = {"Near-full"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)
initial_probe = load_module("initial_probe46", INITIAL_PROBE_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate train-only geometric augmentation for morphology/mode prediction."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-sample-seed-offset", type=int, default=1)
    parser.add_argument("--max-train-wafers", type=int, default=1200)
    parser.add_argument("--max-test-wafers", type=int, default=400)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"],
    )
    return parser.parse_args()


def pattern_mode(label: str) -> str:
    if label in DISCOVERY_PATTERNS:
        return "discovery_first"
    if label in LOW_BIAS_PATTERNS:
        return "low_bias_default"
    if label in COVERAGE_PATTERNS:
        return "coverage32_or_uncertain"
    return "low_bias_default"


def transform_map(wafer_map: np.ndarray, transform: str) -> np.ndarray:
    wafer = np.asarray(wafer_map)
    if transform == "identity":
        return wafer.copy()
    if transform == "rot90":
        return np.rot90(wafer, 1).copy()
    if transform == "rot180":
        return np.rot90(wafer, 2).copy()
    if transform == "rot270":
        return np.rot90(wafer, 3).copy()
    if transform == "flip_lr":
        return np.fliplr(wafer).copy()
    if transform == "flip_ud":
        return np.flipud(wafer).copy()
    if transform == "transpose":
        return wafer.T.copy()
    if transform == "anti_transpose":
        return np.fliplr(np.flipud(wafer).T).copy()
    raise ValueError(f"Unknown transform: {transform}")


def select_ids(ids: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    ids = np.asarray(ids)
    if max_count and len(ids) > max_count:
        rng = np.random.default_rng(seed)
        return np.asarray(rng.choice(ids, size=max_count, replace=False))
    return ids


def feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = {
        "row_index",
        "failureType",
        "failure_group",
        "target_density",
        "target_sample_count",
        "initial_probe_type",
        "transform",
        "source_split",
    }
    return [
        col
        for col in data.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
    ]


def build_features(
    patterned: pd.DataFrame,
    wafer_ids: np.ndarray,
    densities: list[float],
    transforms: list[str],
    source_split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = patterned[patterned.index.isin(set(int(value) for value in wafer_ids))]
    total = len(subset)
    for pos, row in enumerate(subset.itertuples(index=True), start=1):
        failure_type = density_policy.failure_type(row)
        for transform in transforms:
            wafer_map = transform_map(np.asarray(row.waferMap), transform)
            valid_count = int(initial_probe.valid_die_mask(wafer_map).sum())
            for density in densities:
                target_count = int(np.ceil(valid_count * density))
                first_mask = initial_probe.make_initial_coverage_mask(wafer_map, target_count)
                rows.append(
                    {
                        "row_index": int(row.Index),
                        "failureType": failure_type,
                        "failure_group": initial_probe.GROUP_MAP.get(failure_type, "other"),
                        "target_density": density,
                        "target_sample_count": target_count,
                        "initial_probe_type": "coverage_initial",
                        "transform": transform,
                        "source_split": source_split,
                        **initial_probe.sample_geometry_features(wafer_map, first_mask),
                    }
                )
        if pos % 200 == 0 or pos == total:
            print(f"{source_split} augmented morphology features: {pos:,}/{total:,}")
    data = pd.DataFrame.from_records(rows)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def evaluate_variant(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    variant_name: str,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_columns(train_data)
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for density in sorted(train_data["target_density"].unique()):
        train = train_data[np.isclose(train_data["target_density"], density)].copy()
        test = test_data[np.isclose(test_data["target_density"], density)].copy()
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(train[cols], train["failureType"])
        proba = model.predict_proba(test[cols])
        pred = model.classes_[np.argmax(proba, axis=1)]
        ranked_labels = model.classes_[np.argsort(proba, axis=1)[:, ::-1]]
        y_true = test["failureType"].astype(str).to_numpy()
        top2 = np.mean(
            [label in set(ranked_labels[idx, : min(2, ranked_labels.shape[1])]) for idx, label in enumerate(y_true)]
        )
        top3 = np.mean(
            [label in set(ranked_labels[idx, : min(3, ranked_labels.shape[1])]) for idx, label in enumerate(y_true)]
        )
        true_mode = test["failureType"].map(pattern_mode)
        pred_mode = pd.Series(pred, index=test.index).map(pattern_mode)
        dangerous = (pred_mode == "discovery_first") & (true_mode != "discovery_first")
        missed = (pred_mode != "discovery_first") & (true_mode == "discovery_first")
        records.append(
            {
                "variant": variant_name,
                "target_density": density,
                "train_rows": len(train),
                "test_rows": len(test),
                "exact_top1_accuracy": float(accuracy_score(test["failureType"], pred)),
                "exact_top2_accuracy": float(top2),
                "exact_top3_accuracy": float(top3),
                "mode_accuracy": float((pred_mode.to_numpy() == true_mode.to_numpy()).mean()),
                "dangerous_aggressive_error_rate": float(dangerous.mean()),
                "missed_discovery_error_rate": float(missed.mean()),
                "mean_top1_confidence": float(np.max(proba, axis=1).mean()),
            }
        )
        pred_df = test[["row_index", "failureType", "target_density"]].copy()
        pred_df["variant"] = variant_name
        pred_df["pred_label"] = pred
        pred_df["true_mode"] = true_mode.to_numpy()
        pred_df["pred_mode"] = pred_mode.to_numpy()
        pred_df["top1_confidence"] = np.max(proba, axis=1)
        pred_df["top1_correct"] = pred_df["pred_label"] == pred_df["failureType"]
        pred_df["mode_correct"] = pred_df["true_mode"] == pred_df["pred_mode"]
        predictions.append(pred_df)
    return pd.DataFrame.from_records(records), pd.concat(predictions, ignore_index=True)


def plot_results(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    long = data.melt(
        id_vars=["variant", "density_pct"],
        value_vars=[
            "exact_top1_accuracy",
            "exact_top2_accuracy",
            "exact_top3_accuracy",
            "mode_accuracy",
            "dangerous_aggressive_error_rate",
        ],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10.2, 5.6))
    sns.lineplot(data=long, x="density_pct", y="value", hue="metric", style="variant", marker="o")
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "augmentation_morphology_metrics.png", dpi=180)
    plt.close()

    gain = summary.pivot_table(
        index="target_density",
        columns="variant",
        values="mode_accuracy",
        aggfunc="mean",
    ).reset_index()
    if {"augmented", "baseline"}.issubset(gain.columns):
        gain["mode_accuracy_delta"] = gain["augmented"] - gain["baseline"]
        plt.figure(figsize=(7.4, 4.6))
        sns.barplot(data=gain, x="target_density", y="mode_accuracy_delta")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Initial probe density")
        plt.ylabel("Mode accuracy delta (augmented - baseline)")
        plt.tight_layout()
        plt.savefig(fig_dir / "augmentation_mode_accuracy_delta.png", dpi=180)
        plt.close()


def write_report(summary: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> None:
    lines = [
        "# Augmented Morphology Evaluation v1",
        "",
        "Purpose: test whether train-only geometric augmentation improves morphology and operating-mode prediction.",
        "",
        "Leakage rule:",
        "",
        "```text",
        "original wafer IDs are split first;",
        "only train wafers are augmented;",
        "test wafers remain original/unaugmented.",
        "```",
        "",
        f"Train wafers used: {args.max_train_wafers if args.max_train_wafers else 'all train split'}",
        f"Test wafers used: {args.max_test_wafers if args.max_test_wafers else 'all test split'}",
        f"Transforms: {', '.join(args.transforms)}",
        "",
        "## Results",
        "",
    ]
    for row in summary.sort_values(["target_density", "variant"]).itertuples(index=False):
        lines.append(
            f"- {row.variant}, {row.target_density:.0%}: "
            f"top1={row.exact_top1_accuracy:.3f}, "
            f"top2={row.exact_top2_accuracy:.3f}, "
            f"top3={row.exact_top3_accuracy:.3f}, "
            f"mode={row.mode_accuracy:.3f}, "
            f"dangerous={row.dangerous_aggressive_error_rate:.3f}, "
            f"missed={row.missed_discovery_error_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If augmented mode accuracy and dangerous aggressive error improve on the unaugmented test wafers, geometric augmentation is a valid next training upgrade.",
            "If exact top1 improves but mode accuracy does not, the benefit is less relevant to the final policy.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    patterned = pd.read_pickle(args.patterned)
    train_wafers, test_wafers = density_policy.split_wafers(
        patterned,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_wafers = select_ids(train_wafers, args.max_train_wafers, args.seed)
    test_wafers = select_ids(
        test_wafers,
        args.max_test_wafers,
        args.seed + args.test_sample_seed_offset,
    )
    densities = [float(value) for value in args.densities]

    baseline_train = build_features(patterned, train_wafers, densities, ["identity"], "train_baseline")
    augmented_train = build_features(patterned, train_wafers, densities, args.transforms, "train_augmented")
    test_data = build_features(patterned, test_wafers, densities, ["identity"], "test_original")

    baseline_summary, baseline_predictions = evaluate_variant(
        baseline_train,
        test_data,
        "baseline",
        args.n_estimators,
        args.n_jobs,
        args.seed,
    )
    augmented_summary, augmented_predictions = evaluate_variant(
        augmented_train,
        test_data,
        "augmented",
        args.n_estimators,
        args.n_jobs,
        args.seed,
    )
    summary = pd.concat([baseline_summary, augmented_summary], ignore_index=True)
    predictions = pd.concat([baseline_predictions, augmented_predictions], ignore_index=True)

    summary.to_csv(args.out_dir / "augmented_morphology_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "augmented_morphology_predictions.csv", index=False)
    baseline_train.head(0).to_csv(args.out_dir / "feature_schema.csv", index=False)
    plot_results(summary, args.fig_dir)
    write_report(summary, args, args.out_dir / "augmented_morphology_report.md")

    print(f"wrote augmented morphology outputs to {args.out_dir}")
    print(f"wrote augmented morphology figures to {args.fig_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
