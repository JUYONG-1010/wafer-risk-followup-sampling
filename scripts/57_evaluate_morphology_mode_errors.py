from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_OUT_DIR = Path("data") / "processed" / "morphology_mode_errors_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "47_morphology_mode_errors_v1"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"

DISCOVERY_PATTERNS = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Scratch"}
LOW_BIAS_PATTERNS = {"Random"}
COVERAGE_PATTERNS = {"Near-full"}
MODE_ORDER = ["discovery_first", "low_bias_default", "coverage32_or_uncertain"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", DENSITY_POLICY_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact morphology errors vs operating-mode errors."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=350)
    parser.add_argument("--max-defect-candidates", type=int, default=80)
    parser.add_argument("--max-normal-candidates", type=int, default=120)
    parser.add_argument("--point-estimators", type=int, default=40)
    parser.add_argument("--morph-estimators", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--confidence-thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.7, 0.8])
    return parser.parse_args()


def pattern_mode(label: str) -> str:
    if label in DISCOVERY_PATTERNS:
        return "discovery_first"
    if label in LOW_BIAS_PATTERNS:
        return "low_bias_default"
    if label in COVERAGE_PATTERNS:
        return "coverage32_or_uncertain"
    return "low_bias_default"


def topk_labels(classes: np.ndarray, probs: np.ndarray, k: int) -> list[str]:
    order = np.argsort(probs)[::-1][:k]
    return [str(classes[idx]) for idx in order]


def evaluate_predictions(args: argparse.Namespace) -> pd.DataFrame:
    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    densities = [float(value) for value in args.densities]
    train_wafers, test_wafers = density_policy.split_wafers(
        patterned,
        test_size=args.test_size,
        seed=args.seed,
    )
    morph_models, morph_columns, morph_lookup = density_policy.train_morph_models(
        morph_data,
        train_wafers,
        densities,
        args,
    )

    eval_df = patterned[patterned.index.isin(set(int(value) for value in test_wafers))]
    records: list[dict[str, object]] = []
    for row in eval_df.itertuples(index=True):
        row_index = int(row.Index)
        true_label = density_policy.failure_type(row)
        true_mode = pattern_mode(true_label)
        for density in densities:
            morph_row = morph_lookup[(row_index, density)]
            cols = morph_columns[density]
            exact_model = morph_models[(density, "exact")]
            x_morph = pd.DataFrame([morph_row[cols].to_dict()])
            probs = exact_model.predict_proba(x_morph)[0]
            labels_top3 = topk_labels(exact_model.classes_, probs, 3)
            pred_label = labels_top3[0]
            pred_mode = pattern_mode(pred_label)
            records.append(
                {
                    "row_index": row_index,
                    "target_density": density,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "true_mode": true_mode,
                    "pred_mode": pred_mode,
                    "top1_confidence": float(np.max(probs)),
                    "top2_label": labels_top3[1] if len(labels_top3) > 1 else pd.NA,
                    "top3_label": labels_top3[2] if len(labels_top3) > 2 else pd.NA,
                    "top1_correct": pred_label == true_label,
                    "top2_correct": true_label in labels_top3[:2],
                    "top3_correct": true_label in labels_top3[:3],
                    "mode_correct": pred_mode == true_mode,
                    "dangerous_aggressive_error": (
                        pred_mode == "discovery_first" and true_mode != "discovery_first"
                    ),
                    "missed_discovery_error": (
                        pred_mode != "discovery_first" and true_mode == "discovery_first"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("target_density", observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            exact_top1_accuracy=("top1_correct", "mean"),
            exact_top2_accuracy=("top2_correct", "mean"),
            exact_top3_accuracy=("top3_correct", "mean"),
            mode_accuracy=("mode_correct", "mean"),
            dangerous_aggressive_error_rate=("dangerous_aggressive_error", "mean"),
            missed_discovery_error_rate=("missed_discovery_error", "mean"),
            mean_top1_confidence=("top1_confidence", "mean"),
        )
        .reset_index()
    )


def summarize_by_pattern(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["target_density", "true_label"], observed=False)
        .agg(
            wafers=("row_index", "nunique"),
            exact_top1_accuracy=("top1_correct", "mean"),
            mode_accuracy=("mode_correct", "mean"),
            dangerous_aggressive_error_rate=("dangerous_aggressive_error", "mean"),
            missed_discovery_error_rate=("missed_discovery_error", "mean"),
            mean_top1_confidence=("top1_confidence", "mean"),
        )
        .reset_index()
    )


def evaluate_confidence_gating(predictions: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for density, group in predictions.groupby("target_density", observed=False):
        for threshold in thresholds:
            aggressive = (
                (group["pred_mode"] == "discovery_first")
                & (group["top1_confidence"] >= threshold)
            )
            true_discovery = group["true_mode"] == "discovery_first"
            selected_count = int(aggressive.sum())
            rows.append(
                {
                    "target_density": density,
                    "confidence_threshold": threshold,
                    "wafers": int(group["row_index"].nunique()),
                    "aggressive_selected_count": selected_count,
                    "aggressive_selected_rate": float(aggressive.mean()),
                    "aggressive_precision": (
                        float((true_discovery & aggressive).sum() / selected_count)
                        if selected_count
                        else np.nan
                    ),
                    "aggressive_recall": (
                        float((true_discovery & aggressive).sum() / true_discovery.sum())
                        if true_discovery.sum()
                        else np.nan
                    ),
                    "dangerous_aggressive_error_rate": float(
                        ((~true_discovery) & aggressive).mean()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def plot_outputs(
    summary: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    gating: pd.DataFrame,
    predictions: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    long = data.melt(
        id_vars=["density_pct"],
        value_vars=["exact_top1_accuracy", "exact_top2_accuracy", "exact_top3_accuracy", "mode_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    plt.figure(figsize=(8.8, 5.2))
    sns.lineplot(data=long, x="density_pct", y="accuracy", hue="metric", marker="o")
    plt.ylim(0, 1.02)
    plt.xlabel("Initial probe density (%)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "exact_vs_mode_accuracy.png", dpi=180)
    plt.close()

    pivot = pattern_summary.pivot_table(
        index="true_label",
        columns="target_density",
        values="mode_accuracy",
        aggfunc="mean",
    )
    pivot.columns = [f"{col:.0%}" for col in pivot.columns]
    plt.figure(figsize=(7.4, 5.2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.title("Operating-mode accuracy by true pattern")
    plt.tight_layout()
    plt.savefig(fig_dir / "pattern_mode_accuracy_heatmap.png", dpi=180)
    plt.close()

    gate = gating.copy()
    gate["density_pct"] = gate["target_density"] * 100.0
    plt.figure(figsize=(8.8, 5.2))
    sns.lineplot(
        data=gate,
        x="confidence_threshold",
        y="dangerous_aggressive_error_rate",
        hue="density_pct",
        marker="o",
        palette="viridis",
    )
    plt.xlabel("Top1 confidence threshold for discovery-first")
    plt.ylabel("Dangerous aggressive error rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "confidence_gating_dangerous_error.png", dpi=180)
    plt.close()

    labels = sorted(predictions["true_label"].unique())
    for density, group in predictions.groupby("target_density", observed=False):
        cm = confusion_matrix(group["true_label"], group["pred_label"], labels=labels)
        plt.figure(figsize=(7.4, 6.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted top1 label")
        plt.ylabel("True label")
        plt.title(f"Morphology confusion matrix: {density:.0%}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_matrix_density_{density:.2f}.png", dpi=180)
        plt.close()


def write_report(
    summary: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    gating: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Morphology Mode Error Evaluation v1",
        "",
        "Purpose: check whether exact morphology errors actually hurt operating-mode selection.",
        "",
        "Operating modes:",
        "",
        "```text",
        "discovery_first: Center, Donut, Edge-Loc, Edge-Ring, Loc, Scratch",
        "low_bias_default: Random",
        "coverage32_or_uncertain: Near-full",
        "```",
        "",
        "## Global Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}: exact top1={row.exact_top1_accuracy:.3f}, "
            f"top2={row.exact_top2_accuracy:.3f}, top3={row.exact_top3_accuracy:.3f}, "
            f"mode accuracy={row.mode_accuracy:.3f}, "
            f"dangerous aggressive error={row.dangerous_aggressive_error_rate:.3f}, "
            f"missed discovery error={row.missed_discovery_error_rate:.3f}"
        )
    lines.extend(["", "## Pattern-Level Mode Accuracy", ""])
    for row in pattern_summary.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, {row.true_label}: "
            f"mode accuracy={row.mode_accuracy:.3f}, "
            f"dangerous aggressive error={row.dangerous_aggressive_error_rate:.3f}, "
            f"missed discovery error={row.missed_discovery_error_rate:.3f}, "
            f"confidence={row.mean_top1_confidence:.3f}"
        )
    lines.extend(["", "## Confidence Gating", ""])
    for row in gating.itertuples(index=False):
        lines.append(
            f"- {row.target_density:.0%}, threshold={row.confidence_threshold:.2f}: "
            f"aggressive selected={row.aggressive_selected_rate:.3f}, "
            f"aggressive precision={row.aggressive_precision:.3f}, "
            f"aggressive recall={row.aggressive_recall:.3f}, "
            f"dangerous aggressive error={row.dangerous_aggressive_error_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If mode accuracy is much higher than exact top1 accuracy, the current classifier may be sufficient for policy selection.",
            "If dangerous aggressive errors remain high, add confidence gating or improve the classifier before recommending discovery-first mode.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    predictions = evaluate_predictions(args)
    summary = summarize(predictions)
    pattern_summary = summarize_by_pattern(predictions)
    gating = evaluate_confidence_gating(predictions, args.confidence_thresholds)

    predictions.to_csv(args.out_dir / "morphology_mode_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "morphology_mode_summary.csv", index=False)
    pattern_summary.to_csv(args.out_dir / "morphology_mode_pattern_summary.csv", index=False)
    gating.to_csv(args.out_dir / "morphology_confidence_gating.csv", index=False)
    plot_outputs(summary, pattern_summary, gating, predictions, args.fig_dir)
    write_report(summary, pattern_summary, gating, args.out_dir / "morphology_mode_errors_report.md")

    print(f"wrote morphology mode-error outputs to {args.out_dir}")
    print(f"wrote morphology mode-error figures to {args.fig_dir}")
    print(summary.round(4).to_string(index=False))
    print(gating.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
