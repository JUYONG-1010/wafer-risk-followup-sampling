from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "morphology_learning_curve_v1"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "55_morphology_learning_curve_v1"
AUG_SCRIPT = Path("scripts") / "60_evaluate_augmented_morphology.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aug = load_module("augmented_morphology60", AUG_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate morphology/mode learning curve from first-pass sparse features."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 200, 350, 700, 1200])
    parser.add_argument("--max-test-wafers", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"],
    )
    return parser.parse_args()


def split_wafers(patterned: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return aug.density_policy.split_wafers(patterned, test_size=test_size, seed=seed)


def select_train_ids(train_ids: np.ndarray, train_size: int, seed: int) -> np.ndarray:
    train_ids = np.asarray(train_ids)
    if len(train_ids) <= train_size:
        return train_ids
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(train_ids, size=train_size, replace=False))


def evaluate_learning_curve(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    patterned = pd.read_pickle(args.patterned)
    train_ids, test_ids = split_wafers(patterned, args.test_size, args.seed)
    test_ids = aug.select_ids(test_ids, args.max_test_wafers, args.seed + 10_000)
    test_data = aug.build_features(
        patterned,
        test_ids,
        [float(v) for v in args.densities],
        ["identity"],
        "test",
    )

    summaries: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for train_size in args.train_sizes:
        selected_train = select_train_ids(train_ids, int(train_size), args.seed + int(train_size))
        baseline_train = aug.build_features(
            patterned,
            selected_train,
            [float(v) for v in args.densities],
            ["identity"],
            f"train_{train_size}_baseline",
        )
        augmented_train = aug.build_features(
            patterned,
            selected_train,
            [float(v) for v in args.densities],
            list(args.transforms),
            f"train_{train_size}_augmented",
        )
        for variant_name, train_data in [
            ("baseline", baseline_train),
            ("augmented", augmented_train),
        ]:
            summary, pred = aug.evaluate_variant(
                train_data,
                test_data,
                variant_name=variant_name,
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
                seed=args.seed + int(train_size),
            )
            summary["train_wafers"] = int(len(selected_train))
            summary["requested_train_wafers"] = int(train_size)
            summary["test_wafers"] = int(len(test_ids))
            pred["train_wafers"] = int(len(selected_train))
            pred["requested_train_wafers"] = int(train_size)
            summaries.append(summary)
            predictions.append(pred)
        partial_summary = pd.concat(summaries, ignore_index=True)
        partial_predictions = pd.concat(predictions, ignore_index=True)
        partial_summary.to_csv(
            args.out_dir / "morphology_learning_curve_summary_partial.csv",
            index=False,
        )
        partial_predictions.to_csv(
            args.out_dir / "morphology_learning_curve_predictions_partial.csv",
            index=False,
        )
        print(f"learning curve train size completed: {train_size}")
    return pd.concat(summaries, ignore_index=True), pd.concat(predictions, ignore_index=True)


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "baseline"][
        [
            "requested_train_wafers",
            "target_density",
            "exact_top1_accuracy",
            "mode_accuracy",
            "dangerous_aggressive_error_rate",
            "missed_discovery_error_rate",
        ]
    ].rename(
        columns={
            "exact_top1_accuracy": "baseline_exact_top1_accuracy",
            "mode_accuracy": "baseline_mode_accuracy",
            "dangerous_aggressive_error_rate": "baseline_dangerous_aggressive_error_rate",
            "missed_discovery_error_rate": "baseline_missed_discovery_error_rate",
        }
    )
    out = summary.merge(baseline, on=["requested_train_wafers", "target_density"], how="left")
    out["exact_top1_delta_vs_baseline"] = (
        out["exact_top1_accuracy"] - out["baseline_exact_top1_accuracy"]
    )
    out["mode_accuracy_delta_vs_baseline"] = (
        out["mode_accuracy"] - out["baseline_mode_accuracy"]
    )
    out["dangerous_error_delta_vs_baseline"] = (
        out["dangerous_aggressive_error_rate"]
        - out["baseline_dangerous_aggressive_error_rate"]
    )
    out["missed_discovery_delta_vs_baseline"] = (
        out["missed_discovery_error_rate"]
        - out["baseline_missed_discovery_error_rate"]
    )
    return out


def plot_outputs(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = summary.copy()
    data["density_pct"] = data["target_density"] * 100.0
    for metric, filename, ylabel in [
        ("exact_top1_accuracy", "learning_curve_exact_top1.png", "Exact morphology top1 accuracy"),
        ("mode_accuracy", "learning_curve_mode_accuracy.png", "Operating-mode accuracy"),
        (
            "dangerous_aggressive_error_rate",
            "learning_curve_dangerous_error.png",
            "Dangerous aggressive error rate",
        ),
        (
            "missed_discovery_error_rate",
            "learning_curve_missed_discovery.png",
            "Missed discovery error rate",
        ),
    ]:
        plt.figure(figsize=(9.2, 5.4))
        sns.lineplot(
            data=data,
            x="requested_train_wafers",
            y=metric,
            hue="density_pct",
            style="variant",
            marker="o",
            palette="viridis",
        )
        plt.xlabel("Training wafers")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    aug_data = data[data["variant"] == "augmented"].copy()
    plt.figure(figsize=(8.8, 5.2))
    sns.lineplot(
        data=aug_data,
        x="requested_train_wafers",
        y="mode_accuracy_delta_vs_baseline",
        hue="density_pct",
        marker="o",
        palette="viridis",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Training wafers")
    plt.ylabel("Augmented mode accuracy delta vs baseline")
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curve_augmentation_mode_delta.png", dpi=180)
    plt.close()


def slope_view(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, density), group in summary.groupby(["variant", "target_density"], observed=False):
        group = group.sort_values("requested_train_wafers")
        first = group.iloc[0]
        last = group.iloc[-1]
        prev = group.iloc[-2] if len(group) > 1 else first
        rows.append(
            {
                "variant": variant,
                "target_density": density,
                "first_train_wafers": int(first.requested_train_wafers),
                "last_train_wafers": int(last.requested_train_wafers),
                "exact_top1_first": float(first.exact_top1_accuracy),
                "exact_top1_last": float(last.exact_top1_accuracy),
                "exact_top1_total_gain": float(last.exact_top1_accuracy - first.exact_top1_accuracy),
                "exact_top1_last_step_gain": float(last.exact_top1_accuracy - prev.exact_top1_accuracy),
                "mode_first": float(first.mode_accuracy),
                "mode_last": float(last.mode_accuracy),
                "mode_total_gain": float(last.mode_accuracy - first.mode_accuracy),
                "mode_last_step_gain": float(last.mode_accuracy - prev.mode_accuracy),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_report(summary: pd.DataFrame, slopes: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        "# Morphology Learning Curve v1",
        "",
        "Purpose: check whether first-pass morphology/mode models are undertrained or near plateau.",
        "",
        "Important: this audit evaluates morphology/mode prediction, not the full point-risk ranking model.",
        "",
        "## Largest Train Size Result",
        "",
    ]
    max_train = int(summary["requested_train_wafers"].max())
    focus = summary[summary["requested_train_wafers"] == max_train].copy()
    for row in focus.sort_values(["target_density", "variant"]).itertuples(index=False):
        lines.append(
            f"- {row.variant}, {row.target_density:.0%}: "
            f"top1 {row.exact_top1_accuracy:.3f}, "
            f"mode {row.mode_accuracy:.3f}, "
            f"dangerous {row.dangerous_aggressive_error_rate:.3f}, "
            f"missed discovery {row.missed_discovery_error_rate:.3f}"
        )
    lines.extend(["", "## Learning Slope", ""])
    for row in slopes.sort_values(["target_density", "variant"]).itertuples(index=False):
        lines.append(
            f"- {row.variant}, {row.target_density:.0%}: "
            f"top1 total gain {row.exact_top1_total_gain:+.3f}, "
            f"last-step gain {row.exact_top1_last_step_gain:+.3f}; "
            f"mode total gain {row.mode_total_gain:+.3f}, "
            f"last-step gain {row.mode_last_step_gain:+.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "```text",
            "large positive last-step gain -> model likely still benefits from more data/model work",
            "near-zero or unstable last-step gain -> morphology model is closer to plateau under current features",
            "```",
            "",
            "Next audit step: repeated split stability for the follow-up policy metrics.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary, predictions = evaluate_learning_curve(args)
    summary = add_deltas(summary)
    slopes = slope_view(summary)

    summary.to_csv(args.out_dir / "morphology_learning_curve_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "morphology_learning_curve_predictions.csv", index=False)
    slopes.to_csv(args.out_dir / "morphology_learning_curve_slopes.csv", index=False)
    plot_outputs(summary, args.fig_dir)
    write_report(summary, slopes, args.out_dir / "morphology_learning_curve_report.md")

    print(f"wrote morphology learning curve outputs to {args.out_dir}")
    print(f"wrote morphology learning curve figures to {args.fig_dir}")
    print(
        summary[
            summary["requested_train_wafers"] == summary["requested_train_wafers"].max()
        ][
            [
                "variant",
                "target_density",
                "exact_top1_accuracy",
                "mode_accuracy",
                "dangerous_aggressive_error_rate",
                "missed_discovery_error_rate",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(slopes.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
