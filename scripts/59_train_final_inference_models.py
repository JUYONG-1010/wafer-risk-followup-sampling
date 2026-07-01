from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_ranking import FEATURE_COLUMNS


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_MORPH_DATASET = (
    Path("data")
    / "processed"
    / "initial_probe_density_v1"
    / "initial_probe_density_dataset.csv"
)
DEFAULT_MODEL_DIR = Path("models") / "final_inference_v1"
DENSITY_POLICY_SCRIPT = Path("scripts") / "47_evaluate_density_followup_policy.py"


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
        description="Train final inference models on all labeled patterned wafers."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--morph-dataset", type=Path, default=DEFAULT_MORPH_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-wafers", type=int, default=0)
    parser.add_argument("--max-defect-candidates", type=int, default=6)
    parser.add_argument("--max-normal-candidates", type=int, default=14)
    parser.add_argument("--point-estimators", type=int, default=60)
    parser.add_argument("--morph-estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--default-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--first-ratio-weight", type=float, default=0.25)
    return parser.parse_args()


def class_counts(patterned: pd.DataFrame) -> pd.DataFrame:
    counts = patterned["failureType_clean"].astype(str).value_counts().rename_axis("failureType")
    return counts.reset_index(name="wafers")


def save_morph_models(
    model_dir: Path,
    morph_models,
    morph_columns,
    densities: list[float],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    morph_dir = model_dir / "morphology"
    morph_dir.mkdir(parents=True, exist_ok=True)
    for density in densities:
        for target_name in ["exact", "group"]:
            model = morph_models[(density, target_name)]
            filename = f"morph_{target_name}_{density_policy.density_key(density)}.joblib"
            path = morph_dir / filename
            joblib.dump(model, path)
            records.append(
                {
                    "density": density,
                    "target": target_name,
                    "path": str(path),
                    "classes": [str(value) for value in model.classes_],
                    "feature_count": len(morph_columns[density]),
                }
            )
    (morph_dir / "morph_feature_columns.json").write_text(
        json.dumps({str(k): v for k, v in morph_columns.items()}, indent=2),
        encoding="utf-8",
    )
    return records


def write_report(
    args: argparse.Namespace,
    model_dir: Path,
    patterned: pd.DataFrame,
    point_train: pd.DataFrame,
    point_metrics: dict[str, float],
    global_target_ratio: float,
    class_table: pd.DataFrame,
    morph_records: list[dict[str, object]],
) -> None:
    lines = [
        "# Final Inference Model Training v1",
        "",
        "Purpose: train final inference/demo models after policy evaluation was locked.",
        "",
        "Important:",
        "",
        "```text",
        "These all-data models are for final inference/demo.",
        "Do not use them to report held-out performance metrics.",
        "Held-out metrics must come from the earlier train/test evaluations.",
        "```",
        "",
        "## Training Data",
        "",
        f"- patterned wafers used: {len(patterned)}",
        f"- densities: {', '.join(f'{d:.0%}' for d in args.densities)}",
        f"- point candidate rows: {len(point_train)}",
        f"- point candidate sampling per wafer/density: max defect {args.max_defect_candidates}, max normal {args.max_normal_candidates}",
        f"- global target defect ratio: {global_target_ratio:.6f}",
        "",
        "## Class Counts",
        "",
    ]
    for row in class_table.itertuples(index=False):
        lines.append(f"- {row.failureType}: {row.wafers}")
    lines.extend(
        [
            "",
            "## Point Model",
            "",
            f"- estimators: {args.point_estimators}",
            f"- train ROC-AUC: {point_metrics['train_point_roc_auc']:.4f}",
            f"- train average precision: {point_metrics['train_point_average_precision']:.4f}",
            f"- feature count: {len(FEATURE_COLUMNS)}",
            f"- artifact: `{model_dir / 'point_model.joblib'}`",
            "",
            "## Morphology Models",
            "",
        ]
    )
    for record in morph_records:
        lines.append(
            f"- {record['density']:.0%} {record['target']}: "
            f"{len(record['classes'])} classes, artifact `{record['path']}`"
        )
    lines.extend(
        [
            "",
            "## Default Policy Settings",
            "",
            f"- default confidence threshold: {args.default_confidence_threshold:.2f}",
            f"- follow-up budget top-k: {args.top_k}",
            "",
        ]
    )
    (model_dir / "final_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    densities = [float(value) for value in args.densities]

    patterned = pd.read_pickle(args.patterned)
    morph_data = pd.read_csv(args.morph_dataset)
    train_wafers = patterned.index.to_numpy()
    if args.max_train_wafers and len(train_wafers) > args.max_train_wafers:
        rng = np.random.default_rng(args.seed)
        train_wafers = rng.choice(train_wafers, size=args.max_train_wafers, replace=False)
        train_wafers = np.asarray(train_wafers)

    point_train = density_policy.build_point_training_data(patterned, train_wafers, densities, args)
    point_model = density_policy.train_point_model(point_train, args)
    point_metrics = density_policy.point_model_metrics(point_model, point_train)
    morph_models, morph_columns, _ = density_policy.train_morph_models(
        morph_data,
        train_wafers,
        densities,
        args,
    )
    global_target_ratio = density_policy.mean_actual_defect_ratio(patterned, train_wafers)

    point_model_path = args.model_dir / "point_model.joblib"
    joblib.dump(point_model, point_model_path)
    (args.model_dir / "point_feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2),
        encoding="utf-8",
    )

    morph_records = save_morph_models(args.model_dir, morph_models, morph_columns, densities)
    class_table = class_counts(patterned[patterned.index.isin(set(int(v) for v in train_wafers))])
    class_table.to_csv(args.model_dir / "training_class_counts.csv", index=False)
    point_train_summary = pd.DataFrame(
        [
            {
                "point_train_rows": len(point_train),
                "positive_rows": int(point_train["label_candidate_is_defect"].sum()),
                "negative_rows": int((point_train["label_candidate_is_defect"] == 0).sum()),
                "positive_rate": float(point_train["label_candidate_is_defect"].mean()),
            }
        ]
    )
    point_train_summary.to_csv(args.model_dir / "point_training_summary.csv", index=False)

    manifest = {
        "model_version": "final_inference_v1",
        "purpose": "final inference/demo model trained after split-based evaluation",
        "patterned_path": str(args.patterned),
        "morph_dataset_path": str(args.morph_dataset),
        "patterned_wafers_total": int(len(patterned)),
        "train_wafers_used": int(len(train_wafers)),
        "densities": densities,
        "point_model_path": str(point_model_path),
        "morphology_models": morph_records,
        "point_feature_columns_path": str(args.model_dir / "point_feature_columns.json"),
        "morph_feature_columns_path": str(args.model_dir / "morphology" / "morph_feature_columns.json"),
        "max_defect_candidates": args.max_defect_candidates,
        "max_normal_candidates": args.max_normal_candidates,
        "point_estimators": args.point_estimators,
        "morph_estimators": args.morph_estimators,
        "default_confidence_threshold": args.default_confidence_threshold,
        "top_k": args.top_k,
        "global_target_ratio": global_target_ratio,
        "point_train_metrics": point_metrics,
        "note": "Use earlier train/test results for performance claims; this all-data model is for final inference/demo.",
    }
    (args.model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_report(
        args,
        args.model_dir,
        patterned[patterned.index.isin(set(int(v) for v in train_wafers))],
        point_train,
        point_metrics,
        global_target_ratio,
        class_table,
        morph_records,
    )

    print(f"wrote final inference models to {args.model_dir}")
    print(class_table.to_string(index=False))
    print(point_train_summary.round(4).to_string(index=False))
    print(pd.DataFrame([point_metrics]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
