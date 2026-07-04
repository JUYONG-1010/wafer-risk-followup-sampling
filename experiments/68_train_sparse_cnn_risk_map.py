from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import defect_mask, make_coverage_sampling_mask, sampling_metrics, valid_die_mask, wafer_center


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "sparse_cnn_risk_map_v1_smoke"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "58_sparse_cnn_risk_map_v1_smoke"
DEFAULT_MODEL_PATH = Path("models") / "sparse_cnn_risk_map_v1_smoke.pt"


def load_density_policy_module():
    module_path = PROJECT_ROOT / "experiments" / "47_evaluate_density_followup_policy.py"
    spec = importlib.util.spec_from_file_location("density_policy47", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density_policy = load_density_policy_module()


@dataclass(frozen=True)
class SampleKey:
    row_index: int
    target_density: float


class SparseRiskCNN(nn.Module):
    def __init__(self, in_channels: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe sparse-input CNN risk map prototype."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.03, 0.10])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-train-wafers", type=int, default=300)
    parser.add_argument("--max-test-wafers", type=int, default=120)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def limit_ids(
    patterned: pd.DataFrame,
    ids: np.ndarray,
    limit: int,
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=int)
    if not limit or len(ids) <= limit:
        return ids
    labels = patterned.loc[ids].apply(density_policy.failure_type, axis=1).astype(str)
    counts = labels.value_counts()
    stratify = labels if counts.min() >= 2 else None
    if stratify is None:
        ids = rng.choice(ids, size=limit, replace=False)
    else:
        ids, _ = train_test_split(
            ids,
            train_size=limit,
            random_state=seed,
            stratify=stratify,
        )
    return ids


def validation_split(
    patterned: pd.DataFrame,
    train_ids: np.ndarray,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_ids = np.asarray(train_ids, dtype=int)
    if len(train_ids) < 10 or val_size <= 0:
        return train_ids, np.asarray([], dtype=int)
    labels = patterned.loc[train_ids].apply(density_policy.failure_type, axis=1).astype(str)
    counts = labels.value_counts()
    stratify = labels if counts.min() >= 2 else None
    core_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(core_ids, dtype=int), np.asarray(val_ids, dtype=int)


def make_sample_keys(ids: np.ndarray, densities: list[float]) -> list[SampleKey]:
    return [SampleKey(int(row_index), float(density)) for row_index in ids for density in densities]


def coordinate_channels(wafer_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    yy, xx = np.indices(valid.shape)
    cy, cx = wafer_center(valid)
    y_scale = max(float(valid.shape[0] - 1), 1.0)
    x_scale = max(float(valid.shape[1] - 1), 1.0)
    y_norm = ((yy - cy) / y_scale).astype(np.float32)
    x_norm = ((xx - cx) / x_scale).astype(np.float32)
    radius = np.sqrt(y_norm**2 + x_norm**2).astype(np.float32)
    if valid.any():
        max_radius = float(radius[valid].max())
        if max_radius > 0:
            radius = radius / max_radius
    return y_norm, x_norm, radius


def make_tensors(
    wafer_map: np.ndarray,
    density: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    valid = valid_die_mask(wafer_map)
    defects = defect_mask(wafer_map)
    first = density_policy.make_initial_coverage_mask(wafer_map, density)
    first_valid = first & valid
    first_defect = first & defects
    first_normal = first_valid & ~defects
    unknown_valid = valid & ~first_valid
    y_norm, x_norm, radius = coordinate_channels(wafer_map)
    channels = np.stack(
        [
            valid.astype(np.float32),
            first_normal.astype(np.float32),
            first_defect.astype(np.float32),
            unknown_valid.astype(np.float32),
            y_norm.astype(np.float32),
            x_norm.astype(np.float32),
            radius.astype(np.float32),
        ],
        axis=0,
    )
    target = defects.astype(np.float32)
    candidate_mask = unknown_valid.astype(np.float32)
    x = torch.from_numpy(channels).unsqueeze(0).to(device)
    y = torch.from_numpy(target).unsqueeze(0).to(device)
    mask = torch.from_numpy(candidate_mask).unsqueeze(0).to(device)
    return x, y, mask, first_valid


def estimate_pos_weight(patterned: pd.DataFrame, keys: list[SampleKey]) -> float:
    positives = 0
    negatives = 0
    for key in keys:
        wafer_map = np.asarray(patterned.at[key.row_index, "waferMap"])
        valid = valid_die_mask(wafer_map)
        defects = defect_mask(wafer_map)
        first = density_policy.make_initial_coverage_mask(wafer_map, key.target_density)
        candidates = valid & ~first
        positives += int((candidates & defects).sum())
        negatives += int((candidates & ~defects).sum())
    if positives == 0:
        return 1.0
    return float(min(max(negatives / positives, 1.0), 30.0))


def train_model(
    model: nn.Module,
    patterned: pd.DataFrame,
    train_keys: list[SampleKey],
    val_keys: list[SampleKey],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor], int]:
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = torch.tensor(estimate_pos_weight(patterned, train_keys), dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_keys))
        losses: list[float] = []
        for position in order:
            key = train_keys[int(position)]
            wafer_map = np.asarray(patterned.at[key.row_index, "waferMap"])
            x, y, mask, _ = make_tensors(wafer_map, key.target_density, device)
            logits = model(x)
            loss_map = F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=pos_weight,
                reduction="none",
            )
            denom = mask.sum().clamp_min(1.0)
            loss = (loss_map * mask).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        record = {"epoch": epoch, "train_loss": epoch_loss, "pos_weight": float(pos_weight.cpu())}
        monitor_score = -epoch_loss
        if val_keys:
            _, val_summary = evaluate_model(model, patterned, val_keys, args, device)
            val_ap = float(val_summary["mean_average_precision"].mean())
            val_auc = float(val_summary["mean_roc_auc"].mean())
            val_gain = float(val_summary["mean_top32_coverage_gain_pct"].mean())
            record.update(
                {
                    "val_mean_average_precision": val_ap,
                    "val_mean_roc_auc": val_auc,
                    "val_mean_top32_coverage_gain_pct": val_gain,
                }
            )
            monitor_score = val_ap
        improved = monitor_score > best_score + args.min_delta
        if improved:
            best_score = monitor_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        record["best_epoch"] = best_epoch
        record["early_stop_counter"] = stale_epochs
        history.append(record)
        val_text = ""
        if val_keys:
            val_text = (
                f" val_ap={record['val_mean_average_precision']:.5f}"
                f" val_auc={record['val_mean_roc_auc']:.5f}"
                f" val_gain={record['val_mean_top32_coverage_gain_pct']:.2f}%"
            )
        print(
            f"epoch {epoch}/{args.epochs} train_loss={epoch_loss:.5f}"
            f" pos_weight={float(pos_weight.cpu()):.2f}{val_text}"
            f" best_epoch={best_epoch}"
        )
        model.train()
        if val_keys and args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break
    return pd.DataFrame(history), best_state, best_epoch


def topk_mask_from_scores(scores: np.ndarray, candidate_mask: np.ndarray, top_k: int) -> np.ndarray:
    selected = np.zeros_like(candidate_mask, dtype=bool)
    coords = np.column_stack(np.nonzero(candidate_mask))
    if len(coords) == 0:
        return selected
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    values = scores[yy, xx]
    k = min(int(top_k), len(values))
    if k <= 0:
        return selected
    order = np.argsort(values)[::-1][:k]
    chosen = coords[order]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def top_fraction_iou(scores: np.ndarray, labels: np.ndarray, candidate_mask: np.ndarray, fraction: float) -> float:
    coords = np.column_stack(np.nonzero(candidate_mask))
    if len(coords) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(coords) * fraction)))
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    values = scores[yy, xx]
    selected = np.zeros_like(candidate_mask, dtype=bool)
    chosen = coords[np.argsort(values)[::-1][:k]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    truth = labels.astype(bool) & candidate_mask
    union = selected | truth
    if not union.any():
        return float("nan")
    return float((selected & truth).sum() / union.sum())


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def evaluate_model(
    model: nn.Module,
    patterned: pd.DataFrame,
    test_keys: list[SampleKey],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    records: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for key in test_keys:
            wafer_map = np.asarray(patterned.at[key.row_index, "waferMap"])
            x, _, _, first_mask = make_tensors(wafer_map, key.target_density, device)
            logits = model(x).squeeze(0).cpu().numpy()
            scores = 1.0 / (1.0 + np.exp(-logits))
            valid = valid_die_mask(wafer_map)
            defects = defect_mask(wafer_map)
            candidates = valid & ~first_mask
            yy, xx = np.nonzero(candidates)
            if len(yy) == 0:
                continue
            y_true = defects[yy, xx].astype(int)
            y_score = scores[yy, xx].astype(float)
            selected = topk_mask_from_scores(scores, candidates, args.top_k)
            coverage_followup = make_coverage_sampling_mask(
                wafer_map,
                n_points=args.top_k,
                existing_mask=first_mask,
            )
            cnn_metrics = sampling_metrics(wafer_map, first_mask | selected)
            coverage_metrics = sampling_metrics(wafer_map, first_mask | coverage_followup)
            cnn_coverage = float(cnn_metrics["defect_coverage"])
            coverage32_coverage = float(coverage_metrics["defect_coverage"])
            if coverage32_coverage > 0:
                row_gain_pct = 100.0 * (cnn_coverage - coverage32_coverage) / coverage32_coverage
            else:
                row_gain_pct = float("nan")
            records.append(
                {
                    "row_index": int(key.row_index),
                    "failureType": density_policy.failure_type(patterned.loc[key.row_index]),
                    "target_density": float(key.target_density),
                    "candidate_count": int(candidates.sum()),
                    "candidate_defects": int((candidates & defects).sum()),
                    "roc_auc": safe_auc(y_true, y_score),
                    "average_precision": safe_average_precision(y_true, y_score),
                    "top10pct_iou": top_fraction_iou(scores, defects, candidates, 0.10),
                    "top32_defect_coverage": cnn_coverage,
                    "coverage32_defect_coverage": coverage32_coverage,
                    "top32_coverage_gain_pct": row_gain_pct,
                    "top32_sampled_defects": int(cnn_metrics["sampled_defects"]),
                    "coverage32_sampled_defects": int(coverage_metrics["sampled_defects"]),
                    "absolute_error": float(cnn_metrics["absolute_error"]),
                    "coverage32_absolute_error": float(coverage_metrics["absolute_error"]),
                    "absolute_error_delta": float(cnn_metrics["absolute_error"])
                    - float(coverage_metrics["absolute_error"]),
                    "severe_miss": int(cnn_metrics["severe_miss"]),
                    "coverage32_severe_miss": int(coverage_metrics["severe_miss"]),
                }
            )
    rows = pd.DataFrame.from_records(records)
    summary = (
        rows.groupby("target_density", dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top10pct_iou=("top10pct_iou", "mean"),
            mean_top32_defect_coverage=("top32_defect_coverage", "mean"),
            mean_coverage32_defect_coverage=("coverage32_defect_coverage", "mean"),
            mean_top32_coverage_gain_pct=("top32_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            coverage32_severe_miss_rate=("coverage32_severe_miss", "mean"),
        )
        .reset_index()
    )
    summary["mean_top32_coverage_gain_pct"] = (
        100.0
        * (
            summary["mean_top32_defect_coverage"]
            - summary["mean_coverage32_defect_coverage"]
        )
        / summary["mean_coverage32_defect_coverage"].replace(0.0, np.nan)
    )
    return rows, summary


def pattern_summary(eval_rows: pd.DataFrame) -> pd.DataFrame:
    if eval_rows.empty:
        return pd.DataFrame()
    return (
        eval_rows.groupby(["target_density", "failureType"], dropna=False)
        .agg(
            wafers=("row_index", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_top10pct_iou=("top10pct_iou", "mean"),
            mean_top32_defect_coverage=("top32_defect_coverage", "mean"),
            mean_coverage32_defect_coverage=("coverage32_defect_coverage", "mean"),
            mean_top32_coverage_gain_pct=("top32_coverage_gain_pct", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            severe_miss_rate=("severe_miss", "mean"),
            candidate_defects=("candidate_defects", "mean"),
        )
        .reset_index()
        .sort_values(["target_density", "failureType"])
    )


def plot_training_history(history: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked BCE loss")
    ax.set_title("Sparse CNN training loss")
    ax.grid(alpha=0.25)
    fig.savefig(fig_dir / "sparse_cnn_training_loss.png", dpi=180)
    plt.close(fig)

    if "val_mean_average_precision" in history.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        ax.plot(history["epoch"], history["val_mean_average_precision"], marker="o", label="val AP")
        ax.plot(history["epoch"], history["val_mean_roc_auc"], marker="o", label="val ROC-AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation metric")
        ax.set_title("Sparse CNN validation metrics")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.savefig(fig_dir / "sparse_cnn_validation_metrics.png", dpi=180)
        plt.close(fig)


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    x = summary["target_density"].to_numpy(dtype=float)
    axes[0].plot(x, summary["mean_roc_auc"], marker="o")
    axes[0].set_title("ROC-AUC")
    axes[1].plot(x, summary["mean_average_precision"], marker="o")
    axes[1].set_title("Average Precision")
    axes[2].plot(x, summary["mean_top32_coverage_gain_pct"], marker="o")
    axes[2].axhline(0.0, color="#666666", linewidth=0.8)
    axes[2].set_title("Top32 gain vs coverage32")
    for ax in axes:
        ax.set_xlabel("Initial probe density")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:.0%}" for value in x])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[2].set_ylabel("Gain (%)")
    fig.savefig(fig_dir / "sparse_cnn_summary_metrics.png", dpi=180)
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    train_count: int,
    val_count: int,
    test_count: int,
    best_epoch: int,
    device: torch.device,
) -> None:
    lines = [
        "# Sparse CNN Risk Map Prototype",
        "",
        "This is a CNN branch smoke/prototype run. It is not yet accepted as a replacement for the frozen non-CNN deliverable pack.",
        "",
        "## Input Contract",
        "",
        "Allowed CNN input channels:",
        "",
        "- valid die mask",
        "- first-pass observed normal dies",
        "- first-pass observed defective dies",
        "- unmeasured valid die mask",
        "- normalized y coordinate",
        "- normalized x coordinate",
        "- normalized radial coordinate",
        "",
        "Forbidden inference-time inputs:",
        "",
        "- dense hidden defect labels",
        "- true failureType",
        "- actual defect ratio",
        "- total defect count",
        "- hidden defect coordinates outside the first-pass observation",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(
            {
                "torch_version": torch.__version__,
                "device": str(device),
                "epochs": args.epochs,
                "densities": args.densities,
                "max_train_wafers": args.max_train_wafers,
                "max_test_wafers": args.max_test_wafers,
                "train_sample_count": train_count,
                "validation_sample_count": val_count,
                "test_sample_count": test_count,
                "best_epoch": best_epoch,
                "patience": args.patience,
                "top_k": args.top_k,
                "hidden_channels": args.hidden_channels,
            },
            indent=2,
        ),
        "```",
        "",
        "## Training History",
        "",
        history.round(5).to_string(index=False),
        "",
        "## Evaluation Summary",
        "",
        summary.round(5).to_string(index=False),
        "",
        "## Acceptance Rule",
        "",
        "This CNN should only be adopted if a full run beats the frozen non-CNN pack under the same sparse-input leakage rules.",
        "",
        "Pattern-wise metrics are written to `sparse_cnn_eval_pattern_summary.csv` and should be checked before making any claim about minority patterns.",
        "",
    ]
    (args.out_dir / "sparse_cnn_risk_map_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")

    rng = np.random.default_rng(args.seed)
    patterned = pd.read_pickle(args.patterned)
    train_ids, test_ids = density_policy.split_wafers(patterned, test_size=args.test_size, seed=args.seed)
    train_ids = limit_ids(patterned, train_ids, args.max_train_wafers, rng, args.seed)
    test_ids = limit_ids(patterned, test_ids, args.max_test_wafers, rng, args.seed + 1)
    train_ids, val_ids = validation_split(patterned, train_ids, args.val_size, args.seed)
    densities = [float(value) for value in args.densities]
    train_keys = make_sample_keys(train_ids, densities)
    val_keys = make_sample_keys(val_ids, densities)
    test_keys = make_sample_keys(test_ids, densities)

    model = SparseRiskCNN(in_channels=7, hidden=args.hidden_channels).to(device)
    history, best_state, best_epoch = train_model(model, patterned, train_keys, val_keys, args, device)
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    eval_rows, summary = evaluate_model(model, patterned, test_keys, args, device)
    eval_pattern = pattern_summary(eval_rows)

    history.to_csv(args.out_dir / "sparse_cnn_training_history.csv", index=False)
    eval_rows.to_csv(args.out_dir / "sparse_cnn_eval_rows.csv", index=False)
    summary.to_csv(args.out_dir / "sparse_cnn_eval_summary.csv", index=False)
    eval_pattern.to_csv(args.out_dir / "sparse_cnn_eval_pattern_summary.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "torch_version": torch.__version__,
            "best_epoch": best_epoch,
            "input_channels": [
                "valid_die_mask",
                "first_pass_normal",
                "first_pass_defect",
                "unknown_valid_mask",
                "normalized_y",
                "normalized_x",
                "normalized_radius",
            ],
        },
        args.model_path,
    )
    plot_training_history(history, args.fig_dir)
    plot_summary(summary, args.fig_dir)
    write_report(args, history, summary, len(train_keys), len(val_keys), len(test_keys), best_epoch, device)

    print(f"wrote sparse CNN tables to {args.out_dir}")
    print(f"wrote sparse CNN figures to {args.fig_dir}")
    print(f"wrote sparse CNN model to {args.model_path}")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
