from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampling import defect_mask, make_coverage_sampling_mask, sampling_metrics, valid_die_mask


DEFAULT_PATTERNED = Path("data") / "processed" / "subsets" / "patterned_subset.pkl"
DEFAULT_OUT_DIR = Path("data") / "processed" / "sparse_cnn_risk_map_batched_v1_smoke"
DEFAULT_FIG_DIR = Path("outputs") / "figures" / "71_sparse_cnn_risk_map_batched_v1_smoke"
DEFAULT_MODEL_PATH = Path("models") / "sparse_cnn_risk_map_batched_v1_smoke.pt"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


density_policy = load_module("density_policy47", Path("scripts") / "47_evaluate_density_followup_policy.py")
sparse_cnn = load_module("sparse_cnn68", Path("scripts") / "68_train_sparse_cnn_risk_map.py")


@dataclass(frozen=True)
class CachedWafer:
    wafer_map: np.ndarray
    valid: np.ndarray
    defects: np.ndarray
    y_norm: np.ndarray
    x_norm: np.ndarray
    radius: np.ndarray
    failure_type: str


@dataclass(frozen=True)
class CachedSample:
    row_index: int
    target_density: float
    first_mask: np.ndarray
    first_valid: np.ndarray
    first_defect: np.ndarray
    first_normal: np.ndarray
    unknown_valid: np.ndarray


class SparseRiskCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32) -> None:
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
        description="Train sparse-input CNN with cached masks and batched padded tensors."
    )
    parser.add_argument("--patterned", type=Path, default=DEFAULT_PATTERNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resume", type=Path, default=None)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--drop-coordinate-channels", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def build_wafer_cache(patterned: pd.DataFrame, row_ids: np.ndarray) -> dict[int, CachedWafer]:
    cache: dict[int, CachedWafer] = {}
    for row_index in sorted(set(int(value) for value in row_ids)):
        row = patterned.loc[row_index]
        wafer_map = np.asarray(row["waferMap"])
        valid = valid_die_mask(wafer_map)
        y_norm, x_norm, radius = sparse_cnn.coordinate_channels(wafer_map)
        cache[row_index] = CachedWafer(
            wafer_map=wafer_map,
            valid=valid,
            defects=defect_mask(wafer_map),
            y_norm=y_norm,
            x_norm=x_norm,
            radius=radius,
            failure_type=density_policy.failure_type(row),
        )
    return cache


def build_sample_cache(
    wafer_cache: dict[int, CachedWafer],
    row_ids: np.ndarray,
    densities: list[float],
) -> list[CachedSample]:
    samples: list[CachedSample] = []
    for row_index in row_ids:
        cached = wafer_cache[int(row_index)]
        for density in densities:
            first = density_policy.make_initial_coverage_mask(cached.wafer_map, float(density))
            first_valid = first & cached.valid
            first_defect = first & cached.defects
            first_normal = first_valid & ~cached.defects
            unknown_valid = cached.valid & ~first_valid
            samples.append(
                CachedSample(
                    row_index=int(row_index),
                    target_density=float(density),
                    first_mask=first,
                    first_valid=first_valid,
                    first_defect=first_defect,
                    first_normal=first_normal,
                    unknown_valid=unknown_valid,
                )
            )
    return samples


class CachedSparseDataset(Dataset):
    def __init__(
        self,
        wafer_cache: dict[int, CachedWafer],
        samples: list[CachedSample],
        include_coordinates: bool,
    ) -> None:
        self.wafer_cache = wafer_cache
        self.samples = samples
        self.include_coordinates = include_coordinates

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        wafer = self.wafer_cache[sample.row_index]
        channels = [
            wafer.valid.astype(np.float32),
            sample.first_normal.astype(np.float32),
            sample.first_defect.astype(np.float32),
            sample.unknown_valid.astype(np.float32),
        ]
        if self.include_coordinates:
            channels.extend(
                [
                    wafer.y_norm.astype(np.float32),
                    wafer.x_norm.astype(np.float32),
                    wafer.radius.astype(np.float32),
                ]
            )
        x = np.stack(channels, axis=0)
        return {
            "x": x,
            "y": wafer.defects.astype(np.float32),
            "mask": sample.unknown_valid.astype(np.float32),
            "first_mask": sample.first_mask.astype(bool),
            "row_index": sample.row_index,
            "target_density": sample.target_density,
        }


def pad_2d(arr: np.ndarray, height: int, width: int, fill_value: float | bool = 0) -> np.ndarray:
    out = np.full((height, width), fill_value, dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def collate_padded(batch: list[dict[str, object]]) -> dict[str, object]:
    height = max(item["x"].shape[1] for item in batch)  # type: ignore[index]
    width = max(item["x"].shape[2] for item in batch)  # type: ignore[index]
    channels = batch[0]["x"].shape[0]  # type: ignore[index]
    xs = np.zeros((len(batch), channels, height, width), dtype=np.float32)
    ys = np.zeros((len(batch), height, width), dtype=np.float32)
    masks = np.zeros((len(batch), height, width), dtype=np.float32)
    first_masks = np.zeros((len(batch), height, width), dtype=bool)
    shapes: list[tuple[int, int]] = []
    row_indices: list[int] = []
    densities: list[float] = []
    for i, item in enumerate(batch):
        x = item["x"]  # type: ignore[assignment]
        y = item["y"]  # type: ignore[assignment]
        mask = item["mask"]  # type: ignore[assignment]
        first = item["first_mask"]  # type: ignore[assignment]
        h, w = y.shape
        xs[i, :, :h, :w] = x
        ys[i, :h, :w] = y
        masks[i, :h, :w] = mask
        first_masks[i, :h, :w] = first
        shapes.append((h, w))
        row_indices.append(int(item["row_index"]))
        densities.append(float(item["target_density"]))
    return {
        "x": torch.from_numpy(xs),
        "y": torch.from_numpy(ys),
        "mask": torch.from_numpy(masks),
        "first_mask": first_masks,
        "shapes": shapes,
        "row_index": row_indices,
        "target_density": densities,
    }


def estimate_pos_weight_from_cache(
    wafer_cache: dict[int, CachedWafer],
    samples: list[CachedSample],
) -> float:
    positives = 0
    negatives = 0
    for sample in samples:
        defects = wafer_cache[sample.row_index].defects
        candidates = sample.unknown_valid
        positives += int((candidates & defects).sum())
        negatives += int((candidates & ~defects).sum())
    if positives == 0:
        return 1.0
    return float(min(max(negatives / positives, 1.0), 30.0))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def topk_mask_from_scores(scores: np.ndarray, candidate_mask: np.ndarray, top_k: int) -> np.ndarray:
    selected = np.zeros_like(candidate_mask, dtype=bool)
    coords = np.column_stack(np.nonzero(candidate_mask))
    if len(coords) == 0:
        return selected
    yy = coords[:, 0].astype(int)
    xx = coords[:, 1].astype(int)
    values = scores[yy, xx]
    chosen = coords[np.argsort(values)[::-1][: min(top_k, len(values))]]
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


def make_loader(
    wafer_cache: dict[int, CachedWafer],
    samples: list[CachedSample],
    include_coordinates: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CachedSparseDataset(wafer_cache, samples, include_coordinates=include_coordinates)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_padded,
    )


def train_model(
    model: nn.Module,
    wafer_cache: dict[int, CachedWafer],
    train_samples: list[CachedSample],
    val_samples: list[CachedSample],
    args: argparse.Namespace,
    device: torch.device,
    include_coordinates: bool,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = torch.tensor(
        estimate_pos_weight_from_cache(wafer_cache, train_samples),
        dtype=torch.float32,
        device=device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")
    start_epoch = 1
    best_score = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history_records: list[dict[str, float]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history_records = list(checkpoint.get("history", []))
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        stale_epochs = int(checkpoint.get("stale_epochs", stale_epochs))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint.get("best_state_dict", model.state_dict()).items()
        }

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loader = make_loader(
            wafer_cache,
            train_samples,
            include_coordinates,
            args.batch_size,
            args.num_workers,
            shuffle=True,
        )
        losses: list[float] = []
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                logits = model(x)
                loss_map = F.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "pos_weight": float(pos_weight.cpu()),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        monitor_score = -epoch_loss
        if val_samples:
            _, val_summary = evaluate_model(
                model,
                wafer_cache,
                val_samples,
                args,
                device,
                include_coordinates,
            )
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
        history_records.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_state_dict": best_state,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history_records,
            "args": vars(args),
        }
        torch.save(checkpoint, args.model_path.with_suffix(".checkpoint.pt"))

        val_text = ""
        if val_samples:
            val_text = (
                f" val_ap={record['val_mean_average_precision']:.5f}"
                f" val_auc={record['val_mean_roc_auc']:.5f}"
                f" val_gain={record['val_mean_top32_coverage_gain_pct']:.2f}%"
            )
        print(
            f"epoch {epoch}/{args.epochs} train_loss={epoch_loss:.5f}"
            f" seconds={record['epoch_seconds']:.1f}"
            f" pos_weight={float(pos_weight.cpu()):.2f}{val_text}"
            f" best_epoch={best_epoch}",
            flush=True,
        )
        if val_samples and args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}", flush=True)
            break
    return pd.DataFrame(history_records), best_state, best_epoch


def evaluate_model(
    model: nn.Module,
    wafer_cache: dict[int, CachedWafer],
    samples: list[CachedSample],
    args: argparse.Namespace,
    device: torch.device,
    include_coordinates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    records: list[dict[str, float | int | str]] = []
    loader = make_loader(
        wafer_cache,
        samples,
        include_coordinates,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x).detach().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            for i, (row_index, density, shape) in enumerate(
                zip(batch["row_index"], batch["target_density"], batch["shapes"], strict=True)
            ):
                h, w = shape
                wafer = wafer_cache[int(row_index)]
                scores = probs[i, :h, :w]
                first_mask = batch["first_mask"][i, :h, :w].astype(bool)
                candidates = wafer.valid & ~first_mask
                yy, xx = np.nonzero(candidates)
                if len(yy) == 0:
                    continue
                y_true = wafer.defects[yy, xx].astype(int)
                y_score = scores[yy, xx].astype(float)
                selected = topk_mask_from_scores(scores, candidates, args.top_k)
                coverage_followup = make_coverage_sampling_mask(
                    wafer.wafer_map,
                    n_points=args.top_k,
                    existing_mask=first_mask,
                )
                cnn_metrics = sampling_metrics(wafer.wafer_map, first_mask | selected)
                coverage_metrics = sampling_metrics(wafer.wafer_map, first_mask | coverage_followup)
                cnn_coverage = float(cnn_metrics["defect_coverage"])
                coverage32_coverage = float(coverage_metrics["defect_coverage"])
                if coverage32_coverage > 0:
                    row_gain_pct = 100.0 * (cnn_coverage - coverage32_coverage) / coverage32_coverage
                else:
                    row_gain_pct = float("nan")
                records.append(
                    {
                        "row_index": int(row_index),
                        "failureType": wafer.failure_type,
                        "target_density": float(density),
                        "candidate_count": int(candidates.sum()),
                        "candidate_defects": int((candidates & wafer.defects).sum()),
                        "roc_auc": safe_auc(y_true, y_score),
                        "average_precision": safe_average_precision(y_true, y_score),
                        "top10pct_iou": top_fraction_iou(scores, wafer.defects, candidates, 0.10),
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


def write_report(
    args: argparse.Namespace,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    train_count: int,
    val_count: int,
    test_count: int,
    best_epoch: int,
    device: torch.device,
    include_coordinates: bool,
) -> None:
    lines = [
        "# Batched Sparse CNN Risk Map Run",
        "",
        "This run uses cached wafer masks and padded mini-batches to reduce CPU/GPU transfer overhead.",
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
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "include_coordinates": include_coordinates,
                "use_amp": args.use_amp,
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
    ]
    (args.out_dir / "sparse_cnn_batched_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")
    include_coordinates = not args.drop_coordinate_channels
    input_channels = 7 if include_coordinates else 4

    rng = np.random.default_rng(args.seed)
    patterned = pd.read_pickle(args.patterned)
    train_ids, test_ids = density_policy.split_wafers(patterned, test_size=args.test_size, seed=args.seed)
    train_ids = sparse_cnn.limit_ids(patterned, train_ids, args.max_train_wafers, rng, args.seed)
    test_ids = sparse_cnn.limit_ids(patterned, test_ids, args.max_test_wafers, rng, args.seed + 1)
    train_ids, val_ids = sparse_cnn.validation_split(patterned, train_ids, args.val_size, args.seed)
    densities = [float(value) for value in args.densities]
    all_ids = np.concatenate([train_ids, val_ids, test_ids])
    cache_start = time.perf_counter()
    wafer_cache = build_wafer_cache(patterned, all_ids)
    train_samples = build_sample_cache(wafer_cache, train_ids, densities)
    val_samples = build_sample_cache(wafer_cache, val_ids, densities)
    test_samples = build_sample_cache(wafer_cache, test_ids, densities)
    cache_seconds = time.perf_counter() - cache_start
    print(
        f"prepared cache wafers={len(wafer_cache)} train_samples={len(train_samples)}"
        f" val_samples={len(val_samples)} test_samples={len(test_samples)}"
        f" seconds={cache_seconds:.1f}",
        flush=True,
    )
    print(
        f"device={device} input_channels={input_channels} batch_size={args.batch_size}"
        f" num_workers={args.num_workers} amp={args.use_amp}",
        flush=True,
    )

    model = SparseRiskCNN(in_channels=input_channels, hidden=args.hidden_channels).to(device)
    history, best_state, best_epoch = train_model(
        model,
        wafer_cache,
        train_samples,
        val_samples,
        args,
        device,
        include_coordinates,
    )
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    eval_rows, summary = evaluate_model(
        model,
        wafer_cache,
        test_samples,
        args,
        device,
        include_coordinates,
    )
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
                *(
                    ["normalized_y", "normalized_x", "normalized_radius"]
                    if include_coordinates
                    else []
                ),
            ],
        },
        args.model_path,
    )
    sparse_cnn.plot_training_history(history, args.fig_dir)
    sparse_cnn.plot_summary(summary, args.fig_dir)
    write_report(
        args,
        history,
        summary,
        len(train_samples),
        len(val_samples),
        len(test_samples),
        best_epoch,
        device,
        include_coordinates,
    )
    print(f"wrote batched CNN tables to {args.out_dir}")
    print(f"wrote batched CNN figures to {args.fig_dir}")
    print(f"wrote batched CNN model to {args.model_path}")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
