"""Evaluate a trained STFPM student on a category's test split.

Scores every test image with the anomaly map and reports image-level ROC AUC,
pixel-level ROC AUC, and the best achievable IoU. Ground-truth masks are used
for scoring only, never training.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

import config
from src import data, metrics
from src.data import build_transform
from src.model import STFPM, anomaly_map, resolve_device

# Metrics that are meaningful to average across categories (the IoU threshold
# is category-specific, so it is reported per category but never averaged).
_MEAN_KEYS = ("image_auroc", "pixel_auroc", "best_iou")


def load_model(device: torch.device) -> STFPM:
    """Build an STFPM model and load the saved student weights.

    Args:
        device: Device to place the model on.

    Returns:
        The model in eval mode, ready for inference.

    Raises:
        FileNotFoundError: If no trained student weights exist yet.
    """
    if not config.STUDENT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"No trained student at {config.STUDENT_WEIGHTS}. Run training first."
        )
    model = STFPM(config.BACKBONE, config.FEATURE_LAYERS).to(device)
    state = torch.load(config.STUDENT_WEIGHTS, map_location=device)
    model.student.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: STFPM, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute the smoothed anomaly map for one image at model resolution.

    Args:
        model: The trained model.
        image: RGB image, shape (H, W, 3), uint8.
        device: Device the model lives on.

    Returns:
        Anomaly map of shape (IMG_SIZE, IMG_SIZE); higher is more anomalous.
    """
    transform = build_transform(config.IMG_SIZE)
    tensor = cast(torch.Tensor, transform(np.ascontiguousarray(image)))
    tensor = tensor.unsqueeze(0).to(device)
    teacher_feats, student_feats = model(tensor)
    amap = anomaly_map(teacher_feats, student_feats, (config.IMG_SIZE, config.IMG_SIZE))
    if config.SMOOTH_SIGMA > 0:
        kernel = 2 * int(round(3 * config.SMOOTH_SIGMA)) + 1  # cover +/-3 sigma
        amap = gaussian_blur(
            amap, kernel_size=[kernel, kernel], sigma=[config.SMOOTH_SIGMA]
        )
    return amap.squeeze().cpu().numpy()


def _save_metrics(results: dict[str, float], path: Path | None = None) -> Path:
    """Write evaluation metrics, with their run configuration, to JSON.

    Args:
        results: Metrics as returned by `evaluate`.
        path: Destination file, or None for the default
            ``metrics_<backbone>_<category>.json`` under `config.RESULTS_DIR`.

    Returns:
        The path written to.
    """
    if path is None:
        path = config.RESULTS_DIR / f"metrics_{config.BACKBONE}_{config.CATEGORY}.json"
    payload = {
        "category": config.CATEGORY,
        "backbone": config.BACKBONE,
        "feature_layers": list(config.FEATURE_LAYERS),
        "img_size": config.IMG_SIZE,
        "epochs": config.EPOCHS,
        **results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def save_summary(rows: list[dict], path: Path | None = None) -> dict:
    """Write per-category rows and their category means to one JSON file.

    The payload records the run configuration, one row per category, and the
    ``mean`` over categories of image/pixel AUROC and best IoU.

    Args:
        rows: Per-category results, each with a ``category`` key plus the
            metric keys returned by `score`.
        path: Destination file, or None for the default
            ``summary_<backbone>.json`` under `config.RESULTS_DIR`.

    Returns:
        The summary payload as written.
    """
    if path is None:
        path = config.RESULTS_DIR / f"summary_{config.BACKBONE}.json"
    mean = {
        key: sum(row[key] for row in rows) / len(rows) for key in _MEAN_KEYS
    }
    payload = {
        "backbone": config.BACKBONE,
        "feature_layers": list(config.FEATURE_LAYERS),
        "img_size": config.IMG_SIZE,
        "epochs": config.EPOCHS,
        "num_categories": len(rows),
        "categories": rows,
        "mean": mean,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved summary -> {path}")
    return payload


def print_summary(summary: dict) -> None:
    """Print a table of per-category metrics and their category mean.

    Args:
        summary: Summary payload as built by `save_summary`.
    """
    mean = summary["mean"]
    print(f"\n{'category':<12}{'image AUROC':>14}{'pixel AUROC':>14}{'best IoU':>12}")
    print("-" * 52)
    for row in summary["categories"]:
        print(
            f"{row['category']:<12}{row['image_auroc']:>14.4f}"
            f"{row['pixel_auroc']:>14.4f}{row['best_iou']:>12.4f}"
        )
    print("-" * 52)
    print(
        f"{'MEAN':<12}{mean['image_auroc']:>14.4f}"
        f"{mean['pixel_auroc']:>14.4f}{mean['best_iou']:>12.4f}"
    )


@torch.no_grad()
def score(
    model: STFPM,
    samples: list[data.VisaSample],
    device: torch.device,
    progress: bool = False,
) -> dict[str, float]:
    """Score a model over samples and return metrics, writing no files.

    Shared by `evaluate` and by per-epoch monitoring during training: the model
    is switched to eval mode and restored afterwards, so it is safe to call
    mid-training. Pixel metrics are computed at `config.IMG_SIZE` resolution
    for both the anomaly map and the ground-truth mask.

    Args:
        model: The model to score.
        samples: The test samples to score over.
        device: Device the model lives on.
        progress: If True, show a per-image progress bar.

    Returns:
        Metrics with keys ``image_auroc``, ``pixel_auroc``, ``best_iou``,
        and ``iou_threshold``.
    """
    was_training = model.training
    model.eval()

    image_scores, image_labels = [], []
    pixel_maps, pixel_masks = [], []
    for sample in tqdm(samples, desc="scoring", leave=False, disable=not progress):
        amap = predict(model, data.load_image(sample), device)
        image_scores.append(float(amap.max()))
        image_labels.append(int(sample.is_anomaly))

        mask = data.load_mask(sample)
        gt = (
            data.resize_mask(mask, config.IMG_SIZE)
            if mask is not None
            else np.zeros_like(amap, dtype=bool)
        )
        pixel_maps.append(amap)
        pixel_masks.append(gt)

    pixel_scores = np.concatenate([m.ravel() for m in pixel_maps])
    pixel_labels = np.concatenate([m.ravel() for m in pixel_masks])
    best, best_thr = metrics.best_iou(pixel_labels, pixel_scores)

    if was_training:
        model.train()

    return {
        "image_auroc": metrics.roc_auc(np.array(image_labels), np.array(image_scores)),
        "pixel_auroc": metrics.roc_auc(pixel_labels, pixel_scores),
        "best_iou": best,
        "iou_threshold": best_thr,
    }


def evaluate() -> dict[str, float]:
    """Score the trained student over the category's test split and save metrics.

    Returns:
        Metrics with keys ``image_auroc``, ``pixel_auroc``, ``best_iou``,
        and ``iou_threshold``.
    """
    device = resolve_device()
    model = load_model(device)

    samples = data.load_samples(category=config.CATEGORY, split="test")
    print(f"Evaluating on {len(samples)} test '{config.CATEGORY}' images.")

    results = score(model, samples, device, progress=True)
    print(
        f"image AUROC={results['image_auroc']:.4f}  "
        f"pixel AUROC={results['pixel_auroc']:.4f}  "
        f"best IoU={results['best_iou']:.4f} @ thr={results['iou_threshold']:.4g}"
    )
    saved = _save_metrics(results)
    print(f"Saved metrics -> {saved}")
    return results
