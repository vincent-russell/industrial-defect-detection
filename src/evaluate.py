"""Evaluate a trained STFPM student on a category's test split.

Runs the student over every test image, turns the teacher/student discrepancy
into a per-pixel anomaly map, and reports three numbers: image-level ROC AUC
(is the image anomalous?), pixel-level ROC AUC (which pixels?), and the best
achievable IoU. Ground-truth masks are used for scoring only — never training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

import config
from src import data, metrics
from src.data import build_transform
from src.model import STFPM, anomaly_map
from src.train import resolve_device


def load_model(device: torch.device) -> STFPM:
    """Build an STFPM model and load the saved student weights.

    Args:
        device (torch.device): Device to place the model on.

    Returns:
        STFPM: The model in eval mode, ready for inference.

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
        model (STFPM): The trained model.
        image (np.ndarray): RGB image, shape (H, W, 3), uint8.
        device (torch.device): Device the model lives on.

    Returns:
        np.ndarray: Anomaly map of shape (IMG_SIZE, IMG_SIZE); higher is more
            anomalous.
    """
    transform = build_transform(config.IMG_SIZE)
    tensor = transform(np.ascontiguousarray(image)).unsqueeze(0).to(device)
    teacher_feats, student_feats = model(tensor)
    amap = anomaly_map(teacher_feats, student_feats, (config.IMG_SIZE, config.IMG_SIZE))
    if config.SMOOTH_SIGMA > 0:
        kernel = 2 * int(round(3 * config.SMOOTH_SIGMA)) + 1  # cover +/-3 sigma
        amap = gaussian_blur(amap, kernel_size=kernel, sigma=config.SMOOTH_SIGMA)
    return amap.squeeze().cpu().numpy()


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to a square of side `size`.

    Args:
        mask (np.ndarray): Boolean mask of shape (H, W).
        size (int): Target side length.

    Returns:
        np.ndarray: Boolean mask of shape (size, size).
    """
    resized = Image.fromarray(mask.astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    return np.asarray(resized) > 0


def save_metrics(results: dict[str, float], path: Path | None = None) -> Path:
    """Write evaluation metrics to a self-describing JSON file under `results/`.

    The saved payload records the run configuration (category, backbone, feature
    layers, image size, epochs) alongside the metric values, so a result file is
    interpretable on its own without consulting `config`.

    Args:
        results (dict[str, float]): Metrics as returned by `evaluate`.
        path (Path | None): Destination file, or None to derive a default name
            of ``metrics_<backbone>_<category>.json`` under `config.RESULTS_DIR`.

    Returns:
        Path: The path the metrics were written to.
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


@torch.no_grad()
def score(
    model: STFPM,
    samples: list[data.VisaSample],
    device: torch.device,
    progress: bool = False,
) -> dict[str, float]:
    """Score a model over samples and return metrics, doing no I/O.

    This is the pure scoring core shared by `evaluate` (full run, saves a file)
    and by per-epoch monitoring during training (called repeatedly, no output).
    The model is switched to eval mode for scoring and restored to its previous
    mode afterwards, so it is safe to call mid-training. Pixel-level metrics are
    computed at `config.IMG_SIZE` resolution — both the anomaly map and the
    ground-truth mask — which keeps memory modest and matches the resolution the
    model actually predicts at.

    Args:
        model (STFPM): The model to score (trained or mid-training).
        samples (list[data.VisaSample]): The test samples to score over.
        device (torch.device): Device the model lives on.
        progress (bool): If True, show a per-image progress bar.

    Returns:
        dict[str, float]: Metrics with keys ``image_auroc``, ``pixel_auroc``,
            ``best_iou``, and ``iou_threshold``.
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
            _resize_mask(mask, config.IMG_SIZE)
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
        dict[str, float]: Metrics with keys ``image_auroc``, ``pixel_auroc``,
            ``best_iou``, and ``iou_threshold``.
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
    saved = save_metrics(results)
    print(f"Saved metrics -> {saved}")
    return results
