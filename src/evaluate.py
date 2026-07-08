"""Evaluate a trained STFPM student on a category's test split.

Runs the student over every test image, turns the teacher/student discrepancy
into a per-pixel anomaly map, and reports three numbers: image-level ROC AUC
(is the image anomalous?), pixel-level ROC AUC (which pixels?), and the best
achievable IoU. Ground-truth masks are used for scoring only — never training.
"""

from __future__ import annotations

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


def evaluate() -> dict[str, float]:
    """Score the trained student over the category's test split.

    Pixel-level metrics are computed at `config.IMG_SIZE` resolution (both the
    anomaly map and the ground-truth mask), which keeps memory modest and is the
    resolution the model actually predicts at.

    Returns:
        dict[str, float]: Metrics with keys ``image_auroc``, ``pixel_auroc``,
            ``best_iou``, and ``iou_threshold``.
    """
    device = resolve_device()
    model = load_model(device)

    samples = data.load_samples(category=config.CATEGORY, split="test")
    print(f"Evaluating on {len(samples)} test '{config.CATEGORY}' images.")

    image_scores, image_labels = [], []
    pixel_maps, pixel_masks = [], []
    for sample in tqdm(samples, desc="evaluating", leave=False):
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

    results = {
        "image_auroc": metrics.roc_auc(np.array(image_labels), np.array(image_scores)),
        "pixel_auroc": metrics.roc_auc(pixel_labels, pixel_scores),
        "best_iou": best,
        "iou_threshold": best_thr,
    }
    print(
        f"image AUROC={results['image_auroc']:.4f}  "
        f"pixel AUROC={results['pixel_auroc']:.4f}  "
        f"best IoU={results['best_iou']:.4f} @ thr={results['iou_threshold']:.4g}"
    )
    return results
