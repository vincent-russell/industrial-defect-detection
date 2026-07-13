"""Render result figures: training curves and qualitative test examples.

The example panel shows, for a handful of test images, the input, the
ground-truth defect mask, and the predicted anomaly heatmap side by side.
Nothing here feeds back into scoring.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src import data
from src.evaluate import load_model, predict
from src.model import resolve_device

matplotlib.use("Agg")  # non-interactive backend: render straight to files


def _resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear-resize an RGB image to `size` x `size`.

    Args:
        image: RGB image of shape (H, W, 3), uint8.
        size: Target side length.

    Returns:
        Resized RGB image of shape (size, size, 3), uint8.
    """
    return np.asarray(
        Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
    )


def _normalize(amap: np.ndarray) -> np.ndarray:
    """Min-max scale an anomaly map to [0, 1] for display.

    Args:
        amap: Anomaly map with arbitrary positive range.

    Returns:
        The map rescaled to [0, 1]; all-zeros if the map is constant.
    """
    lo, hi = float(amap.min()), float(amap.max())
    return (amap - lo) / (hi - lo) if hi > lo else np.zeros_like(amap)


def save_training_curves(
    history: dict[str, list[float]] | None = None, path: Path | None = None
) -> Path:
    """Plot the training history: loss and (if tracked) test AUROC per epoch.

    Draws the mean distillation loss on the left axis and, when per-epoch
    scoring was enabled during training, the image- and pixel-level test AUROC
    on a shared right axis (a monitoring diagnostic, noted as such on the
    figure).

    Args:
        history: Per-epoch arrays as returned by `train`, or None to load the
            saved ``history_<backbone>_<category>.json`` from
            `config.RESULTS_DIR`.
        path: Destination PNG, or None for the default
            ``training_<backbone>_<category>.png`` under `config.RESULTS_DIR`.

    Returns:
        The path the figure was written to.

    Raises:
        FileNotFoundError: If `history` is None and no saved history exists.
    """
    if path is None:
        path = config.RESULTS_DIR / f"training_{config.BACKBONE}_{config.CATEGORY}.png"
    if history is None:
        hist_path = config.RESULTS_DIR / f"history_{config.BACKBONE}_{config.CATEGORY}.json"
        if not hist_path.exists():
            raise FileNotFoundError(
                f"No training history at {hist_path}. Train the model first."
            )
        with open(hist_path, encoding="utf-8") as f:
            loaded: dict[str, list[float]] = json.load(f)
        history = loaded

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(history["epoch"], history["loss"], color="tab:blue", label="train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("distillation loss", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")
    ax_loss.grid(True, alpha=0.3)
    handles, labels = ax_loss.get_legend_handles_labels()

    # Second axis for the test-AUROC diagnostic, only if it was recorded.
    if history.get("eval_epoch"):
        ax_auc = ax_loss.twinx()
        ax_auc.plot(
            history["eval_epoch"], history["image_auroc"],
            "o-", color="tab:orange", label="image AUROC (test)",
        )
        ax_auc.plot(
            history["eval_epoch"], history["pixel_auroc"],
            "s-", color="tab:green", label="pixel AUROC (test)",
        )
        ax_auc.set_ylabel("test AUROC")
        ax_auc.set_ylim(0.0, 1.02)
        h2, l2 = ax_auc.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
        fig.text(
            0.5, 0.005,
            "AUROC is a test-split diagnostic (monitoring only; not used for model selection).",
            ha="center", fontsize=8, style="italic", color="0.4",
        )

    ax_loss.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"STFPM ({config.BACKBONE}) — {config.CATEGORY} training", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _select_examples(
    samples: list[data.VisaSample], count: int
) -> list[data.VisaSample]:
    """Pick a representative set of test samples to visualise.

    Chooses anomalous samples with a ground-truth mask, spread evenly across
    the test split, plus one normal sample for contrast.

    Args:
        samples: The category's full test split.
        count: Total number of examples to return, including the normal.

    Returns:
        The selected samples, anomalies first.
    """
    anomalies = [s for s in samples if s.is_anomaly and s.mask_path is not None]
    normals = [s for s in samples if not s.is_anomaly]

    n_anom = max(count - 1, 1) if normals else count
    n_anom = min(n_anom, len(anomalies))
    idx = np.linspace(0, len(anomalies) - 1, n_anom).round().astype(int)
    chosen = [anomalies[i] for i in dict.fromkeys(idx.tolist())]  # ordered, unique

    if normals:
        chosen.append(normals[len(normals) // 2])
    return chosen


def save_examples(path: Path | None = None) -> Path:
    """Render and save a qualitative panel of test-set predictions.

    Builds a grid with one row per example and three columns: the input, the
    ground-truth defect mask overlaid in red, and the predicted anomaly
    heatmap.

    Args:
        path: Destination PNG, or None for the default
            ``examples_<backbone>_<category>.png`` under `config.RESULTS_DIR`.

    Returns:
        The path the figure was written to.

    Raises:
        FileNotFoundError: If no trained student weights exist yet.
    """
    if path is None:
        path = config.RESULTS_DIR / f"examples_{config.BACKBONE}_{config.CATEGORY}.png"

    device = resolve_device()
    model = load_model(device)
    samples = data.load_samples(category=config.CATEGORY, split="test")
    examples = _select_examples(samples, config.NUM_FIGURE_EXAMPLES)

    size = config.IMG_SIZE
    titles = ("Input", "Ground truth", "Anomaly map")
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(9, 3 * len(examples)), squeeze=False
    )

    for row, sample in enumerate(examples):
        disp = _resize_rgb(data.load_image(sample), size)
        amap = _normalize(predict(model, data.load_image(sample), device))
        mask = data.load_mask(sample)
        gt = data.resize_mask(mask, size) if mask is not None else np.zeros((size, size), bool)

        # Column 0 — the input image, tagged with its true label on the y-axis.
        axes[row][0].imshow(disp)
        axes[row][0].set_ylabel(sample.label, fontsize=11)

        # Column 1 — input with the ground-truth defect in translucent red.
        axes[row][1].imshow(disp)
        red = np.zeros((size, size, 4))
        red[gt] = (1.0, 0.0, 0.0, 0.5)
        axes[row][1].imshow(red)

        # Column 2 — input with the predicted anomaly heatmap on top.
        axes[row][2].imshow(disp)
        axes[row][2].imshow(amap, cmap="inferno", alpha=0.5, vmin=0.0, vmax=1.0)

        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            if row == 0:
                axes[row][col].set_title(titles[col], fontsize=12)

    fig.suptitle(
        f"STFPM ({config.BACKBONE}) — {config.CATEGORY}", fontsize=14, y=0.995
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
