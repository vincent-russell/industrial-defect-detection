"""Train the STFPM student on normal images for one category.

The teacher is frozen; only the student is optimised, learning to reproduce the
teacher's feature maps on defect-free images. Nothing anomalous is seen during
training — the model simply builds a representation of "normal", and anomalies
show up later as places the student cannot match the teacher.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src import data
from src.data import VisaDataset, build_transform
from src.model import STFPM, distillation_loss


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs for reproducible runs.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Return the configured device, falling back to CPU if CUDA is unavailable.

    Returns:
        torch.device: The device to run on.
    """
    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config.DEVICE)


def save_history(history: dict[str, list[float]], path: Path | None = None) -> Path:
    """Write the training history to a self-describing JSON file under `results/`.

    Records the run configuration alongside the per-epoch arrays, so the file is
    interpretable on its own and can be plotted later without a live model.

    Args:
        history (dict[str, list[float]]): Per-epoch arrays as built by `train`.
        path (Path | None): Destination file, or None to derive a default name of
            ``history_<backbone>_<category>.json`` under `config.RESULTS_DIR`.

    Returns:
        Path: The path the history was written to.
    """
    if path is None:
        path = config.RESULTS_DIR / f"history_{config.BACKBONE}_{config.CATEGORY}.json"
    payload = {
        "category": config.CATEGORY,
        "backbone": config.BACKBONE,
        "feature_layers": list(config.FEATURE_LAYERS),
        "img_size": config.IMG_SIZE,
        "epochs": config.EPOCHS,
        **history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def train() -> dict[str, list[float]]:
    """Train the student on the category's normal images and save its weights.

    Reads the run parameters from `config` and trains for `config.EPOCHS`,
    recording the mean distillation loss each epoch. If `config.EVAL_EVERY_EPOCHS`
    is positive, the model is also scored on the test split at that cadence (and
    on the final epoch) as a convergence diagnostic — monitoring only, never used
    for model selection or early stopping. Writes the student `state_dict` to
    `config.STUDENT_WEIGHTS` and the per-epoch history to `results/`.

    Returns:
        dict[str, list[float]]: The training history, with keys ``epoch`` and
            ``loss`` and, when per-epoch scoring is enabled, ``eval_epoch``,
            ``image_auroc``, and ``pixel_auroc``.
    """
    # Local import breaks a cycle: evaluate imports resolve_device from this module.
    from src import evaluate

    set_seed(config.SEED)
    device = resolve_device()

    samples = data.load_samples(
        category=config.CATEGORY, split="train", label="normal"
    )
    loader = DataLoader(
        VisaDataset(samples, build_transform(config.IMG_SIZE)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    print(f"Training on {len(samples)} normal '{config.CATEGORY}' images.")

    # Load the test split once for the per-epoch diagnostic (empty if disabled).
    eval_samples = (
        data.load_samples(category=config.CATEGORY, split="test")
        if config.EVAL_EVERY_EPOCHS > 0
        else []
    )

    model = STFPM(config.BACKBONE, config.FEATURE_LAYERS).to(device)
    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    history: dict[str, list[float]] = {
        "epoch": [], "loss": [], "eval_epoch": [], "image_auroc": [], "pixel_auroc": []
    }

    model.train()
    for epoch in range(1, config.EPOCHS + 1):
        running, seen = 0.0, 0
        for images in tqdm(loader, desc=f"epoch {epoch}/{config.EPOCHS}", leave=False):
            images = images.to(device)
            teacher_feats, student_feats = model(images)
            loss = distillation_loss(teacher_feats, student_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        epoch_loss = running / seen
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        msg = f"epoch {epoch}/{config.EPOCHS}  loss={epoch_loss:.4f}"

        # Diagnostic scoring on the test split (restores train mode internally).
        if eval_samples and (epoch % config.EVAL_EVERY_EPOCHS == 0 or epoch == config.EPOCHS):
            scored = evaluate.score(model, eval_samples, device)
            history["eval_epoch"].append(epoch)
            history["image_auroc"].append(scored["image_auroc"])
            history["pixel_auroc"].append(scored["pixel_auroc"])
            msg += (
                f"  image AUROC={scored['image_auroc']:.4f}"
                f"  pixel AUROC={scored['pixel_auroc']:.4f}"
            )
        print(msg)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.student.state_dict(), config.STUDENT_WEIGHTS)
    print(f"Saved student weights -> {config.STUDENT_WEIGHTS}")
    saved = save_history(history)
    print(f"Saved training history -> {saved}")
    return history
