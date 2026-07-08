"""Train the STFPM student on normal images for one category.

The teacher is frozen; only the student is optimised, learning to reproduce the
teacher's feature maps on defect-free images. Nothing anomalous is seen during
training — the model simply builds a representation of "normal", and anomalies
show up later as places the student cannot match the teacher.
"""

from __future__ import annotations

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


def train() -> Path:
    """Train the student on the category's normal images and save its weights.

    Reads the run parameters from `config`, trains for `config.EPOCHS`, and
    writes the student `state_dict` to `config.STUDENT_WEIGHTS`.

    Returns:
        Path: The path the trained student weights were saved to.
    """
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

    model = STFPM(config.BACKBONE, config.FEATURE_LAYERS).to(device)
    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

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
        print(f"epoch {epoch}/{config.EPOCHS}  loss={running / seen:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.student.state_dict(), config.STUDENT_WEIGHTS)
    print(f"Saved student weights -> {config.STUDENT_WEIGHTS}")
    return config.STUDENT_WEIGHTS
