"""VisA dataset: download, official split, and sample loading.

VisA (Zou et al., ECCV 2022) is a visual anomaly benchmark of 12 object
categories. Each category has *normal* and *anomaly* images, plus pixel-level
masks for the anomalies — exactly what we need for segmentation.

Source (verified 2026-06): the tarball lives on Amazon's public S3 bucket and
the canonical train/test split lives in the spot-diff GitHub repo as
`split_csv/1cls.csv`. Using that official split keeps our results comparable to
the literature instead of inventing our own.

On-disk layout after extraction (paths in the split CSV are relative to the
dataset root, i.e. the directory that holds the category folders):

    <root>/<category>/Data/Images/Normal/*.JPG
    <root>/<category>/Data/Images/Anomaly/*.JPG
    <root>/<category>/Data/Masks/Anomaly/*.png      # masks only for anomalies

Pure Python. Everything runs locally: the ~16 GB tarball downloads and extracts
under `data/` (gitignored). Budget disk and time for the first download; it is
idempotent, so reruns are cheap.
"""

from __future__ import annotations

import csv
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import config

# --- Sources (verified live 2026-06) ------------------------------------------

VISA_TAR_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
# Official 1-class train/test split shipped in the spot-diff repo (not in the tar).
SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"

# The 12 object categories, in the dataset's own order.
CATEGORIES = (
    "candle", "capsules", "cashew", "chewing_gum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)

# Where things land under the local data directory (all gitignored).
VISA_DIR = config.DATA_DIR / "VisA"          # extraction target
VISA_TAR = config.DATA_DIR / "VisA_20220922.tar"
SPLIT_CSV = config.DATA_DIR / "split_csv_1cls.csv"


@dataclass(frozen=True)
class VisaSample:
    """One labelled image, with absolute paths resolved on disk."""

    category: str           # e.g. "candle"
    split: str              # "train" or "test"
    label: str              # "normal" or "anomaly"
    image_path: Path
    mask_path: Path | None  # ground-truth mask; None for normal samples

    @property
    def is_anomaly(self) -> bool:
        return self.label == "anomaly"


# --- Download / extract -------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with a progress bar (skips if already present)."""
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")  # download to a temp name...
    print(f"Downloading {url}\n        -> {dest}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted URL)
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)  # ...and rename only on success, so partials never look done.


def _find_dataset_root(search_dir: Path) -> Path | None:
    """Locate the directory holding the category folders.

    The tar's top-level folder name isn't something we want to hardcode, so we
    search instead: a directory is the dataset root if it contains "candle".
    """
    if (search_dir / "candle").is_dir():
        return search_dir
    for child in sorted(search_dir.iterdir()):
        if child.is_dir() and (child / "candle").is_dir():
            return child
    return None


def download_visa(keep_tar: bool = False) -> Path:
    """Download and extract VisA; return the dataset root (dir of categories).

    Idempotent: skips the download if the tar is present and skips extraction if
    the categories are already on disk. Set `keep_tar=False` (default) to delete
    the ~16 GB archive after extracting.
    """
    config.ensure_dirs()

    root = _find_dataset_root(VISA_DIR) if VISA_DIR.exists() else None
    if root is not None:
        print(f"VisA already extracted at: {root}")
        return root

    _download(VISA_TAR_URL, VISA_TAR)

    print(f"Extracting {VISA_TAR} -> {VISA_DIR}")
    VISA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(VISA_TAR) as tar:
        tar.extractall(VISA_DIR)  # noqa: S202 (trusted archive)

    root = _find_dataset_root(VISA_DIR)
    if root is None:
        raise RuntimeError(
            f"Extraction finished but no category folders found under {VISA_DIR}."
        )
    if not keep_tar:
        VISA_TAR.unlink()
    print(f"VisA ready at: {root}")
    return root


# --- Split + samples ----------------------------------------------------------

def load_samples(
    category: str | None = None,
    split: str | None = None,
    label: str | None = None,
) -> list[VisaSample]:
    """Return VisA samples from the official 1-class split, optionally filtered.

    Resolves every image/mask to an absolute path under the extracted dataset.
    Downloads the small split CSV on first use; assumes the images themselves
    are already extracted (call `download_visa()` first).
    """
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; choose from {CATEGORIES}")

    _download(SPLIT_CSV_URL, SPLIT_CSV)

    root = _find_dataset_root(VISA_DIR) if VISA_DIR.exists() else None
    if root is None:
        raise FileNotFoundError(
            f"VisA images not found under {VISA_DIR}. Run download_visa() first."
        )

    samples: list[VisaSample] = []
    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if category and row["object"] != category:
                continue
            if split and row["split"] != split:
                continue
            if label and row["label"] != label:
                continue
            mask = row["mask"].strip()
            samples.append(
                VisaSample(
                    category=row["object"],
                    split=row["split"],
                    label=row["label"],
                    image_path=root / row["image"],
                    mask_path=(root / mask) if mask else None,
                )
            )
    return samples


# --- Pixel loaders ------------------------------------------------------------

def load_image(sample: VisaSample) -> np.ndarray:
    """Load a sample's image as an RGB uint8 array, shape (H, W, 3)."""
    with Image.open(sample.image_path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(sample: VisaSample) -> np.ndarray | None:
    """Load a sample's ground-truth mask as a boolean array (H, W).

    Returns None for normal samples (which have no mask). VisA masks are stored
    with non-zero pixels marking the defect, so we threshold at > 0.
    """
    if sample.mask_path is None:
        return None
    with Image.open(sample.mask_path) as m:
        return np.asarray(m.convert("L")) > 0
