"""VisA dataset: download, official split, sample loading, and a torch dataset.

VisA (Zou et al., ECCV 2022) is a visual anomaly benchmark of 12 object
categories with normal and anomalous images, plus pixel-level defect masks for
the anomalies. The official one-class train/test split (`1cls.csv`) is used so
results stay comparable to the literature.

On-disk layout after extraction (paths in the split CSV are relative to the
dataset root, i.e. the directory holding the category folders):

    <root>/<category>/Data/Images/Normal/*.JPG
    <root>/<category>/Data/Images/Anomaly/*.JPG
    <root>/<category>/Data/Masks/Anomaly/*.png      # masks only for anomalies

The ~16 GB tarball downloads and extracts under `data/` on first use; download
and extraction are idempotent.
"""

import csv
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import config

# ImageNet statistics — the teacher backbone was pretrained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Sources -------------------------------------------------------------------

VISA_TAR_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
# Official 1-class train/test split from the spot-diff repo (not in the tar).
SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"

# The 12 object categories, in the dataset's own order.
CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)

# Local paths under the data directory.
VISA_DIR = config.DATA_DIR / "VisA"          # extraction target
VISA_TAR = config.DATA_DIR / "VisA_20220922.tar"
SPLIT_CSV = config.DATA_DIR / "split_csv_1cls.csv"


@dataclass(frozen=True)
class VisaSample:
    """One labelled VisA image with resolved on-disk paths.

    Attributes:
        category (str): Object category, e.g. "candle".
        split (str): Dataset split, "train" or "test".
        label (str): Sample label, "normal" or "anomaly".
        image_path (Path): Absolute path to the RGB image on disk.
        mask_path (Path | None): Absolute path to the ground-truth mask, or None
            for normal samples.
    """

    category: str
    split: str
    label: str
    image_path: Path
    mask_path: Path | None

    @property
    def is_anomaly(self) -> bool:
        """Whether this sample is labelled as an anomaly.

        Returns:
            bool: True if the label is "anomaly", False otherwise.
        """
        return self.label == "anomaly"


# --- Download / extract -------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Stream a URL to a local file with a progress bar.

    Downloads to a temporary ``.part`` file and renames it into place on
    success, so an interrupted download never looks complete. Skips the
    download if `dest` already exists.

    Args:
        url (str): Source URL.
        dest (Path): Destination file path.
    """
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n        -> {dest}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted URL)
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)


def _find_dataset_root(search_dir: Path) -> Path | None:
    """Locate the directory that holds the VisA category folders.

    A directory is the dataset root if it contains a "candle" folder; this
    avoids hardcoding the tar's top-level folder name.

    Args:
        search_dir (Path): Directory to search, either the dataset root itself
            or its immediate parent.

    Returns:
        Path | None: The dataset root, or None if not found.
    """
    if (search_dir / "candle").is_dir():
        return search_dir
    for child in sorted(search_dir.iterdir()):
        if child.is_dir() and (child / "candle").is_dir():
            return child
    return None


def download_visa(keep_tar: bool = False) -> Path:
    """Download and extract VisA, returning the dataset root.

    Idempotent: skips the download if the tar is present and skips extraction
    if the categories are already on disk.

    Args:
        keep_tar (bool): If False (default), delete the archive after
            extracting.

    Returns:
        Path: The dataset root, i.e. the directory holding the category folders.

    Raises:
        RuntimeError: If extraction finishes but no category folders are found.
    """
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
    """Load VisA samples from the official 1-class split, optionally filtered.

    Downloads the split CSV on first use; the images themselves must already
    be extracted (call `download_visa` first).

    Args:
        category (str | None): Restrict to one of `CATEGORIES`, or None for all.
        split (str | None): Restrict to "train" or "test", or None for both.
        label (str | None): Restrict to "normal" or "anomaly", or None for both.

    Returns:
        list[VisaSample]: The matching samples with resolved on-disk paths.

    Raises:
        ValueError: If `category` is not a known VisA category.
        FileNotFoundError: If the extracted VisA images cannot be found.
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
    """Load a sample's image as an RGB array.

    Args:
        sample (VisaSample): The sample to load.

    Returns:
        np.ndarray: uint8 array of shape (H, W, 3).
    """
    with Image.open(sample.image_path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(sample: VisaSample) -> np.ndarray | None:
    """Load a sample's ground-truth defect mask as a boolean array.

    Args:
        sample (VisaSample): The sample to load.

    Returns:
        np.ndarray | None: Boolean array of shape (H, W) where True marks
            defect pixels, or None for normal samples.
    """
    if sample.mask_path is None:
        return None
    with Image.open(sample.mask_path) as m:
        return np.asarray(m.convert("L")) > 0


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to `size` x `size`.

    Args:
        mask (np.ndarray): Boolean mask of shape (H, W).
        size (int): Target side length in pixels.

    Returns:
        np.ndarray: Boolean mask of shape (size, size).
    """
    resized = Image.fromarray(mask.astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


# --- Torch dataset ------------------------------------------------------------

def build_transform(img_size: int) -> transforms.Compose:
    """Build the image transform: resize, tensor, ImageNet-normalise.

    Args:
        img_size (int): Side length in pixels of the square model input.

    Returns:
        transforms.Compose: Maps an (H, W, 3) uint8 RGB array to a normalised
            float tensor of shape (3, img_size, img_size).
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class VisaDataset(Dataset):
    """A torch dataset of VisA images (no labels — training uses normals only).

    Attributes:
        samples (list[VisaSample]): The samples to serve.
        transform (transforms.Compose): Transform applied to each loaded image.
    """

    def __init__(self, samples: list[VisaSample], transform: transforms.Compose):
        """Wrap a list of samples with an image transform.

        Args:
            samples (list[VisaSample]): Samples to serve.
            transform (transforms.Compose): Transform from `build_transform`.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            int: Sample count.
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        """Load and transform the image at `index`.

        Args:
            index (int): Position in `samples`.

        Returns:
            torch.Tensor: Normalised image tensor of shape (3, H, W).
        """
        image = load_image(self.samples[index])
        return self.transform(np.ascontiguousarray(image))
