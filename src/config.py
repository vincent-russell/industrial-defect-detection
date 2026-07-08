"""Project paths and experiment configuration.

Two things live here, kept deliberately separate:

* **Project layout** — the fixed on-disk directories, as module-level constants.
  Everything is derived from the repo root, so paths are correct no matter where
  Python is launched from.
* **Experiment configuration** — every tunable knob for a run, gathered into one
  typed, immutable `Config`. Prefer this over scattering literals through the
  code or reaching for a YAML file: a dataclass gives type checking, IDE
  autocomplete, computed fields (e.g. the checkpoint path for a SAM variant), and
  cheap provenance (`Config.to_json`) — with no extra dependency.

Pure Python — safe to import from any entry point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Project layout -----------------------------------------------------------

# Repo root. This file lives at <root>/src/config.py, so go up two levels.
ROOT = Path(__file__).resolve().parents[1]

# Local, gitignored working directories (created on demand by `ensure_dirs`).
DATA_DIR = ROOT / "data"        # dataset cache (VisA tarball + extraction)
MODELS_DIR = ROOT / "models"    # pretrained SAM weights (.pth), downloaded
RESULTS_DIR = ROOT / "results"  # generated outputs (masks, figures, metrics)

# Curated figures committed to the repo (for the README / portfolio) live here.
ASSETS_DIR = ROOT / "assets"


# --- SAM checkpoints ----------------------------------------------------------

# Meta's published weight filenames and official download URLs, per variant.
# VRAM footprint grows vit_b < vit_l < vit_h; pick to fit the local GPU.
SAM_CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
SAM_URLS = {
    variant: f"https://dl.fbaipublicfiles.com/segment_anything/{name}"
    for variant, name in SAM_CHECKPOINTS.items()
}


# --- Experiment configuration -------------------------------------------------

@dataclass(frozen=True)
class Config:
    """All parameters for one run, in a single typed, immutable place.

    Construct with overrides at an entry point, e.g. ``Config(sam_variant="vit_l")``.
    Frozen so a run's settings can't be mutated halfway through.
    """

    # Data selection
    category: str = "candle"          # one of data.CATEGORIES
    split: str = "test"               # "train" or "test"

    # Model
    sam_variant: str = "vit_b"        # vit_b | vit_l | vit_h  (VRAM small -> large)
    device: str = "cuda"              # "cuda" or "cpu"

    # SAM automatic mask generation
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95

    # Reproducibility
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sam_variant not in SAM_CHECKPOINTS:
            raise ValueError(
                f"Unknown sam_variant {self.sam_variant!r}; "
                f"choose from {tuple(SAM_CHECKPOINTS)}"
            )

    @property
    def sam_checkpoint(self) -> Path:
        """Local path to this variant's weight file (may not exist yet)."""
        return MODELS_DIR / SAM_CHECKPOINTS[self.sam_variant]

    @property
    def sam_url(self) -> str:
        """Official download URL for this variant's weight file."""
        return SAM_URLS[self.sam_variant]

    def to_json(self, path: Path) -> None:
        """Write these settings alongside a run's outputs, for provenance."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def ensure_dirs() -> None:
    """Create the working directories if they don't exist yet."""
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
