"""Editable parameters for the pipeline — the single place to change a run.

Everything here is a plain module-level value: no classes to instantiate and no
functions to call, so it is obvious what to change and where. `main.py` and the
modules in `src/` read these values; edit them here rather than hard-coding
literals elsewhere.
"""

from pathlib import Path

# =============================================================================
# Run parameters — the knobs you actually tune between runs.
# =============================================================================

# Data selection
CATEGORY = "candle"   # VisA object category (see CATEGORIES in src/data.py)
SPLIT = "test"        # "train" or "test"

# Segment Anything model
SAM_VARIANT = "vit_b"  # "vit_b" | "vit_l" | "vit_h"  (VRAM footprint small -> large)
DEVICE = "cuda"        # "cuda" or "cpu"

# SAM automatic mask generation
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.95

# Reproducibility
SEED = 0


# =============================================================================
# Project paths — derived from the repo root; rarely edited.
# =============================================================================

ROOT = Path(__file__).resolve().parent  # config.py lives at the repo root

DATA_DIR = ROOT / "data"        # dataset cache (VisA tarball + extraction)
MODELS_DIR = ROOT / "models"    # pretrained SAM weights (.pth), downloaded
RESULTS_DIR = ROOT / "results"  # generated outputs (masks, figures, metrics)
ASSETS_DIR = ROOT / "assets"    # curated figures committed for the README


# =============================================================================
# SAM checkpoint reference — the weight files Meta publishes; rarely edited.
# The active variant is chosen by SAM_VARIANT above.
# =============================================================================

SAM_CHECKPOINT_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
