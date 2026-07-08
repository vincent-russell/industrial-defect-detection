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
CATEGORY = "cashew"   # VisA object category (see CATEGORIES in src/data.py)

# Model — a frozen ImageNet teacher and a same-architecture student are compared
# across a feature pyramid (student-teacher feature-pyramid matching, STFPM).
BACKBONE = "resnet18"  # "resnet18" | "resnet34" | "wide_resnet50_2"
FEATURE_LAYERS = ("layer1", "layer2", "layer3")  # pyramid taps (shallow -> deep)
DEVICE = "cuda"        # "cuda" or "cpu"

# Training — the student learns to reproduce the teacher's features on *normal*
# images only. Defaults follow the STFPM paper (SGD, 256px, 100 epochs).
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Diagnostic: track image/pixel AUROC on the test split every N epochs during
# training (0 disables). Monitoring only — never used for model selection or
# early stopping, so it does not leak into the reported result.
EVAL_EVERY_EPOCHS = 10

# Evaluation — the anomaly map is smoothed before scoring; sigma in pixels at
# IMG_SIZE resolution (0 disables smoothing).
SMOOTH_SIGMA = 4.0

# Qualitative figures — how many test examples to render in the results panel
# (anomalies, evenly spaced across the test split, plus one normal for contrast).
NUM_FIGURE_EXAMPLES = 6

# Reproducibility
SEED = 0


# =============================================================================
# Project paths — derived from the repo root; rarely edited.
# =============================================================================

ROOT = Path(__file__).resolve().parent  # config.py lives at the repo root

DATA_DIR = ROOT / "data"        # dataset cache (VisA tarball + extraction)
MODELS_DIR = ROOT / "models"    # trained student weights (.pth)
RESULTS_DIR = ROOT / "results"  # generated outputs (maps, figures, metrics)
ASSETS_DIR = ROOT / "assets"    # curated figures committed for the README

# Where the trained student for the active backbone/category is saved and loaded.
STUDENT_WEIGHTS = MODELS_DIR / f"stfpm_{BACKBONE}_{CATEGORY}.pth"
