"""Run parameters, as plain module-level constants."""

from pathlib import Path

# =============================================================================
# Run parameters
# =============================================================================

# One VisA object category, or "all" to sweep every category and report the
# category mean. Categories: candle, capsules, cashew, chewinggum, fryum,
# macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum.
CATEGORY = "all"

# Model
BACKBONE = "resnet18"  # "resnet18" | "resnet34" | "wide_resnet50_2"
FEATURE_LAYERS = ("layer1", "layer2", "layer3")  # pyramid taps (shallow -> deep)
DEVICE = "cuda"        # "cuda" or "cpu"

# Training. Defaults follow the STFPM paper (SGD, 256px, 100 epochs).
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Track image/pixel AUROC on the test split every N epochs during training
# (0 disables). Monitoring only — never used for model selection.
EVAL_EVERY_EPOCHS = 10

# Gaussian smoothing of the anomaly map before scoring; sigma in pixels at
# IMG_SIZE resolution (0 disables).
SMOOTH_SIGMA = 4.0

# Number of test examples rendered in the qualitative figure.
NUM_FIGURE_EXAMPLES = 6

# Reproducibility
SEED = 0


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"        # dataset cache
MODELS_DIR = ROOT / "models"    # trained student weights
RESULTS_DIR = ROOT / "results"  # generated outputs (figures, metrics)
ASSETS_DIR = ROOT / "assets"    # curated figures committed for the README

# Trained student weights for the active backbone/category.
STUDENT_WEIGHTS = MODELS_DIR / f"stfpm_{BACKBONE}_{CATEGORY}.pth"
