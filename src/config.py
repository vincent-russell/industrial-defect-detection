"""Central paths and environment configuration.

Keeps every filesystem path in one place. Everything runs locally under the
repo: the dataset cache, model checkpoints, and generated results all live in
gitignored directories at the repo root.

Pure Python — safe to import from `scripts/` and any entry point.
"""

from pathlib import Path

# Repo root. This file lives at <root>/src/config.py, so go up two levels.
ROOT = Path(__file__).resolve().parents[1]

# Local, gitignored working directories.
DATA_DIR = ROOT / "data"              # dataset cache (VisA tarball + extraction)
CHECKPOINTS_DIR = ROOT / "checkpoints"  # SAM model weights
RESULTS_DIR = ROOT / "results"        # generated outputs (masks, figures, metrics)

# Curated figures committed to the repo (for the README) live here.
ASSETS_DIR = ROOT / "assets"


def ensure_dirs() -> None:
    """Create the working directories if they don't exist yet."""
    for d in (DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
