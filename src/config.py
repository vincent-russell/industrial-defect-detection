"""Central paths and environment configuration.

Keeps every filesystem path in one place and adapts to where the code runs:
locally (under the repo) or on Google Colab (persisting big files to Drive,
since Colab's local disk is wiped each session).

Pure Python — safe to import from both `scripts/` and the Colab notebook.
"""

from pathlib import Path

# Repo root. This file lives at <root>/src/config.py, so go up two levels.
ROOT = Path(__file__).resolve().parents[1]


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # type: ignore # noqa: F401  (only present on Colab)

        return True
    except ImportError:
        return False


# On Colab, cache the dataset and results to Google Drive so they survive
# session resets. Locally, keep them under the repo (both are gitignored).
if in_colab():
    DRIVE = Path("/content/drive/MyDrive/industrial-defect-detection")
    DATA_DIR = DRIVE / "data"
    RESULTS_DIR = DRIVE / "results"
else:
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"

# Curated figures committed to the repo (for the README) live here.
ASSETS_DIR = ROOT / "assets"


def ensure_dirs() -> None:
    """Create the data/results/assets directories if they don't exist yet."""
    for d in (DATA_DIR, RESULTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
