# CLAUDE.md — project context

Context for AI-assisted development, and a quick orientation for anyone reading the repo.

## What this project is
Pixel-level segmentation of manufacturing defects on the **VisA** benchmark using Meta AI's
**Segment Anything Model (SAM)**, compared against a simple anomaly-detection baseline.
Goal: a clean, reproducible, well-documented computer-vision pipeline — not state-of-the-art
performance.

## How development works
- **Local (VS Code), no GPU:** write and iterate code. Real logic lives in `src/` as pure
  Python modules. Exploratory work uses `# %%`-celled scripts in `scripts/`, run cell-by-cell
  in VS Code's Jupyter Interactive Window (they import from `src/`).
- **Colab (GPU):** the only place SAM actually runs. A single thin notebook in `notebooks/`
  clones the repo, installs requirements, mounts Drive, and runs code from `src/`.
- **Storage:** code → GitHub; dataset (~16 GB) and bulky outputs → Google Drive (Colab's
  local disk is wiped each session); a few curated showcase figures → committed to `assets/`.

## Conventions
- `src/*.py` is **pure Python** — no `!`/`%` notebook magics (those live only in the notebook).
- Large or regenerated artifacts (dataset, model weights, results) are gitignored.
- Keep modules small and readable. Commits are authored by Vincent Russell, small and clear.

## Folder structure
```
src/         pure-Python modules (logic)
scripts/     # %% interactive dev scripts (import from src/)
notebooks/   thin Colab launcher (.ipynb)
assets/      curated figures committed for the README
data/        dataset cache (gitignored; on Colab → Google Drive)
results/     generated outputs (gitignored)
```

## Build order (incremental — one piece at a time)
1. [x] Repository setup (README, .gitignore, MIT license, first push)
2. [x] Folder structure + workflow
3. [ ] VisA download + loader (verify the live source URL before hardcoding; dataset ~16 GB)
4. [ ] Single end-to-end SAM inference example on Colab
5. [ ] Simple baseline + evaluation (IoU)

## Dataset
VisA (Zou et al., ECCV 2022) — https://github.com/amazon-science/spot-diff
