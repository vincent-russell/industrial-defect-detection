# CLAUDE.md — project context

Context for AI-assisted development, and a quick orientation for anyone reading the repo.

## What this project is
Pixel-level segmentation of manufacturing defects on the **VisA** benchmark using Meta AI's
**Segment Anything Model (SAM)**, compared against a simple anomaly-detection baseline.
Goal: a clean, reproducible, well-documented computer-vision pipeline — not state-of-the-art
performance.

## How development works
Everything runs locally on a machine with an NVIDIA GPU.
- **Write logic in `src/`** as pure Python modules.
- **Iterate in `scripts/`:** exploratory work uses `# %%`-celled scripts, run cell-by-cell
  in VS Code's Jupyter Interactive Window (they import from `src/`).
- **Run SAM on the local GPU:** inference uses CUDA directly. Pick the SAM variant to fit
  available VRAM (`vit_b`/`vit_l` are lighter than `vit_h`).
- **Storage:** code → GitHub; dataset (~16 GB), model weights, and bulky outputs → local
  gitignored dirs (`data/`, `checkpoints/`, `results/`); curated showcase figures → `assets/`.

## Conventions
- `src/*.py` is **pure Python** — no `!`/`%` notebook magics.
- Large or regenerated artifacts (dataset, model weights, results) are gitignored.
- Keep modules small and readable. Commits are authored by Vincent Russell, small and clear.

## Folder structure
```
src/          pure-Python modules (logic)
scripts/      # %% interactive dev scripts (import from src/)
assets/       curated figures committed for the README
checkpoints/  SAM model weights (gitignored)
data/         dataset cache (gitignored)
results/      generated outputs (gitignored)
```

## Build order (incremental — one piece at a time)
1. [x] Repository setup (README, .gitignore, MIT license, first push)
2. [x] Folder structure + workflow
3. [x] VisA download + loader (verify the live source URL before hardcoding; dataset ~16 GB)
4. [ ] Single end-to-end SAM inference example (local GPU)
5. [ ] Simple baseline + evaluation (IoU)

## Dataset
VisA (Zou et al., ECCV 2022) — https://github.com/amazon-science/spot-diff
