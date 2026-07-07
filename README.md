# Industrial Defect Detection with Segment Anything (SAM)

Pixel-level segmentation of manufacturing defects on the **VisA** benchmark, using
Meta AI's **Segment Anything Model (SAM)** — compared against a simple anomaly-detection
baseline.

> 🚧 **Status:** work in progress. This repo is being built step by step as a clean,
> well-documented example of a modern computer-vision workflow.

## Overview

Industrial inspection asks a simple question: *given a photo of a part, which pixels
(if any) are defective?* This project explores answering it with a vision foundation
model (SAM) rather than a model trained from scratch, and measures how that compares to
a straightforward baseline.

The emphasis is on a **clean, reproducible, well-explained pipeline** — not on
state-of-the-art numbers.

## Dataset

[**VisA (Visual Anomaly)**](https://github.com/amazon-science/spot-diff) — ~10,800 images
across 12 object categories, with normal and defective samples and pixel-level
ground-truth masks for the defects.

> Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
> and Segmentation*, ECCV 2022.

## Project structure

```text
industrial-defect-detection/
├── src/          # Python modules (pure Python): config, data, SAM, baseline, evaluation
├── scripts/      # interactive dev scripts (# %% cells, VS Code Jupyter window)
├── assets/       # curated figures committed for this README
├── checkpoints/  # SAM model weights (gitignored)
├── data/         # dataset cache (gitignored)
└── results/      # generated outputs (gitignored)
```

Everything runs locally in VS Code, with SAM inference on a local NVIDIA GPU.
See [CLAUDE.md](CLAUDE.md) for the full workflow.

## Roadmap

- [x] Repository setup
- [ ] Download and load the VisA dataset
- [ ] Single end-to-end SAM inference example
- [ ] Simple baseline for comparison
- [ ] Evaluation (IoU and friends)

## Tech

Python · PyTorch (CUDA) · Segment Anything (SAM)

## License

[MIT](LICENSE)
