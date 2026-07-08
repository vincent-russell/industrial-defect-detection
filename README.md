# Industrial Defect Detection with Student–Teacher Feature Matching (STFPM)

Pixel-level detection of manufacturing defects on the **VisA** benchmark, using
**Student–Teacher Feature-Pyramid Matching (STFPM)**, trained on defect-free images.

> 🚧 **Status:** work in progress.

## Overview

The task: given a photo of a part, identify which pixels (if any) are defective. Defects
are rare and varied, so the approach is **anomaly detection** — model normal appearance
and flag deviations.

A frozen ImageNet backbone (the *teacher*) defines a reference feature space. A second
backbone of the same architecture (the *student*) is trained to reproduce the teacher's
features on normal images only. At test time, pixels where the student fails to match the
teacher — across several feature-pyramid levels — are scored as anomalous. Defect masks
are used for evaluation only, never for training.

## Method

- **Teacher / student:** identical backbones (default ResNet-18); the teacher is frozen
  with ImageNet weights, the student is trained from scratch.
- **Objective:** on normal images, minimise the distance between L2-normalised teacher
  and student feature maps at each pyramid level.
- **Anomaly map:** per-level teacher–student discrepancies are upsampled and multiplied,
  so a pixel scores high only where multiple scales agree — small defects surface at
  shallow levels, structural ones at deep levels.
- **Metrics:** image-level ROC AUC, pixel-level ROC AUC, and best-achievable IoU.

> Wang et al., *Student-Teacher Feature Pyramid Matching for Anomaly Detection*, BMVC 2021.

## Dataset

[**VisA (Visual Anomaly)**](https://github.com/amazon-science/spot-diff) — ~10,800 images
across 12 object categories, with normal and defective samples and pixel-level
ground-truth masks for the defects. Training uses the official one-class split
(`1cls`): **normal images only** in train, normal + anomalous in test.

> Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
> and Segmentation*, ECCV 2022.

## Project structure

```text
industrial-defect-detection/
├── main.py       # entry point: download → train → evaluate
├── config.py     # editable run parameters (flat constants)
├── src/          # Python modules: data, model, train, evaluate, metrics
├── assets/       # curated figures committed for this README
├── models/       # trained student weights (gitignored)
├── data/         # dataset cache (gitignored)
└── results/      # generated outputs (gitignored)
```

Everything runs locally in VS Code, with training and inference on a local NVIDIA GPU.
Run the pipeline from `main.py` (F5, or `python main.py`); all parameters live in
`config.py`. See [CLAUDE.md](CLAUDE.md) for the full workflow.

## Roadmap

- [x] Repository setup
- [x] Download and load the VisA dataset
- [x] Train STFPM on normal images (one category)
- [x] Evaluation (image/pixel ROC AUC, IoU)
- [ ] Result figures and qualitative examples

## Tech

Python · PyTorch (CUDA) · torchvision

## License

[MIT](LICENSE)
