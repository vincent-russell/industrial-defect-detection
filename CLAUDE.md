# CLAUDE.md — project context

Context for AI-assisted development, and a quick orientation for anyone reading the repo.

## What this project is
Pixel-level detection of manufacturing defects on the **VisA** benchmark using
**Student–Teacher Feature-Pyramid Matching (STFPM)** — a frozen ImageNet teacher backbone
and a same-architecture student trained on *normal* images only; anomalies show up as
places the student fails to match the teacher across a feature pyramid. Goal: a clean,
reproducible, documented pipeline, not state-of-the-art numbers.

## How development works
Everything runs locally on a machine with an NVIDIA GPU.
- **Write logic in `src/`** as pure Python modules (functions and classes).
- **Run everything from `main.py`:** the single entry point imports from `src/`
  and wires the pipeline together. Run it with F5 / `python main.py`, or
  highlight a block and "Run Selection" to iterate.
- **Parameters live in `config.py`** at the repo root — a flat module of plain,
  editable values (constants), *not* classes or functions and *not* YAML. It is
  the one obvious place to change a run; `main.py` and `src/` read from it.
- **Train and run on the local GPU:** training and inference use CUDA directly. Pick the
  backbone to fit available VRAM (`resnet18`/`resnet34` are lighter than `wide_resnet50_2`).
- **Storage:** code → GitHub; dataset (~16 GB), model weights, and bulky outputs → local
  gitignored dirs (`data/`, `models/`, `results/`); curated showcase figures → `assets/`.

## Conventions
- `src/*.py` is **pure Python** — no `!`/`%` notebook magics.
- **Docstrings: Google style, always** — every function and class has a docstring
  with an imperative one-line summary and `Args:` / `Returns:` / `Raises:`
  (or `Attributes:`) sections, e.g. `category: the object category`. Types live
  in the signature annotations only — never repeat them in the docstring; the
  docstring describes meaning (semantics, shapes, units, valid values). Keep
  this consistent across the whole codebase.
- Large or regenerated artifacts (dataset, model weights, results) are gitignored.
- Keep modules small and readable. Commits are authored by Vincent Russell, small and clear.

## Folder structure
```
main.py       entry point (wires the pipeline)
config.py     editable run parameters (flat constants)
src/          pure-Python modules (logic)
assets/       curated figures committed for the README
models/       trained student weights (gitignored)
data/         dataset cache (gitignored)
results/      generated outputs (gitignored)
```

## Build order (incremental — one piece at a time)
1. [x] Repository setup (README, .gitignore, MIT license, first push)
2. [x] Folder structure + workflow
3. [x] VisA download + loader (verify the live source URL before hardcoding; dataset ~16 GB)
4. [x] STFPM model + training on normal images (local GPU)
5. [x] Evaluation (image/pixel ROC AUC, IoU)
6. [ ] Result figures and qualitative examples

## Dataset
VisA (Zou et al., ECCV 2022) — https://github.com/amazon-science/spot-diff
Training uses the official one-class split (`1cls`): normal-only in train, normal +
anomalous in test. STFPM: Wang et al., BMVC 2021.
