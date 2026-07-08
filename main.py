"""Entry point for the defect-detection pipeline.

Run the whole thing from here — press F5 in VS Code (or `python main.py`), or
highlight a block and "Run Selection" to iterate interactively. All logic lives
in `src/`; this file only wires it together and sets the run's parameters.

Parameters come from `src.config.Config`. Edit `CFG` below (or construct it with
overrides, e.g. `Config(sam_variant="vit_l", category="pcb1")`) to change a run.
"""

from __future__ import annotations

from src import config, data
from src.config import Config

# Parameters for this run — the one place to change what happens.
CFG = Config()


def main(cfg: Config = CFG) -> None:
    """Download VisA if needed, then load and summarise the selected split."""
    config.ensure_dirs()
    data.download_visa()

    samples = data.load_samples(category=cfg.category, split=cfg.split)
    anomalies = [s for s in samples if s.is_anomaly]
    print(
        f"{cfg.category} / {cfg.split}: {len(samples)} samples "
        f"({len(anomalies)} anomalous, {len(samples) - len(anomalies)} normal)"
    )

    # Peek at one anomalous sample and its ground-truth mask.
    if anomalies:
        sample = anomalies[0]
        image = data.load_image(sample)
        mask = data.load_mask(sample)
        print(f"Example: {sample.image_path.name}  image={image.shape}", end="")
        print(f"  mask={mask.shape}  defect_px={int(mask.sum())}" if mask is not None else "")

    # Next: SAM inference + baseline + IoU evaluation (build order steps 4-5).


if __name__ == "__main__":
    main()
