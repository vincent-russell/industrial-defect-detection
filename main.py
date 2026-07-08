"""Entry point for the defect-detection pipeline.

Run the whole thing from here — press F5 in VS Code (or `python main.py`), or
highlight a block and "Run Selection" to iterate interactively. All logic lives
in `src/`; this file only wires it together.

Parameters live in `config.py` at the repo root. Edit the values there to change
what a run does — this file reads them, it does not define them.
"""

from __future__ import annotations

import config
from src import data


def main() -> None:
    """Download VisA if needed, then load and summarise the selected split.

    Reads `config.CATEGORY` and `config.SPLIT` to choose what to load, prints a
    normal/anomaly breakdown, and peeks at one anomalous sample with its mask.
    """
    data.download_visa()

    samples = data.load_samples(category=config.CATEGORY, split=config.SPLIT)
    anomalies = [s for s in samples if s.is_anomaly]
    print(
        f"{config.CATEGORY} / {config.SPLIT}: {len(samples)} samples "
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
