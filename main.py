"""Entry point for the defect-detection pipeline.

Run the whole thing from here — press F5 in VS Code (or `python main.py`), or
highlight a block and "Run Selection" to iterate interactively. All logic lives
in `src/`; this file only wires it together.

Parameters live in `config.py` at the repo root. Edit the values there to change
what a run does — this file reads them, it does not define them.
"""

from __future__ import annotations

import config
from src import data, evaluate, train, visualize


def main() -> None:
    """Run the STFPM pipeline end to end for the configured category.

    Ensures VisA is available, trains the student on normal images if it has not
    been trained yet (saving training curves to `results/`), evaluates it on the
    test split (saving metrics to `results/`), and renders a qualitative example
    panel there too. Delete the saved weights (or change
    `config.BACKBONE`/`config.CATEGORY`) to force retraining.
    """
    data.download_visa()

    if not config.STUDENT_WEIGHTS.exists():
        history = train.train()
        curves = visualize.save_training_curves(history)
        print(f"Saved training curves -> {curves}")
    else:
        print(f"Using existing student weights: {config.STUDENT_WEIGHTS}")

    evaluate.evaluate()

    figure = visualize.save_examples()
    print(f"Saved example figure -> {figure}")


if __name__ == "__main__":
    main()
