"""Entry point for the defect-detection pipeline.

Run the whole thing from here — press F5 in VS Code (or `python main.py`), or
highlight a block and "Run Selection" to iterate interactively. All logic lives
in `src/`; this file only wires it together.

Parameters live in `config.py` at the repo root. Edit the values there to change
what a run does — this file reads them, it does not define them.
"""

from __future__ import annotations

import config
from src import data, sweep


def main() -> None:
    """Run the STFPM pipeline for the category selected in `config.py`.

    Ensures VisA is available, then dispatches on `config.CATEGORY`. A category
    name (e.g. "cashew") runs that one category: it trains the student on its
    normal images (unless weights already exist), evaluates on the test split,
    and renders a qualitative example panel — all under `results/`. The sentinel
    "all" does the same for every category and additionally writes a
    `summary_<backbone>.json` with the per-category metrics and their
    category-mean image/pixel AUROC, the numbers to compare against the published
    VisA leaderboard. Delete a category's saved weights to force it to retrain.
    """
    data.download_visa()
    if config.CATEGORY == "all":
        sweep.sweep()
    else:
        sweep.run_category(config.CATEGORY)


if __name__ == "__main__":
    main()
