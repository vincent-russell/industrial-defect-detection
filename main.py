"""Entry point for the defect-detection pipeline.

Run the whole thing from here — press F5 in VS Code (or `python main.py`), or
highlight a block and "Run Selection" to iterate interactively. All logic lives
in `src/`; this file only wires it together.

Parameters live in `config.py` at the repo root. Edit the values there to change
what a run does — this file reads them, it does not define them.
"""

from __future__ import annotations

from src import data, sweep


def main() -> None:
    """Run the STFPM pipeline over all 12 VisA categories and aggregate.

    Ensures VisA is available, then sweeps every category: for each one it trains
    the student on normal images (unless weights already exist), evaluates on the
    test split, and renders a qualitative example panel — all under `results/`.
    Finally it writes a `summary_<backbone>.json` with the per-category metrics
    and their category-mean image/pixel AUROC, the numbers to compare against the
    published VisA leaderboard. Delete a category's saved weights to force it to
    retrain; to run a single category instead, call `sweep.run_category(name)`.
    """
    data.download_visa()
    sweep.sweep()


if __name__ == "__main__":
    main()
