"""Entry point for the defect-detection pipeline.

Run with `python main.py`. Logic lives in `src/`; parameters in `config.py`.
"""

from __future__ import annotations

import config
from src import data, sweep


def main() -> None:
    """Run the STFPM pipeline for the category selected in `config.py`.

    Ensures VisA is available, then trains (unless saved weights exist),
    evaluates, and renders figures for `config.CATEGORY`. If the category is
    "all", every category is run and a summary with category-mean metrics is
    written alongside the per-category results.
    """
    data.download_visa()
    if config.CATEGORY == "all":
        sweep.sweep()
    else:
        sweep.run_category(config.CATEGORY)


if __name__ == "__main__":
    main()
