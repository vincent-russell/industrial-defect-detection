"""Entry point for the defect-detection pipeline.

Run with `python main.py`. Logic lives in `src/`; parameters in `config.py`.
This module wires the pipeline together: it runs one category end to end, or
sweeps every VisA category and aggregates the results. Published VisA numbers
are the mean over all 12 categories, so the sweep trains and scores one model
per category and averages the metrics.
"""

import config
from src import data, evaluate, figures, train


def _select_category(category: str) -> None:
    """Point the shared config at one category for the next train/eval calls.

    Mutates the module-level `config` so functions that read `config.CATEGORY`
    and `config.STUDENT_WEIGHTS` at call time act on `category`. The weights
    path is re-derived from the current `config.MODELS_DIR` and
    `config.BACKBONE`.

    Args:
        category: VisA category to switch to; must be in `data.CATEGORIES`.
    """
    config.CATEGORY = category
    config.STUDENT_WEIGHTS = (
        config.MODELS_DIR / f"stfpm_{config.BACKBONE}_{category}.pth"
    )


def run_category(category: str, make_figures: bool = True) -> dict[str, float]:
    """Train (if needed) and evaluate one category, returning its metrics.

    Trains the student only when no saved weights exist (also saving the
    training curves), then evaluates on the test split.

    Args:
        category: VisA category to run; must be in `data.CATEGORIES`.
        make_figures: If True, also render the qualitative example panel.

    Returns:
        Metrics with keys ``image_auroc``, ``pixel_auroc``, ``best_iou``,
        and ``iou_threshold``.
    """
    _select_category(category)
    print(f"\n=== {category} ===")

    if config.STUDENT_WEIGHTS.exists():
        print(f"Using existing student weights: {config.STUDENT_WEIGHTS}")
    else:
        history = train.train()
        figures.save_training_curves(history)

    results = evaluate.evaluate()
    if make_figures:
        figures.save_examples()
    return results


def sweep(
    categories: tuple[str, ...] = data.CATEGORIES, make_figures: bool = True
) -> dict:
    """Run every category end to end and aggregate the category-mean metrics.

    Trains (if needed) and evaluates each category, then writes a single
    summary file and prints a table of the results.

    Args:
        categories: Categories to sweep, defaulting to all 12.
        make_figures: If True, also render each category's example panel.

    Returns:
        The summary payload as written by `evaluate.save_summary`, including
        the per-category ``categories`` rows and the ``mean`` over categories.
    """
    rows = [
        {"category": category, **run_category(category, make_figures=make_figures)}
        for category in categories
    ]
    summary = evaluate.save_summary(rows)
    evaluate.print_summary(summary)
    if make_figures:
        figures.save_summary_figure(summary)
    return summary


def main() -> None:
    """Run the STFPM pipeline for the category selected in `config.py`.

    Ensures VisA is available, then trains (unless saved weights exist),
    evaluates, and renders figures for `config.CATEGORY`. If the category is
    "all", every category is run and a summary with category-mean metrics is
    written alongside the per-category results.
    """
    data.download_visa()
    if config.CATEGORY == "all":
        sweep()
    else:
        run_category(config.CATEGORY)


if __name__ == "__main__":
    main()
