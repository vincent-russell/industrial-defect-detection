"""Sweep the STFPM pipeline over every VisA category and aggregate the results.

Published VisA numbers are the mean over all 12 categories, so one model is
trained and scored per category and the metrics are averaged. Each iteration
sets `config.CATEGORY` (and the derived weights path) and calls the existing
per-category `train`, `evaluate`, and `visualize` code; a summary file collects
the per-category rows and their category means.
"""

import json
from pathlib import Path

import config
from src import data, evaluate, train, visualize

# Metrics carried from each category's evaluation into the summary.
_METRIC_KEYS = ("image_auroc", "pixel_auroc", "best_iou", "iou_threshold")
# Metrics averaged across categories.
_MEAN_KEYS = ("image_auroc", "pixel_auroc", "best_iou")


def _select_category(category: str) -> None:
    """Point the shared config at one category for the next train/eval calls.

    Mutates the module-level `config` so functions that read `config.CATEGORY`
    and `config.STUDENT_WEIGHTS` at call time act on `category`. The weights
    path is re-derived from the current `config.MODELS_DIR` and
    `config.BACKBONE`.

    Args:
        category (str): VisA category to switch to; must be in `data.CATEGORIES`.
    """
    config.CATEGORY = category
    config.STUDENT_WEIGHTS = (
        config.MODELS_DIR / f"stfpm_{config.BACKBONE}_{category}.pth"
    )


def run_category(category: str, figures: bool = True) -> dict[str, float]:
    """Train (if needed) and evaluate one category, returning its metrics.

    Trains the student only when no saved weights exist (also saving the
    training curves), then evaluates on the test split.

    Args:
        category (str): VisA category to run; must be in `data.CATEGORIES`.
        figures (bool): If True, also render the qualitative example panel.

    Returns:
        dict[str, float]: Metrics with keys ``image_auroc``, ``pixel_auroc``,
            ``best_iou``, and ``iou_threshold``.
    """
    _select_category(category)
    print(f"\n=== {category} ===")

    if config.STUDENT_WEIGHTS.exists():
        print(f"Using existing student weights: {config.STUDENT_WEIGHTS}")
    else:
        history = train.train()
        visualize.save_training_curves(history)

    results = evaluate.evaluate()
    if figures:
        visualize.save_examples()
    return results


def _save_summary(rows: list[dict], path: Path | None = None) -> Path:
    """Write the per-category rows and their category means to one JSON file.

    The payload records the run configuration, one row per category, and the
    ``mean`` over categories of image/pixel AUROC and best IoU.

    Args:
        rows (list[dict]): Per-category results as built by `sweep`, each with
            a ``category`` key plus the metric keys.
        path (Path | None): Destination file, or None for the default
            ``summary_<backbone>.json`` under `config.RESULTS_DIR`.

    Returns:
        Path: The path written to.
    """
    if path is None:
        path = config.RESULTS_DIR / f"summary_{config.BACKBONE}.json"
    mean = {
        key: sum(row[key] for row in rows) / len(rows) for key in _MEAN_KEYS
    }
    payload = {
        "backbone": config.BACKBONE,
        "feature_layers": list(config.FEATURE_LAYERS),
        "img_size": config.IMG_SIZE,
        "epochs": config.EPOCHS,
        "num_categories": len(rows),
        "categories": rows,
        "mean": mean,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _print_table(rows: list[dict], mean: dict[str, float]) -> None:
    """Print a table of per-category metrics and their mean.

    Args:
        rows (list[dict]): Per-category results, each with a ``category`` key
            plus the metric keys.
        mean (dict[str, float]): Category means keyed by `_MEAN_KEYS`.
    """
    print(f"\n{'category':<12}{'image AUROC':>14}{'pixel AUROC':>14}{'best IoU':>12}")
    print("-" * 52)
    for row in rows:
        print(
            f"{row['category']:<12}{row['image_auroc']:>14.4f}"
            f"{row['pixel_auroc']:>14.4f}{row['best_iou']:>12.4f}"
        )
    print("-" * 52)
    print(
        f"{'MEAN':<12}{mean['image_auroc']:>14.4f}"
        f"{mean['pixel_auroc']:>14.4f}{mean['best_iou']:>12.4f}"
    )


def sweep(
    categories: tuple[str, ...] = data.CATEGORIES, figures: bool = True
) -> dict:
    """Run every category end to end and aggregate the category-mean metrics.

    Trains (if needed) and evaluates each category, then writes a single
    summary file and prints a table of the results.

    Args:
        categories (tuple[str, ...]): Categories to sweep, defaulting to all 12.
        figures (bool): If True, also render each category's example panel.

    Returns:
        dict: The summary payload as written by `_save_summary`, including the
            per-category ``categories`` rows and the ``mean`` over categories.
    """
    rows: list[dict] = []
    for category in categories:
        results = run_category(category, figures=figures)
        rows.append({"category": category, **{k: results[k] for k in _METRIC_KEYS}})

    path = _save_summary(rows)
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    _print_table(rows, summary["mean"])
    print(f"\nSaved summary -> {path}")
    return summary
