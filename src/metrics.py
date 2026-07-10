"""Detection and segmentation metrics, computed in numpy.

ROC AUC (image- and pixel-level scoring) and the best achievable IoU over a
threshold sweep.
"""

from __future__ import annotations

import numpy as np


def _average_ranks(values_sorted: np.ndarray) -> np.ndarray:
    """Return average (tie-aware) ranks for an already-sorted array.

    Args:
        values_sorted (np.ndarray): 1-D array sorted in ascending order.

    Returns:
        np.ndarray: 1-based ranks, with ties assigned their group's mean rank.
    """
    _, inverse, counts = np.unique(
        values_sorted, return_inverse=True, return_counts=True
    )
    ends = np.cumsum(counts)          # last 1-based rank of each tie group
    starts = ends - counts            # count of elements before each group
    group_avg = (starts + ends + 1) / 2.0
    return group_avg[inverse]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC via the Mann-Whitney rank-sum identity.

    Args:
        labels (np.ndarray): Binary labels (True/1 for the positive class).
        scores (np.ndarray): Real-valued scores; higher should mean positive.

    Returns:
        float: Area under the ROC curve, or NaN if only one class is present.
    """
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = _average_ranks(scores[order])
    pos_rank_sum = ranks[labels[order]].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_iou(
    labels: np.ndarray, scores: np.ndarray, num_thresholds: int = 100
) -> tuple[float, float]:
    """Sweep thresholds and return the best IoU and the threshold that hit it.

    Args:
        labels (np.ndarray): Binary ground-truth mask, flattened over all pixels.
        scores (np.ndarray): Anomaly scores aligned with `labels`.
        num_thresholds (int): Number of thresholds sampled across the score range.

    Returns:
        tuple[float, float]: The best IoU and its threshold. IoU is NaN if there
            are no positive ground-truth pixels.
    """
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if not labels.any():
        return float("nan"), float("nan")

    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    gt_area = int(labels.sum())
    best, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        pred = scores >= thr
        intersection = int(np.logical_and(pred, labels).sum())
        union = int(pred.sum()) + gt_area - intersection
        iou = intersection / union if union else 0.0
        if iou > best:
            best, best_thr = iou, float(thr)
    return best, best_thr
