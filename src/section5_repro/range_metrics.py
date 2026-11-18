from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .metrics import average_precision


METRICS = ("auprc", "auroc", "vus_pr", "vus_roc")


def align_available(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-loader support policy: common prefix and joint finite filtering."""
    label = np.asarray(labels).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    length = min(len(label), len(score))
    label = label[:length]
    score = score[:length]
    valid = np.isfinite(label) & np.isfinite(score)
    label = (label[valid] != 0).astype(np.int8)
    score = score[valid]
    return label, score


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    label, score = align_available(labels, scores)
    if not len(label) or np.unique(label).size < 2:
        return float("nan")
    positives = float(np.sum(label == 1))
    negatives = float(np.sum(label == 0))
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    positive_rank_sum = float(np.sum(full_ranks[label == 1]))
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1.0) / 2.0
        )
        / (positives * negatives)
    )


def _prepare_desc_score_context(score: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    sorted_scores = score[order]
    threshold_indices = np.linspace(0, len(score) - 1, 250).astype(int)
    thresholds = sorted_scores[threshold_indices]
    pred_counts = np.searchsorted(
        -sorted_scores, -thresholds, side="right"
    ).astype(np.float64)
    return {
        "order": order,
        "thresholds": thresholds,
        "pred_counts": pred_counts,
        "pred_counts_int": pred_counts.astype(np.int64),
    }


def _positive_ranges_from_label(
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(label).reshape(-1) != 0
    if mask.size == 0 or not bool(np.any(mask)):
        empty = np.array([], dtype=np.int64)
        return empty, empty
    padded = np.empty(mask.size + 2, dtype=np.bool_)
    padded[0] = False
    padded[1:-1] = mask
    padded[-1] = False
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1).astype(np.int64)
    ends = (np.flatnonzero(changes == -1) - 1).astype(np.int64)
    return starts, ends


def _merge_sorted_ranges(
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[int, int]]:
    if len(starts) == 0:
        return []
    merged: list[tuple[int, int]] = []
    current_start = int(starts[0])
    current_end = int(ends[0])
    for raw_start, raw_end in zip(starts[1:], ends[1:]):
        start, end = int(raw_start), int(raw_end)
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _extended_label_and_ranges(
    labels_original: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    window: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    labels = labels_original.astype(np.float64, copy=True)
    length = len(labels)
    half_window = int(window) // 2
    if int(window) > 0 and half_window > 0 and len(starts) > 0:
        for offset in range(1, half_window + 1):
            weight = math.sqrt(1.0 - float(offset) / float(window))
            positions = starts - offset
            positions = positions[positions >= 0]
            if len(positions):
                np.add.at(labels, positions, weight)
        for offset in range(1, half_window):
            weight = math.sqrt(1.0 - float(offset) / float(window))
            positions = ends + offset
            positions = positions[positions < length]
            if len(positions):
                np.add.at(labels, positions, weight)
        np.minimum(labels, 1.0, out=labels)
    if half_window > 0:
        support_starts = np.maximum(starts - half_window, 0)
        support_ends = np.minimum(ends + half_window - 1, length - 1)
    else:
        support_starts, support_ends = starts, ends
    return labels, _merge_sorted_ranges(support_starts, support_ends)


def range_auc_volume(
    labels_original: np.ndarray,
    scores: np.ndarray,
    window_size: int = 12,
    score_context: Optional[dict[str, np.ndarray]] = None,
) -> tuple[float, float]:
    """Return VUS-ROC and VUS-PR using the frozen 250-threshold protocol."""
    labels, score = align_available(labels_original, scores)
    if not len(labels) or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    context = score_context or _prepare_desc_score_context(score)
    order = context["order"]
    thresholds = context["thresholds"]
    pred_counts = context["pred_counts"]
    pred_counts_int = context["pred_counts_int"]
    range_starts, range_ends = _positive_ranges_from_label(labels)
    original_positives = float(np.sum(labels))
    auc_values: list[float] = []
    ap_values: list[float] = []
    for window in range(int(window_size) + 1):
        extended, ranges = _extended_label_and_ranges(
            labels, range_starts, range_ends, window
        )
        if not ranges:
            auc_values.append(float("nan"))
            ap_values.append(float("nan"))
            continue
        positives = (original_positives + float(np.sum(extended))) / 2.0
        negatives = float(len(labels)) - positives
        if positives <= 0.0 or negatives <= 0.0:
            auc_values.append(float("nan"))
            ap_values.append(float("nan"))
            continue
        sorted_labels = extended[order]
        cumulative_true = np.cumsum(sorted_labels, dtype=np.float64)
        true_positive = cumulative_true[np.maximum(pred_counts_int - 1, 0)]
        true_positive = np.where(pred_counts_int > 0, true_positive, 0.0)
        segment_maximum = np.array(
            [np.max(score[start : end + 1]) for start, end in ranges],
            dtype=np.float64,
        )
        sorted_segment_maximum = np.sort(segment_maximum)[::-1]
        existence = np.searchsorted(
            -sorted_segment_maximum, -thresholds, side="right"
        ).astype(np.float64)
        existence_ratio = existence / float(len(ranges))
        recall = np.minimum(true_positive / positives, 1.0)
        true_positive_rate = recall * existence_ratio
        false_positive_rate = (pred_counts - true_positive) / negatives
        precision = np.divide(
            true_positive,
            pred_counts,
            out=np.zeros_like(true_positive),
            where=pred_counts != 0,
        )
        tpr_curve = np.concatenate(([0.0], true_positive_rate, [1.0]))
        fpr_curve = np.concatenate(([0.0], false_positive_rate, [1.0]))
        precision_curve = np.concatenate(([1.0], precision))
        auc_values.append(
            float(
                np.sum(
                    (fpr_curve[1:] - fpr_curve[:-1])
                    * ((tpr_curve[1:] + tpr_curve[:-1]) / 2.0)
                )
            )
        )
        ap_values.append(
            float(
                np.sum(
                    (tpr_curve[1:-1] - tpr_curve[:-2])
                    * ((precision_curve[1:] + precision_curve[:-1]) / 2.0)
                )
            )
        )
    return float(np.nanmean(auc_values)), float(np.nanmean(ap_values))


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    window_size: int = 12,
) -> dict[str, float]:
    label, score = align_available(labels, scores)
    if not len(label) or np.unique(label).size < 2:
        return {name: float("nan") for name in METRICS}
    vus_roc, vus_pr = range_auc_volume(label, score, window_size)
    return {
        "auprc": average_precision(label, score),
        "auroc": roc_auc(label, score),
        "vus_pr": vus_pr,
        "vus_roc": vus_roc,
    }
