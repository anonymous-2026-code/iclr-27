from __future__ import annotations

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Stable AP implementation used by the paper exporter."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s):
        raise ValueError(f"Score/label length mismatch: {len(s)} != {len(y)}")
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    true_positive = np.cumsum(sorted_labels == 1, dtype=np.float64)
    false_positive = np.cumsum(sorted_labels == 0, dtype=np.float64)
    positives = float(np.sum(sorted_labels == 1))
    if positives <= 0:
        return float("nan")
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def binary_prediction_threshold(
    source_scores: np.ndarray,
    test_scores: np.ndarray,
    anomaly_ratio_percent: float,
) -> float:
    combined = np.concatenate(
        [
            np.asarray(source_scores, dtype=np.float64).reshape(-1),
            np.asarray(test_scores, dtype=np.float64).reshape(-1),
        ]
    )
    return float(np.percentile(combined, 100.0 - anomaly_ratio_percent))


def anomaly_ratio_for_dataset(dataset: str) -> float:
    # This affects only the auxiliary binary predictions. Section 5 uses AUPRC.
    return 0.5 if dataset == "SMD" else 1.0

