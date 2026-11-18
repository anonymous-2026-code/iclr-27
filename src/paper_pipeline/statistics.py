from __future__ import annotations

import dataclasses
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclasses.dataclass(frozen=True)
class BootstrapEstimate:
    estimand: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    replicates: int
    bootstrap_seed: int
    cluster_unit: str
    n_clusters: int
    n_valid: int
    n_total: int
    undefined_reason: str | None


def _finite_quantiles(values: Iterable[float], confidence: float) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


def dataset_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    estimand: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    cluster_column: str = "dataset",
) -> BootstrapEstimate:
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("Bootstrap confidence must lie in (0,1)")
    if cluster_column not in frame:
        raise ValueError(f"Missing cluster column {cluster_column!r}")
    clusters = tuple(sorted(str(item) for item in frame[cluster_column].dropna().unique()))
    if not clusters:
        return BootstrapEstimate(
            estimand=estimand,
            estimate=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence=confidence,
            replicates=replicates,
            bootstrap_seed=seed,
            cluster_unit=cluster_column,
            n_clusters=0,
            n_valid=0,
            n_total=len(frame),
            undefined_reason="no-dataset-clusters",
        )
    grouped = {
        cluster: frame.loc[frame[cluster_column].astype(str) == cluster].copy()
        for cluster in clusters
    }
    estimate = float(statistic(frame))
    generator = np.random.default_rng(seed)
    sampled_values: list[float] = []
    for _ in range(replicates):
        sampled = generator.integers(0, len(clusters), size=len(clusters))
        pieces = []
        for draw_index, cluster_index in enumerate(sampled):
            piece = grouped[clusters[int(cluster_index)]].copy()
            piece["__bootstrap_cluster_instance"] = draw_index
            pieces.append(piece)
        sampled_values.append(float(statistic(pd.concat(pieces, ignore_index=True))))
    lower, upper = _finite_quantiles(sampled_values, confidence)
    finite_replicates = int(np.isfinite(sampled_values).sum())
    reason = None
    if not math.isfinite(estimate):
        reason = "point-estimate-undefined"
    elif finite_replicates == 0:
        reason = "all-bootstrap-replicates-undefined"
    return BootstrapEstimate(
        estimand=estimand,
        estimate=estimate,
        ci_lower=lower,
        ci_upper=upper,
        confidence=confidence,
        replicates=replicates,
        bootstrap_seed=seed,
        cluster_unit=cluster_column,
        n_clusters=len(clusters),
        n_valid=finite_replicates,
        n_total=replicates,
        undefined_reason=reason,
    )


def category_proportion(category: str) -> Callable[[pd.DataFrame], float]:
    def statistic(frame: pd.DataFrame) -> float:
        values = frame["category"].dropna().astype(str)
        return float((values == category).mean()) if len(values) else float("nan")

    return statistic


def spearman_statistic(x_column: str, y_column: str) -> Callable[[pd.DataFrame], float]:
    def statistic(frame: pd.DataFrame) -> float:
        values = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 3:
            return float("nan")
        if values[x_column].nunique() < 2 or values[y_column].nunique() < 2:
            return float("nan")
        result = spearmanr(values[x_column], values[y_column])
        return float(result.statistic)

    return statistic


def estimates_frame(estimates: Iterable[BootstrapEstimate]) -> pd.DataFrame:
    return pd.DataFrame(dataclasses.asdict(item) for item in estimates)
