from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MaskDecision:
    mask: torch.Tensor
    diagnostics: Mapping[str, Any]


class Selector:
    uses_labels = False

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        raise NotImplementedError

    def observe_after_decision(self, scores: torch.Tensor) -> None:
        return None


class StaticThresholdSelector(Selector):
    def __init__(self, threshold: float, percentile: float) -> None:
        self.threshold = float(threshold)
        self.percentile = float(percentile)

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        _require_finite(scores)
        return MaskDecision(
            scores < self.threshold,
            {"threshold": self.threshold, "percentile": self.percentile},
        )


class OraclePointSelector(Selector):
    uses_labels = True

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        if labels.shape != scores.shape:
            raise ValueError(
                f"Oracle label/score mismatch: {tuple(labels.shape)} != "
                f"{tuple(scores.shape)}"
            )
        return MaskDecision(labels == 0, {"oracle": "benchmark_normal_point"})


class ExternalPointMaskSelector(Selector):
    """Use a prespecified global point mask without changing evaluation labels."""

    def __init__(self, mask: np.ndarray, window: int, stride: int) -> None:
        self.global_mask = np.asarray(mask, dtype=np.int8).reshape(-1)
        self.window = int(window)
        self.stride = int(stride)

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        start = int(window_index) * self.stride
        stop = start + self.window
        selected = self.global_mask[start:stop]
        if selected.shape != (scores.shape[-1],):
            raise ValueError(
                f"External mask ended before window {window_index}: "
                f"slice={selected.shape}, scores={tuple(scores.shape)}"
            )
        mask = torch.as_tensor(selected, device=scores.device, dtype=torch.bool)
        return MaskDecision(
            mask.unsqueeze(0).expand_as(scores),
            {"global_start": start, "global_stop": stop},
        )


class BottomKSelector(Selector):
    def __init__(self, k: int = 11, expected_window: int = 12) -> None:
        if not 1 <= int(k) <= int(expected_window):
            raise ValueError("Bottom-k requires 1 <= k <= window")
        self.k = int(k)
        self.expected_window = int(expected_window)

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        _require_finite(scores)
        if scores.ndim != 2 or scores.shape[1] != self.expected_window:
            raise ValueError(
                f"Bottom-k expects [B,{self.expected_window}], got "
                f"{tuple(scores.shape)}"
            )
        order = torch.argsort(scores, dim=1, stable=True)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, order[:, : self.k], True)
        return MaskDecision(
            mask,
            {
                "k": self.k,
                "window_length": self.expected_window,
                "selected_fraction": self.k / self.expected_window,
            },
        )


class _TreapNode:
    __slots__ = ("key", "priority", "left", "right", "size")

    def __init__(self, key: tuple[float, int], priority: int) -> None:
        self.key = key
        self.priority = priority
        self.left: _TreapNode | None = None
        self.right: _TreapNode | None = None
        self.size = 1


def _node_size(node: _TreapNode | None) -> int:
    return 0 if node is None else node.size


def _refresh(node: _TreapNode) -> None:
    node.size = 1 + _node_size(node.left) + _node_size(node.right)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _priority(value: float, sequence: int) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return _splitmix64(bits ^ _splitmix64(sequence))


def _rotate_left(node: _TreapNode) -> _TreapNode:
    child = node.right
    assert child is not None
    node.right = child.left
    child.left = node
    _refresh(node)
    _refresh(child)
    return child


def _rotate_right(node: _TreapNode) -> _TreapNode:
    child = node.left
    assert child is not None
    node.left = child.right
    child.right = node
    _refresh(node)
    _refresh(child)
    return child


def _insert(node: _TreapNode | None, item: _TreapNode) -> _TreapNode:
    if node is None:
        return item
    if item.key < node.key:
        node.left = _insert(node.left, item)
        if node.left.priority < node.priority:
            node = _rotate_right(node)
    else:
        node.right = _insert(node.right, item)
        if node.right.priority < node.priority:
            node = _rotate_left(node)
    _refresh(node)
    return node


def _kth(node: _TreapNode | None, index: int) -> float:
    if node is None or not 0 <= index < _node_size(node):
        raise IndexError(index)
    left = _node_size(node.left)
    if index < left:
        return _kth(node.left, index)
    if index == left:
        return node.key[0]
    return _kth(node.right, index - left - 1)


class ExactOnlineQuantile:
    """Exact expanding quantile with expected O(log N) insertion/query."""

    def __init__(
        self,
        values: Iterable[float] | None = None,
        dtype: np.dtype[Any] | str = np.float64,
    ) -> None:
        self._root: _TreapNode | None = None
        self._values: list[float] = []
        self._dtype = np.dtype(dtype)
        if self._dtype.kind != "f":
            raise ValueError("Quantile tracker requires a floating dtype")
        if values is not None:
            self.extend(values)

    def add(self, value: float) -> None:
        typed = float(self._dtype.type(value))
        if not math.isfinite(typed):
            raise ValueError("Quantile history contains NaN or Inf")
        sequence = len(self._values)
        node = _TreapNode((typed, sequence), _priority(typed, sequence))
        self._root = _insert(self._root, node)
        self._values.append(typed)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(float(value))

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1 or not self._values:
            raise ValueError("Quantile requires q in [0,1] and non-empty history")
        probe = np.quantile(
            np.asarray([0, 1], dtype=self._dtype), q, method="linear"
        )
        compute_dtype = np.asarray(probe).dtype
        location = compute_dtype.type(len(self._values) - 1) * compute_dtype.type(q)
        low, high = math.floor(location), math.ceil(location)
        lower = _kth(self._root, low)
        if low == high:
            return lower
        upper = _kth(self._root, high)
        fraction = compute_dtype.type(location - low)
        return float(
            np.quantile(
                np.asarray([lower, upper], dtype=self._dtype),
                fraction,
                method="linear",
            )
        )

    def __len__(self) -> int:
        return len(self._values)


class CausalExpandingP95Selector(Selector):
    def __init__(self, validation_scores: np.ndarray) -> None:
        values = np.asarray(validation_scores).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Expanding-P95 needs finite validation scores")
        self.tracker = ExactOnlineQuantile(values.tolist(), dtype=values.dtype)
        self.pending = False

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        if self.pending:
            raise RuntimeError("observe_after_decision must follow select")
        _require_finite(scores)
        threshold = self.tracker.quantile(0.95)
        history = len(self.tracker)
        self.pending = True
        return MaskDecision(
            scores < threshold,
            {"threshold": threshold, "history_size_before": history},
        )

    def observe_after_decision(self, scores: torch.Tensor) -> None:
        if not self.pending:
            raise RuntimeError("select must precede observe_after_decision")
        self.tracker.extend(scores.detach().cpu().reshape(-1).tolist())
        self.pending = False


class CadenceController:
    def __init__(self, optimizer: torch.optim.Optimizer, cadence: int) -> None:
        if int(cadence) not in (1, 2, 4, 8, 16):
            raise ValueError("Cadence must be one of 1, 2, 4, 8, or 16")
        self.optimizer = optimizer
        self.cadence = int(cadence)
        self.windows = 0
        self.steps = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor) -> bool:
        (loss / self.cadence).backward()
        self.windows += 1
        if self.windows % self.cadence:
            return False
        self.optimizer.step()
        self.steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def discard_tail(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)


def anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    binary = (np.asarray(labels).reshape(-1) != 0).astype(np.int8)
    changes = np.diff(np.pad(binary, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _boundary_width(length: int, fraction: float) -> int:
    if length <= 1 or fraction <= 0:
        return 0
    return min(max(1, math.ceil(fraction * length)), (length - 1) // 2)


def perturb_oracle_mask(
    labels: np.ndarray,
    name: str,
    fraction: float,
    mask_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(labels).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Evaluation labels contain non-finite values")
    if not 0 <= float(fraction) <= 1:
        raise ValueError("Perturbation fraction must be in [0,1]")
    name = str(name).strip().lower()
    allowed = {"fn_contamination", "boundary_erosion", "boundary_dilation"}
    if name not in allowed:
        raise ValueError(f"Unknown oracle-mask perturbation {name!r}")

    binary = (values != 0).astype(np.int8)
    original = (binary == 0).astype(np.int8)
    changed = np.zeros(len(binary), dtype=bool)
    segments = anomaly_segments(binary)
    if name == "fn_contamination" and fraction > 0:
        remaining = math.ceil(float(fraction) * int(binary.sum()))
        rng = np.random.default_rng(int(mask_seed))
        for segment_id in rng.permutation(len(segments)):
            if remaining <= 0:
                break
            start, end = segments[int(segment_id)]
            length = end - start + 1
            take = min(length, remaining)
            offset = (
                0
                if take == length
                else int(rng.integers(0, length + 1 - take))
            )
            changed[start + offset : start + offset + take] = True
            remaining -= take
    elif name == "boundary_erosion":
        for start, end in segments:
            width = _boundary_width(end - start + 1, float(fraction))
            changed[start : start + width] = True
            changed[end - width + 1 : end + 1] = True
    elif name == "boundary_dilation":
        for start, end in segments:
            width = _boundary_width(end - start + 1, float(fraction))
            changed[max(0, start - width) : start] = True
            changed[end + 1 : min(len(binary), end + 1 + width)] = True
        changed &= binary == 0

    perturbed = original.copy()
    if name in {"fn_contamination", "boundary_erosion"}:
        perturbed[changed] = 1
    else:
        perturbed[changed] = 0
    actual = original != perturbed
    eligible = int(perturbed.sum())
    included = int(np.sum((binary == 1) & (perturbed == 1)))
    manifest: dict[str, Any] = {
        "perturbation_name": name,
        "perturbation_fraction": float(fraction),
        "mask_seed": int(mask_seed),
        "sequence_length": int(len(binary)),
        "anomaly_segment_count": len(segments),
        "original_normal_count": int((binary == 0).sum()),
        "original_anomaly_count": int(binary.sum()),
        "changed_mask_points": int(actual.sum()),
        "original_update_eligible_points": int(original.sum()),
        "perturbed_update_eligible_points": eligible,
        "benchmark_anomaly_points_included_in_updates": included,
        "contamination_among_update_eligible_points": (
            float(included / eligible) if eligible else None
        ),
        "normal_coverage": float(
            np.sum((binary == 0) & (perturbed == 1))
            / max(1, int((binary == 0).sum()))
        ),
    }
    return perturbed, manifest


@dataclass(frozen=True)
class FPMReference:
    score_threshold: float
    hard: torch.Tensor
    moderate: torch.Tensor
    inverse_covariance: torch.Tensor
    distance_threshold: float
    validation_windows: int
    hard_topk: int
    covariance_rank: int
    covariance_condition: float
    latent_dim: int


@torch.no_grad()
def normalized_latent(encoder: nn.Module, raw: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(
        encoder(raw.reshape(raw.shape[0], -1)), p=2, dim=1
    )


@torch.no_grad()
def near_any(
    latent: torch.Tensor,
    reference: torch.Tensor,
    inverse_covariance: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if len(reference) == 0:
        return torch.zeros(
            latent.shape[0], dtype=torch.bool, device=latent.device
        )
    delta = latent[:, None, :] - reference[None, :, :]
    distance = ((delta @ inverse_covariance) * delta).sum(-1)
    return (distance < float(threshold)).any(1)


class FPMSelector(Selector):
    def __init__(self, source_encoder: nn.Module, reference: FPMReference) -> None:
        self.source_encoder = source_encoder
        self.reference = reference

    def select(
        self,
        scores: torch.Tensor,
        *,
        raw: torch.Tensor,
        labels: torch.Tensor,
        window_index: int,
    ) -> MaskDecision:
        window_score = scores.mean(1)
        with torch.no_grad():
            latent = normalized_latent(self.source_encoder, raw).double()
            hard = (window_score > self.reference.score_threshold) & near_any(
                latent,
                self.reference.hard,
                self.reference.inverse_covariance,
                self.reference.distance_threshold,
            )
            moderate = (window_score < self.reference.score_threshold) & near_any(
                latent,
                self.reference.moderate,
                self.reference.inverse_covariance,
                self.reference.distance_threshold,
            )
            selected = hard | moderate
        return MaskDecision(
            selected[:, None].expand_as(scores),
            {
                "hard_selected": int(hard.sum()),
                "moderate_selected": int(moderate.sum()),
                "window_selected": int(selected.sum()),
            },
        )


def _require_finite(value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Selector received NaN or Inf")
