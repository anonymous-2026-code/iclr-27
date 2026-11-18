from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import (
    atomic_json,
    atomic_numpy,
    completion_matches,
    finite_tensor,
    sha256_array,
    sha256_file,
    sha256_state,
)
from .data import Entity, PreparedSeries, entity_manifest
from .engine import (
    RunOptions,
    _build_tta_optimizer,
    _load_checkpoint,
    _point_loss,
    build_loaders,
    evaluate_frozen,
    evaluate_tta,
    resolve_device,
    set_seed,
    train_source,
)
from .metrics import average_precision
from .model import WindowAutoencoder
from .paths import Workspace, assert_owned_path, checkpoint_path
from .protocol import Protocol, TTAConfig
from .sana import SANAPipeline
from .supplement_protocol import (
    FPMProtocol,
    SupplementProtocol,
    SupplementVariant,
    variants_for,
)
from .supplement_selectors import (
    BottomKSelector,
    CadenceController,
    CausalExpandingP95Selector,
    ExternalPointMaskSelector,
    FPMReference,
    FPMSelector,
    MaskDecision,
    OraclePointSelector,
    Selector,
    StaticThresholdSelector,
    normalized_latent,
    perturb_oracle_mask,
)


SelectorFactory = Callable[
    [nn.Module, DataLoader, torch.device], tuple[Selector, dict[str, Any]]
]


def supplement_result_dir(
    workspace: Workspace,
    experiment: str,
    variant: str,
    entity: Entity,
    seed: int,
) -> Path:
    return (
        workspace.results
        / "supplements"
        / experiment
        / variant
        / entity.dataset
        / (entity.subset or "__root__")
        / f"seed{int(seed)}"
    )


def _load_completed(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def _result_complete(
    directory: Path,
    protocol: SupplementProtocol,
    experiment: str,
    variant: str,
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    workspace: Workspace,
    required: Iterable[str] = (),
) -> bool:
    files = (
        "test_energy.npy",
        "true.npy",
        "result.json",
        "completed.json",
        *tuple(required),
    )
    if not all((directory / name).is_file() for name in files):
        return False
    if not completion_matches(
        directory,
        protocol_id=protocol.protocol_id,
        experiment=experiment,
        variant=variant,
    ):
        return False
    marker_path = directory / "completed.json"
    result_path = directory / "result.json"
    try:
        # Both files are emitted from the same payload.  Requiring byte identity
        # rejects a stale marker paired with a partially overwritten result.
        marker_bytes = marker_path.read_bytes()
        if marker_bytes != result_path.read_bytes():
            return False
        marker = json.loads(marker_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    checkpoint = checkpoint_path(workspace, entity, seed, source_protocol)
    expected_identity = {
        "complete": True,
        "protocol_id": protocol.protocol_id,
        "experiment": experiment,
        "variant": variant,
        "dataset": entity.dataset,
        "subset": entity.subset,
        "seed": int(seed),
        "source_protocol_id": source_protocol.protocol_id,
        "source_checkpoint_path": str(checkpoint),
        "entity": entity_manifest(entity),
    }
    if any(marker.get(key) != value for key, value in expected_identity.items()):
        return False
    if not checkpoint.is_file():
        return False
    try:
        if marker.get("source_checkpoint_sha256") != sha256_file(checkpoint):
            return False
    except OSError:
        return False

    array_hashes = {
        "test_energy.npy": "score_sha256",
        "true.npy": "label_sha256",
        "update_mask.npy": "update_mask_sha256",
        "selected_window_mask.npy": "selected_window_mask_sha256",
        "perturbed_true.npy": "perturbed_label_sha256",
    }
    arrays: dict[str, np.ndarray] = {}
    for filename, hash_key in array_hashes.items():
        should_exist = filename in files or hash_key in marker
        if not should_exist:
            continue
        path = directory / filename
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return False
        if marker.get(hash_key) != sha256_array(value):
            return False
        arrays[filename] = np.asarray(value).reshape(-1)

    score = arrays["test_energy.npy"]
    labels = arrays["true.npy"]
    if (
        len(score) != int(marker.get("test_points", -1))
        or len(labels) != len(score)
        or not np.isfinite(score).all()
        or not set(np.unique(labels)).issubset({0, 1})
    ):
        return False
    if "update_mask.npy" in arrays and len(arrays["update_mask.npy"]) != len(score):
        return False
    if (
        "selected_window_mask.npy" in arrays
        and len(arrays["selected_window_mask.npy"])
        != int(marker.get("test_windows", -1))
    ):
        return False

    file_hashes = {
        "tta_pointwise_analysis.parquet": "point_trace_sha256",
        "window_trace.parquet": "window_trace_sha256",
    }
    for filename, hash_key in file_hashes.items():
        if filename not in files:
            continue
        try:
            if marker.get(hash_key) != sha256_file(directory / filename):
                return False
        except OSError:
            return False
    return True


def _pointwise_labels(labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    value = labels.to(scores.device)
    if value.ndim == 3:
        value = value.float().mean(-1)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape != scores.shape:
        raise ValueError(
            f"Label/score shape mismatch: {tuple(value.shape)} != "
            f"{tuple(scores.shape)}"
        )
    return value


def _runtime_device_payload(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def _source_checkpoint_metadata(
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    workspace: Workspace,
) -> tuple[Path, dict[str, Any]]:
    checkpoint = checkpoint_path(workspace, entity, seed, source_protocol)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing repository-owned checkpoint: {checkpoint}")
    return checkpoint, {
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_protocol_id": source_protocol.protocol_id,
    }


def _validation_scores(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_x, _ in validation_loader:
            x = batch_x.float().to(device)
            score = _point_loss(model, x, stage="test")
            finite_tensor("validation score", score, -1)
            values.append(score.detach().cpu().numpy())
    if not values:
        raise RuntimeError("Source validation loader was empty")
    result = np.concatenate(values).reshape(-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Source validation scores are non-finite")
    return result


def _build_fpm_selector(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    config: FPMProtocol,
) -> tuple[FPMSelector, dict[str, Any]]:
    if not isinstance(model, WindowAutoencoder):
        raise TypeError("The E4/E7 FPM transplant requires WindowAutoencoder")
    source_encoder = copy.deepcopy(model.encoder).to(device).eval()
    for parameter in source_encoder.parameters():
        parameter.requires_grad_(False)
    encoder_hash = sha256_state(source_encoder)
    window_scores: list[torch.Tensor] = []
    latent_values: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_x, _ in validation_loader:
            x = batch_x.float().to(device)
            point = _point_loss(model, x, stage="test")
            finite_tensor("FPM validation score", point, -1)
            window_scores.append(point.mean(1).cpu())
            latent_values.append(normalized_latent(source_encoder, x).cpu())
    scores = torch.cat(window_scores).double().to(device)
    latents = torch.cat(latent_values).double().to(device)
    if len(scores) < 2:
        raise ValueError("FPM covariance requires at least two validation windows")
    if not bool(torch.isfinite(scores).all() and torch.isfinite(latents).all()):
        raise FloatingPointError("FPM reference contains NaN or Inf")

    count = len(scores)
    hard_topk = int(count * config.hard_tail_percent / 100.0)
    score_array = scores.cpu().numpy()
    score_threshold = float(np.percentile(score_array, config.score_percentile))
    q1, q3 = np.percentile(
        score_array,
        [config.moderate_low_percentile, config.moderate_high_percentile],
    )
    # Frozen historical edge: argpartition(a, -0)[-0:] means the full bank.
    hard_indices = (
        torch.arange(count, device=device)
        if hard_topk == 0
        else torch.topk(
            scores, hard_topk, largest=True, sorted=False
        ).indices
    )
    hard = latents[hard_indices]
    moderate = latents[(scores > float(q1)) & (scores < float(q3))]
    centered = latents - latents.mean(0)
    covariance = centered.T @ centered / (count - 1)
    if config.covariance_regularization:
        covariance = covariance + (
            torch.eye(
                covariance.shape[0],
                dtype=covariance.dtype,
                device=covariance.device,
            )
            * config.covariance_regularization
        )
    inverse = (
        torch.linalg.pinv(covariance)
        if config.inverse_policy == "pinv"
        else torch.linalg.inv(covariance)
    )
    finite_tensor("FPM inverse covariance", inverse, -1)
    latent_dim = int(latents.shape[1])
    reference = FPMReference(
        score_threshold=score_threshold,
        hard=hard,
        moderate=moderate,
        inverse_covariance=inverse,
        distance_threshold=float(
            chi2.ppf(config.chi2_probability, df=latent_dim)
        ),
        validation_windows=count,
        hard_topk=hard_topk,
        covariance_rank=int(torch.linalg.matrix_rank(covariance).item()),
        covariance_condition=float(torch.linalg.cond(covariance).item()),
        latent_dim=latent_dim,
    )
    metadata = {
        "selector": "CANDI-inspired FPM-style selector",
        "score_percentile": config.score_percentile,
        "score_threshold": score_threshold,
        "hard_tail_percent": config.hard_tail_percent,
        "hard_topk": hard_topk,
        "hard_bank_size": len(hard),
        "hard_bank_zero_topk_full_validation": hard_topk == 0,
        "moderate_bank": (
            f"strict Q{config.moderate_low_percentile:g}-"
            f"Q{config.moderate_high_percentile:g}"
        ),
        "moderate_bank_size": len(moderate),
        "chi2_probability": config.chi2_probability,
        "mahalanobis_threshold": reference.distance_threshold,
        "covariance_inverse": config.inverse_policy,
        "covariance_regularization": config.covariance_regularization,
        "covariance_rank": reference.covariance_rank,
        "covariance_condition": reference.covariance_condition,
        "latent_dim": latent_dim,
        "validation_windows": count,
        "source_encoder_sha256": encoder_hash,
        "representation": "L2-normalized frozen source AE encoder latent",
        "hard_rule": "score > threshold and distance < chi2 threshold",
        "moderate_rule": "score < threshold and distance < chi2 threshold",
    }
    return FPMSelector(source_encoder, reference), metadata


def _write_pointwise_artifacts(
    directory: Path,
    *,
    score: np.ndarray,
    labels: np.ndarray,
    update_mask: np.ndarray,
    point_trace: pd.DataFrame,
    window_trace: pd.DataFrame,
    result: dict[str, Any],
    complete: bool,
    alternate_labels: np.ndarray | None = None,
) -> None:
    atomic_numpy(directory / "test_energy.npy", score.astype(np.float32))
    atomic_numpy(directory / "true.npy", labels.astype(np.int8))
    atomic_numpy(directory / "update_mask.npy", update_mask.astype(bool))
    if alternate_labels is not None:
        atomic_numpy(
            directory / "perturbed_true.npy",
            alternate_labels.astype(np.int8),
        )
    point_trace.to_parquet(
        directory / "tta_pointwise_analysis.parquet",
        engine="pyarrow",
        index=False,
    )
    window_trace.to_parquet(
        directory / "window_trace.parquet",
        engine="pyarrow",
        index=False,
    )
    result["point_trace_sha256"] = sha256_file(
        directory / "tta_pointwise_analysis.parquet"
    )
    result["window_trace_sha256"] = sha256_file(
        directory / "window_trace.parquet"
    )
    atomic_json(directory / "result.json", result)
    atomic_json(
        directory / ("completed.json" if complete else "smoke_completed.json"),
        result,
    )


def run_pointwise_online(
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    supplement: SupplementProtocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries,
    validation_scores: np.ndarray,
    experiment: str,
    variant: str,
    optimizer_config: TTAConfig,
    selector_factory: SelectorFactory,
    cadence: int = 1,
    alternate_labels_from_mask: bool = False,
    extra_metadata: dict[str, Any] | None = None,
    max_test_windows: int | None = None,
) -> dict[str, Any]:
    directory = supplement_result_dir(
        workspace, experiment, variant, entity, seed
    )
    assert_owned_path(directory, workspace)
    if (
        max_test_windows is None
        and options.resume
        and not options.force
        and _result_complete(
            directory,
            supplement,
            experiment,
            variant,
            entity=entity,
            seed=seed,
            source_protocol=source_protocol,
            workspace=workspace,
            required=(
                "update_mask.npy",
                "tta_pointwise_analysis.parquet",
                "window_trace.parquet",
            ),
        )
    ):
        return _load_completed(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    checkpoint, source_meta = _source_checkpoint_metadata(
        entity, seed, source_protocol, workspace
    )
    model = _load_checkpoint(
        checkpoint, series.train.shape[1], source_protocol, device
    )
    model._tta_scope = "full_model"  # type: ignore[attr-defined]
    initial_model_hash = sha256_state(model)
    _, validation_loader, test_loader = build_loaders(
        series,
        source_protocol,
        test_batch_size=1,
        num_workers=options.num_workers,
    )
    selector, selector_metadata = selector_factory(
        model, validation_loader, device
    )
    optimizer = _build_tta_optimizer(model, optimizer_config)
    controller = CadenceController(optimizer, cadence)

    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    did_step_chunks: list[np.ndarray] = []
    window_rows: list[dict[str, Any]] = []
    selected_normal = 0
    selected_anomaly = 0
    normal_total = 0
    masked_mass = 0.0
    started = time.time()

    for window_index, (batch_x, batch_y) in enumerate(test_loader):
        if max_test_windows is not None and window_index >= max_test_windows:
            break
        raw = batch_x.float().to(device)
        model.eval()
        with torch.no_grad():
            score_before = _point_loss(model, raw, stage="test")
        finite_tensor("pre-update score", score_before, window_index)
        # Practical selectors must make their immutable decision before test
        # labels are transferred or converted.  Only explicit oracle selectors
        # receive labels at decision time.
        selector_labels = (
            _pointwise_labels(batch_y, score_before)
            if selector.uses_labels
            else torch.empty_like(score_before)
        )
        decision = selector.select(
            score_before.detach(),
            raw=raw,
            labels=selector_labels,
            window_index=window_index,
        )
        if decision.mask.shape != score_before.shape:
            raise ValueError(
                f"Selector mask mismatch at window {window_index}: "
                f"{tuple(decision.mask.shape)} != {tuple(score_before.shape)}"
            )
        mask = decision.mask.bool()
        selector.observe_after_decision(score_before.detach())
        y_point = _pointwise_labels(batch_y, score_before)

        model.train()
        loss_points = _point_loss(model, raw, stage="tta")
        finite_tensor("adaptation point loss", loss_points, window_index)
        masked_loss = (loss_points * mask.float()).mean()
        finite_tensor("masked adaptation loss", masked_loss, window_index)

        score_chunks.append(score_before.detach().cpu().numpy())
        label_chunks.append(y_point.detach().cpu().numpy())
        mask_chunks.append(mask.detach().cpu().numpy())
        selected = int(mask.sum().item())
        normal = y_point == 0
        selected_normal += int((mask & normal).sum().item())
        selected_anomaly += int((mask & ~normal).sum().item())
        normal_total += int(normal.sum().item())
        masked_mass += float((score_before * mask).sum().item())
        did_step = controller.backward(masked_loss)
        did_step_chunks.append(
            np.full(mask.shape, did_step, dtype=np.int8)
        )
        if did_step:
            for name, parameter in model.named_parameters():
                finite_tensor(f"parameter:{name}", parameter, window_index)
        window_rows.append(
            {
                "window_index": window_index,
                "selected_points": selected,
                "masked_loss": float(masked_loss.detach().item()),
                "optimizer_step": bool(did_step),
                "selector_diagnostics": json.dumps(
                    dict(decision.diagnostics), sort_keys=True
                ),
            }
        )
    controller.discard_tail()
    if not score_chunks:
        raise RuntimeError("Online test loader yielded no windows")
    score = np.concatenate(score_chunks).reshape(-1)
    truth = np.concatenate(label_chunks).reshape(-1).astype(np.int8)
    selected = np.concatenate(mask_chunks).reshape(-1).astype(bool)
    did_step = np.concatenate(did_step_chunks).reshape(-1).astype(np.int8)
    if (
        len(score) != len(truth)
        or len(score) != len(selected)
        or not np.isfinite(score).all()
    ):
        raise RuntimeError("Online output is non-finite or misaligned")
    alternate = (1 - selected.astype(np.int8)) if alternate_labels_from_mask else None
    metrics = {"auprc": average_precision(truth, score)}
    if alternate is not None:
        alternate_metrics = {
            "auprc": average_precision(alternate, score)
        }
    else:
        alternate_metrics = {}
    point_trace = pd.DataFrame(
        {
            "pre_update_score": score.astype(np.float32),
            "true_label": truth,
            "point_index": np.arange(len(score), dtype=np.int64),
            "window_index": (
                np.arange(len(score), dtype=np.int64)
                // source_protocol.sequence_length
            ),
            "simple_used_for_update": selected.astype(np.int8),
            "did_update": did_step,
        }
    )
    if alternate is not None:
        point_trace["perturbed_true_label"] = alternate
    window_trace = pd.DataFrame(window_rows)
    complete = max_test_windows is None
    result: dict[str, Any] = {
        "complete": complete,
        "protocol_id": supplement.protocol_id,
        "experiment": experiment,
        "variant": variant,
        "dataset": entity.dataset,
        "subset": entity.subset,
        "seed": int(seed),
        "entity": entity_manifest(entity),
        **source_meta,
        "initial_model_sha256": initial_model_hash,
        "final_model_sha256": sha256_state(model),
        "optimizer_key": optimizer_config.key,
        "optimizer": optimizer_config.optimizer,
        "learning_rate": optimizer_config.learning_rate,
        "weight_decay": 0.0,
        "update_scope": "full_model",
        "sequence_length": source_protocol.sequence_length,
        "train_stride": source_protocol.train_stride,
        "test_stride": source_protocol.test_stride,
        "test_batch_size": 1,
        "cadence": cadence,
        "online_order": (
            "pre-gradient score -> immutable selection -> observe -> "
            "masked backward -> cadence step"
        ),
        "loss": "(point_loss * point_mask).mean() / cadence",
        "test_windows": len(window_rows),
        "test_points": len(score),
        "optimizer_steps": controller.steps,
        "discarded_tail_windows": controller.windows % cadence,
        "selected_points": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "selected_normal_points": selected_normal,
        "selected_anomalous_points": selected_anomaly,
        "selector_purity": (
            selected_normal / int(selected.sum()) if selected.any() else None
        ),
        "normal_coverage": (
            selected_normal / normal_total if normal_total else None
        ),
        "masked_update_mass": masked_mass,
        "score_sha256": sha256_array(score),
        "label_sha256": sha256_array(truth),
        "update_mask_sha256": sha256_array(selected),
        **metrics,
        "alternate_label_metrics": alternate_metrics,
        "selector_config": selector_metadata,
        "wall_seconds": time.time() - started,
        **_runtime_device_payload(device),
        **(extra_metadata or {}),
    }
    if alternate is not None:
        result["perturbed_label_sha256"] = sha256_array(alternate)
    if isinstance(selector, FPMSelector):
        expected = str(selector_metadata["source_encoder_sha256"])
        actual = sha256_state(selector.source_encoder)
        if actual != expected:
            raise AssertionError("Frozen FPM source encoder changed")
        result["selector_encoder_final_sha256"] = actual
    _write_pointwise_artifacts(
        directory,
        score=score,
        labels=truth,
        update_mask=selected,
        point_trace=point_trace,
        window_trace=window_trace,
        result=result,
        complete=complete,
        alternate_labels=alternate,
    )
    return result


def run_m2n2_online(
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    supplement: SupplementProtocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries,
    validation_scores: np.ndarray,
    variant: SupplementVariant,
    max_test_windows: int | None = None,
) -> dict[str, Any]:
    experiment = "e2"
    directory = supplement_result_dir(
        workspace, experiment, variant.name, entity, seed
    )
    assert_owned_path(directory, workspace)
    if (
        max_test_windows is None
        and options.resume
        and not options.force
        and _result_complete(
            directory,
            supplement,
            experiment,
            variant.name,
            entity=entity,
            seed=seed,
            source_protocol=source_protocol,
            workspace=workspace,
            required=(
                "update_mask.npy",
                "tta_pointwise_analysis.parquet",
                "window_trace.parquet",
            ),
        )
    ):
        return _load_completed(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    checkpoint, source_meta = _source_checkpoint_metadata(
        entity, seed, source_protocol, workspace
    )
    model = _load_checkpoint(
        checkpoint, series.train.shape[1], source_protocol, device
    )
    model.eval()
    initial_hash = sha256_state(model)
    gamma = supplement.e2.gamma[entity.dataset]
    threshold = float(
        np.percentile(validation_scores, supplement.e2.m2n2_percentile)
    )
    _, _, test_loader = build_loaders(
        series,
        source_protocol,
        test_batch_size=1,
        num_workers=options.num_workers,
    )
    optimizer: torch.optim.Optimizer | None = None
    if variant.arm != "m2n2_dt_only":
        if variant.optimizer_key is None:
            raise ValueError(f"{variant.name} is missing an optimizer")
        optimizer = _build_tta_optimizer(
            model, source_protocol.tta_by_key(variant.optimizer_key)
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    mean = torch.zeros(1, 1, series.test.shape[1], device=device)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    selected_normal = 0
    selected_anomaly = 0
    normal_total = 0
    first_window: dict[str, Any] = {}
    started = time.time()

    for window_index, (batch_x, batch_y) in enumerate(test_loader):
        if max_test_windows is not None and window_index >= max_test_windows:
            break
        raw = batch_x.float().to(device)
        with torch.no_grad():
            current = raw.mean(dim=(0, 1), keepdim=True)
            mean.lerp_(current, 1.0 - gamma)
        finite_tensor("M2N2 EMA mean", mean, window_index)
        detrended = raw - mean
        finite_tensor("M2N2 detrended input", detrended, window_index)
        if optimizer is None:
            with torch.no_grad():
                reconstruction = model(detrended) + mean
                point = (reconstruction - raw).square().mean(-1)
            mask = torch.zeros_like(point, dtype=torch.bool)
        else:
            reconstruction = model(detrended) + mean
            point = (reconstruction - raw).square().mean(-1)
            if variant.arm == "m2n2_practical_p95":
                mask = point < threshold
            elif variant.arm == "m2n2_oracle_point":
                # This is the sole M2N2 arm whose selection may inspect labels.
                mask = _pointwise_labels(batch_y, point) == 0
            else:
                raise ValueError(f"Unknown M2N2 arm {variant.arm!r}")
        finite_tensor("M2N2 reconstruction", reconstruction, window_index)
        finite_tensor("M2N2 pre-gradient score", point, window_index)
        y_point = _pointwise_labels(batch_y, point)
        scores.append(point.detach().cpu().numpy())
        labels.append(y_point.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())
        normal = y_point == 0
        selected_normal += int((mask & normal).sum().item())
        selected_anomaly += int((mask & ~normal).sum().item())
        normal_total += int(normal.sum().item())
        if window_index == 0:
            first_window = {
                "ema_sha256": sha256_array(mean.detach().cpu().numpy()),
                "detrended_input_sha256": sha256_array(
                    detrended.detach().cpu().numpy()
                ),
                "reconstruction_sha256": sha256_array(
                    reconstruction.detach().cpu().numpy()
                ),
                "score_sha256": sha256_array(point.detach().cpu().numpy()),
            }
        if optimizer is not None:
            loss = (point * mask.float()).mean()
            finite_tensor("M2N2 masked loss", loss, window_index)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, parameter in model.named_parameters():
                finite_tensor(f"M2N2 parameter:{name}", parameter, window_index)
        rows.append(
            {
                "window_index": window_index,
                "selected_points": int(mask.sum().item()),
                "optimizer_step": optimizer is not None,
                "ema_sha256": sha256_array(mean.detach().cpu().numpy()),
            }
        )
    if not scores:
        raise RuntimeError("M2N2 test loader yielded no windows")
    score = np.concatenate(scores).reshape(-1)
    truth = np.concatenate(labels).reshape(-1).astype(np.int8)
    selected = np.concatenate(masks).reshape(-1).astype(bool)
    if len(score) != len(truth) or not np.isfinite(score).all():
        raise RuntimeError("M2N2 output is non-finite or misaligned")
    if optimizer is None and sha256_state(model) != initial_hash:
        raise AssertionError("M2N2 detrending-only detector changed")
    metrics = {"auprc": average_precision(truth, score)}
    complete = max_test_windows is None
    optimizer_config = (
        None
        if variant.optimizer_key is None
        else source_protocol.tta_by_key(variant.optimizer_key)
    )
    result: dict[str, Any] = {
        "complete": complete,
        "protocol_id": supplement.protocol_id,
        "experiment": experiment,
        "variant": variant.name,
        "method": "standardized_m2n2",
        "arm": variant.arm,
        "dataset": entity.dataset,
        "subset": entity.subset,
        "seed": int(seed),
        "entity": entity_manifest(entity),
        **source_meta,
        "initial_model_sha256": initial_hash,
        "final_model_sha256": sha256_state(model),
        "optimizer_key": variant.optimizer_key,
        "optimizer": (
            optimizer_config.optimizer if optimizer_config is not None else "none"
        ),
        "learning_rate": (
            optimizer_config.learning_rate if optimizer_config is not None else 0.0
        ),
        "gamma": gamma,
        "validation_score_sha256": sha256_array(validation_scores),
        "validation_percentile": supplement.e2.m2n2_percentile,
        "numeric_threshold": threshold,
        "validation_ema_updates": 0,
        "validation_normalizer_state": "fresh_zero",
        "sequence_length": source_protocol.sequence_length,
        "train_stride": source_protocol.train_stride,
        "test_stride": source_protocol.test_stride,
        "test_batch_size": 1,
        "update_scope": "full_model",
        "online_order": (
            "current raw window updates EMA -> detrend -> reconstruct and "
            "restore EMA -> pre-gradient score -> mask -> update"
        ),
        "loss": "(point_loss * point_mask).mean()",
        "ema_updates": len(rows),
        "optimizer_steps": len(rows) if optimizer is not None else 0,
        "test_windows": len(rows),
        "test_points": len(score),
        "selected_points": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "selected_normal_points": selected_normal,
        "selected_anomalous_points": selected_anomaly,
        "selector_purity": (
            selected_normal / int(selected.sum()) if selected.any() else None
        ),
        "normal_coverage": (
            selected_normal / normal_total if normal_total else None
        ),
        "score_sha256": sha256_array(score),
        "label_sha256": sha256_array(truth),
        "update_mask_sha256": sha256_array(selected),
        "first_window": first_window,
        **metrics,
        "wall_seconds": time.time() - started,
        **_runtime_device_payload(device),
    }
    point_trace = pd.DataFrame(
        {
            "pre_update_score": score.astype(np.float32),
            "true_label": truth,
            "point_index": np.arange(len(score), dtype=np.int64),
            "window_index": (
                np.arange(len(score), dtype=np.int64)
                // source_protocol.sequence_length
            ),
            "simple_used_for_update": selected.astype(np.int8),
            "did_update": np.full(
                len(score), optimizer is not None, dtype=np.int8
            ),
        }
    )
    _write_pointwise_artifacts(
        directory,
        score=score,
        labels=truth,
        update_mask=selected,
        point_trace=point_trace,
        window_trace=pd.DataFrame(rows),
        result=result,
        complete=complete,
    )
    return result


def _changed_state_names(
    before: dict[str, torch.Tensor], module: nn.Module
) -> list[str]:
    return [
        name
        for name, value in module.state_dict().items()
        if not torch.equal(before[name], value.detach().cpu())
    ]


def run_e7_arms(
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    supplement: SupplementProtocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries,
    variants: tuple[SupplementVariant, ...],
    max_test_windows: int | None = None,
    optimizer_config: TTAConfig | None = None,
) -> list[dict[str, Any]]:
    # The submitted E7 default remains AdamW 1e-4. Extensions pass their
    # explicit configuration and use separate output workspaces.
    config = optimizer_config or source_protocol.tta_by_key("adamw_ttlr0.0001")
    device = resolve_device(options.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    checkpoint, source_meta = _source_checkpoint_metadata(
        entity, seed, source_protocol, workspace
    )
    source = _load_checkpoint(
        checkpoint, series.train.shape[1], source_protocol, device
    )
    if not isinstance(source, WindowAutoencoder):
        raise TypeError("E7 requires the AE source backbone")
    source.eval()
    source_hash = sha256_state(source)
    _, validation_loader, test_loader = build_loaders(
        series,
        source_protocol,
        test_batch_size=1,
        num_workers=options.num_workers,
    )
    fpm, fpm_metadata = _build_fpm_selector(
        source, validation_loader, device, supplement.fpm
    )

    set_seed(seed + supplement.e7.sana_seed_offset)
    template = SANAPipeline(
        copy.deepcopy(source),
        series.test.shape[1],
        window=source_protocol.sequence_length,
    ).to(device).eval()
    initial_state = {
        name: value.detach().clone()
        for name, value in template.state_dict().items()
    }
    sana_template = nn.ModuleDict(
        {"sana_in": template.sana_in, "sana_out": template.sana_out}
    )
    sana_init_hash = sha256_state(sana_template)
    with torch.inference_mode():
        probe = next(iter(validation_loader))[0].float().to(device)[:1]
        source_reconstruction = source(probe)
        identity_reconstruction = template(probe)
        identity_reconstruction_max_abs = float(
            (identity_reconstruction - source_reconstruction).abs().max().item()
        )
        source_score = (source_reconstruction - probe).square().mean(-1)
        identity_score = (identity_reconstruction - probe).square().mean(-1)
        identity_score_max_abs = float(
            (identity_score - source_score).abs().max().item()
        )
    if identity_reconstruction_max_abs != 0.0 or identity_score_max_abs != 0.0:
        raise AssertionError("SANA zero gates are not exact identity")

    results: list[dict[str, Any]] = []
    for variant in variants:
        directory = supplement_result_dir(
            workspace, "e7", variant.name, entity, seed
        )
        assert_owned_path(directory, workspace)
        if (
            max_test_windows is None
            and options.resume
            and not options.force
            and _result_complete(
                directory,
                supplement,
                "e7",
                variant.name,
                entity=entity,
                seed=seed,
                source_protocol=source_protocol,
                workspace=workspace,
                required=("selected_window_mask.npy", "update_mask.npy"),
            )
        ):
            results.append(_load_completed(directory))
            continue
        directory.mkdir(parents=True, exist_ok=True)
        pipeline = SANAPipeline(
            copy.deepcopy(source),
            series.test.shape[1],
            window=source_protocol.sequence_length,
        ).to(device)
        pipeline.load_state_dict(initial_state, strict=True)
        pipeline.eval()
        backbone_before = {
            name: value.detach().cpu().clone()
            for name, value in pipeline.backbone.state_dict().items()
        }
        sana_module = nn.ModuleDict(
            {"sana_in": pipeline.sana_in, "sana_out": pipeline.sana_out}
        )
        sana_before = {
            name: value.detach().cpu().clone()
            for name, value in sana_module.state_dict().items()
        }
        optimizer = (
            None
            if variant.arm == "frozen_sana"
            else _build_tta_optimizer(sana_module, config)
        )
        score_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        selections: list[np.ndarray] = []
        point_masks: list[np.ndarray] = []
        steps = 0
        anomaly_selected = 0
        normal_selected = 0
        started = time.time()
        latent_probe: tuple[torch.Tensor, torch.Tensor] | None = None

        for window_index, (batch_x, batch_y) in enumerate(test_loader):
            if max_test_windows is not None and window_index >= max_test_windows:
                break
            raw = batch_x.float().to(device)
            if latent_probe is None:
                latent_probe = (
                    raw.detach().clone(),
                    normalized_latent(fpm.source_encoder, raw).cpu(),
                )
            reconstruction = pipeline(raw)
            point = (reconstruction - raw).square().mean(-1)
            finite_tensor("E7 score", point, window_index)
            fpm_window: torch.Tensor | None = None
            if variant.arm in {
                "practical_fpm_sana",
                "oracle_clean_fpm_sana",
            }:
                fpm_decision = fpm.select(
                    point.detach(),
                    raw=raw,
                    labels=torch.empty_like(point),
                    window_index=window_index,
                )
                fpm_window = fpm_decision.mask.all(1)
            labels = _pointwise_labels(batch_y, point)
            all_normal = (labels == 0).all(1)
            if variant.arm == "frozen_sana":
                selected_window = torch.zeros_like(all_normal)
            elif variant.arm == "practical_fpm_sana":
                assert fpm_window is not None
                selected_window = fpm_window
            elif variant.arm == "oracle_normal_window_sana":
                selected_window = all_normal
            elif variant.arm == "oracle_clean_fpm_sana":
                assert fpm_window is not None
                selected_window = fpm_window & all_normal
            else:
                raise ValueError(f"Unknown E7 arm {variant.arm!r}")
            selected_points = selected_window[:, None].expand_as(labels)
            score_chunks.append(point.detach().cpu().numpy())
            label_chunks.append(labels.detach().cpu().numpy())
            selections.append(selected_window.detach().cpu().numpy())
            point_masks.append(selected_points.detach().cpu().numpy())
            normal_selected += int(
                (selected_points & (labels == 0)).sum().item()
            )
            anomaly_selected += int(
                (selected_points & (labels != 0)).sum().item()
            )
            if bool(selected_window.any()):
                if optimizer is None:
                    raise AssertionError("Frozen-SANA selected an update window")
                loss = (
                    reconstruction[selected_window] - raw[selected_window]
                ).square().mean()
                finite_tensor("E7 whole-window loss", loss, window_index)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None
                    for parameter in pipeline.backbone.parameters()
                ):
                    raise AssertionError("Frozen E7 backbone received a gradient")
                optimizer.step()
                steps += 1
        if not score_chunks:
            raise RuntimeError("E7 test loader yielded no windows")
        score = np.concatenate(score_chunks).reshape(-1)
        truth = np.concatenate(label_chunks).reshape(-1).astype(np.int8)
        selected_windows = np.concatenate(selections).reshape(-1).astype(bool)
        selected_points = np.concatenate(point_masks).reshape(-1).astype(bool)
        if (
            len(score) != len(truth)
            or len(score)
            != source_protocol.sequence_length * len(selected_windows)
            or not np.isfinite(score).all()
        ):
            raise RuntimeError("E7 output is non-finite or misaligned")
        backbone_changed = any(
            not torch.equal(backbone_before[name], value.detach().cpu())
            for name, value in pipeline.backbone.state_dict().items()
        )
        if backbone_changed:
            raise AssertionError("E7 source backbone changed")
        assert latent_probe is not None
        if not torch.equal(
            latent_probe[1],
            normalized_latent(fpm.source_encoder, latent_probe[0]).cpu(),
        ):
            raise AssertionError("E7 frozen FPM latent encoder changed")
        metrics = {"auprc": average_precision(truth, score)}
        sana_changed = _changed_state_names(sana_before, sana_module)
        complete = max_test_windows is None
        result: dict[str, Any] = {
            "complete": complete,
            "protocol_id": supplement.protocol_id,
            "experiment": "e7",
            "variant": variant.name,
            "arm": variant.arm,
            "dataset": entity.dataset,
            "subset": entity.subset,
            "seed": int(seed),
            "entity": entity_manifest(entity),
            **source_meta,
            "source_model_sha256": source_hash,
            "sana_init_sha256": sana_init_hash,
            "identity_reconstruction_max_abs": identity_reconstruction_max_abs,
            "identity_score_max_abs": identity_score_max_abs,
            "sequence_length": source_protocol.sequence_length,
            "train_stride": source_protocol.train_stride,
            "test_stride": source_protocol.test_stride,
            "test_batch_size": 1,
            "selector_type": (
                "fpm_hard_plus_moderate"
                if "fpm" in str(variant.arm)
                else "none"
            ),
            "oracle_type": (
                "normal_window"
                if variant.arm == "oracle_normal_window_sana"
                else "clean_fpm"
                if variant.arm == "oracle_clean_fpm_sana"
                else "none"
            ),
            "optimizer": "none" if optimizer is None else config.optimizer,
            "optimizer_key": None if optimizer is None else config.key,
            "learning_rate": 0.0 if optimizer is None else config.learning_rate,
            "update_scope": "sana_in_and_sana_out_only",
            "loss": "whole-window all-time/all-feature squared-error mean",
            "online_order": (
                "score -> FPM decision -> label-only oracle/diagnostics -> "
                "record -> selected-window step"
            ),
            "test_windows": len(selected_windows),
            "test_points": len(score),
            "selected_windows": int(selected_windows.sum()),
            "selected_points": int(selected_points.sum()),
            "selected_normal_points": normal_selected,
            "selected_anomalous_points": anomaly_selected,
            "optimizer_steps": steps,
            "backbone_changed": False,
            "latent_probe_bitwise_equal": True,
            "sana_changed_parameter_names": sana_changed,
            "sana_parameter_count": sum(
                parameter.numel() for parameter in pipeline.sana_parameters()
            ),
            "backbone_parameter_count": sum(
                parameter.numel()
                for parameter in pipeline.backbone.parameters()
            ),
            "fpm_config": fpm_metadata,
            "score_sha256": sha256_array(score),
            "label_sha256": sha256_array(truth),
            "selected_window_mask_sha256": sha256_array(selected_windows),
            "update_mask_sha256": sha256_array(selected_points),
            "final_sana_sha256": sha256_state(sana_module),
            **metrics,
            "wall_seconds": time.time() - started,
            **_runtime_device_payload(device),
        }
        atomic_numpy(directory / "test_energy.npy", score.astype(np.float32))
        atomic_numpy(directory / "true.npy", truth)
        atomic_numpy(
            directory / "selected_window_mask.npy", selected_windows
        )
        atomic_numpy(directory / "update_mask.npy", selected_points)
        atomic_json(directory / "result.json", result)
        atomic_json(
            directory
            / ("completed.json" if complete else "smoke_completed.json"),
            result,
        )
        results.append(result)
        if sha256_state(source) != source_hash:
            raise AssertionError("Shared E7 source model changed between arms")
    return results


def _ensure_source_and_controls(
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
    optimizer_keys: Iterable[str] = (),
    controls: bool,
) -> PreparedSeries:
    source_model, series = train_source(
        entity, seed, source_protocol, workspace, options
    )
    del source_model
    if controls:
        evaluate_frozen(
            entity,
            seed,
            source_protocol,
            workspace,
            options,
            series=series,
        )
        for key in dict.fromkeys(optimizer_keys):
            config = source_protocol.tta_by_key(key)
            evaluate_tta(
                entity,
                seed,
                config,
                oracle=False,
                protocol=source_protocol,
                workspace=workspace,
                options=options,
                series=series,
            )
            evaluate_tta(
                entity,
                seed,
                config,
                oracle=True,
                protocol=source_protocol,
                workspace=workspace,
                options=options,
                series=series,
            )
    return series


def _calibration_for_series(
    *,
    entity: Entity,
    seed: int,
    source_protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries,
) -> np.ndarray:
    device = resolve_device(options.device)
    set_seed(seed)
    model = _load_checkpoint(
        checkpoint_path(workspace, entity, seed, source_protocol),
        series.train.shape[1],
        source_protocol,
        device,
    )
    _, validation_loader, _ = build_loaders(
        series,
        source_protocol,
        test_batch_size=1,
        num_workers=options.num_workers,
    )
    scores = _validation_scores(model, validation_loader, device)
    del model
    return scores


def _static_factory(
    validation_scores: np.ndarray, percentile: float
) -> SelectorFactory:
    threshold = float(np.percentile(validation_scores, percentile))

    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return StaticThresholdSelector(threshold, percentile), {
            "selector": f"static_p{percentile:g}",
            "percentile": percentile,
            "threshold": threshold,
            "comparison": "strict_lt",
            "validation_score_sha256": sha256_array(validation_scores),
        }

    return build


def _oracle_factory() -> SelectorFactory:
    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return OraclePointSelector(), {
            "selector": "benchmark_normal_point_oracle",
            "labels_used": True,
        }

    return build


def _expanding_factory(
    validation_scores: np.ndarray,
) -> SelectorFactory:
    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return CausalExpandingP95Selector(validation_scores), {
            "selector": "causal_expanding_p95",
            "initial_history_points": len(validation_scores),
            "validation_score_sha256": sha256_array(validation_scores),
            "comparison": "strict_lt",
            "observe_order": "decision_before_current_scores_enter_history",
        }

    return build


def _bottomk_factory(k: int, window: int) -> SelectorFactory:
    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return BottomKSelector(k, window), {
            "selector": f"bottomk_low_score_k{k}",
            "k": k,
            "window": window,
            "tie_policy": "stable_time_order",
        }

    return build


def _fpm_factory(config: FPMProtocol) -> SelectorFactory:
    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return _build_fpm_selector(model, validation_loader, device, config)

    return build


def _external_mask_factory(
    mask: np.ndarray, window: int, stride: int, manifest: dict[str, Any]
) -> SelectorFactory:
    def build(
        model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Selector, dict[str, Any]]:
        return ExternalPointMaskSelector(mask, window, stride), {
            "selector": "prespecified_perturbed_oracle_update_mask",
            **manifest,
        }

    return build


def run_supplement_entity(
    *,
    experiment: str,
    entity: Entity,
    seed: int,
    supplement: SupplementProtocol,
    workspace: Workspace,
    options: RunOptions,
    variant_names: Iterable[str] | None = None,
    max_test_windows: int | None = None,
) -> list[dict[str, Any]]:
    name = supplement.validate_experiment(experiment)
    if name == "e5":
        raise ValueError("E5 is score-only post-processing; use `export e5`.")
    if not supplement.task_is_eligible(name, entity.dataset, seed):
        raise ValueError(
            f"{entity.dataset}/seed{seed} is outside the locked {name.upper()} grid"
        )
    source_protocol = supplement.source_protocol()
    variants = variants_for(
        supplement, name, entity.dataset, seed
    )
    if variant_names is not None:
        requested = tuple(variant_names)
        by_name = {variant.name: variant for variant in variants}
        unknown = set(requested) - set(by_name)
        if unknown:
            raise ValueError(
                f"Unknown/ineligible {name.upper()} variants: {sorted(unknown)}"
            )
        variants = tuple(by_name[item] for item in requested)
    if not variants:
        raise ValueError(f"No {name.upper()} variant selected")
    optimizer_keys = tuple(
        variant.optimizer_key
        for variant in variants
        if variant.optimizer_key is not None
    )
    controls = name in {"e1", "e2", "e3", "e4", "e6"}
    series = _ensure_source_and_controls(
        entity=entity,
        seed=seed,
        source_protocol=source_protocol,
        workspace=workspace,
        options=options,
        optimizer_keys=optimizer_keys,
        controls=controls,
    )
    if name == "e7":
        return run_e7_arms(
            entity=entity,
            seed=seed,
            source_protocol=source_protocol,
            supplement=supplement,
            workspace=workspace,
            options=options,
            series=series,
            variants=variants,
            max_test_windows=max_test_windows,
        )
    validation_scores = _calibration_for_series(
        entity=entity,
        seed=seed,
        source_protocol=source_protocol,
        workspace=workspace,
        options=options,
        series=series,
    )
    results: list[dict[str, Any]] = []
    for variant in variants:
        if name == "e1":
            assert variant.perturbation is not None
            assert variant.fraction is not None
            assert variant.optimizer_key is not None
            mask, manifest = perturb_oracle_mask(
                series.test_labels,
                variant.perturbation,
                variant.fraction,
                supplement.e1.mask_seed,
            )
            manifest.update(
                {
                    "original_mask_sha256": sha256_array(
                        (series.test_labels == 0).astype(np.int8)
                    ),
                    "perturbed_mask_sha256": sha256_array(mask),
                    "evaluation_label_sha256": sha256_array(
                        (series.test_labels != 0).astype(np.int8)
                    ),
                }
            )
            results.append(
                run_pointwise_online(
                    entity=entity,
                    seed=seed,
                    source_protocol=source_protocol,
                    supplement=supplement,
                    workspace=workspace,
                    options=options,
                    series=series,
                    validation_scores=validation_scores,
                    experiment=name,
                    variant=variant.name,
                    optimizer_config=source_protocol.tta_by_key(
                        variant.optimizer_key
                    ),
                    selector_factory=_external_mask_factory(
                        mask,
                        source_protocol.sequence_length,
                        source_protocol.test_stride,
                        manifest,
                    ),
                    alternate_labels_from_mask=True,
                    extra_metadata={"perturbation_manifest": manifest},
                    max_test_windows=max_test_windows,
                )
            )
        elif name == "e2" and str(variant.arm).startswith("m2n2_"):
            results.append(
                run_m2n2_online(
                    entity=entity,
                    seed=seed,
                    source_protocol=source_protocol,
                    supplement=supplement,
                    workspace=workspace,
                    options=options,
                    series=series,
                    validation_scores=validation_scores,
                    variant=variant,
                    max_test_windows=max_test_windows,
                )
            )
        else:
            assert variant.optimizer_key is not None
            selector_factory: SelectorFactory
            cadence = variant.cadence or 1
            if name == "e2":
                if variant.selector == "static_p99":
                    selector_factory = _static_factory(
                        validation_scores, 99.0
                    )
                elif variant.selector == "static_p99p5":
                    selector_factory = _static_factory(
                        validation_scores, 99.5
                    )
                elif variant.selector == "causal_expanding_p95":
                    selector_factory = _expanding_factory(validation_scores)
                elif variant.selector == f"bottomk_k{supplement.e2.bottom_k}":
                    selector_factory = _bottomk_factory(
                        supplement.e2.bottom_k,
                        source_protocol.sequence_length,
                    )
                else:
                    raise ValueError(
                        f"Unknown E2 selector {variant.selector!r}"
                    )
            elif name in {"e3", "e6"}:
                selector_factory = (
                    _oracle_factory()
                    if variant.oracle
                    else _static_factory(validation_scores, 95.0)
                )
            elif name == "e4":
                selector_factory = _fpm_factory(supplement.fpm)
            else:
                raise ValueError(f"Unhandled supplement {name!r}")
            results.append(
                run_pointwise_online(
                    entity=entity,
                    seed=seed,
                    source_protocol=source_protocol,
                    supplement=supplement,
                    workspace=workspace,
                    options=options,
                    series=series,
                    validation_scores=validation_scores,
                    experiment=name,
                    variant=variant.name,
                    optimizer_config=source_protocol.tta_by_key(
                        variant.optimizer_key
                    ),
                    selector_factory=selector_factory,
                    cadence=cadence,
                    max_test_windows=max_test_windows,
                )
            )
    return results
