from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader

from .data import Entity, PreparedSeries, WindowDataset, entity_manifest, prepare_series
from .metrics import anomaly_ratio_for_dataset, average_precision, binary_prediction_threshold
from .model import WindowAutoencoder, pointwise_reconstruction_loss
from .section6_models import LSTMED, USAD
from .paths import (
    Workspace,
    assert_owned_path,
    checkpoint_path,
    frozen_result_dir,
    source_setting,
    tta_result_dir,
    tta_setting,
)
from .protocol import Protocol, TTAConfig


@dataclass(frozen=True)
class RunOptions:
    device: str = "cuda"
    num_workers: int = 0
    force: bool = False
    resume: bool = True
    log_test_loss_during_source_training: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(normalized)
    raise ValueError(f"Unsupported device {requested!r}; use cuda or cpu")


def _loader(
    values: np.ndarray,
    labels: np.ndarray,
    window: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = WindowDataset(values, labels, window, stride)
    if len(dataset) == 0:
        raise ValueError(
            f"No windows: rows={len(values)}, window={window}, stride={stride}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def build_loaders(
    series: PreparedSeries,
    protocol: Protocol,
    test_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = _loader(
        series.train,
        np.zeros(len(series.train), dtype=np.float32),
        protocol.sequence_length,
        protocol.train_stride,
        protocol.train_batch_size,
        True,
        num_workers,
    )
    validation_loader = _loader(
        series.validation,
        np.zeros(len(series.validation), dtype=np.float32),
        protocol.sequence_length,
        protocol.train_stride,
        protocol.train_batch_size,
        False,
        num_workers,
    )
    test_loader = _loader(
        series.test,
        series.test_labels,
        protocol.sequence_length,
        protocol.test_stride,
        test_batch_size,
        False,
        num_workers,
    )
    return train_loader, validation_loader, test_loader


def _mean_loader_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch_x, _ in loader:
            x = batch_x.float().to(device)
            losses.append(float(_point_loss(model, x, stage="test").mean().item()))
    model.train()
    return float(np.mean(losses))


def _consume_legacy_loader_seed() -> None:
    """Preserve the executable snapshot's train-shuffle RNG progression.

    PyTorch creates a base seed whenever a ``DataLoader`` iterator is built,
    including for a sequential loader with ``num_workers=0``.  The legacy
    source-training loop iterated over the test loader once per epoch solely
    to log a diagnostic loss.  Full-paper mode must not access test features
    during source training, but omitting that iterator also shifts the global
    RNG state and therefore changes the next epoch's shuffled train order.

    One CPU int64 draw is exactly the base-seed transition performed by the
    legacy iterator.  Consuming it here preserves source-training parity
    without reading test features or labels.
    """

    torch.empty((), dtype=torch.int64).random_()


def _atomic_torch_save(state: dict[str, torch.Tensor], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(target)


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_metadata(
    entity: Entity,
    seed: int,
    protocol: Protocol,
    device: torch.device,
) -> dict[str, object]:
    cuda_name = None
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "protocol": protocol.as_dict(),
        "entity": entity_manifest(entity),
        "seed": int(seed),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_device_name": cuda_name,
        "cuda_runtime": torch.version.cuda,
    }


def _load_checkpoint(
    path: Path,
    channels: int,
    protocol: Protocol,
    device: torch.device,
) -> nn.Module:
    model = _build_model(protocol, channels).to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def _build_model(protocol: Protocol, channels: int) -> nn.Module:
    if protocol.model_name == "win_ae":
        return WindowAutoencoder(protocol.sequence_length, channels)
    if protocol.model_name == "USAD":
        return USAD(protocol.sequence_length, channels)
    if protocol.model_name == "LSTMED":
        return LSTMED(protocol.sequence_length, channels)
    raise ValueError(f"Unsupported reconstruction backbone {protocol.model_name!r}")


def _point_loss(
    model: nn.Module, x: torch.Tensor, stage: str, epoch_index: int = 0
) -> torch.Tensor:
    if isinstance(model, (USAD, LSTMED)):
        return model.point_loss(x, stage=stage, epoch_index=epoch_index)
    _, loss = pointwise_reconstruction_loss(model, x)
    return loss


def _scope_parameters(model: nn.Module, scope: str) -> list[nn.Parameter]:
    if scope == "full_model":
        return list(model.parameters())
    if isinstance(model, (USAD, LSTMED)):
        return model.trainable_for_scope(scope)
    if isinstance(model, WindowAutoencoder) and scope == "reconstruction_path":
        return list(model.decoder.parameters())
    raise ValueError(f"Unsupported update scope {scope!r} for {type(model).__name__}")


def _marker_matches_run(
    metadata: dict[str, object],
    protocol: Protocol,
    entity: Entity,
    seed: int,
    setting: str,
    kind: str,
) -> bool:
    return (
        metadata.get("protocol_id") == protocol.protocol_id
        and metadata.get("seed") == int(seed)
        and metadata.get("setting") == setting
        and metadata.get("kind") == kind
        and metadata.get("entity") == entity_manifest(entity)
    )


def _checkpoint_complete(
    path: Path,
    protocol: Protocol,
    entity: Entity,
    seed: int,
) -> bool:
    metadata_path = path.parent / "complete.json"
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _marker_matches_run(
        metadata,
        protocol,
        entity,
        seed,
        source_setting(entity, seed, protocol),
        "source_checkpoint",
    )


def train_source(
    entity: Entity,
    seed: int,
    protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
) -> tuple[nn.Module, PreparedSeries]:
    device = resolve_device(options.device)
    path = checkpoint_path(workspace, entity, seed, protocol)
    assert_owned_path(path, workspace)
    series = prepare_series(entity.directory, protocol.train_fraction)
    channels = series.train.shape[1]

    if (
        options.resume
        and not options.force
        and _checkpoint_complete(path, protocol, entity, seed)
    ):
        return _load_checkpoint(path, channels, protocol, device), series
    if path.exists() and not options.force:
        raise FileExistsError(
            f"Incomplete checkpoint already exists: {path}. "
            "Use --force for this repository-owned target or remove it explicitly."
        )

    set_seed(seed)
    model = _build_model(protocol, channels).to(device)
    train_loader, validation_loader, test_loader = build_loaders(
        series,
        protocol,
        test_batch_size=protocol.frozen_test_batch_size,
        num_workers=options.num_workers,
    )
    optimizer = Adam(model.parameters(), lr=protocol.source_learning_rate)
    checkpoint_dir = path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    best_loss = float("inf")
    best_score: float | None = None
    counter = 0
    started = time.time()

    for epoch_index in range(protocol.train_epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        train_losses: list[float] = []
        model.train()
        for batch_x, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch_x.float().to(device)
            loss = _point_loss(model, x, stage="train", epoch_index=epoch_index).mean()
            train_losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
        train_loss = float(np.mean(train_losses))
        validation_loss = _mean_loader_loss(model, validation_loader, device)
        if options.log_test_loss_during_source_training:
            test_loss = _mean_loader_loss(model, test_loader, device)
        else:
            _consume_legacy_loader_seed()
            test_loss = None
        score = -validation_loss
        improved = best_score is None or score >= best_score
        if improved:
            best_score = score
            best_loss = validation_loss
            counter = 0
            _atomic_torch_save(model.state_dict(), path)
        else:
            counter += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "test_loss_logging_only": test_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_so_far": int(improved),
                "epoch_seconds": time.time() - epoch_started,
            }
        )
        if counter >= protocol.early_stopping_patience:
            break
        # Exact "type1" schedule from the executable snapshot.
        next_rate = protocol.source_learning_rate * (0.5 ** (epoch - 1))
        for group in optimizer.param_groups:
            group["lr"] = next_rate

    log_path = checkpoint_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = _runtime_metadata(entity, seed, protocol, device)
    metadata.update(
        {
            "kind": "source_checkpoint",
            "protocol_id": protocol.protocol_id,
            "setting": source_setting(entity, seed, protocol),
            "epochs_completed": len(rows),
            "best_validation_loss": best_loss,
            "test_loss_logging_enabled": (
                options.log_test_loss_during_source_training
            ),
            "elapsed_seconds": time.time() - started,
        }
    )
    _json_write(checkpoint_dir / "complete.json", metadata)
    return _load_checkpoint(path, channels, protocol, device), series


def _score_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            x = batch_x.float().to(device)
            point_loss = _point_loss(model, x, stage="test")
            scores.append(point_loss.detach().cpu().numpy())
            labels.append(np.asarray(batch_y))
    return (
        np.concatenate(scores, axis=0).reshape(-1),
        np.concatenate(labels, axis=0).reshape(-1),
    )


def _result_complete(
    directory: Path,
    protocol: Protocol,
    tta: bool,
    entity: Entity,
    seed: int,
    setting: str,
) -> bool:
    required = ["test_energy.npy", "true.npy", "metrics.json", "complete.json"]
    if tta:
        required.append("tta_pointwise_analysis.parquet")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _marker_matches_run(
        complete,
        protocol,
        entity,
        seed,
        setting,
        "tta_result" if tta else "frozen_result",
    )


def evaluate_frozen(
    entity: Entity,
    seed: int,
    protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries | None = None,
) -> dict[str, object]:
    directory = frozen_result_dir(workspace, entity, seed, protocol)
    assert_owned_path(directory, workspace)
    setting = source_setting(entity, seed, protocol)
    if (
        options.resume
        and not options.force
        and _result_complete(
            directory, protocol, False, entity, seed, setting
        )
    ):
        return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)
    if series is None:
        series = prepare_series(entity.directory, protocol.train_fraction)
    path = checkpoint_path(workspace, entity, seed, protocol)
    if not _checkpoint_complete(path, protocol, entity, seed):
        raise FileNotFoundError(f"Repository-owned source checkpoint is not complete: {path}")

    # A fresh process in the original runner seeded before model construction.
    set_seed(seed)
    model = _load_checkpoint(path, series.train.shape[1], protocol, device)
    train_loader, _, test_loader = build_loaders(
        series,
        protocol,
        test_batch_size=protocol.frozen_test_batch_size,
        num_workers=options.num_workers,
    )
    started = time.time()
    source_scores, _ = _score_loader(model, train_loader, device)
    test_scores, test_labels = _score_loader(model, test_loader, device)
    threshold = binary_prediction_threshold(
        source_scores,
        test_scores,
        anomaly_ratio_for_dataset(entity.dataset),
    )
    prediction = (test_scores > threshold).astype(np.int8)
    np.save(directory / "test_energy.npy", test_scores)
    np.save(directory / "true.npy", test_labels.astype(np.int8))
    np.save(directory / "pred.npy", prediction)
    metrics: dict[str, object] = {
        "setting": source_setting(entity, seed, protocol),
        "dataset": entity.dataset,
        "subset": entity.subset,
        "seed": int(seed),
        "scenario": "frozen",
        "auprc": average_precision(test_labels, test_scores),
        "points": int(len(test_scores)),
        "threshold_auxiliary": threshold,
        "elapsed_seconds": time.time() - started,
    }
    _json_write(directory / "metrics.json", metrics)
    complete = _runtime_metadata(entity, seed, protocol, device)
    complete.update(
        {
            "kind": "frozen_result",
            "protocol_id": protocol.protocol_id,
            "setting": metrics["setting"],
        }
    )
    _json_write(directory / "complete.json", complete)
    return metrics


def _calibration_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    scores, _ = _score_loader(model, loader, device)
    if not len(scores):
        raise RuntimeError("Calibration loader yielded no score")
    return scores


def _build_tta_optimizer(
    model: nn.Module,
    config: TTAConfig,
) -> torch.optim.Optimizer:
    parameters = _scope_parameters(model, getattr(model, "_tta_scope", "full_model"))
    if config.optimizer == "sgd":
        return SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
        )
    if config.optimizer == "adamw":
        return AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    raise ValueError(f"Unsupported TTA optimizer {config.optimizer!r}")


def evaluate_tta(
    entity: Entity,
    seed: int,
    config: TTAConfig,
    oracle: bool,
    protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
    series: PreparedSeries | None = None,
) -> dict[str, object]:
    directory = tta_result_dir(workspace, entity, seed, protocol, config, oracle)
    assert_owned_path(directory, workspace)
    setting = tta_setting(entity, seed, protocol, config, oracle)
    if (
        options.resume
        and not options.force
        and _result_complete(
            directory, protocol, True, entity, seed, setting
        )
    ):
        return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    device = resolve_device(options.device)
    if series is None:
        series = prepare_series(entity.directory, protocol.train_fraction)
    path = checkpoint_path(workspace, entity, seed, protocol)
    if not _checkpoint_complete(path, protocol, entity, seed):
        raise FileNotFoundError(f"Repository-owned source checkpoint is not complete: {path}")

    set_seed(seed)
    model = _load_checkpoint(path, series.train.shape[1], protocol, device)
    model._tta_scope = protocol.update_scope  # type: ignore[attr-defined]
    train_loader, validation_loader, test_loader = build_loaders(
        series,
        protocol,
        test_batch_size=protocol.tta_test_batch_size,
        num_workers=options.num_workers,
    )
    train_scores = _calibration_scores(model, train_loader, device)
    validation_scores = _calibration_scores(model, validation_loader, device)
    tau = float(np.percentile(validation_scores, protocol.selector_quantile * 100.0))
    optimizer = _build_tta_optimizer(model, config)

    pre_scores: list[np.ndarray] = []
    post_scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pseudo_masks: list[np.ndarray] = []
    update_masks: list[np.ndarray] = []
    did_updates: list[np.ndarray] = []
    started = time.time()

    for batch_x, batch_y in test_loader:
        x = batch_x.float().to(device)
        y = batch_y.float().to(device)
        model.eval()
        with torch.no_grad():
            score_before = _point_loss(model, x, stage="test")
        model.train()
        optimizer.zero_grad()
        loss_per_point = _point_loss(model, x, stage="tta")
        pseudo_mask = (loss_per_point < tau).float()
        update_mask = (y == 0).float() if oracle else pseudo_mask

        # Intentional executable-snapshot semantics:
        # masked entries are zeroed and the loss is averaged over the whole window.
        # The optimizer step is still called for an all-zero mask.
        masked_loss = (loss_per_point * update_mask).mean()
        masked_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score_after = _point_loss(model, x, stage="test")
        pre_scores.append(score_before.detach().cpu().numpy())
        post_scores.append(score_after.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        pseudo_masks.append(pseudo_mask.detach().cpu().numpy().astype(np.int8))
        update_masks.append(update_mask.detach().cpu().numpy().astype(np.int8))
        did_updates.append(np.ones_like(update_masks[-1], dtype=np.int8))

    test_scores = np.concatenate(pre_scores, axis=0).reshape(-1)
    post = np.concatenate(post_scores, axis=0).reshape(-1)
    test_labels = np.concatenate(labels, axis=0).reshape(-1).astype(np.int8)
    pseudo = np.concatenate(pseudo_masks, axis=0).reshape(-1)
    selected = np.concatenate(update_masks, axis=0).reshape(-1)
    did_update = np.concatenate(did_updates, axis=0).reshape(-1)
    point_index = np.arange(len(test_scores), dtype=np.int64)

    threshold = binary_prediction_threshold(
        train_scores,
        test_scores,
        anomaly_ratio_for_dataset(entity.dataset),
    )
    prediction = (test_scores > threshold).astype(np.int8)
    np.save(directory / "test_energy.npy", test_scores)
    np.save(directory / "true.npy", test_labels)
    np.save(directory / "pred.npy", prediction)
    trace = pd.DataFrame(
        {
            "pre_update_score": test_scores.astype(np.float32),
            "post_update_score": post.astype(np.float32),
            "score_delta": (post - test_scores).astype(np.float32),
            "true_label": test_labels,
            "point_index": point_index,
            "pseudo_label": pseudo,
            "simple_used_for_update": selected,
            "did_update": did_update,
        }
    )
    if oracle:
        trace["oracle_normal_mask"] = selected
    trace.to_parquet(
        directory / "tta_pointwise_analysis.parquet",
        engine="pyarrow",
        index=False,
    )
    metrics: dict[str, object] = {
        "setting": tta_setting(entity, seed, protocol, config, oracle),
        "dataset": entity.dataset,
        "subset": entity.subset,
        "seed": int(seed),
        "scenario": "oracle" if oracle else "practical",
        "optimizer_key": config.key,
        "tta_optimizer": config.optimizer,
        "tta_learning_rate": config.learning_rate,
        "auprc": average_precision(test_labels, test_scores),
        "points": int(len(test_scores)),
        "selector_tau": tau,
        "selected_point_fraction": float(selected.mean()),
        "threshold_auxiliary": threshold,
        "elapsed_seconds": time.time() - started,
    }
    _json_write(directory / "metrics.json", metrics)
    complete = _runtime_metadata(entity, seed, protocol, device)
    complete.update(
        {
            "kind": "tta_result",
            "protocol_id": protocol.protocol_id,
            "setting": metrics["setting"],
            "scenario": metrics["scenario"],
            "optimizer_key": config.key,
        }
    )
    _json_write(directory / "complete.json", complete)
    return metrics


def run_entity_seed(
    entity: Entity,
    seed: int,
    protocol: Protocol,
    workspace: Workspace,
    options: RunOptions,
    scenarios: Iterable[str] = ("frozen", "practical", "oracle"),
    optimizer_keys: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    selected_scenarios = tuple(scenarios)
    invalid = set(selected_scenarios) - {"frozen", "practical", "oracle"}
    if invalid:
        raise ValueError(f"Unknown scenarios: {sorted(invalid)}")
    configs = (
        tuple(protocol.tta_configs)
        if optimizer_keys is None
        else tuple(protocol.tta_by_key(key) for key in optimizer_keys)
    )
    _, series = train_source(entity, seed, protocol, workspace, options)
    records: list[dict[str, object]] = []
    if "frozen" in selected_scenarios:
        records.append(
            evaluate_frozen(entity, seed, protocol, workspace, options, series=series)
        )
    for scenario in ("practical", "oracle"):
        if scenario not in selected_scenarios:
            continue
        for config in configs:
            records.append(
                evaluate_tta(
                    entity,
                    seed,
                    config,
                    oracle=scenario == "oracle",
                    protocol=protocol,
                    workspace=workspace,
                    options=options,
                    series=series,
                )
            )
    return records
