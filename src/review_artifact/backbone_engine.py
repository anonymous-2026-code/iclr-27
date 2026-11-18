"""Portable Table 2 loops, following the submitted historical producers.

Model definitions and state-dict names are unchanged. In particular USAD
validation uses compute_loss(stage='val'), and LSTMED scores in eval mode
but adapts in teacher-forcing train mode. The AE engine is not modified.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pandas as pd
import torch

from section5_repro.artifacts import sha256_file
from section5_repro.data import entity_manifest, prepare_series
from section5_repro.engine import build_loaders, set_seed, resolve_device, _consume_legacy_loader_seed
from section5_repro.paths import checkpoint_path, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol, repository_root
from .aggregate import point_auprc
from .historical_models.usad import Model as USAD
from .historical_models.lstmed import Model as LSTMED

SCOPES = ("full_model", "reconstruction_path")
BACKBONES = ("USAD", "LSTMED")


def load_backbone_config():
    with (repository_root() / "configs/backbones.toml").open("rb") as handle:
        config = tomllib.load(handle)
    expected = {
        "protocol_id": "table2-submission-v1", "backbones": list(BACKBONES),
        "update_scopes": list(SCOPES),
        "USAD": {"model_id": "base_usad_w0p5", "latent_dim": 5, "score_weight": .5, "frozen_test_batch_size": 64},
        "LSTMED": {"model_id": "lstmed_12_12_12_core", "latent_dim": 5, "layers": 1, "dropout": 0., "frozen_test_batch_size": 1},
    }
    if config != expected:
        raise ValueError("Backbone settings differ from submitted Table 2")
    return config


def protocol_for(backbone, scope="full_model"):
    config = load_backbone_config()
    if backbone not in BACKBONES or scope not in SCOPES:
        raise ValueError("Unknown Table 2 backbone/update scope")
    return replace(load_protocol(), protocol_id=config["protocol_id"],
                   model_id=config[backbone]["model_id"], model_name=backbone,
                   frozen_test_batch_size=config[backbone]["frozen_test_batch_size"], update_scope=scope)


def build_model(backbone, channels, device):
    config = load_backbone_config()[backbone]
    args = SimpleNamespace(task_name="anomaly_detection", seq_len=12, enc_in=channels,
                           latent_dim=config["latent_dim"], usad_score_weight=config.get("score_weight", .5),
                           e_layers=config.get("layers", 1), dropout=config.get("dropout", 0.),
                           use_gpu=torch.device(device).type == "cuda")
    return {"USAD": USAD, "LSTMED": LSTMED}[backbone](args).float().to(device)


def source_loss(model, x, stage, epoch=None):
    outputs = model(x)
    if isinstance(model, USAD):
        return model.compute_loss(x, outputs=outputs, stage=stage, epoch=epoch, reduction="mean")
    return torch.nn.functional.mse_loss(outputs, x)


def point_score(model, x):
    outputs = model(x)
    if isinstance(model, USAD):
        return model.compute_score(x, outputs=outputs)
    return (outputs - x).square().mean(dim=-1)


def tta_loss(model, x):
    outputs = model(x)
    if isinstance(model, USAD):
        return model.compute_tta_loss(x, outputs=outputs, reduction="none")
    return (outputs - x).square().mean(dim=-1)


def identity(entity, seed, protocol, kind, **extra):
    return {"entity": entity_manifest(entity), "seed": seed, "kind": kind,
            "protocol": json.loads(json.dumps(protocol.as_dict())), **extra}


def completed(directory, expected):
    marker = directory / "complete.json"
    if not marker.exists():
        if directory.exists():
            raise FileExistsError(f"Incomplete output; inspect explicitly before retrying: {directory}")
        return False
    actual = json.loads(marker.read_text())
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Completed output identity differs: {directory}")
    return True


def write_complete(directory, metadata):
    (directory / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_source(backbone, series, device, path):
    model = build_model(backbone, series.train.shape[1], device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model


def train_source(entity, seed, backbone, workspace, device, series):
    protocol = protocol_for(backbone)
    path = checkpoint_path(workspace, entity, seed, protocol)
    meta = identity(entity, seed, protocol, "source")
    if completed(path.parent, meta):
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    set_seed(seed)
    model = build_model(backbone, series.train.shape[1], device)
    train, validation, _ = build_loaders(series, protocol, protocol.frozen_test_batch_size, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.source_learning_rate)
    path.parent.mkdir(parents=True)
    best, patience, rows = None, 0, []
    for epoch in range(protocol.train_epochs):
        model.train()
        losses = []
        for x, _ in train:
            optimizer.zero_grad(set_to_none=True)
            loss = source_loss(model, x.float().to(device), "train", epoch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite source training loss")
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for x, _ in validation:
                validation_losses.append(source_loss(model, x.float().to(device), "val").item())
        value = float(np.average(validation_losses))
        if not np.isfinite(value):
            raise ValueError("Non-finite source validation loss")
        # Historical test-loss logging did not select the checkpoint. Preserve
        # its DataLoader RNG draw without accessing test values during fitting.
        _consume_legacy_loader_seed()
        improved = best is None or value <= best
        if improved:
            best, patience = value, 0
            torch.save(model.state_dict(), path)
        else:
            patience += 1
        rows.append({"epoch": epoch + 1, "train_loss": float(np.average(losses)),
                     "validation_loss": value, "learning_rate": optimizer.param_groups[0]["lr"],
                     "best_so_far": improved})
        if patience >= protocol.early_stopping_patience:
            break
        for group in optimizer.param_groups:
            group["lr"] = protocol.source_learning_rate * .5 ** epoch
    pd.DataFrame(rows).to_csv(path.parent / "training_log.csv", index=False)
    write_complete(path.parent, {**meta, "best_validation_loss": best, "epochs_completed": len(rows)})
    return path


def score_loader(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(point_score(model, x.float().to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores).reshape(-1), np.concatenate(labels).reshape(-1).astype(np.int8)


def save_scores(directory, meta, scores, labels, expected_length):
    if len(scores) != expected_length or len(labels) != expected_length:
        raise ValueError("Backbone stream length mismatch; cropping is forbidden")
    valid, value, reason = point_auprc(labels, scores)
    np.save(directory / "test_energy.npy", scores)
    np.save(directory / "true.npy", labels)
    np.save(directory / "point_index.npy", np.arange(expected_length, dtype=np.int64))
    write_complete(directory, {**meta, "raw_valid": valid, "corrected_auprc": value if valid else None,
                               "invalid_reason": reason})
    if not valid:
        print(f"INVALID complete stream: {directory.name}: {reason}", flush=True)


def run_entity(entity, seed, backbone, workspace, requested_device):
    device = resolve_device(requested_device)
    series = prepare_series(entity.directory)
    protocol = protocol_for(backbone)
    source = train_source(entity, seed, backbone, workspace, device, series)
    source_hash = sha256_file(source)
    expected_length = len(series.test) // 12 * 12
    frozen = frozen_result_dir(workspace, entity, seed, protocol)
    meta = identity(entity, seed, protocol, "frozen", source_checkpoint_sha256=source_hash)
    if not completed(frozen, meta):
        frozen.mkdir(parents=True)
        set_seed(seed)
        model = load_source(backbone, series, device, source)
        train, _, test = build_loaders(series, protocol, protocol.frozen_test_batch_size, 0)
        score_loader(model, train, device)  # Preserve historical inference/RNG order.
        score, labels = score_loader(model, test, device)
        save_scores(frozen, meta, score, labels, expected_length)
    for scope in SCOPES:
        protocol = protocol_for(backbone, scope)
        for config in protocol.tta_configs:
            for oracle in (False, True):
                directory = tta_result_dir(workspace, entity, seed, protocol, config, oracle)
                meta = identity(entity, seed, protocol, "oracle" if oracle else "practical",
                                optimizer_key=config.key, source_checkpoint_sha256=source_hash)
                if completed(directory, meta):
                    continue
                directory.mkdir(parents=True)
                set_seed(seed)
                model = load_source(backbone, series, device, source)
                train, validation, test = build_loaders(series, protocol, 1, 0)
                parameters = [p for _, p in model.get_tta_named_trainable_params(scope)]
                optimizer = (torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0., weight_decay=0.)
                             if config.optimizer == "sgd" else
                             torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0.))
                score_loader(model, train, device)
                calibration, _ = score_loader(model, validation, device)
                if not np.isfinite(calibration).all():
                    raise ValueError("Non-finite source calibration scores")
                tau = float(np.percentile(calibration, 95.))
                before, after, truth, selected = [], [], [], []
                for x, y in test:
                    x, y = x.float().to(device), y.to(device)
                    model.eval()
                    with torch.no_grad():
                        score = point_score(model, x)
                    model.train()
                    loss_points = tta_loss(model, x)
                    mask = (y == 0).float() if oracle else (loss_points < tau).float()
                    optimizer.zero_grad()
                    (loss_points * mask).mean().backward()
                    optimizer.step()  # Includes empty-mask windows, as submitted.
                    model.eval()
                    with torch.no_grad():
                        post = point_score(model, x)
                    before.append(score.detach().cpu().numpy())
                    after.append(post.cpu().numpy())
                    truth.append(y.cpu().numpy())
                    selected.append(mask.cpu().numpy())
                scores = np.concatenate(before).reshape(-1)
                labels = np.concatenate(truth).reshape(-1).astype(np.int8)
                pd.DataFrame({"pre_update_score": scores, "post_update_score": np.concatenate(after).reshape(-1),
                              "true_label": labels, "point_index": np.arange(len(labels)),
                              "simple_used_for_update": np.concatenate(selected).reshape(-1).astype(np.int8)
                              }).to_parquet(directory / "tta_pointwise_analysis.parquet", index=False)
                save_scores(directory, {**meta, "selector_tau": tau}, scores, labels, expected_length)
    return 17
