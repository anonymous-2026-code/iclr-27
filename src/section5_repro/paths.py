from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .data import Entity
from .protocol import Protocol, TTAConfig


@dataclass(frozen=True)
class Workspace:
    root: Path
    checkpoint_root: Path | None = None

    @classmethod
    def create(
        cls,
        root: Path,
        checkpoint_root: Path | None = None,
    ) -> "Workspace":
        resolved = root.resolve()
        resolved_checkpoints = (
            checkpoint_root.resolve()
            if checkpoint_root is not None
            else resolved / "checkpoints"
        )
        resolved_checkpoints.mkdir(parents=True, exist_ok=True)
        for child in ("results", "logs", "manifests", "reports"):
            (resolved / child).mkdir(parents=True, exist_ok=True)
        return cls(root=resolved, checkpoint_root=resolved_checkpoints)

    @property
    def checkpoints(self) -> Path:
        return self.checkpoint_root or self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def source_setting(entity: Entity, seed: int, protocol: Protocol) -> str:
    return (
        f"{protocol.model_id}_{protocol.model_name}_{entity.dataset}"
        f"_sub{entity.subset}"
        f"_sl{protocol.sequence_length}"
        f"_st{protocol.train_stride}"
        f"_bs{protocol.train_batch_size}"
        f"_ep{protocol.train_epochs}"
        f"_lr{protocol.source_learning_rate}"
        f"_seed{int(seed)}"
    )


def tta_setting(
    entity: Entity,
    seed: int,
    protocol: Protocol,
    config: TTAConfig,
    oracle: bool,
) -> str:
    upper_bound = "_UB" if oracle else ""
    scope = (
        "" if protocol.update_scope == "full_model"
        else f"_scope{protocol.update_scope}"
    )
    return (
        f"{source_setting(entity, seed, protocol)}"
        f"_TTA_simple{upper_bound}"
        f"_ttlr{config.learning_rate}"
        f"_opt{config.optimizer}"
        f"_tbs{protocol.tta_test_batch_size}"
        f"{scope}"
    )


def checkpoint_path(
    workspace: Workspace,
    entity: Entity,
    seed: int,
    protocol: Protocol,
) -> Path:
    return (
        workspace.checkpoints
        / source_setting(entity, seed, protocol)
        / "checkpoint.pth"
    )


def frozen_result_dir(
    workspace: Workspace,
    entity: Entity,
    seed: int,
    protocol: Protocol,
) -> Path:
    return workspace.results / source_setting(entity, seed, protocol)


def tta_result_dir(
    workspace: Workspace,
    entity: Entity,
    seed: int,
    protocol: Protocol,
    config: TTAConfig,
    oracle: bool,
) -> Path:
    return workspace.results / tta_setting(entity, seed, protocol, config, oracle)


def safe_log_stem(entity: Entity, seed: int) -> str:
    raw = f"{entity.dataset}_{entity.subset or 'single'}_seed{seed}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def assert_owned_path(path: Path, workspace: Workspace) -> None:
    resolved = path.resolve()
    allowed = (workspace.root.resolve(), workspace.checkpoints.resolve())
    for root in allowed:
        try:
            resolved.relative_to(root)
            return
        except ValueError:
            continue
    raise ValueError(
        f"Refusing path outside repository workspace/checkpoint roots: {resolved}"
    )
