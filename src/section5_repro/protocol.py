from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class TTAConfig:
    key: str
    optimizer: str
    learning_rate: float


@dataclasses.dataclass(frozen=True)
class Protocol:
    protocol_id: str
    source_snapshot: str
    model_id: str
    model_name: str
    datasets: tuple[str, ...]
    multi_subset_datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    sequence_length: int
    train_stride: int
    test_stride: int
    train_batch_size: int
    frozen_test_batch_size: int
    tta_test_batch_size: int
    train_epochs: int
    early_stopping_patience: int
    source_learning_rate: float
    source_optimizer: str
    source_lr_schedule: str
    train_fraction: float
    selector_quantile: float
    update_scope: str
    clear_gain_pp: float
    tta_configs: tuple[TTAConfig, ...]

    def is_multi_subset(self, dataset: str) -> bool:
        return dataset in self.multi_subset_datasets

    def tta_by_key(self, key: str) -> TTAConfig:
        for config in self.tta_configs:
            if config.key == key:
                return config
        raise KeyError(f"Unknown TTA config {key!r}")

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["tta_configs"] = [dataclasses.asdict(item) for item in self.tta_configs]
        return payload


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_protocol_path() -> Path:
    return repository_root() / "configs" / "section5_ae.toml"


def load_protocol(path: str | Path | None = None) -> Protocol:
    config_path = Path(path) if path is not None else default_protocol_path()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    tta = tuple(
        TTAConfig(
            key=str(item["key"]),
            optimizer=str(item["optimizer"]).lower(),
            learning_rate=float(item["learning_rate"]),
        )
        for item in raw.pop("tta_config")
    )
    datasets = tuple(raw.pop("datasets"))
    multi_subset_datasets = tuple(raw.pop("multi_subset_datasets"))
    seeds = tuple(int(seed) for seed in raw.pop("seeds"))
    protocol = Protocol(
        **raw,
        datasets=datasets,
        multi_subset_datasets=multi_subset_datasets,
        seeds=seeds,
        tta_configs=tta,
    )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Protocol) -> None:
    exact_fields = {
        "protocol_id": "section5-ae-paper-results-v1",
        "source_snapshot": "anonymous-submission-v1",
        "model_id": "base_win_ae",
        "model_name": "win_ae",
        "datasets": (
            "MSL",
            "SMAP",
            "SMD",
            "PSM",
            "SWAT2015",
            "PUMP",
            "Daphnet",
            "GECCO",
            "Genesis",
        ),
        "multi_subset_datasets": ("MSL", "SMAP", "SMD"),
        "seeds": (42, 43, 44, 45, 46),
        "sequence_length": 12,
        "train_stride": 12,
        "test_stride": 12,
        "train_batch_size": 64,
        "frozen_test_batch_size": 64,
        "tta_test_batch_size": 1,
        "train_epochs": 10,
        "early_stopping_patience": 3,
        "source_optimizer": "adam",
        "source_lr_schedule": "type1",
        "update_scope": "full_model",
    }
    for field, expected_value in exact_fields.items():
        actual_value = getattr(protocol, field)
        if actual_value != expected_value:
            raise ValueError(
                f"Locked paper protocol field {field!r}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )
    exact_floats = {
        "source_learning_rate": 1e-4,
        "train_fraction": 0.8,
        "selector_quantile": 0.95,
        "clear_gain_pp": 0.5,
    }
    for field, expected_value in exact_floats.items():
        actual_value = getattr(protocol, field)
        if not math.isclose(
            actual_value, expected_value, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                f"Locked paper protocol field {field!r}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )
    expected = {
        ("sgd", 1e-3),
        ("sgd", 1e-4),
        ("adamw", 1e-4),
        ("adamw", 3e-4),
    }
    actual = {(item.optimizer, item.learning_rate) for item in protocol.tta_configs}
    if len(protocol.tta_configs) != 4 or actual != expected:
        raise ValueError("The paper protocol requires exactly four optimizer configurations.")
    expected_keys = (
        "sgd_ttlr0.001",
        "sgd_ttlr0.0001",
        "adamw_ttlr0.0001",
        "adamw_ttlr0.0003",
    )
    if tuple(item.key for item in protocol.tta_configs) != expected_keys:
        raise ValueError(
            "The paper protocol requires the locked optimizer key order."
        )
