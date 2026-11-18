from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


LABEL_COLUMN = "label"
MULTI_SUBSET_DATASETS = frozenset({"MSL", "SMAP", "SMD"})


@dataclass(frozen=True, order=True)
class Entity:
    dataset: str
    subset: str
    directory: Path

    @property
    def display_name(self) -> str:
        return f"{self.dataset}/{self.subset}" if self.subset else self.dataset


@dataclass(frozen=True)
class PreparedSeries:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    full_source: np.ndarray
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


class WindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        stride: int,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.values.ndim != 2:
            raise ValueError(f"Expected 2-D feature matrix, got {self.values.shape}")
        if self.labels.shape != (len(self.values),):
            raise ValueError(
                f"Labels must have shape ({len(self.values)},), got {self.labels.shape}"
            )
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")

    def __len__(self) -> int:
        if len(self.values) < self.window_size:
            return 0
        return (len(self.values) - self.window_size) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(index) * self.stride
        stop = start + self.window_size
        return self.values[start:stop], self.labels[start:stop]


def read_processed_pair(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = directory / "train.parquet"
    test_path = directory / "test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Missing train.parquet/test.parquet under {directory}"
        )
    train = pd.read_parquet(train_path, engine="pyarrow")
    test = pd.read_parquet(test_path, engine="pyarrow")
    if test.shape[1] < 2:
        raise ValueError(f"{test_path} must contain features plus a label column")
    if str(test.columns[-1]).lower() != LABEL_COLUMN:
        raise ValueError(f"{test_path}: last column must be named 'label'")
    if str(train.columns[-1]).lower() != LABEL_COLUMN:
        raise ValueError(f"{train_path}: last column must be named 'label'")
    if list(train.columns) != list(test.columns):
        raise ValueError(f"Train/test columns differ under {directory}")
    return train, test


def prepare_series(directory: Path, train_fraction: float = 0.8) -> PreparedSeries:
    train_df, test_df = read_processed_pair(directory)
    feature_names = tuple(str(column) for column in train_df.columns[:-1])
    source_raw = train_df.iloc[:, :-1].to_numpy()
    test_raw = test_df.iloc[:, :-1].to_numpy()
    labels = test_df.iloc[:, -1].to_numpy(dtype=np.float32)
    if not np.isfinite(source_raw).all() or not np.isfinite(test_raw).all():
        raise ValueError(f"Non-finite feature value found under {directory}")
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError(f"Non-binary test labels found under {directory}")

    # This intentionally fits on the complete official source-side normal series,
    # matching the executable used for the paper.
    scaler = StandardScaler()
    scaler.fit(source_raw)
    source = scaler.transform(source_raw).astype(np.float32)
    test = scaler.transform(test_raw).astype(np.float32)
    split = int(len(source) * float(train_fraction))
    if split <= 0 or split >= len(source):
        raise ValueError(f"Invalid chronological source split for {directory}")
    return PreparedSeries(
        train=source[:split],
        validation=source[split:],
        test=test,
        test_labels=labels,
        full_source=source,
        feature_names=feature_names,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
    )


def discover_entities(
    processed_root: Path,
    datasets: Iterable[str] | None = None,
) -> list[Entity]:
    requested = list(datasets) if datasets is not None else [
        "MSL",
        "SMAP",
        "SMD",
        "PSM",
        "SWAT2015",
        "PUMP",
        "Daphnet",
        "GECCO",
        "Genesis",
    ]
    entities: list[Entity] = []
    for dataset in requested:
        dataset_dir = processed_root / dataset
        if dataset in MULTI_SUBSET_DATASETS:
            if not dataset_dir.is_dir():
                raise FileNotFoundError(f"Processed dataset directory missing: {dataset_dir}")
            subset_dirs = sorted(
                path
                for path in dataset_dir.iterdir()
                if path.is_dir()
                and (path / "train.parquet").is_file()
                and (path / "test.parquet").is_file()
            )
            if not subset_dirs:
                raise FileNotFoundError(f"No processed subsets found under {dataset_dir}")
            entities.extend(
                Entity(dataset=dataset, subset=path.name, directory=path)
                for path in subset_dirs
            )
        else:
            if not (dataset_dir / "train.parquet").is_file():
                raise FileNotFoundError(f"Processed dataset missing: {dataset_dir}")
            entities.append(Entity(dataset=dataset, subset="", directory=dataset_dir))
    return entities


def semantic_frame_hash(frame: pd.DataFrame) -> str:
    """Hash values rather than Parquet bytes, which vary across PyArrow releases."""
    if str(frame.columns[-1]).lower() != LABEL_COLUMN:
        raise ValueError("Semantic hashing expects label as the final column")
    features = np.ascontiguousarray(
        frame.iloc[:, :-1].to_numpy(dtype="<f8", copy=True)
    )
    labels = np.ascontiguousarray(
        frame.iloc[:, -1].to_numpy(dtype=np.int8, copy=True)
    )
    digest = hashlib.sha256()
    digest.update(b"section5-semantic-frame-v1\0")
    digest.update(json.dumps(list(map(str, frame.columns)), ensure_ascii=True).encode())
    digest.update(np.asarray(features.shape, dtype="<i8").tobytes())
    digest.update(features.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=256)
def entity_manifest(entity: Entity) -> dict[str, object]:
    train, test = read_processed_pair(entity.directory)
    test_labels = test.iloc[:, -1].to_numpy(dtype=np.int8)
    return {
        "dataset": entity.dataset,
        "subset": entity.subset,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": int(train.shape[1] - 1),
        "test_anomalies": int(test_labels.sum()),
        "train_semantic_sha256": semantic_frame_hash(train),
        "test_semantic_sha256": semantic_frame_hash(test),
    }


def build_processed_manifest(
    processed_root: Path,
    datasets: Iterable[str] | None = None,
) -> dict[str, object]:
    entities = discover_entities(processed_root, datasets=datasets)
    records = [entity_manifest(entity) for entity in entities]
    return {
        "format": "section5-processed-manifest-v1",
        "entity_count": len(records),
        "entities": records,
    }


def iter_manifest_differences(
    actual: dict[str, object],
    expected: dict[str, object],
) -> Iterator[str]:
    actual_rows = {
        (str(row["dataset"]), str(row["subset"])): row
        for row in actual.get("entities", [])
    }
    expected_rows = {
        (str(row["dataset"]), str(row["subset"])): row
        for row in expected.get("entities", [])
    }
    for key in sorted(set(actual_rows) | set(expected_rows)):
        if key not in actual_rows:
            yield f"missing entity: {key[0]}/{key[1]}"
            continue
        if key not in expected_rows:
            yield f"unexpected entity: {key[0]}/{key[1]}"
            continue
        for field in (
            "train_rows",
            "test_rows",
            "features",
            "test_anomalies",
            "train_semantic_sha256",
            "test_semantic_sha256",
        ):
            if actual_rows[key].get(field) != expected_rows[key].get(field):
                yield (
                    f"{key[0]}/{key[1]} {field}: "
                    f"actual={actual_rows[key].get(field)!r}, "
                    f"expected={expected_rows[key].get(field)!r}"
                )
