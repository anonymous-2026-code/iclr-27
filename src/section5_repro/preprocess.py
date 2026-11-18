from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN, build_processed_manifest


@dataclass(frozen=True)
class RawSources:
    telemanom: Path
    smd: Path
    psm: Path
    tab: Path
    tsb_ad: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        telemanom: Path | None = None,
        smd: Path | None = None,
        psm: Path | None = None,
        tab: Path | None = None,
        tsb_ad: Path | None = None,
    ) -> "RawSources":
        def choose(override: Path | None, candidates: list[str]) -> Path:
            if override is not None:
                return override.resolve()
            for name in candidates:
                candidate = root / name
                if candidate.exists():
                    return candidate.resolve()
            return (root / candidates[0]).resolve()

        return cls(
            telemanom=choose(telemanom, ["telemanom", "SMAP_MSL"]),
            smd=choose(smd, ["smd", "ServerMachineDataset"]),
            psm=choose(psm, ["psm", "PSM"]),
            tab=choose(tab, ["tab", "TAB"]),
            tsb_ad=choose(tsb_ad, ["tsb-ad", "TSB-AD", "tsb_ad"]),
        )


def _clean_pair(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = train_data.copy()
    test = test_data.copy()
    if label_column not in train:
        train[label_column] = 0
    if label_column not in test:
        raise ValueError(f"Test frame lacks {label_column!r}")
    train_labels = train.pop(label_column).reset_index(drop=True)
    test_labels = test.pop(label_column).reset_index(drop=True)

    valid_columns: list[object] = []
    dropped: dict[str, str] = {}
    for column in train.columns:
        series = train[column]
        if series.isna().all():
            dropped[str(column)] = "all_nan"
            continue
        if pd.api.types.is_numeric_dtype(series):
            valid_columns.append(column)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if not coerced.isna().all():
            valid_columns.append(column)
        else:
            dropped[str(column)] = "non_numeric"

    train = train.loc[:, valid_columns].copy()
    for column in valid_columns:
        if column not in test:
            test[column] = 0
    test = test.loc[:, valid_columns].copy()
    for frame in (train, test):
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    train = train.ffill().bfill().fillna(0)
    test = test.ffill().bfill().fillna(0)

    train_variance = train.var()
    test_variance = test.var()
    zero_variance = [
        column
        for column in valid_columns
        if np.isclose(train_variance[column], 0, atol=1e-8)
        and np.isclose(test_variance[column], 0, atol=1e-8)
    ]
    for column in zero_variance:
        dropped[str(column)] = "zero_variance_both_splits"
    if zero_variance:
        train = train.drop(columns=zero_variance)
        test = test.drop(columns=zero_variance)

    train[LABEL_COLUMN] = np.asarray(train_labels, dtype=np.int8)
    test[LABEL_COLUMN] = np.asarray(test_labels, dtype=np.int8)
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "features": train.shape[1] - 1,
        "dropped_columns": dropped,
    }
    return train, test, report


def _save_pair(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train.to_parquet(train_path, engine="pyarrow", index=False)
    test.to_parquet(test_path, engine="pyarrow", index=False)
    (output_dir / "preprocess_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def preprocess_telemanom(raw: Path, output_root: Path) -> list[str]:
    label_path = raw / "labeled_anomalies.csv"
    train_dir = raw / "train"
    test_dir = raw / "test"
    if not label_path.is_file():
        raise FileNotFoundError(f"Telemanom labels missing: {label_path}")
    labels = pd.read_csv(label_path)
    labels["source_chan_id"] = labels["chan_id"]
    p2_rows = labels.index[
        (labels["spacecraft"] == "SMAP") & (labels["chan_id"] == "P-2")
    ].tolist()
    if len(p2_rows) >= 2:
        labels.loc[p2_rows[0], "chan_id"] = "P-2-0"
        labels.loc[p2_rows[0], "source_chan_id"] = "P-2"

    completed: list[str] = []
    for _, row in labels.iterrows():
        dataset = str(row["spacecraft"])
        subset = str(row["chan_id"])
        source_subset = str(row["source_chan_id"])
        x_train = np.load(train_dir / f"{source_subset}.npy")
        x_test = np.load(test_dir / f"{source_subset}.npy")
        columns = [str(index) for index in range(x_train.shape[1])]
        train = pd.DataFrame(x_train, columns=columns)
        test = pd.DataFrame(x_test, columns=columns)
        train[LABEL_COLUMN] = np.zeros(len(train), dtype=np.int8)
        test_labels = np.zeros(len(test), dtype=np.int8)
        for start, end in ast.literal_eval(str(row["anomaly_sequences"])):
            start = max(0, int(start))
            end = min(len(test_labels), int(end))
            test_labels[start:end] = 1
        test[LABEL_COLUMN] = test_labels
        train, test, report = _clean_pair(train, test)
        report.update(
            {
                "source": "Telemanom",
                "dataset": dataset,
                "subset": subset,
                "source_subset": source_subset,
            }
        )
        _save_pair(output_root / dataset / subset, train, test, report)
        completed.append(f"{dataset}/{subset}")
    return completed


def _load_smd_matrix(path: Path, expected_dimension: int = 38) -> np.ndarray:
    rows: list[list[np.float32]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if len(parts) != expected_dimension:
                continue
            try:
                rows.append([np.float32(value) for value in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No valid SMD rows in {path}")
    return np.asarray(rows, dtype=np.float32)


def preprocess_smd(raw: Path, output_root: Path) -> list[str]:
    train_dir = raw / "train"
    test_dir = raw / "test"
    label_dir = raw / "test_label"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"SMD train directory missing: {train_dir}")
    completed: list[str] = []
    for train_path in sorted(train_dir.glob("machine-*.txt")):
        server = train_path.stem
        subset = server.replace("machine", "M", 1)
        x_train = _load_smd_matrix(train_path)
        x_test = _load_smd_matrix(test_dir / train_path.name)
        labels = np.loadtxt(label_dir / train_path.name, dtype=np.int64)
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(x_test):
            raise ValueError(f"SMD label length mismatch for {server}")
        columns = [str(index) for index in range(x_train.shape[1])]
        train = pd.DataFrame(x_train, columns=columns)
        test = pd.DataFrame(x_test, columns=columns)
        train[LABEL_COLUMN] = 0
        test[LABEL_COLUMN] = (labels != 0).astype(np.int8)
        train, test, report = _clean_pair(train, test)
        report.update({"source": "OmniAnomaly", "dataset": "SMD", "subset": subset})
        _save_pair(output_root / "SMD" / subset, train, test, report)
        completed.append(f"SMD/{subset}")
    if len(completed) != 28:
        raise ValueError(f"Expected 28 SMD entities, processed {len(completed)}")
    return completed


def _time_columns(frame: pd.DataFrame) -> list[object]:
    columns = []
    for column in frame.columns:
        normalized = (
            str(column).strip().lower().replace(" ", "").replace("-", "_")
        )
        if (
            "timestamp" in normalized
            or normalized in {"date", "time"}
            or normalized.startswith("date_")
            or normalized.startswith("time_")
        ):
            columns.append(column)
    return columns


def preprocess_psm(raw: Path, output_root: Path) -> list[str]:
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "test_label.csv")
    train = train.drop(columns=_time_columns(train))
    test = test.drop(columns=_time_columns(test))
    train[LABEL_COLUMN] = 0
    label_values = (
        labels[LABEL_COLUMN].to_numpy()
        if LABEL_COLUMN in labels
        else labels.iloc[:, 0].to_numpy()
    )
    if len(label_values) != len(test):
        raise ValueError("PSM test/label row count mismatch")
    test[LABEL_COLUMN] = (label_values != 0).astype(np.int8)
    train, test, report = _clean_pair(train, test)
    report.update({"source": "RANSynCoders", "dataset": "PSM", "subset": ""})
    _save_pair(output_root / "PSM", train, test, report)
    return ["PSM"]


def _tab_wide(path: Path) -> pd.DataFrame:
    long_frame = pd.read_csv(path)
    required = {"date", "cols", "data"}
    if not required.issubset(long_frame.columns):
        raise ValueError(f"{path} lacks TAB long-format columns {sorted(required)}")
    independent_label = next(
        (
            column
            for column in long_frame.columns
            if str(column).lower()
            in {
                "label",
                "attack",
                "normal/attack",
                "machine_status",
                "class",
                "is_anomaly",
            }
            and column not in required
        ),
        None,
    )
    labels = None
    if independent_label is not None:
        labels = (
            long_frame[["date", independent_label]]
            .drop_duplicates()
            .set_index("date")
        )
    wide = long_frame.pivot(index="date", columns="cols", values="data").reset_index()
    if labels is not None:
        wide = wide.set_index("date").join(labels).reset_index()
        wide = wide.rename(columns={independent_label: LABEL_COLUMN})
    if LABEL_COLUMN not in wide:
        candidates = [
            column
            for column in wide.columns
            if str(column).lower()
            in {"label", "attack", "normal/attack", "machine_status", "class", "is_anomaly"}
            or "ttack" in str(column)
            or "abel" in str(column)
        ]
        if candidates:
            wide = wide.rename(columns={candidates[0]: LABEL_COLUMN})
    if LABEL_COLUMN not in wide:
        raise ValueError(f"No label series found in TAB file {path}")
    try:
        wide["date"] = pd.to_datetime(wide["date"])
    except Exception:
        pass
    return wide.sort_values("date").reset_index(drop=True).drop(columns=["date"])


def _binary_labels(values: Iterable[object]) -> np.ndarray:
    if isinstance(values, (np.ndarray, pd.Series)):
        array = np.asarray(values)
    else:
        array = np.asarray(list(values))
    if np.issubdtype(array.dtype, np.number):
        return (array != 0).astype(np.int8)
    strings = np.asarray([str(value).strip().upper() for value in array])
    return (~np.isin(strings, ["NORMAL", "0", "BENIGN", "FALSE"])).astype(np.int8)


def preprocess_swat(raw: Path, output_root: Path) -> list[str]:
    wide = _tab_wide(raw / "SWAT.csv")
    split = 495_000
    if len(wide) <= split:
        raise ValueError(f"TAB SWAT has only {len(wide)} rows")
    train = wide.iloc[:split].copy()
    test = wide.iloc[split:].copy()
    train[LABEL_COLUMN] = 0
    test[LABEL_COLUMN] = _binary_labels(test[LABEL_COLUMN].to_numpy())
    train, test, report = _clean_pair(train, test)
    report.update({"source": "TAB", "dataset": "SWAT2015", "subset": ""})
    _save_pair(output_root / "SWAT2015", train, test, report)
    return ["SWAT2015"]


def preprocess_pump(raw: Path, output_root: Path) -> list[str]:
    wide = _tab_wide(raw / "PUMP.csv")
    split = int(len(wide) * 0.5)
    train = wide.iloc[:split].copy()
    test = wide.iloc[split:].copy()
    train[LABEL_COLUMN] = 0
    test[LABEL_COLUMN] = _binary_labels(test[LABEL_COLUMN].to_numpy())
    train, test, report = _clean_pair(train, test)
    report.update({"source": "TAB", "dataset": "PUMP", "subset": ""})
    _save_pair(output_root / "PUMP", train, test, report)
    return ["PUMP"]


_TSB_FILES = {
    "Daphnet": "018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv",
    "GECCO": "173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv",
    "Genesis": "001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv",
}


def preprocess_tsb_dataset(raw: Path, output_root: Path, dataset: str) -> list[str]:
    filename = _TSB_FILES[dataset]
    path = raw / filename
    match = re.search(r"_tr_(\d+)_", filename)
    if match is None:
        raise ValueError(f"Cannot parse TSB-AD split from {filename}")
    split = int(match.group(1))
    frame = pd.read_csv(path)
    if split <= 0 or split >= len(frame):
        raise ValueError(f"Invalid TSB-AD split {split} for {filename}")
    raw_label = frame.columns[-1]
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    test_labels = (test[raw_label].to_numpy() != 0).astype(np.int8)
    train = train.drop(columns=[raw_label])
    test = test.drop(columns=[raw_label])
    train[LABEL_COLUMN] = 0
    test[LABEL_COLUMN] = test_labels
    train, test, report = _clean_pair(train, test)
    report.update(
        {
            "source": "TSB-AD",
            "dataset": dataset,
            "subset": "",
            "raw_filename": filename,
        }
    )
    _save_pair(output_root / dataset, train, test, report)
    return [dataset]


PREPROCESS_ORDER = (
    "MSL",
    "SMAP",
    "SMD",
    "PSM",
    "SWAT2015",
    "PUMP",
    "Daphnet",
    "GECCO",
    "Genesis",
)


def preprocess_all(
    sources: RawSources,
    output_root: Path,
    datasets: Iterable[str] | None = None,
) -> dict[str, object]:
    selected = set(datasets or PREPROCESS_ORDER)
    unknown = selected - set(PREPROCESS_ORDER)
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    completed: list[str] = []
    if selected.intersection({"MSL", "SMAP"}):
        telemanom_completed = preprocess_telemanom(sources.telemanom, output_root)
        completed.extend(
            name for name in telemanom_completed if name.split("/", 1)[0] in selected
        )
    if "SMD" in selected:
        completed.extend(preprocess_smd(sources.smd, output_root))
    if "PSM" in selected:
        completed.extend(preprocess_psm(sources.psm, output_root))
    if "SWAT2015" in selected:
        completed.extend(preprocess_swat(sources.tab, output_root))
    if "PUMP" in selected:
        completed.extend(preprocess_pump(sources.tab, output_root))
    for dataset in ("Daphnet", "GECCO", "Genesis"):
        if dataset in selected:
            completed.extend(preprocess_tsb_dataset(sources.tsb_ad, output_root, dataset))

    manifest = build_processed_manifest(output_root, datasets=sorted(selected))
    manifest["completed"] = completed
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
