from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from section5_repro.data import build_processed_manifest, iter_manifest_differences
from section5_repro.preprocess import RawSources, preprocess_all

from .config import _load_json_yaml, repository_root


BUNDLE_PREFIXES = {
    "telemanom": "telemanom",
    "smd": "smd",
    "psm": "psm",
    "tab": "tab",
    "tsb_ad_m": "tsb-ad",
}

SOURCE_ALIASES = {
    "telemanom": ("telemanom", "SMAP_MSL"),
    "smd": ("smd", "SMD", "ServerMachineDataset"),
    "psm": ("psm", "PSM"),
    "tab": ("tab", "TAB"),
    "tsb-ad": ("tsb-ad", "TSB-AD", "tsb_ad"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_sources(path: str | Path | None = None) -> dict[str, Any]:
    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else repository_root() / "configs" / "data_sources.yaml"
    )
    payload = _load_json_yaml(config_path)
    if payload.get("schema_version") != "iclr27-data-sources-v1":
        raise ValueError("Unsupported data source schema")
    return payload


def _selected_bundles(
    config: Mapping[str, Any], datasets: Iterable[str]
) -> tuple[str, ...]:
    selected = set(datasets)
    bundles = tuple(
        name
        for name, bundle in config["bundles"].items()
        if selected.intersection(bundle["datasets"])
    )
    covered = {
        dataset
        for name in bundles
        for dataset in config["bundles"][name]["datasets"]
        if dataset in selected
    }
    if covered != selected:
        raise ValueError(f"No data source mapping for {sorted(selected - covered)}")
    return bundles


def _actual_files(root: Path, prefixes: Iterable[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for prefix in prefixes:
        aliases = SOURCE_ALIASES[prefix]
        source = next(
            (root / alias for alias in aliases if (root / alias).is_dir()),
            root / prefix,
        )
        if not source.is_dir():
            continue
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            files[f"{prefix}/{path.relative_to(source).as_posix()}"] = path
    return files


def validate_raw_data(
    raw_root: Path,
    *,
    datasets: Iterable[str],
    config_path: str | Path | None = None,
    strict_unexpected: bool = True,
) -> dict[str, Any]:
    root = raw_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw-data root does not exist: {root}")
    config = load_data_sources(config_path)
    selected = tuple(dict.fromkeys(str(item) for item in datasets))
    bundles = _selected_bundles(config, selected)
    prefixes = tuple(BUNDLE_PREFIXES[name] for name in bundles)
    manifest_path = repository_root() / str(config["raw_manifest"])
    if sha256_file(manifest_path) != config["raw_manifest_sha256"]:
        raise ValueError("Committed raw manifest checksum differs from data_sources.yaml")
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = set(config.get("excluded_legacy_snapshot_non_inputs", ()))
    expected = {
        str(row["path"]): row
        for row in legacy["files"]
        if str(row["path"]).split("/", 1)[0] in prefixes
        and str(row["path"]) not in excluded
    }
    actual = _actual_files(root, prefixes)
    problems: list[str] = []
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"missing: {name}")
    if strict_unexpected:
        for name in sorted(set(actual) - set(expected) - excluded):
            problems.append(f"unexpected: {name}")
    checked_bytes = 0
    for name in sorted(set(expected).intersection(actual)):
        record = expected[name]
        path = actual[name]
        size = path.stat().st_size
        checked_bytes += size
        if size != int(record["bytes"]):
            problems.append(
                f"size mismatch: {name} expected={record['bytes']} actual={size}"
            )
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            problems.append(f"SHA-256 mismatch: {name}")
    status = "pass" if not problems else "fail"
    report = {
        "schema_version": "iclr27-raw-validation-v1",
        "status": status,
        "raw_root": str(root),
        "datasets": list(selected),
        "bundles": list(bundles),
        "expected_file_count": len(expected),
        "matched_file_count": len(set(expected).intersection(actual)),
        "checked_bytes": checked_bytes,
        "excluded_non_inputs": sorted(excluded),
        "strict_unexpected": strict_unexpected,
        "problems": problems,
    }
    if problems:
        preview = "\n".join(f"- {item}" for item in problems[:100])
        raise ValueError(f"Raw-data validation failed:\n{preview}")
    return report


def _filtered_reference(
    manifest: Mapping[str, Any], datasets: Iterable[str]
) -> dict[str, Any]:
    selected = set(datasets)
    rows = [
        row for row in manifest.get("entities", ()) if row.get("dataset") in selected
    ]
    return {"format": manifest.get("format"), "entity_count": len(rows), "entities": rows}


def verify_processed_data(
    processed_root: Path,
    *,
    datasets: Iterable[str],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = processed_root.expanduser().resolve()
    config = load_data_sources(config_path)
    manifest_path = repository_root() / str(config["processed_manifest"])
    if sha256_file(manifest_path) != config["processed_manifest_sha256"]:
        raise ValueError("Committed processed manifest checksum differs from data_sources.yaml")
    expected = _filtered_reference(
        json.loads(manifest_path.read_text(encoding="utf-8")), datasets
    )
    actual = build_processed_manifest(root, datasets=datasets)
    differences = list(iter_manifest_differences(actual, expected))
    if differences:
        preview = "\n".join(f"- {item}" for item in differences[:100])
        raise ValueError(f"Processed-data parity failed:\n{preview}")
    return {
        "schema_version": "iclr27-processed-validation-v1",
        "status": "pass",
        "processed_root": str(root),
        "datasets": list(datasets),
        "entity_count": actual["entity_count"],
        "reference_manifest_sha256": config["processed_manifest_sha256"],
    }


def prepare_data(
    raw_root: Path,
    processed_root: Path,
    *,
    datasets: Iterable[str],
    resume: bool,
) -> dict[str, Any]:
    raw = raw_root.expanduser().resolve()
    processed = processed_root.expanduser().resolve()
    selected = tuple(dict.fromkeys(str(item) for item in datasets))
    validate_raw_data(raw, datasets=selected)
    if processed.exists():
        if not resume:
            raise FileExistsError(
                f"Processed target exists and --no-resume forbids reuse: {processed}"
            )
        verified = verify_processed_data(processed, datasets=selected)
        return {**verified, "resumed": True}
    sources = RawSources.from_root(raw)
    manifest = preprocess_all(sources, processed, datasets=selected)
    verified = verify_processed_data(processed, datasets=selected)
    return {**verified, "resumed": False, "completed": manifest.get("completed", [])}
