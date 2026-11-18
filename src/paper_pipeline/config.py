from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ScopeValidationError(ValueError):
    """Raised when a paper-facing result contract is internally inconsistent."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json_yaml(path: Path) -> dict[str, Any]:
    """Load the repository's JSON-syntax YAML without an optional dependency.

    JSON is a strict subset of YAML 1.2. Keeping paper contracts in this subset
    makes exact hashing and clean-checkout parsing independent of PyYAML.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeValidationError(f"Cannot load JSON-syntax YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScopeValidationError(f"Top-level scope must be a mapping: {path}")
    return payload


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _product(values: Mapping[str, Any], *, profile_id: str, kind: str) -> int:
    result = 1
    if not values:
        raise ScopeValidationError(f"{profile_id}: empty {kind} count basis")
    for name, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ScopeValidationError(
                f"{profile_id}: {kind}.{name} must be a non-negative integer"
            )
        result *= raw
    return result


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    main_required: bool
    paper_section: str
    description: str
    logical_cell_definition: str
    expected_logical_cells: int
    expected_physical_runs: int
    logical_count_basis: Mapping[str, int]
    physical_count_basis: Mapping[str, int]
    output_artifact_ids: tuple[str, ...]
    reuses_profiles: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class PaperScope:
    path: Path
    schema_version: str
    profile_id: str
    datasets: tuple[str, ...]
    dataset_entity_counts: Mapping[str, int]
    seeds: tuple[int, ...]
    optimizer_ids: tuple[str, ...]
    metrics: tuple[str, ...]
    profiles: tuple[ProfileSpec, ...]
    artifacts: Mapping[str, str]
    expected_logical_cells: int
    expected_physical_runs: int
    config_hash: str
    raw: Mapping[str, Any]

    @property
    def entity_count(self) -> int:
        return sum(self.dataset_entity_counts.values())

    def profile(self, profile_id: str) -> ProfileSpec:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(profile_id)


def _profile(raw: Mapping[str, Any], artifact_ids: set[str]) -> ProfileSpec:
    required = {
        "id",
        "main_required",
        "paper_section",
        "description",
        "logical_cell_definition",
        "expected_logical_cells",
        "expected_physical_runs",
        "count_basis",
        "output_artifact_ids",
    }
    missing = required - set(raw)
    if missing:
        raise ScopeValidationError(f"Profile missing keys: {sorted(missing)}")
    profile_id = str(raw["id"])
    basis = raw["count_basis"]
    if not isinstance(basis, dict) or set(basis) != {"logical", "physical"}:
        raise ScopeValidationError(
            f"{profile_id}: count_basis must contain logical and physical"
        )
    logical = _product(basis["logical"], profile_id=profile_id, kind="logical")
    physical = _product(basis["physical"], profile_id=profile_id, kind="physical")
    expected_logical = int(raw["expected_logical_cells"])
    expected_physical = int(raw["expected_physical_runs"])
    if logical != expected_logical:
        raise ScopeValidationError(
            f"{profile_id}: logical count {logical} != declared {expected_logical}"
        )
    if physical != expected_physical:
        raise ScopeValidationError(
            f"{profile_id}: physical count {physical} != declared {expected_physical}"
        )
    outputs = tuple(str(item) for item in raw["output_artifact_ids"])
    unknown = set(outputs) - artifact_ids
    if unknown:
        raise ScopeValidationError(
            f"{profile_id}: unknown output artifact ids {sorted(unknown)}"
        )
    return ProfileSpec(
        profile_id=profile_id,
        main_required=bool(raw["main_required"]),
        paper_section=str(raw["paper_section"]),
        description=str(raw["description"]),
        logical_cell_definition=str(raw["logical_cell_definition"]),
        expected_logical_cells=expected_logical,
        expected_physical_runs=expected_physical,
        logical_count_basis=dict(basis["logical"]),
        physical_count_basis=dict(basis["physical"]),
        output_artifact_ids=outputs,
        reuses_profiles=tuple(str(item) for item in raw.get("reuses_profiles", ())),
        raw=dict(raw),
    )


def load_paper_scope(path: str | Path | None = None) -> PaperScope:
    scope_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else repository_root() / "configs" / "paper" / "iclr27_main_v1.yaml"
    )
    raw = _load_json_yaml(scope_path)
    if raw.get("schema_version") != "iclr27-paper-scope-v1":
        raise ScopeValidationError("Unsupported or missing paper scope schema_version")
    datasets = tuple(str(item) for item in raw.get("datasets", ()))
    if len(datasets) != len(set(datasets)) or not datasets:
        raise ScopeValidationError("datasets must be non-empty and unique")
    counts_raw = raw.get("dataset_entity_counts")
    if not isinstance(counts_raw, dict) or set(counts_raw) != set(datasets):
        raise ScopeValidationError("dataset_entity_counts must exactly cover datasets")
    counts = {str(key): int(value) for key, value in counts_raw.items()}
    if any(value <= 0 for value in counts.values()) or sum(counts.values()) != 116:
        raise ScopeValidationError("ICLR27 main v1 must resolve to 116 physical entities")
    seeds = tuple(int(item) for item in raw.get("seeds", ()))
    if seeds != (42, 43, 44, 45, 46):
        raise ScopeValidationError("ICLR27 main v1 requires seeds 42--46")
    optimizers_raw = raw.get("optimizers", ())
    optimizer_ids = tuple(str(item["id"]) for item in optimizers_raw)
    if len(optimizer_ids) != 4 or len(set(optimizer_ids)) != 4:
        raise ScopeValidationError("ICLR27 main v1 requires four unique optimizers")
    metrics = tuple(str(item) for item in raw.get("metrics", ()))
    if metrics != ("auprc", "vus_pr", "auroc", "vus_roc"):
        raise ScopeValidationError("Metric registry order is part of the v1 contract")
    artifacts_raw = raw.get("paper_artifacts")
    if not isinstance(artifacts_raw, dict):
        raise ScopeValidationError("paper_artifacts must be a mapping")
    artifacts = {str(key): str(value) for key, value in artifacts_raw.items()}
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ScopeValidationError("profiles must be a non-empty list")
    profiles = tuple(_profile(item, set(artifacts)) for item in profiles_raw)
    profile_ids = tuple(item.profile_id for item in profiles)
    if len(profile_ids) != len(set(profile_ids)):
        raise ScopeValidationError("profile ids must be unique")
    for profile in profiles:
        unknown = set(profile.reuses_profiles) - set(profile_ids)
        if unknown:
            raise ScopeValidationError(
                f"{profile.profile_id}: unknown reused profiles {sorted(unknown)}"
            )
    required = tuple(item for item in profiles if item.main_required)
    logical_total = sum(item.expected_logical_cells for item in required)
    physical_total = sum(item.expected_physical_runs for item in required)
    expected_totals = raw.get("expected_totals", {})
    if logical_total != int(expected_totals.get("required_logical_cells", -1)):
        raise ScopeValidationError(
            f"Required logical total {logical_total} differs from declared total"
        )
    if physical_total != int(expected_totals.get("unique_physical_runs", -1)):
        raise ScopeValidationError(
            f"Physical total {physical_total} differs from declared total"
        )
    return PaperScope(
        path=scope_path,
        schema_version=str(raw["schema_version"]),
        profile_id=str(raw["profile_id"]),
        datasets=datasets,
        dataset_entity_counts=counts,
        seeds=seeds,
        optimizer_ids=optimizer_ids,
        metrics=metrics,
        profiles=profiles,
        artifacts=artifacts,
        expected_logical_cells=logical_total,
        expected_physical_runs=physical_total,
        config_hash=canonical_hash(raw),
        raw=raw,
    )
