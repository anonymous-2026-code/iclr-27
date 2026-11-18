from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Any

from .protocol import Protocol, load_protocol, repository_root


EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7")


@dataclasses.dataclass(frozen=True)
class E1Protocol:
    mask_seed: int
    fractions: tuple[float, ...]
    perturbations: tuple[str, ...]
    main_fraction: float
    main_optimizer_key: str
    auxiliary_optimizer_key: str
    auxiliary_seeds: tuple[int, ...]
    representative_datasets: tuple[str, ...]
    baseline_unavailable_datasets: tuple[str, ...]
    baseline_unavailable_optimizer_key: str
    baseline_unavailable_seeds: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class E2Protocol:
    static_percentiles: tuple[float, ...]
    expanding_percentile: float
    bottom_k: int
    m2n2_percentile: float
    gamma: dict[str, float]


@dataclasses.dataclass(frozen=True)
class E3Protocol:
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    cadences: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FPMProtocol:
    score_percentile: float
    hard_tail_percent: float
    moderate_low_percentile: float
    moderate_high_percentile: float
    chi2_probability: float
    covariance_regularization: float
    inverse_policy: str


@dataclasses.dataclass(frozen=True)
class E5Protocol:
    vus_window: int
    rank_thresholds: int
    alignment_policy: str


@dataclasses.dataclass(frozen=True)
class E6Protocol:
    cadences: tuple[int, ...]
    optional_unrun_cadences: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class E7Protocol:
    arms: tuple[str, ...]
    d_model: int
    heads: int
    d_ff: int
    sana_seed_offset: int


@dataclasses.dataclass(frozen=True)
class SupplementProtocol:
    protocol_id: str
    source_protocol_id: str
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    sequence_length: int
    train_stride: int
    test_stride: int
    optimizer_key: str
    gain_threshold_pp: float
    e1: E1Protocol
    e2: E2Protocol
    e3: E3Protocol
    fpm: FPMProtocol
    e5: E5Protocol
    e6: E6Protocol
    e7: E7Protocol

    def source_protocol(self) -> Protocol:
        source = load_protocol()
        if source.protocol_id != self.source_protocol_id:
            raise ValueError(
                f"Supplement/source protocol mismatch: {self.source_protocol_id!r} "
                f"!= {source.protocol_id!r}"
            )
        return source

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate_experiment(self, experiment: str) -> str:
        name = experiment.strip().lower()
        if name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown supplement {experiment!r}; choose one of {EXPERIMENTS}"
            )
        return name

    def task_is_eligible(self, experiment: str, dataset: str, seed: int) -> bool:
        name = self.validate_experiment(experiment)
        if dataset not in self.datasets or seed not in self.seeds:
            return False
        if name == "e3":
            return dataset in self.e3.datasets and seed in self.e3.seeds
        return name != "e5"


@dataclasses.dataclass(frozen=True)
class SupplementVariant:
    name: str
    optimizer_key: str | None = None
    perturbation: str | None = None
    fraction: float | None = None
    selector: str | None = None
    cadence: int | None = None
    oracle: bool | None = None
    arm: str | None = None


def _fraction_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def e1_configuration_unavailable(
    protocol: SupplementProtocol,
    dataset: str,
    seed: int,
    optimizer_key: str,
) -> bool:
    """Whether the canonical E1 target records a baseline-NF logical cell."""
    return (
        dataset in protocol.e1.baseline_unavailable_datasets
        and int(seed) in protocol.e1.baseline_unavailable_seeds
        and optimizer_key == protocol.e1.baseline_unavailable_optimizer_key
    )


def variants_for(
    protocol: SupplementProtocol,
    experiment: str,
    dataset: str,
    seed: int,
    *,
    include_known_unavailable: bool = False,
) -> tuple[SupplementVariant, ...]:
    """Return only newly executed arms; reused Section 5 controls are dependencies."""
    name = protocol.validate_experiment(experiment)
    if not protocol.task_is_eligible(name, dataset, seed):
        return ()
    if name == "e1":
        rows: list[SupplementVariant] = []
        optimizers = [protocol.e1.main_optimizer_key]
        if seed in protocol.e1.auxiliary_seeds:
            optimizers.append(protocol.e1.auxiliary_optimizer_key)
        for fraction in protocol.e1.fractions:
            if (
                math.isclose(fraction, 0.05)
                and seed not in protocol.e1.auxiliary_seeds
            ):
                continue
            for perturbation in protocol.e1.perturbations:
                if (
                    math.isclose(fraction, 0.05)
                    and perturbation != "fn_contamination"
                    and dataset not in protocol.e1.representative_datasets
                ):
                    continue
                for optimizer in optimizers:
                    if (
                        not include_known_unavailable
                        and e1_configuration_unavailable(
                            protocol, dataset, seed, optimizer
                        )
                    ):
                        continue
                    rows.append(
                        SupplementVariant(
                            name=(
                                f"{optimizer}__{perturbation}"
                                f"__frac{_fraction_tag(fraction)}"
                            ),
                            optimizer_key=optimizer,
                            perturbation=perturbation,
                            fraction=fraction,
                        )
                    )
        return tuple(rows)
    if name == "e2":
        variants = [
            SupplementVariant(
                "static_p99_practical",
                protocol.optimizer_key,
                selector="static_p99",
            ),
            SupplementVariant(
                "static_p99p5_practical",
                protocol.optimizer_key,
                selector="static_p99p5",
            ),
            SupplementVariant(
                "causal_expanding_p95_practical",
                protocol.optimizer_key,
                selector="causal_expanding_p95",
            ),
            SupplementVariant(
                f"bottomk_k{protocol.e2.bottom_k}_practical",
                protocol.optimizer_key,
                selector=f"bottomk_k{protocol.e2.bottom_k}",
            ),
            SupplementVariant("m2n2_dt_only", arm="m2n2_dt_only"),
            SupplementVariant(
                "m2n2_practical_p95",
                protocol.optimizer_key,
                arm="m2n2_practical_p95",
            ),
            SupplementVariant(
                "m2n2_oracle_point",
                protocol.optimizer_key,
                arm="m2n2_oracle_point",
            ),
        ]
        return tuple(variants)
    if name == "e3":
        return tuple(
            SupplementVariant(
                f"cadence_b{cadence}_{scenario}",
                protocol.optimizer_key,
                cadence=cadence,
                oracle=scenario == "oracle",
            )
            for cadence in protocol.e3.cadences
            if cadence != 1
            for scenario in ("practical", "oracle")
        )
    if name == "e4":
        return (
            SupplementVariant(
                "fpm_style_practical",
                protocol.optimizer_key,
                selector="fpm_hard_plus_moderate",
            ),
        )
    if name == "e6":
        return tuple(
            SupplementVariant(
                f"cadence_b{cadence}_{scenario}",
                protocol.optimizer_key,
                cadence=cadence,
                oracle=scenario == "oracle",
            )
            for cadence in protocol.e6.cadences
            if cadence != 1
            for scenario in ("practical", "oracle")
        )
    if name == "e7":
        return tuple(
            SupplementVariant(arm, arm=arm) for arm in protocol.e7.arms
        )
    return ()


def default_supplement_protocol_path() -> Path:
    return repository_root() / "configs" / "supplements.toml"


def load_supplement_protocol(
    path: str | Path | None = None,
) -> SupplementProtocol:
    config_path = (
        Path(path) if path is not None else default_supplement_protocol_path()
    )
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    e1_raw = raw.pop("e1")
    e2_raw = raw.pop("e2")
    gamma = {str(key): float(value) for key, value in e2_raw.pop("gamma").items()}
    e3_raw = raw.pop("e3")
    fpm_raw = raw.pop("fpm")
    e5_raw = raw.pop("e5")
    e6_raw = raw.pop("e6")
    e7_raw = raw.pop("e7")
    for key in ("datasets", "seeds"):
        raw[key] = tuple(raw[key])
    for key in (
        "fractions",
        "perturbations",
        "auxiliary_seeds",
        "representative_datasets",
        "baseline_unavailable_datasets",
        "baseline_unavailable_seeds",
    ):
        e1_raw[key] = tuple(e1_raw[key])
    for key in ("static_percentiles",):
        e2_raw[key] = tuple(e2_raw[key])
    for key in ("datasets", "seeds", "cadences"):
        e3_raw[key] = tuple(e3_raw[key])
    for key in ("cadences", "optional_unrun_cadences"):
        e6_raw[key] = tuple(e6_raw[key])
    e7_raw["arms"] = tuple(e7_raw["arms"])
    protocol = SupplementProtocol(
        **raw,
        e1=E1Protocol(**e1_raw),
        e2=E2Protocol(**e2_raw, gamma=gamma),
        e3=E3Protocol(**e3_raw),
        fpm=FPMProtocol(**fpm_raw),
        e5=E5Protocol(**e5_raw),
        e6=E6Protocol(**e6_raw),
        e7=E7Protocol(**e7_raw),
    )
    validate_supplement_protocol(protocol)
    return protocol


def validate_supplement_protocol(protocol: SupplementProtocol) -> None:
    source = protocol.source_protocol()
    locked = {
        "protocol_id": "supplements-paper-results-v1",
        "source_protocol_id": source.protocol_id,
        "datasets": source.datasets,
        "seeds": source.seeds,
        "sequence_length": 12,
        "train_stride": 12,
        "test_stride": 12,
        "optimizer_key": "adamw_ttlr0.0001",
        "gain_threshold_pp": 0.5,
    }
    for field, expected in locked.items():
        actual = getattr(protocol, field)
        if actual != expected:
            raise ValueError(
                f"Locked supplement field {field!r}: "
                f"expected {expected!r}, got {actual!r}"
            )
    if protocol.e1 != E1Protocol(
        mask_seed=20260725,
        fractions=(0.05, 0.10),
        perturbations=(
            "fn_contamination",
            "boundary_erosion",
            "boundary_dilation",
        ),
        main_fraction=0.10,
        main_optimizer_key="adamw_ttlr0.0001",
        auxiliary_optimizer_key="sgd_ttlr0.001",
        auxiliary_seeds=(42, 43, 44),
        representative_datasets=("Daphnet", "MSL", "PUMP", "SWAT2015"),
        baseline_unavailable_datasets=("MSL", "SMAP", "SMD"),
        baseline_unavailable_optimizer_key="sgd_ttlr0.001",
        baseline_unavailable_seeds=(42, 43, 44),
    ):
        raise ValueError("E1 protocol differs from the final label-2x2 design.")
    expected_gamma = {
        "Daphnet": 0.99,
        "GECCO": 0.99,
        "Genesis": 0.99,
        "MSL": 0.60,
        "PSM": 0.99,
        "PUMP": 0.99,
        "SMAP": 0.80,
        "SMD": 0.99,
        "SWAT2015": 0.99999,
    }
    if (
        protocol.e2.static_percentiles != (95.0, 99.0, 99.5)
        or protocol.e2.expanding_percentile != 95.0
        or protocol.e2.bottom_k != 11
        or protocol.e2.m2n2_percentile != 95.0
        or protocol.e2.gamma != expected_gamma
    ):
        raise ValueError("E2 protocol differs from the final canonical selector grid.")
    if protocol.e3 != E3Protocol(
        datasets=("SWAT2015", "PUMP", "MSL", "Daphnet"),
        seeds=(42, 43, 44),
        cadences=(1, 4),
    ):
        raise ValueError("E3 protocol differs from the preregistered pilot.")
    if protocol.fpm != FPMProtocol(
        score_percentile=99.5,
        hard_tail_percent=0.5,
        moderate_low_percentile=25.0,
        moderate_high_percentile=75.0,
        chi2_probability=0.05,
        covariance_regularization=0.0,
        inverse_policy="pinv",
    ):
        raise ValueError("E4/E7 FPM protocol differs from the frozen transplant.")
    if protocol.e5 != E5Protocol(
        vus_window=12,
        rank_thresholds=250,
        alignment_policy="common_prefix_joint_finite",
    ):
        raise ValueError("E5 metric protocol differs from the available-run audit.")
    if protocol.e6 != E6Protocol(
        cadences=(1, 2, 4, 8), optional_unrun_cadences=(16,)
    ):
        raise ValueError("E6 cadence sweep differs from the final grid.")
    if protocol.e7 != E7Protocol(
        arms=(
            "frozen_sana",
            "practical_fpm_sana",
            "oracle_normal_window_sana",
            "oracle_clean_fpm_sana",
        ),
        d_model=512,
        heads=8,
        d_ff=512,
        sana_seed_offset=7000,
    ):
        raise ValueError("E7 SANA protocol differs from the frozen four-arm port.")
