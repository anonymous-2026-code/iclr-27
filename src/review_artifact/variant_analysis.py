"""Matched Section 4.4 analysis, using the submitted corrected PR-curve AUC."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from section5_repro.artifacts import sha256_array, sha256_file
from section5_repro.data import Entity
from section5_repro.paths import Workspace, checkpoint_path, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol
from section5_repro.supplement_engine import supplement_result_dir
from section5_repro.supplement_protocol import load_supplement_protocol
from .aggregate import FROZEN, PRACTICAL, ORACLE, CATEGORIES, category, point_auprc
from .cli import entity_records
from .score_shift import read_core_arrays
from .variants import GROUPS

CORE = "core_oracle_heterogeneity"
COMPARATORS = "benchmark_label_comparators"
ADAMW = "adamw_ttlr0.0001"
PERTURBATIONS = ("fn_contamination", "boundary_erosion", "boundary_dilation")
CHANNELS = ("mask-only", "evaluation-only", "joint")


def metric_row(entry, seed, profile, arm, score, labels, *, perturbation="", channel=""):
    valid, value, reason = point_auprc(labels, score)
    return {"dataset": entry["dataset"], "subset": entry["subset"], "seed": seed,
            "profile_id": profile, "arm": arm, "perturbation": perturbation,
            "sensitivity_channel": channel, "raw_valid": valid, "invalid_reason": reason,
            "corrected_auprc": value, "label_digest": sha256_array(labels)}


def read_variant(directory, entry, seed, experiment, variant, checkpoint_hash):
    meta = json.loads((directory / "completed.json").read_text())
    expected = {"complete": True, "protocol_id": load_supplement_protocol().protocol_id,
                "source_protocol_id": load_protocol().protocol_id, "dataset": entry["dataset"],
                "subset": entry["subset"], "seed": seed, "entity": entry,
                "experiment": experiment, "variant": variant, "source_checkpoint_sha256": checkpoint_hash}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Variant identity/checkpoint mismatch: {directory}")
    score = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
    labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
    if len(score) != entry["test_rows"] // 12 * 12 or len(labels) != len(score):
        raise ValueError(f"Variant point support mismatch: {directory}")
    if sha256_array(score) != meta["score_sha256"] or sha256_array(labels) != meta["label_sha256"]:
        raise ValueError(f"Saved variant array mismatch: {directory}")
    return score, labels, meta


def collect_metrics(workspace_root):
    protocol = load_protocol()
    config = protocol.tta_by_key(ADAMW)
    workspace = Workspace(workspace_root)
    rows = []
    for entry in entity_records():
        entity = Entity(entry["dataset"], entry["subset"], Path("unused"))
        for seed in protocol.seeds:
            checkpoint_hash = sha256_file(checkpoint_path(workspace, entity, seed, protocol))
            controls = {}
            for arm, directory in (
                (FROZEN, frozen_result_dir(workspace, entity, seed, protocol)),
                (PRACTICAL, tta_result_dir(workspace, entity, seed, protocol, config, False)),
                (ORACLE, tta_result_dir(workspace, entity, seed, protocol, config, True)),
            ):
                score, labels = read_core_arrays(directory, entry, seed)
                if controls and not np.array_equal(labels, original_labels):
                    raise ValueError(f"Core comparator labels differ: {directory}")
                original_labels = labels
                controls[arm] = score
                rows.append(metric_row(entry, seed, CORE, arm, score, labels))
            for group, (experiment, profile, variants) in GROUPS.items():
                for variant in variants:
                    directory = supplement_result_dir(workspace, experiment, variant, entity, seed)
                    score, labels, meta = read_variant(directory, entry, seed, experiment, variant, checkpoint_hash)
                    if not np.array_equal(labels, original_labels):
                        raise ValueError(f"Unmatched variant evaluation labels: {directory}")
                    if group != "labels":
                        rows.append(metric_row(entry, seed, profile, variant, score, labels))
                        continue
                    perturbation = variant.split("__")[1]
                    changed = np.load(directory / "perturbed_true.npy", allow_pickle=False).reshape(-1)
                    if len(changed) != len(labels) or sha256_array(changed) != meta["perturbed_label_sha256"]:
                        raise ValueError(f"Perturbed label mismatch: {directory}")
                    for channel in CHANNELS:
                        evaluation_labels = labels if channel == "mask-only" else changed
                        evaluation_score = controls[ORACLE] if channel == "evaluation-only" else score
                        rows.append(metric_row(entry, seed, profile, variant, evaluation_score, evaluation_labels,
                                               perturbation=perturbation, channel=channel))
                    for arm in (FROZEN, PRACTICAL):
                        rows.append(metric_row(entry, seed, COMPARATORS, arm, controls[arm], changed,
                                               perturbation=perturbation, channel="perturbed"))
    return pd.DataFrame(rows)


def select(frame, profile, arm, perturbation="", channel=""):
    return frame.loc[(frame.profile_id == profile) & (frame.arm == arm)
                     & (frame.perturbation == perturbation) & (frame.sensitivity_channel == channel)].copy()


def compare(name, baseline, practical, oracle, forced_support=None):
    keys = ["dataset", "subset", "seed"]
    expected = {(e["dataset"], e["subset"], s) for e in entity_records() for s in load_protocol().seeds}
    def keyed(frame, prefix):
        if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"{name}: incomplete or duplicated {prefix} physical support")
        cols = ["raw_valid", "corrected_auprc", "label_digest"]
        return frame[keys + cols].rename(columns={c: f"{prefix}_{c}" for c in cols})
    paired = keyed(baseline, "frozen").merge(keyed(practical, "practical"), on=keys, validate="one_to_one")
    paired = paired.merge(keyed(oracle, "oracle"), on=keys, validate="one_to_one")
    if not ((paired.frozen_label_digest == paired.practical_label_digest)
            & (paired.frozen_label_digest == paired.oracle_label_digest)).all():
        raise ValueError(f"{name}: mismatched evaluation labels")
    paired["all_valid"] = paired[["frozen_raw_valid", "practical_raw_valid", "oracle_raw_valid"]].all(axis=1)
    support = {key for key, group in paired.groupby(["dataset", "subset"], dropna=False)
               if group.all_valid.all() and (forced_support is None or key in forced_support)}
    paired["retained"] = [(d, s) in support for d, s in zip(paired.dataset, paired.subset)]
    rows = []
    for (dataset, seed), group in paired.loc[paired.retained].groupby(["dataset", "seed"], sort=True):
        practical_delta = float(np.median(group.practical_corrected_auprc - group.frozen_corrected_auprc))
        oracle_delta = float(np.median(group.oracle_corrected_auprc - group.frozen_corrected_auprc))
        rows.append({"analysis": name, "dataset": dataset, "seed": seed,
                     "corrected_practical_delta": practical_delta, "corrected_oracle_delta": oracle_delta,
                     "corrected_category": category(practical_delta, oracle_delta), "retained_entity_count": len(group),
                     "excluded_entities": ";".join(sorted(set(paired.loc[(paired.dataset == dataset) & ~paired.retained, "subset"])))})
    if len(rows) != 45:
        raise ValueError(f"{name}: expected 45 logical cells, found {len(rows)}")
    return pd.DataFrame(rows)


def summarize(outcomes):
    rows = []
    for name, group in outcomes.groupby("analysis", sort=False):
        counts = group.corrected_category.value_counts()
        rows.append({"analysis": name, **{c: int(counts.get(c, 0)) for c in CATEGORIES}, "logical_cells": len(group)})
    return pd.DataFrame(rows)


def analyze_metrics(physical, core_cells):
    physical = physical.copy()
    for column in ("subset", "perturbation", "sensitivity_channel"):
        physical[column] = physical[column].fillna("")
    keys = ["profile_id", "dataset", "subset", "seed", "arm", "perturbation", "sensitivity_channel"]
    if len(physical) != 18560 or physical.duplicated(keys).any():
        raise ValueError("Expected all 18,560 submitted variant/comparator metric rows")
    if not physical.raw_valid.isin([True, False]).all():
        raise ValueError("Invalid raw validity flags")
    if not np.isfinite(physical.loc[physical.raw_valid, "corrected_auprc"]).all():
        raise ValueError("Undefined metric marked valid")
    if len(core_cells) != 180 or core_cells.duplicated(["dataset", "optimizer", "seed"]).any():
        raise ValueError("Expected 180 core logical cells")
    forced = set()
    for entry in entity_records():
        group = core_cells.loc[core_cells.dataset == entry["dataset"]]
        excluded = group.excluded_entities.fillna("")
        if len(group) != 20 or excluded.nunique() != 1:
            raise ValueError("Core cells do not declare a single global support per dataset")
        if entry["subset"] not in set(excluded.iloc[0].split(";")) - {""}:
            forced.add((entry["dataset"], entry["subset"]))
    cf, cp, co = (select(physical, CORE, arm) for arm in (FROZEN, PRACTICAL, ORACLE))
    robustness = [compare("Cadence B=1 / Static-P95", cf, cp, co, forced)]
    for name, group, arms in (
        ("M2N2", "m2n2", ("m2n2_dt_only", "m2n2_practical_p95", "m2n2_oracle_point")),
        ("FPM+SANA normal-window Oracle", "fpm_sana", ("frozen_sana", "practical_fpm_sana", "oracle_normal_window_sana")),
        ("FPM+SANA clean-FPM Oracle", "fpm_sana", ("frozen_sana", "practical_fpm_sana", "oracle_clean_fpm_sana")),
    ):
        robustness.append(compare(name, *(select(physical, GROUPS[group][1], arm) for arm in arms)))
    robustness.insert(2, compare("FPM-only", cf, select(physical, GROUPS["fpm"][1], "fpm_style_practical"), co))
    for cadence in (2, 4, 8):
        robustness.append(compare(f"Cadence B={cadence}", cf,
                                  select(physical, GROUPS["cadence"][1], f"cadence_b{cadence}_practical"),
                                  select(physical, GROUPS["cadence"][1], f"cadence_b{cadence}_oracle")))
    reference = robustness[0][["dataset", "seed", "corrected_category"]].rename(columns={"corrected_category": "original_category"})
    reference_cells = core_cells.loc[core_cells.optimizer == ADAMW]
    check = robustness[0].merge(reference_cells, on=["dataset", "seed"], validate="one_to_one")
    if len(check) != 45:
        raise ValueError("Core AdamW comparison is incomplete")
    for arm in ("practical", "oracle"):
        np.testing.assert_allclose(check[f"corrected_{arm}_delta"], check[f"corrected_auprc_{arm}_delta"], rtol=0, atol=1e-14)
    label_outcomes = []
    for perturbation in PERTURBATIONS:
        variant = f"{ADAMW}__{perturbation}__frac0p1"
        for channel in CHANNELS:
            baseline, practical = (cf, cp) if channel == "mask-only" else (
                select(physical, COMPARATORS, FROZEN, perturbation, "perturbed"),
                select(physical, COMPARATORS, PRACTICAL, perturbation, "perturbed"))
            oracle = select(physical, GROUPS["labels"][1], variant, perturbation, channel)
            label_outcomes.append(compare(f"{perturbation}/{channel}", baseline, practical, oracle, forced))
    labels = pd.concat(label_outcomes, ignore_index=True).merge(reference, on=["dataset", "seed"], validate="many_to_one")
    labels["category_retained"] = labels.corrected_category == labels.original_category
    summary = summarize(labels)
    retention = labels.groupby("analysis").category_retained.agg(["sum", "mean"]).rename(columns={"sum": "retained", "mean": "retention_rate"})
    summary = summary.merge(retention, on="analysis", validate="one_to_one").sort_values("analysis")
    robustness = pd.concat(robustness, ignore_index=True)
    return robustness, summarize(robustness), labels, summary


def aggregate(workspace, core_derived, output, metrics=None):
    if output.exists():
        raise FileExistsError(output)
    physical = pd.read_csv(metrics) if metrics is not None else collect_metrics(workspace)
    core_cells = pd.read_csv(core_derived / "core_cells.csv")
    robustness, robust_summary, labels, label_summary = analyze_metrics(physical, core_cells)
    output.mkdir(parents=True)
    for name, frame in (("variant_physical", physical), ("robustness_outcomes", robustness),
                        ("robustness_summary", robust_summary), ("label_perturbation_outcomes", labels),
                        ("label_perturbation_summary", label_summary)):
        frame.to_csv(output / f"{name}.csv", index=False)
    print(json.dumps({"physical_metrics": len(physical), "robustness_cells": len(robustness),
                      "label_perturbation_cells": len(labels)}, indent=2))
