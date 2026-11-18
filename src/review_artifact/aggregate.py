"""Submitted PR-curve AUC and median of matched entity differences.

The engine's legacy metrics.json contains AP and is deliberately not read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from section5_repro.data import Entity
from section5_repro.paths import Workspace, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol, repository_root

TAU = 0.005
CATEGORIES = ("BI", "OOG", "NI", "POG")
FROZEN = "frozen"
PRACTICAL = "practical_static_p95"
ORACLE = "oracle_benchmark_normal_point"


def category(practical, oracle):
    pg, og = practical > TAU, oracle > TAU
    return "BI" if pg and og else "POG" if pg else "OOG" if og else "NI"


def point_auprc(labels, scores):
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("Score/label length mismatch; cropping is forbidden.")
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be finite and binary.")
    if not np.isfinite(scores).all():
        return False, float("nan"), "nonfinite-score"
    if np.unique(labels).size != 2:
        return False, float("nan"), "single-class-labels"
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return True, float(auc(recall, precision)), ""


def aggregate_physical(physical):
    """Require all 5,220 score runs, then derive one support across the whole grid."""
    physical = physical.copy()
    physical["subset"] = physical["subset"].fillna("")
    protocol = load_protocol()
    entities = json.loads((repository_root() / "reference/data/processed_manifest.json").read_text())["entities"]
    keys = ["dataset", "subset", "seed", "optimizer_id", "arm"]
    expected = set()
    for row in entities:
        for seed in protocol.seeds:
            expected.add((row["dataset"], row["subset"], seed, "shared", FROZEN))
            for config in protocol.tta_configs:
                for arm in (PRACTICAL, ORACLE):
                    expected.add((row["dataset"], row["subset"], seed, config.key, arm))
    if physical.duplicated(keys).any() or set(physical[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected all 5,220 distinct protocol score runs; missing or extra rows are not allowed.")
    if not physical.raw_valid.isin([True, False]).all():
        raise ValueError("raw_valid must contain explicit boolean validity flags.")
    if not np.isfinite(physical.loc[physical.raw_valid, "corrected_auprc"]).all():
        raise ValueError("A valid physical run has an undefined AUPRC.")
    support = physical.groupby(["dataset", "subset"], sort=True).raw_valid.all().reset_index()
    support["support_status"] = np.where(support.raw_valid, "retained", "excluded")
    kept = set(support.loc[support.raw_valid, ["dataset", "subset"]].itertuples(index=False, name=None))
    lookup = physical.set_index(keys).corrected_auprc
    rows = []
    for dataset in protocol.datasets:
        subsets = sorted(s for d, s in kept if d == dataset)
        if not subsets:
            raise ValueError(f"No valid common support for {dataset}; no zero substitution is allowed.")
        excluded = support.loc[(support.dataset == dataset) & ~support.raw_valid, "subset"].tolist()
        for config in protocol.tta_configs:
            for seed in protocol.seeds:
                p_delta, o_delta = [], []
                for subset in subsets:
                    frozen = lookup.loc[(dataset, subset, seed, "shared", FROZEN)]
                    p_delta.append(lookup.loc[(dataset, subset, seed, config.key, PRACTICAL)] - frozen)
                    o_delta.append(lookup.loc[(dataset, subset, seed, config.key, ORACLE)] - frozen)
                practical, oracle = float(np.median(p_delta)), float(np.median(o_delta))
                rows.append({
                    "dataset": dataset, "optimizer": config.key, "seed": seed,
                    "corrected_auprc_practical_delta": practical, "corrected_auprc_oracle_delta": oracle,
                    "corrected_auprc_category": category(practical, oracle),
                    "retained_entity_count": len(subsets), "excluded_entities": ";".join(excluded),
                    "gain_threshold": TAU, "support_policy": "global-common-physical-support",
                    "aggregation": "median-of-physical-paired-deltas",
                })
    return pd.DataFrame(rows), support.drop(columns="raw_valid")


def aggregate_workspace(workspace_root: Path, protocol, output: Path):
    from .cli import entity_records
    workspace = Workspace(workspace_root)
    rows, objectives = [], []
    for entry in entity_records():
        entity = Entity(entry["dataset"], entry["subset"], Path("unused"))
        expected_length = entry["test_rows"] // protocol.sequence_length * protocol.sequence_length
        for seed in protocol.seeds:
            frozen_dir = frozen_result_dir(workspace, entity, seed, protocol)
            frozen_labels = np.load(frozen_dir / "true.npy", allow_pickle=False).reshape(-1)
            if len(frozen_labels) != expected_length:
                raise ValueError(f"Unexpected point support in {frozen_dir}")
            runs = [("shared", FROZEN, frozen_dir)]
            runs += [(config.key, arm, tta_result_dir(workspace, entity, seed, protocol, config, arm == ORACLE))
                     for config in protocol.tta_configs for arm in (PRACTICAL, ORACLE)]
            for optimizer, arm, directory in runs:
                complete = json.loads((directory / "complete.json").read_text())
                if (complete["entity"] != entry or complete["seed"] != seed
                        or complete["protocol"] != json.loads(json.dumps(protocol.as_dict())) or complete["setting"] != directory.name):
                    raise ValueError(f"Protocol/entity/seed mismatch in {directory}")
                scores = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
                labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
                if not np.array_equal(labels, frozen_labels):
                    raise ValueError(f"Unmatched evaluation labels in {directory}")
                valid, value, reason = point_auprc(labels, scores)
                rows.append({"dataset": entity.dataset, "subset": entity.subset, "seed": seed,
                             "optimizer_id": optimizer, "arm": arm, "raw_valid": valid,
                             "invalid_reason": reason, "corrected_auprc": value})
                if arm == ORACLE:
                    trace = pd.read_parquet(directory / "tta_pointwise_analysis.parquet")
                    if len(trace) != len(labels) or not np.array_equal(trace.true_label, labels):
                        raise ValueError(f"Unmatched objective trace in {directory}")
                    normal = trace.true_label.to_numpy() == 0
                    difference = trace.pre_update_score.to_numpy(dtype=float) - trace.post_update_score.to_numpy(dtype=float)
                    reduction = float(np.mean(difference[normal])) if normal.any() else float("nan")
                    objectives.append({"dataset": entity.dataset, "optimizer_id": optimizer, "seed": seed,
                                       "oracle_normal_loss_reduction": reduction})
    physical = pd.DataFrame(rows)
    cells, support = aggregate_physical(physical)
    # Objective diagnostics have their own finite support, as in the submission.
    objective = pd.DataFrame(objectives).replace([np.inf, -np.inf], np.nan)
    objective = objective.groupby(["dataset", "optimizer_id", "seed"]).oracle_normal_loss_reduction.median().reset_index()
    outcome = cells.rename(columns={"optimizer": "optimizer_id", "corrected_auprc_oracle_delta": "oracle_delta_auprc",
                                   "corrected_auprc_practical_delta": "practical_delta_auprc", "corrected_auprc_category": "category"})
    objective = outcome.merge(objective, on=["dataset", "optimizer_id", "seed"], validate="one_to_one")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("core_physical", physical), ("core_cells", cells), ("core_support", support), ("core_objective", objective)):
        frame.to_csv(output / f"{name}.csv", index=False)
    print(json.dumps({"logical_cells": len(cells), "retained_entities": int((support.support_status == "retained").sum()),
                      "counts": cells.corrected_auprc_category.value_counts().to_dict()}, indent=2))
