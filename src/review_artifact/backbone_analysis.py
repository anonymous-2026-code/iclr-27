"""Matched Table 2 aggregation from raw scores or retained historical metrics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from section5_repro.artifacts import sha256_file, sha256_array
from section5_repro.data import Entity
from section5_repro.paths import Workspace, checkpoint_path, frozen_result_dir, tta_result_dir
from .aggregate import FROZEN, PRACTICAL, ORACLE, category, point_auprc
from .backbone_engine import BACKBONES, SCOPES, protocol_for
from .cli import entity_records


def stem(backbone, scope):
    return f"{backbone}_{'FULL' if scope == 'full_model' else 'DECODER_ONLY'}"


def collect_metrics(workspace_root):
    workspace = Workspace(workspace_root)
    rows = []
    for backbone in BACKBONES:
        for entry in entity_records():
            entity = Entity(entry["dataset"], entry["subset"], Path("unused"))
            for seed in protocol_for(backbone).seeds:
                source_hash = sha256_file(checkpoint_path(workspace, entity, seed, protocol_for(backbone)))
                for scope in SCOPES:
                    protocol = protocol_for(backbone, scope)
                    runs = [("shared", FROZEN, frozen_result_dir(workspace, entity, seed, protocol_for(backbone)))]
                    runs += [(c.key, arm, tta_result_dir(workspace, entity, seed, protocol, c, arm == ORACLE))
                             for c in protocol.tta_configs for arm in (PRACTICAL, ORACLE)]
                    frozen_labels = None
                    taus = {}
                    for optimizer, arm, directory in runs:
                        meta = json.loads((directory / "complete.json").read_text())
                        expected_protocol = protocol_for(backbone) if arm == FROZEN else protocol
                        expected = {"entity": entry, "seed": seed, "source_checkpoint_sha256": source_hash,
                                    "kind": {FROZEN: "frozen", PRACTICAL: "practical", ORACLE: "oracle"}[arm],
                                    "protocol": json.loads(json.dumps(expected_protocol.as_dict()))}
                        if arm != FROZEN:
                            expected["optimizer_key"] = optimizer
                        if any(meta.get(k) != v for k, v in expected.items()):
                            raise ValueError(f"Backbone identity/protocol/checkpoint mismatch: {directory}")
                        scores = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
                        labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
                        points = np.load(directory / "point_index.npy", allow_pickle=False).reshape(-1)
                        expected_length = entry["test_rows"] // 12 * 12
                        if len(scores) != expected_length or len(labels) != expected_length or not np.array_equal(points, np.arange(expected_length)):
                            raise ValueError(f"Invalid backbone point support: {directory}")
                        if frozen_labels is None:
                            frozen_labels = labels
                        if not np.array_equal(labels, frozen_labels):
                            raise ValueError(f"Unmatched backbone labels: {directory}")
                        if arm != FROZEN:
                            if optimizer in taus and taus[optimizer] != meta["selector_tau"]:
                                raise ValueError(f"Unmatched Practical/Oracle calibration: {directory}")
                            taus[optimizer] = meta["selector_tau"]
                        valid, value, reason = point_auprc(labels, scores)
                        rows.append({"backbone": backbone, "update_scope": scope, "dataset": entity.dataset,
                                     "subset": entity.subset, "seed": seed, "optimizer_id": optimizer, "arm": arm,
                                     "raw_valid": valid, "corrected_auprc": value, "invalid_reason": reason,
                                     "label_digest": sha256_array(labels)})
    return pd.DataFrame(rows)


def aggregate_block(frame, backbone, scope, declared_support=None):
    protocol = protocol_for(backbone, scope)
    entries = entity_records()
    inventory = {(e["dataset"], e["subset"]) for e in entries}
    frame = frame.copy()
    frame["subset"] = frame.subset.fillna("")
    if declared_support is not None:
        support = declared_support.copy()
        support["entity"] = support.entity.fillna("").replace({"__root__": ""})
        if (support.duplicated(["dataset", "entity"]).any()
                or set(support[["dataset", "entity"]].itertuples(index=False, name=None)) != inventory
                or not support.status.isin(["retained", "excluded"]).all()):
            raise ValueError("Invalid declared historical support")
        eligible = set(support.loc[support.status == "retained", ["dataset", "entity"]].itertuples(index=False, name=None))
    else:
        eligible = inventory
    expected = set()
    for dataset, subset in eligible:
        for seed in protocol.seeds:
            expected.add((dataset, subset, seed, "shared", FROZEN))
            expected.update((dataset, subset, seed, c.key, a) for c in protocol.tta_configs for a in (PRACTICAL, ORACLE))
    keys = ["dataset", "subset", "seed", "optimizer_id", "arm"]
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"Incomplete {backbone}/{scope} grid on its declared support")
    if not frame.raw_valid.isin([True, False]).all() or not np.isfinite(frame.loc[frame.raw_valid, "corrected_auprc"]).all():
        raise ValueError("Invalid backbone metric validity flags")
    if declared_support is not None and not frame.raw_valid.all():
        raise ValueError("A retained historical metric is undefined")
    if declared_support is None:
        support = frame.groupby(["dataset", "subset"]).raw_valid.all().reset_index().rename(columns={"subset": "entity"})
        support["status"] = np.where(support.raw_valid, "retained", "excluded")
        support["reasons"] = np.where(support.raw_valid, "", "undefined complete score stream in matched grid")
        support = support.drop(columns="raw_valid")
    kept = set(support.loc[support.status == "retained", ["dataset", "entity"]].itertuples(index=False, name=None))
    lookup = frame.set_index(keys)
    rows = []
    for dataset in protocol.datasets:
        subsets = sorted(s for d, s in kept if d == dataset)
        if not subsets:
            raise ValueError(f"No valid common support: {backbone}/{scope}/{dataset}")
        for seed in protocol.seeds:
            for config in protocol.tta_configs:
                p_delta, o_delta = [], []
                for subset in subsets:
                    frozen = lookup.loc[(dataset, subset, seed, "shared", FROZEN)]
                    practical = lookup.loc[(dataset, subset, seed, config.key, PRACTICAL)]
                    oracle = lookup.loc[(dataset, subset, seed, config.key, ORACLE)]
                    if len({r.label_digest for r in (frozen, practical, oracle)}) != 1:
                        raise ValueError("Matched backbone arms have different labels")
                    p_delta.append(practical.corrected_auprc - frozen.corrected_auprc)
                    o_delta.append(oracle.corrected_auprc - frozen.corrected_auprc)
                p, o = float(np.median(p_delta)), float(np.median(o_delta))
                rows.append({"backbone": backbone, "update_scope": "full_model" if scope == "full_model" else "decoder_only",
                             "dataset": dataset, "seed": seed, "optimizer": config.optimizer, "lr": config.learning_rate,
                             "delta_practical": p, "delta_oracle": o, "category": category(p, o),
                             "retained_entities": len(subsets), "threshold": .005})
    return pd.DataFrame(rows), support


def aggregate(workspace, output, metrics=None, support_root=None):
    if output.exists():
        raise FileExistsError(output)
    if (metrics is None) != (support_root is None):
        raise ValueError("Saved metric replay requires both --metrics and --support-root")
    physical = collect_metrics(workspace) if metrics is None else pd.read_csv(metrics)
    expected_blocks = {(b, s) for b in BACKBONES for s in SCOPES}
    if set(physical[["backbone", "update_scope"]].itertuples(index=False, name=None)) != expected_blocks:
        raise ValueError("All four Table 2 blocks are required")
    results = []
    for backbone in BACKBONES:
        for scope in SCOPES:
            frame = physical.loc[(physical.backbone == backbone) & (physical.update_scope == scope)]
            declared = pd.read_csv(support_root / f"{stem(backbone, scope)}_COMMON_SUPPORT.csv") if metrics is not None else None
            cells, support = aggregate_block(frame, backbone, scope, declared)
            results.append((backbone, scope, cells, support))
    (output / "backbones").mkdir(parents=True)
    physical.to_csv(output / "backbone_physical.csv", index=False)
    for backbone, scope, cells, support in results:
        cells.to_csv(output / "backbones" / f"{stem(backbone, scope)}_ICLR27_OUTCOMES.csv", index=False)
        support.to_csv(output / "backbones" / f"{stem(backbone, scope)}_COMMON_SUPPORT.csv", index=False)
    print(json.dumps({"physical_metric_memberships": len(physical), "logical_cells": sum(len(r[2]) for r in results),
                      "support_mode": "declared historical support" if metrics else "derived from complete fresh grid"}))
