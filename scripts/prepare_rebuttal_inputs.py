#!/usr/bin/env python3
"""Index freshly reproduced AE artifacts for the preserved rebuttal scripts.

This performs score postprocessing only. It never trains or changes weights.
The original rebuttal readers require physical manifests in addition to the
compact submission tables; these are generated from this package's workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from review_artifact.aggregate import aggregate_physical, point_auprc
from review_artifact.cli import entity_records
from review_artifact.variant_analysis import read_variant
from review_artifact.variants import GROUPS
from section5_repro.artifacts import sha256_array, sha256_file
from section5_repro.data import Entity, entity_manifest
from section5_repro.engine import _checkpoint_complete
from section5_repro.paths import Workspace, checkpoint_path, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol
from section5_repro.supplement_engine import supplement_result_dir

OUT = ROOT / "workspace/rebuttal_inputs"
CORE_FILE = OUT / "p3_iclr27_main_auprc_corrected_physical.parquet"
DETAIL = OUT / "p3_iclr27_corrected_auprc"
VERSION = "sklearn-pr-curve-trapezoid-v1"


def _array_sha256(value: np.ndarray) -> str:
    # Original VUS inventory encoding; required by the preserved VUS reader.
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8").reshape(-1))
    digest = hashlib.sha256()
    digest.update(b"iclr27-score-vector-v1\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def index_core_run(directory, entity, seed, optimizer, arm, protocol, data_meta, checkpoint_hash, manifest):
    complete = json.loads((directory / "complete.json").read_text())
    if (complete["entity"] != data_meta or complete["seed"] != seed
            or complete["protocol"] != json.loads(json.dumps(protocol.as_dict()))
            or complete["setting"] != directory.name):
        raise ValueError(f"Mismatched core identity/protocol: {directory}")
    scores = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
    labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
    expected = data_meta["test_rows"] // protocol.sequence_length * protocol.sequence_length
    if len(labels) != expected:
        raise ValueError(f"Unexpected complete-window prefix: {directory}")
    valid, value, reason = point_auprc(labels, scores)
    metrics = json.loads((directory / "metrics.json").read_text())
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as handle:
        json.dump({"status": "complete", "source_checkpoint_hash": checkpoint_hash,
                   "data_manifest": data_meta, "result_metadata": metrics,
                   "origin": "fresh anonymous-package workspace"}, handle, indent=2)
    fingerprint = hashlib.sha256(json.dumps(data_meta, sort_keys=True).encode()).hexdigest()
    return {
        "physical_run_id": f"core/{entity.dataset}/{entity.subset or '__root__'}/{seed}/{optimizer}/{arm}",
        "dataset": entity.dataset, "subset": entity.subset, "seed": seed,
        "optimizer_id": optimizer, "arm": arm, "backbone": "AE",
        "profile_id": "core_oracle_heterogeneity", "raw_valid": valid,
        "invalid_reason": reason, "corrected_auprc": value, "corrected_pr_curve_auc": value,
        "corrected_metric_version": VERSION, "n_total": len(labels),
        "n_valid": int((np.isfinite(scores) & np.isfinite(labels)).sum()),
        "source_checkpoint_hash": checkpoint_hash, "data_fingerprint": fingerprint,
        "score_sha256": sha256_file(directory / "test_energy.npy"),
        "label_sha256": sha256_file(directory / "true.npy"), "label_digest": sha256_array(labels),
        "manifest_path": str(manifest), "score_path": str(directory / "test_energy.npy"),
        "label_path": str(directory / "true.npy"),
    }


def core(derived):
    OUT.mkdir(parents=True, exist_ok=False)
    DETAIL.mkdir()
    protocol = load_protocol()
    workspace = Workspace(ROOT / "workspace/ae")
    records = []
    for entry in entity_records():
        entity = Entity(entry["dataset"], entry["subset"], ROOT / "data/processed" / entry["dataset"] / entry["subset"])
        data_meta = entity_manifest(entity)
        if data_meta != entry:
            raise ValueError(f"Prepared benchmark identity changed: {entity.directory}")
        for seed in protocol.seeds:
            checkpoint = checkpoint_path(workspace, entity, seed, protocol)
            if not _checkpoint_complete(checkpoint, protocol, entity, seed):
                raise ValueError(f"Incomplete source checkpoint: {checkpoint}")
            checkpoint_hash = sha256_file(checkpoint)
            runs = [("shared", "frozen", frozen_result_dir(workspace, entity, seed, protocol))]
            runs += [(c.key, arm, tta_result_dir(workspace, entity, seed, protocol, c, arm == "oracle_benchmark_normal_point"))
                     for c in protocol.tta_configs for arm in ("practical_static_p95", "oracle_benchmark_normal_point")]
            reference_labels = np.load(runs[0][2] / "true.npy", allow_pickle=False).reshape(-1)
            for optimizer, arm, directory in runs:
                np.testing.assert_array_equal(reference_labels, np.load(directory / "true.npy", allow_pickle=False).reshape(-1))
                manifest = OUT / "manifests" / entity.dataset / (entity.subset or "__root__") / str(seed) / optimizer / f"{arm}.json"
                records.append(index_core_run(directory, entity, seed, optimizer, arm, protocol, data_meta, checkpoint_hash, manifest))
    physical = pd.DataFrame(records)
    cells, support = aggregate_physical(physical)
    expected_cells = pd.read_csv(derived / "core_cells.csv")
    keys = ["dataset", "optimizer", "seed"]
    checked = cells.merge(expected_cells, on=keys, suffixes=("", "_reference"), validate="one_to_one")
    if len(checked) != 180:
        raise ValueError("Incomplete core aggregation")
    for name in ("practical", "oracle"):
        col = f"corrected_auprc_{name}_delta"
        np.testing.assert_allclose(checked[col], checked[col + "_reference"], rtol=0, atol=1e-14)
    # Rebuttal controls deliberately use the original fixed 106-entity support.
    historical = pd.DataFrame(json.loads((ROOT / "configs/rebuttal_support.json").read_text()))
    got = set(support.loc[support.support_status == "retained", ["dataset", "subset"]].itertuples(index=False, name=None))
    wanted = set(historical[["dataset", "subset"]].itertuples(index=False, name=None))
    if got != wanted:
        raise ValueError("Fresh support differs from the rebuttal's fixed 106 entities; no substitution is allowed")
    physical.to_parquet(CORE_FILE, index=False)
    support = support.rename(columns={"subset": "entity"})
    support["entity"] = support.entity.replace("", "__root__")
    support.to_csv(OUT / "p3_iclr27_corrected_auprc_common_support_manifest.csv", index=False)
    cells.to_csv(OUT / "p3_iclr27_corrected_auprc_common_support_cells.csv", index=False)
    shutil.copyfile(derived / "core_objective.csv", DETAIL / "corrected_core_outcomes_with_objective.csv")
    print(f"Indexed {len(physical)} physical runs and {len(cells)} core cells.")


def variants(derived):
    target = DETAIL / "physical_auprc.parquet"
    if target.exists():
        raise FileExistsError(target)
    protocol, workspace = load_protocol(), Workspace(ROOT / "workspace/ae")
    rows = pd.read_parquet(CORE_FILE).to_dict("records")
    for entry in entity_records():
        entity = Entity(entry["dataset"], entry["subset"], ROOT / "data/processed" / entry["dataset"] / entry["subset"])
        for seed in protocol.seeds:
            checkpoint_hash = sha256_file(checkpoint_path(workspace, entity, seed, protocol))
            for name in ("m2n2", "fpm", "fpm_sana"):
                experiment, profile, arms = GROUPS[name]
                for arm in arms:
                    directory = supplement_result_dir(workspace, experiment, arm, entity, seed)
                    scores, labels, _ = read_variant(directory, entry, seed, experiment, arm, checkpoint_hash)
                    valid, value, reason = point_auprc(labels, scores)
                    rows.append({"physical_run_id": f"{profile}/{entity.dataset}/{entity.subset or '__root__'}/{seed}/{arm}",
                        "profile_id": profile, "dataset": entity.dataset, "subset": entity.subset, "seed": seed,
                        "optimizer_id": "none" if arm in ("m2n2_dt_only", "frozen_sana") else "adamw_ttlr0.0001",
                        "arm": arm, "raw_valid": valid, "invalid_reason": reason, "corrected_auprc": value,
                        "label_digest": sha256_array(labels), "score_path": str(directory / "test_energy.npy"),
                        "label_path": str(directory / "true.npy")})
    pd.DataFrame(rows).to_parquet(target, index=False)
    shutil.copyfile(derived / "robustness_outcomes.csv", DETAIL / "robustness_outcomes.csv")
    print("Indexed the submitted method controls for the four-optimizer extension.")


def vus():
    from section5_repro.range_metrics import range_auc_volume
    target = OUT / "iclr27_main_v1_physical.parquet"
    if target.exists():
        raise FileExistsError(target)
    records = []
    for row in pd.read_parquet(CORE_FILE).to_dict("records"):
        labels = np.load(row["label_path"], allow_pickle=False).reshape(-1)
        scores = np.load(row["score_path"], allow_pickle=False).reshape(-1)
        valid, _, _ = point_auprc(labels, scores)
        # Never finite-filter a retained stream. Invalid streams are excluded
        # by the declared global support, not repaired for this metric.
        value = range_auc_volume(labels, scores, window_size=12)[1] if valid else np.nan
        records.append({**row, "value": value, "metric": "vus_pr",
                        "base_score_sha256": _array_sha256(scores),
                        "metric_version": "range-volume-v1-window12-thresholds250"})
    pd.DataFrame(records).to_parquet(target, index=False)
    print("Computed VUS-PR from saved scores with window=12 and 250 thresholds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("core", "variants", "vus", "score-shift"))
    parser.add_argument("--derived", type=Path, help="Fresh core/variant/score-shift aggregation directory")
    args = parser.parse_args()
    if args.mode != "vus" and args.derived is None:
        parser.error("--derived is required; bundled historical evidence must not be mixed with fresh runs")
    if args.mode == "core":
        core(args.derived)
    elif args.mode == "variants":
        variants(args.derived)
    elif args.mode == "vus":
        vus()
    else:
        target = DETAIL / "score_shift_full_analysis.csv"
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(args.derived / target.name, target)


if __name__ == "__main__":
    main()
