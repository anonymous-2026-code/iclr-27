#!/usr/bin/env python3
"""Extend the current ICLR AE variants to four optimizers, in isolated outputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from sklearn.metrics import auc, precision_recall_curve

from section5_repro import supplement_engine as engine
from section5_repro.artifacts import atomic_json, sha256_file
from section5_repro.data import Entity, discover_entities, entity_manifest, prepare_series
from section5_repro.engine import RunOptions
from section5_repro.paths import Workspace, checkpoint_path
from section5_repro.protocol import load_protocol
from section5_repro.supplement_protocol import SupplementVariant, load_supplement_protocol

PHYSICAL = ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/physical_auprc.parquet"
EXISTING_OPT = "adamw_ttlr0.0001"
NEW_OPTS = ("sgd_ttlr0.001", "sgd_ttlr0.0001", "adamw_ttlr0.0003")
PROFILES = {"m2n2": "standardized_m2n2_matched_diagnostic",
            "fpm": "fpm_only_selector_control", "sana": "matched_fpm_sana_diagnostic"}
ARMS = {"m2n2_practical_p95": "e2", "m2n2_oracle_point": "e2",
        "fpm_style_practical": "e4", "practical_fpm_sana": "e7",
        "oracle_normal_window_sana": "e7", "oracle_clean_fpm_sana": "e7"}
EXPECTED = {"Daphnet": 1, "GECCO": 1, "Genesis": 1, "MSL": 27, "PSM": 1,
            "PUMP": 1, "SMAP": 55, "SMD": 28, "SWAT2015": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def metric(labels, scores):
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Nonfinite or misaligned scores; no finite-prefix filtering")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected both benchmark classes")
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return float(auc(recall, precision))


def output_dir(base, optimizer, arm, output_root=None):
    workspace = Workspace((Path(output_root) if output_root else HERE / "outputs") / optimizer,
                          ROOT / "workspace/ae/checkpoints")
    entity = Entity(base["dataset"], base["subset"], Path(base["directory"]))
    return engine.supplement_result_dir(workspace, ARMS[arm], arm, entity, base["seed"])


def prepare():
    if (HERE / "plan.json").exists():
        raise FileExistsError("Plan already exists; no overwrite")
    protocol = load_protocol()
    entities = discover_entities(ROOT / "data/processed")
    counts = pd.Series([e.dataset for e in entities]).value_counts().to_dict()
    if counts != EXPECTED:
        raise ValueError(f"Expected original 116-entity variant support: {counts}")
    physical = pd.read_parquet(PHYSICAL).fillna({"subset": ""})
    lookup = {(r.profile_id, r.dataset, r.subset, int(r.seed), r.optimizer_id, r.arm): r
              for r in physical.itertuples()}
    bases, tasks = {}, []
    for entity in entities:
        for seed in protocol.seeds:
            base_id = f"{entity.dataset}/{entity.subset or '__root__'}/seed{seed}"
            refs = {}

            def reference(key, profile, optimizer, arm):
                r = lookup[profile, entity.dataset, entity.subset, seed, optimizer, arm]
                for path in (r.score_path, r.label_path):
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                refs[key] = {"physical_run_id": r.physical_run_id, "score_path": r.score_path,
                             "label_path": r.label_path, "label_digest": r.label_digest,
                             "valid": bool(r.raw_valid), "invalid_reason": r.invalid_reason,
                             "auprc": float(r.corrected_auprc) if r.raw_valid else None}

            reference("m2n2_frozen", PROFILES["m2n2"], "none", "m2n2_dt_only")
            reference("fpm_frozen", "core_oracle_heterogeneity", "shared", "frozen")
            reference("sana_frozen", PROFILES["sana"], "none", "frozen_sana")
            for arm, experiment in ARMS.items():
                profile = PROFILES[{"e2": "m2n2", "e4": "fpm", "e7": "sana"}[experiment]]
                reference(arm, profile, EXISTING_OPT, arm)
            for config in protocol.tta_configs:
                reference(f"fpm_oracle:{config.key}", "core_oracle_heterogeneity", config.key,
                          "oracle_benchmark_normal_point")
            meta = json.loads(Path(refs["m2n2_frozen"]["score_path"]).with_name("result.json").read_text())
            checkpoint = checkpoint_path(Workspace(HERE, ROOT / "workspace/ae/checkpoints"), entity, seed, protocol)
            if not checkpoint.with_name("complete.json").is_file():
                raise FileNotFoundError(checkpoint.with_name("complete.json"))
            checkpoint_hash = sha256_file(checkpoint)
            if checkpoint_hash != meta["source_checkpoint_sha256"] or entity_manifest(entity) != meta["entity"]:
                raise ValueError(f"Source or data no longer matches ICLR variants: {base_id}")
            estimated_seconds = 0.
            for arm in ARMS:
                old = json.loads(Path(refs[arm]["score_path"]).with_name("result.json").read_text())
                if old["source_checkpoint_sha256"] != checkpoint_hash or not old["complete"]:
                    raise ValueError(f"Unmatched existing variant: {base_id}/{arm}")
                estimated_seconds += float(old["wall_seconds"])
            for key, ref in refs.items():
                if ref["valid"] and ref["label_digest"] != refs["fpm_frozen"]["label_digest"]:
                    raise ValueError(f"Existing label mismatch: {base_id}/{key}")
            bases[base_id] = {"dataset": entity.dataset, "subset": entity.subset, "seed": seed,
                              "directory": str(entity.directory), "checkpoint": str(checkpoint),
                              "source_checkpoint_sha256": checkpoint_hash, "data_manifest": meta["entity"],
                              "references": refs}
            for optimizer in NEW_OPTS:
                tasks.append({"base_id": base_id, "optimizer": optimizer,
                              "estimated_seconds": estimated_seconds})
    tasks.sort(key=lambda t: (-t["estimated_seconds"], t["base_id"], t["optimizer"]))
    loads = [0.] * 4
    for index, task in enumerate(tasks):
        shard = min(range(4), key=lambda s: loads[s])
        task.update(task_id=index, shard=shard)
        loads[shard] += task["estimated_seconds"]
    if len(bases) != 580 or len(tasks) != 1740:
        raise ValueError("Unexpected expansion size")
    invalid_controls = sum(not ref["valid"] for b in bases.values()
                           for key, ref in b["references"].items() if key.startswith("fpm_oracle:"))
    plan = {"created_at": now(), 
            "source_protocol": protocol.as_dict(), "supplement_protocol": load_supplement_protocol().as_dict(),
            "nominal_entities": EXPECTED, "seeds": list(protocol.seeds), "new_optimizers": list(NEW_OPTS),
            "reused_optimizer": EXISTING_OPT, "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4],
            "new_physical_runs": 10440, "logical_cells_per_comparison": 180, "comparisons": 4,
            "new_runs_per_family": {"m2n2": 3480, "fpm": 1740, "sana": 5220},
            "existing_invalid_fpm_oracle_controls": invalid_controls,
            "support_policy": "Original 116 entities; any missing/invalid member makes its nominal cell undefined; no entity deletion",
            "failure_policy": "Record numerical failures without retries or hyperparameter changes; integrity failures stop the batch",
            "reuse_policy": "Current ICLR artifacts only; Frozen shared; historical M2N2 extension excluded",
            "metric": "sklearn-pr-curve-trapezoid-v1; entity-paired deltas then dataset median; strict delta > 0.005",
            "estimated_shard_hours_from_existing_runs": [v / 3600 for v in loads],
            "bases": bases, "tasks": tasks}
    atomic_json(HERE / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ["new_physical_runs", "new_runs_per_family",
                     "existing_invalid_fpm_oracle_controls", "estimated_shard_hours_from_existing_runs"]}, indent=2))


def run_one(base, optimizer, arm, series, validation_scores, output_root=None, max_test_windows=None):
    protocol = load_protocol()
    config = protocol.tta_by_key(optimizer)
    supplement = replace(load_supplement_protocol(), protocol_id=f"iclr27-variants-180-{optimizer}-v1")
    workspace = Workspace((Path(output_root) if output_root else HERE / "outputs") / optimizer,
                          ROOT / "workspace/ae/checkpoints")
    entity = Entity(base["dataset"], base["subset"], Path(base["directory"]))
    directory = output_dir(base, optimizer, arm, output_root)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite run: {directory}")
    args = dict(entity=entity, seed=base["seed"], source_protocol=protocol, supplement=supplement,
                workspace=workspace, options=RunOptions(device="cuda", num_workers=0, resume=False), series=series,
                max_test_windows=max_test_windows)
    if ARMS[arm] == "e2":
        result = engine.run_m2n2_online(**args, validation_scores=validation_scores,
                                       variant=SupplementVariant(arm, optimizer_key=optimizer, arm=arm))
    elif ARMS[arm] == "e4":
        result = engine.run_pointwise_online(**args, validation_scores=validation_scores,
                    experiment="e4", variant=arm, optimizer_config=config,
                    selector_factory=engine._fpm_factory(supplement.fpm))
    else:
        result = engine.run_e7_arms(**args, variants=(SupplementVariant(arm, arm=arm),),
                                    optimizer_config=config)[0]
    scores = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
    labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
    expected_points = len(series.test_labels) // 12 * 12
    if max_test_windows is not None:
        expected_points = min(expected_points, max_test_windows * 12)
    np.testing.assert_array_equal(labels, series.test_labels[:expected_points])
    if result["source_checkpoint_sha256"] != base["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint identity changed")
    if result["optimizer"] != config.optimizer or result["learning_rate"] != config.learning_rate:
        raise ValueError("Executed optimizer differs from requested optimizer")
    mask = np.load(directory / "update_mask.npy", allow_pickle=False).reshape(-1)
    if arm == "m2n2_oracle_point":
        np.testing.assert_array_equal(mask, labels == 0)
    if ARMS[arm] == "e7":
        old = json.loads(Path(base["references"]["sana_frozen"]["score_path"]).with_name("result.json").read_text())
        if result["sana_init_sha256"] != old["sana_init_sha256"] or result["backbone_changed"]:
            raise ValueError("SANA initialization/source differs from matched Frozen")
        if result["update_scope"] != "sana_in_and_sana_out_only":
            raise ValueError("Unexpected SANA update scope")
        normal_windows = np.repeat((labels.reshape(-1, 12) == 0).all(axis=1), 12)
        if arm == "oracle_normal_window_sana":
            np.testing.assert_array_equal(mask, normal_windows)
        elif arm == "oracle_clean_fpm_sana" and np.any(mask & ~normal_windows):
            raise ValueError("Clean-FPM Oracle selected a non-normal window")
    record = {"dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
              "optimizer": optimizer, "arm": arm,
              "status": "complete" if max_test_windows is None else "validation_prefix",
              "valid": max_test_windows is None,
              "auprc": metric(labels, scores) if max_test_windows is None else None,
              "points": len(labels), "score_path": str(directory / "test_energy.npy"),
              "label_path": str(directory / "true.npy"), "output_path": str(directory),
              "source_checkpoint_sha256": result["source_checkpoint_sha256"],
              "wall_seconds": result["wall_seconds"], "finished_at": now()}
    atomic_json(directory / "extension_metrics.json", record)
    return record


def load_base_inputs(base):
    protocol = load_protocol()
    entity = Entity(base["dataset"], base["subset"], Path(base["directory"]))
    if entity_manifest(entity) != base["data_manifest"]:
        raise ValueError("Processed data changed since planning")
    if sha256_file(Path(base["checkpoint"])) != base["source_checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since planning")
    series = prepare_series(entity.directory, protocol.train_fraction)
    validation = engine._calibration_for_series(entity=entity, seed=base["seed"], source_protocol=protocol,
        workspace=Workspace(HERE / "outputs", ROOT / "workspace/ae/checkpoints"),
        options=RunOptions(device="cuda", num_workers=0, resume=False), series=series)
    old = json.loads(Path(base["references"]["m2n2_practical_p95"]["score_path"]).with_name("result.json").read_text())
    if abs(float(np.percentile(validation, 95)) - old["numeric_threshold"]) > 1e-10:
        raise ValueError("Source-validation calibration differs from existing M2N2")
    return series, validation


def worker(shard):
    configure_threads()
    torch.cuda.set_device(0)
    plan = json.loads((HERE / "plan.json").read_text())
    tasks = [t for t in plan["tasks"] if t["shard"] == shard]
    progress = {"status": "running", "shard": shard, "pid": os.getpid(), "started_at": now(),
                "planned": len(tasks) * len(ARMS), "completed": 0, "numerical_failures": 0}
    path = HERE / "progress" / f"gpu{shard}.json"
    atomic_json(path, progress)
    print(f"START gpu={shard} runs={progress['planned']} affinity={sorted(os.sched_getaffinity(0))}", flush=True)
    try:
        for task in tasks:
            base = plan["bases"][task["base_id"]]
            series, validation = load_base_inputs(base)
            for arm in ARMS:
                progress.update(current_base=task["base_id"], current_optimizer=task["optimizer"], current_arm=arm)
                atomic_json(path, progress)
                print(f"RUN {task['base_id']} {task['optimizer']} {arm}", flush=True)
                try:
                    result = run_one(base, task["optimizer"], arm, series, validation)
                except FloatingPointError as exc:
                    # A divergent trajectory is an explicit experimental outcome,
                    # not a reason to alter its learning rate or stop other runs.
                    traceback.print_exc()
                    directory = output_dir(base, task["optimizer"], arm)
                    atomic_json(directory / "failure.json", {"status": "numerical_failure", "valid": False,
                        "dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
                        "optimizer": task["optimizer"], "arm": arm, "error": str(exc),
                        "traceback": traceback.format_exc(), "finished_at": now()})
                    progress["numerical_failures"] += 1
                else:
                    progress["completed"] += 1
                    print(f"DONE auprc={result['auprc']:.9f}", flush=True)
                progress["updated_at"] = now()
                atomic_json(path, progress)
        progress.update(status="finished", finished_at=now())
    except BaseException as exc:
        progress.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
        raise
    finally:
        atomic_json(path, progress)


def supervise():
    plan = json.loads((HERE / "plan.json").read_text())
    if json.loads((HERE / "validation.json").read_text())["status"] != "passed":
        raise ValueError("Validation must pass before full execution")
    if (HERE / "status.json").exists():
        raise FileExistsError("Batch has already been started")
    status = {"status": "starting", "started_at": now(), "pid": os.getpid(),
              "planned_new_runs": plan["new_physical_runs"],
              "existing_invalid_fpm_oracle_controls": plan["existing_invalid_fpm_oracle_controls"], "workers": []}
    processes, handles = [], []
    (HERE / "logs").mkdir(exist_ok=False)
    atomic_json(HERE / "status.json", status)
    try:
        for gpu, core in zip(plan["devices"], plan["cpu_cores"]):
            handle = (HERE / "logs" / f"gpu{gpu}.log").open("x")
            handles.append(handle)
            process = subprocess.Popen(["taskset", "-c", str(core), sys.executable, "-B", str(HERE / "run.py"),
                    "worker", "--shard", str(gpu)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            status["workers"].append({"gpu": gpu, "cpu_core": core, "pid": process.pid})
        status["status"] = "running"
        atomic_json(HERE / "status.json", status)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(p.wait) for p in processes]):
                if future.result() != 0:
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError("Worker integrity/execution failure; inspect per-GPU progress and logs")
        subprocess.run([sys.executable, "-B", str(HERE / "aggregate.py")], cwd=ROOT, check=True)
        progress = [json.loads((HERE / "progress" / f"gpu{i}.json").read_text()) for i in plan["devices"]]
        failed = sum(p["numerical_failures"] for p in progress)
        summary = json.loads((HERE / "derived/summary.json").read_text())
        status.update(status="complete" if summary["valid_grid_complete"] else "finished_with_invalid_results",
                      execution_finished=True, valid_grid_complete=summary["valid_grid_complete"],
                      completed=sum(p["completed"] for p in progress), numerical_failures=failed, finished_at=now())
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        status.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
        raise
    finally:
        for p in processes:
            p.wait()
        for handle in handles:
            handle.close()
        atomic_json(HERE / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "supervise"))
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "supervise":
        supervise()
    elif args.shard is None:
        parser.error("worker requires --shard")
    else:
        worker(args.shard)
