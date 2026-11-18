#!/usr/bin/env python3
"""Isolated all-normal-window Oracle experiment; reuse the submitted AE updater."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve

from section5_repro.artifacts import atomic_json, sha256_file
from section5_repro.data import Entity, discover_entities, prepare_series
from section5_repro.engine import RunOptions
from section5_repro.paths import Workspace, checkpoint_path
from section5_repro.protocol import load_protocol
from section5_repro.supplement_engine import run_pointwise_online, supplement_result_dir
from section5_repro.supplement_protocol import load_supplement_protocol
from section5_repro.supplement_selectors import (
    CadenceController, ExternalPointMaskSelector, OraclePointSelector,
)

EXPERIMENT = "ae_all_normal_window"
METRIC_VERSION = "sklearn-pr-curve-trapezoid-v1"
SUPPORT = ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_manifest.csv"
PHYSICAL = ROOT / "workspace/rebuttal_inputs/p3_iclr27_main_auprc_corrected_physical.parquet"
EXPECTED = {"Daphnet": 1, "GECCO": 1, "Genesis": 1, "MSL": 25,
            "PSM": 1, "PUMP": 1, "SMAP": 52, "SMD": 23, "SWAT2015": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def corrected_auprc(labels, scores):
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Non-finite or misaligned score array; no filtering is allowed")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires the original binary two-class labels")
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return float(auc(recall, precision))


def window_mask(labels, window=12):
    labels = np.asarray(labels).reshape(-1)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Nonbinary benchmark labels")
    used = labels[:len(labels) // window * window].reshape(-1, window)
    clean = (used == 0).all(axis=1)
    anomalous = (used == 1).all(axis=1)
    mask = np.repeat(clean, window)
    normal_total = int((used == 0).sum())
    return mask, {
        "all_normal_windows": int(clean.sum()),
        "all_anomalous_windows": int(anomalous.sum()),
        "mixed_windows": int((~clean & ~anomalous).sum()),
        "all_normal_window_fraction": float(clean.mean()),
        "normal_points_total": normal_total,
        "normal_points_dropped": normal_total - int(mask.sum()),
        "normal_points_dropped_fraction": (
            (normal_total - int(mask.sum())) / normal_total if normal_total else None),
    }


def prepare(devices):
    target = HERE / "plan.json"
    if target.exists():
        raise FileExistsError(f"Plan already exists: {target}")
    protocol = load_protocol()
    support = pd.read_csv(SUPPORT, keep_default_na=False)
    kept = support[support.support_status == "retained"]
    if kept.groupby("dataset").size().to_dict() != EXPECTED:
        raise ValueError("Canonical retained support differs from the fixed design")
    retained = {(r.dataset, "" if r.entity == "__root__" else r.entity)
                for r in kept.itertuples()}
    entities = {(e.dataset, e.subset): e for e in discover_entities(ROOT / "data/processed")}
    physical = pd.read_parquet(PHYSICAL).fillna({"subset": ""})
    old_oracle = physical[physical.arm == "oracle_benchmark_normal_point"]
    lookup = {(r.dataset, r.subset, int(r.seed), r.optimizer_id): r
              for r in old_oracle.itertuples()}
    tasks = []
    workspace = Workspace(HERE / "outputs", ROOT / "workspace/ae/checkpoints")
    for dataset, subset in sorted(retained):
        entity = entities[dataset, subset]
        for seed in protocol.seeds:
            checkpoint = checkpoint_path(workspace, entity, seed, protocol)
            if not checkpoint.is_file() or not checkpoint.with_name("complete.json").is_file():
                raise FileNotFoundError(checkpoint)
            actual_checkpoint_hash = sha256_file(checkpoint)
            references = {}
            for config in protocol.tta_configs:
                row = lookup[dataset, subset, seed, config.key]
                manifest = json.loads(Path(row.manifest_path).read_text())
                if manifest["source_checkpoint_hash"] != actual_checkpoint_hash:
                    raise ValueError(f"Checkpoint differs from submitted Oracle: {checkpoint}")
                if not np.isfinite(row.corrected_pr_curve_auc):
                    raise ValueError(f"Invalid canonical reference: {row.physical_run_id}")
                for field in ("score_path", "label_path"):
                    if not Path(getattr(row, field)).is_file():
                        raise FileNotFoundError(getattr(row, field))
                references[config.key] = {
                    "score_path": row.score_path, "label_path": row.label_path,
                    "auprc": float(row.corrected_pr_curve_auc),
                    "source_checkpoint_sha256": actual_checkpoint_hash,
                }
            tasks.append({"dataset": dataset, "subset": subset, "seed": seed,
                          "directory": str(entity.directory), "references": references,
                          "cost": (entity.directory / "test.parquet").stat().st_size})
    if len(tasks) != 530:
        raise ValueError(f"Expected 530 tasks, got {len(tasks)}")
    loads = [0] * len(devices)
    tasks.sort(key=lambda r: (-r["cost"], r["dataset"], r["subset"], r["seed"]))
    for i, task in enumerate(tasks):
        shard = min(range(len(devices)), key=lambda j: loads[j])
        task.update(task_id=i, shard=shard, gpu=devices[shard])
        loads[shard] += task["cost"]
    plan = {"experiment": EXPERIMENT, "created_at": now(), "devices": devices,
            "source_protocol": protocol.as_dict(), "support": str(SUPPORT),
            "retained_entities": 106, "entity_seed_tasks": 530,
            "physical_runs": 2120, "logical_cells": 180, "tasks": tasks,
            "python": sys.executable, "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    atomic_json(target, plan)
    print(json.dumps({k: plan[k] for k in ("devices", "entity_seed_tasks", "physical_runs", "logical_cells")}))


def run_one(task, config, workspace_root, *, point_oracle=False):
    protocol = load_protocol()
    supplement = replace(load_supplement_protocol(), protocol_id="iclr27-rebuttal-all-normal-window-v1")
    entity = Entity(task["dataset"], task["subset"], Path(task["directory"]))
    workspace = Workspace.create(workspace_root, ROOT / "workspace/ae/checkpoints")
    series = prepare_series(entity.directory, protocol.train_fraction)
    mask, diagnostics = window_mask(series.test_labels, protocol.sequence_length)
    variant = f"point_parity_{config.key}" if point_oracle else config.key
    directory = supplement_result_dir(workspace, EXPERIMENT, variant, entity, task["seed"])
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment run: {directory}")

    def factory(model, validation_loader, device):
        if point_oracle:
            return OraclePointSelector(), {"oracle": "benchmark_normal_point", "validation_only": True}
        selector = ExternalPointMaskSelector(mask, protocol.sequence_length, protocol.test_stride)
        selector.uses_labels = True
        return selector, {"oracle": "all_normal_window", "uses_benchmark_labels": True,
                          "window_length": 12, "zero_mask_keeps_optimizer_step": True}

    result = run_pointwise_online(
        entity=entity, seed=task["seed"], source_protocol=protocol,
        supplement=supplement, workspace=workspace,
        options=RunOptions(device="cuda", num_workers=0, resume=False), series=series,
        # This interface argument is unused by the external/oracle selector.
        validation_scores=np.empty(0), experiment=EXPERIMENT, variant=variant,
        optimizer_config=config, selector_factory=factory, cadence=1,
        extra_metadata={"rebuttal_arm": "point_oracle_validation" if point_oracle else "all_normal_window_oracle",
                        "support_policy": "submitted-global-common-106-entities"},
    )
    score = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
    labels = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
    selected = np.load(directory / "update_mask.npy", allow_pickle=False).reshape(-1)
    reference = task["references"][config.key]
    if result["source_checkpoint_sha256"] != reference["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint no longer matches the canonical reference")
    expected_labels = np.load(reference["label_path"], allow_pickle=False).reshape(-1)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Evaluation labels/prefix differ from the submitted experiment")
    expected_mask = labels == 0 if point_oracle else mask
    if not np.array_equal(selected, expected_mask):
        raise ValueError("Recorded mask does not match the prespecified Oracle rule")
    if result["optimizer_steps"] != len(labels) // 12 or result["selected_anomalous_points"] != 0:
        raise ValueError("Optimizer schedule or normal-only selection violated")
    trace = pd.read_parquet(directory / "window_trace.parquet")
    if not trace.optimizer_step.all():
        raise ValueError("A scheduled optimizer step was skipped")
    if not point_oracle:
        if not (trace.loc[trace.selected_points == 0, "masked_loss"] == 0).all():
            raise ValueError("Rejected windows contribute nonzero data loss")
        result.update(diagnostics)
    result.update(legacy_average_precision=result["auprc"],
                  auprc=corrected_auprc(labels, score), metric_version=METRIC_VERSION)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "completed.json", result)
    return result, score, labels


def validate(plan):
    output = HERE / "validation"
    if output.exists():
        raise FileExistsError(output)
    # Mixed-window normals must be rejected; an incomplete tail stays outside evaluation.
    toy = np.array([0] * 12 + [0] * 11 + [1] + [1] * 12 + [0] * 5)
    mask, stats = window_mask(toy)
    np.testing.assert_array_equal(mask, [True] * 12 + [False] * 24)
    assert stats["mixed_windows"] == 1 and stats["normal_points_dropped"] == 11
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    controller = CadenceController(optimizer, 1)
    controller.backward(parameter.square())
    before = parameter.detach().clone()
    controller.backward(parameter.square() * 0)
    assert controller.steps == 2 and not torch.equal(before, parameter.detach())
    task = next(t for t in plan["tasks"] if t["dataset"] == "Daphnet" and t["seed"] == 42)
    config = load_protocol().tta_by_key("adamw_ttlr0.0001")
    original, score, labels = run_one(task, config, output, point_oracle=True)
    reference = task["references"][config.key]
    old_score = np.load(reference["score_path"], allow_pickle=False).reshape(-1)
    np.testing.assert_allclose(score, old_score, rtol=1e-5, atol=1e-6)
    if abs(original["auprc"] - reference["auprc"]) > 1e-8:
        raise AssertionError("Point Oracle corrected PR-AUC parity failed")
    clean, _, _ = run_one(task, config, output)
    report = {"status": "passed", "finished_at": now(),
              "toy_mask_and_zero_step": "passed", "point_oracle_parity": "passed",
              "point_oracle_max_score_error": float(np.max(np.abs(score - old_score))),
              "point_oracle_auprc_error": abs(original["auprc"] - reference["auprc"]),
              "all_normal_full_stream": "passed", "validation_dataset": "Daphnet",
              "validation_seed": 42, "validation_optimizer": config.key,
              "all_normal_windows": clean["all_normal_windows"],
              "mixed_windows": clean["mixed_windows"], "optimizer_steps": clean["optimizer_steps"]}
    atomic_json(HERE / "validation.json", report)
    print(json.dumps(report, indent=2))


def worker(plan, shard):
    protocol = load_protocol()
    for task in (t for t in plan["tasks"] if t["shard"] == shard):
        for config in protocol.tta_configs:
            print(f"START {task['task_id']} {task['dataset']}/{task['subset']} seed={task['seed']} {config.key}", flush=True)
            result, _, _ = run_one(task, config, HERE / "outputs")
            print(f"DONE {task['task_id']} {config.key} auprc={result['auprc']:.10g} coverage={result['normal_coverage']} seconds={result['wall_seconds']:.2f}", flush=True)


def supervise(plan):
    validation = json.loads((HERE / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("Preflight validation has not passed")
    if (HERE / "outputs/results").exists():
        raise FileExistsError("Formal results already exist; no automatic rerun or overwrite")
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles = [], []
    status = {"status": "running", "started_at": now(), "pid": os.getpid(),
              "devices": plan["devices"], "expected_physical_runs": 2120}
    atomic_json(HERE / "status.json", status)
    try:
        for shard, gpu in enumerate(plan["devices"]):
            handle = (logs / f"gpu{gpu}.log").open("x")
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        "worker", "--shard", str(shard)],
                                       cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
        status["workers"] = [{"gpu": gpu, "pid": p.pid} for gpu, p in zip(plan["devices"], processes)]
        atomic_json(HERE / "status.json", status)
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            futures = {pool.submit(p.wait): p for p in processes}
            for future in as_completed(futures):
                code = future.result()
                if code:
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError(f"Worker {futures[future].pid} exited with code {code}; see GPU logs")
        subprocess.run([sys.executable, str(HERE / "aggregate.py")], cwd=ROOT, check=True)
        status.update(status="complete", finished_at=now())
        atomic_json(HERE / "status.json", status)
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        status.update(status="failed", finished_at=now(), error=str(exc))
        atomic_json(HERE / "status.json", status)
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "worker", "run"))
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        devices = [int(v) for v in args.devices.split(",")]
        if len(devices) != 2 or len(set(devices)) != 2:
            raise ValueError("This launch requires two distinct GPU devices")
        prepare(devices)
        return
    plan = json.loads((HERE / "plan.json").read_text())
    if args.mode in {"validate", "worker"}:
        # The reused runner resets CUDA memory counters before loading its model.
        # Explicit device initialization is required before that counter call.
        torch.cuda.set_device(0)
    if args.mode == "validate":
        validate(plan)
    elif args.mode == "worker":
        if args.shard not in range(len(plan["devices"])):
            raise ValueError("Invalid worker shard")
        worker(plan, args.shard)
    else:
        supervise(plan)


if __name__ == "__main__":
    main()
