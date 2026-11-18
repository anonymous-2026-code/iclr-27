#!/usr/bin/env python3
"""Four-GPU rebuttal controls; source protocol/artifacts remain read-only."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, precision_recall_curve
import torch

from section5_repro.artifacts import atomic_json, finite_tensor, sha256_file
from section5_repro.data import Entity, discover_entities, entity_manifest, prepare_series
from section5_repro.engine import (
    _build_tta_optimizer, _checkpoint_complete, _load_checkpoint, _point_loss,
    _score_loader, build_loaders, set_seed,
)
from section5_repro.paths import Workspace, checkpoint_path
from section5_repro.protocol import load_protocol

NORMALIZATION = "selected_count_normalization"
SHIFT = "controlled_distribution_shift"
EXPERIMENTS = (NORMALIZATION, SHIFT)
ARM_NAMES = {"frozen": "frozen", "practical": "practical_static_p95",
             "oracle": "oracle_benchmark_normal_point"}
VERSION = "sklearn-pr-curve-trapezoid-v1"
CONDITIONS = {"identity": (1.0, 0.0), "offset_0p5": (1.0, 0.5),
              "offset_1p0": (1.0, 1.0), "scale_1p25": (1.25, 0.0),
              "scale_1p5": (1.5, 0.0)}
SHIFT_DATASETS = ("Daphnet", "Genesis", "GECCO")
EXPECTED_ENTITIES = {"Daphnet": 1, "GECCO": 1, "Genesis": 1, "MSL": 25,
                     "PSM": 1, "PUMP": 1, "SMAP": 52, "SMD": 23, "SWAT2015": 1}
PHYSICAL = ROOT / "workspace/rebuttal_inputs/p3_iclr27_main_auprc_corrected_physical.parquet"
SUPPORT = ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_manifest.csv"


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def metric(labels, scores):
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Non-finite or misaligned scores; filtering is forbidden")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The complete original evaluation requires both binary classes")
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return float(auc(recall, precision))


def reduction(loss_points, mask, mode):
    if loss_points.shape != mask.shape or loss_points.shape != (1, 12):
        raise ValueError("Expected aligned single-window [1,12] point losses/mask")
    selected = int(mask.sum().item())
    weighted = loss_points * mask.to(loss_points.dtype)
    if mode == "full":
        return weighted.mean(), selected
    if mode != "selected":
        raise ValueError(mode)
    # An empty selection keeps an attached zero gradient and the scheduled step.
    return (weighted.sum() / selected if selected else weighted.sum() * 0.0), selected


def transform(series, condition):
    scale, offset = CONDITIONS[condition]
    active = np.ptp(series.full_source, axis=0) > 0
    if condition == "identity":
        return series
    if not active.any():
        raise ValueError("Prespecified shift dataset has no nonconstant source feature")
    test = series.test.copy()
    test[:, active] = test[:, active] * np.float32(scale) + np.float32(offset)
    if not np.isfinite(test).all():
        raise FloatingPointError("Non-finite transformed features")
    return replace(series, test=test)


def reference(base, arm, optimizer):
    return base["references"]["shared:frozen" if arm == "frozen" else f"{optimizer}:{arm}"]


def run_directory(task, base, optimizer, arm, output_root=None):
    root = Path(output_root) if output_root is not None else HERE.parent / task["experiment"] / "runs"
    return (root / task["condition"] / optimizer / arm / base["dataset"] /
            (base["subset"] or "__root__") / f"seed{base['seed']}")


def task_runs(task, protocol):
    if task["experiment"] == SHIFT:
        yield "shared", "frozen"
    for config in protocol.tta_configs:
        yield config.key, "practical"
        yield config.key, "oracle"


def prepare():
    if (HERE / "plan.json").exists():
        raise FileExistsError("Refusing to replace the prespecified launch plan")
    protocol = load_protocol()
    support = pd.read_csv(SUPPORT, keep_default_na=False)
    kept = support[support.support_status == "retained"]
    if kept.groupby("dataset").size().to_dict() != EXPECTED_ENTITIES:
        raise ValueError("Submitted common support differs from the fixed design")
    retained = {(r.dataset, "" if r.entity == "__root__" else r.entity) for r in kept.itertuples()}
    entities = {(e.dataset, e.subset): e for e in discover_entities(ROOT / "data/processed")}
    frame = pd.read_parquet(PHYSICAL).fillna({"subset": ""})
    lookup = {(r.dataset, r.subset, int(r.seed), r.optimizer_id, r.arm): r for r in frame.itertuples()}
    if len(lookup) != len(frame):
        raise ValueError("Duplicate original physical reference")
    timing = pd.read_csv(HERE.parent / "ae_all_normal_window/derived/physical.csv").fillna({"subset": ""})
    seconds = timing.groupby(["dataset", "subset", "seed"]).wall_seconds.mean().to_dict()
    bases, tasks, shift_rows = {}, [], []
    for dataset, subset in sorted(retained):
        entity = entities[dataset, subset]
        data_meta = entity_manifest(entity)
        if dataset in SHIFT_DATASETS:
            series = prepare_series(entity.directory, protocol.train_fraction)
            n = len(series.test_labels) // 12 * 12
            normal = series.test_labels[:n] == 0
            active = np.ptp(series.full_source, axis=0) > 0
            for condition in CONDITIONS:
                shifted = transform(series, condition)
                for j, feature in enumerate(series.feature_names):
                    shift_rows.append({"dataset": dataset, "condition": condition, "feature": feature,
                        "active_feature": bool(active[j]), "normal_test_points": int(normal.sum()),
                        "source_train_points": len(series.train),
                        "wasserstein1": wasserstein_distance(series.train[:, j], shifted.test[:n, j][normal])})
        for seed in protocol.seeds:
            checkpoint = checkpoint_path(Workspace(HERE, ROOT / "workspace/ae/checkpoints"), entity, seed, protocol)
            if not _checkpoint_complete(checkpoint, protocol, entity, seed):
                raise ValueError(f"Incomplete/mismatched source checkpoint: {checkpoint}")
            checkpoint_hash = sha256_file(checkpoint)
            refs = {}
            keys = [("shared", "frozen")] + [(c.key, arm) for c in protocol.tta_configs
                                                       for arm in ("practical", "oracle")]
            expected_labels = None
            for optimizer, arm in keys:
                row = lookup[dataset, subset, seed, optimizer, ARM_NAMES[arm]]
                meta = json.loads(Path(row.manifest_path).read_text())
                if (meta["source_checkpoint_hash"] != checkpoint_hash or meta["data_manifest"] != data_meta
                        or meta["status"] != "complete"):
                    raise ValueError(f"Original run provenance mismatch: {row.physical_run_id}")
                if (row.corrected_metric_version != VERSION or not np.isfinite(row.corrected_pr_curve_auc)
                        or row.n_valid != row.n_total):
                    raise ValueError(f"Invalid canonical metric: {row.physical_run_id}")
                for path in (row.score_path, row.label_path):
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                labels = np.load(row.label_path, allow_pickle=False).reshape(-1)
                if expected_labels is None:
                    expected_labels = labels
                elif not np.array_equal(labels, expected_labels):
                    raise ValueError("Original F/P/O evaluation labels differ")
                refs[f"{optimizer}:{arm}"] = {"score_path": row.score_path, "label_path": row.label_path,
                    "manifest_path": row.manifest_path, "auprc": float(row.corrected_pr_curve_auc),
                    "physical_run_id": row.physical_run_id}
            base_id = f"{dataset}/{subset or '__root__'}/seed{seed}"
            bases[base_id] = {"dataset": dataset, "subset": subset, "seed": seed,
                "directory": str(entity.directory), "checkpoint": str(checkpoint),
                "source_checkpoint_sha256": checkpoint_hash, "data_manifest": data_meta, "references": refs}
            cost = seconds[dataset, subset, seed]
            tasks.append({"base_id": base_id, "experiment": NORMALIZATION, "condition": "identity",
                          "estimated_seconds": cost * 8, "physical_runs": 8})
            if dataset in SHIFT_DATASETS:
                for condition in CONDITIONS:
                    if condition != "identity":
                        tasks.append({"base_id": base_id, "experiment": SHIFT, "condition": condition,
                                      "estimated_seconds": cost * 9, "physical_runs": 9})
    if len(bases) != 530 or len(tasks) != 590 or sum(t["physical_runs"] for t in tasks) != 4780:
        raise ValueError("Task grid does not match 530 bases / 590 tasks / 4780 runs")
    loads = [0.0] * 4
    tasks.sort(key=lambda t: (-t["estimated_seconds"], t["experiment"], t["base_id"], t["condition"]))
    for task_id, task in enumerate(tasks):
        shard = min(range(4), key=lambda j: loads[j])
        task.update(task_id=task_id, shard=shard)
        loads[shard] += task["estimated_seconds"]
    plan = {"created_at": now(), "protocol": protocol.as_dict(), "metric_version": VERSION,
        "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4], "bases": bases, "tasks": tasks,
        "expected_runs": {NORMALIZATION: 4240, SHIFT: 540},
        "expected_cells": {NORMALIZATION: 180, SHIFT: 240}, "conditions": CONDITIONS,
        "shard_estimated_hours": [v / 3600 for v in loads],
        "python": sys.executable, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    if not set(plan["cpu_cores"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("Requested four CPU cores are not available to this process")
    # These are analysis outputs of the prespecified transformations, not model results.
    shift_frame = pd.DataFrame(shift_rows)
    target = HERE.parent / SHIFT / "input_diagnostics"
    target.mkdir(parents=True, exist_ok=False)
    shift_frame.to_csv(target / "feature_shift.csv", index=False)
    shift_frame.groupby(["dataset", "condition"]).agg(
        w1_median=("wasserstein1", "median"), w1_mean=("wasserstein1", "mean"),
        active_features=("active_feature", "sum"), features=("feature", "size")
    ).reset_index().to_csv(target / "shift_summary.csv", index=False)
    atomic_json(HERE / "plan.json", plan)
    print(json.dumps({"bases": len(bases), "tasks": len(tasks), "expected_runs": plan["expected_runs"],
                      "shard_estimated_hours": plan["shard_estimated_hours"]}), flush=True)


def run_one(base, task, optimizer_key, arm, source_series, *, output_root=None, reduction_override=None):
    protocol = load_protocol()
    directory = run_directory(task, base, optimizer_key, arm, output_root)
    directory.mkdir(parents=True, exist_ok=False)
    series = transform(source_series, task["condition"])
    mode = reduction_override or ("selected" if task["experiment"] == NORMALIZATION else "full")
    device = torch.device("cuda:0")
    set_seed(base["seed"])
    model = _load_checkpoint(Path(base["checkpoint"]), series.train.shape[1], protocol, device)
    model._tta_scope = "full_model"
    batch_size = protocol.frozen_test_batch_size if arm == "frozen" else 1
    _, validation_loader, test_loader = build_loaders(series, protocol, batch_size, 0)
    started = time.perf_counter()
    started_cpu = time.process_time()
    tau = None
    steps, mask_parts, rows = 0, [], []
    if arm == "frozen":
        scores, labels = _score_loader(model, test_loader, device)
        selected = np.zeros(len(scores), dtype=bool)
        final_displacement = step_sum = 0.0
    else:
        calibration, _ = _score_loader(model, validation_loader, device)
        if not np.isfinite(calibration).all():
            raise FloatingPointError("Invalid source-validation scores")
        tau = float(np.percentile(calibration, 95.0))
        config = protocol.tta_by_key(optimizer_key)
        optimizer = _build_tta_optimizer(model, config)
        params = list(model.parameters())
        initial = [p.detach().clone() for p in params]
        score_parts, label_parts = [], []
        for window, (batch_x, batch_y) in enumerate(test_loader):
            x = batch_x.float().to(device)
            model.eval()
            with torch.no_grad():
                score_before = _point_loss(model, x, stage="test")
            finite_tensor("pre-update scores", score_before, window)
            # Practical decisions never consume benchmark labels.
            if arm == "practical":
                mask = score_before < tau
            elif arm == "oracle":
                mask = batch_y.to(device) == 0
            else:
                raise ValueError(arm)
            model.train()
            optimizer.zero_grad()
            point_loss = _point_loss(model, x, stage="tta")
            loss, k = reduction(point_loss, mask, mode)
            finite_tensor("adaptation loss", loss, window)
            previous = [p.detach().clone() for p in params]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                step_l2 = torch.stack([(p.detach() - old).square().sum()
                                       for p, old in zip(params, previous)]).sum().sqrt()
            finite_tensor("actual parameter step L2", step_l2, window)
            steps += 1
            score_parts.append(score_before.detach().cpu().numpy())
            label_parts.append(batch_y.numpy())
            mask_parts.append(mask.detach().cpu().numpy())
            rows.append({"window": window, "selected_points": k, "masked_loss": float(loss.item()),
                         "optimizer_step": True, "parameter_step_l2": float(step_l2.item())})
        scores = np.concatenate(score_parts).reshape(-1)
        labels = np.concatenate(label_parts).reshape(-1)
        selected = np.concatenate(mask_parts).reshape(-1).astype(bool)
        final_displacement = float(torch.stack([(p.detach() - old).square().sum()
                                  for p, old in zip(params, initial)]).sum().sqrt().item())
        step_sum = sum(r["parameter_step_l2"] for r in rows)
    labels = labels.astype(np.int8)
    expected = np.load(reference(base, arm, optimizer_key)["label_path"], allow_pickle=False).reshape(-1)
    if not np.array_equal(labels, expected):
        raise ValueError("Original labels/evaluation prefix changed")
    if len(labels) % 12 or (arm != "frozen" and steps != len(labels) // 12):
        raise ValueError("Window/step schedule changed")
    if arm == "oracle" and not np.array_equal(selected, labels == 0):
        raise ValueError("Oracle mask differs from benchmark-normal points")
    if arm == "practical" and not np.array_equal(selected, scores < tau):
        raise ValueError("Practical mask differs from source P95 pre-update scores")
    if not np.isfinite([final_displacement, step_sum]).all():
        raise FloatingPointError("Invalid parameter diagnostic")
    value = metric(labels, scores)
    count = int(selected.sum())
    normal_count = int((selected & (labels == 0)).sum())
    result = {"complete": True, "finished_at": now(), "experiment": task["experiment"],
        "condition": task["condition"], "dataset": base["dataset"], "subset": base["subset"],
        "seed": base["seed"], "optimizer": optimizer_key, "arm": arm, "reduction": mode,
        "source_checkpoint_path": base["checkpoint"], "source_checkpoint_sha256": base["source_checkpoint_sha256"],
        "metric_version": VERSION, "auprc": value, "selector_tau": tau,
        "test_points": len(labels), "test_windows": len(labels) // 12, "optimizer_steps": steps,
        "selected_points": count, "selected_fraction": float(selected.mean()),
        "selected_normal_points": normal_count, "selected_anomalous_points": count - normal_count,
        "normal_coverage": normal_count / int((labels == 0).sum()),
        "selector_purity": normal_count / count if count else None,
        "zero_selected_windows": sum(r["selected_points"] == 0 for r in rows) if rows else None,
        "parameter_step_l2_sum": step_sum, "final_source_parameter_l2": final_displacement,
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - started_cpu,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "output_path": str(directory)}
    np.save(directory / "test_energy.npy", scores)
    np.save(directory / "true.npy", labels)
    np.save(directory / "update_mask.npy", selected)
    if rows:
        pd.DataFrame(rows).to_parquet(directory / "window_trace.parquet", index=False)
    atomic_json(directory / "result.json", result)
    return result, scores, labels


def worker(plan, shard):
    torch.cuda.set_device(0)
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Each worker must have exactly one CPU core")
    for task in (t for t in plan["tasks"] if t["shard"] == shard):
        base = plan["bases"][task["base_id"]]
        entity = Entity(base["dataset"], base["subset"], Path(base["directory"]))
        if entity_manifest(entity) != base["data_manifest"] or sha256_file(Path(base["checkpoint"])) != base["source_checkpoint_sha256"]:
            raise ValueError(f"Input changed since preparation: {task['base_id']}")
        series = prepare_series(entity.directory, load_protocol().train_fraction)
        for optimizer, arm in task_runs(task, load_protocol()):
            print(f"START {task['task_id']} {task['experiment']} {task['base_id']} {task['condition']} {optimizer} {arm}", flush=True)
            result, _, _ = run_one(base, task, optimizer, arm, series)
            print(f"DONE {task['task_id']} {optimizer} {arm} auprc={result['auprc']:.10g} seconds={result['wall_seconds']:.2f}", flush=True)


def supervise(plan):
    if json.loads((HERE / "validation.json").read_text())["status"] != "passed":
        raise RuntimeError("Preflight validation has not passed")
    for experiment in EXPERIMENTS:
        if (HERE.parent / experiment / "runs").exists() or (HERE.parent / experiment / "derived").exists():
            raise FileExistsError(f"Formal outputs already exist: {experiment}")
    if (HERE / "status.json").exists():
        raise FileExistsError("Refusing to replace an existing batch status")
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    status = {"status": "running", "started_at": now(), "pid": os.getpid(),
              "devices": plan["devices"], "expected_runs": plan["expected_runs"]}
    processes, handles = [], []
    atomic_json(HERE / "status.json", status)
    try:
        for shard, gpu in enumerate(plan["devices"]):
            handle = (logs / f"gpu{gpu}.log").open("x")
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(["taskset", "-c", str(plan["cpu_cores"][shard]), sys.executable,
                str(HERE / "run.py"), "worker", "--shard", str(shard)], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(proc)
        status["workers"] = [{"gpu": gpu, "pid": p.pid, "cpu_core": core}
            for gpu, p, core in zip(plan["devices"], processes, plan["cpu_cores"])]
        atomic_json(HERE / "status.json", status)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(p.wait): p for p in processes}
            for future in as_completed(futures):
                code = future.result()
                if code:
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError(f"Worker {futures[future].pid} failed with exit code {code}; inspect its log")
        subprocess.run([sys.executable, str(HERE / "aggregate.py")], cwd=ROOT, check=True)
        status.update(status="complete", finished_at=now())
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        status.update(status="failed", finished_at=now(), error=str(exc))
        raise
    finally:
        atomic_json(HERE / "status.json", status)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "supervise"))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    configure_threads()
    if args.mode == "prepare":
        prepare()
        return
    plan = json.loads((HERE / "plan.json").read_text())
    if json.loads(json.dumps(load_protocol().as_dict())) != plan["protocol"]:
        raise ValueError("Source protocol changed after planning")
    if args.mode == "worker":
        if args.shard not in range(4):
            raise ValueError("Invalid worker shard")
        worker(plan, args.shard)
    else:
        supervise(plan)


if __name__ == "__main__":
    main()
