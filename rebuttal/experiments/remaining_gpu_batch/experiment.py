#!/usr/bin/env python3
"""Prespecified count matching and window-length controls, with exact reuse."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_controls"))
import run as controls
# Keep this batch's validate/aggregate modules ahead of the older batch's names.
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import torch
from section5_repro.engine import train_source, RunOptions

ROOT = controls.ROOT
COUNT = HERE.parent / "matched_update_count"
WINDOW = HERE.parent / "window_length_sensitivity"
SAMPLE_SEEDS = (2026110701, 2026110702, 2026110703)
WINDOW_DATASETS = ("Daphnet", "GECCO", "PUMP")
LENGTHS = (24, 48)
ARMS = ("practical", "oracle")


def original_inputs(base, optimizer, verify=False):
    refs = {a: controls.reference(base, a, optimizer) for a in ARMS}
    scores = np.load(refs["practical"]["score_path"], allow_pickle=False).reshape(-1)
    labels = np.load(refs["practical"]["label_path"], allow_pickle=False).reshape(-1)
    if len(labels) % 12 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid original complete score stream")
    metadata = json.loads(Path(refs["practical"]["manifest_path"]).read_text())
    if (metadata["status"] != "complete" or metadata["data_manifest"] != base["data_manifest"]
            or metadata["source_checkpoint_hash"] != base["source_checkpoint_sha256"]):
        raise ValueError("Original Practical provenance differs")
    tau = float(metadata["result_metadata"]["selector_tau"])
    if verify:
        for arm in ("frozen", *ARMS):
            ref = controls.reference(base, arm, optimizer)
            truth = np.load(ref["label_path"], allow_pickle=False).reshape(-1)
            values = np.load(ref["score_path"], allow_pickle=False).reshape(-1)
            if not np.array_equal(truth, labels) or abs(controls.metric(truth, values) - ref["auprc"]) > 1e-12:
                raise ValueError("Canonical source metric/labels changed")
    return labels, {"practical": (scores < tau).reshape(-1, 12),
                    "oracle": (labels == 0).reshape(-1, 12)}, tau


def thin_mask(mask, counts, rng):
    """Uniform within-window sampling through independent random priorities."""
    if mask.dtype != bool or mask.ndim != 2 or mask.shape[1] != 12:
        raise ValueError("Expected Boolean [windows,12] mask")
    if counts.shape != (len(mask),) or (counts < 0).any() or (counts > mask.sum(1)).any():
        raise ValueError("Invalid matched count")
    priorities = np.where(mask, rng.random(mask.shape), np.inf)
    order = np.argsort(priorities, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(12), order.shape), axis=1)
    selected = mask & (ranks < counts[:, None])
    if not np.array_equal(selected.sum(1), counts) or np.any(selected & ~mask):
        raise ValueError("Sampling violated count/subset contract")
    return selected


def matched_masks(original, task_index):
    counts = np.minimum(original["practical"].sum(1), original["oracle"].sum(1))
    return {(seed, arm): thin_mask(original[arm], counts,
                np.random.default_rng(np.random.SeedSequence([seed, task_index, ai])))
            for seed in SAMPLE_SEEDS for ai, arm in enumerate(ARMS)}


def deduplicate(original, masks):
    representatives = [({"kind": "original", "arm": a}, original[a]) for a in ARMS]
    slots, executions = [], []
    for (seed, arm), mask in masks.items():
        matches = [ref for ref, other in representatives if np.array_equal(mask, other)]
        if matches:
            reference = dict(matches[0])
        else:
            reference = {"kind": "new", "sample_seed": seed, "arm": arm}
            executions.append(dict(reference))
            representatives.append((dict(reference), mask))
        slots.append({"sample_seed": seed, "arm": arm, "reference": reference})
    return slots, executions


def count_directory(base, optimizer, reference, root=COUNT):
    return (Path(root) / "runs" / f"sample{reference['sample_seed']}" / optimizer / reference["arm"]
            / base["dataset"] / (base["subset"] or "__root__") / f"seed{base['seed']}")


def window_protocol(length):
    if length not in (12, *LENGTHS):
        raise ValueError("Unplanned window length")
    old = controls.load_protocol()
    return old if length == 12 else replace(old, sequence_length=length, train_stride=length,
        test_stride=length, protocol_id=f"{old.protocol_id}-rebuttal-window{length}")


def window_directory(base, length, optimizer, arm, root=WINDOW):
    directory = Path(root) / "runs" / f"window{length}" / base["dataset"]
    # Preserve all original single-entity paths; distinguish retained subseries.
    if base["subset"]:
        directory = directory / base["subset"]
    return directory / f"seed{base['seed']}" / optimizer / arm


def source_path(base, length):
    entity = controls.Entity(base["dataset"], base["subset"], Path(base["directory"]))
    return controls.checkpoint_path(controls.Workspace(WINDOW), entity, base["seed"], window_protocol(length))


def window_runs(protocol):
    return [("shared", "frozen")] + [(c.key, a) for c in protocol.tta_configs for a in ARMS]


def prepare():
    if (HERE / "plan.json").exists():
        raise FileExistsError("Prespecified plan exists")
    old = json.loads((HERE.parent / "gpu_controls/plan.json").read_text())
    if json.loads(json.dumps(controls.load_protocol().as_dict())) != old["protocol"]:
        raise ValueError("Original protocol changed")
    times = pd.read_csv(COUNT.parent / "selected_count_normalization/derived/physical.csv", keep_default_na=False)
    costs = {(r.dataset, r.subset, int(r.seed), r.optimizer, r.arm): r.wall_seconds for r in times.itertuples()}
    tasks, original_reuse, within_reuse = [], 0, 0
    for base_index, (base_id, base) in enumerate(sorted(old["bases"].items())):
        for config in controls.load_protocol().tta_configs:
            index = len(tasks)
            _, original, _ = original_inputs(base, config.key, verify=True)
            masks = matched_masks(original, index)
            slots, executions = deduplicate(original, masks)
            original_reuse += sum(s["reference"]["kind"] == "original" for s in slots)
            within_reuse += len(slots) - len(executions) - sum(s["reference"]["kind"] == "original" for s in slots)
            cost = sum(costs[base["dataset"], base["subset"], base["seed"], config.key, e["arm"]] for e in executions)
            tasks.append({"task_id": index, "experiment": "count", "base_id": base_id,
                "optimizer": config.key, "slots": slots, "executions": executions,
                "estimated_seconds": 1.3 * cost})
        if (base_index + 1) % 100 == 0:
            print(f"PREPARED {base_index+1}/530 bases", flush=True)
    for base_id, base in sorted(old["bases"].items()):
        if base["dataset"] not in WINDOW_DATASETS:
            continue
        if base["subset"]:
            raise ValueError("Window pilot requires the declared single entities")
        for length in LENGTHS:
            cost = sum(costs[base["dataset"], "", base["seed"], c.key, a]
                       for c in controls.load_protocol().tta_configs for a in ARMS)
            tasks.append({"task_id": len(tasks), "experiment": "window", "base_id": base_id,
                "length": length, "common_points": base["data_manifest"]["test_rows"] // 48 * 48,
                "estimated_seconds": 1.5 * cost * length / 12 + 30})
    if len(old["bases"]) != 530 or len(tasks) != 2150:
        raise ValueError("Expected 2120 count tasks and 30 window/source tasks")
    loads = [0.0] * 4
    for t in sorted(tasks, key=lambda t: (-t["estimated_seconds"], t["task_id"])):
        shard = min(range(4), key=lambda i: loads[i])
        t["shard"] = shard
        loads[shard] += t["estimated_seconds"]
    new_count = sum(len(t["executions"]) for t in tasks if t["experiment"] == "count")
    if new_count + original_reuse + within_reuse != 12720:
        raise ValueError("Exact-reuse accounting does not close")
    plan = {"created_at": controls.now(), "protocol": old["protocol"], "metric_version": controls.VERSION,
        "bases": old["bases"], "tasks": tasks, "sample_seeds": list(SAMPLE_SEEDS),
        "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4], "reserved_4gpu_hours": 12,
        "estimated_shard_hours": [x/3600 for x in loads],
        "expected": {"count_logical_arm_observations": 12720, "count_unique_new_runs": new_count,
            "count_original_reuse": original_reuse, "count_within_task_reuse": within_reuse,
            "window_new_sources": 30, "window_new_evaluations": 270},
        "reuse_audit": str(ROOT / "docs/guides/REBUTTAL.md")}
    controls.atomic_json(HERE / "plan.json", plan)
    print(json.dumps({"expected": plan["expected"], "estimated_shard_hours": plan["estimated_shard_hours"]}), flush=True)


def execute(base, protocol, checkpoint, series, directory, optimizer_key, arm, expected_labels,
            *, fixed_mask=None, expected_tau=None, parity_reference=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    controls.set_seed(base["seed"])
    model = controls._load_checkpoint(Path(checkpoint), series.train.shape[1], protocol, device)
    model._tta_scope = "full_model"
    width = protocol.sequence_length
    _, val_loader, loader = controls.build_loaders(series, protocol,
        protocol.frozen_test_batch_size if arm == "frozen" else 1, 0)
    started, cpu_start = time.perf_counter(), time.process_time()
    calibration, _ = controls._score_loader(model, val_loader, device)
    if not len(calibration) or not np.isfinite(calibration).all():
        raise ValueError("Invalid source-validation scores")
    tau = float(np.percentile(calibration, 95.0))
    if expected_tau is not None and tau != expected_tau:
        raise ValueError(f"Original source P95 changed: {tau} != {expected_tau}")
    rows, masks = [], []
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    if arm == "frozen":
        if fixed_mask is not None:
            raise ValueError("Frozen cannot receive a selection mask")
        scores, labels = controls._score_loader(model, loader, device)
        selection = np.zeros(len(scores), dtype=bool)
    else:
        optimizer = controls._build_tta_optimizer(model, protocol.tta_by_key(optimizer_key))
        score_parts, label_parts = [], []
        if fixed_mask is not None and fixed_mask.shape != (len(loader), width):
            raise ValueError("Fixed schedule must cover every original window")
        for step, (batch_x, batch_y) in enumerate(loader):
            x = batch_x.float().to(device)
            model.eval()
            with torch.no_grad():
                before = controls._point_loss(model, x, stage="test")
            controls.finite_tensor("pre-update scores", before, step)
            if fixed_mask is not None:
                mask = torch.as_tensor(fixed_mask[step:step+1], dtype=torch.bool, device=device)
            elif arm == "practical":
                mask = before < tau
            elif arm == "oracle":
                mask = batch_y.to(device) == 0
            else:
                raise ValueError(arm)
            if mask.shape != before.shape or before.shape != (1, width):
                raise ValueError("Window/loss/mask shape differs")
            model.train()
            optimizer.zero_grad()
            points = controls._point_loss(model, x, stage="tta")
            loss = (points * mask.to(points.dtype)).mean()
            controls.finite_tensor("masked loss", loss, step)
            previous = [p.detach().clone() for p in params]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                displacement = torch.stack([(p.detach()-old).square().sum() for p,old in zip(params,previous)]).sum().sqrt()
            controls.finite_tensor("parameter step", displacement, step)
            score_parts.append(before.detach().cpu().numpy())
            label_parts.append(batch_y.numpy())
            masks.append(mask.detach().cpu().numpy())
            rows.append({"window": step, "selected_points": int(mask.sum().item()), "masked_loss": float(loss.item()),
                         "optimizer_step": True, "parameter_step_l2": float(displacement.item())})
        scores = np.concatenate(score_parts).reshape(-1)
        labels = np.concatenate(label_parts).reshape(-1)
        selection = np.concatenate(masks).reshape(-1).astype(bool)
    labels = labels.astype(np.int8)
    if not np.array_equal(labels, expected_labels) or len(labels) % width:
        raise ValueError("Evaluation labels/common prefix changed")
    if arm != "frozen" and len(rows) != len(labels)//width:
        raise ValueError("Optimizer-step schedule changed")
    if fixed_mask is not None and not np.array_equal(selection, fixed_mask.reshape(-1)):
        raise ValueError("Replayed mask differs")
    if fixed_mask is None:
        expected = labels == 0 if arm == "oracle" else scores < tau if arm == "practical" else np.zeros(len(labels),bool)
        if not np.array_equal(selection, expected):
            raise ValueError("Online selection rule differs")
    value = controls.metric(labels, scores)
    parity_error = None
    if parity_reference is not None:
        old = np.load(parity_reference["score_path"], allow_pickle=False).reshape(-1)
        np.testing.assert_allclose(scores, old, rtol=1e-5, atol=1e-6)
        if abs(value-parity_reference["auprc"]) > 1e-8:
            raise ValueError("Full-stream metric parity failed")
        parity_error = float(np.max(np.abs(scores-old)))
    with torch.no_grad():
        final_step = float(torch.stack([(p.detach()-old).square().sum() for p,old in zip(params,initial)]).sum().sqrt().item())
    selected = int(selection.sum())
    selected_normal = int((selection & (labels==0)).sum())
    result = {"complete": True, "finished_at": controls.now(), "dataset": base["dataset"],
        "subset": base["subset"], "seed": base["seed"], "optimizer": optimizer_key, "arm": arm,
        "window_length": width, "train_stride": protocol.train_stride, "test_stride": protocol.test_stride,
        "source_checkpoint_path": str(checkpoint), "source_checkpoint_sha256": controls.sha256_file(Path(checkpoint)),
        "data_manifest": base["data_manifest"], "metric_version": controls.VERSION, "auprc": value,
        "test_points": len(labels), "test_windows": len(labels)//width, "test_anomalies": int(labels.sum()),
        "test_rows_omitted": base["data_manifest"]["test_rows"]-len(labels), "parameter_count": sum(p.numel() for p in params),
        "selector_tau": tau, "selection_kind": "fixed_matched_replay" if fixed_mask is not None else "online",
        "optimizer_steps": len(rows), "zero_selected_windows": sum(r["selected_points"]==0 for r in rows),
        "selected_points": selected, "selected_normal_points": selected_normal, "selected_anomalous_points": selected-selected_normal,
        "normal_coverage": selected_normal/int((labels==0).sum()), "selector_purity": selected_normal/selected if selected else None,
        "parameter_step_l2_sum": sum(r["parameter_step_l2"] for r in rows), "final_source_parameter_l2": final_step,
        "max_score_parity_error": parity_error, "wall_seconds": time.perf_counter()-started,
        "cpu_seconds": time.process_time()-cpu_start, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "output_path": str(directory)}
    np.save(directory/"test_energy.npy", scores)
    np.save(directory/"true.npy", labels)
    np.save(directory/"update_mask.npy", selection)
    if rows:
        pd.DataFrame(rows).to_parquet(directory/"window_trace.parquet", index=False)
    controls.atomic_json(directory/"result.json", result)
    return result


def check_base(base):
    entity = controls.Entity(base["dataset"], base["subset"], Path(base["directory"]))
    if (controls.entity_manifest(entity) != base["data_manifest"]
            or controls.sha256_file(Path(base["checkpoint"])) != base["source_checkpoint_sha256"]):
        raise ValueError("Input identity changed after planning")
    return entity


def run_count_task(plan, task, preexecuted=()):
    if not task["executions"]:
        return
    base = plan["bases"][task["base_id"]]
    check_base(base)
    labels, original, tau = original_inputs(base, task["optimizer"])
    masks = matched_masks(original, task["task_id"])
    slots, executions = deduplicate(original, masks)
    if slots != task["slots"] or executions != task["executions"]:
        raise ValueError("Fixed matching/reuse schedule changed")
    series = controls.prepare_series(Path(base["directory"]), controls.load_protocol().train_fraction)
    for reference in executions:
        directory = count_directory(base, task["optimizer"], reference)
        identity = {"task_id": task["task_id"], "sample_seed": reference["sample_seed"], "arm": reference["arm"]}
        if identity in preexecuted:
            meta = json.loads((directory/"result.json").read_text())
            if not meta["complete"] or not np.array_equal(np.load(directory/"update_mask.npy",allow_pickle=False),
                    masks[reference["sample_seed"],reference["arm"]].reshape(-1)):
                raise ValueError("Explicitly reused validation trajectory differs")
            print(f"REUSE validated formal count {identity}",flush=True)
            continue
        result = execute(base, controls.load_protocol(), base["checkpoint"], series, directory,
            task["optimizer"], reference["arm"], labels,
            fixed_mask=masks[reference["sample_seed"], reference["arm"]], expected_tau=tau)
        print(f"DONE count {task['task_id']} sample{reference['sample_seed']} {reference['arm']} seconds={result['wall_seconds']:.2f}", flush=True)


def run_window_task(plan, task, *, record_numerical_failures=False):
    base = plan["bases"][task["base_id"]]
    entity = check_base(base)
    protocol = window_protocol(task["length"])
    path = source_path(base, task["length"])
    if path.parent.exists():
        raise FileExistsError(f"Refusing to retrain or overwrite source: {path}")
    _, series = train_source(entity, base["seed"], protocol, controls.Workspace(WINDOW),
        RunOptions(device="cuda:0", num_workers=0, resume=False, force=False, log_test_loss_during_source_training=False))
    if not controls._checkpoint_complete(path, protocol, entity, base["seed"]):
        raise ValueError("New source checkpoint did not complete")
    n = task["common_points"]
    series = replace(series, test=series.test[:n], test_labels=series.test_labels[:n])
    expected = np.load(controls.reference(base,"frozen","shared")["label_path"], allow_pickle=False).reshape(-1)[:n]
    for optimizer, arm in window_runs(protocol):
        directory = window_directory(base,task["length"],optimizer,arm)
        try:
            result = execute(base, protocol, path, series, directory, optimizer, arm, expected)
        except FloatingPointError as exc:
            if not record_numerical_failures:
                raise
            # A failed trajectory stays invalid on the fixed entity support.
            # Never retry it, retain a finite prefix, or alter the optimizer.
            controls.atomic_json(directory/"failure.json", {"complete": False, "status": "invalid",
                "dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
                "optimizer": optimizer, "arm": arm, "window_length": task["length"],
                "source_checkpoint_sha256": controls.sha256_file(path), "data_manifest": base["data_manifest"],
                "metric_version": controls.VERSION, "error_type": type(exc).__name__, "error": str(exc),
                "finished_at": controls.now(), "expected_test_points": n})
            print(f"INVALID window {task['task_id']} length{task['length']} {optimizer} {arm}: {exc}", flush=True)
            continue
        print(f"DONE window {task['task_id']} length{task['length']} {optimizer} {arm} seconds={result['wall_seconds']:.2f}", flush=True)


def prepare_window_extension():
    if (WINDOW/"extension_plan.json").exists():
        raise FileExistsError("Window extension plan already exists")
    old = json.loads((HERE/"plan.json").read_text())
    if json.loads(json.dumps(controls.load_protocol().as_dict())) != old["protocol"]:
        raise ValueError("Original E17 protocol changed")
    support = pd.read_csv(controls.SUPPORT, keep_default_na=False)
    retained = {(r.dataset, "" if r.entity == "__root__" else r.entity)
                for r in support[support.support_status == "retained"].itertuples()}
    expected_bases = {(d, s, seed) for d,s in retained for seed in old["protocol"]["seeds"]}
    actual_bases = {(b["dataset"],b["subset"],b["seed"]) for b in old["bases"].values()}
    if (actual_bases != expected_bases or len(actual_bases) != 530
            or support[support.support_status == "retained"].groupby("dataset").size().to_dict() != controls.EXPECTED_ENTITIES):
        raise ValueError("Canonical retained entity/seed support changed")
    timing = pd.read_csv(WINDOW.parent/"selected_count_normalization/derived/physical.csv", keep_default_na=False)
    costs = timing.groupby(["dataset","subset","seed"]).wall_seconds.sum().to_dict()
    tasks, loads = [], [0.0]*4
    for base_id, base in sorted(old["bases"].items()):
        for length in LENGTHS:
            tasks.append({"task_id": len(tasks), "experiment": "window", "base_id": base_id,
                "length": length, "common_points": base["data_manifest"]["test_rows"]//48*48,
                "reuse_existing_e17": base["dataset"] in WINDOW_DATASETS,
                "estimated_seconds": costs[base["dataset"],base["subset"],base["seed"]]*12/length*1.5+5})
    for task in sorted((t for t in tasks if not t["reuse_existing_e17"]), key=lambda t: (-t["estimated_seconds"],t["task_id"])):
        shard = min(range(4), key=lambda i: loads[i])
        task["shard"] = shard
        loads[shard] += task["estimated_seconds"]
    original = pd.read_csv(WINDOW/"derived/cells.csv")
    if len(original) != 180 or set(original.dataset) != set(WINDOW_DATASETS):
        raise ValueError("Expected the original three-family E17 grid before extension")
    plan = {"created_at": controls.now(), "extension_of": str(HERE/"plan.json"),
        "protocol": old["protocol"], "metric_version": controls.VERSION, "bases": old["bases"],
        "window_datasets": list(controls.EXPECTED_ENTITIES), "tasks": tasks,
        "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4],
        "estimated_shard_hours": [v/3600 for v in loads],
        "original_e17_cells": original.to_dict(orient="records"),
        "original_e17_dataset_summary": pd.read_csv(WINDOW/"derived/dataset_summary.csv").to_dict(orient="records"),
        "expected": {"retained_entities": 106, "excluded_entities": len(support)-len(retained),
            "extension_new_sources": 1030, "extension_new_evaluations": 9270,
            "reused_e17_sources": 30, "reused_e17_evaluations": 270,
            "reused_length12_sources": 530, "reused_length12_evaluations": 4770,
            "logical_cells_per_length": 180, "logical_cells": 540,
            "physical_evaluations": 14310, "paired_entity_cells": 6360},
        "invalid_policy": "No entity exclusions or retries; undefined full-support cells are reported explicitly."}
    if sum(not t["reuse_existing_e17"] for t in tasks) != 1030:
        raise ValueError("Unexpected extension task inventory")
    directories = [window_directory(old["bases"][t["base_id"]],t["length"],o,a)
                   for t in tasks for o,a in window_runs(window_protocol(t["length"]))]
    if len(set(directories)) != 9540:
        raise ValueError("Window result path collision")
    for task in tasks:
        if not task["reuse_existing_e17"]:
            base = old["bases"][task["base_id"]]
            if source_path(base,task["length"]).parent.exists() or any(
                    window_directory(base,task["length"],o,a).exists() for o,a in window_runs(window_protocol(task["length"]))):
                raise FileExistsError(f"Unplanned existing extension artifact: {task}")
    controls.atomic_json(WINDOW/"extension_plan.json",plan)
    print(json.dumps({"expected": plan["expected"], "estimated_shard_hours": plan["estimated_shard_hours"]}),flush=True)


def window_extension_worker(plan, shard):
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Each worker must be bound to one CPU core")
    torch.cuda.set_device(0)
    tasks = sorted((t for t in plan["tasks"] if t.get("shard")==shard),key=lambda t:-t["estimated_seconds"])
    progress = {"status": "running", "shard": shard, "pid": os.getpid(), "completed_tasks": 0,
                "expected_tasks": len(tasks), "successful_evaluations": 0, "invalid_evaluations": 0}
    for task in tasks:
        progress.update(current_task=task,updated_at=controls.now())
        controls.atomic_json(WINDOW/"progress"/f"gpu{shard}.json",progress)
        print(f"START {task['task_id']} window {task['base_id']} length{task['length']}",flush=True)
        run_window_task(plan,task,record_numerical_failures=True)
        base = plan["bases"][task["base_id"]]
        for optimizer,arm in window_runs(window_protocol(task["length"])):
            directory = window_directory(base,task["length"],optimizer,arm)
            progress["successful_evaluations"] += int((directory/"result.json").is_file())
            progress["invalid_evaluations"] += int((directory/"failure.json").is_file())
        progress["completed_tasks"] += 1
        progress["updated_at"] = controls.now()
        controls.atomic_json(WINDOW/"progress"/f"gpu{shard}.json",progress)
    progress.update(status="complete",updated_at=controls.now(),current_task=None)
    controls.atomic_json(WINDOW/"progress"/f"gpu{shard}.json",progress)


def worker(plan, shard):
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Each worker must be bound to one CPU core")
    torch.cuda.set_device(0)
    validation = json.loads((HERE/"validation.json").read_text())
    for task in sorted((t for t in plan["tasks"] if t["shard"]==shard), key=lambda t: -t["estimated_seconds"]):
        if task["experiment"] == "count" and not task["executions"]:
            continue
        if task["task_id"] in validation["completed_window_task_ids"]:
            if task["experiment"] != "window":
                raise ValueError("Unexpected preflight reuse")
            base=plan["bases"][task["base_id"]]
            for optimizer,arm in window_runs(window_protocol(task["length"])):
                meta=json.loads((window_directory(base,task["length"],optimizer,arm)/"result.json").read_text())
                if not meta["complete"]:
                    raise ValueError("Preflight formal task is incomplete")
            print(f"REUSE validated formal window task {task['task_id']}", flush=True)
            continue
        print(f"START {task['task_id']} {task['experiment']} {task['base_id']}", flush=True)
        if task["experiment"] == "count":
            run_count_task(plan,task,validation["completed_count_runs"])
        else:
            run_window_task(plan,task)


def supervise(plan, *, window_extension=False):
    validation_path = WINDOW/"extension_validation.json" if window_extension else HERE/"validation.json"
    status_path = WINDOW/"extension_status.json" if window_extension else HERE/"status.json"
    validation=json.loads(validation_path.read_text())
    if validation["status"] != "passed":
        raise RuntimeError("Finite validation has not passed")
    if status_path.exists() or (not window_extension and ((COUNT/"derived").exists() or (WINDOW/"derived").exists())):
        raise FileExistsError("Batch status or derived output already exists")
    logs=WINDOW/"extension_logs" if window_extension else HERE/"logs"
    logs.mkdir(exist_ok=True)
    status={"status":"running","started_at":controls.now(),"pid":os.getpid(),"expected":plan["expected"],
            "completed_window_task_ids_at_launch":validation.get("completed_window_task_ids",[]),
            "completed_count_runs_at_launch":validation.get("completed_count_runs",[]),"devices":plan["devices"]}
    processes,handles=[],[]
    controls.atomic_json(status_path,status)
    try:
        for shard,gpu in enumerate(plan["devices"]):
            handle=(logs/f"gpu{gpu}.log").open("x")
            handles.append(handle)
            p=subprocess.Popen(["taskset","-c",str(plan["cpu_cores"][shard]),sys.executable,
                "-B",str(HERE/"experiment.py"),"window-worker" if window_extension else "worker","--shard",str(shard)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1"),
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            processes.append(p)
        status["workers"]=[{"gpu":g,"pid":p.pid,"cpu_core":c} for g,p,c in zip(plan["devices"],processes,plan["cpu_cores"])]
        controls.atomic_json(status_path,status)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(p.wait):p for p in processes}
            for job in as_completed(jobs):
                if job.result():
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError(f"Worker {jobs[job].pid} failed; inspect log")
        command = [sys.executable,"-B",str(HERE/"aggregate.py")]
        if window_extension:
            command.append("--window-only")
        subprocess.run(command,cwd=ROOT,check=True)
        final_status = json.loads((WINDOW/"derived/summary.json").read_text())["status"] if window_extension else "complete"
        status.update(status=final_status,finished_at=controls.now())
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        status.update(status="failed",finished_at=controls.now(),error=str(exc))
        raise
    finally:
        controls.atomic_json(status_path,status)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","validate","worker","supervise",
        "window-prepare","window-validate","window-worker","window-supervise"))
    parser.add_argument("--shard",type=int)
    args=parser.parse_args()
    controls.configure_threads()
    if args.mode=="prepare":
        prepare()
    elif args.mode=="window-prepare":
        prepare_window_extension()
    else:
        extension = args.mode.startswith("window-")
        plan=json.loads((WINDOW/"extension_plan.json" if extension else HERE/"plan.json").read_text())
        if json.loads(json.dumps(controls.load_protocol().as_dict()))!=plan["protocol"]:
            raise ValueError("Protocol changed since plan")
        if args.mode=="window-validate":
            from validate import validate_window_extension
            validate_window_extension(plan)
        elif args.mode=="validate":
            from validate import validate
            validate(plan)
        elif args.mode in ("worker","window-worker"):
            if args.shard not in range(4):
                raise ValueError("Invalid worker shard")
            (window_extension_worker if extension else worker)(plan,args.shard)
        else:
            supervise(plan,window_extension=extension)
