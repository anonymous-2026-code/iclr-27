#!/usr/bin/env python3
"""Read-only source-validation inference for causal calibration/Frozen audit."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_controls"))
import run as controls
import numpy as np
import pandas as pd
import torch


def score_directory(base):
    return HERE / "source_scores" / base["dataset"] / (base["subset"] or "__root__") / f"seed{base['seed']}"


def worker(plan, shard):
    torch.cuda.set_device(0)
    protocol = controls.load_protocol()
    device = torch.device("cuda:0")
    for task in plan["tasks"]:
        if task["shard"] != shard:
            continue
        base = plan["bases"][task["base_id"]]
        out = score_directory(base)
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        entity = controls.Entity(base["dataset"], base["subset"], Path(base["directory"]))
        if (controls.entity_manifest(entity) != base["data_manifest"] or
                controls.sha256_file(Path(base["checkpoint"])) != base["source_checkpoint_sha256"]):
            raise ValueError(f"Source/data identity changed: {task['base_id']}")
        series = controls.prepare_series(entity.directory, protocol.train_fraction)
        controls.set_seed(base["seed"])
        model = controls._load_checkpoint(Path(base["checkpoint"]), series.train.shape[1], protocol, device)
        _, loader, _ = controls.build_loaders(series, protocol, protocol.frozen_test_batch_size, 0)
        scores, _ = controls._score_loader(model, loader, device)
        if (not np.isfinite(scores).all() or len(scores) != len(series.validation) // 12 * 12
                or not len(scores) or (scores < 0).any()):
            raise ValueError("Invalid source-validation scores/prefix")
        tau = float(np.percentile(scores, 95))
        errors = []
        for key, reference in base["references"].items():
            if key == "shared:frozen":
                continue
            old_tau = json.loads(Path(reference["manifest_path"]).read_text())["result_metadata"]["selector_tau"]
            errors.append(abs(tau - old_tau))
            if abs(tau - old_tau) > 1e-10:
                raise ValueError(f"Source P95 differs from original run: {task['base_id']} / {key}")
        np.save(out / "scores.npy", scores)
        result = {"complete": True, "dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
            "source_checkpoint_sha256": base["source_checkpoint_sha256"], "points": len(scores),
            "source_validation_mean": float(scores.mean(dtype=np.float64)),
            "source_validation_std": float(scores.std(dtype=np.float64)),
            "source_validation_median": float(np.median(scores)), "p95": tau,
            "p99": float(np.percentile(scores, 99)), "max_p95_reference_error": max(errors),
            "wall_seconds": time.perf_counter() - started, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "score_path": str(out / "scores.npy"), "new_source_training": False}
        controls.atomic_json(out / "result.json", result)
        print(f"DONE {task['base_id']} points={len(scores)} p95_error={max(errors):.3g} seconds={result['wall_seconds']:.2f}", flush=True)


def supervise():
    if (HERE / "source_status.json").exists() or (HERE / "source_scores").exists():
        raise FileExistsError("Source scoring already exists; no automatic overwrite")
    old = json.loads((HERE.parent / "gpu_controls/plan.json").read_text())
    tasks = [{"base_id": key, "cost": base["data_manifest"]["train_rows"] * base["data_manifest"]["features"]}
             for key, base in old["bases"].items()]
    tasks.sort(key=lambda t: (-t["cost"], t["base_id"]))
    loads = [0] * 4
    for task in tasks:
        shard = min(range(4), key=lambda s: loads[s])
        task["shard"] = shard
        loads[shard] += task["cost"]
    plan = {"created_at": controls.now(), "bases": old["bases"], "tasks": tasks,
            "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4], "expected": 530}
    controls.atomic_json(HERE / "source_plan.json", plan)
    logs = HERE / "source_logs"
    logs.mkdir(exist_ok=True)
    status = {"status": "running", "started_at": controls.now(), "pid": os.getpid(), "expected": 530}
    processes, handles = [], []
    controls.atomic_json(HERE / "source_status.json", status)
    try:
        for shard in range(4):
            handle = (logs / f"gpu{shard}.log").open("x")
            handles.append(handle)
            proc = subprocess.Popen(["taskset", "-c", str(plan["cpu_cores"][shard]), sys.executable,
                str(HERE / "source_scores.py"), "worker", "--shard", str(shard)], cwd=controls.ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan["devices"][shard]), PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(proc)
        status["workers"] = [{"gpu": gpu, "pid": p.pid} for gpu, p in enumerate(processes)]
        controls.atomic_json(HERE / "source_status.json", status)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(p.wait): p for p in processes}
            for future in as_completed(jobs):
                if future.result():
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError(f"Source scoring worker failed: {jobs[future].pid}")
        expected = {score_directory(b) / "result.json" for b in plan["bases"].values()}
        if set((HERE / "source_scores").glob("*/*/seed*/result.json")) != expected:
            raise ValueError("Incomplete source-validation output grid")
        records = [json.loads(p.read_text()) for p in sorted(expected)]
        if len(records) != 530 or not all(r["complete"] for r in records):
            raise ValueError("Expected 530 complete source scores")
        pd.DataFrame(records).to_csv(HERE / "source_validation_summary.csv", index=False)
        status.update(status="complete", finished_at=controls.now(), completed=530)
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        status.update(status="failed", finished_at=controls.now(), error=str(exc))
        raise
    finally:
        controls.atomic_json(HERE / "source_status.json", status)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    controls.configure_threads()
    if args.mode == "run":
        supervise()
    else:
        if args.shard not in range(4):
            raise ValueError("Invalid shard")
        worker(json.loads((HERE / "source_plan.json").read_text()), args.shard)
