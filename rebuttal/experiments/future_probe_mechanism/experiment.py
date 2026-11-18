#!/usr/bin/env python3
"""Unseen-future diagnostic probes along unchanged original Oracle trajectories."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
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
DATASETS = ("Daphnet", "GECCO", "PUMP")


def task_directory(task, root=None):
    base = Path(root) if root is not None else HERE / "runs"
    return base / task["base"]["dataset"] / task["optimizer"] / f"seed{task['base']['seed']}"


def future_indices(labels, window, maximum=64):
    first = (window + 1) * 12
    groups = {}
    for name, value in (("normal", 0), ("anomaly", 1)):
        indices = np.flatnonzero(labels[first:] == value) + first
        if len(indices):
            indices = indices[np.linspace(0, len(indices) - 1, min(maximum, len(indices)), dtype=int)]
        if (indices < first).any() or len(np.unique(indices)) != len(indices):
            raise ValueError("Future probe indexing violated temporal independence")
        groups[name] = indices
    return groups


def pair_fraction(normal, anomaly):
    if not len(normal) or not len(anomaly):
        return None
    return float(np.searchsorted(np.sort(normal), anomaly, side="left").mean() / len(normal))


def probe(model, test_windows, groups, gradients):
    parameters = list(model.parameters())
    points = np.concatenate(list(groups.values()))
    if not len(points):
        return {k: np.empty(0) for k in groups}, {k: None for k in groups}
    window_ids = np.unique(points // 12)
    x = torch.as_tensor(test_windows[window_ids], device="cuda:0", dtype=torch.float32)
    model.eval()
    with torch.set_grad_enabled(gradients):
        loss = controls._point_loss(model, x, stage="test")
        controls.finite_tensor("probe scores", loss, -1)
        class_scores = {}
        for name, indices in groups.items():
            if len(indices):
                rows = torch.as_tensor(np.searchsorted(window_ids, indices // 12), device=x.device)
                cols = torch.as_tensor(indices % 12, device=x.device)
                class_scores[name] = loss[rows, cols]
        grad = {name: None for name in groups}
        if gradients:
            nonempty = list(class_scores)
            for i, name in enumerate(nonempty):
                grad[name] = torch.autograd.grad(class_scores[name].mean(), parameters,
                    retain_graph=i < len(nonempty) - 1, create_graph=False)
        values = {name: class_scores[name].detach().cpu().numpy() if name in class_scores else np.empty(0)
                  for name in groups}
    if gradients and any(p.grad is not None for p in parameters):
        raise ValueError("Diagnostic gradients contaminated optimizer .grad")
    return values, grad


def fixed_summary(values):
    """Retrospective fixed-probe statistics; never used to select updates."""
    result = {}
    for name, scores in values.items():
        result[f"{name}_mean"] = float(scores.mean(dtype=np.float64))
        for label, q in (("q10", .1), ("q50", .5), ("q90", .9)):
            result[f"{name}_{label}"] = float(np.quantile(scores, q))
    result["margin"] = result["anomaly_mean"] - result["normal_mean"]
    result["pair_fraction"] = pair_fraction(values["normal"], values["anomaly"])
    labels = np.concatenate([np.full(len(values[k]), v) for k, v in (("normal", 0), ("anomaly", 1))])
    result["auprc"] = controls.metric(labels, np.concatenate([values["normal"], values["anomaly"]]))
    return result


def run_one(task, root=None, fixed_diagnostics=None, *, save_probe_vectors=False,
            sampling_sensitivity=False):
    base = task["base"]
    protocol = controls.load_protocol()
    config = protocol.tta_by_key(task["optimizer"])
    out = task_directory(task, root)
    out.mkdir(parents=True, exist_ok=False)
    entity = controls.Entity(base["dataset"], base["subset"], Path(base["directory"]))
    if (controls.entity_manifest(entity) != base["data_manifest"] or
            controls.sha256_file(Path(base["checkpoint"])) != base["source_checkpoint_sha256"]):
        raise ValueError("Source model/data changed")
    series = controls.prepare_series(entity.directory, protocol.train_fraction)
    count = len(series.test) // 12
    truth = series.test_labels[:count * 12].astype(np.int8)
    test_windows = series.test[:count * 12].reshape(count, 12, -1)
    checkpoints = {j * count // 20 for j in range(20)}
    if len(checkpoints) != 20:
        raise ValueError("Not enough windows for the prespecified 20 checkpoints")
    controls.set_seed(base["seed"])
    model = controls._load_checkpoint(Path(base["checkpoint"]), series.train.shape[1], protocol, torch.device("cuda:0"))
    model._tta_scope = "full_model"
    optimizer = controls._build_tta_optimizer(model, config)
    params = list(model.parameters())
    scores, records = [], []
    vector_records = []
    fixed_records, fixed_scores = [], []
    if fixed_diagnostics is not None:
        fixed_groups = {k: np.asarray(v, dtype=int) for k, v in fixed_diagnostics["groups"].items()}
        cumulative_steps = set(fixed_diagnostics["checkpoint_steps"])
        for name, value in (("normal", 0), ("anomaly", 1)):
            indices = fixed_groups[name]
            if not len(indices) or not np.all(truth[indices] == value):
                raise ValueError("Fixed diagnostic labels/indices differ")

        def record_checkpoint(step):
            values, _ = probe(model, test_windows, fixed_groups, gradients=False)
            summary = fixed_summary(values)
            fixed_records.append({"seed": base["seed"], "optimizer": config.key,
                                  "steps_completed": step, **summary})
            for name, indices in fixed_groups.items():
                fixed_scores.extend({"steps_completed": step, "class": name,
                                     "point_index": int(i), "score": float(s)}
                                    for i, s in zip(indices, values[name]))

        record_checkpoint(0)
    started = time.perf_counter()
    for t in range(count):
        x = torch.as_tensor(test_windows[t:t + 1], dtype=torch.float32, device="cuda:0")
        mask = torch.as_tensor(truth[t * 12:(t + 1) * 12] == 0, device=x.device).reshape(1, 12)
        model.eval()
        with torch.no_grad():
            score_before = controls._point_loss(model, x, stage="test")
        controls.finite_tensor("current pre-update score", score_before, t)
        scores.append(score_before.detach().cpu().numpy())
        optimizer.zero_grad()
        if t in checkpoints:
            groups = future_indices(truth, t)
            before, grads = probe(model, test_windows, groups, gradients=True)
            if sampling_sensitivity:
                # One alternative, fixed by time alone: midpoints of equal-count
                # class strata instead of the primary endpoint-spaced ranks.
                alternative_groups = {}
                for name, value in (("normal", 0), ("anomaly", 1)):
                    eligible = np.flatnonzero(truth[(t + 1) * 12:] == value) + (t + 1) * 12
                    k = min(64, len(eligible))
                    ranks = np.floor((np.arange(k) + .5) * len(eligible) / k).astype(int) if k else np.empty(0, dtype=int)
                    alternative_groups[name] = eligible[ranks]
                alternative_before, _ = probe(model, test_windows, alternative_groups, gradients=False)
            if fixed_diagnostics is not None:
                fixed_before, _ = probe(model, test_windows, fixed_groups, gradients=False)
            previous = [p.detach().clone() for p in params]
        model.train()
        loss_points = controls._point_loss(model, x, stage="tta")
        loss = (loss_points * mask.float()).mean()
        controls.finite_tensor("Oracle full-window loss", loss, t)
        loss.backward()
        optimizer.step()
        if fixed_diagnostics is not None and t + 1 in cumulative_steps:
            record_checkpoint(t + 1)
        if t not in checkpoints:
            continue
        with torch.no_grad():
            delta = [p.detach() - old for p, old in zip(params, previous)]
            step_l2 = float(torch.stack([d.square().sum() for d in delta]).sum().sqrt().item())
            model.eval()
            current_after = controls._point_loss(model, x, stage="test")
        after, _ = probe(model, test_windows, groups, gradients=False)
        if save_probe_vectors:
            def vector_record(design, membership, pre, post):
                return {"window": t, "design": design,
                        **{name + "_indices": membership[name].tolist() for name in membership},
                        **{name + "_labels": [value] * len(membership[name])
                           for name, value in (("normal", 0), ("anomaly", 1))},
                        **{name + "_before": pre[name].tolist() for name in membership},
                        **{name + "_after": post[name].tolist() for name in membership}}
            vector_records.append(vector_record("primary", groups, before, after))
            if sampling_sensitivity:
                alternative_after, _ = probe(model, test_windows, alternative_groups, gradients=False)
                vector_records.append(vector_record("midpoint_sensitivity", alternative_groups,
                                                    alternative_before, alternative_after))
        selected = int(mask.sum().item())
        current_change = float((current_after - score_before)[mask].mean().item()) if selected else None
        record = {"dataset": base["dataset"], "seed": base["seed"], "optimizer": config.key,
            "window": t, "windows_total": count, "selected_points": selected,
            "masked_loss": float(loss.item()), "parameter_step_l2": step_l2,
            "current_selected_normal_delta": current_change,
            "pair_fraction_before": pair_fraction(before["normal"], before["anomaly"]),
            "pair_fraction_after": pair_fraction(after["normal"], after["anomaly"])}
        for name in groups:
            n = len(groups[name])
            record[f"{name}_points"] = n
            record[f"{name}_indices"] = json.dumps(groups[name].tolist())
            record[f"{name}_before"] = float(before[name].mean(dtype=np.float64)) if n else None
            record[f"{name}_after"] = float(after[name].mean(dtype=np.float64)) if n else None
            actual = float((after[name].astype(np.float64) - before[name]).mean()) if n else None
            predicted = float(torch.stack([(g * d).sum() for g, d in zip(grads[name], delta)]).sum().item()) if n else None
            record[f"{name}_delta"] = actual
            record[f"{name}_predicted_delta"] = predicted
            record[f"{name}_linearization_error"] = actual - predicted if n else None
        paired = bool(len(groups["normal"]) and len(groups["anomaly"]))
        record["paired_probe_defined"] = paired
        record["margin_delta"] = record["anomaly_delta"] - record["normal_delta"] if paired else None
        record["predicted_margin_delta"] = record["anomaly_predicted_delta"] - record["normal_predicted_delta"] if paired else None
        record["pair_fraction_delta"] = record["pair_fraction_after"] - record["pair_fraction_before"] if paired else None
        if fixed_diagnostics is not None:
            record["current_normal_before"] = float(score_before[mask].mean().item()) if selected else None
            record["current_normal_after"] = float(current_after[mask].mean().item()) if selected else None
            record["masked_loss_after"] = float((current_after * mask.float()).mean().item())
            record["masked_loss_delta"] = record["masked_loss_after"] - record["masked_loss"]
            fixed_after, _ = probe(model, test_windows, fixed_groups, gradients=False)
            fb, fa = fixed_summary(fixed_before), fixed_summary(fixed_after)
            for key in ("normal_mean", "anomaly_mean", "margin", "pair_fraction", "auprc"):
                record[f"fixed_{key}_delta"] = fa[key] - fb[key]
        for k, v in record.items():
            if isinstance(v, float) and not np.isfinite(v):
                raise FloatingPointError(f"Nonfinite probe diagnostic: {k}")
        records.append(record)
    for p in params:
        controls.finite_tensor("final model parameter", p.detach(), count)
    score = np.concatenate(scores).reshape(-1)
    ref = controls.reference(base, "oracle", config.key)
    reference = np.load(ref["score_path"], allow_pickle=False).reshape(-1)
    labels = np.load(ref["label_path"], allow_pickle=False).reshape(-1)
    np.testing.assert_array_equal(truth, labels)
    np.testing.assert_allclose(score, reference, rtol=1e-5, atol=1e-6)
    value = controls.metric(truth, score)
    metric_error = abs(value - ref["auprc"])
    if metric_error > 1e-8 or len(records) != 20:
        raise ValueError("Original Oracle metric parity or checkpoint count failed")
    np.save(out / "test_energy.npy", score)
    np.save(out / "true.npy", truth)
    pd.DataFrame(records).to_csv(out / "probes.csv", index=False)
    if save_probe_vectors:
        (out / "probe_vectors.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in vector_records))
    if fixed_diagnostics is not None:
        if [r["steps_completed"] for r in fixed_records] != sorted(cumulative_steps):
            raise ValueError("Cumulative checkpoints differ from the prespecified schedule")
        pd.DataFrame(fixed_records).to_csv(out / "cumulative.csv", index=False)
        pd.DataFrame(fixed_scores).to_csv(out / "fixed_scores.csv", index=False)
    result = {"complete": True, "dataset": base["dataset"], "seed": base["seed"], "optimizer": config.key,
        "auprc": value, "max_score_parity_error": float(np.max(np.abs(score - reference))),
        "auprc_parity_error": metric_error, "observed_windows": 20, "optimizer_steps": count,
        "paired_probe_windows": sum(r["paired_probe_defined"] for r in records),
        "source_checkpoint_sha256": base["source_checkpoint_sha256"], "wall_seconds": time.perf_counter() - started,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0))}
    if fixed_diagnostics is not None:
        result.update(subset=base["subset"], cumulative_checkpoints=len(fixed_records),
                      frozen_auprc=controls.reference(base, "frozen", "shared")["auprc"])
        result["online_delta_auprc"] = value - result["frozen_auprc"]
    controls.atomic_json(out / "result.json", result)
    return result


def make_plan():
    old = json.loads((HERE.parent / "gpu_controls/plan.json").read_text())
    tasks = [{"base": base, "optimizer": c["key"]} for base in old["bases"].values()
             if base["dataset"] in DATASETS for c in old["protocol"]["tta_configs"]]
    if len(tasks) != 60:
        raise ValueError("Expected 3 datasets x 5 seeds x 4 optimizers")
    # Reuse measured original-scale per-run cost for deterministic balanced assignment.
    measured = pd.read_csv(HERE.parent / "selected_count_normalization/derived/physical.csv")
    costs = measured[(measured.arm == "oracle") & measured.dataset.isin(DATASETS)].set_index(["dataset", "seed", "optimizer"]).wall_seconds.to_dict()
    tasks.sort(key=lambda t: (-costs[t["base"]["dataset"], t["base"]["seed"], t["optimizer"]], t["base"]["dataset"], t["base"]["seed"], t["optimizer"]))
    loads = [0.0] * 4
    for i, t in enumerate(tasks):
        shard = min(range(4), key=lambda s: loads[s])
        t.update(task_id=i, shard=shard)
        loads[shard] += costs[t["base"]["dataset"], t["base"]["seed"], t["optimizer"]]
    return {"created_at": controls.now(), "tasks": tasks, "expected_runs": 60,
            "devices": [0, 1, 2, 3], "cpu_cores": sorted(os.sched_getaffinity(0))[:4], "prior_shard_seconds": loads}


def validate(plan):
    toy = np.array([0] * 12 + [0, 1] * 30, dtype=np.int8)
    for t in (0, 2, 4):
        for points in future_indices(toy, t).values():
            assert (points >= (t + 1) * 12).all()
    assert pair_fraction(np.array([0., 1.]), np.array([1., 2.])) == .75
    assert pair_fraction(np.empty(0), np.array([1.])) is None
    selected = [t for t in plan["tasks"] if t["base"]["dataset"] == "Daphnet" and t["base"]["seed"] == 42
                and t["optimizer"] in ("sgd_ttlr0.001", "adamw_ttlr0.0001")]
    if len(selected) != 2:
        raise ValueError("Validation pair differs")
    outputs = []
    for task in selected:
        result = run_one(task, HERE / "validation")
        outputs.append(result)
        print(json.dumps(result), flush=True)
    controls.atomic_json(HERE / "validation.json", {"status": "passed", "finished_at": controls.now(),
        "future_index_and_strict_pair_toy": "passed", "probe_grad_no_training_accumulation": "passed", "parity": outputs})


def aggregate(plan):
    paths = {task_directory(t) / "result.json": t for t in plan["tasks"]}
    if set((HERE / "runs").glob("*/*/seed*/result.json")) != set(paths):
        raise ValueError("Incomplete/unexpected formal probe run grid")
    runs, probes = [], []
    for path, task in paths.items():
        r = json.loads(path.read_text())
        if (not r["complete"] or r["dataset"] != task["base"]["dataset"] or r["seed"] != task["base"]["seed"]
                or r["optimizer"] != task["optimizer"] or r["auprc_parity_error"] > 1e-8):
            raise ValueError("Probe result identity/parity failure")
        f = pd.read_csv(path.parent / "probes.csv")
        if len(f) != 20 or len(f.window.unique()) != 20:
            raise ValueError("Expected exactly 20 prescribed probe checkpoints")
        runs.append(r)
        probes.append(f)
    run_frame, observations = pd.DataFrame(runs), pd.concat(probes, ignore_index=True)
    summaries = []
    for key, g in observations.groupby(["dataset", "optimizer", "seed"]):
        valid = g[g.paired_probe_defined]
        active = g[g.selected_points > 0]
        summaries.append({"dataset": key[0], "optimizer": key[1], "seed": key[2],
            "checkpoints": len(g), "two_class_probe_checkpoints": len(valid), "normal_update_checkpoints": len(active),
            "normal_probe_checkpoints": int((g.normal_points > 0).sum()),
            "anomaly_probe_checkpoints": int((g.anomaly_points > 0).sum()),
            "current_normal_improves_fraction": float((active.current_selected_normal_delta < 0).mean()) if len(active) else np.nan,
            "future_normal_decreases_fraction": float((g.loc[g.normal_points > 0, "normal_delta"] < 0).mean()),
            "future_anomaly_decreases_fraction": float((g.loc[g.anomaly_points > 0, "anomaly_delta"] < 0).mean()) if (g.anomaly_points > 0).any() else np.nan,
            "margin_decreases_fraction": float((valid.margin_delta < 0).mean()) if len(valid) else np.nan,
            "pair_order_worsens_fraction": float((valid.pair_fraction_delta < 0).mean()) if len(valid) else np.nan,
            "normal_delta_median": g.normal_delta.median(), "anomaly_delta_median": g.anomaly_delta.median(),
            "margin_delta_median": valid.margin_delta.median(),
            "normal_linearization_abs_error_median": g.normal_linearization_error.abs().median(),
            "anomaly_linearization_abs_error_median": g.anomaly_linearization_error.abs().median()})
    per_run = pd.DataFrame(summaries)
    dataset_optimizer = per_run.groupby(["dataset", "optimizer"]).median(numeric_only=True).drop(columns="seed").reset_index()
    out = HERE / "derived"
    out.mkdir(exist_ok=False)
    for name, f in (("runs", run_frame), ("probes", observations), ("per_run", per_run), ("dataset_optimizer", dataset_optimizer)):
        f.to_csv(out / f"{name}.csv", index=False)
    lines = ["# One-step responses on unseen future samples: complete diagnostics", "", "All 60 original point-Oracle trajectories are replayed at 20 predefined times per stream. All complete detection scores pass parity against the original references.",
        "The same unseen future points are used before and after each step; probe pools differ across observation times. Future labels are used only for offline grouping and never enter updates.", "",
        "Each fraction below is first calculated over valid observation times within a run, then summarized by the median over five seeds. Observation times are not treated as independent experiments.", "",
        "| Dataset | Optimizer | Current normal-fit improvement fraction | Future anomaly-score decrease fraction | Future margin decrease fraction | Future ordering deterioration fraction |",
        "|---|---|---:|---:|---:|---:|"]
    for r in dataset_optimizer.itertuples():
        lines.append(f"| {r.dataset} | {r.optimizer} | {r.current_normal_improves_fraction:.1%} | {r.future_anomaly_decreases_fraction:.1%} | {r.margin_decreases_fraction:.1%} | {r.pair_order_worsens_fraction:.1%} |")
    lines += ["", "Negative-sign fractions include tiny numerical changes. Interpret them alongside continuous magnitudes, floating-point precision, and first-order residuals in probes.csv. Parameter-displacement inner products predict responses to the same single step; they do not establish a causal decomposition of full-stream AUPRC.",
        "Probe groups contain at most 64 points per class, selected at evenly spaced times. Their class balance does not represent the full test distribution. Missing classes remain undefined. A one-step anomaly-score decrease or margin compression does not automatically imply representation collapse.", "",
        "[Step records and indices](probes.csv), [valid support per run](per_run.csv), [optimizer summaries](dataset_optimizer.csv).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    controls.atomic_json(out / "summary.json", {"status": "complete", "finished_at": controls.now(), "runs": 60,
        "observation_windows": len(observations), "two_class_windows": int(observations.paired_probe_defined.sum()),
        "max_score_parity_error": float(run_frame.max_score_parity_error.max()),
        "max_auprc_parity_error": float(run_frame.auprc_parity_error.max()),
        "summed_worker_run_hours": float(run_frame.wall_seconds.sum() / 3600)})


def supervise(plan):
    if json.loads((HERE / "validation.json").read_text())["status"] != "passed":
        raise ValueError("Validation has not passed")
    if (HERE / "runs").exists() or (HERE / "status.json").exists():
        raise FileExistsError("Formal probe batch already exists")
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    processes, handles = [], []
    status = {"status": "running", "started_at": controls.now(), "pid": os.getpid(), "expected_runs": 60}
    controls.atomic_json(HERE / "status.json", status)
    try:
        for shard in range(4):
            handle = (logs / f"gpu{shard}.log").open("x")
            handles.append(handle)
            p = subprocess.Popen(["taskset", "-c", str(plan["cpu_cores"][shard]), sys.executable,
                str(HERE / "experiment.py"), "worker", "--shard", str(shard)], cwd=controls.ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan["devices"][shard]), PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(p)
        status["workers"] = [{"gpu": gpu, "pid": p.pid} for gpu, p in enumerate(processes)]
        controls.atomic_json(HERE / "status.json", status)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(p.wait): p for p in processes}
            for job in as_completed(jobs):
                if job.result():
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    raise RuntimeError(f"Probe worker {jobs[job].pid} failed")
        aggregate(plan)
        status.update(status="complete", finished_at=controls.now())
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        status.update(status="failed", finished_at=controls.now(), error=str(exc))
        raise
    finally:
        controls.atomic_json(HERE / "status.json", status)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "run", "worker"))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    controls.configure_threads()
    if args.mode == "prepare":
        if (HERE / "plan.json").exists():
            raise FileExistsError("Probe plan already exists")
        controls.atomic_json(HERE / "plan.json", make_plan())
    else:
        plan = json.loads((HERE / "plan.json").read_text())
        if args.mode in ("validate", "worker"):
            torch.cuda.set_device(0)
        if args.mode == "validate":
            validate(plan)
        elif args.mode == "run":
            supervise(plan)
        else:
            if args.shard not in range(4):
                raise ValueError("Invalid shard")
            for task in plan["tasks"]:
                if task["shard"] == args.shard:
                    print(f"START {task['task_id']} {task['base']['dataset']} seed{task['base']['seed']} {task['optimizer']}", flush=True)
                    result = run_one(task)
                    print(f"DONE {task['task_id']} max_score_error={result['max_score_parity_error']:.3g} seconds={result['wall_seconds']:.2f}", flush=True)
