#!/usr/bin/env python3
"""Prespecified input controls; reuse all submitted source/F/P/O artifacts."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "gpu_controls"))
import run as controls
import numpy as np
import pandas as pd
import torch

OUTPUT = HERE.parent / "oracle_input_controls"
CONDITIONS = ("sanitized_source_mean", "clean_skip_step")
OPTIMIZERS = tuple(c.key for c in controls.load_protocol().tta_configs)
KEYS = ["dataset", "subset", "seed", "optimizer"]
CATEGORIES = ("BI", "OOG", "NI", "POG")
GRAD_COLUMNS = ["window", "normal_points", "original_loss", "sanitized_loss",
                "original_gradient_l2", "sanitized_gradient_l2", "difference_l2",
                "cosine", "relative_difference", "normal_reconstruction_difference_rms"]


def category(p, o):
    if not np.isfinite([p, o]).all():
        return "undefined"
    return "BI" if p > .005 and o > .005 else "POG" if p > .005 else "OOG" if o > .005 else "NI"


def directory(base, condition, optimizer):
    return OUTPUT / "runs" / condition / optimizer / base["dataset"] / (base["subset"] or "__root__") / f"seed{base['seed']}"


def prepare():
    if (HERE / "plan.json").exists():
        raise FileExistsError("Prespecified plan already exists")
    available_cores = sorted(os.sched_getaffinity(0))
    if len(available_cores) < 6:
        raise ValueError("Input-control launcher requires six available CPU cores")
    supervisor_core, analysis_core = available_cores[:2]
    worker_cores = available_cores[2:6]
    os.sched_setaffinity(0, {supervisor_core})
    source = HERE.parent / "causal_score_calibration"
    if json.loads((source / "source_status.json").read_text())["status"] != "complete":
        raise ValueError("Existing source-validation scoring is incomplete")
    old = json.loads((source / "source_plan.json").read_text())
    bases = old["bases"]
    if len(bases) != 530:
        raise ValueError("Expected 530 canonical bases")
    entities = pd.DataFrame([b for b in bases.values()])[['dataset', 'subset']].drop_duplicates()
    if entities.groupby('dataset').size().to_dict() != controls.EXPECTED_ENTITIES:
        raise ValueError("Canonical 106-entity support differs")
    timing = pd.read_csv(HERE.parent / "ae_all_normal_window/derived/physical.csv", keep_default_na=False)
    seconds = timing.groupby(["dataset", "subset", "seed"]).wall_seconds.sum().to_dict()
    tasks, loads = [], [0.] * 4
    for base_id, base in bases.items():
        for ref in base["references"].values():
            for field in ("score_path", "label_path", "manifest_path"):
                if not Path(ref[field]).is_file():
                    raise FileNotFoundError(ref[field])
        tasks.append({"base_id": base_id, "estimated_seconds": 2.4 * seconds[base['dataset'], base['subset'], base['seed']]})
    tasks.sort(key=lambda t: (-t['estimated_seconds'], t['base_id']))
    for task in tasks:
        shard = min(range(4), key=lambda i: loads[i])
        task['shard'] = shard
        loads[shard] += task['estimated_seconds']
    plan = {"created_at": controls.now(), "bases": bases, "tasks": tasks,
            "conditions": list(CONDITIONS), "optimizers": list(OPTIMIZERS),
            "protocol": controls.load_protocol().as_dict(), "metric_version": controls.VERSION,
            "devices": [0, 1, 2, 3], "cpu_cores": worker_cores,
            "analysis_core": analysis_core, "supervisor_core": supervisor_core,
            "expected_gpu_runs": 4240, "gpu_cells_per_condition": 180,
            "cpu_analyses": ["anomaly_ratio_association", "temporal_score_drift", "affiliation_f"],
            "estimated_shard_hours": [v / 3600 for v in loads], "python": sys.executable}
    controls.atomic_json(HERE / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ('expected_gpu_runs', 'estimated_shard_hours', 'cpu_analyses')}), flush=True)


def source_reference(series):
    value = series.train.mean(axis=0, dtype=np.float64).astype(np.float32)
    if value.shape != (series.train.shape[1],) or not np.isfinite(value).all():
        raise ValueError("Invalid source-training reference")
    return value


def sanitized_input(x, normal, reference):
    if x.shape[:2] != normal.shape or reference.shape != (x.shape[-1],):
        raise ValueError("Input/mask/reference dimensions differ")
    return torch.where(normal[..., None], x, reference.reshape(1, 1, -1))


def reconstruction_loss(model, x_input, x_target, mask):
    reconstruction = model(x_input)
    points = (reconstruction - x_target).square().mean(dim=-1)
    return (points * mask.to(points.dtype)).mean(), reconstruction


def gradient_diagnostic(model, x, normal, sanitized_loss, sanitized_reconstruction, window):
    params = list(model.parameters())
    original_loss, original_reconstruction = reconstruction_loss(model, x, x, normal)
    original = torch.autograd.grad(original_loss, params)
    changed = torch.autograd.grad(sanitized_loss, params, retain_graph=True)
    dot = torch.stack([(a * b).sum() for a, b in zip(original, changed)]).sum()
    n1 = torch.stack([a.square().sum() for a in original]).sum().sqrt()
    n2 = torch.stack([a.square().sum() for a in changed]).sum().sqrt()
    difference = torch.stack([(a-b).square().sum() for a, b in zip(original, changed)]).sum().sqrt()
    reconstruction_difference = (original_reconstruction[normal] - sanitized_reconstruction[normal]).square().mean().sqrt()
    values = torch.stack([original_loss, sanitized_loss, dot, n1, n2, difference, reconstruction_difference])
    controls.finite_tensor("same-state gradient diagnostic", values, window)
    return dict(zip(GRAD_COLUMNS, [window, int(normal.sum()), float(original_loss), float(sanitized_loss),
        float(n1), float(n2), float(difference), float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else np.nan,
        float(difference / n1) if n1 > 0 else np.nan, float(reconstruction_difference)]))


def run_one(base, optimizer_key, condition, series, *, out=None, diagnostics=True):
    if condition not in (*CONDITIONS, 'original_point'):
        raise ValueError(condition)
    target = Path(out) if out is not None else directory(base, condition, optimizer_key)
    target.mkdir(parents=True, exist_ok=False)
    protocol = controls.load_protocol()
    device = torch.device('cuda:0')
    controls.set_seed(base['seed'])
    model = controls._load_checkpoint(Path(base['checkpoint']), series.train.shape[1], protocol, device)
    model._tta_scope = 'full_model'
    optimizer = controls._build_tta_optimizer(model, protocol.tta_by_key(optimizer_key))
    _, _, loader = controls.build_loaders(series, protocol, 1, 0)
    reference = source_reference(series)
    fixed = torch.as_tensor(reference, device=device)
    labels_expected = np.load(controls.reference(base, 'oracle', optimizer_key)['label_path'], allow_pickle=False).reshape(-1).astype(np.int8)
    if not np.array_equal(series.test_labels[:len(labels_expected)], labels_expected) or len(labels_expected) != len(loader)*12:
        raise ValueError('Original label/prefix support differs')
    counts = (labels_expected.reshape(-1, 12) == 0).sum(1)
    mixed = np.flatnonzero((counts > 0) & (counts < 12))
    probe_windows = set(mixed[np.linspace(0, len(mixed)-1, min(20, len(mixed)), dtype=int)].tolist()) if len(mixed) else set()
    if condition != 'sanitized_source_mean' or not diagnostics:
        probe_windows = set()
    params = list(model.parameters())
    scores, masks, traces, probes = [], [], [], []
    started, cpu_start = time.perf_counter(), time.process_time()
    for window, (batch_x, batch_y) in enumerate(loader):
        x = batch_x.float().to(device)
        normal = batch_y.to(device) == 0
        model.eval()
        with torch.no_grad():
            score = controls._point_loss(model, x, stage='test')
        controls.finite_tensor('original-input pre-update score', score, window)
        model.train()
        optimizer.zero_grad()
        update = condition != 'clean_skip_step' or bool(normal.all())
        mask = normal if condition != 'clean_skip_step' else torch.full_like(normal, update)
        replaced = int((~normal).sum()) if condition == 'sanitized_source_mean' else 0
        loss_value, displacement = 0., 0.
        if update:
            adapted_input = sanitized_input(x, normal, fixed) if condition == 'sanitized_source_mean' else x
            loss, reconstructed = reconstruction_loss(model, adapted_input, x, mask)
            controls.finite_tensor('masked adaptation loss', loss, window)
            if window in probe_windows:
                probes.append(gradient_diagnostic(model, x, normal, loss, reconstructed, window))
            previous = [p.detach().clone() for p in params]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                step = torch.stack([(p.detach()-old).square().sum() for p, old in zip(params, previous)]).sum().sqrt()
            controls.finite_tensor('actual optimizer displacement', step, window)
            loss_value, displacement = float(loss.detach()), float(step)
        scores.append(score.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())
        traces.append({'window': window, 'normal_points': int(normal.sum()), 'selected_points': int(mask.sum()),
            'replaced_points': replaced, 'optimizer_step': update, 'masked_loss': loss_value,
            'parameter_step_l2': displacement})
    score_array = np.concatenate(scores).reshape(-1)
    mask_array = np.concatenate(masks).reshape(-1)
    trace = pd.DataFrame(traces)
    expected_mask = labels_expected == 0 if condition != 'clean_skip_step' else np.repeat(counts == 12, 12)
    if not np.array_equal(mask_array, expected_mask) or len(probes) != len(probe_windows):
        raise ValueError('Mask/diagnostic count differs from the fixed design')
    value = controls.metric(labels_expected, score_array)
    result = {'complete': True, 'finished_at': controls.now(), 'condition': condition,
        'dataset': base['dataset'], 'subset': base['subset'], 'seed': base['seed'], 'optimizer': optimizer_key,
        'source_checkpoint_sha256': base['source_checkpoint_sha256'], 'source_checkpoint_path': base['checkpoint'],
        'data_manifest': base['data_manifest'], 'metric_version': controls.VERSION, 'auprc': value,
        'test_points': len(labels_expected), 'test_windows': len(trace), 'optimizer_steps': int(trace.optimizer_step.sum()),
        'selected_points': int(mask_array.sum()), 'mixed_windows': len(mixed), 'all_normal_windows': int((counts == 12).sum()),
        'replaced_points': int(trace.replaced_points.sum()), 'gradient_observations': len(probes),
        'optimizer_state_steps': sorted({int(state['step']) for state in optimizer.state.values() if 'step' in state}),
        'gradient_state_trajectory': 'sanitized_source_mean' if condition == 'sanitized_source_mean' else None,
        'source_training_reference': reference.tolist() if condition == 'sanitized_source_mean' else None,
        'source_training_points': len(series.train), 'parameter_step_l2_sum': float(trace.parameter_step_l2.sum()),
        'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu_start,
        'output_path': str(target), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'nice': os.getpriority(os.PRIO_PROCESS, 0)}
    np.save(target/'test_energy.npy', score_array)
    np.save(target/'true.npy', labels_expected)
    np.save(target/'update_mask.npy', mask_array)
    trace.to_parquet(target/'window_trace.parquet', index=False)
    pd.DataFrame(probes, columns=GRAD_COLUMNS).to_parquet(target/'gradient_probes.parquet', index=False)
    controls.atomic_json(target/'result.json', result)
    return result, score_array


def worker(plan, shard):
    torch.cuda.set_device(0)
    validated = json.loads((HERE/'validation.json').read_text())['formal_runs']
    for task in (t for t in plan['tasks'] if t['shard'] == shard):
        base = plan['bases'][task['base_id']]
        print(f"START {task['base_id']} CPU_load={os.getloadavg()[0]:.2f}", flush=True)
        entity = controls.Entity(base['dataset'], base['subset'], Path(base['directory']))
        if controls.entity_manifest(entity) != base['data_manifest'] or controls.sha256_file(Path(base['checkpoint'])) != base['source_checkpoint_sha256']:
            raise ValueError('Original data/checkpoint changed')
        series = controls.prepare_series(entity.directory, controls.load_protocol().train_fraction)
        for condition in CONDITIONS:
            for optimizer in OPTIMIZERS:
                out = directory(base, condition, optimizer)
                if str(out) in validated:
                    if not json.loads((out/'result.json').read_text())['complete']:
                        raise ValueError('Explicitly reused formal validation is incomplete')
                    print(f'REUSE {out}', flush=True)
                    continue
                result, _ = run_one(base, optimizer, condition, series)
                print(f"DONE {condition} {optimizer} AUPRC={result['auprc']:.10g} seconds={result['wall_seconds']:.2f}", flush=True)


def aggregate(plan):
    old = pd.read_csv(HERE.parent/'ae_all_normal_window/derived/physical.csv', keep_default_na=False)
    old = old.set_index(KEYS)
    records, gradient_rows = [], []
    for base in plan['bases'].values():
        for optimizer in OPTIMIZERS:
            refs = {a: controls.reference(base, a, optimizer) for a in ('frozen', 'practical', 'oracle')}
            labels = np.load(refs['oracle']['label_path'], allow_pickle=False).reshape(-1).astype(np.int8)
            counts = (labels.reshape(-1, 12) == 0).sum(1)
            identity = {k: base[k] for k in ('dataset', 'subset', 'seed')}
            identity['optimizer'] = optimizer
            legacy = old.loc[tuple(identity[k] for k in KEYS)]
            for condition in CONDITIONS:
                out = directory(base, condition, optimizer)
                meta = json.loads((out/'result.json').read_text())
                if (not meta['complete'] or any(meta[k] != v for k, v in identity.items())
                    or meta['condition'] != condition or meta['source_checkpoint_sha256'] != base['source_checkpoint_sha256']
                    or meta['data_manifest'] != base['data_manifest'] or meta['metric_version'] != controls.VERSION):
                    raise ValueError(f'Completed result identity mismatch: {out}')
                values = np.load(out/'test_energy.npy', allow_pickle=False)
                truth = np.load(out/'true.npy', allow_pickle=False)
                mask = np.load(out/'update_mask.npy', allow_pickle=False)
                expected_mask = labels == 0 if condition == CONDITIONS[0] else np.repeat(counts == 12, 12)
                if not np.array_equal(truth, labels) or not np.array_equal(mask, expected_mask):
                    raise ValueError('Saved evaluation labels/mask changed')
                if abs(controls.metric(truth, values)-meta['auprc']) > 1e-12:
                    raise ValueError('Saved PR-AUC does not reproduce')
                trace = pd.read_parquet(out/'window_trace.parquet')
                expected_steps = np.ones(len(counts), dtype=bool) if condition == CONDITIONS[0] else counts == 12
                if (len(trace) != len(counts) or not np.array_equal(trace.optimizer_step, expected_steps)
                    or not np.array_equal(trace.selected_points, expected_mask.reshape(-1, 12).sum(1))
                    or meta['optimizer_steps'] != int(expected_steps.sum())
                    or not (trace.loc[~trace.optimizer_step, ['masked_loss', 'parameter_step_l2']] == 0).all().all()):
                    raise ValueError('Step/mask trace violates control')
                grad = pd.read_parquet(out/'gradient_probes.parquet')
                if len(grad) != meta['gradient_observations']:
                    raise ValueError('Gradient diagnostic artifact incomplete')
                if len(grad):
                    gradient_rows.extend([{**identity, **r} for r in grad.to_dict('records')])
                f, p, original = (refs[a]['auprc'] for a in ('frozen', 'practical', 'oracle'))
                records.append({**identity, 'condition': condition, 'auprc': meta['auprc'],
                    'frozen_auprc': f, 'practical_auprc': p, 'original_oracle_auprc': original,
                    'practical_delta': p-f, 'oracle_delta': meta['auprc']-f,
                    'original_oracle_delta': original-f, 'change_vs_point_oracle': meta['auprc']-original,
                    'change_vs_zero_step_clean': meta['auprc']-legacy.window_oracle_auprc,
                    'optimizer_steps': meta['optimizer_steps'], 'selected_points': meta['selected_points'],
                    'gradient_observations': len(grad), 'output_path': str(out)})
    if len(records) != 4240 or len(list((OUTPUT/'runs').rglob('result.json'))) != 4240:
        raise ValueError('Incomplete or unexpected formal GPU grid')
    physical = pd.DataFrame(records)
    effect_cols = ['practical_delta', 'oracle_delta', 'original_oracle_delta', 'change_vs_point_oracle', 'change_vs_zero_step_clean']
    cells = physical.groupby(['condition', 'dataset', 'optimizer', 'seed'])[effect_cols].median().reset_index()
    if len(cells) != 360:
        raise ValueError('Expected 180 cells for each input control')
    cells['category'] = [category(p, o) for p, o in zip(cells.practical_delta, cells.oracle_delta)]
    cells['original_category'] = [category(p, o) for p, o in zip(cells.practical_delta, cells.original_oracle_delta)]
    canonical = pd.read_csv(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv')
    check = cells.merge(canonical, on=['dataset', 'optimizer', 'seed'], validate='many_to_one')
    np.testing.assert_allclose(check.practical_delta, check.corrected_auprc_practical_delta, atol=1e-14, rtol=0)
    np.testing.assert_allclose(check.original_oracle_delta, check.corrected_auprc_oracle_delta, atol=1e-14, rtol=0)
    counts = pd.crosstab(cells.condition, cells.category).reindex(columns=CATEGORIES, fill_value=0).reset_index()
    summary = cells.groupby(['condition', 'dataset'])[effect_cols].median().reset_index()
    transitions = cells.groupby(['condition', 'original_category', 'category']).size().rename('cells').reset_index()
    gradients = pd.DataFrame(gradient_rows)
    if not len(gradients):
        raise ValueError('Expected some mixed-window diagnostics across the grid')
    grad_run = gradients.groupby(KEYS).agg(observations=('window', 'size'), defined_cosines=('cosine', 'count'),
        median_cosine=('cosine', 'median'), median_relative_difference=('relative_difference', 'median'),
        median_gradient_difference=('difference_l2', 'median')).reset_index()
    grad_summary = grad_run.groupby(['dataset', 'optimizer']).median(numeric_only=True).reset_index()
    out = OUTPUT/'derived'
    out.mkdir(exist_ok=False)
    for name, frame in [('physical', physical), ('cells', cells), ('outcome_counts', counts),
        ('dataset_summary', summary), ('transitions', transitions), ('gradient_probes', gradients),
        ('gradient_per_run', grad_run), ('update_diagnostics', grad_summary)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    text = ['# Oracle input controls: complete results', '', 'Each of the two controls has 2,120 Oracle trajectories, using the same 106 entities, four optimizers, and five seeds.',
            'Practical/Frozen results are reused read-only. All evaluation scores come from the original unsanitized inputs.', '',
            '| Condition | BI | OOG | NI | POG |', '|---|---:|---:|---:|---:|']
    text += [f'| {r.condition} | {r.BI} | {r.OOG} | {r.NI} | {r.POG} |' for r in counts.itertuples()]
    text += ['', '| Dataset | Condition | Oracle-Frozen pp | Oracle-original point Oracle pp | Oracle-previous clean zero-gradient step pp |',
             '|---|---|---:|---:|---:|']
    text += [f'| {r.dataset} | {r.condition} | {100*r.oracle_delta:.4f} | {100*r.change_vs_point_oracle:.4f} | {100*r.change_vs_zero_step_clean:.4f} |' for r in summary.itertuples()]
    text += ['', 'Sanitization preserves normal-loss timing but introduces an artificial reference. Skipping steps reduces normal-point updates and changes optimizer-state evolution.',
             'Gradient diagnostics compare the two inputs at the same parameter state of the sanitized trajectory; they do not causally attribute the original Oracle trajectory. Zero-norm cosines/ratios are undefined.',
             'NI does not imply negative gain. Different conditions are not automatically interpreted as robustness successes.', '',
             '[Design](../../../../docs/guides/REBUTTAL.md), [cells](cells.csv), [gradient summary](update_diagnostics.csv).']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    controls.atomic_json(out/'summary.json', {'status': 'complete', 'finished_at': controls.now(),
        'new_oracle_runs': len(physical), 'logical_cells': len(cells), 'gradient_observations': len(gradients),
        'gradient_runs': len(grad_run), 'outcome_counts': counts.to_dict('records'),
        'canonical_practical_and_original_oracle_regression': 'passed'})


def supervise(plan):
    validation = json.loads((HERE/'validation.json').read_text())
    if validation['status'] != 'passed':
        raise RuntimeError('Validation has not passed')
    if (HERE/'status.json').exists():
        raise FileExistsError('Already launched; no automatic relaunch')
    logs = HERE/'logs'
    logs.mkdir(exist_ok=False)
    status = {'status': 'running', 'started_at': controls.now(), 'pid': os.getpid(),
              'expected_gpu_runs': 4240, 'reused_formal_validation_runs': validation['formal_runs'], 'completed_workers': []}
    processes, handles, workers = [], [], []
    controls.atomic_json(HERE/'status.json', status)
    try:
        specs = [(f'gpu{gpu}', plan['cpu_cores'][i], str(gpu), [str(HERE/'experiment.py'), 'worker', '--shard', str(i)])
                 for i, gpu in enumerate(plan['devices'])]
        specs.append(('cpu', plan['analysis_core'], '', [str(HERE/'cpu_analysis.py')]))
        for name, core, gpu, args in specs:
            handle = (logs/f'{name}.log').open('x')
            handles.append(handle)
            p = subprocess.Popen(['taskset', '-c', str(core), sys.executable, *args], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1'),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(p)
            workers.append({'name': name, 'cpu_core': core, 'gpu': gpu, 'pid': p.pid})
        status['workers'] = workers
        controls.atomic_json(HERE/'status.json', status)
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs = {pool.submit(p.wait): (p, w) for p, w in zip(processes, workers)}
            for job in as_completed(jobs):
                p, w = jobs[job]
                code = job.result()
                if code:
                    for other in processes:
                        if other.poll() is None:
                            other.terminate()
                    raise RuntimeError(f"{w['name']} pid={p.pid} exit={code}; inspect logs")
                status['completed_workers'].append(w['name'])
                controls.atomic_json(HERE/'status.json', status)
        aggregate(plan)
        status.update(status='complete', finished_at=controls.now())
        print('COMPLETE: both input controls and all three CPU analyses', flush=True)
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        status.update(status='failed', finished_at=controls.now(), error=str(exc))
        raise
    finally:
        controls.atomic_json(HERE/'status.json', status)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'aggregate', 'supervise'])
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    controls.configure_threads()
    if args.mode == 'prepare':
        prepare()
    else:
        plan = json.loads((HERE/'plan.json').read_text())
        if args.mode == 'worker':
            worker(plan, args.shard)
        elif args.mode == 'aggregate':
            aggregate(plan)
        else:
            supervise(plan)


if __name__ == '__main__':
    main()
