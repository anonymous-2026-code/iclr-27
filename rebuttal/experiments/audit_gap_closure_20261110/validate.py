#!/usr/bin/env python3
"""Finite protocol checks; full-grid jobs are launched separately."""
import copy
import json
import os
from pathlib import Path
import time
import experiment as runner
import cpu_analysis as cpu
import numpy as np
import pandas as pd
import torch
from section5_repro.model import WindowAutoencoder


def validate_synthetic():
    torch.manual_seed(42)
    model = WindowAutoencoder(12, 2)
    x = torch.randn(1, 12, 2)
    normal = torch.ones(1, 12, dtype=torch.bool)
    reference = torch.tensor([.3, -.7])
    original, _ = runner.reconstruction_loss(model, x, x, normal)
    modified, _ = runner.reconstruction_loss(model, runner.sanitized_input(x, normal, reference), x, normal)
    torch.testing.assert_close(original, modified, rtol=0, atol=0)
    grads1 = torch.autograd.grad(original, tuple(model.parameters()))
    grads2 = torch.autograd.grad(modified, tuple(model.parameters()))
    for a, b in zip(grads1, grads2):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    normal[:, [1, 4, 7]] = False
    sanitized = runner.sanitized_input(x, normal, reference)
    torch.testing.assert_close(sanitized[normal], x[normal], rtol=0, atol=0)
    torch.testing.assert_close(sanitized[~normal], reference.expand(3, 2), rtol=0, atol=0)
    loss, reconstructed = runner.reconstruction_loss(model, sanitized, x, normal)
    direct = ((reconstructed-x).square().mean(-1)*normal).sum()/12
    torch.testing.assert_close(loss, direct, rtol=1e-6, atol=1e-8)
    clone = copy.deepcopy(model)
    diagnostic = runner.gradient_diagnostic(model, x, normal, loss, reconstructed, 0)
    assert all(p.grad is None for p in model.parameters())
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    opt2 = torch.optim.AdamW(clone.parameters(), lr=1e-4, weight_decay=0)
    loss.backward(); opt1.step()
    clone_loss, _ = runner.reconstruction_loss(clone, sanitized, x, normal)
    clone_loss.backward(); opt2.step()
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    empty = torch.zeros_like(normal)
    opt1.zero_grad()
    empty_loss, _ = runner.reconstruction_loss(model, runner.sanitized_input(x, empty, reference), x, empty)
    empty_loss.backward()
    assert float(empty_loss) == 0 and all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    opt1.step()
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    assert {int(s['step']) for s in opt1.state.values()} == {2}
    assert {int(s['step']) for s in opt2.state.values()} == {1}

    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0, 1, 1, 2, 2, 2, 1, 0], dtype=float)
    blocks, temporal = cpu.temporal_one(scores, labels, np.array([0., 1., 2.]), block_size=4)
    brute = np.mean([float(a > n)+.5*float(a == n) for n in scores[labels == 0] for a in scores[labels == 1]])
    assert abs(temporal['global_pair_accuracy']-brute) < 1e-14
    assert abs(temporal['within_global_contribution']+temporal['cross_global_contribution']-brute) < 1e-14
    one_class_blocks, _ = cpu.temporal_one(np.arange(8.), np.array([0]*4+[1]*4), np.arange(3.), block_size=4)
    assert np.isnan(one_class_blocks[0]['anomaly_mean']) and np.isnan(one_class_blocks[1]['normal_mean'])
    perfect = cpu.affiliation_one(labels.astype(float), labels, .5)
    empty_prediction = cpu.affiliation_one(np.zeros(8), labels, .5)
    assert perfect['defined'] and abs(perfect['affiliation_f']-1) < 1e-14
    assert not empty_prediction['defined'] and np.isnan(empty_prediction['affiliation_f'])
    assert cpu.convert_vector_to_events([0, 1, 0]) == [(1, 2)]
    metric_rows = []
    for q in (95, 99):
        for entity in ('a', 'b'):
            for optimizer, arm in [('shared', 'frozen')] + [(o, a) for o in runner.OPTIMIZERS for a in ('practical', 'oracle')]:
                metric_rows.append({'quantile': q, 'dataset': 'toy', 'subset': entity, 'seed': 42,
                    'optimizer': optimizer, 'arm': arm, 'affiliation_f': np.nan if entity == 'b' and arm == 'oracle' else .5})
    _, cells = cpu.affiliation_cells(pd.DataFrame(metric_rows))
    assert len(cells) == 8 and (cells.category == 'undefined').all()
    assert (cells.entities_valid == 1).all() and cells.practical_delta.isna().all()
    return {'input_mask_target_invariants': 'passed', 'normal_window_exact_loss_gradient': 'passed',
            'gradient_probe_parameter_update_invariance': 'passed', 'adamw_empty_step_vs_skip': 'passed',
            'exact_pair_decomposition_and_ties': 'passed', 'class_absence_preserved': 'passed',
            'affiliation_perfect_empty_singleton_and_support': 'passed',
            'toy_mixed_gradient_difference_l2': diagnostic['difference_l2']}


def main():
    if (runner.HERE/'validation.json').exists():
        raise FileExistsError('Validation already completed')
    runner.controls.configure_threads()
    torch.cuda.set_device(0)
    start = time.perf_counter()
    checks = validate_synthetic()
    plan = json.loads((runner.HERE/'plan.json').read_text())
    base = plan['bases']['Daphnet/__root__/seed42']
    entity = runner.controls.Entity(base['dataset'], base['subset'], Path(base['directory']))
    if runner.controls.entity_manifest(entity) != base['data_manifest'] or runner.controls.sha256_file(Path(base['checkpoint'])) != base['source_checkpoint_sha256']:
        raise ValueError('Validation input identity differs')
    series = runner.controls.prepare_series(entity.directory, runner.controls.load_protocol().train_fraction)
    sources, _ = cpu.load_source(base)
    labels, flows = cpu.load_flows(base)
    for optimizer, arm, values, ref in flows:
        _, result = cpu.temporal_one(values, labels, sources)
        assert np.isfinite(result['global_pair_accuracy'])
        cpu.affiliation_one(values, labels, float(np.percentile(sources, 95)))
    formal, parity = [], []
    for optimizer in ('sgd_ttlr0.001', 'adamw_ttlr0.0001'):
        meta, values = runner.run_one(base, optimizer, 'original_point', series,
            out=runner.HERE/'validation'/'original_point'/optimizer)
        reference = runner.controls.reference(base, 'oracle', optimizer)
        expected = np.load(reference['score_path'], allow_pickle=False).reshape(-1)
        np.testing.assert_allclose(values, expected, atol=1e-6, rtol=1e-5)
        assert abs(meta['auprc']-reference['auprc']) < 1e-10
        parity.append({'optimizer': optimizer, 'max_score_error': float(np.max(np.abs(values-expected))),
                       'auprc_error': abs(meta['auprc']-reference['auprc'])})
        for condition in runner.CONDITIONS:
            meta, values = runner.run_one(base, optimizer, condition, series)
            trace = pd.read_parquet(Path(meta['output_path'])/'window_trace.parquet')
            counts = (labels.reshape(-1, 12) == 0).sum(1)
            expected_steps = np.ones(len(counts), dtype=bool) if condition == runner.CONDITIONS[0] else counts == 12
            np.testing.assert_array_equal(trace.optimizer_step, expected_steps)
            assert meta['optimizer_steps'] == int(expected_steps.sum())
            if optimizer.startswith('adamw'):
                assert meta['optimizer_state_steps'] == [meta['optimizer_steps']]
            assert (trace.loc[~trace.optimizer_step, 'parameter_step_l2'] == 0).all()
            assert np.isfinite(values).all()
            formal.append(meta['output_path'])
    record = {'status': 'passed', 'finished_at': runner.controls.now(), 'checks': checks,
              'real_flow_cpu_checks': 'Daphnet seed42 all nine arms/configurations',
              'original_oracle_parity': parity, 'formal_runs': formal,
              'wall_seconds': time.perf_counter()-start, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    runner.controls.atomic_json(runner.HERE/'validation.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
