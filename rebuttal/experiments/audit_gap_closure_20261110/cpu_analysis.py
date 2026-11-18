#!/usr/bin/env python3
"""Three bounded CPU analyses of already saved canonical score streams."""
from __future__ import annotations
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import sys
import time
import experiment as runner
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

HERE, ROOT = runner.HERE, runner.ROOT
SOURCE = HERE.parent/'causal_score_calibration'
AFFILIATION = ROOT/'rebuttal/vendor'
if not (AFFILIATION/'affiliation/metrics.py').is_file():
    raise FileNotFoundError(AFFILIATION/'affiliation/metrics.py')
sys.path.insert(0, str(AFFILIATION))
from affiliation.metrics import pr_from_events
from affiliation.generics import convert_vector_to_events

BLOCK = 512 * 12
IDENTITY = ['dataset', 'subset', 'seed', 'optimizer', 'arm']


def load_source(base):
    directory = SOURCE/'source_scores'/base['dataset']/(base['subset'] or '__root__')/f"seed{base['seed']}"
    meta = json.loads((directory/'result.json').read_text())
    scores = np.load(directory/'scores.npy', allow_pickle=False).reshape(-1)
    if (not meta['complete'] or meta['source_checkpoint_sha256'] != base['source_checkpoint_sha256']
        or len(scores) != meta['points'] or not np.isfinite(scores).all() or not len(scores)):
        raise ValueError('Source score history identity/values invalid')
    for q in (95, 99):
        if abs(float(np.percentile(scores, q))-meta[f'p{q}']) > 1e-10:
            raise ValueError('Saved source threshold differs')
    return scores, meta


def load_flows(base):
    flows, common = [], None
    for key, ref in sorted(base['references'].items()):
        optimizer, arm = key.split(':')
        scores = np.load(ref['score_path'], allow_pickle=False).reshape(-1)
        labels = np.load(ref['label_path'], allow_pickle=False).reshape(-1).astype(np.int8)
        if common is None:
            common = labels
        if not np.array_equal(common, labels) or len(labels) % 12:
            raise ValueError('Paired labels/original complete-window prefix differ')
        if abs(runner.controls.metric(labels, scores)-ref['auprc']) > 1e-12:
            raise ValueError('Original scalar PR-AUC no longer matches raw scores')
        meta = json.loads(Path(ref['manifest_path']).read_text())
        if (meta['status'] != 'complete' or meta['source_checkpoint_hash'] != base['source_checkpoint_sha256']
            or meta['data_manifest'] != base['data_manifest']):
            raise ValueError('Original flow identity differs')
        flows.append((optimizer, arm, scores, ref))
    if len(flows) != 9:
        raise ValueError('Expected Frozen and eight original adaptive flows')
    return common, flows


def identity(base, optimizer, arm):
    return {'dataset': base['dataset'], 'subset': base['subset'], 'seed': base['seed'],
            'optimizer': optimizer, 'arm': arm}


def ratio(numerator, denominator):
    return numerator/denominator if denominator else np.nan


def pair_counts(normal, anomaly):
    """Anomaly-higher concordance; exact tie credit of one half."""
    if not len(normal) or not len(anomaly):
        return 0., 0
    ordered = np.sort(normal)
    left = np.searchsorted(ordered, anomaly, side='left')
    right = np.searchsorted(ordered, anomaly, side='right')
    return float((left.astype(np.float64)+right).sum()/2), int(len(normal))*int(len(anomaly))


def distribution(values):
    if not len(values):
        return {'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'q25': np.nan, 'q75': np.nan, 'iqr': np.nan}
    values = np.asarray(values, dtype=np.float64)
    q25, med, q75 = np.quantile(values, [.25, .5, .75])
    return {'count': len(values), 'mean': float(values.mean()), 'median': med,
            'std': float(values.std(ddof=0)), 'q25': q25, 'q75': q75, 'iqr': q75-q25}


def temporal_one(scores, labels, source, block_size=BLOCK):
    source = np.asarray(source, dtype=np.float64)
    source_mean, source_std = float(source.mean()), float(source.std(ddof=0))
    epsilon = max(1e-12, 1e-6*float(np.sqrt(np.mean(source**2))))
    scale = max(source_std, epsilon)
    rows, within_wins, within_pairs = [], 0., 0
    for block, start in enumerate(range(0, len(labels), block_size)):
        truth, values = labels[start:start+block_size], scores[start:start+block_size]
        normal, anomaly = values[truth == 0], values[truth == 1]
        wins, pairs = pair_counts(normal, anomaly)
        within_wins += wins
        within_pairs += pairs
        row = {'block': block, 'start_point': start, 'points': len(truth),
               'source_mean': source_mean, 'source_std': source_std, 'source_scale': scale,
               'source_floor_active': source_std < epsilon, 'local_pair_accuracy': ratio(wins, pairs),
               'local_pair_count': pairs}
        for name, values in [('all', values), ('normal', normal), ('anomaly', anomaly)]:
            summary = distribution(values)
            row.update({f'{name}_{k}': v for k, v in summary.items()})
            row[f'{name}_source_z_mean'] = (summary['mean']-source_mean)/scale
            row[f'{name}_source_z_median'] = (summary['median']-source_mean)/scale
        rows.append(row)
    wins, pairs = pair_counts(scores[labels == 0], scores[labels == 1])
    cross_wins, cross_pairs = wins-within_wins, pairs-within_pairs
    if cross_wins < -1e-8 or cross_wins > cross_pairs+1e-8 or cross_pairs < 0:
        raise ValueError('Within/cross pair-count decomposition invalid')
    within_component, cross_component = within_wins/pairs, cross_wins/pairs
    if abs(wins/pairs-within_component-cross_component) > 1e-12:
        raise ValueError('Exact pair decomposition failed')
    summary = {'global_pair_accuracy': wins/pairs, 'within_pair_accuracy': ratio(within_wins, within_pairs),
        'cross_pair_accuracy': ratio(cross_wins, cross_pairs), 'within_weight': within_pairs/pairs,
        'cross_weight': cross_pairs/pairs, 'within_global_contribution': within_component,
        'cross_global_contribution': cross_component, 'normal_points': int((labels == 0).sum()),
        'anomaly_points': int((labels == 1).sum()), 'blocks': len(rows),
        'both_class_blocks': sum(r['normal_count'] > 0 and r['anomaly_count'] > 0 for r in rows)}
    return rows, summary


def temporal_analysis(plan):
    out = HERE.parent/'temporal_score_drift/derived'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    blocks, records = [], []
    for index, base in enumerate(plan['bases'].values(), 1):
        source, _ = load_source(base)
        labels, flows = load_flows(base)
        for optimizer, arm, scores, ref in flows:
            tag = identity(base, optimizer, arm)
            rows, summary = temporal_one(scores, labels, source)
            blocks.extend({**tag, **r} for r in rows)
            records.append({**tag, **summary, 'global_pr_auc': ref['auprc'], 'score_path': ref['score_path']})
        if index % 25 == 0 or index == 530:
            print(f'TEMPORAL {index}/530 load={os.getloadavg()[0]:.2f}', flush=True)
    if len(records) != 4770:
        raise ValueError('Temporal flow inventory incomplete')
    frame, block_frame = pd.DataFrame(records), pd.DataFrame(blocks)
    measures = ['global_pair_accuracy', 'within_global_contribution', 'cross_global_contribution', 'global_pr_auc']
    frozen = frame[frame.arm == 'frozen'][['dataset', 'subset', 'seed', *measures]].rename(columns={k: f'frozen_{k}' for k in measures})
    paired = frame[frame.arm != 'frozen'].merge(frozen, on=['dataset', 'subset', 'seed'], validate='many_to_one')
    for metric in measures:
        paired[f'{metric}_delta'] = paired[metric]-paired[f'frozen_{metric}']
    np.testing.assert_allclose(paired.global_pair_accuracy_delta,
        paired.within_global_contribution_delta+paired.cross_global_contribution_delta, atol=1e-12, rtol=0)
    delta = [f'{k}_delta' for k in measures]
    cells = paired.groupby(['dataset', 'optimizer', 'seed', 'arm'])[delta].median().reset_index()
    summary = cells.groupby(['dataset', 'arm'])[delta].median().reset_index()
    if len(cells) != 360:
        raise ValueError('Expected 360 arm-specific temporal cells')
    block_keys = ['dataset', 'subset', 'seed', 'block']
    block_measures = [f'{c}_{s}' for c in ('normal', 'anomaly') for s in ('mean', 'median', 'std', 'iqr', 'source_z_mean', 'source_z_median')]
    block_frozen = block_frame[block_frame.arm == 'frozen'][block_keys+block_measures].rename(columns={k: 'frozen_'+k for k in block_measures})
    block_pairs = block_frame[block_frame.arm != 'frozen'].merge(block_frozen, on=block_keys, validate='many_to_one')
    for metric in block_measures:
        block_pairs[f'{metric}_delta'] = block_pairs[metric]-block_pairs[f'frozen_{metric}']
    block_summary = block_pairs.groupby(['dataset', 'optimizer', 'arm', 'block'])[[f'{m}_delta' for m in block_measures]].median().reset_index()
    for name, data in [('physical', frame), ('time_blocks', block_frame), ('paired_entities', paired),
        ('cells', cells), ('dataset_summary', summary), ('paired_time_blocks', block_pairs), ('time_block_summary', block_summary)]:
        data.to_csv(out/f'{name}.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, ('Daphnet', 'GECCO', 'PUMP', 'SWAT2015')):
        for arm, style in [('frozen', '--'), ('oracle', '-')]:
            group = block_frame[(block_frame.dataset == dataset) & (block_frame.arm == arm)
                & ((block_frame.optimizer == 'shared') | (block_frame.optimizer == 'adamw_ttlr0.0001'))]
            med = group.groupby('block')[['start_point', 'normal_median', 'anomaly_median']].median()
            for cls, color in [('normal', 'tab:blue'), ('anomaly', 'tab:red')]:
                ax.plot(med.start_point, med[f'{cls}_median'], style, color=color, label=f'{arm} {cls}', linewidth=1.1)
        ax.set(title=dataset, xlabel='Block start point', ylabel='Raw class-median score')
        ax.legend(fontsize=7)
    fig.suptitle('Fixed chronological blocks; AdamW 1e-4; median over five seeds')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'temporal_class_scores.{suffix}', dpi=160)
    plt.close(fig)
    text = ['# Temporal score scale and pair ordering: complete analysis', '', '4,770 original streams; consecutive blocks of 6,144 points, including the final partial block. Missing-class statistics are undefined.',
        'All Frozen/Practical/Oracle blocks use the same original observations. Fixed source-scale statistics are not online calibration; completed z-score experiments were not rerun.', '',
        '| Dataset | Oracle global PR-AUC delta pp | Global pair-ordering delta pp | Within-block pair contribution delta pp | Cross-block pair contribution delta pp |', '|---|---:|---:|---:|---:|']
    text += [f'| {r.dataset} | {100*r.global_pr_auc_delta:.4f} | {100*r.global_pair_accuracy_delta:.4f} | {100*r.within_global_contribution_delta:.4f} | {100*r.cross_global_contribution_delta:.4f} |'
        for r in summary[summary.arm == 'oracle'].itertuples()]
    text += ['', 'Pair statistics measure AUROC-style ordering with ties=1/2; **they do not decompose AUPRC**. The additive identity holds strictly for each physical pair, but separately aggregated medians need not add up.',
        'Temporal changes in Frozen reflect a fixed model responding to changing inputs. Adaptation minus Frozen measures an additional model-trajectory response; it does not uniquely identify causal contributions from baseline drift, representation changes, or composition changes.',
        'Within-block and cross-block pairs use all points in the stream. A block missing one class has zero within-block pairs, but its points still participate in cross-block comparisons.',
        'Time curves show four predefined datasets; complete details cover all nine. The strongest associations were not selected based on results.', '',
        '[Cells](cells.csv), [all time blocks](time_blocks.csv), [design](../../../../docs/guides/REBUTTAL.md).']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    runner.controls.atomic_json(out/'summary.json', {'status': 'complete', 'finished_at': runner.controls.now(),
        'physical_flows': len(frame), 'time_block_rows': len(block_frame), 'arm_specific_cells': len(cells),
        'exact_pair_decomposition': 'passed', 'raw_metric_regression': 'passed',
        'wall_seconds': time.perf_counter()-started, 'cpu_affinity': sorted(os.sched_getaffinity(0))})


def association_row(frame, x, y, scope):
    finite = frame[[x, y]].dropna()
    n = len(finite)
    defined = n > 1 and finite[x].nunique() > 1 and finite[y].nunique() > 1
    return {'scope': scope, 'predictor': x, 'response': y, 'n': n,
            'spearman_rho': float(spearmanr(finite[x], finite[y]).statistic) if defined else np.nan,
            'defined': defined, 'inference': 'descriptive_only'}


def contamination_analysis(plan):
    out = HERE.parent/'anomaly_ratio_association/derived'
    out.mkdir(parents=True, exist_ok=False)
    shifts = pd.read_csv(HERE.parent/'normal_shift_headroom/derived/entity_shift.csv', keep_default_na=False)
    shifts['anomaly_points'] = shifts.evaluation_points-shifts.benchmark_normal_points
    if len(shifts) != 106 or shifts.duplicated(['dataset', 'subset']).any():
        raise ValueError('Anomaly-ratio support differs')
    records = []
    for base in plan['bases'].values():
        for optimizer in runner.OPTIMIZERS:
            values = {arm: runner.controls.reference(base, arm, optimizer)['auprc'] for arm in ('frozen', 'practical', 'oracle')}
            records.append({'dataset': base['dataset'], 'subset': base['subset'], 'seed': base['seed'], 'optimizer': optimizer,
                'oracle_minus_practical': values['oracle']-values['practical'], **{f'{a}_auprc': v for a, v in values.items()}})
    paired = pd.DataFrame(records).merge(shifts[['dataset', 'subset', 'evaluation_points', 'anomaly_points', 'test_anomaly_fraction']],
        on=['dataset', 'subset'], validate='many_to_one')
    if len(paired) != 2120:
        raise ValueError('Expected 2120 entity/config pairs')
    np.testing.assert_allclose(shifts.test_anomaly_fraction, shifts.anomaly_points/shifts.evaluation_points, atol=1e-15, rtol=0)
    cells = paired.groupby(['dataset', 'optimizer', 'seed']).oracle_minus_practical.median().reset_index()
    entities = paired.groupby(['dataset', 'subset']).agg(oracle_minus_practical=('oracle_minus_practical', 'median'),
        test_anomaly_fraction=('test_anomaly_fraction', 'first'), evaluation_points=('evaluation_points', 'first'),
        anomaly_points=('anomaly_points', 'first')).reset_index()
    summary = cells.groupby('dataset').oracle_minus_practical.median().rename('oracle_minus_practical').reset_index()
    covariates = shifts.groupby('dataset').agg(anomaly_ratio_median=('test_anomaly_fraction', 'median'),
        anomalies=('anomaly_points', 'sum'), points=('evaluation_points', 'sum'), entities=('subset', 'size')).reset_index()
    covariates['anomaly_ratio_pooled'] = covariates.anomalies/covariates.points
    summary = summary.merge(covariates, on='dataset', validate='one_to_one')
    associations = [association_row(summary, x, 'oracle_minus_practical', '9_dataset_families')
                    for x in ('anomaly_ratio_median', 'anomaly_ratio_pooled')]
    associations.append(association_row(entities, 'test_anomaly_fraction', 'oracle_minus_practical', '106_entities_nested_in_families'))
    associations += [association_row(entities[entities.dataset == d], 'test_anomaly_fraction', 'oracle_minus_practical', f'within_{d}') for d in ('MSL', 'SMAP', 'SMD')]
    assoc = pd.DataFrame(associations)
    for name, frame in [('paired_entities', paired), ('cells', cells), ('entities', entities), ('dataset_summary', summary), ('associations', assoc)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.scatter(summary.anomaly_ratio_median*100, summary.oracle_minus_practical*100)
    for r in summary.itertuples():
        ax.annotate(r.dataset, (r.anomaly_ratio_median*100, r.oracle_minus_practical*100), xytext=(3, 4), textcoords='offset points', fontsize=8)
    ax.axhline(0, color='grey', linewidth=.7)
    ax.set(xlabel='Median entity anomaly ratio (%)', ylabel='Median paired Oracle minus Practical (pp)', title='Nine dataset families; scored-prefix support')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'anomaly_ratio_vs_oracle_practical.{suffix}', dpi=160)
    plt.close(fig)
    text = ['# Anomaly ratio and Oracle-minus-Practical: complete analysis', '',
        'The main table covers nine datasets. Ratios and performance use the original scoring prefixes of 106 entities. O-P effects are paired within entities, then aggregated by entity median and configuration median.', '',
        '| Dataset | Entities | Median entity anomaly ratio % | Point-pooled ratio % | Oracle-Practical pp |', '|---|---:|---:|---:|---:|']
    text += [f'| {r.dataset} | {r.entities} | {100*r.anomaly_ratio_median:.4f} | {100*r.anomaly_ratio_pooled:.4f} | {100*r.oracle_minus_practical:.4f} |' for r in summary.itertuples()]
    text += ['', '| Scope | Ratio definition | n | Descriptive Spearman |', '|---|---|---:|---:|']
    text += [f'| {r.scope} | {r.predictor} | {r.n} | {r.spearman_rho:.4f} |' for r in assoc.itertuples()]
    text += ['', 'The 106 entities are nested within nine datasets; the 20 seed/optimizer repetitions are not independent datasets. No significance test assuming such independence was performed.',
        'The full-test anomaly ratio is neither contamination of Practical-selected samples nor drift in the normal conditional distribution. This association alone cannot confirm or reject a low-shift hypothesis.',
        '[Entities](entities.csv), [cells](cells.csv), [design](../../../../docs/guides/REBUTTAL.md).']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    runner.controls.atomic_json(out/'summary.json', {'status': 'complete', 'finished_at': runner.controls.now(),
        'entities': 106, 'physical_pairs': 2120, 'logical_cells': len(cells), 'families': 9,
        'associations': assoc.to_dict('records'), 'cpu_affinity': sorted(os.sched_getaffinity(0))})


def affiliation_one(scores, labels, threshold):
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or scores.shape != labels.shape:
        raise ValueError('Invalid Affiliation input')
    predicted = convert_vector_to_events(scores > threshold)
    truth = convert_vector_to_events(labels)
    result = pr_from_events(predicted, truth, (0, len(labels)))
    p, r = result['Affiliation_Precision'], result['Affiliation_Recall']
    defined = bool(np.isfinite([p, r]).all() and p+r > 0)
    f = 2*p*r/(p+r) if defined else np.nan
    if defined and not (0 <= p <= 1+1e-12 and 0 <= r <= 1+1e-12 and 0 <= f <= 1+1e-12):
        raise ValueError('Affiliation outside probability range')
    reason = '' if defined else 'empty_prediction' if not predicted else 'undefined_precision_or_recall_or_harmonic_mean'
    return {'precision': p, 'recall': r, 'affiliation_f': f, 'defined': defined, 'undefined_reason': reason,
            'predicted_events': len(predicted), 'ground_truth_events': len(truth), 'threshold': threshold,
            'predicted_positive_points': int((scores > threshold).sum()), 'points': len(labels)}


def affiliation_cells(physical):
    records = []
    for quantile in (95, 99):
        frame = physical[physical['quantile'] == quantile]
        lookup = frame.set_index(IDENTITY).affiliation_f.to_dict()
        bases = frame[['dataset', 'subset', 'seed']].drop_duplicates()
        for b in bases.itertuples():
            f = lookup[b.dataset, b.subset, b.seed, 'shared', 'frozen']
            for optimizer in runner.OPTIMIZERS:
                p = lookup[b.dataset, b.subset, b.seed, optimizer, 'practical']
                o = lookup[b.dataset, b.subset, b.seed, optimizer, 'oracle']
                valid = bool(np.isfinite([f, p, o]).all())
                records.append({'quantile': quantile, 'dataset': b.dataset, 'subset': b.subset, 'seed': b.seed,
                    'optimizer': optimizer, 'frozen_f': f, 'practical_f': p, 'oracle_f': o, 'valid': valid,
                    'practical_delta': p-f if valid else np.nan, 'oracle_delta': o-f if valid else np.nan})
    paired = pd.DataFrame(records)
    cells = []
    for key, g in paired.groupby(['quantile', 'dataset', 'optimizer', 'seed']):
        complete = bool(g.valid.all())
        p, o = g.practical_delta.median(), g.oracle_delta.median()
        cells.append(dict(zip(['quantile', 'dataset', 'optimizer', 'seed'], key)) | {
            'entities_total': len(g), 'entities_valid': int(g.valid.sum()), 'defined': complete,
            'practical_delta': p if complete else np.nan, 'oracle_delta': o if complete else np.nan,
            'conditional_practical_delta': p, 'conditional_oracle_delta': o,
            'category': runner.category(p, o) if complete else 'undefined'})
    return paired, pd.DataFrame(cells)


def affiliation_analysis(plan):
    out = HERE.parent/'affiliation_f/derived'
    out.mkdir(parents=True, exist_ok=False)
    started, records = time.perf_counter(), []
    for index, base in enumerate(plan['bases'].values(), 1):
        _, source_meta = load_source(base)
        labels, flows = load_flows(base)
        for optimizer, arm, scores, ref in flows:
            for quantile in (95, 99):
                record = affiliation_one(scores, labels, source_meta[f'p{quantile}'])
                records.append({**identity(base, optimizer, arm), 'quantile': quantile,
                    **record, 'original_pr_auc': ref['auprc'], 'score_path': ref['score_path']})
        if index % 25 == 0 or index == 530:
            print(f'AFFILIATION {index}/530 load={os.getloadavg()[0]:.2f}', flush=True)
    physical = pd.DataFrame(records)
    if len(physical) != 9540 or physical.duplicated(['quantile', *IDENTITY]).any():
        raise ValueError('Expected 4770 distinct flows for each source threshold')
    paired, cells = affiliation_cells(physical)
    if len(cells) != 360:
        raise ValueError('Expected 180 Affiliation cells per source threshold')
    canonical = pd.read_csv(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv')
    cells = cells.merge(canonical[['dataset', 'optimizer', 'seed', 'corrected_auprc_category']],
        on=['dataset', 'optimizer', 'seed'], validate='many_to_one').rename(columns={'corrected_auprc_category': 'original_pr_auc_category'})
    counts = pd.crosstab(cells['quantile'], cells.category).reindex(columns=[*runner.CATEGORIES, 'undefined'], fill_value=0).reset_index()
    summaries = []
    for (quantile, dataset), g in cells.groupby(['quantile', 'dataset']):
        summaries.append({'quantile': quantile, 'dataset': dataset, 'cells': len(g), 'defined_cells': int(g.defined.sum()),
            'practical_delta': g.practical_delta.median() if g.defined.all() else np.nan,
            'oracle_delta': g.oracle_delta.median() if g.defined.all() else np.nan,
            'conditional_defined_cell_practical_delta': g.practical_delta.median(),
            'conditional_defined_cell_oracle_delta': g.oracle_delta.median()})
    summary = pd.DataFrame(summaries)
    transitions = cells.groupby(['quantile', 'original_pr_auc_category', 'category']).size().rename('cells').reset_index()
    for name, frame in [('physical', physical), ('paired_entities', paired), ('cells', cells),
        ('dataset_summary', summary), ('outcome_counts', counts), ('transitions', transitions)]:
        frame.to_csv(out/f'{name}.csv', index=False)
    text = ['# Source-threshold Affiliation-F: complete results', '',
        'The same 106 entities and original scores are used. Frozen source-validation P95 is primary and P99 is a sensitivity setting. All three arms share the same threshold; test labels were not used for tuning.',
        'Events are merged as `[i,i+1)` intervals; Trange covers the full scoring prefix. Undefined precision/F for empty predictions is retained rather than filled with zero.', '',
        '| Source quantile | BI | OOG | NI | POG | Undefined cells |', '|---|---:|---:|---:|---:|---:|']
    text += [f'| P{r.quantile} | {r.BI} | {r.OOG} | {r.NI} | {r.POG} | {r.undefined} |' for r in counts.itertuples()]
    text += ['', '| Dataset | Source quantile | Defined full-support cells / 20 | Oracle-Frozen pp (reported only when all 20 cells are defined) |', '|---|---|---:|---:|']
    text += [f'| {r.dataset} | P{r.quantile} | {r.defined_cells} | {100*r.oracle_delta:.4f} |' for r in summary.itertuples()]
    text += ['', 'This evaluates an event operating point at a source threshold, rather than threshold-free PR-AUC. +0.5 pp is used only for descriptive classification.',
        'Primary cells require all predefined entities to be defined for all three arms. Conditional columns cover only available pairs and must not be described as full-support results.',
        'Category retention must use explicit valid-support denominators. Undefined cells must not be counted as NI or directly compared with the original overall proportions over 180 cells.',
        f'Core implementation: `{AFFILIATION}/affiliation/metrics.py`, using `pr_from_events` without a test-threshold search wrapper.', '',
        '[Cells and support](cells.csv), [predicted/true events and definition status](physical.csv), [design](../../../../docs/guides/REBUTTAL.md).']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    runner.controls.atomic_json(out/'summary.json', {'status': 'complete', 'finished_at': runner.controls.now(),
        'physical_metric_rows': len(physical), 'physical_undefined': int((~physical.defined).sum()),
        'cells': len(cells), 'outcome_counts': counts.to_dict('records'), 'test_label_threshold_search': False,
        'undefined_imputation': False, 'implementation': str(AFFILIATION/'affiliation/metrics.py'),
        'wall_seconds': time.perf_counter()-started, 'cpu_affinity': sorted(os.sched_getaffinity(0))})


def main():
    runner.controls.configure_threads()
    plan = json.loads((HERE/'plan.json').read_text())
    if sorted(os.sched_getaffinity(0)) != [plan['analysis_core']]:
        raise RuntimeError('CPU analyses require the prespecified single-core affinity')
    print(f'START CPU analyses load={os.getloadavg()[0]:.2f}', flush=True)
    for name, function in [('anomaly_ratio_association', contamination_analysis),
                           ('temporal_score_drift', temporal_analysis), ('affiliation_f', affiliation_analysis)]:
        print(f'START {name}', flush=True)
        function(plan)
        print(f'COMPLETE {name}', flush=True)


if __name__ == '__main__':
    main()
