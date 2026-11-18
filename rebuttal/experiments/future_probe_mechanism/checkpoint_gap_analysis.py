#!/usr/bin/env python3
"""CPU-only post-processing of frozen same-step future-probe summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'rebuttal/experiments/future_probe_mechanism'
KEY = ['dataset', 'optimizer', 'seed']

def read(path):
    return pd.read_csv(path, float_precision='round_trip')

def table(f):
    return '```text\n' + f.to_string(index=False, float_format=lambda v: f'{v:.9g}') + '\n```\n'

def summarize(g):
    v = g[g.validity_status == 'valid_two_class']
    d = v.Delta_G
    p = v.pair_fraction_delta
    r = dict(observations=len(g), valid_observations=len(v), invalid_observations=len(g)-len(v))
    for name, mask in [('improving', d > 0), ('deteriorating', d < 0), ('unchanged', d == 0)]:
        r[name+'_count'] = int(mask.sum())
        r[name+'_fraction'] = float(mask.mean()) if len(v) else np.nan
    r.update(mean_Delta_G=d.mean(), median_Delta_G=d.median(), net_gap_change=np.nan,
             net_gap_status='undefined_changing_probe_pools_and_unsampled_steps')
    for name, mask in [('improving', p > 0), ('deteriorating', p < 0), ('unchanged', p == 0)]:
        r['pair_'+name+'_count'] = int(mask.sum())
        r['pair_'+name+'_fraction'] = float(mask.mean()) if len(v) else np.nan
    return r

def main(out):
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((SOURCE/'plan.json').read_text())
    frames, results = [], []
    for task in plan['tasks']:
        b, opt = task['base'], task['optimizer']
        folder = SOURCE/'runs'/b['dataset']/opt/f"seed{b['seed']}"
        f = read(folder/'probes.csv')
        result = json.loads((folder/'result.json').read_text())
        labels = np.load(folder/'true.npy', allow_pickle=False).reshape(-1)
        ref = b['references'][opt+':oracle']
        np.testing.assert_array_equal(labels, np.load(ref['label_path'], allow_pickle=False).reshape(-1))
        np.testing.assert_array_equal(np.load(folder/'test_energy.npy'), np.load(ref['score_path']).reshape(-1))
        n = result['optimizer_steps']
        assert result['complete'] and result['max_score_parity_error'] == 0
        assert result['source_checkpoint_sha256'] == b['source_checkpoint_sha256']
        assert f.window.tolist() == [j*n//20 for j in range(20)]
        assert (f.windows_total == n).all() and len(labels) == 12*n
        for col, value in [('dataset', b['dataset']), ('optimizer', opt), ('seed', b['seed'])]:
            assert (f[col] == value).all()
        for row in f.itertuples():
            for name, label in [('normal', 0), ('anomaly', 1)]:
                indices = np.asarray(json.loads(getattr(row, name+'_indices')), dtype=int)
                candidates = np.flatnonzero(labels[(row.window+1)*12:] == label)+(row.window+1)*12
                expected = candidates[np.linspace(0, len(candidates)-1, min(64,len(candidates)), dtype=int)] if len(candidates) else candidates
                np.testing.assert_array_equal(indices, expected)
                assert len(indices) == getattr(row,name+'_points')
                if not len(indices):
                    assert pd.isna(getattr(row,name+'_before')) and pd.isna(getattr(row,name+'_after'))
        f['source_probe_csv'] = str((folder/'probes.csv').relative_to(ROOT))
        f['source_checkpoint'] = b['checkpoint']
        f['source_checkpoint_sha256'] = b['source_checkpoint_sha256']
        f['before_state_id'] = [f"{b['dataset']}/{opt}/seed{b['seed']}:steps_completed={t}" for t in f.window]
        f['after_state_id'] = [f"{b['dataset']}/{opt}/seed{b['seed']}:steps_completed={t+1}" for t in f.window]
        f['state_provenance'] = 'saved_diagnostic_of_original_replay;intermediate_weights_not_required'
        f['probe_identity_evidence'] = 'same_groups_object_in_before_and_after_calls;indices_revalidated_against_saved_labels'
        frames.append(f)
        results.append({**{k:result[k] for k in KEY}, 'auprc':result['auprc'], 'optimizer_steps':n})
    obs = pd.concat(frames, ignore_index=True).sort_values(KEY+['window']).reset_index(drop=True)
    assert len(obs)==1200 and not obs.duplicated(KEY+['window']).any()
    frozen_aggregate = read(SOURCE/'derived/probes.csv').sort_values(KEY+['window']).reset_index(drop=True)
    # The historical aggregator used default CSV float parsing before re-export.
    pd.testing.assert_frame_equal(obs[frozen_aggregate.columns], frozen_aggregate,
                                  check_exact=False, atol=2e-13, rtol=2e-15)
    obs['validity_status'] = np.where((obs.normal_points>0)&(obs.anomaly_points>0),'valid_two_class','undefined_single_class')
    assert (obs.paired_probe_defined == (obs.validity_status=='valid_two_class')).all()
    obs['G_before'] = obs.anomaly_before-obs.normal_before
    obs['G_after'] = obs.anomaly_after-obs.normal_after
    obs['Delta_G'] = obs.G_after-obs.G_before
    valid = obs.validity_status=='valid_two_class'
    assert valid.sum()==1000 and obs.loc[~valid,'Delta_G'].isna().all()
    assert (obs.selected_points==0).sum()==60
    assert (obs.parameter_step_l2==0).sum()==40
    assert ((obs.parameter_step_l2==0)&(obs.selected_points>0)).sum()==0
    assert (obs.loc[obs.parameter_step_l2==0,'Delta_G']==0).all()
    np.testing.assert_allclose(obs.loc[valid,'Delta_G'],obs.loc[valid,'margin_delta'],atol=2e-12,rtol=1e-8)
    np.testing.assert_allclose(obs.loc[valid,'pair_fraction_after']-obs.loc[valid,'pair_fraction_before'],obs.loc[valid,'pair_fraction_delta'],atol=1e-15,rtol=0)
    obs['gap_change_class'] = np.select([valid&(obs.Delta_G>0),valid&(obs.Delta_G<0),valid&(obs.Delta_G==0)],['improving','deteriorating','unchanged'],default='undefined')
    core = read(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv').rename(columns={'optimizer_id':'optimizer','category':'final_outcome'})
    assert len(core)==180 and not core.duplicated(KEY).any()
    expected_categories = np.select([(core.practical_delta_auprc>.005)&(core.oracle_delta_auprc>.005),core.oracle_delta_auprc>.005,core.practical_delta_auprc>.005],['BI','OOG','POG'],default='NI')
    assert (core.final_outcome==expected_categories).all()
    assert core.final_outcome.value_counts().to_dict()=={'NI':90,'OOG':63,'BI':27}
    traj = pd.DataFrame([{**dict(zip(KEY,k)), **summarize(g)} for k,g in obs.groupby(KEY)])
    traj = traj.merge(core[KEY+['final_outcome','oracle_delta_auprc','practical_delta_auprc','retained_entity_count']],on=KEY,validate='one_to_one',how='left')
    assert len(traj)==60 and traj.final_outcome.notna().all() and (traj.retained_entity_count==1).all()
    # Independently match authoritative paired effects to the plan's saved arm metrics.
    run_frame = pd.DataFrame(results)
    for task in plan['tasks']:
        b,opt=task['base'],task['optimizer']
        row=traj[(traj.dataset==b['dataset'])&(traj.optimizer==opt)&(traj.seed==b['seed'])].iloc[0]
        refs=b['references']
        frozen=refs['shared:frozen']['auprc']
        np.testing.assert_allclose(row.oracle_delta_auprc,refs[opt+':oracle']['auprc']-frozen,atol=1e-14,rtol=0)
        np.testing.assert_allclose(row.practical_delta_auprc,refs[opt+':practical']['auprc']-frozen,atol=1e-14,rtol=0)
    obs=obs.merge(traj[KEY+['final_outcome']],on=KEY,validate='many_to_one')
    comparisons=[]
    for category in ['BI','OOG','NI','POG']:
        t=traj[traj.final_outcome==category]
        r=dict(final_outcome=category,trajectories=len(t),observations=int(t.observations.sum()),valid_observations=int(t.valid_observations.sum()),invalid_observations=int(t.invalid_observations.sum()))
        for col in ['improving_fraction','deteriorating_fraction','unchanged_fraction','mean_Delta_G','median_Delta_G','pair_improving_fraction','pair_deteriorating_fraction']:
            for stat in ['mean','median','min','max']:
                r[stat+'_trajectory_'+col]=getattr(t[col],stat)()
        comparisons.append(r)
    comparison=pd.DataFrame(comparisons)
    hetero=traj.groupby(['dataset','optimizer','final_outcome']).agg(trajectories=('seed','size'),valid_observations=('valid_observations','sum'),median_improving_fraction=('improving_fraction','median'),median_deteriorating_fraction=('deteriorating_fraction','median'),mean_trajectory_mean_Delta_G=('mean_Delta_G','mean'),median_trajectory_median_Delta_G=('median_Delta_G','median')).reset_index()
    contrasts=[]
    for cols in [['dataset'],['optimizer'],['dataset','optimizer'],['dataset','seed']]:
        for keys,g in traj.groupby(cols):
            a,b=g[g.final_outcome=='OOG'],g[g.final_outcome=='NI']
            if len(a) and len(b):
                contrasts.append(dict(stratum='/'.join(cols),value=str(keys),OOG_n=len(a),NI_n=len(b),median_improving_OOG_minus_NI=a.improving_fraction.median()-b.improving_fraction.median(),mean_improving_OOG_minus_NI=a.improving_fraction.mean()-b.improving_fraction.mean()))
    contrasts=pd.DataFrame(contrasts,columns=['stratum','value','OOG_n','NI_n','median_improving_OOG_minus_NI','mean_improving_OOG_minus_NI'])
    distributions=[]
    for scope,g in [('all',obs[valid])]+[(str(k),g) for k,g in obs[valid].groupby('dataset')]:
        for metric in ['Delta_G','pair_fraction_delta']:
            for absolute in [False,True]:
                s=g[metric].abs() if absolute else g[metric]
                distributions.append(dict(scope=scope,metric=('abs_' if absolute else '')+metric,n=len(s),**{f'q{q:g}':s.quantile(q) for q in [0,.05,.25,.5,.75,.95,1]}))
    dist=pd.DataFrame(distributions)
    sensitivity=pd.DataFrame([{'scope':label,**summarize(g)} for label,g in [('all_scheduled',obs),('nonempty_normal_mask',obs[obs.selected_points>0]),('nonzero_parameter_displacement',obs[obs.parameter_step_l2>0])]])
    outputs={'checkpoint_gap_observations.csv':obs,'trajectory_gap_summary.csv':traj,'outcome_gap_comparison.csv':comparison,'dataset_optimizer_gap_summary.csv':hetero,'within_stratum_OOG_NI_comparison.csv':contrasts,'gap_magnitude_distribution.csv':dist,'update_denominator_sensitivity.csv':sensitivity}
    for name,f in outputs.items(): f.to_csv(out/name,index=False,float_format='%.17g')
    report = '''# P3 checkpoint score-gap audit — 2026-11-17

## Rebuttal-ready finding

Among 1,000 valid two-class observations at predetermined Oracle steps, mean anomaly-minus-normal separation improved in **463/1,000 (46.3%)**, deteriorated in **497/1,000 (49.7%)**, and was unchanged in **40/1,000 (4.0%)**. These are sampled local observations, not a census of all optimizer updates or independent experimental replicates. Across trajectories, median improving fractions were **52.63% for OOG (30 trajectories)** and **13.64% for NI (20 trajectories)**. This pooled contrast is dataset-confounded: all NI are Daphnet and all OOG are GECCO/PUMP. It does not establish a within-dataset association, general prediction, or causal explanation of NI. The optimizer-stratified direction reverses for AdamW 3e-4. Local mean-gap reduction is not equivalent to ranking deterioration.

## Data integrity and measurement design

Sources: `rebuttal/P3_ICLR27_REBUTTAL_EVIDENCE_FROZEN_20261113.md`, `rebuttal/experiments/future_probe_mechanism/{plan.json,experiment.py,runs/,derived/}`, and `workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv`. New analysis reads artifacts only. No model training, replay, inference, optimizer mutation, or GPU execution was performed.

The frozen grid is Daphnet/GECCO/PUMP × four optimizer settings × seeds 42–46: 60 unique trajectories, 20 observations each, 1,200 records. Valid support per trajectory is respectively 11/19/20, giving **220 + 380 + 400 = 1,000**. Missing-class support is 180 Daphnet + 20 GECCO = 200 observations; undefined quantities stay blank, never zero-filled. Each observed window t brackets exactly one scheduled `optimizer.step()`, from state after t completed steps to state after t+1 completed steps; t=floor(j*N/20), j=0..19. Intermediate weight files are not needed because the required diagnostics were saved during the original replay. State identifiers in the CSV describe these original states, not newly recovered checkpoints.

For each run, the script checks original full-stream scores and labels for exact equality against the plan's authoritative Oracle arrays, source checkpoint identifiers, the 20-step schedule, unique row identities, and all probe indices against saved labels. Probe indices are reconstructed from the original deterministic label-stratified time rule: up to 64 points/class, evenly spaced in the strictly future suffix. No score-based selection occurs. In the recording code, both `probe` calls receive the identical `groups` object without intervening mutation, preserving indices, order, and labels within each observation. This is code/provenance verification: individual per-probe score vectors were not retained for this 60-run batch, so their class means cannot be independently recomputed from point vectors. Saved class means and strict pair fractions suffice for the requested statistics. Checkpoint-level inference is unnecessary.

G_before = anomaly_before − normal_before; G_after = anomaly_after − normal_after; Delta_G = G_after − G_before, computed in float64 from the original per-run saved means with round-trip CSV parsing. The historical derived aggregate used default float parsing and re-export, producing last-digit differences from per-run records; aggregate agreement is checked with atol=2e-13, rtol=2e-15. Original per-run records are the numerical inputs. Signs use strict >0/<0/=0, with no threshold or zero-filling. Cross-checks against saved mean-of-point margin deltas use numerical tolerance only for agreement, never for classification. Arithmetic operation order can change last digits. Scores originally come from float32 model evaluation; extremely small changes should not be overinterpreted.

Probe pools change between observations and most intervening steps are unsampled. Consequently, **net trajectory gap change is undefined**, and the local deltas are not summed into a claimed end-to-end gap change.

## Main counts and denominator sensitivity

The primary denominator includes scheduled empty-mask steps, as did the original protocol. There are 60 empty-mask observations: 40 have zero displacement and 20 have nonzero AdamW history-driven displacement. All 60 have two-class probes; all 40 zero-displacement records have unchanged gaps. The following rows explicitly separate scheduled steps, normal-selected steps, and actual parameter movement. Fractions use each row's `valid_observations` denominator; multiply by 100 for percentages.

'''
    report += table(sensitivity.drop(columns=['net_gap_change','net_gap_status']))
    report += '''
Pooled mean Delta_G is **−0.0418601139** and median is **0** (1,000 observations). These are score-unit summaries, not comparable effect sizes across differently scaled datasets. Strict pair ordering is P(score_anomaly > score_normal), giving ties no credit: **160/1,000 (16.0%) improve**, **169/1,000 (16.9%) deteriorate**, and **671/1,000 (67.1%) are unchanged**. Saved fractions permit exact changes in the fraction correctly ordered, but not identities/counts of pairs that swap in opposite directions and cancel. Mean-gap and pair-order changes are distinct measurements.

## Magnitude distributions

Quantiles use linear interpolation. Pooled raw magnitudes are provided for completeness; dataset-specific distributions should guide interpretation.

'''+table(dist)
    report += '''
## Trajectory-level outcome association

Each row of `trajectory_gap_summary.csv` is one trajectory; each fraction uses its own 11/19/20 valid observations. Original outcome assignments use paired adaptive-minus-Frozen trapezoidal PR-AUC and strict clear-gain cutoff >0.005: BI both arms, OOG Oracle only, NI neither, POG Practical only. The script independently rederives all 180 authoritative categories (27 BI / 63 OOG / 90 NI / 0 POG), validates a one-to-one join, and checks the 60 joined paired effects against the plan's original arm metrics. No category is recomputed from probe scores. NI means no clear gain, not necessarily negative effect.

The table below uses equal trajectory weighting for mean/median statistics. POG has zero support and undefined summaries. Full ranges for every group statistic are in `outcome_gap_comparison.csv`; all individual values are in the trajectory CSV.

'''
    report += table(comparison[['final_outcome','trajectories','observations','valid_observations','median_trajectory_improving_fraction','median_trajectory_deteriorating_fraction','mean_trajectory_mean_Delta_G','median_trajectory_mean_Delta_G','mean_trajectory_median_Delta_G','median_trajectory_median_Delta_G']])
    report += '''
OOG−NI median improving-fraction difference is **+39.00 percentage points**; the equal-trajectory mean difference is **+32.33 pp**. NI has median deteriorating fraction 77.27%, versus OOG 47.37%. Nevertheless, the mean of trajectory mean Delta_G is more negative in OOG (−0.0633928) than NI (−0.00133969), illustrating scale and magnitude sensitivity. These pooled quantities do not support a universal monotonic relation between gap improvement and global AUPRC.

## Dataset / optimizer heterogeneity and repeated configurations

The following table keeps the five seeds within each dataset/optimizer block visible through trajectory counts. All NI observations belong to Daphnet; GECCO contributes BI and OOG, and PUMP only OOG. Dataset and outcome are inseparable for OOG versus NI in this design. There is no dataset, dataset×optimizer, or dataset×seed stratum containing both OOG and NI. A dataset-adjusted effect therefore cannot be identified from these 60 trajectories; a regression or observation-level significance test would not resolve this limitation.

'''+table(hetero)
    report += '''
Optimizer-stratified descriptive contrasts (still comparing different datasets):

'''+table(contrasts)
    report += '''
Three optimizer settings show higher OOG improving fractions, but AdamW 3e-4 reverses direction: PUMP OOG median 40.0% versus Daphnet NI 45.45% (−5.45 pp). Repeated optimizers and matched seeds are not independent datasets. No p-values, independent-observation intervals, or significance claims are made. With only three dataset clusters and no overlapping OOG/NI support, a cluster bootstrap would not establish robustness across datasets. Raw score magnitudes are not normalized post hoc or used to invent a common effect scale.

## Counterexamples and existing local-update evidence

* Frozen Daphnet actual-update example (seed42, AdamW1e-4, window726): all 12 current points normal; current selected-normal score decreases ≈0.00453519 while future gap decreases ≈0.01014562 and strict pair fraction falls 0.465087890625→0.46337890625. This exact observation is retained in the new CSV; it is an existence example, not a frequency estimate.
* PUMP/AdamW3e-4 succeeds as OOG despite a median improving fraction of only 40% and median deteriorating fraction 55% across five seeds. GECCO/SGD1e-3 is OOG despite a negative mean Delta_G in its dataset/optimizer summary. Successful adaptation can include substantial local gap compression.
* Existing SMD local-update evidence is under `rebuttal/experiments/future_probe_mechanism/derived/smd_m1_6/{RESULTS.md,probes.csv,per_seed.csv,cumulative.csv}`; its separate ten M-1-6 trajectories are **not added to this 60-run denominator**. It finds normal-fitting plus strict-pair harm in 10/75 active observations for failed SGD versus 20/75 for successful AdamW. Both final fixed-probe mean gaps compress while mean pair ordering and PR-AUC improve. This independently limits a universal local-harm accumulation explanation. SMD is one physical entity, so its effects must not be mapped to whole-family SMD outcome categories.

## Scientific limitations and legitimate claims

The evidence quantifies the occurrence of local separation deterioration under the original diagnostic design and shows a pooled descriptive difference between outcomes. It supports the statement that normal-only fitting can reduce future separation and sometimes strict ordering. It does **not** show that local gap changes cause final NI/OOG, predict global AUPRC, explain most NI cases, or identify representation collapse. Probe points, optimizer settings, observations, and seeds are dependent. The three selected families do not represent all nine main benchmark families. Most late Daphnet observations have no future anomalies; no observations were replaced to balance outcome support. Sparse observation frequency is not an estimate over every optimizer update. Existing scientific insights and original outcome assignments are unchanged.

No additional GPU experiment is needed to answer this requested saved-artifact analysis. Establishing a within-dataset OOG–NI relation would require different existing support or additional diagnostic experiments; it cannot be obtained by re-evaluating these same states. Such expansion is outside this sprint.

## Independent verification and reproduction

From the repository root:

```bash
python rebuttal/experiments/future_probe_mechanism/checkpoint_gap_analysis.py
python rebuttal/experiments/future_probe_mechanism/verify_checkpoint_gap.py
```

The second program independently reads the original CSVs using Python's csv/Decimal/statistics modules, rederives observation signs, trajectory fractions and means/medians, verifies original outcome assignments and group summaries, and checks the three required CSVs. It also reruns the CPU analysis in a temporary directory and byte-compares every generated analysis artifact. Its machine-readable result is `independent_verification.json`. Verification is a separate implementation in this session, not a separate human audit or new model evaluation. The source point-vector limitation above remains.

All new evidence is separated under `rebuttal/experiments/future_probe_mechanism/checkpoint_gap/`; original frozen artifacts are read-only inputs. Required CSVs, distribution, denominator-sensitivity, optimizer-stratified and heterogeneity tables, report, and verification record form the saved evidence bundle. Reproduction requires the existing saved artifacts and Python environment, including NumPy and pandas. No original experiment launch commands should be rerun.
'''
    (out/'P3_CHECKPOINT_SCORE_GAP_AUDIT.md').write_text(report)
    print(f'Wrote {len(obs)} observations and {len(traj)} trajectories to {out}')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'rebuttal/experiments/future_probe_mechanism/checkpoint_gap')
    main(parser.parse_args().output)
