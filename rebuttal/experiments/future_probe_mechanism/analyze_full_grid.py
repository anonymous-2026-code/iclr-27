#!/usr/bin/env python3
"""Deterministic CPU-only analysis of retained physical and logical Oracle probes."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'rebuttal/experiments/future_probe_mechanism/checkpoint_gap'
KEY=['dataset','optimizer','seed']
PKEY=['dataset','entity','optimizer','seed']
METRICS=['gap_improving_fraction','gap_deteriorating_fraction','mean_Delta_G','median_Delta_G','pair_improving_fraction','pair_deteriorating_fraction','mean_pair_fraction_delta','median_pair_fraction_delta','mean_normal_delta','mean_anomaly_delta']

def read(path):return pd.read_csv(path,float_precision='round_trip',keep_default_na=True)

def pair(n,a):
    if len(n)==0 or len(a)==0:return np.nan
    return float(np.searchsorted(np.sort(n),a,side='left').mean()/len(n))

def derive(f):
    f=f.copy()
    f['G_before']=f.anomaly_before-f.normal_before
    f['G_after']=f.anomaly_after-f.normal_after
    f['Delta_G']=f.G_after-f.G_before
    f['normal_delta']=f.normal_after-f.normal_before
    f['anomaly_delta']=f.anomaly_after-f.anomaly_before
    f['pair_fraction_delta']=f.pair_fraction_after-f.pair_fraction_before
    f['validity_status']=np.where(f.Delta_G.notna(),'valid_two_class','undefined_single_class')
    for col,prefix in [('Delta_G','gap'),('pair_fraction_delta','pair')]:
        f[prefix+'_direction']=np.select([f[col]>0,f[col]<0,f[col]==0],['improving','deteriorating','unchanged'],default='undefined')
    assert (f.Delta_G.notna()==f.pair_fraction_delta.notna()).all()
    return f

def local_summary(g):
    v=g[g.Delta_G.notna()]
    r={'observations':len(g),'valid_observations':len(v),'undefined_observations':len(g)-len(v)}
    for col,prefix in [('Delta_G','gap'),('pair_fraction_delta','pair')]:
        for name,test in [('improving',v[col]>0),('deteriorating',v[col]<0),('unchanged',v[col]==0)]:
            r[prefix+'_'+name+'_count']=int(test.sum())
            r[prefix+'_'+name+'_fraction']=test.mean() if len(v) else np.nan
        r['mean_'+col]=v[col].mean();r['median_'+col]=v[col].median()
        for q in [.05,.25,.75,.95]:r[col+'_q'+str(q)]=v[col].quantile(q)
        r['median_abs_'+col]=v[col].abs().median()
    for col in ['normal_delta','anomaly_delta']:
        r['mean_'+col]=v[col].mean();r['median_'+col]=v[col].median()
        r[col+'_positive_count']=int((v[col]>0).sum());r[col+'_negative_count']=int((v[col]<0).sum())
    return r

def grouped(f,keys):
    return pd.DataFrame([{**dict(zip(keys,k if isinstance(k,tuple) else (k,))),**local_summary(g)} for k,g in f.groupby(keys,dropna=False,sort=True)])

def family_observations(physical):
    records=[]
    for k,g in physical.groupby(KEY+['observation_index'],sort=True):
        v=g[g.Delta_G.notna()]
        r=dict(zip(KEY+['observation_index'],k))
        r.update(physical_entities_expected=len(g),physical_entities_valid=len(v),normal_points=int(v.normal_points.sum()),anomaly_points=int(v.anomaly_points.sum()),all_entities_valid=len(v)==len(g),entity_median_Delta_G=v.Delta_G.median(),entity_median_pair_delta=v.pair_fraction_delta.median())
        for col in ['normal_before','normal_after','anomaly_before','anomaly_after','pair_fraction_before','pair_fraction_after']:
            r[col]=v[col].mean()
        r['trajectory_id']=f'{k[0]}/{k[1]}/seed{k[2]}'
        r['checkpoint_update_id']=r['trajectory_id']+f'/relative_observation={k[3]}'
        records.append(r)
    return derive(pd.DataFrame(records))

def rho(x,y):
    f=pd.DataFrame({'x':x,'y':y}).dropna()
    if len(f)<3 or f.x.nunique()<2 or f.y.nunique()<2:return np.nan
    return f.x.rank().corr(f.y.rank())

def group_distribution(t,cols):
    result=[]
    for k,g in t.groupby(cols,dropna=False,sort=True):
        r=dict(zip(cols,k if isinstance(k,tuple) else (k,)))
        r.update(trajectories=len(g),valid_observations=int(g.valid_observations.sum()),observations=int(g.observations.sum()),physical_trajectories=int(g.physical_trajectories.sum()))
        for metric in METRICS+['entity_balanced_gap_improving_fraction','entity_balanced_gap_deteriorating_fraction','oracle_delta_auprc']:
            for stat in ['mean','median','min','max']:r[metric+'_'+stat]=getattr(g[metric],stat)()
            for q in [.25,.75]:r[metric+'_q'+str(q)]=g[metric].quantile(q)
        result.append(r)
    return pd.DataFrame(result)

def contrast(t,keys):
    result=[]
    for k,g in t.groupby(keys,sort=True):
        a,b=g[g.final_outcome=='OOG'],g[g.final_outcome=='NI']
        r=dict(zip(keys,k if isinstance(k,tuple) else (k,)))
        r.update(OOG_trajectories=len(a),NI_trajectories=len(b),OOG_valid_observations=int(a.valid_observations.sum()),NI_valid_observations=int(b.valid_observations.sum()),eligible=bool(len(a) and len(b)))
        for metric in METRICS+['entity_balanced_gap_improving_fraction','entity_balanced_gap_deteriorating_fraction']:
            for stat in ['mean','median']:
                av,bv=a[metric].dropna(),b[metric].dropna()
                x=getattr(av,stat)() if len(av) else np.nan
                y=getattr(bv,stat)() if len(bv) else np.nan
                r['OOG_'+metric+'_'+stat]=x;r['NI_'+metric+'_'+stat]=y
                r['OOG_minus_NI_'+metric+'_'+stat]=x-y if len(a) and len(b) else np.nan
        result.append(r)
    return pd.DataFrame(result)

def tbl(f):
    return '```text\n'+f.to_string(index=False,float_format=lambda x:f'{x:.8g}')+'\n```\n'

def main(output):
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((OUT/'full_grid_execution_plan.json').read_text())
    core=read(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv').rename(columns={'optimizer_id':'optimizer','category':'final_outcome'})
    assert len(core)==180 and not core.duplicated(KEY).any()
    cats=np.select([(core.practical_delta_auprc>.005)&(core.oracle_delta_auprc>.005),core.oracle_delta_auprc>.005,core.practical_delta_auprc>.005],['BI','OOG','POG'],default='NI')
    assert (core.final_outcome==cats).all() and core.final_outcome.value_counts().to_dict()=={'NI':90,'OOG':63,'BI':27}
    frames=[];alternatives=[];tails=[];provenance=[];checks=[]
    for task in plan['tasks']:
        b,opt=task['base'],task['optimizer']
        folder=Path(task['reuse_directory']) if 'reuse_directory' in task else OUT/'full_grid_runs'/(b['subset'] or '__root__')/b['dataset']/opt/f"seed{b['seed']}"
        result_path=folder/'result.json'
        if not result_path.exists():raise FileNotFoundError(f'Incomplete task {task["task_id"]}: {result_path}')
        r=json.loads(result_path.read_text())
        assert r['complete'] and r['source_checkpoint_sha256']==b['source_checkpoint_sha256'] and r['auprc_parity_error']<=1e-8
        if 'reuse_directory' not in task:assert r['configuration_sha256']==plan['configuration_sha256']
        ref=b['references'][opt+':oracle'];frozen=b['references']['shared:frozen']
        labels=np.load(folder/'true.npy',allow_pickle=False).reshape(-1)
        np.testing.assert_array_equal(labels,np.load(ref['label_path'],allow_pickle=False).reshape(-1))
        score=np.load(folder/'test_energy.npy',allow_pickle=False).reshape(-1)
        original=np.load(ref['score_path'],allow_pickle=False).reshape(-1)
        np.testing.assert_allclose(score,original,atol=1e-6,rtol=1e-5)
        assert abs(np.max(np.abs(score-original))-r['max_score_parity_error'])<1e-15
        frozen_score=np.load(frozen['score_path'],allow_pickle=False).reshape(-1).astype(np.float64)
        np.testing.assert_array_equal(labels,np.load(frozen['label_path'],allow_pickle=False).reshape(-1))
        hard_threshold=np.quantile(frozen_score[labels==0],.85)
        boundary_threshold=np.quantile(frozen_score[labels==1],.30)
        hard=(labels==0)&(frozen_score>=hard_threshold)
        boundary=(labels==1)&(frozen_score<=boundary_threshold)
        drop=frozen_score-original.astype(np.float64)
        hard_drop=float(np.median(drop[hard]));boundary_drop=float(np.median(drop[boundary]))
        def correct_pairs(normal,anomaly):
            return int(np.searchsorted(np.sort(normal),anomaly,side='left').sum())
        npairs=int(hard.sum()*boundary.sum())
        clean_shift=dict(clean_hard_normal_drop=hard_drop,clean_boundary_anomaly_drop=boundary_drop,
                         clean_selective_compression=hard_drop-boundary_drop,
                         clean_net_pair_gain=(correct_pairs(original[hard],original[boundary])-correct_pairs(frozen_score[hard],frozen_score[boundary]))/npairs,
                         hard_normal_points=int(hard.sum()),boundary_anomaly_points=int(boundary.sum()),tail_pairs=npairs)
        f=read(folder/'probes.csv')
        n=len(labels)//12
        assert len(f)==20 and f.window.tolist()==[j*n//20 for j in range(20)]
        assert (f.dataset==b['dataset']).all() and (f.optimizer==opt).all() and (f.seed==b['seed']).all()
        f['entity']=b['subset'] or '__root__';f['observation_index']=np.arange(20)
        f['trajectory_id']=f"{b['dataset']}/{f.entity.iloc[0]}/{opt}/seed{b['seed']}"
        f['checkpoint_update_id']=[f.trajectory_id.iloc[0]+f'/step={t}->{t+1}' for t in f.window]
        f['before_steps_completed']=f.window;f['after_steps_completed']=f.window+1
        f['probe_csv_path']=str((folder/'probes.csv').relative_to(ROOT))
        f['vectors_available']=(folder/'probe_vectors.jsonl').exists()
        f['probe_vectors_path']=str((folder/'probe_vectors.jsonl').relative_to(ROOT)) if f.vectors_available.iloc[0] else ''
        f['source_checkpoint_sha256']=b['source_checkpoint_sha256'];f['configuration_sha256']=task['configuration_sha256']
        for row in f.itertuples():
            first=(row.window+1)*12
            for name,value in [('normal',0),('anomaly',1)]:
                indices=np.array(json.loads(getattr(row,name+'_indices')),dtype=int)
                eligible=np.flatnonzero(labels[first:]==value)+first
                expected=eligible[np.linspace(0,len(eligible)-1,min(64,len(eligible)),dtype=int)] if len(eligible) else eligible
                np.testing.assert_array_equal(indices,expected)
                assert len(indices)==getattr(row,name+'_points')
                assert int((labels[row.window*12:(row.window+1)*12]==0).sum())==row.selected_points
                if not len(indices):assert pd.isna(getattr(row,name+'_before')) and pd.isna(getattr(row,name+'_after'))
        f['normal_labels']=[json.dumps([0]*int(n)) for n in f.normal_points]
        f['anomaly_labels']=[json.dumps([1]*int(n)) for n in f.anomaly_points]
        f=derive(f)
        # Difference-of-means versus mean-of-point-differences has scale-
        # dependent float64 cancellation, especially for large SMD scores.
        scale=f[['normal_before','normal_after','anomaly_before','anomaly_after']].abs().sum(axis=1)
        f['gap_arithmetic_difference']=f.Delta_G-f.margin_delta
        f['gap_arithmetic_roundoff_bound']=16*np.finfo(np.float64).eps*scale
        assert f.Delta_G.isna().equals(f.margin_delta.isna())
        assert (f.loc[f.Delta_G.notna(),'gap_arithmetic_difference'].abs()<=f.loc[f.Delta_G.notna(),'gap_arithmetic_roundoff_bound']).all(), f'{b["dataset"]}/{b["subset"]}/{opt}/{b["seed"]}: gap arithmetic exceeds float64 roundoff bound'
        frames.append(f)
        meta={k:f.iloc[0][k] for k in PKEY}
        if f.vectors_available.iloc[0]:
            records=[json.loads(line) for line in (folder/'probe_vectors.jsonl').read_text().splitlines()]
            assert len(records)==40 and len({(v['window'],v['design']) for v in records})==40
            for v in records:
                row=f[f.window==v['window']].iloc[0]
                common={**meta,'observation_index':int(row.observation_index),'window':v['window'],'trajectory_id':row.trajectory_id,'probe_vectors_path':row.probe_vectors_path}
                vals={}
                for name,label in [('normal',0),('anomaly',1)]:
                    indices=np.array(v[name+'_indices'],dtype=int)
                    assert v[name+'_labels']==[label]*len(indices) and np.all(labels[indices]==label)
                    first=(v['window']+1)*12
                    eligible=np.flatnonzero(labels[first:]==label)+first;k=min(64,len(eligible))
                    ranks=np.linspace(0,len(eligible)-1,k,dtype=int) if v['design']=='primary' else np.floor((np.arange(k)+.5)*len(eligible)/k).astype(int) if k else np.empty(0,dtype=int)
                    np.testing.assert_array_equal(indices,eligible[ranks])
                    vals[name]={phase:np.array(v[name+'_'+phase],dtype=np.float64) for phase in ['before','after']}
                    assert all(len(x)==len(indices) and np.isfinite(x).all() for x in vals[name].values())
                    if v['design']=='primary':
                        assert json.loads(row[name+'_indices'])==indices.tolist()
                        for phase in ['before','after']:
                            if len(indices):np.testing.assert_allclose(vals[name][phase].mean(),row[name+'_'+phase],atol=0,rtol=0)
                if v['design']=='primary':
                    for phase in ['before','after']:
                        np.testing.assert_allclose(pair(vals['normal'][phase],vals['anomaly'][phase]),row['pair_fraction_'+phase],atol=0,rtol=0,equal_nan=True)
                    tail={**common,'hard_threshold_frozen_q85':hard_threshold,'boundary_threshold_frozen_q30':boundary_threshold}
                    for name in ['normal','anomaly']:
                        indices=np.array(v[name+'_indices'],dtype=int)
                        mask=frozen_score[indices]>=hard_threshold if name=='normal' else frozen_score[indices]<=boundary_threshold
                        vals[name]={phase:x[mask] for phase,x in vals[name].items()}
                        tail[name+'_points']=int(mask.sum());tail[name+'_indices']=json.dumps(indices[mask].tolist())
                        for phase in ['before','after']:tail[name+'_'+phase]=vals[name][phase].mean() if mask.any() else np.nan
                    for phase in ['before','after']:tail['pair_fraction_'+phase]=pair(vals['normal'][phase],vals['anomaly'][phase])
                    tails.append(tail)
                else:
                    alt=common.copy()
                    for name in ['normal','anomaly']:
                        alt[name+'_points']=len(vals[name]['before']);alt[name+'_indices']=json.dumps(v[name+'_indices'])
                        for phase in ['before','after']:alt[name+'_'+phase]=vals[name][phase].mean() if len(vals[name][phase]) else np.nan
                    for phase in ['before','after']:alt['pair_fraction_'+phase]=pair(vals['normal'][phase],vals['anomaly'][phase])
                    alternatives.append(alt)
        provenance.append({**meta,**clean_shift,'result_path':str(result_path.relative_to(ROOT)),'max_score_error':r['max_score_parity_error'],'auprc_error':r['auprc_parity_error'],'reused':'reuse_directory' in task,'wall_seconds':r['wall_seconds'],'gpu':r.get('physical_gpu'),'vectors_available':bool(f.vectors_available.iloc[0]),'physical_oracle_delta_auprc':ref['auprc']-frozen['auprc'],'physical_practical_delta_auprc':b['references'][opt+':practical']['auprc']-frozen['auprc']})
    physical=pd.concat(frames,ignore_index=True).sort_values(PKEY+['observation_index']).reset_index(drop=True)
    assert len(physical)==42400 and not physical.duplicated(PKEY+['observation_index']).any()
    assert physical.Delta_G.notna().sum()==plan['expected_valid_physical_observations']
    family=family_observations(physical)
    assert len(family)==3600
    physical_trajectories=grouped(physical,PKEY)
    family_trajectories=grouped(family,KEY)
    entity_balanced=physical_trajectories.groupby(KEY).agg(physical_trajectories=('entity','size'),physical_valid_observations=('valid_observations','sum'),entities_with_valid_observations=('gap_improving_fraction','count'),entity_balanced_gap_improving_fraction=('gap_improving_fraction','mean'),entity_median_gap_improving_fraction=('gap_improving_fraction','median'),entity_balanced_gap_deteriorating_fraction=('gap_deteriorating_fraction','mean'),entity_balanced_pair_improving_fraction=('pair_improving_fraction','mean'),entity_balanced_mean_Delta_G=('mean_Delta_G','mean')).reset_index()
    trajectory=family_trajectories.merge(entity_balanced,on=KEY,validate='one_to_one').merge(core[KEY+['final_outcome','oracle_delta_auprc','practical_delta_auprc','retained_entity_count','oracle_normal_loss_reduction']],on=KEY,validate='one_to_one',how='left')
    assert len(trajectory)==180 and trajectory.final_outcome.notna().all() and (trajectory.physical_trajectories==trajectory.retained_entity_count).all()
    provenance=pd.DataFrame(provenance)
    assert len(provenance)==2120
    physical_trajectories=physical_trajectories.merge(
        provenance[PKEY+['physical_oracle_delta_auprc','physical_practical_delta_auprc']],
        on=PKEY,validate='one_to_one')
    paired=provenance.groupby(KEY)[['physical_oracle_delta_auprc','physical_practical_delta_auprc']].median().reset_index().merge(trajectory,on=KEY,validate='one_to_one')
    np.testing.assert_allclose(paired.physical_oracle_delta_auprc,paired.oracle_delta_auprc,atol=1e-14,rtol=0)
    np.testing.assert_allclose(paired.physical_practical_delta_auprc,paired.practical_delta_auprc,atol=1e-14,rtol=0)
    trajectory['trajectory_id']=[f'{r.dataset}/{r.optimizer}/seed{r.seed}' for r in trajectory.itertuples()]
    trajectory['net_gap_change']=np.nan
    trajectory['net_gap_status']='undefined_changing_probe_pools_and_unsampled_updates'
    for f in [physical,family,physical_trajectories]:
        merged=f.merge(core[KEY+['final_outcome','oracle_delta_auprc']],on=KEY,validate='many_to_one',how='left')
        for c in ['final_outcome','oracle_delta_auprc']:f[c]=merged[c].values
    old=read(OUT/'checkpoint_gap_observations.csv').sort_values(KEY+['window']).reset_index(drop=True)
    old_now=physical[physical.dataset.isin(['Daphnet','GECCO','PUMP'])].sort_values(KEY+['window']).reset_index(drop=True)
    assert len(old_now)==1200
    for c in ['normal_indices','anomaly_indices','Delta_G','G_before','G_after','pair_fraction_delta']:
        if old[c].dtype=='object':assert old[c].equals(old_now[c])
        else:np.testing.assert_array_equal(old[c],old_now[c])
    tables={
        'checkpoint_gap_observations_full_grid.csv':physical,
        'family_checkpoint_gap_observations_full_grid.csv':family,
        'physical_trajectory_gap_summary_full_grid.csv':physical_trajectories,
        'trajectory_gap_summary_full_grid.csv':trajectory,
        'full_grid_parity_provenance.csv':provenance,
        'dataset_outcome_gap_comparison.csv':contrast(trajectory,['dataset']),
        'optimizer_gap_comparison.csv':contrast(trajectory,['optimizer']),
        'dataset_optimizer_outcome_gap_comparison.csv':contrast(trajectory,['dataset','optimizer']),
        'dataset_seed_outcome_gap_comparison.csv':contrast(trajectory,['dataset','seed']),
        'outcome_gap_comparison_full_grid.csv':group_distribution(trajectory,['final_outcome']),
        'dataset_trajectory_gap_distribution.csv':group_distribution(trajectory,['dataset']),
        'dataset_optimizer_gap_comparison_full_grid.csv':group_distribution(trajectory,['dataset','optimizer']),
    }
    # Explicit zero-support POG row, no imputation of undefined measurements.
    if 'POG' not in tables['outcome_gap_comparison_full_grid.csv'].final_outcome.values:
        tables['outcome_gap_comparison_full_grid.csv']=pd.concat([tables['outcome_gap_comparison_full_grid.csv'],pd.DataFrame([{'final_outcome':'POG','trajectories':0,'valid_observations':0,'observations':0,'physical_trajectories':0}])],ignore_index=True)
    summaries=[];agreements=[];magnitudes=[]
    for level,f in [('physical',physical),('family_macro',family)]:
        sets=[('all','all',f)]+[('dataset',k,g) for k,g in f.groupby('dataset')]+[('optimizer',k,g) for k,g in f.groupby('optimizer')]+[('dataset_optimizer','/'.join(k),g) for k,g in f.groupby(['dataset','optimizer'])]
        for scope,value,g in sets:
            summaries.append({'level':level,'scope':scope,'value':value,**local_summary(g)})
            for (a,b),v in g.groupby(['gap_direction','pair_direction']):agreements.append(dict(level=level,scope=scope,value=value,gap_direction=a,pair_direction=b,count=len(v),all_observations=len(g),valid_observations=int(g.Delta_G.notna().sum())))
            for metric in ['Delta_G','normal_delta','anomaly_delta','pair_fraction_delta']:
                v=g[g.Delta_G.notna()][metric]
                magnitudes.append(dict(level=level,scope=scope,value=value,metric=metric,n=len(v),**{f'q{q:g}':v.quantile(q) for q in [0,.05,.25,.5,.75,.95,1]},mean=v.mean(),median_abs=v.abs().median()))
    tables['checkpoint_gap_summary_full_grid.csv']=pd.DataFrame(summaries)
    tables['gap_pair_agreement_full_grid.csv']=pd.DataFrame(agreements)
    tables['gap_magnitude_distribution_full_grid.csv']=pd.DataFrame(magnitudes)
    response_directions=[]
    for level,f in [('physical',physical),('family_macro',family)]:
        for value,g in [('all',f)]+[(k,g) for k,g in f.groupby('dataset')]:
            valid=g[g.Delta_G.notna()].copy()
            for name in ['normal','anomaly']:
                valid[name+'_direction']=np.select([valid[name+'_delta']>0,valid[name+'_delta']<0],['increases','decreases'],default='unchanged')
            for (n,a),v in valid.groupby(['normal_direction','anomaly_direction']):
                response_directions.append(dict(level=level,dataset=value,normal_score_direction=n,anomaly_score_direction=a,count=len(v),valid_denominator=len(valid),fraction=len(v)/len(valid)))
    tables['normal_anomaly_response_directions_full_grid.csv']=pd.DataFrame(response_directions)
    alt=derive(pd.DataFrame(alternatives));tail=derive(pd.DataFrame(tails))
    tables['probe_sensitivity_observations_full_grid.csv']=alt
    tables['tail_gap_observations_full_grid.csv']=tail
    sensitivity=[];tail_summary=[]
    primary_comparable=physical[physical.vectors_available]
    for name,f in [('primary_available_vectors',primary_comparable),('midpoint_sensitivity',alt)]:
        for level,x in [('physical',f),('family_macro',family_observations(f))]:
            for value,g in [('all',x)]+[(k,g) for k,g in x.groupby('dataset')]:sensitivity.append(dict(design=name,level=level,dataset=value,**local_summary(g)))
    for level,f in [('physical',tail),('family_macro_partial_support',family_observations(tail))]:
        for value,g in [('all',f)]+[(k,g) for k,g in f.groupby('dataset')]:tail_summary.append(dict(level=level,dataset=value,**local_summary(g)))
    tables['probe_sampling_sensitivity_full_grid.csv']=pd.DataFrame(sensitivity)
    tables['tail_gap_summary_full_grid.csv']=pd.DataFrame(tail_summary)
    alt_traj=grouped(family_observations(alt),KEY).merge(core[KEY+['final_outcome','oracle_delta_auprc']],on=KEY,validate='one_to_one')
    tail_traj=grouped(family_observations(tail),KEY).merge(core[KEY+['final_outcome','oracle_delta_auprc']],on=KEY,validate='one_to_one')
    primary_matched_traj=grouped(family_observations(primary_comparable),KEY).merge(core[KEY+['final_outcome','oracle_delta_auprc']],on=KEY,validate='one_to_one')
    sensitivity_contrasts=[]
    for design,f in [('primary_available_vectors',primary_matched_traj),('midpoint_sensitivity',alt_traj)]:
        comparisons=contrast(f.assign(entity_balanced_gap_improving_fraction=np.nan,entity_balanced_gap_deteriorating_fraction=np.nan),['dataset'])
        comparisons['design']=design;sensitivity_contrasts.append(comparisons)
    tables['probe_sensitivity_dataset_outcome_comparison.csv']=pd.concat(sensitivity_contrasts,ignore_index=True)
    tables['tail_trajectory_gap_summary_full_grid.csv']=tail_traj
    # Current corrected full-stream SC/NPG; predictor support can exceed detection support.
    shift=read(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/score_shift_full_analysis.csv')
    shift=shift[np.isclose(shift.alpha,.15)&np.isclose(shift.beta,.30)]
    assert len(shift)==180 and not shift.duplicated(KEY).any()
    joined=trajectory.merge(shift[KEY+['selective_compression','net_pair_gain','oracle_delta_auprc','category']],on=KEY,validate='one_to_one',suffixes=('','_shift'))
    np.testing.assert_allclose(joined.oracle_delta_auprc,joined.oracle_delta_auprc_shift,atol=1e-14,rtol=0)
    assert (joined.final_outcome==joined.category).all()
    clean_columns=['clean_selective_compression','clean_net_pair_gain','clean_hard_normal_drop','clean_boundary_anomaly_drop']
    clean_shift_cells=provenance.groupby(KEY)[clean_columns].median().reset_index()
    joined=joined.merge(clean_shift_cells,on=KEY,validate='one_to_one')
    tables['clean_support_score_shift_physical.csv']=provenance[PKEY+clean_columns+['hard_normal_points','boundary_anomaly_points','tail_pairs']]
    tables['clean_support_score_shift_cells.csv']=clean_shift_cells
    tables['local_full_stream_reconciliation.csv']=joined
    correlations=[]
    predictors=METRICS+['entity_balanced_gap_improving_fraction','selective_compression','net_pair_gain','clean_selective_compression','clean_net_pair_gain','oracle_normal_loss_reduction']
    for predictor in predictors:
        for scope,value,g in [('pooled','all',joined)]+[('within_dataset',k,g) for k,g in joined.groupby('dataset')]+[('within_optimizer_cross_dataset',k,g) for k,g in joined.groupby('optimizer')]:
            correlations.append(dict(predictor=predictor,response='oracle_delta_auprc',scope=scope,value=value,n=len(g[[predictor,'oracle_delta_auprc']].dropna()),rho=rho(g[predictor],g.oracle_delta_auprc)))
        ds=joined.groupby('dataset')[[predictor,'oracle_delta_auprc']].median()
        correlations.append(dict(predictor=predictor,response='oracle_delta_auprc',scope='dataset_medians',value='all',n=len(ds),rho=rho(ds[predictor],ds.oracle_delta_auprc)))
        ranks=joined.groupby('dataset')[[predictor,'oracle_delta_auprc']].rank()
        correlations.append(dict(predictor=predictor,response='oracle_delta_auprc',scope='pooled_within_dataset_ranks',value='all',n=len(ranks.dropna()),rho=rho(ranks[predictor],ranks.oracle_delta_auprc)))
        for dataset in sorted(joined.dataset.unique()):
            g=joined[joined.dataset!=dataset]
            correlations.append(dict(predictor=predictor,response='oracle_delta_auprc',scope='leave_one_dataset_out',value=dataset,n=len(g),rho=rho(g[predictor],g.oracle_delta_auprc)))
    for response in ['selective_compression','net_pair_gain','clean_selective_compression','clean_net_pair_gain']:
        for predictor in ['gap_improving_fraction','mean_Delta_G','pair_improving_fraction','mean_pair_fraction_delta']:
            correlations.append(dict(predictor=predictor,response=response,scope='pooled',value='all',n=len(joined),rho=rho(joined[predictor],joined[response])))
            for dataset,g in joined.groupby('dataset'):
                correlations.append(dict(predictor=predictor,response=response,scope='within_dataset',value=dataset,n=len(g),rho=rho(g[predictor],g[response])))
    tables['continuous_gap_associations_full_grid.csv']=pd.DataFrame(correlations)
    examples=[]
    for dataset,g in trajectory.groupby('dataset'):
        for category,sort_ascending in [('NI',False),('OOG',True)]:
            candidates=g[g.final_outcome==category]
            if len(candidates):
                chosen=candidates.sort_values(['gap_improving_fraction','optimizer','seed'],ascending=[sort_ascending,True,True]).iloc[0]
                examples.append({**chosen.to_dict(),'example_rule':'largest_gap_improving_fraction_in_NI' if category=='NI' else 'smallest_gap_improving_fraction_in_OOG'})
    tables['full_grid_descriptive_counterexamples.csv']=pd.DataFrame(examples)
    denominator=[]
    for label,f in [('all_scheduled',physical),('selected_normal_points_positive',physical[physical.selected_points>0]),('nonzero_parameter_displacement',physical[physical.parameter_step_l2>0]),('all_current_points_normal',physical[physical.selected_points==12]),('current_selected_normal_fit_improves',physical[physical.current_selected_normal_delta<0]),('family_all_entities_two_class',family[family.all_entities_valid])]:
        denominator.append(dict(scope=label,**local_summary(f)))
    tables['full_grid_denominator_sensitivity.csv']=pd.DataFrame(denominator)
    for name,f in tables.items():f.to_csv(output/name,index=False,float_format='%.17g')
    manifest={k:dict(path=str((OUT/k).relative_to(ROOT)),rows=len(v),columns=list(v.columns)) for k,v in tables.items()}
    (output/'full_grid_analysis_artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_report(output,plan,tables)
    print('Complete:',len(trajectory),'logical cells;',len(physical_trajectories),'physical trajectories;',len(physical),'physical observations;',physical.Delta_G.notna().sum(),'valid physical;',family.Delta_G.notna().sum(),'valid family')
    print(tbl(tables['checkpoint_gap_summary_full_grid.csv'].query("scope=='all'")[["level","observations","valid_observations","gap_improving_count","gap_improving_fraction","gap_deteriorating_count","gap_deteriorating_fraction","gap_unchanged_count"]]))

def write_report(output,plan,t):
    summary=t['checkpoint_gap_summary_full_grid.csv'];overall=summary[summary.scope=='all']
    d=t['dataset_outcome_gap_comparison.csv'];eligible=d[d.eligible]
    report='''# P3 full-grid checkpoint score-gap audit
+
+## Executive summary
+
+This expansion reuses the original three-family evidence and measures the same local future-probe diagnostic throughout the nine-family clean-support main AE grid. Outcomes are the original paired trapezoidal PR-AUC assignments at a strict >0.005 clear-gain cutoff. The observations are descriptive diagnostics of actual optimizer steps, not independent experimental units or a causal attribution of final detection performance.
+
+'''.replace('\n+','\n')
    report+=tbl(overall[['level','observations','valid_observations','undefined_observations','gap_improving_count','gap_improving_fraction','gap_deteriorating_count','gap_deteriorating_fraction','gap_unchanged_count','gap_unchanged_fraction','mean_Delta_G','median_Delta_G']])
    observations=t['checkpoint_gap_observations_full_grid.csv']
    v=observations[observations.Delta_G.notna()]
    max_roundoff=v.gap_arithmetic_difference.abs().max()
    sign_disagreements=int((np.sign(v.Delta_G)!=np.sign(v.margin_delta)).sum())
    report+=f"\nNumerical audit: difference-of-class-means Delta_G versus the original mean-of-paired-point margin delta differs by at most **{max_roundoff:.12g}**, with **{sign_disagreements} sign disagreements / {len(v):,} valid observations**. All discrepancies satisfy 16 × float64 epsilon × the sum of absolute before/after class means. This score-scale-dependent roundoff bound is for verification only; no gap is thresholded, repaired, or reclassified. Per-observation discrepancies and bounds are saved in the physical observation CSV. Independent integer strict-pair counts can differ from the recorder's two-stage floating division by one ulp; verification allows two ulps of fraction roundoff, with no ordering-sign changes.\n"
    physical=overall[overall.level=='physical'].iloc[0]
    macro=overall[overall.level=='family_macro'].iloc[0]
    report+=f"\nAt the physical-update level, **{physical.gap_improving_fraction:.2%} improve** and **{physical.gap_deteriorating_fraction:.2%} deteriorate** ({int(physical.valid_observations):,} valid). At the equally weighted entity-mean, relative-family-observation level, **{macro.gap_improving_fraction:.2%} improve** and **{macro.gap_deteriorating_fraction:.2%} deteriorate** ({int(macro.valid_observations):,} valid). The two levels answer different weighting/aggregation questions.\n"
    dataset_rows=summary[(summary.level=='physical')&(summary.scope=='dataset')]
    report+=f"\nAt least one physical gap-deteriorating observation occurs in **{int((dataset_rows.gap_deteriorating_count>0).sum())}/9 families**; occurrence alone does not explain failure.\n"
    for r in eligible.itertuples():
        report+=f"\n**{r.dataset}:** OOG n={r.OOG_trajectories}, NI n={r.NI_trajectories}; median family-macro gap-improving fractions **{r.OOG_gap_improving_fraction_median:.2%} versus {r.NI_gap_improving_fraction_median:.2%}**, difference **{100*r.OOG_minus_NI_gap_improving_fraction_median:+.2f} pp**. The corresponding entity-balanced contrast is **{100*r.OOG_minus_NI_entity_balanced_gap_improving_fraction_median:+.2f} pp**.\n"
    seeds=t['dataset_seed_outcome_gap_comparison.csv']
    for dataset,g in seeds[seeds.eligible].groupby('dataset'):
        delta=g.OOG_minus_NI_gap_improving_fraction_median
        report+=f"\nWith source seed matched, {dataset} has a positive OOG−NI median improving-fraction contrast in **{int((delta>0).sum())}/{len(g)} seeds**, ranging **{100*delta.min():+.2f} to {100*delta.max():+.2f} pp**. These compare optimizer settings under a matched source seed, not independent datasets.\n"
    coefficients=t['continuous_gap_associations_full_grid.csv']
    def coefficient(predictor,scope):
        return coefficients[(coefficients.predictor==predictor)&(coefficients.response=='oracle_delta_auprc')&(coefficients.scope==scope)].rho.iloc[0]
    report+=f"\nThe continuous association is weak: pooled gap-improving-fraction versus Oracle−Frozen AUPRC Spearman **rho={coefficient('gap_improving_fraction','pooled'):.3f}** across 180 cells, and **rho={coefficient('gap_improving_fraction','pooled_within_dataset_ranks'):.3f}** after ranking within dataset before pooled Spearman. The entity-balanced counterpart of the latter is **{coefficient('entity_balanced_gap_improving_fraction','pooled_within_dataset_ranks'):.3f}**. These do not support a strong general within-dataset predictor of continuous detection gain.\n"
    optimizer_contrasts=t['optimizer_gap_comparison.csv']
    reversals=optimizer_contrasts[optimizer_contrasts.OOG_minus_NI_gap_improving_fraction_median<0]
    report+=f"\nOptimizer-stratified pooled contrasts are not uniformly positive: **{len(reversals)} of four reverse direction**. These remain cross-dataset comparisons; no within-dataset×optimizer OOG/NI contrast is identifiable.\n"
    signs=np.sign(eligible.OOG_minus_NI_gap_improving_fraction_median)
    if len(signs) and (signs>0).all():
        report+='\nBoth eligible families show higher OOG median improving fractions under the family-macro definition. This remains descriptive and optimizer-confounded; entity-balanced sensitivity and ranking diagnostics must also be considered.\n'
    else:
        report+='\nThe eligible-family contrasts do **not** consistently show higher OOG median improving fractions. A general positive within-dataset association is therefore unsupported under the primary family-macro definition.\n'
    agreement=t['gap_pair_agreement_full_grid.csv'].query("level=='physical' and scope=='all'")
    both=agreement[agreement.gap_direction.isin(['improving','deteriorating'])&agreement.pair_direction.isin(['improving','deteriorating'])]
    n_both=int(both['count'].sum());n_same=int(both.loc[both.gap_direction==both.pair_direction,'count'].sum())
    unchanged_pairs=int(agreement.loc[(agreement.pair_direction=='unchanged')&(agreement.gap_direction!='undefined'),'count'].sum())
    report+=f"\nStrict pair ordering is unchanged in **{unchanged_pairs:,}/{int(physical.valid_observations):,}** valid physical observations. Among the **{n_both:,}** observations where both gap and ordering change, signs agree in **{n_same:,} ({n_same/n_both:.2%})**. Conditioning on both changing is descriptive, not a substitute denominator for the primary proportions.\n"
    report+=f"\nThe original 60-trajectory result remains exactly **46.3% improving / 49.7% deteriorating / 4.0% unchanged**. Expansion changes the family-summary improving fraction by **{100*macro.gap_improving_fraction-46.3:+.2f} pp** and deteriorating fraction by **{100*macro.gap_deteriorating_fraction-49.7:+.2f} pp**. This reflects added datasets and a declared multi-entity aggregation, not a revision of the original observations.\n"
    report+='''
+## Data coverage, provenance, and units
+
+The clean support is 106 physical entities: Daphnet/GECCO/Genesis/PSM/PUMP/SWAT2015 one each; MSL 25; SMAP 52; SMD 23. Across four optimizers and five seeds, this is **180 logical family cells but 2,120 physical Oracle trajectories**. Each physical trajectory contributes 20 observations, giving **42,400 physical observations**. The requested 3,600 observations are family-level relative-progress summaries, not 3,600 actual individual optimizer steps. `checkpoint_gap_observations_full_grid.csv` retains physical identities; `family_checkpoint_gap_observations_full_grid.csv` contains the 3,600 logical summaries. `trajectory_gap_summary_full_grid.csv` has 180 rows; the separate physical summary has 2,120 rows. These denominators must never be interchanged.
+
+All original 60 trajectories and 1,200 observations are reused without modification. Ten compatible SMD M-1-6 trajectories are also reused; **2,050 missing physical trajectories** were replayed from existing source checkpoints, with no source fitting. The first additional SMD replay was a smoke test and required exact online-score parity before launching four GPU workers. Each subsequent replay checks original Oracle score and final PR-AUC parity; all individual result records include runtime, GPU assignment, source/configuration hashes, and completion status. The execution manifest, source inventory, configuration, code hashes, and original score/label paths are saved in the full-grid execution artifacts. Existing checkpoints contain source weights, not recoverable intermediate Oracle states; replay was therefore required. The authoritative source is the November 13 freeze and corrected common-support outcome table, not older pre-clean audits. Exact outcome assignments are 27 BI / 63 OOG / 90 NI / 0 POG, and physical paired arm deltas are reaggregated to independently verify all 180 mappings.
+
+The original 12-point nonoverlapping AE windows, chronological stream order, normal-point masks, `(point_loss * mask).mean()` over all 12 positions, optimizer hyperparameters, zero weight decay, and scheduled empty-mask steps are preserved. Diagnostic gradients do not enter optimizer gradients. Each entity's observations are at t=floor(j*N/20), j=0..19, bracketing the actual step t→t+1. Primary probes are at most 64 points/class evenly spaced by rank in the suffix strictly after the current window. Labels are used only for offline grouping; no score-based primary selection occurs. Sample indices, class labels, before/after class means, pair fractions, and vector-file paths are retained. All new replays save individual probe-score vectors. Seventy reused trajectories lack these vectors and retain the original code/indices/labels identity-verification limit.
+
+For multi-entity families, at each relative progress j, we first identify entities with both probe classes. We average class means and strict pair fractions equally over that exact entity set, unchanged before/after. G_before and G_after are therefore paired family macro averages; Delta_G is their difference. If no entity has both classes, the family observation is undefined. Counts and identities of excluded observations are preserved; no undefined statistic is zero-filled. All retained entities are evaluated under every configuration. Valid future-class support may differ across j because anomaly timing differs; `physical_entities_valid`, `physical_entities_expected`, and `all_entities_valid` expose it.
+
+Family gap sign is the sign of a macro-average local score response; it is not the fraction of physical updates with that sign. Raw score scales can differ by entity. Accordingly, trajectory summaries also report the equal-entity average and median of per-entity improving fractions, and the tables retain physical proportions. These are prespecified aggregation sensitivity checks. They do not turn physical entities into independent dataset families. Cross-time pools change and intervening updates are unsampled: net cumulative gap change is undefined.
+
+## Dataset-level checkpoint results
+
+Fractions use the explicitly shown valid denominator. These are local mean score-gap changes; larger scores indicate anomalies. Strict signs use >0/<0/=0 without thresholding; tiny changes reflect float32 model evaluation and should not be overinterpreted. CPU arithmetic uses float64 and round-trip parsing of original per-run CSVs. Tolerances apply only to numerical verification, never classification.
+
+'''.replace('\n+','\n')
    report+=tbl(summary[(summary.scope=='dataset')][['level','value','observations','valid_observations','gap_improving_fraction','gap_deteriorating_fraction','gap_unchanged_fraction','mean_Delta_G','median_Delta_G','mean_normal_delta','mean_anomaly_delta']])
    report+='\nScheduled steps versus actual movement / fresh normal fitting (conditioning is supplementary; it does not replace the primary denominator):\n\n'
    report+=tbl(t['full_grid_denominator_sensitivity.csv'][['scope','observations','valid_observations','gap_improving_fraction','gap_deteriorating_fraction','gap_unchanged_fraction','pair_improving_fraction','pair_deteriorating_fraction']])
    report+='''
+Normal/anomaly changes are reported separately because gap improvement can result from an anomaly-score increase, a normal-score decrease, or unequal joint changes. Positive gap change does not imply normal suppression. Full quantiles, absolute median magnitudes, and dataset×optimizer breakdowns are saved in the magnitude and summary CSVs.
+
+## Within-dataset OOG versus NI — primary outcome comparison
+
+Only datasets containing both categories can support this comparison. Other datasets are explicitly marked ineligible; their missing contrasts remain undefined. All comparisons use one row per logical trajectory, not checkpoint-level replication. Statistics below compare median trajectory fractions and median trajectory median changes; full group means, quantiles, and counts are retained in CSVs.
+
+'''.replace('\n+','\n')
    report+=tbl(d[['dataset','OOG_trajectories','NI_trajectories','eligible','OOG_valid_observations','NI_valid_observations','OOG_gap_improving_fraction_median','NI_gap_improving_fraction_median','OOG_minus_NI_gap_improving_fraction_median','OOG_minus_NI_median_Delta_G_median','OOG_minus_NI_median_pair_fraction_delta_median','OOG_minus_NI_entity_balanced_gap_improving_fraction_median']])
    report+=f'\nThere are **{len(eligible)} eligible families**. The signs and magnitudes above, including zero or reversed contrasts, determine the conclusion; no consistency is assumed. Dataset×optimizer and dataset×seed tables identify remaining configuration/seed overlap. Without both outcomes inside a stratum, a within-stratum contrast is undefined. **In both eligible families (PSM and SMD), all five NI cells are SGD 1e-4, and the other optimizer settings provide OOG/BI. No dataset×optimizer stratum contains both OOG and NI.** Thus this expansion removes dataset confounding for these comparisons, but not optimizer/configuration confounding. The seed-stratified comparisons keep source seed matched while comparing optimizer settings; they do not identify an optimizer-independent mechanism.\n'
    report+='\n## Pooled outcome distributions\n\n'
    report+=tbl(t['outcome_gap_comparison_full_grid.csv'][['final_outcome','trajectories','valid_observations','gap_improving_fraction_median','gap_deteriorating_fraction_median','mean_Delta_G_mean','median_Delta_G_median','pair_improving_fraction_median','mean_pair_fraction_delta_mean','entity_balanced_gap_improving_fraction_median']])
    report+='\n## Optimizer heterogeneity\n\nThese optimizer-stratified contrasts still pool datasets. Dataset×optimizer contrasts, rather than these pooled contrasts alone, are required for interpreting reversals.\n\n'
    report+=tbl(t['optimizer_gap_comparison.csv'][['optimizer','OOG_trajectories','NI_trajectories','OOG_gap_improving_fraction_median','NI_gap_improving_fraction_median','OOG_minus_NI_gap_improving_fraction_median','OOG_minus_NI_median_pair_fraction_delta_median']])
    report+='\nEligible dataset×optimizer strata:\n\n'
    s=t['dataset_optimizer_outcome_gap_comparison.csv']
    report+=tbl(s[s.eligible][['dataset','optimizer','OOG_trajectories','NI_trajectories','OOG_minus_NI_gap_improving_fraction_median','OOG_minus_NI_median_pair_fraction_delta_median']])
    report+='\n## Continuous detection associations and dataset clustering\n\nSpearman coefficients are descriptive. Per-family coefficients involve 20 repeated optimizer×seed settings; dataset-median coefficients involve nine dataset clusters. Pooled within-dataset rank association ranks each variable within family, then computes Spearman after pooling those ranks, matching the frozen downstream-refresh implementation; this includes a second ranking step for pooled ties and differs from ordinary pooled raw-value Spearman. No significance tests assuming 180 independent datasets or 42,400 independent observations are reported. Leave-one-dataset-out coefficients are saved to expose between-dataset dependence.\n\n'
    c=t['continuous_gap_associations_full_grid.csv']
    report+=tbl(c[(c.response=='oracle_delta_auprc')&c.predictor.isin(['gap_improving_fraction','pair_improving_fraction','mean_Delta_G','entity_balanced_gap_improving_fraction','selective_compression','net_pair_gain','clean_selective_compression','clean_net_pair_gain'])&c.scope.isin(['pooled','dataset_medians','pooled_within_dataset_ranks','within_dataset'])])
    report+='\n## Mean-gap versus strict pairwise ordering\n\nStrict ordering is P(score_anomaly > score_normal), with ties receiving no credit. The following table includes improved/deteriorated/unchanged/undefined combinations. A gap shift need not move even one pair across an ordering boundary, and opposite pair flips can cancel. Mean-gap direction therefore cannot be substituted for local ranking direction or full-stream AUPRC.\n\n'
    report+=tbl(t['gap_pair_agreement_full_grid.csv'].query("scope=='all'"))
    report+='\nSeparate normal/anomaly response directions on valid family observations:\n\n'
    report+=tbl(t['normal_anomaly_response_directions_full_grid.csv'].query("level=='family_macro' and dataset=='all'"))
    report+='\n## Tail-focused supplementary diagnostic\n\nThe full-prefix Frozen q85 of normal scores and q30 of anomaly scores define the original hard-normal and boundary-anomaly sets, including threshold ties. We intersect those sets with each existing primary probe group, preserving before/after membership. Empty intersections remain undefined and are never replaced. The reused 70 trajectories lack individual primary score vectors, so this analysis is unavailable for them; the original three families therefore have no tail results, and two SMD optimizer settings have partial entity support. Tail summaries are supplementary, with their own exact denominators. This is local mean response on the available probe intersection, whereas existing SC uses median full-stream Frozen-to-Oracle score drops and NPG full-stream tail pair transitions. The estimands are different.\n\n'
    report+=tbl(t['tail_gap_summary_full_grid.csv'][['level','dataset','observations','valid_observations','gap_improving_fraction','gap_deteriorating_fraction','mean_Delta_G','pair_improving_fraction','pair_deteriorating_fraction']])
    report+='\n## Predetermined probe sensitivity\n\nExactly one alternative was evaluated during each new replay: select up to 64 midpoints of equal-count time-rank strata within the same future class population, using floor((rank+0.5)*eligible_count/k). The primary design instead includes endpoint-spaced ranks. Neither uses anomaly scores. No subset was selected after observing results. Comparison is restricted to the same newly replayed physical trajectories, excluding all 70 reused runs without vectors. Primary and alternative outcomes and configurations are unchanged; this sensitivity does not estimate uncertainty over all possible probe subsets.\n\n'
    report+=tbl(t['probe_sampling_sensitivity_full_grid.csv'][['design','level','dataset','valid_observations','gap_improving_fraction','gap_deteriorating_fraction','pair_improving_fraction','pair_deteriorating_fraction']])
    sc=t['probe_sensitivity_dataset_outcome_comparison.csv']
    report+='\nOutcome contrasts under both predetermined samplings, restricted to the identical vector-available physical support (partial SMD support for two optimizer settings):\n\n'
    report+=tbl(sc[sc.eligible][['dataset','design','OOG_trajectories','NI_trajectories','OOG_minus_NI_gap_improving_fraction_median','OOG_minus_NI_mean_pair_fraction_delta_median']])
    report+='\n## Reconciliation, counterexamples, and limitations\n\nThe frozen corrected full-grid SC/NPG table is joined one-to-one, with all paired AUPRC changes and outcome labels checked. Its predictors use finite per-cell entity support (MSL 25–27, SMAP 52–55, SMD 23–28), whereas detection outcomes use the fixed clean support (25/52/23). This historical support difference is preserved and qualified. We additionally recompute SC/NPG on exactly the 106 retained entities from existing Frozen and original online Oracle score arrays, using the same q85/q30 tail thresholds, per-entity median score drops, exact strict pair-count differences, and family median aggregation. These are new CPU-only reconciliation tables named `clean_support_score_shift_physical.csv` and `clean_support_score_shift_cells.csv`; the frozen evidence is not modified. Correlations labelled clean_selective_compression and clean_net_pair_gain use this matched support. Existing objective reduction is positive in all 180 cells. Correlations with local metrics are saved in `continuous_gap_associations_full_grid.csv`; their magnitude does not imply the same estimand or a causal pathway. The old pre-clean score-shift audit contains different counts and is background only.\n\nDaphnet remains the direct all-normal update counterexample: seed42/AdamW1e-4/window726 lowers current normal reconstruction while reducing future gap and strict ordering. Successful PUMP/AdamW3e-4 already exhibits median gap deterioration at 55% of observations. Existing SMD M-1-6 evidence finds more locally harmful normal-fitting/ranking steps in successful AdamW than failed SGD, and final fixed-state ranking can improve despite online failure. These examples preclude equating gap compression, ranking harm, and final outcome. The expanded tables must be read with these counterexamples, not used to suppress them.\n\nPhysical one-class late-future observations are missing by design, not repaired. Unequal class support, raw entity/dataset score scales, sparse observation cadence, repeated seeds/optimizers, and availability-limited secondary analyses restrict interpretation. Some NI may reflect limited adaptation demand. Local scores are evaluated on fixed states while online AUPRC combines evolving states over a full stream. Neither the local gap nor strict pair fraction decomposes global AUPRC. No universal mechanism, representation-collapse claim, or causal link to NI/OOG is identified. Generality is limited to the evaluated flattened-window reconstruction AE and existing benchmark protocol.\n\n## Rebuttal-ready claim boundaries\n\nThe tables quantify how often normal-only Oracle steps improve or degrade local normal–anomaly separation throughout the evaluated grid. Whether the association with outcomes survives within datasets is answered by the eligible-family contrasts and configuration sensitivity, including null or opposing results. Only consistent observed directions can support a qualified descriptive association. Even then, local gap improvement is neither a sufficient condition for global AUPRC improvement nor evidence that its deterioration causes NI. The paper’s three original insights are unchanged.\n\n## Reproduction and independent verification\n\nAll outputs are under `rebuttal/experiments/future_probe_mechanism/checkpoint_gap/` with new names. Original 60-run tables and report are preserved. `full_grid_analysis_artifacts.json` enumerates generated table paths, row counts, and columns. `full_grid_parity_provenance.csv` enumerates every physical result path. GPU worker logs and individual vector files remain in the same evidence directory.\n\nRegenerate tables and this report from completed saved artifacts (CPU only):\n\n```bash\npython rebuttal/experiments/future_probe_mechanism/analyze_full_grid.py\npython rebuttal/experiments/future_probe_mechanism/verify_full_grid.py\n```\n\nOriginal expansion execution commands were:\n\n```bash\npython rebuttal/experiments/future_probe_mechanism/full_grid.py prepare\npython rebuttal/experiments/future_probe_mechanism/full_grid.py smoke\npython rebuttal/experiments/future_probe_mechanism/full_grid.py launch\npython rebuttal/experiments/future_probe_mechanism/full_grid.py status\n```\n\nPreparation/launch fail rather than overwrite existing results; these GPU commands are historical provenance, not instructions to rerun completed trajectories. The independent verifier uses a separate arithmetic implementation, rechecks saved vectors, trajectory/outcome mapping, original evidence preservation, and a fresh CPU reproduction. Its result is saved separately; a report generated before that command is not itself evidence of a verification pass.\n'
    report+='\n## Descriptive reconciliation details\n\nThe following correlations compare different estimands; they do not test equality or causality.\n\n'
    report+=tbl(c[c.response.isin(['selective_compression','net_pair_gain','clean_selective_compression','clean_net_pair_gain'])&((c.scope=='pooled')|((c.scope=='within_dataset')&c.value.isin(['PSM','SMD'])))])
    report+='\nIllustrative extreme cases selected after observing the completed tables: per family, the NI cell with the largest improving fraction and OOG cell with the smallest improving fraction (ties ordered by optimizer, seed). This explicit selection is for counterexamples only, not frequency estimation.\n\n'
    report+=tbl(t['full_grid_descriptive_counterexamples.csv'][['dataset','optimizer','seed','final_outcome','gap_improving_fraction','gap_deteriorating_fraction','pair_improving_fraction','oracle_delta_auprc','example_rule']])
    report=report.replace('\n+','\n')
    (output/'P3_CHECKPOINT_SCORE_GAP_FULL_GRID_AUDIT.md').write_text(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    main(parser.parse_args().output)
