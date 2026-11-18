#!/usr/bin/env python3
"""Separate arithmetic verifier for completed full-grid evidence; CPU only."""
import argparse
from collections import defaultdict, Counter
import csv
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import numpy as np

getcontext().prec=50
D=Decimal
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'rebuttal/experiments/future_probe_mechanism/checkpoint_gap'

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def physical_key(r):return (r['dataset'],r['entity'],r['optimizer'],str(r['seed']))
def cell_key(r):return (r['dataset'],r.get('optimizer',r.get('optimizer_id')),str(r['seed']))

def close(a,b,atol=2e-12,rtol=2e-12):
    a,b=float(a),float(b)
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*abs(b),(a,b)

def corr(x,y):
    def ranks(a):
        result=np.empty(len(a),float)
        order=sorted(range(len(a)),key=lambda i:a[i]);i=0
        while i<len(a):
            j=i+1
            while j<len(a) and a[order[j]]==a[order[i]]:j+=1
            for k in range(i,j):result[order[k]]=(i+j+1)/2
            i=j
        return result
    if len(set(x))<2 or len(set(y))<2:return None
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def vector_summary(v):
    result={}
    for name in ['normal','anomaly']:
        result[name+'_points']=len(v[name+'_indices'])
        result[name+'_indices']=json.dumps(v[name+'_indices'])
        for phase in ['before','after']:
            values=v[name+'_'+phase]
            result[name+'_'+phase]=math.fsum(values)/len(values) if values else None
    if result['normal_points'] and result['anomaly_points']:
        import bisect
        for phase in ['before','after']:
            normals=sorted(v['normal_'+phase]);anomalies=v['anomaly_'+phase]
            result['pair_fraction_'+phase]=sum(bisect.bisect_left(normals,a) for a in anomalies)/(len(normals)*len(anomalies))
    else:
        result['pair_fraction_before']=result['pair_fraction_after']=None
    return result

def summaries(values,pairs,normals,anomalies):
    n=len(values)
    output={'valid_observations':n}
    for name,v in [('gap',values),('pair',pairs)]:
        for label,condition in [('improving',lambda x:x>0),('deteriorating',lambda x:x<0),('unchanged',lambda x:x==0)]:
            count=sum(condition(x) for x in v)
            output[name+'_'+label+'_count']=count
            output[name+'_'+label+'_fraction']=count/n if n else None
        metric='Delta_G' if name=='gap' else 'pair_fraction_delta'
        output['mean_'+metric]=statistics.mean(v) if n else None
        output['median_'+metric]=statistics.median(v) if n else None
    for name,v in [('normal',normals),('anomaly',anomalies)]:
        output['mean_'+name+'_delta']=statistics.mean(v) if n else None
        output['median_'+name+'_delta']=statistics.median(v) if n else None
    return output

def main(skip_reproduction):
    plan=json.loads((OUT/'full_grid_execution_plan.json').read_text())
    config=json.dumps(plan['configuration'],sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(config).hexdigest()==plan['configuration_sha256']
    for filename,expected in plan['code_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected
    source_files={t['base']['checkpoint']:t['base']['source_checkpoint_sha256'] for t in plan['tasks']}
    assert len(source_files)==530
    for filename,expected in source_files.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected
    artifact_manifest=json.loads((OUT/'full_grid_analysis_artifacts.json').read_text())
    phys=rows(OUT/'checkpoint_gap_observations_full_grid.csv')
    fam=rows(OUT/'family_checkpoint_gap_observations_full_grid.csv')
    trajectories=rows(OUT/'trajectory_gap_summary_full_grid.csv')
    ptrajectories=rows(OUT/'physical_trajectory_gap_summary_full_grid.csv')
    assert len(phys)==42400 and len(fam)==3600 and len(trajectories)==180 and len(ptrajectories)==2120
    assert len({physical_key(r)+(r['observation_index'],) for r in phys})==42400
    assert len({cell_key(r)+(r['observation_index'],) for r in fam})==3600
    by_physical=defaultdict(list);by_family=defaultdict(list);by_cell_obs=defaultdict(list)
    raw_original={};vector_observations=0;vector_values=0;exact_score_runs=0;new_runs=0
    secondary_expected={};tail_expected={};clean_expected={}
    score_errors=[];metric_errors=[]
    for task in plan['tasks']:
        b,opt=task['base'],task['optimizer']
        folder=Path(task['reuse_directory']) if 'reuse_directory' in task else OUT/'full_grid_runs'/(b['subset'] or '__root__')/b['dataset']/opt/f"seed{b['seed']}"
        run=json.loads((folder/'result.json').read_text())
        assert run['complete'] and run['source_checkpoint_sha256']==b['source_checkpoint_sha256']
        score_errors.append(run['max_score_parity_error']);metric_errors.append(run['auprc_parity_error'])
        labels=np.load(folder/'true.npy',allow_pickle=False).reshape(-1)
        original_labels=np.load(b['references'][opt+':oracle']['label_path'],allow_pickle=False).reshape(-1)
        np.testing.assert_array_equal(labels,original_labels)
        scores=np.load(folder/'test_energy.npy',allow_pickle=False).reshape(-1)
        refscores=np.load(b['references'][opt+':oracle']['score_path'],allow_pickle=False).reshape(-1)
        if np.array_equal(scores,refscores):exact_score_runs+=1
        np.testing.assert_allclose(scores,refscores,rtol=1e-5,atol=1e-6)
        frozen=np.load(b['references']['shared:frozen']['score_path'],allow_pickle=False).reshape(-1).astype(np.float64)
        hard=np.quantile(frozen[labels==0],.85);boundary=np.quantile(frozen[labels==1],.30)
        h=(labels==0)&(frozen>=hard);a=(labels==1)&(frozen<=boundary)
        drop=frozen-refscores.astype(np.float64)
        normal_drop=statistics.median(drop[h].tolist());anomaly_drop=statistics.median(drop[a].tolist())
        def ordering_count(normal,anomaly):
            return int((len(anomaly)-np.searchsorted(np.sort(anomaly),normal,side='right')).sum())
        clean_expected[(b['dataset'],b['subset'] or '__root__',opt,str(b['seed']))]={
            'clean_hard_normal_drop':normal_drop,'clean_boundary_anomaly_drop':anomaly_drop,
            'clean_selective_compression':normal_drop-anomaly_drop,
            'clean_net_pair_gain':(ordering_count(refscores[h],refscores[a])-ordering_count(frozen[h],frozen[a]))/int(h.sum()*a.sum())}
        raw=rows(folder/'probes.csv')
        n=len(labels)//12
        assert len(raw)==20
        for j,r in enumerate(raw):
            assert int(r['window'])==j*n//20
            for name,label in [('normal',0),('anomaly',1)]:
                indices=json.loads(r[name+'_indices'])
                start=(int(r['window'])+1)*12
                eligible=np.where(labels[start:]==label)[0]+start;k=min(64,len(eligible))
                if k:
                    chosen=[int(eligible[int(x)]) for x in np.linspace(0,len(eligible)-1,k,dtype=int)]
                else:chosen=[]
                assert chosen==indices and len(indices)==int(r[name+'_points'])
            key=(b['dataset'],b['subset'] or '__root__',opt,str(b['seed']),str(j))
            assert key not in raw_original
            raw_original[key]=r
        vector_file=folder/'probe_vectors.jsonl'
        if 'reuse_directory' not in task:
            new_runs+=1;assert vector_file.is_file()
            frozen=np.load(b['references']['shared:frozen']['score_path'],allow_pickle=False).reshape(-1).astype(np.float64)
            hard=np.quantile(frozen[labels==0],.85);boundary=np.quantile(frozen[labels==1],.30)
            vectors=[json.loads(x) for x in vector_file.read_text().splitlines()]
            assert len(vectors)==40 and Counter(v['design'] for v in vectors)=={'primary':20,'midpoint_sensitivity':20}
            for v in vectors:
                r=next(r for r in raw if int(r['window'])==v['window'])
                vector_observations+=1
                values={}
                for name,label in [('normal',0),('anomaly',1)]:
                    ids=v[name+'_indices'];assert v[name+'_labels']==[label]*len(ids)
                    assert np.all(labels[np.array(ids,dtype=int)]==label)
                    start=(v['window']+1)*12
                    eligible=np.where(labels[start:]==label)[0]+start;k=min(64,len(eligible))
                    expected=eligible[np.linspace(0,len(eligible)-1,k,dtype=int)] if v['design']=='primary' else eligible[np.floor((np.arange(k)+.5)*len(eligible)/k).astype(int)] if k else []
                    assert list(expected)==ids
                    if v['design']=='primary':assert ids==json.loads(r[name+'_indices'])
                    values[name]={}
                    for phase in ['before','after']:
                        a=v[name+'_'+phase];vector_values+=len(a)
                        assert len(a)==len(ids) and all(math.isfinite(x) for x in a)
                        values[name][phase]=a
                        if v['design']=='primary':
                            if a:close(r[name+'_'+phase],math.fsum(a)/len(a),atol=1e-12,rtol=1e-15)
                            else:assert r[name+'_'+phase]==''
                if v['design']=='primary' and values['normal']['before'] and values['anomaly']['before']:
                    for phase in ['before','after']:
                        normals=sorted(values['normal'][phase]);anomalies=values['anomaly'][phase]
                        # Independently count strictly smaller normals with bisect_left.
                        import bisect
                        correct=sum(bisect.bisect_left(normals,a) for a in anomalies)
                        exact_fraction=correct/(len(normals)*len(anomalies))
                        # Recorder averages per-anomaly counts then divides by
                        # normals; one total-count division can differ by 1 ulp.
                        close(r['pair_fraction_'+phase],exact_fraction,atol=2*math.ulp(exact_fraction),rtol=0)
                j=next(j for j,x in enumerate(raw) if int(x['window'])==v['window'])
                k=(b['dataset'],b['subset'] or '__root__',opt,str(b['seed']),str(j))
                if v['design']=='midpoint_sensitivity':secondary_expected[k]=vector_summary(v)
                else:
                    restricted={}
                    for name in ['normal','anomaly']:
                        ids=v[name+'_indices']
                        keep=[i for i,index in enumerate(ids) if frozen[index]>=hard] if name=='normal' else [i for i,index in enumerate(ids) if frozen[index]<=boundary]
                        restricted[name+'_indices']=[ids[i] for i in keep]
                        for phase in ['before','after']:restricted[name+'_'+phase]=[v[name+'_'+phase][i] for i in keep]
                    tail_expected[k]=vector_summary(restricted)
                    tail_expected[k].update(hard_threshold_frozen_q85=hard,boundary_threshold_frozen_q30=boundary)
    assert len(raw_original)==42400 and new_runs==2050
    clean_physical=rows(OUT/'clean_support_score_shift_physical.csv')
    assert len(clean_physical)==2120
    for r in clean_physical:
        for col,value in clean_expected[physical_key(r)].items():close(r[col],value)
    for r in rows(OUT/'clean_support_score_shift_cells.csv'):
        k=cell_key(r)
        inputs=[v for pk,v in clean_expected.items() if (pk[0],pk[2],pk[3])==k]
        for col in inputs[0]:close(r[col],statistics.median(v[col] for v in inputs))
    assert len(secondary_expected)==len(tail_expected)==41000
    for filename,mapping in [('probe_sensitivity_observations_full_grid.csv',secondary_expected),('tail_gap_observations_full_grid.csv',tail_expected)]:
        actual=rows(OUT/filename)
        assert len(actual)==41000
        assert len({physical_key(r)+(r['observation_index'],) for r in actual})==41000
        for r in actual:
            expected=mapping[physical_key(r)+(r['observation_index'],)]
            for col,value in expected.items():
                if value is None:assert r[col]==''
                elif col.endswith('_indices'):assert r[col]==value
                else:close(r[col],value,rtol=1e-15)
    sign_counts=Counter()
    for r in phys:
        k=physical_key(r)
        original=raw_original[k+(r['observation_index'],)]
        assert int(r['window'])==int(original['window'])
        for c in ['normal_indices','anomaly_indices','selected_points','normal_points','anomaly_points']:assert r[c]==original[c]
        valid=int(original['normal_points'])>0 and int(original['anomaly_points'])>0
        if valid:
            nb,na,ab,aa=[D(original[c]) for c in ['normal_before','normal_after','anomaly_before','anomaly_after']]
            delta=(aa-na)-(ab-nb)
            pair_delta=D(original['pair_fraction_after'])-D(original['pair_fraction_before'])
            ulp_bound=16*sum(math.ulp(float(original[c])) for c in ['normal_before','normal_after','anomaly_before','anomaly_after'])
            for c,x in [('G_before',ab-nb),('G_after',aa-na),('Delta_G',delta),('normal_delta',na-nb),('anomaly_delta',aa-ab)]:close(r[c],x,atol=ulp_bound,rtol=0)
            close(r['pair_fraction_delta'],pair_delta)
            classification='improving' if delta>0 else 'deteriorating' if delta<0 else 'unchanged'
            assert classification==r['gap_direction']
            sign_counts[classification]+=1
        else:
            assert r['Delta_G']==r['pair_fraction_delta']=='' and r['gap_direction']=='undefined'
            sign_counts['undefined']+=1
        by_physical[k].append(r);by_cell_obs[cell_key(r)+(r['observation_index'],)].append(r)
    assert sum(sign_counts[k] for k in ['improving','deteriorating','unchanged'])==33440 and sign_counts['undefined']==8960
    family_signs=Counter()
    for r in fam:
        k=cell_key(r)
        sources=by_cell_obs[k+(r['observation_index'],)]
        valid=[v for v in sources if v['Delta_G']!='']
        assert int(r['physical_entities_expected'])==len(sources) and int(r['physical_entities_valid'])==len(valid)
        assert int(r['normal_points'])==sum(int(v['normal_points']) for v in valid)
        assert int(r['anomaly_points'])==sum(int(v['anomaly_points']) for v in valid)
        if valid:
            # Arithmetic means recomputed with fsum, independently of pandas.
            for col in ['normal_before','normal_after','anomaly_before','anomaly_after','pair_fraction_before','pair_fraction_after']:
                close(r[col],math.fsum(float(v[col]) for v in valid)/len(valid),rtol=1e-15)
            close(r['Delta_G'],(float(r['anomaly_after'])-float(r['normal_after']))-(float(r['anomaly_before'])-float(r['normal_before'])),rtol=0,atol=0)
            close(r['pair_fraction_delta'],float(r['pair_fraction_after'])-float(r['pair_fraction_before']),rtol=0,atol=0)
            d=float(r['Delta_G']);classification='improving' if d>0 else 'deteriorating' if d<0 else 'unchanged'
        else:
            assert r['Delta_G']==r['G_before']==r['G_after']==r['pair_fraction_delta']==''
            classification='undefined'
        assert r['gap_direction']==classification
        family_signs[classification]+=1
        by_family[k].append(r)
    for table,mapping in [(trajectories,by_family),(ptrajectories,by_physical)]:
        for r in table:
            k=cell_key(r) if table is trajectories else physical_key(r)
            g=mapping[k];valid=[x for x in g if x['Delta_G']!='']
            stats=summaries(*[[float(x[c]) for x in valid] for c in ['Delta_G','pair_fraction_delta','normal_delta','anomaly_delta']])
            assert int(r['observations'])==20 and int(r['valid_observations'])==len(valid)
            for c,v in stats.items():
                if v is None:assert r[c]==''
                else:close(r[c],v)
    def scope_rows(table,scope,value):
        if scope=='all':return table
        if scope=='dataset':return [r for r in table if r['dataset']==value]
        if scope=='optimizer':return [r for r in table if r['optimizer']==value]
        dataset,optimizer=value.split('/')
        return [r for r in table if r['dataset']==dataset and r['optimizer']==optimizer]
    for r in rows(OUT/'checkpoint_gap_summary_full_grid.csv'):
        table=phys if r['level']=='physical' else fam
        group=scope_rows(table,r['scope'],r['value']);valid=[x for x in group if x['Delta_G']!='']
        expected=summaries(*[[float(x[c]) for x in valid] for c in ['Delta_G','pair_fraction_delta','normal_delta','anomaly_delta']])
        assert int(r['observations'])==len(group) and int(r['undefined_observations'])==len(group)-len(valid)
        for col,value in expected.items():
            if value is None:assert r[col]==''
            else:close(r[col],value)
    for r in rows(OUT/'gap_pair_agreement_full_grid.csv'):
        table=phys if r['level']=='physical' else fam
        group=scope_rows(table,r['scope'],r['value'])
        count=sum(x['gap_direction']==r['gap_direction'] and x['pair_direction']==r['pair_direction'] for x in group)
        assert int(r['count'])==count and int(r['all_observations'])==len(group)
    for r in rows(OUT/'gap_magnitude_distribution_full_grid.csv'):
        table=phys if r['level']=='physical' else fam
        group=scope_rows(table,r['scope'],r['value']);values=[float(x[r['metric']]) for x in group if x['Delta_G']!='']
        assert int(r['n'])==len(values)
        for col,value in r.items():
            if col.startswith('q'):close(value,np.quantile(values,float(col[1:])))
        close(r['mean'],statistics.mean(values));close(r['median_abs'],statistics.median(abs(x) for x in values))
    core={cell_key(r):r for r in rows(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv')}
    assert len(core)==180
    for r in trajectories:
        k=cell_key(r);c=core[k]
        p,o=D(c['practical_delta_auprc'])>D('.005'),D(c['oracle_delta_auprc'])>D('.005')
        cat='BI' if p and o else 'OOG' if o else 'POG' if p else 'NI'
        assert r['final_outcome']==c['category']==cat
        assert r['net_gap_change']==''
        close(r['oracle_delta_auprc'],c['oracle_delta_auprc'],atol=0,rtol=0)
        entity_runs=[x for x in ptrajectories if cell_key(x)==k]
        assert len(entity_runs)==int(r['physical_trajectories'])==int(r['retained_entity_count'])
        rates=[float(x['gap_improving_fraction']) for x in entity_runs if x['gap_improving_fraction']!='']
        close(r['entity_balanced_gap_improving_fraction'],statistics.mean(rates))
    trajectory_map={cell_key(r):r for r in trajectories}
    for r in phys:
        assert r['final_outcome']==trajectory_map[cell_key(r)]['final_outcome']
    group_comparisons=[('dataset_outcome_gap_comparison.csv',['dataset']),('optimizer_gap_comparison.csv',['optimizer']),('dataset_optimizer_outcome_gap_comparison.csv',['dataset','optimizer']),('dataset_seed_outcome_gap_comparison.csv',['dataset','seed'])]
    for filename,keys in group_comparisons:
        for row in rows(OUT/filename):
            subset=[r for r in trajectories if all(r[k]==row[k] for k in keys)]
            a=[r for r in subset if r['final_outcome']=='OOG'];b=[r for r in subset if r['final_outcome']=='NI']
            assert int(row['OOG_trajectories'])==len(a) and int(row['NI_trajectories'])==len(b)
            assert row['eligible']==str(bool(a and b))
            for name,group in [('OOG',a),('NI',b)]:assert int(row[name+'_valid_observations'])==sum(int(r['valid_observations']) for r in group)
            for column,value in row.items():
                if not column.startswith('OOG_minus_NI_'):continue
                suffix=column[len('OOG_minus_NI_'):];metric,operation=suffix.rsplit('_',1)
                function={'mean':statistics.mean,'median':statistics.median}[operation]
                if a and b:close(value,function([float(r[metric]) for r in a])-function([float(r[metric]) for r in b]))
                else:assert value==''
    for row in rows(OUT/'continuous_gap_associations_full_grid.csv'):
        if row['response']!='oracle_delta_auprc' or row['scope'] not in ['pooled','within_dataset','dataset_medians']:continue
        if row['predictor'] not in trajectories[0]:continue
        subset=[r for r in trajectories if row['scope']!='within_dataset' or r['dataset']==row['value']]
        if row['scope']=='dataset_medians':
            groups=defaultdict(list)
            for r in subset:groups[r['dataset']].append(r)
            x=[statistics.median(float(r[row['predictor']]) for r in g) for g in groups.values()]
            y=[statistics.median(float(r['oracle_delta_auprc']) for r in g) for g in groups.values()]
        else:
            x=[float(r[row['predictor']]) for r in subset];y=[float(r['oracle_delta_auprc']) for r in subset]
        expected=corr(x,y)
        if expected is None:assert row['rho']==''
        else:close(row['rho'],expected)
    # Preservation check against previously frozen three-family outputs.
    original=rows(OUT/'checkpoint_gap_observations.csv')
    current={cell_key(r)+(r['window'],):r for r in phys if r['dataset'] in {'Daphnet','GECCO','PUMP'}}
    assert len(current)==len(original)==1200
    for r in original:
        now=current[cell_key(r)+(r['window'],)]
        for c in ['normal_indices','anomaly_indices','normal_before','normal_after','anomaly_before','anomaly_after','G_before','G_after','Delta_G','pair_fraction_delta']:
            assert now[c]==r[c],(c,now[c],r[c])
    reproduction=[]
    if not skip_reproduction:
        with tempfile.TemporaryDirectory(prefix='p3_full_grid_cpu_') as tmp:
            subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/future_probe_mechanism/analyze_full_grid.py'),'--output',tmp],check=True)
            for name in list(artifact_manifest)+['P3_CHECKPOINT_SCORE_GAP_FULL_GRID_AUDIT.md','full_grid_analysis_artifacts.json']:
                assert (OUT/name).read_bytes()==(Path(tmp)/name).read_bytes(),name
                reproduction.append(name)
    summary={'status':'PASS','logical_cells':180,'physical_trajectories':2120,'family_observations':3600,'physical_observations':42400,'physical_decimal_sign_counts':dict(sign_counts),'family_macro_sign_counts':dict(family_signs),'new_replays':new_runs,'reused_trajectories':70,'reused_original_observations_unchanged':1200,'exact_online_score_parity_runs':exact_score_runs,'max_score_error':max(score_errors),'max_auprc_error':max(metric_errors),'vector_observations_checked':vector_observations,'individual_score_values_checked':vector_values,'byte_reproduced_artifacts':sorted(reproduction),'independent_implementation':'csv/Decimal/math.fsum/statistics plus independent strict pair counts and rank correlations','limitations':'Old 70 primary point vectors unavailable; family statistics are relative-progress macro summaries, not individual updates; no causal or independent-dataset significance claim.'}
    (OUT/'full_grid_independent_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skip-reproduction',action='store_true');main(p.parse_args().skip_reproduction)
