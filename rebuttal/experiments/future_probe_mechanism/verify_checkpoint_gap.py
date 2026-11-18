#!/usr/bin/env python3
"""Independent Decimal-based checks of the frozen gap evidence; no model imports."""
import csv
from decimal import Decimal, getcontext
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

getcontext().prec = 50
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'rebuttal/experiments/future_probe_mechanism'
OUT = ROOT/'rebuttal/experiments/future_probe_mechanism/checkpoint_gap'
D = Decimal

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def key(r):
    return r['dataset'],r.get('optimizer',r.get('optimizer_id')),r['seed']

def close(actual, expected):
    assert actual != '', (actual,expected)
    assert abs(D(actual)-expected) <= D('2e-12')+abs(expected)*D('1e-12'), (actual,expected)

def stats(values):
    n=D(len(values))
    return {'improving_count':sum(x>0 for x in values),'deteriorating_count':sum(x<0 for x in values),'unchanged_count':sum(x==0 for x in values),
            'improving_fraction':D(sum(x>0 for x in values))/n,'deteriorating_fraction':D(sum(x<0 for x in values))/n,'unchanged_fraction':D(sum(x==0 for x in values))/n,
            'mean_Delta_G':statistics.mean(values),'median_Delta_G':statistics.median(values)}

def main():
    plan=json.loads((SOURCE/'plan.json').read_text())
    expected={}
    by_run={}
    for task in plan['tasks']:
        b,opt=task['base'],task['optimizer']
        folder=SOURCE/'runs'/b['dataset']/opt/f"seed{b['seed']}"
        f=rows(folder/'probes.csv')
        runkey=(b['dataset'],opt,str(b['seed']))
        assert runkey not in by_run
        by_run[runkey]=[]
        for r in f:
            k=key(r)+(r['window'],)
            assert k not in expected
            valid=int(r['normal_points'])>0 and int(r['anomaly_points'])>0
            if valid:
                before=D(r['anomaly_before'])-D(r['normal_before'])
                after=D(r['anomaly_after'])-D(r['normal_after'])
                delta=after-before
                by_run[runkey].append(delta)
            else:
                before=after=delta=None
            expected[k]=(r,valid,before,after,delta)
    assert len(expected)==1200 and len(by_run)==60 and sum(map(len,by_run.values()))==1000
    observed=rows(OUT/'checkpoint_gap_observations.csv')
    assert len(observed)==1200
    assert len({key(r)+(r['window'],) for r in observed})==1200
    sign_counts={'improving':0,'deteriorating':0,'unchanged':0,'undefined':0}
    for r in observed:
        raw,valid,before,after,delta=expected[key(r)+(r['window'],)]
        for c in ['normal_indices','anomaly_indices','normal_points','anomaly_points','selected_points']:
            assert r[c]==raw[c]
        for c in ['normal_before','normal_after','anomaly_before','anomaly_after','pair_fraction_before','pair_fraction_after']:
            if raw[c]=='':
                assert r[c]==''
            else:
                close(r[c],D(raw[c]))
        if valid:
            for c,x in [('G_before',before),('G_after',after),('Delta_G',delta)]: close(r[c],x)
            classification='improving' if delta>0 else 'deteriorating' if delta<0 else 'unchanged'
            assert r['validity_status']=='valid_two_class'
        else:
            assert r['G_before']==r['G_after']==r['Delta_G']==''
            classification='undefined'
            assert r['validity_status']=='undefined_single_class'
        assert classification==r['gap_change_class']
        sign_counts[classification]+=1
    core={key(r):r for r in rows(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv')}
    category_counts={}
    for r in core.values():
        p,o=D(r['practical_delta_auprc'])>D('.005'),D(r['oracle_delta_auprc'])>D('.005')
        category='BI' if p and o else 'OOG' if o else 'POG' if p else 'NI'
        assert category==r['category']
        category_counts[category]=category_counts.get(category,0)+1
    assert category_counts=={'BI':27,'OOG':63,'NI':90}
    trajectory=rows(OUT/'trajectory_gap_summary.csv')
    assert len(trajectory)==60 and len({key(r) for r in trajectory})==60
    derived={}
    for r in trajectory:
        k=key(r)
        values=by_run[k]
        s=stats(values)
        assert int(r['observations'])==20 and int(r['valid_observations'])==len(values)
        assert int(r['invalid_observations'])==20-len(values)
        for c,v in s.items(): close(r[c],D(v))
        assert r['net_gap_change']==''
        assert r['final_outcome']==core[k]['category']
        for c in ['oracle_delta_auprc','practical_delta_auprc']: close(r[c],D(core[k][c]))
        rawpairs=[D(v[0]['pair_fraction_after'])-D(v[0]['pair_fraction_before']) for rk,v in expected.items() if rk[:3]==k and v[1]]
        ps=stats(rawpairs)
        for prefix in ['improving','deteriorating','unchanged']:
            for suffix in ['count','fraction']:
                close(r['pair_'+prefix+'_'+suffix],D(ps[prefix+'_'+suffix]))
                s['pair_'+prefix+'_'+suffix]=ps[prefix+'_'+suffix]
        derived[k]=s
    groups=rows(OUT/'outcome_gap_comparison.csv')
    assert [g['final_outcome'] for g in groups]==['BI','OOG','NI','POG']
    for g in groups:
        selected=[s for k,s in derived.items() if core[k]['category']==g['final_outcome']]
        assert int(g['trajectories'])==len(selected)
        selected_keys=[k for k in derived if core[k]['category']==g['final_outcome']]
        n=sum(len(by_run[k]) for k in selected_keys)
        assert int(g['valid_observations'])==n and int(g['observations'])==20*len(selected)
        assert int(g['invalid_observations'])==20*len(selected)-n
        for col,actual in g.items():
            if '_trajectory_' not in col: continue
            operation,metric=col.split('_trajectory_',1)
            if not selected:
                assert actual==''
            else:
                values=[s[metric] for s in selected]
                function={'mean':statistics.mean,'median':statistics.median,'min':min,'max':max}[operation]
                close(actual,D(function(values)))
    # A fresh deterministic execution must reproduce all seven CSVs and the report.
    names=[p.name for p in OUT.iterdir() if p.suffix=='.csv']+['P3_CHECKPOINT_SCORE_GAP_AUDIT.md']
    with tempfile.TemporaryDirectory(prefix='p3_gap_reproduce_') as tmp:
        subprocess.run([sys.executable,str(SOURCE/'checkpoint_gap_analysis.py'),'--output',tmp],check=True,capture_output=True,text=True)
        for name in names:
            assert (OUT/name).read_bytes()==(Path(tmp)/name).read_bytes(),name
    evidence={'status':'PASS','trajectories':60,'observations':1200,'valid_two_class':1000,'undefined_single_class':200,
              'independent_decimal_sign_counts':sign_counts,'authoritative_full_grid_categories':category_counts,
              'independent_recomputed':'original probe CSV signs, per-trajectory statistics, outcome assignments, group statistics and pair fractions',
              'deterministic_byte_comparison':sorted(names),'gpu_used':False,
              'limitation':'same-sample identity is established by saved indices, labels and recording-code provenance; original individual point score vectors were not saved'}
    (OUT/'independent_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
