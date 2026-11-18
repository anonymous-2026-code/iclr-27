#!/usr/bin/env python3
"""Full clean-support Oracle probe expansion: no source training, explicit parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import experiment as future

np, pd, controls = future.np, future.pd, future.controls
ROOT = future.HERE.parents[2]
OUT = ROOT/'rebuttal/experiments/future_probe_mechanism/checkpoint_gap'
PLAN = OUT/'full_grid_execution_plan.json'
OLD = {'Daphnet','GECCO','PUMP'}

def write_json(path,value):
    controls.atomic_json(path,value)

def directory(task):
    b=task['base']
    return future.task_directory(task,OUT/'full_grid_runs'/(b['subset'] or '__root__'))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists(): raise FileExistsError(PLAN)
    previous=json.loads((future.HERE.parent/'gpu_controls/plan.json').read_text())
    protocol=controls.load_protocol()
    assert json.loads(json.dumps(protocol.as_dict()))==previous['protocol'], 'Current original protocol differs'
    core=pd.read_csv(ROOT/'workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/corrected_core_outcomes_with_objective.csv')
    support=pd.read_csv(controls.SUPPORT,keep_default_na=False)
    kept=support[support.support_status=='retained']
    expected={(r.dataset,'' if r.entity=='__root__' else r.entity) for r in kept.itertuples()}
    assert kept.groupby('dataset').size().to_dict()==controls.EXPECTED_ENTITIES
    assert len(previous['bases'])==530
    measured=pd.read_csv(future.HERE.parent/'ae_all_normal_window/derived/physical.csv',keep_default_na=False)
    timings={(r.dataset,r.subset,int(r.seed),r.optimizer):r.wall_seconds for r in measured.itertuples()}
    config={**protocol.as_dict(), 'optimizer_semantics':{'sgd':{'momentum':0,'weight_decay':0},'adamw':{'betas':[.9,.999],'eps':1e-8,'weight_decay':0}},
            'observation_steps':'floor(j*N/20), j=0..19, independently per retained physical entity',
            'primary_sampling':'up to 64 per class, endpoint-spaced ranks in strictly future suffix; original future_indices',
            'sensitivity_sampling':'up to 64 per class, floor((rank+0.5)*eligible_count/k), same eligible suffix',
            'family_aggregation':'At each relative observation j, equal-entity arithmetic mean of class means and pair fractions over entities with BOTH classes; same entities before/after. Family-cell fractions use valid family observations. Also report entity-balanced mean/median per-entity fractions to expose aggregation dependence.',
            'tail_sampling':'intersection of primary probes with original full-prefix Frozen class quantiles; normal >= q85, anomaly <= q30, including ties; no replacement',
            'tail_effect':'local class-mean response and strict ordering, not the full-stream SC/NPG estimand',
            'empty_masks':'retain scheduled step and AdamW history; zero-filled losses forbidden except original masked objective'}
    config_hash=digest(config)
    source_files=[future.HERE/'experiment.py',Path(__file__).resolve(),ROOT/'src/section5_repro/engine.py',ROOT/'src/section5_repro/data.py',ROOT/'src/section5_repro/protocol.py']
    code_hashes={str(p.relative_to(ROOT)):controls.sha256_file(p) for p in source_files}
    checked_entities=set(); source_inventory=[];tasks=[];inventory=[]
    all_checkpoint_files=sorted((ROOT/'workspace/ae/checkpoints').rglob('*.pth'))
    intermediate=[p for p in all_checkpoint_files if 'TTA' in str(p.relative_to(ROOT/'workspace/ae/checkpoints')) or 'oracle' in str(p.relative_to(ROOT/'workspace/ae/checkpoints')).lower()]
    assert not intermediate, 'Inspect unexpectedly available adapted checkpoint states before replay'
    for bid,b in sorted(previous['bases'].items()):
        assert (b['dataset'],b['subset']) in expected
        checkpoint=Path(b['checkpoint'])
        assert checkpoint.is_file() and controls.sha256_file(checkpoint)==b['source_checkpoint_sha256']
        entity=controls.Entity(b['dataset'],b['subset'],Path(b['directory']))
        assert controls._checkpoint_complete(checkpoint,protocol,entity,b['seed'])
        if (b['dataset'],b['subset']) not in checked_entities:
            assert controls.entity_manifest(entity)==b['data_manifest']
            checked_entities.add((b['dataset'],b['subset']))
        source_inventory.append(dict(dataset=b['dataset'],entity=b['subset'] or '__root__',seed=b['seed'],source_checkpoint=str(checkpoint),source_sha256=b['source_checkpoint_sha256'],source_complete=True))
        for opt in [c.key for c in protocol.tta_configs]:
            ref=b['references'][opt+':oracle']
            for rkey in ['shared:frozen',opt+':oracle',opt+':practical']:
                r=b['references'][rkey]
                meta=json.loads(Path(r['manifest_path']).read_text())
                assert meta['source_checkpoint_hash']==b['source_checkpoint_sha256'] and meta['data_manifest']==b['data_manifest'] and meta['status']=='complete'
                assert Path(r['score_path']).is_file() and Path(r['label_path']).is_file()
            labels=np.load(ref['label_path'],allow_pickle=False).reshape(-1)
            score=np.load(ref['score_path'],allow_pickle=False).reshape(-1)
            assert labels.shape==score.shape and np.isfinite(score).all() and set(np.unique(labels))=={0,1}
            n=len(labels)//12
            assert len(labels)==n*12
            expected_valid=sum(all(len(v) for v in future.future_indices(labels,j*n//20).values()) for j in range(20))
            task=dict(base=b,optimizer=opt,base_id=bid,configuration_sha256=config_hash,estimated_seconds=float(timings[b['dataset'],b['subset'],b['seed'],opt]))
            reuse=None
            if b['dataset'] in OLD:
                reuse=future.task_directory(task)
            elif b['dataset']=='SMD' and b['subset']=='M-1-6' and opt in ('sgd_ttlr0.0001','adamw_ttlr0.0001'):
                reuse=future.task_directory(task,future.HERE/'runs/smd_m1_6')
            if reuse is not None:
                result=json.loads((reuse/'result.json').read_text())
                assert result['complete'] and result['source_checkpoint_sha256']==b['source_checkpoint_sha256'] and result['auprc_parity_error']<1e-8
                f=pd.read_csv(reuse/'probes.csv')
                assert len(f)==20 and f.window.tolist()==[j*n//20 for j in range(20)]
                task['reuse_directory']=str(reuse)
                task['estimated_seconds']=0.
            row=core[(core.dataset==b['dataset'])&(core.optimizer_id==opt)&(core.seed==b['seed'])].iloc[0]
            tasks.append(task)
            inventory.append(dict(dataset=b['dataset'],entity=b['subset'] or '__root__',optimizer=opt,seed=b['seed'],source_checkpoint=str(checkpoint),source_sha256=b['source_checkpoint_sha256'],configuration_sha256=config_hash,oracle_score_path=ref['score_path'],oracle_label_path=ref['label_path'],original_metric=ref['auprc'],source_available=True,oracle_scores_available=True,adapted_checkpoint_states_available=False,required_action='reuse_saved_probes' if reuse else 'replay_original_oracle',expected_observations=20,expected_valid_two_class=expected_valid,expected_gpu_seconds=task['estimated_seconds'],outcome=row.category,output_directory=str(reuse or directory(task))))
    priority={'SMD':0,'PSM':1,'SWAT2015':2,'SMAP':3,'MSL':4,'Genesis':5}
    tasks.sort(key=lambda t:(priority.get(t['base']['dataset'],6),-t['estimated_seconds'],t['base_id'],t['optimizer']))
    loads=[0.]*4
    for i,t in enumerate(tasks):
        shard=min(range(4),key=lambda s:loads[s]); t.update(task_id=i,shard=shard);loads[shard]+=t['estimated_seconds']
    assert len(tasks)==2120 and len(checked_entities)==106
    pd.DataFrame(source_inventory).to_csv(OUT/'full_grid_source_checkpoint_inventory.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'full_grid_execution_manifest.csv',index=False)
    pd.DataFrame([{'checkpoint':str(p),'kind':'source','used_in_retained_plan':str(p) in {r['source_checkpoint'] for r in source_inventory}} for p in all_checkpoint_files]).to_csv(OUT/'full_grid_checkpoint_file_inventory.csv',index=False)
    plan=dict(created_at=controls.now(),configuration=config,configuration_sha256=config_hash,code_sha256=code_hashes,tasks=tasks,devices=[0, 1, 2, 3],cpu_cores=sorted(os.sched_getaffinity(0))[:4],python=sys.executable,torch=future.torch.__version__,numpy=np.__version__,pandas=pd.__version__,logical_cells=180,physical_trajectories=2120,expected_physical_observations=42400,expected_family_observations=3600,expected_valid_physical_observations=sum(r['expected_valid_two_class'] for r in inventory),new_replays=sum('reuse_directory' not in t for t in tasks),reused_physical=sum('reuse_directory' in t for t in tasks),estimated_gpu_hours=sum(loads)/3600,estimated_wall_hours=max(loads)/3600,adapted_states_found=len(intermediate))
    write_json(PLAN,plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in ['tasks','configuration','code_sha256']},indent=2))

def run_task(task):
    if 'reuse_directory' in task: raise ValueError('A reused trajectory must not be rerun')
    path=directory(task)
    if path.exists(): raise FileExistsError(path)
    start=controls.now()
    result=future.run_one(task,OUT/'full_grid_runs'/(task['base']['subset'] or '__root__'),save_probe_vectors=True,sampling_sensitivity=True)
    result.update(subset=task['base']['subset'],configuration_sha256=task['configuration_sha256'],task_id=task['task_id'],started_at=start,finished_at=controls.now(),source_checkpoint=task['base']['checkpoint'],code_sha256=json.loads(PLAN.read_text())['code_sha256'])
    write_json(path/'result.json',result)
    print(json.dumps({'task_id':task['task_id'],'dataset':result['dataset'],'entity':result['subset'],'seed':result['seed'],'optimizer':result['optimizer'],'max_score_parity_error':result['max_score_parity_error'],'auprc_parity_error':result['auprc_parity_error'],'seconds':result['wall_seconds']}),flush=True)

def worker(shard,task_id=None):
    controls.configure_threads()
    plan=json.loads(PLAN.read_text())
    os.sched_setaffinity(0,{plan['cpu_cores'][shard]})
    for task in plan['tasks']:
        if task_id is not None:
            if task['task_id']==task_id: run_task(task);return
        elif task['shard']==shard and 'reuse_directory' not in task:
            # The parity-checked smoke run is reused by the main batch.
            if task['task_id']==plan.get('smoke_task_id'): continue
            run_task(task)
    if task_id is not None: raise ValueError(task_id)

def smoke():
    plan=json.loads(PLAN.read_text())
    task=next(t for t in plan['tasks'] if t['base']['dataset']=='SMD' and 'reuse_directory' not in t)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    subprocess.run([sys.executable,__file__,'worker','--shard','0','--task-id',str(task['task_id'])],env=env,check=True)
    result=json.loads((directory(task)/'result.json').read_text())
    assert result['max_score_parity_error']==0 and result['auprc_parity_error']<=1e-8
    # Independent vector/aggregate arithmetic at all 20 smoke observations.
    f=pd.read_csv(directory(task)/'probes.csv',float_precision='round_trip')
    for v in map(json.loads,(directory(task)/'probe_vectors.jsonl').read_text().splitlines()):
        if v['design']!='primary':continue
        r=f[f.window==v['window']].iloc[0]
        for name in ['normal','anomaly']:
            assert json.loads(r[name+'_indices'])==v[name+'_indices']
            for phase in ['before','after']:
                x=np.array(v[name+'_'+phase])
                if len(x):np.testing.assert_allclose(x.mean(),r[name+'_'+phase],atol=0,rtol=0)
    plan['smoke_task_id']=task['task_id'];plan['smoke_verified_at']=controls.now()
    write_json(PLAN,plan)
    write_json(OUT/'full_grid_smoke_verification.json',dict(status='PASS',task_id=task['task_id'],result_path=str(directory(task)/'result.json'),max_score_error=result['max_score_parity_error'],auprc_error=result['auprc_parity_error'],probe_vectors_checked=True))

def launch():
    plan=json.loads(PLAN.read_text())
    assert 'smoke_verified_at' in plan
    target=OUT/'full_grid_launch.json'
    if target.exists():raise FileExistsError(target)
    processes=[]
    logs=OUT/'full_grid_logs';logs.mkdir(exist_ok=False)
    for shard in range(4):
        log=(logs/f'gpu{shard}.log').open('w')
        command=[sys.executable,__file__,'worker','--shard',str(shard)]
        p=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(plan['devices'][shard])},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();processes.append(dict(pid=p.pid,shard=shard,command=command,log=str(logs/f'gpu{shard}.log')))
    write_json(target,dict(started_at=controls.now(),workers=processes))
    print(json.dumps(processes,indent=2))

def status():
    plan=json.loads(PLAN.read_text());records=[]
    for t in plan['tasks']:
        folder=Path(t['reuse_directory']) if 'reuse_directory' in t else directory(t)
        p=folder/'result.json'
        r=json.loads(p.read_text()) if p.exists() else {}
        records.append(dict(task_id=t['task_id'],dataset=t['base']['dataset'],entity=t['base']['subset'] or '__root__',optimizer=t['optimizer'],seed=t['base']['seed'],reuse='reuse_directory' in t,status='complete' if r.get('complete') else 'started_incomplete' if folder.exists() else 'pending',wall_seconds=r.get('wall_seconds'),score_error=r.get('max_score_parity_error'),metric_error=r.get('auprc_parity_error'),result_path=str(p)))
    f=pd.DataFrame(records);f.to_csv(OUT/'full_grid_completion_status.csv',index=False)
    print(f.groupby(['dataset','status']).size().to_string())
    print('completed',sum(f.status=='complete'),'/',len(f))
    return f

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','worker','launch','status']);parser.add_argument('--shard',type=int,default=0);parser.add_argument('--task-id',type=int)
    args=parser.parse_args()
    if args.command=='worker':worker(args.shard,args.task_id)
    else:globals()[args.command]()
