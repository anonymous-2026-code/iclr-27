#!/usr/bin/env python3
"""Launch a detached batch and perform one finite startup/resource check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    validation = json.loads((HERE/'validation.json').read_text())
    if validation['status'] != 'passed':
        raise RuntimeError('Protocol validation must pass before launch')
    if (HERE/'launch.json').exists() or (HERE/'status.json').exists():
        raise FileExistsError('Already launched; no automatic relaunch')
    stamp = datetime.now(timezone.utc)
    priority = max(0, 10-os.getpriority(os.PRIO_PROCESS, 0))
    command = ['nice', '-n', str(priority), 'taskset', '-c', str(plan['supervisor_core']),
               sys.executable, str(HERE/'experiment.py'), 'supervise']
    with (HERE/'launcher.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    estimated = max(plan['estimated_shard_hours'])
    record = {'pid': process.pid, 'started_at': stamp.isoformat(), 'command': command,
        'independent_session': True, 'devices': plan['devices'], 'expected_gpu_runs': plan['expected_gpu_runs'],
        'remaining_gpu_runs': plan['expected_gpu_runs']-len(validation['formal_runs']),
        'estimated_gpu_hours_wall': estimated,
        'estimated_gpu_finish_utc': (stamp+timedelta(hours=estimated)).isoformat(),
        'cpu_analyses': plan['cpu_analyses']}
    (HERE/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'launched': record}), flush=True)
    time.sleep(20)
    status = json.loads((HERE/'status.json').read_text())
    observations = []
    for worker in status.get('workers', []):
        item = dict(worker, alive=Path(f"/proc/{worker['pid']}").exists())
        if item['alive']:
            item['affinity'] = sorted(os.sched_getaffinity(worker['pid']))
            item['nice'] = os.getpriority(os.PRIO_PROCESS, worker['pid'])
        lines = (HERE/'logs'/f"{worker['name']}.log").read_text().splitlines()
        item['traceback_seen'] = any('Traceback (most recent call last)' in line for line in lines)
        item['started'] = any(line.startswith('START ') for line in lines)
        item['completed'] = worker['name'] in status.get('completed_workers', [])
        observations.append(item)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True, timeout=20)
    confirmation = {'confirmed_at': datetime.now(timezone.utc).isoformat(), 'one_time_confirmation': True,
        'status': status['status'], 'workers': observations, 'loadavg': Path('/proc/loadavg').read_text().strip(),
        'gpu_index_memory_mib_utilization_percent': gpu.stdout.strip()}
    (HERE/'startup_confirmation.json').write_text(json.dumps(confirmation, indent=2)+'\n')
    print(json.dumps({'startup': confirmation}), flush=True)
    valid = lambda w: w['completed'] or (w['alive'] and w.get('affinity') == [w['cpu_core']] and w.get('nice', 0) >= 10 and w['started'])
    if status['status'] not in ('running', 'complete') or len(observations) != 5 or not all(valid(w) and not w['traceback_seen'] for w in observations):
        raise RuntimeError('Startup confirmation failed; inspect saved logs, do not automatically relaunch')


if __name__ == '__main__':
    main()
