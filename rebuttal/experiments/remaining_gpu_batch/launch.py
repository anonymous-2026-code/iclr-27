#!/usr/bin/env python3
"""Detach the fixed four-GPU batch, then make exactly one startup check."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
now=lambda:datetime.now(timezone.utc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-extension",action="store_true")
    args=parser.parse_args()
    directory=HERE.parent/"window_length_sensitivity" if args.window_extension else HERE
    prefix="extension_" if args.window_extension else ""
    plan=json.loads((directory/f"{prefix}plan.json").read_text())
    if json.loads((directory/f"{prefix}validation.json").read_text())["status"]!="passed":
        raise RuntimeError("Validation must pass before launch")
    if (directory/f"{prefix}launch.json").exists() or (directory/f"{prefix}status.json").exists():
        raise FileExistsError("Batch already launched; no automatic relaunch")
    adjustment=max(0,10-os.getpriority(os.PRIO_PROCESS,0))
    command=["nice","-n",str(adjustment),"taskset","-c",str(min(os.sched_getaffinity(0))),sys.executable,"-B",str(HERE/"experiment.py"),
             "window-supervise" if args.window_extension else "supervise"]
    stamp=now()
    with (directory/f"{prefix}launcher.log").open("x") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1"))
    record={"pid":process.pid,"command":command,"started_at":stamp.isoformat(),"devices":plan["devices"],
        "expected":plan["expected"],"estimated_shard_hours":plan["estimated_shard_hours"],
        "estimated_finish_utc":(stamp+timedelta(hours=max(plan["estimated_shard_hours"]))).isoformat(),"independent_session":True}
    if not args.window_extension:
        record.update(reserved_4gpu_hours=12,estimated_finish_8h_utc=(stamp+timedelta(hours=8)).isoformat(),
                      reserved_finish_12h_utc=(stamp+timedelta(hours=12)).isoformat())
    (directory/f"{prefix}launch.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"launched":record}),flush=True)
    time.sleep(20)
    status=json.loads((directory/f"{prefix}status.json").read_text())
    workers=[]
    for worker in status.get("workers",[]):
        pid=worker["pid"]
        alive=Path(f"/proc/{pid}").exists()
        item=dict(worker,alive=alive)
        if alive:
            item.update(affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid))
        log=directory/f"{prefix}logs"/f"gpu{worker['gpu']}.log"
        lines=log.read_text().splitlines()
        item["first_line"]=lines[0] if lines else ""
        item["traceback_seen"]=any("Traceback (most recent call last)" in line for line in lines)
        workers.append(item)
    gpu=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],
                       check=True,capture_output=True,text=True,timeout=20)
    confirmation={"confirmed_at":now().isoformat(),"status":status["status"],"workers":workers,
        "loadavg":Path("/proc/loadavg").read_text().strip(),"gpu_index_memory_mib_utilization_percent":gpu.stdout.strip(),
        "one_time_confirmation":True}
    (directory/f"{prefix}startup_confirmation.json").write_text(json.dumps(confirmation,indent=2)+"\n")
    print(json.dumps({"startup":confirmation}),flush=True)
    if status["status"]!="running" or len(workers)!=4 or not all(w["alive"] and not w["traceback_seen"] and w.get("nice",0)>=10
            and w.get("affinity")==[w["cpu_core"]] and w["first_line"].startswith("START ") for w in workers):
        raise RuntimeError("One-time startup confirmation failed; inspect status and logs")


if __name__=="__main__":
    main()
