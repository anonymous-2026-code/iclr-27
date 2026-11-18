#!/usr/bin/env python3
"""Launch the validated four-GPU extension in an independent process session."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    if json.loads((HERE / "validation.json").read_text())["status"] != "passed":
        raise RuntimeError("Full-stream parity and optimizer validation must pass")
    if not json.loads((HERE / "aggregation_validation.json").read_text())["passed"]:
        raise RuntimeError("Aggregation regression must pass")
    for name in ("launch.json", "status.json", "launcher.log"):
        if (HERE / name).exists():
            raise FileExistsError(f"Already launched: {HERE / name}")
    command = ["nice", "-n", "10", "taskset", "-c", str(min(os.sched_getaffinity(0))), sys.executable, "-B",
               str(HERE / "run.py"), "supervise"]
    with (HERE / "launcher.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    record = {"pid": process.pid, "command": command, "independent_session": True,
              "plan": str(HERE / "plan.json"), "status": str(HERE / "status.json")}
    (HERE / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
