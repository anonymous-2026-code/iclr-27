from __future__ import annotations

import concurrent.futures
import csv
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence


def write_task_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty task matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_device_matrix(
    *,
    rows: Sequence[dict[str, object]],
    devices: Sequence[str],
    command_for: Callable[[dict[str, object], str], list[str]],
    log_for: Callable[[dict[str, object]], Path],
    lock_path: Path,
    status_path: Path,
    dry_run: bool,
    fail_fast: bool,
) -> int:
    if not rows:
        raise ValueError("The selected experiment matrix is empty")
    if "cpu" in devices and tuple(devices) != ("cpu",):
        raise ValueError("Use either one CPU worker or a comma-separated GPU list")
    if dry_run:
        return 0
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
    except FileExistsError as error:
        raise RuntimeError(
            f"Another matrix runner may be active: {lock_path}"
        ) from error

    tasks: queue.Queue[dict[str, object]] = queue.Queue()
    for row in rows:
        tasks.put(dict(row))
    print_lock = threading.Lock()
    stop = threading.Event()

    def device_loop(device_id: str) -> list[dict[str, object]]:
        outcomes: list[dict[str, object]] = []
        while not stop.is_set():
            try:
                row = tasks.get_nowait()
            except queue.Empty:
                break
            log_path = log_for(row)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            device_kind = "cpu" if device_id == "cpu" else "cuda"
            if device_kind == "cuda":
                environment["CUDA_VISIBLE_DEVICES"] = device_id
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command_for(row, device_kind),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            outcome = {
                **row,
                "device": device_id,
                "returncode": process.returncode,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            }
            outcomes.append(outcome)
            with print_lock:
                status = "OK" if process.returncode == 0 else "FAIL"
                print(
                    f"[{status}][{device_id}] "
                    f"{row.get('dataset')}/{row.get('subset') or '-'} "
                    f"seed={row.get('seed')} "
                    f"({outcome['elapsed_seconds']:.1f}s) {log_path}",
                    flush=True,
                )
            tasks.task_done()
            if process.returncode != 0 and fail_fast:
                stop.set()
        return outcomes

    outcomes: list[dict[str, object]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(devices)
        ) as executor:
            futures = [
                executor.submit(device_loop, device) for device in devices
            ]
            for future in concurrent.futures.as_completed(futures):
                outcomes.extend(future.result())
    finally:
        lock_path.unlink(missing_ok=True)
    if not outcomes:
        raise RuntimeError("The matrix runner produced no task outcome")
    outcomes.sort(
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("subset")),
            int(row.get("seed", 0)),
        )
    )
    write_task_csv(status_path, outcomes)
    failures = [
        row for row in outcomes if int(row["returncode"]) != 0
    ]
    print(
        f"Completed={len(outcomes) - len(failures)}, "
        f"failed={len(failures)}, status={status_path}",
        flush=True,
    )
    return 1 if failures else 0
