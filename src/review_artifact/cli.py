"""Small review entry point; scientific execution is delegated to the copied engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_pipeline.data import prepare_data, validate_raw_data, verify_processed_data
from section5_repro.data import Entity
from section5_repro.engine import RunOptions, run_entity_seed
from section5_repro.matrix_runner import run_device_matrix
from section5_repro.paths import Workspace
from section5_repro.protocol import load_protocol, repository_root


def entity_records():
    path = repository_root() / "reference/data/processed_manifest.json"
    return json.loads(path.read_text())["entities"]


def task_rows(protocol, datasets: str | None, seeds: str | None):
    selected = tuple(datasets.split(",")) if datasets else protocol.datasets
    selected_seeds = tuple(map(int, seeds.split(","))) if seeds else protocol.seeds
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(protocol.datasets):
        raise ValueError("Datasets must be distinct members of the submitted protocol.")
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds) or set(selected_seeds) - set(protocol.seeds):
        raise ValueError("Seeds must be distinct members of 42,43,44,45,46.")
    return [
        {"dataset": row["dataset"], "subset": row["subset"], "seed": seed}
        for row in entity_records() if row["dataset"] in selected
        for seed in selected_seeds
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Anonymous Frozen / Practical / Oracle reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "prepare-data"):
        sub = commands.add_parser(name)
        sub.add_argument("--raw-root", type=Path, default=Path("data/raw"))
        sub.add_argument("--processed", type=Path, default=Path("data/processed"))
    for name in ("plan", "run", "_worker", "aggregate"):
        sub = commands.add_parser(name)
        sub.add_argument("--workspace", type=Path, default=Path("workspace/ae"))
        if name != "aggregate":
            sub.add_argument("--processed", type=Path, default=Path("data/processed"))
        if name in ("plan", "run"):
            sub.add_argument("--datasets", help="Optional protocol subset; partial runs are not paper results")
            sub.add_argument("--seeds")
            sub.add_argument("--devices", default="0", help="GPU indices, or cpu")
            sub.add_argument("--dry-run", action="store_true")
        if name == "_worker":
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--subset", default="")
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--device", choices=("cpu", "cuda"), required=True)
        if name == "aggregate":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("validate-data", "prepare-data"):
        datasets = load_protocol().datasets
        if args.command == "validate-data":
            result = validate_raw_data(args.raw_root, datasets=datasets)
        else:
            result = prepare_data(args.raw_root, args.processed, datasets=datasets, resume=True)
        print(json.dumps(result, indent=2))
        return 0
    protocol = load_protocol()
    if args.command == "aggregate":
        from .aggregate import aggregate_workspace
        aggregate_workspace(args.workspace.resolve(), protocol, args.output.resolve())
        return 0
    if args.command == "_worker":
        entity = Entity(args.dataset, args.subset, args.processed.resolve() / args.dataset / args.subset)
        records = run_entity_seed(
            entity, args.seed, protocol, Workspace.create(args.workspace),
            RunOptions(device=args.device, log_test_loss_during_source_training=False),
        )
        print(json.dumps({"dataset": args.dataset, "subset": args.subset, "seed": args.seed, "score_runs": len(records)}))
        return 0
    rows = task_rows(protocol, args.datasets, args.seeds)
    print(json.dumps({
        "backbone": "AE", "update_scope": "full_model",
        "source_tasks": len(rows), "score_runs": 9 * len(rows),
        "source_and_score_runs": 10 * len(rows),
        "logical_cells": len({r["dataset"] for r in rows}) * len({r["seed"] for r in rows}) * 4,
        "dry_run": args.command == "plan" or args.dry_run,
    }, indent=2), flush=True)
    if args.command == "plan" or args.dry_run:
        return 0
    verify_processed_data(args.processed, datasets=tuple(sorted({r["dataset"] for r in rows})))
    workspace = args.workspace.resolve()

    def command_for(row, device):
        return [sys.executable, str(repository_root() / "scripts/reproduce.py"), "_worker",
                "--processed", str(args.processed.resolve()), "--workspace", str(workspace),
                "--dataset", row["dataset"], "--subset", row["subset"],
                "--seed", str(row["seed"]), "--device", device]

    return run_device_matrix(
        rows=rows, devices=tuple(args.devices.split(",")), command_for=command_for,
        log_for=lambda r: workspace / "logs" / f"{r['dataset']}__{r['subset'] or 'root'}__{r['seed']}.log",
        lock_path=workspace / ".matrix.lock", status_path=workspace / "status.csv",
        dry_run=False, fail_fast=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
