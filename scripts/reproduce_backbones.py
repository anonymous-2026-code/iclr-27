#!/usr/bin/env python3
"""Table 2: train, score and aggregate the submitted USAD/LSTMED protocol."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_pipeline.data import verify_processed_data
from section5_repro.data import Entity
from section5_repro.paths import Workspace
from section5_repro.matrix_runner import run_device_matrix
from review_artifact.cli import task_rows
from review_artifact.backbone_engine import BACKBONES, protocol_for, run_entity
from review_artifact.backbone_analysis import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "_worker", "aggregate"):
        sub = commands.add_parser(name)
        sub.add_argument("--workspace", type=Path, default=Path("workspace/backbones"))
        if name in ("plan", "run"):
            sub.add_argument("--backbones", default="USAD,LSTMED")
            sub.add_argument("--datasets")
            sub.add_argument("--seeds")
            sub.add_argument("--devices", default="0")
            sub.add_argument("--dry-run", action="store_true")
        if name != "aggregate":
            sub.add_argument("--processed", type=Path, default=Path("data/processed"))
        if name == "_worker":
            sub.add_argument("--backbone", choices=BACKBONES, required=True)
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--subset", default="")
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--device", choices=("cpu", "cuda"), required=True)
        if name == "aggregate":
            sub.add_argument("--metrics", type=Path)
            sub.add_argument("--support-root", type=Path)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.workspace.resolve(), args.output, args.metrics, args.support_root)
        return 0
    if args.command == "_worker":
        entity = Entity(args.dataset, args.subset, args.processed.resolve() / args.dataset / args.subset)
        count = run_entity(entity, args.seed, args.backbone, Workspace.create(args.workspace), args.device)
        print(json.dumps({"backbone": args.backbone, "dataset": args.dataset, "subset": args.subset,
                          "seed": args.seed, "score_runs": count}))
        return 0
    backbones = args.backbones.split(",")
    if not backbones or len(set(backbones)) != len(backbones) or set(backbones) - set(BACKBONES):
        raise ValueError("--backbones must be distinct members of USAD,LSTMED")
    units = task_rows(protocol_for(backbones[0]), args.datasets, args.seeds)
    tasks = [{**r, "backbone": b} for b in backbones for r in units]
    print(json.dumps({"source_tasks": len(tasks), "unique_score_runs": 17 * len(tasks),
                      "logical_cells": len(backbones) * 2 * len({r['dataset'] for r in units}) * len({r['seed'] for r in units}) * 4,
                      "scopes": ["full_model", "reconstruction_path"],
                      "dry_run": args.command == "plan" or args.dry_run}, indent=2), flush=True)
    if args.command == "plan" or args.dry_run:
        return 0
    verify_processed_data(args.processed, datasets=tuple(sorted({r['dataset'] for r in units})))
    workspace = args.workspace.resolve()
    def command_for(row, device):
        return [sys.executable, str(Path(__file__).resolve()), "_worker", "--backbone", row["backbone"],
                "--dataset", row["dataset"], "--subset", row["subset"], "--seed", str(row["seed"]),
                "--processed", str(args.processed.resolve()), "--workspace", str(workspace), "--device", device]
    return run_device_matrix(rows=tasks, devices=tuple(args.devices.split(",")), command_for=command_for,
        log_for=lambda r: workspace / "logs" / r["backbone"] / f"{r['dataset']}__{r['subset'] or 'root'}__{r['seed']}.log",
        lock_path=workspace / ".matrix.lock", status_path=workspace / "status.csv", dry_run=False, fail_fast=True)


if __name__ == "__main__":
    raise SystemExit(main())
