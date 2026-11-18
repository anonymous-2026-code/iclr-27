"""Section 4.4 execution grid; the copied engine implements all updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from paper_pipeline.data import verify_processed_data
from section5_repro.data import Entity
from section5_repro.engine import RunOptions
from section5_repro.matrix_runner import run_device_matrix
from section5_repro.paths import Workspace
from section5_repro.protocol import load_protocol, repository_root
from section5_repro.supplement_engine import run_supplement_entity
from section5_repro.supplement_protocol import load_supplement_protocol, variants_for
from .cli import task_rows

# Exact submitted profile membership, including shared B=1 controls.
GROUPS = {
    "m2n2": ("e2", "standardized_m2n2_matched_diagnostic", (
        "m2n2_dt_only", "m2n2_practical_p95", "m2n2_oracle_point")),
    "fpm": ("e4", "fpm_only_selector_control", ("fpm_style_practical",)),
    "fpm_sana": ("e7", "matched_fpm_sana_diagnostic", (
        "frozen_sana", "practical_fpm_sana", "oracle_normal_window_sana", "oracle_clean_fpm_sana")),
    "labels": ("e1", "benchmark_label_sensitivity", (
        "adamw_ttlr0.0001__fn_contamination__frac0p1",
        "adamw_ttlr0.0001__boundary_erosion__frac0p1",
        "adamw_ttlr0.0001__boundary_dilation__frac0p1")),
    "cadence": ("e6", "update_cadence", tuple(
        f"cadence_b{cadence}_{arm}" for cadence in (2, 4, 8) for arm in ("practical", "oracle"))),
}


def build_plan(groups=None, datasets=None, seeds=None):
    selected = tuple(groups.split(",")) if groups else tuple(GROUPS)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(GROUPS):
        raise ValueError(f"Groups must be distinct members of {tuple(GROUPS)}")
    protocol = load_protocol()
    supplement = load_supplement_protocol()
    rows = task_rows(protocol, datasets, seeds)
    tasks = []
    for name in GROUPS:
        if name not in selected:
            continue
        experiment, _, variants = GROUPS[name]
        for row in rows:
            allowed = {v.name for v in variants_for(supplement, experiment, row["dataset"], row["seed"])}
            if not set(variants).issubset(allowed):
                raise ValueError(f"Submitted variant registry conflicts with protocol: {name}/{row}")
            tasks.append({**row, "group": name})
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "_worker", "aggregate"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--workspace", type=Path, default=Path("workspace/ae"))
        if name != "aggregate":
            sub.add_argument("--processed", type=Path, default=Path("data/processed"))
        if name in ("plan", "run"):
            sub.add_argument("--groups", help="Comma-separated: m2n2,fpm,fpm_sana,labels,cadence")
            sub.add_argument("--datasets")
            sub.add_argument("--seeds")
            sub.add_argument("--devices", default="0")
            sub.add_argument("--dry-run", action="store_true")
        if name == "_worker":
            sub.add_argument("--group", choices=tuple(GROUPS), required=True)
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--subset", default="")
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--device", choices=("cpu", "cuda"), required=True)
        if name == "aggregate":
            sub.add_argument("--metrics", type=Path, help="Replay saved physical metrics instead of reading the workspace")
            sub.add_argument("--core-derived", type=Path, required=True, help="Directory containing the matching core_cells.csv")
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        from .variant_analysis import aggregate
        aggregate(args.workspace.resolve(), args.core_derived.resolve(), args.output.resolve(), args.metrics)
        return 0
    if args.command == "_worker":
        entity = Entity(args.dataset, args.subset, args.processed.resolve() / args.dataset / args.subset)
        experiment, _, variants = GROUPS[args.group]
        records = run_supplement_entity(
            experiment=experiment, entity=entity, seed=args.seed,
            supplement=load_supplement_protocol(), workspace=Workspace.create(args.workspace),
            options=RunOptions(device=args.device, log_test_loss_during_source_training=False), variant_names=variants,
        )
        print(json.dumps({"group": args.group, "dataset": args.dataset, "subset": args.subset,
                          "seed": args.seed, "score_runs": len(records)}))
        return 0
    tasks = build_plan(args.groups, args.datasets, args.seeds)
    counts = {name: sum(t["group"] == name for t in tasks) * len(GROUPS[name][2]) for name in GROUPS}
    print(json.dumps({"tasks": len(tasks), "new_score_runs": sum(counts.values()), "scores_by_group": counts,
                      "optimizer": "AdamW", "learning_rate": 1e-4, "label_fraction": 0.1,
                      "cadences": [1, 2, 4, 8], "core_controls": "reused in the same AE workspace",
                      "dry_run": args.command == "plan" or args.dry_run}, indent=2), flush=True)
    if args.command == "plan" or args.dry_run:
        return 0
    verify_processed_data(args.processed, datasets=tuple(sorted({r["dataset"] for r in tasks})))
    workspace = args.workspace.resolve()
    def command_for(row, device):
        return [sys.executable, str(repository_root() / "scripts/reproduce_variants.py"), "_worker",
                "--workspace", str(workspace), "--processed", str(args.processed.resolve()),
                "--group", row["group"], "--dataset", row["dataset"], "--subset", row["subset"],
                "--seed", str(row["seed"]), "--device", device]
    for name in GROUPS:
        rows = [row for row in tasks if row["group"] == name]
        if not rows:
            continue
        code = run_device_matrix(
            rows=rows, devices=tuple(args.devices.split(",")), command_for=command_for,
            log_for=lambda r: workspace / "logs/variants" / r["group"] / f"{r['dataset']}__{r['subset'] or 'root'}__{r['seed']}.log",
            lock_path=workspace / ".matrix.lock", status_path=workspace / f"status_{name}.csv",
            dry_run=False, fail_fast=True,
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
