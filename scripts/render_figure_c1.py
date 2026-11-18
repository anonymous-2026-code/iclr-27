#!/usr/bin/env python3
"""Figure C.1 from submitted bucket statistics or completed AE score arrays."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_artifact.figure_c1 import _compute_seed_and_aggregate_bucket_stats, save_side_by_side_percentile_curves
from review_artifact.score_shift import read_core_arrays
from review_artifact.cli import entity_records
from section5_repro.data import Entity
from section5_repro.paths import Workspace, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol


def raw_statistics(workspace_root, dataset):
    protocol = load_protocol()
    workspace = Workspace(workspace_root.resolve())
    entry = next(e for e in entity_records() if e["dataset"] == dataset and e["subset"] == "")
    entity = Entity(dataset, "", Path("unused"))
    config = protocol.tta_by_key("adamw_ttlr0.0001")
    records = []
    for seed in protocol.seeds:
        frozen, labels = read_core_arrays(frozen_result_dir(workspace, entity, seed, protocol), entry, seed)
        oracle, oracle_labels = read_core_arrays(tta_result_dir(workspace, entity, seed, protocol, config, True), entry, seed)
        if not np.array_equal(labels, oracle_labels) or not np.isfinite(frozen).all() or not np.isfinite(oracle).all():
            raise ValueError("Figure C.1 needs finite, fully matched score streams")
        records.append(pd.DataFrame({"seed": seed, "true_label": labels, "score_frozen": frozen.astype(float),
                                     "delta_score": oracle.astype(float) - frozen.astype(float)}))
    joined = pd.concat(records, ignore_index=True)
    return {name: _compute_seed_and_aggregate_bucket_stats(joined.loc[joined.true_label == label], 10)
            for name, label in (("normal", 0), ("anomaly", 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--workspace", type=Path, help="Recompute bins from completed AE Frozen/Oracle scores")
    inputs.add_argument("--evidence", type=Path, help="Previously generated bucket-statistics directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panels, arrays, tables = [], [], []
    for dataset in ("PUMP", "Daphnet"):
        if args.workspace:
            stats = raw_statistics(args.workspace, dataset)
        else:
            stats = {name: (pd.read_csv(args.evidence / dataset / f"{name}_bucket_stats_by_seed.csv"),
                            pd.read_csv(args.evidence / dataset / f"{name}_bucket_stats.csv"))
                     for name in ("normal", "anomaly")}
        panel = {}
        for name, (by_seed, summary) in stats.items():
            if (len(by_seed) != 50 or set(by_seed.seed) != set(range(42, 47))
                    or by_seed.duplicated(["seed", "bin_id"]).any()
                    or len(summary) != 10 or set(summary.bin_id) != set(range(10))):
                raise ValueError("Expected ten bins and five seeds per Figure C.1 class")
            panel[f"{name}_stats_df"] = summary
            arrays.extend(summary[col].to_numpy(dtype=float) for col in ("median_delta_q25", "median_delta_q75", "median_delta_median"))
            tables.append((dataset, name, by_seed, summary))
        panels.append(panel)
    values = np.concatenate(arrays)
    if not np.isfinite(values).all():
        raise ValueError("Undefined Figure C.1 bar/quantile statistic")
    lo, hi = min(float(values.min()), 0.), max(float(values.max()), 0.)
    span = hi - lo
    pad = .08 * span if span > 0 else .08 * max(abs(lo), abs(hi), 1.)
    args.output.mkdir(parents=True)
    for dataset, name, by_seed, summary in tables:
        target = args.output / dataset
        target.mkdir(exist_ok=True)
        by_seed.to_csv(target / f"{name}_bucket_stats_by_seed.csv", index=False)
        summary.to_csv(target / f"{name}_bucket_stats.csv", index=False)
    save_side_by_side_percentile_curves(panels, args.output / "fig5_3_2.png", (lo-pad, hi+pad))
    print("PASS: Figure C.1, 200 seed/class/bin rows and 40 displayed median bars.")


if __name__ == "__main__":
    main()
