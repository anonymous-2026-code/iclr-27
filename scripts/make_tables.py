#!/usr/bin/env python3
"""Rebuild submitted numerical tables without training or inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_artifact.aggregate import CATEGORIES, aggregate_physical, category


def write_table(frame, output, name):
    frame.to_csv(output / f"{name}.csv", index=False)
    frame.to_latex(output / f"{name}.tex", index=False, escape=True, float_format=lambda v: f"{v:.2f}")


def counts(practical, oracle, expected_rows):
    if len(practical) != expected_rows or len(oracle) != expected_rows:
        raise ValueError(f"Expected {expected_rows} matched cells.")
    if not np.isfinite(np.asarray([practical, oracle], dtype=float)).all():
        raise ValueError("Undefined cells cannot be classified as NI.")
    classified = pd.Series([category(p, o) for p, o in zip(practical, oracle)])
    return {key: int((classified == key).sum()) for key in CATEGORIES}


def variant_tables(evidence, output):
    robustness = pd.read_csv(evidence / "robustness_outcomes.csv")
    rows = []
    for name, frame in robustness.groupby("analysis", sort=False):
        rows.append({"Analysis": name, **counts(frame.corrected_practical_delta, frame.corrected_oracle_delta, 45)})
    write_table(pd.DataFrame(rows), output, "table_b2_variants")
    perturbation = pd.read_csv(evidence / "label_perturbation_outcomes.csv")
    rows = []
    for name, frame in perturbation.groupby("analysis", sort=True):
        observed = counts(frame.corrected_practical_delta, frame.corrected_oracle_delta, 45)
        labels = np.asarray([category(p, o) for p, o in zip(frame.corrected_practical_delta, frame.corrected_oracle_delta)])
        retained = int((labels == frame.original_category.to_numpy()).sum())
        rows.append({"Analysis": name, **observed, "Retained": retained, "Cells": 45})
    write_table(pd.DataFrame(rows), output, "table_b3_label_retention")


def table2_tables(cells, backbone_root, output):
    if cells.duplicated(["dataset", "optimizer", "seed"]).any():
        raise ValueError("Duplicate AE logical cells")
    core_counts = counts(cells.corrected_auprc_practical_delta, cells.corrected_auprc_oracle_delta, 180)
    rows = [{"Backbone": "AE", "Update scope": "Full model", "Cells": 180, **core_counts}]
    for backbone in ("USAD", "LSTMED"):
        for scope, display in (("FULL", "Full model"), ("DECODER_ONLY", "Decoder only")):
            frame = pd.read_csv(backbone_root / f"{backbone}_{scope}_ICLR27_OUTCOMES.csv")
            if frame.duplicated(["dataset", "optimizer", "lr", "seed"]).any():
                raise ValueError("Duplicate backbone logical cells")
            observed = counts(frame.delta_practical, frame.delta_oracle, 180)
            if [category(p, o) for p, o in zip(frame.delta_practical, frame.delta_oracle)] != frame.category.tolist():
                raise ValueError("Backbone categories do not match paired effects")
            rows.append({"Backbone": backbone, "Update scope": display, "Cells": 180, **observed})
    numeric = pd.DataFrame(rows)
    numeric.to_csv(output / "table2_backbone_counts.csv", index=False)
    display = numeric.copy()
    for col in CATEGORIES:
        display[col] = [f"{n} ({100*n/180:.1f}%)" for n in numeric[col]]
    write_table(display, output, "table2_backbone_outcomes")


def build_tables(evidence, output, core_only=False, backbone_root=None):
    physical = pd.read_csv(evidence / "core_physical.csv")
    cells, support = aggregate_physical(physical)
    # Saved logical cells are an independent consistency check, not the numerical input.
    expected = pd.read_csv(evidence / "core_cells.csv")
    keys = ["dataset", "optimizer", "seed"]
    joined = cells.merge(expected, on=keys, suffixes=("", "_saved"), validate="one_to_one")
    if len(joined) != 180:
        raise ValueError("The saved logical reference must contain exactly 180 matching cells.")
    for col in ("corrected_auprc_practical_delta", "corrected_auprc_oracle_delta"):
        np.testing.assert_allclose(joined[col], joined[col + "_saved"], rtol=0, atol=1e-14)
    if not (joined.corrected_auprc_category == joined.corrected_auprc_category_saved).all():
        raise ValueError("Recomputed categories disagree with the saved reference.")
    output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(output / "core_cells.csv", index=False)
    support.to_csv(output / "core_support.csv", index=False)
    table1 = cells.groupby("dataset").corrected_auprc_oracle_delta.agg(["mean", "median", "min", "max"]) * 100
    table1 = table1.rename(columns={"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum"}).reset_index()
    table1 = table1.rename(columns={"dataset": "Dataset"})
    table1["Dataset"] = table1.Dataset.replace({"SWAT2015": "SWAT"})
    table1.iloc[:, 1:] = table1.iloc[:, 1:].map(lambda x: 0.0 if abs(x) < 0.005 else x)
    write_table(table1, output, "table1_oracle_effects_pp")
    core_counts = counts(cells.corrected_auprc_practical_delta, cells.corrected_auprc_oracle_delta, 180)
    write_table(pd.DataFrame([{"Cells": 180, **core_counts}]), output, "table_b1_core_outcomes")
    if not core_only:
        table2_tables(cells, backbone_root or evidence / "backbones", output)
        variant_tables(evidence, output)
    print(json.dumps({"logical_cells": len(cells), "counts": core_counts, "table2_rows": 5 if not core_only else 0}, indent=2))
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="Fresh aggregation output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-derived", type=Path, help="Directory containing newly aggregated backbones/ CSVs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--core-only", action="store_true", help="Use fresh AE aggregation without historical backbone evidence")
    mode.add_argument("--table2-only", action="store_true", help="Build Table 2 from AE cells and backbone effects")
    mode.add_argument("--variants-only", action="store_true", help="Generate B.2/B.3 from the variant analysis output")
    args = parser.parse_args()
    if args.variants_only:
        args.output.mkdir(parents=True, exist_ok=False)
        variant_tables(args.evidence.resolve(), args.output.resolve())
    elif args.table2_only:
        args.output.mkdir(parents=True, exist_ok=False)
        table2_tables(pd.read_csv(args.evidence / "core_cells.csv"),
                      args.backbone_derived or args.evidence / "backbones", args.output)
    else:
        build_tables(args.evidence.resolve(), args.output.resolve(), args.core_only, args.backbone_derived)
