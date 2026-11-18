#!/usr/bin/env python3
"""Reclassify saved or freshly aggregated rebuttal cells and export CSV/LaTeX.

Undefined cells remain explicit. No missing entities or cells are zero-filled.
These are review navigation tables; the original analysis/plotting scripts are
also distributed under rebuttal/experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("ae_all_normal_window", (), "practical_delta", "window_oracle_delta", "new_category", 180),
    ("oracle_input_controls", ("condition",), "practical_delta", "oracle_delta", "category", 360),
    ("selected_count_normalization", (), "practical_delta", "oracle_delta", "category", 180),
    ("controlled_distribution_shift", ("condition",), "practical_delta", "oracle_delta", "category", 300),
    ("matched_update_count", ("sample_seed",), "practical_delta", "oracle_delta", "category", 540),
    ("causal_score_calibration", ("mode",), "practical_delta", "oracle_delta", "category", 720),
    ("window_length_sensitivity", ("window_length",), "practical_delta", "oracle_delta", "category", 540),
    ("adaptation_variants_180", ("analysis",), "practical_delta", "oracle_delta", "category", 720),
    ("affiliation_f", ("quantile",), "practical_delta", "oracle_delta", "category", 360),
    ("vus_pr_robustness", (), "vus_practical_delta", "vus_oracle_delta", "vus_category", 180),
)


def summarize(evidence, fresh=False):
    totals, datasets = [], []
    for experiment, groups, pcol, ocol, category_col, expected in SPECS:
        folder = evidence / experiment
        if fresh:
            folder /= "derived"
        frame = pd.read_csv(folder / "cells.csv", float_precision="round_trip")
        optimizer = "optimizer_id" if "optimizer_id" in frame else "optimizer"
        if len(frame) != expected or frame.duplicated([*groups, "dataset", optimizer, "seed"]).any():
            raise ValueError(f"{experiment}: expected {expected} unique planned cells")
        p, o = frame[pcol], frame[ocol]
        valid = np.isfinite(p) & np.isfinite(o)
        frame["computed_category"] = np.select(
            [~valid, (p > .005) & (o > .005), p > .005, o > .005],
            ["undefined", "BI", "POG", "OOG"], default="NI")
        declared = frame[category_col].str.replace("UNDEFINED", "undefined", regex=False)
        if not (frame.computed_category == declared).all():
            raise ValueError(f"{experiment}: saved category disagrees with finite paired deltas")
        items = frame.groupby(list(groups), dropna=False, sort=True) if groups else [((), frame)]
        for key, group in items:
            if not isinstance(key, tuple):
                key = (key,)
            setting = "; ".join(f"{k}={v}" for k, v in zip(groups, key)) or "default"
            counts = group.computed_category.value_counts()
            row = {"experiment": experiment, "setting": setting,
                   **{name: int(counts.get(name, 0)) for name in ("BI", "OOG", "NI", "POG", "undefined")},
                   "planned_cells": len(group), "valid_cells": int((group.computed_category != "undefined").sum())}
            totals.append(row)
            for dataset, subset in group.groupby("dataset", sort=True):
                finite = subset[subset.computed_category != "undefined"]
                datasets.append({"experiment": experiment, "setting": setting, "dataset": dataset,
                    "planned_cells": len(subset), "valid_cells": len(finite),
                    "undefined": len(subset) - len(finite),
                    "practical_median_pp": 100 * finite[pcol].median(),
                    "oracle_median_pp": 100 * finite[ocol].median()})
    return pd.DataFrame(totals), pd.DataFrame(datasets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, help="Directory containing previously generated experiment summaries")
    parser.add_argument("--experiments", type=Path, help="Use freshly generated experiments/*/derived instead")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.experiments is None and args.evidence is None:
        parser.error("Provide --experiments or --evidence with locally generated results")
    counts, datasets = summarize(args.experiments or args.evidence, fresh=args.experiments is not None)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("rebuttal_outcomes", counts), ("rebuttal_dataset_effects", datasets)):
        frame.to_csv(args.output / f"{name}.csv", index=False)
        frame.to_latex(args.output / f"{name}.tex", index=False, escape=True, float_format="%.4f")
    print(f"Rebuilt {len(counts)} settings / {counts.planned_cells.sum()} planned cells; "
          f"{counts['undefined'].sum()} undefined cells retained.")


if __name__ == "__main__":
    main()
