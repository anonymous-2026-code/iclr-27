#!/usr/bin/env python3
"""Recompute the submitted dataset-cluster confidence intervals from saved cells."""
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_pipeline.statistics import category_proportion, dataset_cluster_bootstrap, estimates_frame, spearman_statistic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if len(frame) != 180 or frame.duplicated(["dataset", "optimizer_id", "seed"]).any():
        raise ValueError("Expected 180 distinct objective/detection cells")
    specs = [("OOG proportion", category_proportion("OOG"), 2701),
             ("NI proportion", category_proportion("NI"), 2702),
             ("objective/detection Spearman rho", spearman_statistic("oracle_normal_loss_reduction", "oracle_delta_auprc"), 2703)]
    estimates = [dataset_cluster_bootstrap(frame, estimand=name, statistic=fn, replicates=10000, seed=seed)
                 for name, fn, seed in specs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        estimates_frame(estimates).to_csv(handle, index=False)
