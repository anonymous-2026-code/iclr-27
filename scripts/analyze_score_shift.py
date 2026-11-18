#!/usr/bin/env python3
"""Section 4.3: raw-score diagnostics or fast replay of saved cell features."""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_artifact.score_shift import physical_features, aggregate_features, attach_detection, associations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--workspace", type=Path, help="Completed AE workspace; reads raw Frozen/Oracle scores")
    inputs.add_argument("--saved-features", type=Path, help="Replay existing logical features without raw-score recomputation")
    parser.add_argument("--cells", type=Path, required=True, help="Matching corrected core_cells.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    physical = physical_features(args.workspace) if args.workspace else None
    features = aggregate_features(physical) if physical is not None else pd.read_csv(args.saved_features)
    logical = attach_detection(features, pd.read_csv(args.cells))
    stats = associations(logical)
    args.output.mkdir(parents=True)
    if physical is not None:
        physical.to_csv(args.output / "physical_features.csv", index=False)
    logical.to_csv(args.output / "score_shift_full_analysis.csv", index=False)
    stats.to_csv(args.output / "score_shift_associations.csv", index=False)
    print(f"PASS: {len(logical)} logical tail-setting rows; {len(stats)} association estimates.")
