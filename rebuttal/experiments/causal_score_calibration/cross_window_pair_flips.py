#!/usr/bin/env python3
"""Exact, CPU-only postprocessing for the three preselected H=512 cases.

Reads existing artifacts only; emits full-precision results as JSON on stdout.
Transition matrices have raw rows and calibrated columns in I/T/C order
(incorrect=0, tie=0.5, correct=1). No inference or experiment entry point runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import analyze as calibration

HERE = Path(__file__).resolve().parent
CASES = (("Daphnet", ""), ("GECCO", ""), ("SMD", "M-1-6"))
OPTIMIZERS = ("sgd_ttlr0.0001", "adamw_ttlr0.0001")
SEEDS = tuple(range(42, 47))
TOLERANCE = 1e-12


def exact_transitions(raw, calibrated, labels):
    """All positive-negative transitions in O(n log n) time and O(n) memory.

    Sweep ascending raw scores. A Fenwick tree over calibrated normal-score
    ranks counts raw-correct pairs in each calibrated state. Equal-raw groups
    give the raw-tie row; calibrated marginals determine the remaining row.
    All counts use integers; equality means exact equality, with no tolerance.
    """
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    positive = positive[np.argsort(raw[positive], kind="stable")]
    negative = negative[np.argsort(raw[negative], kind="stable")]
    px, py = raw[positive], calibrated[positive]
    nx, ny = raw[negative], calibrated[negative]
    unique_y = np.unique(ny)
    ranks = np.searchsorted(unique_y, ny) + 1
    tree = [0] * (len(unique_y) + 1)

    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    matrix = np.zeros((3, 3), dtype=np.int64)
    unique_x, starts, sizes = np.unique(px, return_index=True, return_counts=True)
    lower = np.searchsorted(nx, unique_x, side="left")
    upper = np.searchsorted(nx, unique_x, side="right")
    inserted = 0
    for start, size, left, right in zip(starts, sizes, lower, upper):
        while inserted < left:
            index = int(ranks[inserted])
            while index < len(tree):
                tree[index] += 1
                index += index & -index
            inserted += 1
        values = py[start:start + size]
        lo = np.searchsorted(unique_y, values, side="left")
        hi = np.searchsorted(unique_y, values, side="right")
        for lrank, hrank in zip(lo, hi):
            less, at_most = prefix(int(lrank)), prefix(int(hrank))
            matrix[2] += (inserted - at_most, at_most - less, less)
        if right > left:
            tied = np.sort(ny[left:right])
            less = np.searchsorted(tied, values, side="left")
            at_most = np.searchsorted(tied, values, side="right")
            matrix[1] += ((right - left) * size - at_most.sum(),
                          (at_most - less).sum(), less.sum())
    sorted_y = np.sort(ny)
    less = np.searchsorted(sorted_y, py, side="left")
    at_most = np.searchsorted(sorted_y, py, side="right")
    columns = np.array([len(negative) * len(positive) - at_most.sum(),
                        (at_most - less).sum(), less.sum()], dtype=np.int64)
    matrix[0] = columns - matrix[1] - matrix[2]
    return matrix


def direct_transitions(raw, calibrated, labels):
    """Direct enumeration used only for small algorithm-validation examples."""
    p, n = labels == 1, labels == 0
    before = 2 * (raw[p, None] > raw[n]) + (raw[p, None] == raw[n])
    after = 2 * (calibrated[p, None] > calibrated[n]) + (calibrated[p, None] == calibrated[n])
    return np.bincount((3 * before + after).ravel(), minlength=9).reshape(3, 3)


def same_window_transitions(raw, calibrated, labels):
    y = labels.reshape(-1, 12)
    mixed = (y.sum(axis=1) > 0) & (y.sum(axis=1) < 12)
    y = y[mixed]
    a, b = raw.reshape(-1, 12)[mixed], calibrated.reshape(-1, 12)[mixed]
    pairs = (y[:, :, None] == 1) & (y[:, None, :] == 0)
    before = 2 * (a[:, :, None] > a[:, None, :]) + (a[:, :, None] == a[:, None, :])
    after = 2 * (b[:, :, None] > b[:, None, :]) + (b[:, :, None] == b[:, None, :])
    return np.bincount((3 * before[pairs] + after[pairs]), minlength=9).reshape(3, 3)


def validate_counter():
    rng = np.random.default_rng(2334)
    coverage = np.zeros((3, 3), dtype=np.int64)
    for trial in range(24):
        labels = rng.integers(0, 2, 48)
        labels[:2] = [0, 1]
        raw = rng.integers(-3, 4, 48).astype(float)
        calibrated = rng.integers(-3, 4, 48).astype(float)
        if trial == 0:
            raw[:] = 0
        if trial == 1:
            calibrated[:] = 0
        if trial == 2:
            calibrated = raw.copy()
        expected = direct_transitions(raw, calibrated, labels)
        np.testing.assert_array_equal(exact_transitions(raw, calibrated, labels), expected)
        within = sum((direct_transitions(raw[i:i + 12], calibrated[i:i + 12], labels[i:i + 12])
                      for i in range(0, 48, 12)), np.zeros((3, 3), dtype=np.int64))
        np.testing.assert_array_equal(same_window_transitions(raw, calibrated, labels), within)
        coverage += expected
    assert (coverage > 0).all(), "Counter checks must exercise all nine transitions"


def concordance(matrix):
    # Doubled credit keeps half-credit ties exact until the final division.
    before = int(matrix.sum(axis=1) @ np.array([0, 1, 2], dtype=np.int64))
    after = int(matrix.sum(axis=0) @ np.array([0, 1, 2], dtype=np.int64))
    denominator = int(matrix.sum())
    return {"denominator": denominator, "raw_credit_x2": before, "h512_credit_x2": after,
            "net_credit_x2": after - before, "raw": before / (2 * denominator),
            "h512": after / (2 * denominator), "change": (after - before) / (2 * denominator)}


def main():
    started = time.perf_counter()
    validate_counter()
    plan = calibration.load_plan()
    saved = pd.read_csv(HERE / "derived/physical.csv", keep_default_na=False)
    temporal = pd.read_csv(HERE.parent / "temporal_score_drift/derived/physical.csv", keep_default_na=False)
    keys = ["dataset", "subset", "seed", "optimizer", "arm"]
    saved = saved.set_index(keys + ["mode"], verify_integrity=True)
    temporal = temporal.set_index(keys, verify_integrity=True)
    records, paired, ids, paths = [], [], set(), set()
    errors = {"e14_pr_auc": 0., "canonical_raw_pr_auc": 0., "e12_raw_auroc": 0.,
              "auroc_levels": 0., "auroc_difference": 0.}
    for dataset, subset in CASES:
        for seed in SEEDS:
            base_id = f"{dataset}/{subset or '__root__'}/seed{seed}"
            base = plan["bases"][base_id]
            assert (base["dataset"], base["subset"], base["seed"]) == (dataset, subset, seed)
            source, _ = calibration.source_scores(base)
            official_labels = pd.read_parquet(Path(base["directory"]) / "test.parquet", columns=["label"])["label"].to_numpy()
            assert len(official_labels) == base["data_manifest"]["test_rows"]
            points = len(official_labels) // 12 * 12
            common_labels = official_labels[:points]
            assert set(np.unique(common_labels)) == {0, 1}
            arms = {}
            for key in ("shared:frozen", *(f"{o}:oracle" for o in OPTIMIZERS)):
                optimizer, arm = key.split(":")
                ref = base["references"][key]
                meta = json.loads(Path(ref["manifest_path"]).read_text())
                assert meta["status"] == "complete" and meta["backbone"] == "AE"
                assert meta["window"] == meta["stride"] == 12
                assert (meta["dataset"], meta["subset"], meta["seed"]) == (dataset, subset, seed)
                assert meta["source_checkpoint_hash"] == base["source_checkpoint_sha256"]
                assert meta["data_manifest"] == base["data_manifest"]
                assert meta["physical_run_id"] == ref["physical_run_id"]
                assert meta["result_metadata"]["points"] == points
                assert meta["arm"] == ("frozen" if arm == "frozen" else "oracle_benchmark_normal_point")
                if arm == "oracle":
                    assert meta["optimizer"] == optimizer.split("_")[0] and meta["learning_rate"] == 1e-4
                assert ref["physical_run_id"] not in ids and ref["score_path"] not in paths
                ids.add(ref["physical_run_id"])
                paths.add(ref["score_path"])
                raw = np.load(ref["score_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
                labels = np.load(ref["label_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
                assert raw.shape == labels.shape == (points,) and np.isfinite(raw).all()
                np.testing.assert_array_equal(labels, common_labels)
                calibrated, mean, std, epsilon = calibration.standardize(raw, source, 512)
                assert calibrated.shape == raw.shape
                assert np.all(np.maximum(std, epsilon) > 0)
                np.testing.assert_array_equal(calibrated.reshape(-1, 12),
                    (raw.reshape(-1, 12) - mean[:, None]) / np.maximum(std, epsilon)[:, None])
                identity = (dataset, subset, seed, optimizer, arm)
                pr = {}
                for mode, values in (("raw", raw), ("z512", calibrated)):
                    pr[mode] = calibration.controls.metric(labels, values)
                    previous = saved.loc[identity + (mode,)]
                    assert int(previous.points) == points
                    error = abs(pr[mode] - float(previous.auprc))
                    assert error <= TOLERANCE, (identity, mode, "E14 PR-AUC", error)
                    errors["e14_pr_auc"] = max(errors["e14_pr_auc"], error)
                error = abs(pr["raw"] - ref["auprc"])
                assert error <= TOLERANCE, (identity, "canonical raw PR-AUC", error)
                errors["canonical_raw_pr_auc"] = max(errors["canonical_raw_pr_auc"], error)
                assert float(saved.loc[identity + ("z512",), "floor_window_fraction"]) == float((std < epsilon).mean())
                total = exact_transitions(raw, calibrated, labels)
                within = same_window_transitions(raw, calibrated, labels)
                assert not np.any(within - np.diag(np.diag(within))), (identity, "Same-window order changed", within)
                cross = total - within
                positives = int(labels.sum())
                per_window = labels.reshape(-1, 12).sum(axis=1).astype(np.int64)
                assert total.sum() == positives * (points - positives)
                assert within.sum() == (per_window * (12 - per_window)).sum()
                assert (total >= 0).all() and (cross >= 0).all()
                all_concordance, cross_concordance = concordance(total), concordance(cross)
                roc = [float(roc_auc_score(labels, s)) for s in (raw, calibrated)]
                for observed, counted in zip(roc, (all_concordance["raw"], all_concordance["h512"])):
                    error = abs(observed - counted)
                    assert error <= TOLERANCE, (identity, "AUROC level", error)
                    errors["auroc_levels"] = max(errors["auroc_levels"], error)
                error = abs(roc[1] - roc[0] - all_concordance["change"])
                assert error <= TOLERANCE, (identity, "AUROC difference", error)
                errors["auroc_difference"] = max(errors["auroc_difference"], error)
                assert temporal.loc[identity, "score_path"] == ref["score_path"]
                error = abs(roc[0] - float(temporal.loc[identity, "global_pair_accuracy"]))
                assert error <= TOLERANCE, (identity, "E12 raw concordance", error)
                errors["e12_raw_auroc"] = max(errors["e12_raw_auroc"], error)
                record = dict(zip(keys, identity))
                record.update(base_id=base_id, physical_run_id=ref["physical_run_id"],
                    score_path=ref["score_path"], label_path=ref["label_path"], manifest_path=ref["manifest_path"],
                    points=points, windows=points // 12, dropped_tail=len(official_labels) - points,
                    positive_points=positives, negative_points=points - positives, source_points=len(source),
                    floor_windows=int((std < epsilon).sum()), total_transitions=total.tolist(),
                    within_transitions=within.tolist(), cross_transitions=cross.tolist(),
                    total_concordance=all_concordance, cross_concordance=cross_concordance,
                    raw_pr_auc=pr["raw"], h512_pr_auc=pr["z512"], raw_auroc=roc[0], h512_auroc=roc[1])
                records.append(record)
                arms[key] = record
                print(f"Validated {base_id} {key}", file=sys.stderr, flush=True)
            frozen = arms["shared:frozen"]
            for optimizer in OPTIMIZERS:
                oracle = arms[f"{optimizer}:oracle"]
                delta_raw = oracle["raw_pr_auc"] - frozen["raw_pr_auc"]
                delta_cal = oracle["h512_pr_auc"] - frozen["h512_pr_auc"]
                paired.append(dict(dataset=dataset, subset=subset, seed=seed, optimizer=optimizer,
                    frozen_id=frozen["physical_run_id"], oracle_id=oracle["physical_run_id"],
                    delta_raw=delta_raw, delta_cal=delta_cal, calibration_interaction=delta_cal - delta_raw,
                    frozen_cross_change=frozen["cross_concordance"]["change"],
                    oracle_cross_change=oracle["cross_concordance"]["change"],
                    cross_interaction=oracle["cross_concordance"]["change"] - frozen["cross_concordance"]["change"]))
    assert len(records) == len(ids) == len(paths) == 45 and len(paired) == 30
    json.dump(dict(physical=records, paired=paired, validation_max_errors=errors,
                   tolerance=TOLERANCE, counter_examples=24, wall_seconds=time.perf_counter() - started),
              sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
