#!/usr/bin/env python3
"""Aggregate only the complete, prespecified all-normal-window experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = "sklearn-pr-curve-trapezoid-v1"


def category(practical, oracle):
    p, o = practical > 0.005, oracle > 0.005
    return "BI" if p and o else "POG" if p else "OOG" if o else "NI"


def paired_cells(frame):
    """Pair physical effects before taking medians, including window minus point."""
    rows = []
    for (dataset, optimizer, seed), group in frame.groupby(["dataset", "optimizer", "seed"], sort=True):
        p = float(group.delta_practical_vs_frozen.median())
        o = float(group.delta_point_vs_frozen.median())
        w = float(group.delta_window_vs_frozen.median())
        rows.append({"dataset": dataset, "optimizer": optimizer, "seed": int(seed),
                     "entity_count": len(group), "practical_delta": p, "point_oracle_delta": o,
                     "window_oracle_delta": w,
                     "window_minus_point_delta": float(group.delta_window_vs_point.median()),
                     "original_category": category(p, o), "new_category": category(p, w),
                     "median_normal_coverage": float(group.normal_coverage.median())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=HERE)
    args = parser.parse_args()
    directory = args.experiment_root.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    support = pd.read_csv(ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_manifest.csv", keep_default_na=False)
    retained = {(r.dataset, "" if r.entity == "__root__" else r.entity)
                for r in support.itertuples() if r.support_status == "retained"}
    if len(retained) != 106:
        raise ValueError("Expected the submitted 106-entity support")
    optimizers = [r["key"] for r in plan["source_protocol"]["tta_configs"]]
    seeds = plan["source_protocol"]["seeds"]
    expected_keys = {(d, s, seed, opt) for d, s in retained for seed in seeds for opt in optimizers}
    if len(expected_keys) != 2120:
        raise ValueError("Expected 2120 prespecified physical comparisons")
    base = pd.read_parquet(ROOT / "workspace/rebuttal_inputs/p3_iclr27_main_auprc_corrected_physical.parquet").fillna({"subset": ""})
    lookup = {(r.dataset, r.subset, int(r.seed), r.optimizer_id, r.arm): r for r in base.itertuples()}
    if len(lookup) != len(base):
        raise ValueError("Duplicate canonical physical reference")
    raw = directory / "outputs/results/supplements/ae_all_normal_window"
    expected_paths = {raw / opt / d / (s or "__root__") / f"seed{seed}" / "result.json"
                      for d, s, seed, opt in expected_keys}
    if set(raw.glob("*/*/*/seed*/result.json")) != expected_paths:
        raise ValueError("Formal outputs are incomplete or outside the prespecified grid; refusing full summary")
    rows = []
    for dataset, subset, seed, optimizer in sorted(expected_keys):
        run = raw / optimizer / dataset / (subset or "__root__") / f"seed{seed}"
        payload = (run / "result.json").read_bytes()
        if payload != (run / "completed.json").read_bytes():
            raise ValueError(f"Mismatched completion record: {run}")
        meta = json.loads(payload)
        identity = {"dataset": dataset, "subset": subset, "seed": seed,
                    "optimizer_key": optimizer, "metric_version": VERSION,
                    "complete": True, "rebuttal_arm": "all_normal_window_oracle"}
        if any(meta.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Run identity or metric version differs: {run}")
        labels = np.load(run / "true.npy", allow_pickle=False).reshape(-1)
        score = np.load(run / "test_energy.npy", allow_pickle=False).reshape(-1)
        mask = np.load(run / "update_mask.npy", allow_pickle=False).reshape(-1)
        if labels.shape != score.shape or len(labels) % 12 or not np.isfinite(score).all():
            raise ValueError(f"Invalid full-stream score: {run}")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"Invalid evaluation labels: {run}")
        expected_mask = np.repeat((labels.reshape(-1, 12) == 0).all(axis=1), 12)
        if not np.array_equal(mask, expected_mask):
            raise ValueError(f"Incorrect all-normal-window mask: {run}")
        if meta["optimizer_steps"] != len(labels) // 12 or meta["selected_anomalous_points"] != 0:
            raise ValueError(f"Update schedule or purity violation: {run}")
        precision, recall, _ = precision_recall_curve(labels, score, pos_label=1)
        new_auc = float(auc(recall, precision))
        if abs(new_auc - meta["auprc"]) > 1e-12:
            raise ValueError(f"Stored metric differs from saved scores: {run}")
        refs = [lookup[dataset, subset, seed, "shared", "frozen"],
                lookup[dataset, subset, seed, optimizer, "practical_static_p95"],
                lookup[dataset, subset, seed, optimizer, "oracle_benchmark_normal_point"]]
        for ref in refs:
            if not np.array_equal(labels, np.load(ref.label_path, allow_pickle=False).reshape(-1)):
                raise ValueError(f"Reference label alignment failure: {run} / {ref.arm}")
            manifest = json.loads(Path(ref.manifest_path).read_text())
            if manifest["source_checkpoint_hash"] != meta["source_checkpoint_sha256"]:
                raise ValueError(f"Reference source mismatch: {run} / {ref.arm}")
        frozen, practical, point = [float(r.corrected_pr_curve_auc) for r in refs]
        if not np.isfinite([frozen, practical, point]).all():
            raise ValueError(f"Undefined canonical metric: {run}")
        coverage_fields = ("test_points", "test_windows", "all_normal_windows", "mixed_windows",
                           "all_anomalous_windows", "normal_points_total", "selected_points",
                           "selected_normal_points", "selected_anomalous_points", "normal_points_dropped",
                           "normal_points_dropped_fraction", "all_normal_window_fraction", "normal_coverage")
        normal_total = int((labels == 0).sum())
        if meta["normal_points_total"] != normal_total or meta["selected_points"] != int(mask.sum()):
            raise ValueError(f"Incorrect data-use counts: {run}")
        rows.append({"dataset": dataset, "subset": subset, "seed": seed, "optimizer": optimizer,
                     "frozen_auprc": frozen, "practical_auprc": practical,
                     "point_oracle_auprc": point, "window_oracle_auprc": new_auc,
                     "delta_practical_vs_frozen": practical - frozen,
                     "delta_point_vs_frozen": point - frozen,
                     "delta_window_vs_frozen": new_auc - frozen,
                     "delta_window_vs_point": new_auc - point,
                     **{k: meta[k] for k in coverage_fields},
                     "optimizer_steps": meta["optimizer_steps"], "wall_seconds": meta["wall_seconds"],
                     "metric_version": VERSION, "output_path": str(run)})
    physical = pd.DataFrame(rows)
    cells = paired_cells(physical)
    if len(cells) != 180:
        raise ValueError("Incomplete logical grid")
    old = pd.read_csv(ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv")
    check = cells.merge(old, on=["dataset", "optimizer", "seed"], validate="one_to_one")
    if len(check) != 180:
        raise ValueError("Canonical cell identities do not match")
    for new_col, old_col in (("practical_delta", "corrected_auprc_practical_delta"),
                             ("point_oracle_delta", "corrected_auprc_oracle_delta")):
        np.testing.assert_allclose(check[new_col], check[old_col], rtol=0, atol=1e-14)
    if not (check.original_category == check.corrected_auprc_category).all():
        raise ValueError("Original categories do not reproduce the submission")
    # Data-use counts depend on entity labels, not on optimizer or seed.
    usage_columns = ["dataset", "subset", *coverage_fields]
    usage = physical[usage_columns].drop_duplicates()
    if len(usage) != 106 or usage.duplicated(["dataset", "subset"]).any():
        raise ValueError("Entity label-derived data use changed across seeds/optimizers")
    dataset_usage = usage.groupby("dataset")[["test_windows", "all_normal_windows", "mixed_windows",
        "all_anomalous_windows", "normal_points_total", "selected_normal_points", "normal_points_dropped"]].sum()
    dataset_usage["pooled_normal_coverage"] = dataset_usage.selected_normal_points / dataset_usage.normal_points_total
    dataset_usage["pooled_all_normal_window_fraction"] = dataset_usage.all_normal_windows / dataset_usage.test_windows
    summaries = []
    for dataset, group in cells.groupby("dataset", sort=True):
        summaries.append({"dataset": dataset, "cells": len(group),
            "window_delta_mean": float(group.window_oracle_delta.mean()),
            "window_delta_median": float(group.window_oracle_delta.median()),
            "window_delta_min": float(group.window_oracle_delta.min()),
            "window_delta_max": float(group.window_oracle_delta.max()),
            "window_minus_point_median": float(group.window_minus_point_delta.median()),
            **{name: int((group.new_category == name).sum()) for name in ("BI", "OOG", "NI", "POG")},
            **dataset_usage.loc[dataset].to_dict()})
    dataset_summary = pd.DataFrame(summaries)
    counts = pd.DataFrame([{"category": c, "point_oracle": int((cells.original_category == c).sum()),
                           "window_oracle": int((cells.new_category == c).sum())} for c in ("BI", "OOG", "NI", "POG")])
    transitions = cells.groupby(["original_category", "new_category"]).size().rename("cells").reset_index()
    output = directory / "derived"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output}")
    output.mkdir()
    for name, frame in (("physical", physical), ("cells", cells), ("outcome_counts", counts),
                        ("transitions", transitions), ("dataset_summary", dataset_summary), ("entity_data_use", usage)):
        frame.to_csv(output / f"{name}.csv", index=False)
    summary = {"status": "complete", "physical_runs": 2120, "logical_cells": 180,
               "retained_entities": 106, "metric_version": VERSION,
               "window_oracle_clear_gain_cells": int((cells.window_oracle_delta > 0.005).sum()),
               "window_oracle_negative_effect_cells": int((cells.window_oracle_delta < 0).sum()),
               "outcome_counts": counts.to_dict(orient="records"),
               "summed_worker_run_hours": float(physical.wall_seconds.sum() / 3600),
               "aggregation": "median-of-entity-paired-differences"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Main AE all-normal-window Oracle: complete results", "",
             "This report aggregates all 2,120 physical runs, covering the fixed 106 entities and 180 logical cells.",
             "The primary metric is trapezoidal PR-AUC without point adjustment. Effects are paired within entities before taking dataset medians.", "",
             "| Category | Original point Oracle | New window Oracle |", "|---|---:|---:|"]
    lines += [f"| {r.category} | {r.point_oracle} | {r.window_oracle} |" for r in counts.itertuples()]
    lines += ["", "| Dataset | New Oracle-Frozen median effect (pp) | New Oracle-point Oracle median effect (pp) | Pooled normal-point retention |",
              "|---|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {100*r.window_delta_median:.4f} | {100*r.window_minus_point_median:.4f} | {r.pooled_normal_coverage:.4%} |"
              for r in dataset_summary.itertuples()]
    lines += ["", "Normal-point usage is aggregated over unique entities; seeds and optimizers are not counted as additional data.",
              "NI means the gain does not exceed 0.5 pp; it does not imply that every result deteriorates. The all-normal-window rule both reduces normal update points and changes when update information arrives, so differences cannot be attributed entirely to anomalous inputs.",
              "This report establishes neither a unique failure mechanism nor generalization to untested architectures, update objectives, or semantic normality.", "",
              "Details: [cells.csv](cells.csv), [transitions.csv](transitions.csv), [dataset_summary.csv](dataset_summary.csv), [entity_data_use.csv](entity_data_use.csv).", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
