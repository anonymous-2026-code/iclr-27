#!/usr/bin/env python3
"""Reaggregate existing VUS-PR on the submitted global common support."""
from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed_cpus[0]})

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

START_WALL, START_CPU = time.perf_counter(), time.process_time()
START_TIME = datetime.now(timezone.utc).isoformat()
START_LOAD = os.getloadavg()

import numpy as np
import pandas as pd
import pyarrow as pa

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = "range-volume-v1-window12-thresholds250"
PR_VERSION = "sklearn-pr-curve-trapezoid-v1"
CATEGORIES = ("BI", "OOG", "NI", "POG")
EXPECTED = {"Daphnet": 1, "GECCO": 1, "Genesis": 1, "MSL": 25,
            "PSM": 1, "PUMP": 1, "SMAP": 52, "SMD": 23, "SWAT2015": 1}
OPTIMIZERS = ("sgd_ttlr0.001", "sgd_ttlr0.0001", "adamw_ttlr0.0001", "adamw_ttlr0.0003")
SEEDS = tuple(range(42, 47))
KEYS = ["dataset", "subset", "seed", "optimizer_id"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def category(practical, oracle):
    p, o = practical > 0.005, oracle > 0.005
    return "BI" if p and o else "POG" if p else "OOG" if o else "NI"


def paired_effects(physical):
    fields = ["vus_pr", "corrected_pr_curve_auc", "source_checkpoint_hash",
              "data_fingerprint", "label_sha256", "n_total"]
    parts = {}
    for arm, prefix in (("frozen", "f"), ("practical_static_p95", "p"),
                        ("oracle_benchmark_normal_point", "o")):
        keys = KEYS[:-1] if prefix == "f" else KEYS
        current = physical.loc[physical.arm == arm, keys + fields].copy()
        parts[prefix] = current.rename(columns={name: f"{prefix}_{name}" for name in fields})
    paired = parts["p"].merge(parts["o"], on=KEYS, validate="one_to_one")
    paired = paired.merge(parts["f"], on=KEYS[:-1], validate="many_to_one")
    require(len(paired) == 2120, "Expected 2120 matched physical triples")
    for name in ("source_checkpoint_hash", "data_fingerprint", "label_sha256", "n_total"):
        require((paired[f"f_{name}"] == paired[f"p_{name}"]).all()
                and (paired[f"f_{name}"] == paired[f"o_{name}"]).all(),
                f"Paired source/data/label identity differs: {name}")
    for metric, field in (("vus", "vus_pr"), ("auprc", "corrected_pr_curve_auc")):
        for arm, prefix in (("practical", "p"), ("oracle", "o")):
            paired[f"{metric}_{arm}_delta"] = paired[f"{prefix}_{field}"] - paired[f"f_{field}"]
    return paired


def load_inputs():
    derived = ROOT / "workspace/rebuttal_inputs"
    support = pd.read_csv(derived / "p3_iclr27_corrected_auprc_common_support_manifest.csv",
                          keep_default_na=False)
    support = support[support.support_status == "retained"]
    require(support.groupby("dataset").size().to_dict() == EXPECTED,
            "Canonical support does not match the prescribed 106 entities")
    retained = {(r.dataset, "" if r.entity == "__root__" else r.entity) for r in support.itertuples()}
    require(len(retained) == 106, "Duplicate support entity")
    canonical = pd.read_parquet(derived / "p3_iclr27_main_auprc_corrected_physical.parquet")
    original = pd.read_parquet(derived / "iclr27_main_v1_physical.parquet")
    vus = original[(original.profile_id == "core_oracle_heterogeneity") & (original.metric == "vus_pr")].copy()
    for frame in (canonical, vus):
        frame["subset"] = frame["subset"].fillna("")
        require(len(frame) == 5220 and frame.physical_run_id.nunique() == 5220,
                "Expected unique 5220-run core physical inventory")
    require(set(canonical.physical_run_id) == set(vus.physical_run_id), "VUS and PR-AUC inventories differ")
    physical = canonical.merge(vus, on="physical_run_id", suffixes=("", "_vus"), validate="one_to_one")
    for name in ("dataset", "subset", "seed", "arm", "optimizer_id", "backbone", "profile_id"):
        require((physical[name] == physical[f"{name}_vus"]).all(), f"Joined identity differs: {name}")
    physical = physical[[ (r.dataset, r.subset) in retained for r in physical.itertuples() ]].copy()
    require(len(physical) == 4770, "Expected 4770 retained physical results")
    expected_keys = {(d, s, seed, "shared", "frozen") for d, s in retained for seed in SEEDS}
    expected_keys |= {(d, s, seed, opt, arm) for d, s in retained for seed in SEEDS
                     for opt in OPTIMIZERS for arm in ("practical_static_p95", "oracle_benchmark_normal_point")}
    actual_keys = set(physical[[*KEYS, "arm"]].itertuples(index=False, name=None))
    require(actual_keys == expected_keys, "Retained physical keys differ from the fixed grid")
    require((physical.metric_version == VERSION).all(), "Unexpected VUS implementation version")
    require((physical.corrected_metric_version == PR_VERSION).all(), "Unexpected corrected PR-AUC version")
    for suffix in ("", "_vus"):
        require((physical[f"n_valid{suffix}"] == physical[f"n_total{suffix}"]).all(),
                "Joint-finite filtering is forbidden on the retained support")
    require((physical.n_total == physical.n_total_vus).all(), "Joined score lengths differ")
    require(np.isfinite(physical[["value", "corrected_pr_curve_auc"]].to_numpy()).all(),
            "Undefined retained metric; no entity may be removed")
    require(physical[["value", "corrected_pr_curve_auc"]].ge(0).all().all()
            and physical[["value", "corrected_pr_curve_auc"]].le(1).all().all(), "Metric outside [0,1]")
    for name in ("source_checkpoint_hash", "data_fingerprint", "label_sha256"):
        require(physical[name].notna().all() and physical[name].astype(str).str.len().eq(64).all(),
                f"Missing identity metadata: {name}")
    columns = ["physical_run_id", *KEYS, "arm", "value", "corrected_pr_curve_auc", "n_total",
               "source_checkpoint_hash", "data_fingerprint", "label_sha256", "score_sha256",
               "base_score_sha256", "score_path", "label_path", "metric_version", "corrected_metric_version"]
    physical = physical[columns].rename(columns={"value": "vus_pr"})
    paired = paired_effects(physical)
    effects = [f"{m}_{a}_delta" for m in ("vus", "auprc") for a in ("practical", "oracle")]
    group_keys = ["dataset", "optimizer_id", "seed"]
    cells = paired.groupby(group_keys, sort=True)[effects].median().reset_index()
    cells["retained_entities"] = paired.groupby(group_keys, sort=True).size().to_numpy()
    require(len(cells) == 180, "Expected 180 complete cells")
    for metric in ("vus", "auprc"):
        cells[f"{metric}_category"] = [category(p, o) for p, o in
            cells[[f"{metric}_practical_delta", f"{metric}_oracle_delta"]].itertuples(index=False, name=None)]
    old = pd.read_csv(derived / "p3_iclr27_corrected_auprc_common_support_cells.csv").rename(columns={"optimizer": "optimizer_id"})
    checked = cells.merge(old, on=group_keys, validate="one_to_one")
    require(len(checked) == 180, "Canonical cell keys differ")
    for arm in ("practical", "oracle"):
        np.testing.assert_allclose(checked[f"auprc_{arm}_delta"], checked[f"corrected_auprc_{arm}_delta"],
                                   rtol=0, atol=1e-14)
    require((checked.auprc_category == checked.corrected_auprc_category).all(), "Canonical category mismatch")
    require(cells.auprc_category.value_counts().to_dict() == {"NI": 90, "OOG": 63, "BI": 27},
            "Canonical headline counts differ")
    return physical, paired, cells


def summarize(cells):
    counts = pd.DataFrame([{"category": c, "auprc_cells": int((cells.auprc_category == c).sum()),
                           "vus_pr_cells": int((cells.vus_category == c).sum())} for c in CATEGORIES])
    transitions = pd.crosstab(cells.auprc_category, cells.vus_category).reindex(
        index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    transitions = transitions.rename_axis(index="auprc_category", columns="vus_category").stack().rename("cells").reset_index()
    rows = []
    for dataset, group in cells.groupby("dataset", sort=True):
        row = {"dataset": dataset, "cells": len(group),
               "category_agreement": float((group.auprc_category == group.vus_category).mean())}
        for metric in ("vus", "auprc"):
            for arm in ("practical", "oracle"):
                values = group[f"{metric}_{arm}_delta"]
                for stat in ("mean", "median", "min", "max"):
                    row[f"{metric}_{arm}_{stat}"] = float(getattr(values, stat)())
            for category_name in CATEGORIES:
                row[f"{metric}_{category_name}"] = int((group[f"{metric}_category"] == category_name).sum())
        rows.append(row)
    dataset_summary = pd.DataFrame(rows)
    associations = []
    for arm in ("practical", "oracle"):
        x, y = f"auprc_{arm}_delta", f"vus_{arm}_delta"
        for level, frame in (("180_repeated_cells", cells),
                             ("9_dataset_medians", cells.groupby("dataset")[[x, y]].median())):
            associations.append({"arm": arm, "level": level, "n": len(frame),
                                 "spearman_rho": float(frame[x].corr(frame[y], method="spearman")),
                                 "sign_agreement": float((np.sign(frame[x]) == np.sign(frame[y])).mean())})
    return counts, transitions, dataset_summary, pd.DataFrame(associations)


def plot(cells, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = dict(zip(sorted(EXPECTED), plt.get_cmap("tab10").colors))
    for ax, arm in zip(axes, ("practical", "oracle")):
        for dataset, group in cells.groupby("dataset", sort=True):
            ax.scatter(100 * group[f"auprc_{arm}_delta"], 100 * group[f"vus_{arm}_delta"],
                       s=16, alpha=0.65, color=colors[dataset], label=dataset)
        ax.axvline(0, color="gray", linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title=f"{arm.capitalize()} minus Frozen", xlabel="PR-AUC paired effect (pp)",
               ylabel="VUS-PR paired effect (pp)")
    axes[1].legend(fontsize=7, loc="best", frameon=False)
    fig.savefig(output / "metric_effect_comparison.png", dpi=180)
    fig.savefig(output / "metric_effect_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = HERE / "derived"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    physical, paired, cells = load_inputs()
    counts, transitions, datasets, associations = summarize(cells)
    output.mkdir()
    for name, frame in (("physical", physical), ("paired_entities", paired), ("cells", cells),
                        ("outcome_counts", counts), ("transitions", transitions),
                        ("dataset_summary", datasets), ("associations", associations)):
        frame.to_csv(output / f"{name}.csv", index=False)
    plot(cells, output)
    agreement = float((cells.auprc_category == cells.vus_category).mean())
    summary = {"status": "complete", "retained_entities": 106, "physical_results": 4770,
               "physical_comparisons": 2120, "logical_cells": 180, "new_model_runs": 0,
               "vus_recomputed": False, "vus_metric_version": VERSION, "auprc_metric_version": PR_VERSION,
               "descriptive_vus_gain_cutoff": 0.005, "category_agreement": agreement,
               "outcome_counts": counts.to_dict("records"), "associations": associations.to_dict("records"),
               "canonical_auprc_regression": "passed", "source_label_pairing": "passed",
               "wall_seconds": time.perf_counter() - START_WALL,
               "process_cpu_seconds": time.process_time() - START_CPU,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "started_at": START_TIME, "finished_at": datetime.now(timezone.utc).isoformat(),
               "system_load_start": START_LOAD, "system_load_end": os.getloadavg(),
               "cpu_affinity": sorted(os.sched_getaffinity(0)), "pyarrow_cpu_threads": pa.cpu_count(),
               "pyarrow_io_threads": pa.io_thread_count(),
               "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# VUS-PR metric robustness results", "",
             "Existing VUS-PR values are reused for the fixed submitted 106 entities, 4,770 physical results, and 180 cells. No training, TTA, or VUS recomputation is performed.",
             "Effects relative to Frozen are paired within entities, then summarized by dataset medians. Corrected PR-AUC continuous effects and the 27/63/90/0 categories are reproduced.", "",
             "| Category | Submitted PR-AUC | VUS-PR |", "|---|---:|---:|"]
    lines += [f"| {r.category} | {r.auprc_cells} | {r.vus_pr_cells} |" for r in counts.itertuples()]
    lines += ["", f"Category agreement is {agreement:.2%}. VUS delta > 0.005 is a descriptive absolute-effect threshold for this analysis; it does not imply the same application utility as the PR-AUC threshold. NI includes near-zero changes and decreases.", "",
              "| Dataset | Median PR-AUC Oracle-Frozen effect (pp) | Median VUS-PR Oracle-Frozen effect (pp) | Category agreement |",
              "|---|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {100*r.auprc_oracle_median:.4f} | {100*r.vus_oracle_median:.4f} | {r.category_agreement:.1%} |"
              for r in datasets.itertuples()]
    lines += ["", "| Comparison | Aggregation level | Spearman rho | Positive/zero/negative direction agreement |", "|---|---|---:|---:|"]
    lines += [f"| {r.arm} | {r.level} | {r.spearman_rho:.4f} | {r.sign_agreement:.2%} |" for r in associations.itertuples()]
    lines += ["", "The 180 cells contain optimizer/seed repetitions within the same datasets. They are not treated as 180 independent datasets for significance testing. Metric association is not causal evidence, and VUS and point-level PR-AUC measure different targets.",
              f"Reused version: `{VERSION}`. Metadata and pairing checks do not independently validate the VUS implementation.",
              "", "XXXX", "",
              "Full continuous effects and ranges: [dataset_summary.csv](dataset_summary.csv). Cells and category transitions: [cells.csv](cells.csv), [transitions.csv](transitions.csv).",
              f"", f"Process wall time={summary['wall_seconds']:.2f}s, CPU time={summary['process_cpu_seconds']:.2f}s, max RSS={summary['max_rss_kib']/1024:.1f} MiB; CPU affinity={summary['cpu_affinity']}. All computation threads are set to 1. Machine load is in [summary.json](summary.json).", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
