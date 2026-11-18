#!/usr/bin/env python3
"""CPU-only source/test-normal marginal shift and Frozen headroom analysis."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

START = time.perf_counter()
CPU_START = time.process_time()
LOAD_START = os.getloadavg()
STARTED_AT = datetime.now(timezone.utc).isoformat()

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = {"Daphnet": 1, "GECCO": 1, "Genesis": 1, "MSL": 25,
            "PSM": 1, "PUMP": 1, "SMAP": 52, "SMD": 23, "SWAT2015": 1}
SEEDS = tuple(range(42, 47))
OPTS = ("sgd_ttlr0.001", "sgd_ttlr0.0001", "adamw_ttlr0.0001", "adamw_ttlr0.0003")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs():
    root = ROOT / "workspace/rebuttal_inputs"
    support = pd.read_csv(root / "p3_iclr27_corrected_auprc_common_support_manifest.csv", keep_default_na=False)
    kept = support[support.support_status == "retained"].copy()
    require(kept.groupby("dataset").size().to_dict() == EXPECTED, "Canonical support differs")
    kept["subset"] = kept.entity.replace("__root__", "")
    physical = pd.read_parquet(root / "p3_iclr27_main_auprc_corrected_physical.parquet").fillna({"subset": ""})
    physical = physical.merge(kept[["dataset", "subset"]], on=["dataset", "subset"], validate="many_to_one")
    require(len(physical) == 4770, "Expected 4770 retained physical records")
    require((physical.n_valid == physical.n_total).all(), "Retained score arrays were filtered")
    require((physical.corrected_metric_version == "sklearn-pr-curve-trapezoid-v1").all(), "Incorrect metric version")
    values = physical.corrected_pr_curve_auc.to_numpy()
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid corrected metrics")
    keys = ["dataset", "subset", "seed", "optimizer_id", "arm"]
    require(not physical.duplicated(keys).any(), "Duplicate physical reference")
    entities = set(kept[["dataset", "subset"]].itertuples(index=False, name=None))
    expected_keys = {(d, s, seed, "shared", "frozen") for d, s in entities for seed in SEEDS}
    expected_keys |= {(d, s, seed, opt, arm) for d, s in entities for seed in SEEDS
                     for opt in OPTS for arm in ("practical_static_p95", "oracle_benchmark_normal_point")}
    require(set(physical[keys].itertuples(index=False, name=None)) == expected_keys, "Physical grid mismatch")
    frozen = physical[physical.arm == "frozen"].copy()
    frozen["headroom"] = 1 - frozen.corrected_pr_curve_auc
    frozen["frozen_ge_0_99"] = frozen.corrected_pr_curve_auc >= 0.99
    frozen["clear_gain_impossible"] = frozen.headroom <= 0.005
    base = frozen[["dataset", "subset", "seed", "corrected_pr_curve_auc"]].rename(columns={"corrected_pr_curve_auc": "frozen_auprc"})
    effects = physical[physical.arm != "frozen"].merge(base, on=["dataset", "subset", "seed"], validate="many_to_one")
    effects["delta"] = effects.corrected_pr_curve_auc - effects.frozen_auprc
    rebuilt = effects.groupby(["dataset", "optimizer_id", "seed", "arm"]).delta.median().unstack("arm").reset_index()
    old = pd.read_csv(root / "p3_iclr27_corrected_auprc_common_support_cells.csv").rename(columns={"optimizer": "optimizer_id"})
    checked = rebuilt.merge(old, on=["dataset", "optimizer_id", "seed"], validate="one_to_one")
    require(len(checked) == 180, "Canonical logical grid mismatch")
    for arm, name in (("practical_static_p95", "practical"), ("oracle_benchmark_normal_point", "oracle")):
        np.testing.assert_allclose(checked[arm], checked[f"corrected_auprc_{name}_delta"], rtol=0, atol=1e-14)
    bounds = frozen.groupby(["dataset", "seed"]).headroom.median().rename("paired_effect_upper_bound").reset_index()
    bounds["clear_gain_impossible"] = bounds.paired_effect_upper_bound <= 0.005
    cells = old.merge(bounds, on=["dataset", "seed"], validate="many_to_one")
    require((cells.corrected_auprc_oracle_delta <= cells.paired_effect_upper_bound + 1e-14).all(), "Oracle exceeds exact headroom bound")
    require(not (cells.clear_gain_impossible & (cells.corrected_auprc_oracle_delta > 0.005)).any(), "Ceiling contradicts clear-gain classification")
    return kept, frozen, effects, bounds, cells


def compute_shift(kept, frozen):
    feature_rows, entity_rows = [], []
    for index, r in enumerate(kept.sort_values(["dataset", "subset"]).itertuples(), 1):
        directory = ROOT / "data/processed" / r.dataset
        if r.subset:
            directory /= r.subset
        source_frame = pd.read_parquet(directory / "train.parquet")
        test_frame = pd.read_parquet(directory / "test.parquet")
        require(list(source_frame.columns) == list(test_frame.columns), f"Column mismatch: {directory}")
        require(source_frame.columns[-1] == "label", f"Missing label column: {directory}")
        source_raw = source_frame.iloc[:, :-1].to_numpy()
        test_raw = test_frame.iloc[:, :-1].to_numpy()
        labels = test_frame.iloc[:, -1].to_numpy()
        require(np.isfinite(source_raw).all() and np.isfinite(test_raw).all(), f"Nonfinite features: {directory}")
        require(set(np.unique(labels)).issubset({0, 1}), f"Nonbinary test labels: {directory}")
        # Match the original model preprocessing; source labels are not used to select rows.
        scaler = StandardScaler().fit(source_raw)
        source = scaler.transform(source_raw).astype(np.float32)
        test = scaler.transform(test_raw).astype(np.float32)
        split = int(len(source) * 0.8)
        prefix = len(test) // 12 * 12
        require(0 < split < len(source) and prefix > 0, f"Invalid source/test length: {directory}")
        labels = labels[:prefix]
        references = frozen[(frozen.dataset == r.dataset) & (frozen.subset == r.subset)]
        require(len(references) == 5, f"Frozen seed grid differs: {directory}")
        for reference in references.itertuples():
            require(np.array_equal(labels, np.load(reference.label_path, allow_pickle=False).reshape(-1)),
                    f"Test prefix differs from canonical labels: {directory}")
        train, validation = source[:split], source[split:]
        normal = test[:prefix][labels == 0]
        require(len(normal) > 0, f"No benchmark-normal evaluation point: {directory}")
        distances, references_w, means = [], [], []
        constant = scaler.var_ == 0
        for j, name in enumerate(source_frame.columns[:-1]):
            wt = float(wasserstein_distance(train[:, j], normal[:, j]))
            wv = float(wasserstein_distance(train[:, j], validation[:, j]))
            mean_shift = abs(float(train[:, j].mean(dtype=np.float64) - normal[:, j].mean(dtype=np.float64)))
            require(mean_shift <= wt + 1e-10 * max(1, wt), f"Mean-shift/W1 inequality failed: {directory}/{name}")
            feature_rows.append({"dataset": r.dataset, "subset": r.subset, "feature": str(name),
                "source_mean": float(scaler.mean_[j]), "source_scale": float(scaler.scale_[j]),
                "constant_source_feature": bool(constant[j]), "w_train_testnormal": wt,
                "w_train_validation": wv, "absolute_mean_shift_test": mean_shift})
            distances.append(wt); references_w.append(wv); means.append(mean_shift)
        distances, references_w = np.array(distances), np.array(references_w)
        active = ~constant
        entity_rows.append({"dataset": r.dataset, "subset": r.subset,
            "source_train_points": len(train), "source_validation_points": len(validation),
            "evaluation_points": prefix, "benchmark_normal_points": len(normal),
            "test_anomaly_fraction": float((labels == 1).mean()),
            "source_label_nonzero_points": int((source_frame.iloc[:, -1].to_numpy() != 0).sum()),
            "feature_count": len(constant), "constant_source_features": int(constant.sum()),
            "constant_features_changed_in_test": int((constant & (distances > 0)).sum()),
            "drift_median": float(np.median(distances)), "drift_mean": float(distances.mean()),
            "reference_median": float(np.median(references_w)), "reference_mean": float(references_w.mean()),
            "excess_over_source_reference": float(np.median(distances) - np.median(references_w)),
            "absolute_mean_shift_median": float(np.median(means)),
            "nonconstant_drift_median": float(np.median(distances[active])) if active.any() else np.nan,
            "nonconstant_reference_median": float(np.median(references_w[active])) if active.any() else np.nan,
            "nonconstant_diagnostic_status": "defined" if active.any() else "undefined:no-nonconstant-source-feature"})
        print(f"ENTITY {index}/106 {r.dataset}/{r.subset} normal_points={len(normal)} features={len(constant)} load={os.getloadavg()[0]:.2f}", flush=True)
    return pd.DataFrame(feature_rows), pd.DataFrame(entity_rows)


def summaries(entity, frozen, effects, cells):
    f_entity = frozen.groupby(["dataset", "subset"]).corrected_pr_curve_auc.median().rename("frozen_median_over_seeds")
    o_entity = effects[effects.arm == "oracle_benchmark_normal_point"].groupby(["dataset", "subset"]).delta.median().rename("oracle_delta_median_over_configs")
    entity = entity.merge(f_entity, on=["dataset", "subset"], validate="one_to_one").merge(o_entity, on=["dataset", "subset"], validate="one_to_one")
    rows = []
    drift_fields = ("drift_median", "drift_mean", "reference_median", "reference_mean",
                    "excess_over_source_reference", "nonconstant_drift_median", "nonconstant_reference_median")
    for dataset, group in entity.groupby("dataset", sort=True):
        f = frozen[frozen.dataset == dataset]
        c = cells[cells.dataset == dataset]
        rows.append({"dataset": dataset, "entities": len(group),
            **{name: float(group[name].median()) for name in drift_fields},
            "nonconstant_defined_entities": int(group.nonconstant_drift_median.notna().sum()),
            "features_total": int(group.feature_count.sum()), "constant_source_features": int(group.constant_source_features.sum()),
            "frozen_median": float(f.corrected_pr_curve_auc.median()),
            "frozen_min": float(f.corrected_pr_curve_auc.min()), "frozen_max": float(f.corrected_pr_curve_auc.max()),
            "headroom_median": float(f.headroom.median()),
            "entity_seed_count": len(f), "frozen_ge_0_99": int(f.frozen_ge_0_99.sum()),
            "entity_seed_clear_gain_impossible": int(f.clear_gain_impossible.sum()),
            "oracle_delta_median": float(c.corrected_auprc_oracle_delta.median()),
            "oracle_delta_min": float(c.corrected_auprc_oracle_delta.min()),
            "oracle_delta_max": float(c.corrected_auprc_oracle_delta.max()),
            "NI_cells": int((c.corrected_auprc_category == "NI").sum()),
            "ceiling_limited_cells": int(c.clear_gain_impossible.sum()),
            "ceiling_limited_NI_cells": int((c.clear_gain_impossible & (c.corrected_auprc_category == "NI")).sum()),
            "normal_points": int(group.benchmark_normal_points.sum()),
            "evaluation_points": int(group.evaluation_points.sum())})
    datasets = pd.DataFrame(rows)
    associations = []
    for feature in ("drift_median", "drift_mean", "excess_over_source_reference", "nonconstant_drift_median", "headroom_median"):
        valid = datasets[[feature, "oracle_delta_median"]].dropna()
        defined = len(valid) >= 3 and valid[feature].nunique() > 1 and valid.oracle_delta_median.nunique() > 1
        rho = float(spearmanr(valid[feature], valid.oracle_delta_median).statistic) if defined else None
        associations.append({"predictor": feature, "outcome": "canonical_dataset_median_oracle_delta",
                             "datasets": len(valid), "spearman_rho": rho,
                             "status": "descriptive" if defined else "undefined:insufficient-variation"})
    return entity, datasets, pd.DataFrame(associations)


def make_plot(datasets, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for color, row in zip(colors, datasets.itertuples()):
        points = [(row.reference_median, row.drift_median),
                  (row.drift_median, 100 * row.oracle_delta_median),
                  (row.frozen_median, 100 * row.oracle_delta_median)]
        for panel, (ax, (x, y)) in enumerate(zip(axes, points)):
            ax.scatter(x, y, s=30, color=color)
            offsets = {
                (0, "Daphnet"): (-45, -12), (0, "PSM"): (-28, 5),
                (1, "Genesis"): (22, 24), (1, "MSL"): (4, -25),
                (1, "SMAP"): (26, -10), (1, "PUMP"): (-32, 5),
                (2, "Genesis"): (4, 18), (2, "MSL"): (18, 8),
                (2, "SMAP"): (4, -22),
            }
            offset = offsets.get((panel, row.dataset), (4, 4))
            ax.annotate(row.dataset, (x, y), xytext=offset, textcoords="offset points", fontsize=7,
                        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": color}
                        if (panel, row.dataset) in offsets else None)
    axes[0].set(xlabel="Source-train vs validation W1", ylabel="Source-train vs test-normal W1",
                title="Marginal shift (dataset medians)")
    axes[0].set_xscale("symlog", linthresh=0.001)
    axes[0].set_yscale("symlog", linthresh=0.001)
    axes[1].set(xlabel="Source-train vs test-normal W1", ylabel="Oracle minus Frozen (pp)", title="Shift and detection effect")
    axes[1].set_xscale("symlog", linthresh=0.001)
    axes[2].set(xlabel="Frozen PR-AUC (entity/seed median)", ylabel="Oracle minus Frozen (pp)", title="Frozen performance and effect", xlim=(-0.03, 1.03))
    for ax in axes[1:]:
        ax.axhline(0, color="gray", linewidth=0.7)
    fig.savefig(output / "shift_headroom.png", dpi=180)
    fig.savefig(output / "shift_headroom.pdf")
    plt.close(fig)


def main():
    output = HERE / "derived"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}")
    kept, frozen, effects, bounds, cells = load_inputs()
    features, entity = compute_shift(kept, frozen)
    entity, datasets, associations = summaries(entity, frozen, effects, cells)
    output.mkdir()
    frames = {"feature_shift": features, "entity_shift": entity,
              "frozen_headroom": frozen, "dataset_seed_headroom_bounds": bounds,
              "canonical_cells_with_headroom": cells, "dataset_summary": datasets, "associations": associations}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    make_plot(datasets, output)
    summary = {"status": "complete", "entities": 106, "frozen_entity_seed_records": 530,
        "logical_cells": 180, "features": len(features), "source_labels_are_not_independent_normality_evidence": True,
        "constant_source_features": int(features.constant_source_feature.sum()),
        "nonconstant_diagnostic_undefined_entities": int(entity.nonconstant_drift_median.isna().sum()),
        "frozen_ge_0_99": int(frozen.frozen_ge_0_99.sum()),
        "entity_seed_ceiling_limited": int(frozen.clear_gain_impossible.sum()),
        "NI_cells": int((cells.corrected_auprc_category == "NI").sum()),
        "ceiling_limited_NI_cells": int((cells.clear_gain_impossible & (cells.corrected_auprc_category == "NI")).sum()),
        "datasets_test_shift_above_source_reference": int((datasets.excess_over_source_reference > 0).sum()),
        "associations": [{**r, "spearman_rho": None if pd.isna(r["spearman_rho"]) else r["spearman_rho"]}
                         for r in associations.to_dict("records")], "new_model_runs": 0,
        "canonical_effect_regression": "passed", "label_alignment": "passed", "headroom_bound": "passed",
        "wall_seconds": time.perf_counter() - START, "process_cpu_seconds": time.process_time() - CPU_START,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "started_at": STARTED_AT, "finished_at": datetime.now(timezone.utc).isoformat(),
        "system_load_start": LOAD_START, "system_load_end": os.getloadavg(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "pyarrow_cpu_threads": pa.cpu_count(), "pyarrow_io_threads": pa.io_thread_count(),
        "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# Normal-distribution shift and Frozen headroom: results", "",
        "The submitted 106 entities, 530 Frozen entity-seed observations, and 180 logical cells are fixed. No retraining or TTA was performed. Paired performance effects reproduce canonical results.",
        "Shift comparisons use the original source-training distribution and benchmark-normal test points in the scoring prefix. Some source labels were set to zero during preprocessing, which does not independently certify semantic normality.", "",
        "| Dataset | W1 test (entity median) | W1 validation (entity median) | Entity median of feature median differences | Median Frozen AUPRC | Median Oracle gain (pp) | NI cells | Ceiling-limited NI cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {r.drift_median:.5g} | {r.reference_median:.5g} | {r.excess_over_source_reference:.5g} | {r.frozen_median:.5f} | {100*r.oracle_delta_median:.4f} | {r.NI_cells} | {r.ceiling_limited_NI_cells} |" for r in datasets.itertuples()]
    lines += ["", f"Of 530 unique Frozen observations, {summary['entity_seed_ceiling_limited']} have headroom no greater than 0.005. The mathematical upper bound on dataset paired effects places {summary['ceiling_limited_NI_cells']} of the original 90 NI cells under the same hard ceiling.",
        "This bound is the median entity headroom within a dataset-seed, not the difference between independently aggregated arm medians. Metric headroom in other cells does not establish an exploitable adaptation opportunity for the current reconstruction update.", "",
        "| Predefined descriptive metric | Datasets | Spearman rho with Oracle gain |", "|---|---:|---:|"]
    for r in associations.itertuples():
        rho_text = f"{r.spearman_rho:.4f}" if pd.notna(r.spearman_rho) else "undefined"
        lines.append(f"| {r.predictor} | {r.datasets} | {rho_text} |")
    lines += ["", f"Of {len(features)} entity features, {summary['constant_source_features']} are constant on source data. The primary analysis retains the original model StandardScaler (constant scale=1); these distances are not all in source-standard-deviation units. Nonconstant-feature sensitivity and individual feature distances are fully saved.",
        "The primary association uses only nine equally weighted datasets. The 180 repeated configurations are not treated as independent samples for significance testing. Feature medians may hide sparse changes; interpret them alongside means and nonconstant-feature sensitivity.",
        "Source-validation differences provide a source-domain temporal reference, not a no-drift null hypothesis. All W1 values are one-dimensional empirical marginal distances; they do not cover joint dependencies, periodic/spectral structure, or full temporal dynamics. Low W1 does not establish absence of shift, and high W1 does not establish that adaptation should help. This report describes associations, not causal explanations.", "",
        "XXXX", "",
        "Detailed tables: [dataset_summary.csv](dataset_summary.csv), [entity_shift.csv](entity_shift.csv), [feature_shift.csv](feature_shift.csv), [frozen_headroom.csv](frozen_headroom.csv), [canonical_cells_with_headroom.csv](canonical_cells_with_headroom.csv).",
        f"Process wall time {summary['wall_seconds']:.2f}s, CPU {summary['process_cpu_seconds']:.2f}s, peak RSS {summary['max_rss_kib']/1024:.1f} MiB; single-core affinity={summary['cpu_affinity']}, nice={summary['nice']}.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
