#!/usr/bin/env python3
"""Frozen normal-score and fixed-source-workpoint diagnostic, CPU only."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import resource
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "causal_score_calibration"))
import analyze as calibration
import numpy as np
import pandas as pd
controls = calibration.controls


def main():
    controls.configure_threads()
    started, cpu_start, load_start = time.perf_counter(), time.process_time(), os.getloadavg()
    out = HERE / "derived"
    if out.exists():
        raise FileExistsError(out)
    plan = calibration.load_plan()
    rows = []
    for base_id, base in sorted(plan["bases"].items()):
        source, meta = calibration.source_scores(base)
        reference = controls.reference(base, "frozen", "shared")
        score = np.load(reference["score_path"], allow_pickle=False).reshape(-1)
        labels = np.load(reference["label_path"], allow_pickle=False).reshape(-1)
        value = controls.metric(labels, score)
        if abs(value - reference["auprc"]) > 1e-12:
            raise ValueError("Frozen canonical metric mismatch")
        normal, abnormal = score[labels == 0], score[labels == 1]
        sd = float(source.std())
        for quantile in (95, 99):
            threshold = float(np.percentile(source, quantile))
            fpr = float((normal > threshold).mean())
            source_exceedance = float((source > threshold).mean())
            rows.append({"dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
                "quantile": quantile, "threshold": threshold, "normal_points": len(normal),
                "anomalous_points": len(abnormal), "source_validation_points": len(source),
                "source_exceedance": source_exceedance, "normal_fpr": fpr,
                "fpr_minus_source_exceedance": fpr - source_exceedance,
                "anomaly_tpr": float((abnormal > threshold).mean()),
                "normal_false_positives": int((normal > threshold).sum()),
                "source_score_mean": float(source.mean()), "source_score_median": float(np.median(source)),
                "source_score_std": sd, "normal_score_mean": float(normal.mean(dtype=np.float64)),
                "normal_score_median": float(np.median(normal)),
                "normal_median_shift_in_source_sd": (float(np.median(normal)) - float(np.median(source))) / sd if sd > 0 else np.nan,
                "standardized_shift_defined": sd > 0, "frozen_auprc": value})
    physical = pd.DataFrame(rows)
    if len(physical) != 1060 or physical.duplicated(["dataset", "subset", "seed", "quantile"]).any():
        raise ValueError("Frozen 530 x 2 diagnostic grid differs")
    fields = ["source_exceedance", "normal_fpr", "fpr_minus_source_exceedance", "anomaly_tpr", "frozen_auprc",
              "source_score_median", "normal_score_median", "normal_median_shift_in_source_sd"]
    grouped = physical.groupby(["dataset", "seed", "quantile"])
    seed_summary = grouped[fields].median()
    seed_summary["entities"] = grouped.size()
    seed_summary["standardized_shift_defined_entities"] = grouped.standardized_shift_defined.sum()
    seed_summary = seed_summary.reset_index()
    summary = seed_summary.groupby(["dataset", "quantile"])[fields + ["entities", "standardized_shift_defined_entities"]].median().reset_index()
    old = pd.read_csv(controls.ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv")
    oracle = old.groupby("dataset").agg(oracle_delta=("corrected_auprc_oracle_delta", "median"),
        original_NI=("corrected_auprc_category", lambda s: int((s == "NI").sum()))).reset_index()
    shift = pd.read_csv(HERE.parent / "normal_shift_headroom/derived/dataset_summary.csv")[["dataset", "drift_median"]]
    summary = summary.merge(oracle, on="dataset", validate="many_to_one").merge(shift, on="dataset", validate="many_to_one")
    out.mkdir()
    for name, frame in (("physical", physical), ("dataset_seed", seed_summary), ("dataset_summary", summary)):
        frame.to_csv(out / f"{name}.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for quantile in (95, 99):
        selected = summary[summary["quantile"] == quantile].sort_values("dataset")
        axes[0].plot(selected.dataset, selected.normal_fpr * 100, "o-", label=f"Source P{quantile}")
    axes[0].set(ylabel="Target-normal FPR (%)", title="Fixed source workpoints (diagnostic)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    selected = summary[summary["quantile"] == 95]
    for r in selected.itertuples():
        axes[1].scatter(r.normal_fpr * 100, r.oracle_delta * 100, s=25)
        offsets = {"Genesis": (7, 24), "MSL": (40, -8), "SMAP": (5, -19), "GECCO": (-43, 7)}
        axes[1].annotate(r.dataset, (r.normal_fpr * 100, r.oracle_delta * 100),
                         xytext=offsets.get(r.dataset, (4, 4)), textcoords="offset points", fontsize=7,
                         arrowprops={"arrowstyle": "-", "lw": .4} if r.dataset in offsets else None)
    axes[1].set(xlabel="Target-normal FPR at source P95 (%)", ylabel="Original Oracle gain over Frozen (pp)")
    axes[1].axhline(0, color="grey", linewidth=.7)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"frozen_need.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# Original Frozen normal scores and diagnostic false positives", "", "All 106 entities and five seeds. Thresholds are P95/P99 of the original validation scores from the same source checkpoint; these are not PR-AUC thresholds or deployment certification.",
        "Dataset tables first take the entity median for each seed, then the median across five seeds. The accompanying Oracle effect is the median over the original 20 cells.", "",
        "| Dataset | Original marginal W1 | P95 normal FPR | P99 normal FPR | P95 anomaly TPR | Original Oracle-F (pp) | Original NI/20 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    p99 = summary[summary["quantile"] == 99].set_index("dataset")
    for r in selected.itertuples():
        lines.append(f"| {r.dataset} | {r.drift_median:.5f} | {r.normal_fpr:.2%} | {p99.loc[r.dataset,'normal_fpr']:.2%} | {r.anomaly_tpr:.2%} | {100*r.oracle_delta:.4f} | {r.original_NI} |")
    lines += ["", "Source-validation exceedance is computed on the same calibration samples. Increased normal FPR may reflect scale drift and does not independently establish ranking degradation or an exploitable update opportunity. Interpret alongside causal calibration and continuous Oracle effects.",
        "Source-normal follows the original data-protocol assumption: all-zero processed source labels are not independent semantic certification of normality. A zero source-score variance makes the standardized median difference undefined; denominators are listed in dataset_seed.csv, while other metrics for that entity are retained.", "",
        "[Entity results](physical.csv), [dataset table](dataset_summary.csv), [figure](frozen_need.png).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    controls.atomic_json(out / "summary.json", {"status": "complete", "finished_at": controls.now(),
        "frozen_observations": 530, "diagnostic_workpoints": [95, 99],
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "load_start": load_start, "load_end": os.getloadavg()})
    print("COMPLETE Frozen adaptation-need audit", flush=True)


if __name__ == "__main__":
    main()
