#!/usr/bin/env python3
"""R1-W3/R2-W1/R3-W1: saved-score analysis only, with no model execution.

Four chronological periods use the existing checkpoint boundaries, independent
of scores: [0,493,987,1480,1974] windows. Compare all saved checkpoint score
vectors, online scores and Frozen scores at the identical 512 fixed indices.
Also analyze the complete Frozen/online streams. Strict pair ordering is
partitioned into same-period and different-period pairs, with exact weights.
This is an additive pair-fraction identity, never a decomposition of PR-AUC.
Paired score-ratio quantiles describe temporal changes, not a calibration/TTA
method or proof of scalar score rescaling. No checkpoint selection/search.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "derived/smd_m1_6"
RUNS = HERE / "runs/smd_m1_6/SMD"
OUT = HERE / "derived/smd_m1_6_dynamics"


def metric(labels, scores):
    if len(np.unique(labels)) != 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def pair_count(normal, anomaly):
    return int(np.searchsorted(np.sort(normal), anomaly, side="left").sum())


def describe(indices, labels, scores, frozen, boundaries, identity):
    assert np.isfinite(scores).all() and (frozen > 0).all()
    periods = np.searchsorted(boundaries[1:], indices, side="right")
    assert (periods < 4).all()
    n, a = scores[labels == 0], scores[labels == 1]
    total = len(n) * len(a)
    cells, temporal = [], []
    within_count = within_correct = cross_count = cross_correct = 0
    for i in range(4):
        for j in range(4):
            av, nv = scores[(labels == 1) & (periods == i)], scores[(labels == 0) & (periods == j)]
            pairs, correct = len(av)*len(nv), pair_count(nv, av)
            cells.append({**identity, "anomaly_period": i+1, "normal_period": j+1,
                          "pairs": pairs, "correct": correct, "weight": pairs/total,
                          "pair_fraction": correct/pairs if pairs else np.nan,
                          "contribution": correct/total})
            if i == j:
                within_count += pairs
                within_correct += correct
            else:
                cross_count += pairs
                cross_correct += correct
        use = periods == i
        ll, ss, ff = labels[use], scores[use], frozen[use]
        nn, aa = ss[ll == 0], ss[ll == 1]
        row = {**identity, "period": i+1, "normal_points": len(nn), "anomaly_points": len(aa),
               "auprc": metric(ll, ss), "pair_fraction": pair_count(nn, aa)/(len(nn)*len(aa)) if len(nn)*len(aa) else np.nan}
        for name, value in (("normal", 0), ("anomaly", 1)):
            points, ratios = ss[ll == value], ss[ll == value]/ff[ll == value]
            row[name + "_mean"] = points.mean() if len(points) else np.nan
            for suffix, quantile in (("q10", .1), ("median", .5), ("q90", .9)):
                row[name + "_" + suffix] = np.quantile(points, quantile) if len(points) else np.nan
                row[name + "_paired_ratio_" + suffix] = np.quantile(ratios, quantile) if len(ratios) else np.nan
        temporal.append(row)
    assert total == within_count + cross_count
    assert pair_count(n, a) == within_correct + cross_correct
    summary = {**identity, "normal_points": len(n), "anomaly_points": len(a),
               "auprc": metric(labels, scores), "pair_fraction": pair_count(n, a)/total,
               "within_pair_fraction": within_correct/within_count,
               "cross_pair_fraction": cross_correct/cross_count,
               "within_weight": within_count/total, "cross_weight": cross_count/total,
               "within_contribution": within_correct/total, "cross_contribution": cross_correct/total}
    return summary, temporal, cells


def main():
    plan = json.loads((SOURCE / "plan.json").read_text())
    boundaries = np.array(plan["fixed_diagnostics"]["checkpoint_steps"])*12
    fixed_indices = np.sort(np.concatenate(list(plan["fixed_diagnostics"]["groups"].values())))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "design.json").write_text(json.dumps({
        "source_plan": str(SOURCE / "plan.json"), "seeds": plan["seeds"], "optimizers": plan["optimizers"],
        "period_boundaries_points": boundaries.tolist(), "fixed_points": len(fixed_indices),
        "pair_definition": "strict anomaly > normal; condition on same vs different chronological periods",
        "score_views": "Frozen, online, and all four saved adapted-checkpoint evaluations",
        "full_stream_control": "Frozen and online arrays only; no full-stream adapted-checkpoint inference",
        "model_executions": 0, "metric": "sklearn PR-curve trapezoidal area",
    }, indent=2) + "\n")
    summaries, temporal, cells, verification = [], [], [], []
    for base in plan["bases"]:
        frozen_ref = base["references"]["shared:frozen"]
        frozen = np.load(frozen_ref["score_path"]).reshape(-1)
        labels = np.load(frozen_ref["label_path"]).reshape(-1).astype(int)
        assert len(labels) == boundaries[-1] and set(np.unique(labels)) == {0, 1}
        np.testing.assert_allclose(metric(labels, frozen), frozen_ref["auprc"], rtol=0, atol=1e-8)
        for optimizer in plan["optimizers"]:
            directory = RUNS / optimizer / f"seed{base['seed']}"
            online = np.load(directory / "test_energy.npy").reshape(-1)
            ref = base["references"][optimizer + ":oracle"]
            np.testing.assert_array_equal(online, np.load(ref["score_path"]).reshape(-1))
            np.testing.assert_array_equal(labels, np.load(directory / "true.npy").reshape(-1))
            np.testing.assert_allclose(metric(labels, online), ref["auprc"], rtol=0, atol=1e-8)
            verification.append(dict(seed=base["seed"], optimizer=optimizer, frozen_auprc=metric(labels, frozen),
                                     online_auprc=metric(labels, online), online_delta_auprc=metric(labels, online)-metric(labels, frozen)))
            raw = pd.read_csv(directory / "fixed_scores.csv")
            checkpoint_metrics = pd.read_csv(directory / "cumulative.csv").set_index("steps_completed")
            fixed_views = {"frozen": frozen[fixed_indices], "online": online[fixed_indices]}
            for step, g in raw.groupby("steps_completed"):
                g = g.sort_values("point_index")
                np.testing.assert_array_equal(g.point_index, fixed_indices)
                np.testing.assert_array_equal((g["class"] == "anomaly").astype(int), labels[fixed_indices])
                values = g.score.to_numpy()
                np.testing.assert_allclose(metric(labels[fixed_indices], values), checkpoint_metrics.loc[step, "auprc"], rtol=0, atol=1e-12)
                if step == 0:
                    np.testing.assert_allclose(values, fixed_views["frozen"], rtol=1e-5, atol=1e-6)
                else:
                    fixed_views[f"checkpoint_{step}"] = values
            for cohort, ids, views in (("fixed", fixed_indices, fixed_views),
                                       ("full_stream", np.arange(len(labels)), {"frozen": frozen, "online": online})):
                for mode, values in views.items():
                    result, by_period, by_cell = describe(ids, labels[ids], values, frozen[ids], boundaries,
                        dict(cohort=cohort, optimizer=optimizer, seed=base["seed"], mode=mode))
                    summaries.append(result)
                    temporal.extend(by_period)
                    cells.extend(by_cell)
    frames = {"score_views": pd.DataFrame(summaries), "period_metrics": pd.DataFrame(temporal),
              "pair_cells": pd.DataFrame(cells)}
    metrics = {"score_views": ("auprc", "pair_fraction", "within_pair_fraction", "cross_pair_fraction", "within_contribution", "cross_contribution"),
               "period_metrics": ("auprc", "pair_fraction"), "pair_cells": ("pair_fraction", "contribution")}
    extra = {"score_views": [], "period_metrics": ["period"], "pair_cells": ["anomaly_period", "normal_period"]}
    for name, frame in frames.items():
        keys = ["cohort", "optimizer", "seed"] + extra[name]
        baseline = frame[frame["mode"] == "frozen"][keys + list(metrics[name])]
        merged = frame.merge(baseline, on=keys, suffixes=("", "_frozen"), validate="many_to_one")
        for m in metrics[name]:
            merged[m + "_from_frozen"] = merged[m] - merged[m + "_frozen"]
        if name == "score_views":
            np.testing.assert_allclose(merged.pair_fraction_from_frozen,
                merged.within_contribution_from_frozen + merged.cross_contribution_from_frozen, rtol=0, atol=1e-12)
        merged.to_csv(OUT / (name + ".csv"), index=False)
        group = ["cohort", "optimizer", "mode"] + extra[name]
        means = merged.groupby(group).mean(numeric_only=True).drop(columns="seed").reset_index()
        means.to_csv(OUT / (name + "_means.csv"), index=False)
        if name == "score_views":
            print(means[["cohort", "optimizer", "mode", "auprc", "auprc_from_frozen", "pair_fraction_from_frozen",
                         "within_contribution_from_frozen", "cross_contribution_from_frozen"]].to_string(index=False))
    pd.DataFrame(verification).to_csv(OUT / "original_metric_verification.csv", index=False)
    print("PASS: 10 original streams and 50 checkpoint evaluations verified; exact pair decomposition; no model inference.")


if __name__ == "__main__":
    main()
