#!/usr/bin/env python3
"""R2-W1 B/C: ten original SMD M-1-6 replays with isolated diagnostics.

Run prepare, run --optimizer {sgd,adamw} (one process/GPU), then aggregate.
Twenty chronological local probes retain the original Future Probe protocol.
Fixed probes: 256 evenly spaced points/class over the evaluation prefix;
checkpoint states after floor(j*N/4) updates, j=0..4. Also measure their
local responses at the same twenty updates, without using them for learning.
Primary harmful update: selected normal loss decreases by >1e-8 AND the
future strict pair fraction decreases. Margin harm is separately defined as
normal loss decrease AND margin decrease by >1e-8. Denominator: all scheduled
updates with a nonempty normal mask and both future classes. No reselection.
All seeds are paired. Aggregation is descriptive, with no independence claim
for updates, probe points, or checkpoint states. PR-AUC is PR-curve trapezoid
area, as in the corrected original artifacts, not average precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import experiment as future

np, pd, controls = future.np, future.pd, future.controls
HERE = Path(__file__).resolve().parent
OUT = HERE / "derived/smd_m1_6"
RUNS = HERE / "runs/smd_m1_6"
OPTIMIZERS = ("sgd_ttlr0.0001", "adamw_ttlr0.0001")
TOL = 1e-8


def prepare():
    original = json.loads((HERE.parent / "gpu_controls/plan.json").read_text())
    bases = [original["bases"][f"SMD/M-1-6/seed{s}"] for s in range(42, 47)]
    protocol = controls.load_protocol()
    assert protocol.model_name == "win_ae" and protocol.sequence_length == protocol.test_stride == 12
    series = controls.prepare_series(Path(bases[0]["directory"]), protocol.train_fraction)
    count = len(series.test) // 12
    truth = series.test_labels[:count * 12]
    # Reuse chronological, label-stratified indexing; window -1 means whole stream.
    groups = future.future_indices(truth, -1, maximum=256)
    if any(len(v) != 256 for v in groups.values()):
        raise ValueError("The prespecified fixed probe needs 256 points per class")
    plan = {
        "created_at": controls.now(), "dataset": "SMD", "subset": "M-1-6",
        "seeds": list(range(42, 47)), "optimizers": list(OPTIMIZERS),
        "bases": bases, "windows": count, "evaluated_points": len(truth),
        "local_steps": [j * count // 20 for j in range(20)],
        "fixed_diagnostics": {"groups": {k: v.tolist() for k, v in groups.items()},
                              "checkpoint_steps": [j * count // 4 for j in range(5)]},
        "harmful_ranking": "selected_normal_delta < -1e-8 AND future_pair_delta < 0",
        "harmful_margin": "selected_normal_delta < -1e-8 AND future_margin_delta < -1e-8",
        "local_denominator": "selected_points > 0 AND both future classes present",
        "fixed_sampling": "256 evenly spaced time indices/class; same across all states and seeds",
        "fixed_probe_overlap": "retrospective set; normal points still enter adaptation at their natural online time",
        "metric": controls.VERSION, "score_tolerance": TOL,
        "interval_analysis": "compare local fixed-probe sums with checkpoint changes in each chronological quarter",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    controls.atomic_json(OUT / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("windows", "evaluated_points", "local_steps")}), flush=True)


def run(optimizer):
    controls.configure_threads()
    plan = json.loads((OUT / "plan.json").read_text())
    for base in plan["bases"]:
        task = {"base": base, "optimizer": optimizer + "_ttlr0.0001"}
        result = future.run_one(task, RUNS, plan["fixed_diagnostics"])
        print(json.dumps(result), flush=True)


def local_summary(g):
    active = g[(g.selected_points > 0) & g.paired_probe_defined]
    improved = active.current_selected_normal_delta < -TOL
    changing = active[active.margin_delta.abs() > TOL]
    error = active.predicted_margin_delta - active.margin_delta
    return {
        "scheduled": len(g), "active_paired": len(active),
        "empty_mask": int((g.selected_points == 0).sum()),
        "undefined_future_pair": int((~g.paired_probe_defined).sum()),
        "normal_improved_fraction": float(improved.mean()),
        "harmful_pair_fraction": float((improved & (active.pair_fraction_delta < 0)).mean()),
        "harmful_margin_fraction": float((improved & (active.margin_delta < -TOL)).mean()),
        "joint_anomaly_pair_harm_fraction": float((improved & (active.anomaly_delta < -TOL) & (active.pair_fraction_delta < 0)).mean()),
        "normal_objective_delta_mean": float(active.masked_loss_delta.mean()),
        "normal_update_delta_mean": float(active.current_selected_normal_delta.mean()),
        "probe_normal_delta_mean": float(active.normal_delta.mean()),
        "probe_anomaly_delta_mean": float(active.anomaly_delta.mean()),
        "margin_delta_mean": float(active.margin_delta.mean()),
        "margin_delta_median": float(active.margin_delta.median()),
        "pair_delta_mean": float(active.pair_fraction_delta.mean()),
        "pair_delta_sum": float(active.pair_fraction_delta.sum()),
        "prediction_sign_agreement": float((np.sign(changing.margin_delta) == np.sign(changing.predicted_margin_delta)).mean()),
        "prediction_sign_count": len(changing),
        "prediction_correlation": float(active.margin_delta.corr(active.predicted_margin_delta)),
        "prediction_mae": float(error.abs().mean()),
        "prediction_relative_l1_error": float(error.abs().sum() / active.margin_delta.abs().sum()),
        "fixed_local_margin_mean": float(active.fixed_margin_delta.mean()),
        "fixed_local_pair_mean": float(active.fixed_pair_fraction_delta.mean()),
        "fixed_local_harmful_pair_fraction": float((improved & (active.fixed_pair_fraction_delta < 0)).mean()),
    }


def aggregate():
    plan = json.loads((OUT / "plan.json").read_text())
    results, probes, cumulative, intervals, online_fixed = [], [], [], [], []
    for optimizer in OPTIMIZERS:
        for base in plan["bases"]:
            path = future.task_directory({"base": base, "optimizer": optimizer}, RUNS)
            result = json.loads((path / "result.json").read_text())
            assert result["complete"] and result["auprc_parity_error"] <= 1e-8
            assert result["optimizer_steps"] == plan["windows"]
            results.append(result)
            local = pd.read_csv(path / "probes.csv")
            assert local.window.tolist() == plan["local_steps"]
            probes.append(local)
            c = pd.read_csv(path / "cumulative.csv")
            assert c.steps_completed.tolist() == plan["fixed_diagnostics"]["checkpoint_steps"]
            for metric in ("normal_mean", "anomaly_mean", "margin", "pair_fraction", "auprc"):
                c[metric + "_from_frozen"] = c[metric] - c.iloc[0][metric]
            cumulative.append(c)
            # Post-hoc descriptive control prompted by disagreement between
            # online and cumulative diagnostics. No new model execution.
            frozen_scores = np.load(controls.reference(base, "frozen", "shared")["score_path"]).reshape(-1)
            online_scores = np.load(path / "test_energy.npy").reshape(-1)
            fixed_groups = plan["fixed_diagnostics"]["groups"]
            baseline = future.fixed_summary({k: frozen_scores[v] for k, v in fixed_groups.items()})
            online = future.fixed_summary({k: online_scores[v] for k, v in fixed_groups.items()})
            online_fixed.append({"seed": base["seed"], "optimizer": optimizer,
                **{f"frozen_{k}": baseline[k] for k in ("margin", "pair_fraction", "auprc")},
                **{f"online_{k}": online[k] for k in ("margin", "pair_fraction", "auprc")},
                **{f"online_{k}_from_frozen": online[k] - baseline[k]
                   for k in ("margin", "pair_fraction", "auprc")}})
            for quarter in range(4):
                left, right = c.iloc[quarter], c.iloc[quarter + 1]
                updates = local[(local.window >= left.steps_completed) & (local.window < right.steps_completed)]
                row = {"optimizer": optimizer, "seed": base["seed"], "quarter": quarter + 1,
                       "sampled_updates": len(updates)}
                for metric in ("margin", "pair_fraction", "auprc"):
                    row[f"local_fixed_{metric}_sum"] = float(updates[f"fixed_{metric}_delta"].sum())
                    row[f"checkpoint_{metric}_change"] = float(right[metric] - left[metric])
                intervals.append(row)
    runs, p, c = pd.DataFrame(results), pd.concat(probes), pd.concat(cumulative)
    for seed in range(42, 47):
        # Same source checkpoint and exact initial fixed-set scores for each pair.
        assert runs[runs.seed == seed].source_checkpoint_sha256.nunique() == 1
        initial = c[(c.seed == seed) & (c.steps_completed == 0)]
        for name in ("normal_mean", "anomaly_mean", "margin", "pair_fraction", "auprc"):
            np.testing.assert_allclose(initial[name], initial[name].iloc[0], rtol=0, atol=0)
        a = p[(p.seed == seed) & (p.optimizer == OPTIMIZERS[0])]
        b = p[(p.seed == seed) & (p.optimizer == OPTIMIZERS[1])]
        assert a.normal_indices.tolist() == b.normal_indices.tolist()
        assert a.anomaly_indices.tolist() == b.anomaly_indices.tolist()
    per_seed, summaries = [], []
    for (optimizer, seed), g in p.groupby(["optimizer", "seed"]):
        per_seed.append({"optimizer": optimizer, "seed": seed, **local_summary(g)})
    per_seed = pd.DataFrame(per_seed).merge(runs[["optimizer", "seed", "auprc", "frozen_auprc", "online_delta_auprc"]])
    for optimizer, g in p.groupby("optimizer"):
        r = runs[runs.optimizer == optimizer]
        final = c[(c.optimizer == optimizer) & (c.steps_completed == plan["windows"])]
        summaries.append({"optimizer": optimizer, **local_summary(g),
            "online_delta_auprc_mean": r.online_delta_auprc.mean(),
            "online_delta_auprc_sd": r.online_delta_auprc.std(ddof=1),
            **{f"final_{m}_from_frozen_mean": final[m + "_from_frozen"].mean()
               for m in ("normal_mean", "anomaly_mean", "margin", "pair_fraction", "auprc")}})
    files = {"runs.csv": runs, "probes.csv": p, "per_seed.csv": per_seed,
             "summary.csv": pd.DataFrame(summaries), "cumulative.csv": c,
             "intervals.csv": pd.DataFrame(intervals), "online_fixed.csv": pd.DataFrame(online_fixed)}
    associations = []
    for optimizer, g in files["intervals.csv"].groupby("optimizer"):
        for metric in ("margin", "pair_fraction", "auprc"):
            local, measured = g[f"local_fixed_{metric}_sum"], g[f"checkpoint_{metric}_change"]
            associations.append({"optimizer": optimizer, "metric": metric, "intervals": len(g),
                "pearson_correlation": local.corr(measured),
                "sign_agreement": float((np.sign(local) == np.sign(measured)).mean())})
    files["interval_associations.csv"] = pd.DataFrame(associations)
    active = p[(p.selected_points > 0) & p.paired_probe_defined].copy()
    active["harmful_pair"] = (active.current_selected_normal_delta < -TOL) & (active.pair_fraction_delta < 0)
    files["local_by_step.csv"] = active.groupby(["optimizer", "window"]).agg(
        harmful_pair_fraction=("harmful_pair", "mean"), margin_delta=("margin_delta", "mean"),
        pair_delta=("pair_fraction_delta", "mean")).reset_index()
    files["checkpoint_means.csv"] = c.groupby(["optimizer", "steps_completed"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    for name, frame in files.items():
        frame.to_csv(OUT / name, index=False)
    print(files["summary.csv"].to_string(index=False))
    print(files["checkpoint_means.csv"][["optimizer", "steps_completed", "margin", "pair_fraction", "auprc"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--optimizer", choices=("sgd", "adamw"))
    args = parser.parse_args()
    if args.action == "run":
        if args.optimizer is None:
            parser.error("run requires --optimizer")
        run(args.optimizer)
    elif args.action == "prepare":
        prepare()
    else:
        aggregate()
