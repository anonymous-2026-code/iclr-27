#!/usr/bin/env python3
"""Strict complete-grid aggregation for the two prespecified rebuttal controls."""
from __future__ import annotations

import json
from pathlib import Path

import run as runner
import numpy as np
import pandas as pd

HERE, ROOT = runner.HERE, runner.ROOT
CATEGORIES = ("BI", "OOG", "NI", "POG")


def category(p, o):
    return "BI" if p > .005 and o > .005 else "POG" if p > .005 else "OOG" if o > .005 else "NI"


def old_values(base, optimizer):
    return tuple(runner.reference(base, arm, optimizer)["auprc"] for arm in ("frozen", "practical", "oracle"))


def canonical_check(cells):
    old = pd.read_csv(ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv")
    joined = cells.merge(old, on=["dataset", "optimizer", "seed"], validate="one_to_one")
    if len(joined) != 180:
        raise ValueError("Canonical cell regression requires all 180 cells")
    for a, b in (("original_practical_delta", "corrected_auprc_practical_delta"),
                 ("original_oracle_delta", "corrected_auprc_oracle_delta")):
        np.testing.assert_allclose(joined[a], joined[b], rtol=0, atol=1e-14)
    if not (joined.original_category == joined.corrected_auprc_category).all():
        raise ValueError("Canonical outcome categories do not reproduce")


def normalization_frames(plan, raw):
    lookup = {(r.dataset, r.subset, r.seed, r.optimizer, r.arm): r.auprc for r in raw.itertuples()}
    if len(lookup) != 4240 or len(raw) != 4240:
        raise ValueError("Expected exactly 4240 unique normalization physical runs")
    paired = []
    for base in plan["bases"].values():
        for config in plan["protocol"]["tta_configs"]:
            optimizer = config["key"]
            f, p, o = old_values(base, optimizer)
            key = base["dataset"], base["subset"], base["seed"], optimizer
            pn, on = lookup[(*key, "practical")], lookup[(*key, "oracle")]
            paired.append({"dataset": key[0], "subset": key[1], "seed": key[2], "optimizer": optimizer,
                "frozen_auprc": f, "original_practical_auprc": p, "original_oracle_auprc": o,
                "practical_auprc": pn, "oracle_auprc": on,
                "original_practical_delta": p - f, "original_oracle_delta": o - f,
                "practical_delta": pn - f, "oracle_delta": on - f,
                "practical_minus_original": pn - p, "oracle_minus_original": on - o})
    paired = pd.DataFrame(paired)
    effects = ["original_practical_delta", "original_oracle_delta", "practical_delta", "oracle_delta",
               "practical_minus_original", "oracle_minus_original"]
    group = paired.groupby(["dataset", "optimizer", "seed"], sort=True)
    cells = group[effects].median().reset_index()
    cells["entities"] = group.size().values
    cells["original_category"] = [category(p, o) for p, o in zip(cells.original_practical_delta, cells.original_oracle_delta)]
    cells["category"] = [category(p, o) for p, o in zip(cells.practical_delta, cells.oracle_delta)]
    canonical_check(cells)
    return paired, cells


def shift_cells(plan, raw):
    lookup = {(r.dataset, r.seed, r.condition, r.optimizer, r.arm): r.auprc for r in raw.itertuples()}
    if len(lookup) != 540 or len(raw) != 540:
        raise ValueError("Expected exactly 540 unique shifted physical runs")
    rows = []
    for base in plan["bases"].values():
        if base["dataset"] not in runner.SHIFT_DATASETS:
            continue
        for config in plan["protocol"]["tta_configs"]:
            optimizer = config["key"]
            f0, p0, o0 = old_values(base, optimizer)
            for condition in runner.CONDITIONS:
                if condition == "identity":
                    f, p, o = f0, p0, o0
                else:
                    key = base["dataset"], base["seed"], condition
                    f = lookup[(*key, "shared", "frozen")]
                    p, o = (lookup[(*key, optimizer, arm)] for arm in ("practical", "oracle"))
                rows.append({"dataset": base["dataset"], "seed": base["seed"], "optimizer": optimizer,
                    "condition": condition, "frozen_auprc": f, "practical_auprc": p, "oracle_auprc": o,
                    "practical_delta": p - f, "oracle_delta": o - f,
                    "frozen_minus_identity": f - f0,
                    "practical_gain_minus_identity": (p - f) - (p0 - f0),
                    "oracle_gain_minus_identity": (o - f) - (o0 - f0),
                    "original_category": category(p0 - f0, o0 - f0), "category": category(p - f, o - f)})
    cells = pd.DataFrame(rows)
    if len(cells) != 300 or len(cells[cells.condition != "identity"]) != 240:
        raise ValueError("Shift logical grid is incomplete")
    return cells


def read_runs(plan):
    protocol = runner.load_protocol()
    expected = {}
    for task in plan["tasks"]:
        base = plan["bases"][task["base_id"]]
        for optimizer, arm in runner.task_runs(task, protocol):
            path = runner.run_directory(task, base, optimizer, arm)
            if path in expected:
                raise ValueError("Duplicate planned output")
            expected[path] = task, base, optimizer, arm
    actual = {p.parent for experiment in runner.EXPERIMENTS
              for p in (HERE.parent / experiment / "runs").glob("*/*/*/*/*/seed*/result.json")}
    if len(expected) != 4780 or actual != set(expected):
        raise ValueError(f"Incomplete or unexpected formal output: expected 4780, got {len(actual)}")
    rows = []
    for path, (task, base, optimizer, arm) in expected.items():
        result = json.loads((path / "result.json").read_text())
        identity = {"complete": True, "experiment": task["experiment"], "condition": task["condition"],
                    "dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
                    "optimizer": optimizer, "arm": arm, "metric_version": runner.VERSION,
                    "source_checkpoint_sha256": base["source_checkpoint_sha256"],
                    "reduction": "selected" if task["experiment"] == runner.NORMALIZATION else "full"}
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Result identity mismatch: {path}")
        scores = np.load(path / "test_energy.npy", allow_pickle=False).reshape(-1)
        labels = np.load(path / "true.npy", allow_pickle=False).reshape(-1)
        selected = np.load(path / "update_mask.npy", allow_pickle=False).reshape(-1)
        expected_labels = np.load(runner.reference(base, arm, optimizer)["label_path"], allow_pickle=False).reshape(-1)
        if not np.array_equal(labels, expected_labels) or selected.shape != labels.shape:
            raise ValueError(f"Changed/misaligned labels or mask: {path}")
        value = runner.metric(labels, scores)
        if abs(value - result["auprc"]) > 1e-12 or result["test_points"] != len(labels) or len(labels) % 12:
            raise ValueError(f"Metric/prefix check failed: {path}")
        expected_mask = labels == 0 if arm == "oracle" else scores < result["selector_tau"] if arm == "practical" else np.zeros(len(labels), bool)
        if not np.array_equal(selected, expected_mask):
            raise ValueError(f"Selection rule mismatch: {path}")
        if result["selected_points"] != int(selected.sum()) or result["selected_normal_points"] != int((selected & (labels == 0)).sum()):
            raise ValueError(f"Selected counts mismatch: {path}")
        if arm != "frozen":
            trace = pd.read_parquet(path / "window_trace.parquet")
            if (result["optimizer_steps"] != len(labels) // 12 or len(trace) != len(labels) // 12
                    or not trace.optimizer_step.all()
                    or not np.array_equal(trace.selected_points, selected.reshape(-1, 12).sum(axis=1))
                    or not (trace.loc[trace.selected_points == 0, "masked_loss"] == 0).all()
                    or not np.isfinite(trace[["masked_loss", "parameter_step_l2"]].to_numpy()).all()):
                raise ValueError(f"Online step/loss diagnostics mismatch: {path}")
        elif result["optimizer_steps"] != 0:
            raise ValueError("Frozen has optimizer steps")
        rows.append(result)
    return pd.DataFrame(rows)


def summarize(cells, keys):
    rows = []
    for name, group in cells.groupby(keys, sort=True):
        names = name if isinstance(name, tuple) else (name,)
        row = dict(zip(keys, names))
        row.update(cells=len(group), **{c: int((group.category == c).sum()) for c in CATEGORIES})
        for field in ("frozen_auprc", "practical_auprc", "oracle_auprc", "practical_delta", "oracle_delta",
                      "original_practical_delta", "original_oracle_delta", "practical_minus_original",
                      "oracle_minus_original", "frozen_minus_identity", "practical_gain_minus_identity",
                      "oracle_gain_minus_identity"):
            if field in group:
                row[field + "_median"] = float(group[field].median())
                row[field + "_mean"] = float(group[field].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plots(normal_cells, shift_summary, directories):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, arm in zip(axes, ("practical", "oracle")):
        for dataset, group in normal_cells.groupby("dataset", sort=True):
            ax.scatter(group[f"original_{arm}_delta"] * 100, group[f"{arm}_delta"] * 100,
                       s=15, alpha=.65, label=dataset)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
        ax.axhline(.5, color="grey", linewidth=.6)
        ax.axvline(.5, color="grey", linewidth=.6)
        ax.set(title=arm.capitalize(), xlabel="Original gain over Frozen (pp)",
               ylabel="Selected-count-normalized gain (pp)")
    axes[1].legend(fontsize=7, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(directories[runner.NORMALIZATION] / f"normalization_effects.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    conditions = list(runner.CONDITIONS)
    labels = ["Original", "+0.5 SD", "+1.0 SD", "x1.25", "x1.5"]
    colors = {"frozen": "tab:blue", "practical": "tab:orange", "oracle": "tab:green"}
    for column, dataset in enumerate(runner.SHIFT_DATASETS):
        group = shift_summary[shift_summary.dataset == dataset].set_index("condition").loc[conditions]
        for arm in ("frozen", "practical", "oracle"):
            axes[0, column].plot(labels, group[f"{arm}_auprc_median"], "o-", label=arm, color=colors[arm], markersize=4)
        for arm in ("practical", "oracle"):
            axes[1, column].plot(labels, group[f"{arm}_delta_median"] * 100, "o-", label=arm, color=colors[arm], markersize=4)
        axes[0, column].set(title=dataset, ylabel="Median PR-AUC")
        axes[1, column].set(ylabel="Median gain over matched Frozen (pp)")
        axes[1, column].axhline(.5, color="grey", linestyle="--", linewidth=.7)
        for ax in axes[:, column]:
            ax.tick_params(axis="x", labelrotation=25)
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
    for suffix in ("png", "pdf"):
        fig.savefig(directories[runner.SHIFT] / f"controlled_shift_effects.{suffix}", dpi=180)
    plt.close(fig)


def main():
    runner.configure_threads()
    plan = json.loads((HERE / "plan.json").read_text())
    raw = read_runs(plan)
    normal_raw = raw[raw.experiment == runner.NORMALIZATION]
    shift_raw = raw[raw.experiment == runner.SHIFT]
    paired, normal_cells = normalization_frames(plan, normal_raw)
    shifted = shift_cells(plan, shift_raw)
    normal_summary = summarize(normal_cells, ["dataset"])
    shift_summary = summarize(shifted, ["dataset", "condition"])
    shift_diagnostics = pd.read_csv(HERE.parent / runner.SHIFT / "input_diagnostics/shift_summary.csv")
    shift_summary = shift_summary.merge(shift_diagnostics, on=["dataset", "condition"], validate="one_to_one")
    directories = {e: HERE.parent / e / "derived" for e in runner.EXPERIMENTS}
    for directory in directories.values():
        directory.mkdir(exist_ok=False)
    for experiment, runs, cells, summary in ((runner.NORMALIZATION, normal_raw, normal_cells, normal_summary),
                                           (runner.SHIFT, shift_raw, shifted, shift_summary)):
        out = directories[experiment]
        runs.to_csv(out / "physical.csv", index=False)
        cells.to_csv(out / "cells.csv", index=False)
        summary.to_csv(out / "dataset_summary.csv", index=False)
        counts = cells.groupby((["condition"] if experiment == runner.SHIFT else []) + ["category"]).size().rename("cells").reset_index()
        counts.to_csv(out / "outcome_counts.csv", index=False)
        transitions = cells.groupby((["condition"] if experiment == runner.SHIFT else []) + ["original_category", "category"]).size().rename("cells").reset_index()
        transitions.to_csv(out / "transitions.csv", index=False)
        diagnostic_fields = ["selected_fraction", "normal_coverage", "selector_purity", "zero_selected_windows",
                             "parameter_step_l2_sum", "final_source_parameter_l2"]
        updates = runs[runs.arm != "frozen"]
        grouped = updates.groupby(["dataset", "condition", "arm"])
        diagnostics = grouped[diagnostic_fields].median()
        diagnostics["physical_runs"] = grouped.size()
        diagnostics["purity_defined_runs"] = grouped.selector_purity.count()
        diagnostics["no_selected_runs"] = grouped.selected_points.apply(lambda s: int((s == 0).sum()))
        diagnostics.reset_index().to_csv(out / "update_diagnostics.csv", index=False)
        compared = cells if experiment == runner.NORMALIZATION else cells[cells.condition != "identity"]
        runner.atomic_json(out / "summary.json", {"status": "complete", "finished_at": runner.now(),
            "physical_runs": len(runs), "logical_rows": len(cells), "metric_version": runner.VERSION,
            "summed_worker_run_hours": float(runs.wall_seconds.sum() / 3600),
            "classification_comparison_rows": len(compared),
            "classification_agreement": float((compared.original_category == compared.category).mean()),
            "outcome_counts": counts.to_dict(orient="records")})
    paired.to_csv(directories[runner.NORMALIZATION] / "paired_entities.csv", index=False)
    lines = ["# Selected-count normalization: complete results", "", "All 4,240 new adaptation runs are complete: 106 entities and 180 logical cells, with no additional source training.",
             "The primary metric is trapezoidal PR-AUC. Effects are paired within entities before taking dataset medians.", "",
             "| Category | Original full-window | Selected-count |", "|---|---:|---:|"]
    lines += [f"| {c} | {(normal_cells.original_category == c).sum()} | {(normal_cells.category == c).sum()} |" for c in CATEGORIES]
    lines += ["", "| Dataset | Original Oracle-F (pp) | New Oracle-F (pp) | New Practical-F (pp) |",
              "|---|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {100*r.original_oracle_delta_median:.4f} | {100*r.oracle_delta_median:.4f} | {100*r.practical_delta_median:.4f} |" for r in normal_summary.itertuples()]
    lines += ["", "NI means the gain does not exceed +0.5 pp, rather than universal deterioration. Normalization removes the explicit k/12 factor but does not match sample counts, AdamW state, or parameter displacement. The actual Practical mask can change with the model trajectory.",
              "This is a denominator-sensitivity control for loss normalization, not independent identification of a purity effect. Purity is undefined when no points are selected; diagnostic tables retain valid denominators and counts of runs without selections.", "",
              "Details: [cells.csv](cells.csv), [transitions.csv](transitions.csv), [update_diagnostics.csv](update_diagnostics.csv), [figure](normalization_effects.png).", ""]
    (directories[runner.NORMALIZATION] / "RESULTS.md").write_text("\n".join(lines))
    lines = ["# Controlled distribution transformations: complete results", "", "All 60 new Frozen and 480 new P/O runs are complete: 240 nonzero-condition cells, with the original 60 identity cells reused separately.",
             "The three datasets form a diagnostic subset fixed by original performance, rather than representing all nine datasets. All conditions are reported without selection by effect.", "",
             "| Dataset | Condition | Measured median W1 | Median Frozen PR-AUC | P-F (pp) | O-F (pp) | NI/20 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {r.condition} | {r.w1_median:.5f} | {r.frozen_auprc_median:.5f} | {100*r.practical_delta_median:.4f} | {100*r.oracle_delta_median:.4f} | {r.NI} |" for r in shift_summary.itertuples()]
    lines += ["", "P-F/O-F use Frozen under the same transformation. Median paired gains are not differences between independently aggregated arm medians. Transformation strength is not the actual source/test distance and must be interpreted with changes in W1 and Frozen.",
              "Deliberate affine transformations do not guarantee an exploitable gain for the current update and are not a complete model of real drift. Gains, if present, are compatible with an adaptation-need explanation. Remaining NI supports only limited conclusions under these explicit conditions.",
              "Seeds, optimizers, and strengths are not treated as independent datasets for significance testing. Category agreement uses only the 240 nonzero-condition cells. Purity is undefined without selected points, and diagnostic tables retain valid denominators.", "",
              "Details: [cells.csv](cells.csv), [dataset_summary.csv](dataset_summary.csv), [update_diagnostics.csv](update_diagnostics.csv), [figure](controlled_shift_effects.png).", ""]
    (directories[runner.SHIFT] / "RESULTS.md").write_text("\n".join(lines))
    plots(normal_cells, shift_summary, directories)
    print(json.dumps({"status": "complete", "normalization_runs": len(normal_raw), "shift_runs": len(shift_raw)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
