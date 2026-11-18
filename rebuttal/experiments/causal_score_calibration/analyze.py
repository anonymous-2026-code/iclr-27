#!/usr/bin/env python3
"""CPU-only causal, window-aligned rolling score standardization."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import resource
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_controls"))
import run as controls
import numpy as np
import pandas as pd

HORIZONS = (128, 512, 2048)
MODES = ("raw", "z128", "z512", "z2048")
ARMS = ("practical", "oracle")


def load_plan():
    if json.loads((HERE / "source_status.json").read_text())["status"] != "complete":
        raise ValueError("All 530 source-validation scores must be complete")
    plan = json.loads((HERE / "source_plan.json").read_text())
    if len(plan["bases"]) != 530:
        raise ValueError("Expected 530 canonical entity/seed bases")
    return plan


def source_scores(base):
    folder = HERE / "source_scores" / base["dataset"] / (base["subset"] or "__root__") / f"seed{base['seed']}"
    meta = json.loads((folder / "result.json").read_text())
    if not meta["complete"] or meta["source_checkpoint_sha256"] != base["source_checkpoint_sha256"]:
        raise ValueError("Source score identity mismatch")
    scores = np.load(folder / "scores.npy", allow_pickle=False).reshape(-1).astype(np.float64)
    if len(scores) != meta["points"] or len(scores) % 12 or not np.isfinite(scores).all():
        raise ValueError("Invalid source validation history")
    return scores, meta


def standardize(scores, source, windows):
    if len(scores) % 12 or not len(source) or len(source) % 12:
        raise ValueError("Expected complete source/test windows")
    width = windows * 12
    initial = np.asarray(source[-width:], dtype=np.float64)
    values = np.concatenate((initial, np.asarray(scores, dtype=np.float64)))
    # Only positions strictly before each whole current test window are read.
    indices = len(initial) + np.arange(len(scores) // 12) * 12 - 1
    history = pd.Series(values).rolling(width, min_periods=1)
    mean = history.mean().to_numpy()[indices]
    std = history.std(ddof=0).to_numpy()[indices]
    epsilon = max(1e-12, 1e-6 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64))))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise FloatingPointError("Invalid rolling history statistics")
    scale = np.maximum(std, epsilon)
    z = ((np.asarray(scores).reshape(-1, 12) - mean[:, None]) / scale[:, None]).reshape(-1)
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite standardized output")
    return z, mean, std, epsilon


def validate_causality():
    source = np.arange(1, 49, dtype=float) / 10
    scores = np.arange(1, 121, dtype=float) / 7
    z, means, stds, eps = standardize(scores, source, 3)
    history = list(source[-36:])
    expected = []
    for t, window in enumerate(scores.reshape(-1, 12)):
        np.testing.assert_allclose(means[t], np.mean(history), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(stds[t], np.std(history), atol=1e-12, rtol=1e-12)
        expected.extend((window - np.mean(history)) / max(np.std(history), eps))
        history = (history + window.tolist())[-36:]
    np.testing.assert_allclose(z, expected, rtol=1e-12, atol=1e-12)
    changed = scores.copy()
    changed[60:] *= 20
    z_changed, m_changed, s_changed, _ = standardize(changed, source, 3)
    np.testing.assert_array_equal(z_changed[:60], z[:60])
    np.testing.assert_array_equal(m_changed[:6], means[:6])
    np.testing.assert_array_equal(s_changed[:6], stds[:6])
    zero, _, _, epsilon = standardize(np.zeros(36), np.zeros(24), 2)
    assert np.isfinite(zero).all() and (zero == 0).all() and epsilon > 0
    # The same positive affine map preserves within-current-window ranks/ties.
    for raw, transformed in zip(scores.reshape(-1, 12), z.reshape(-1, 12)):
        np.testing.assert_array_equal(np.argsort(raw), np.argsort(transformed))


def local_metrics(scores, labels):
    rows = []
    for block, start in enumerate(range(0, len(labels), 512 * 12)):
        truth = labels[start:start + 512 * 12]
        values = scores[start:start + 512 * 12]
        defined = len(np.unique(truth)) == 2
        rows.append({"block": block, "start_point": start, "points": len(truth),
                     "anomalies": int(truth.sum()), "defined": defined,
                     "auprc": controls.metric(truth, values) if defined else np.nan})
    valid = [r for r in rows if r["defined"]]
    count = sum(r["points"] for r in valid)
    summary = sum(r["points"] * r["auprc"] for r in valid) / count if count else np.nan
    return summary, count / len(labels), len(valid), rows


def category(p, o):
    if not np.isfinite([p, o]).all():
        return "undefined"
    return "BI" if p > .005 and o > .005 else "POG" if p > .005 else "OOG" if o > .005 else "NI"


def build_cells(physical, *, local=False):
    modes = ("raw", "z512") if local else MODES
    rows = []
    measure = "local_auprc" if local else "auprc"
    for mode in modes:
        frame = physical[physical["mode"] == mode]
        frozen = frame[frame.arm == "frozen"][["dataset", "subset", "seed", measure]].rename(columns={measure: "frozen"})
        updated = frame[frame.arm != "frozen"].merge(frozen, on=["dataset", "subset", "seed"], validate="many_to_one")
        updated["delta"] = updated[measure] - updated.frozen
        for key, group in updated.groupby(["dataset", "optimizer", "seed"], sort=True):
            p = group[group.arm == "practical"]
            o = group[group.arm == "oracle"]
            if not p.subset.tolist() == o.subset.tolist():
                raise ValueError("P/O entity pairing order differs")
            if not p.delta.notna().to_list() == o.delta.notna().to_list():
                raise ValueError("P/O/F local support differs")
            valid = int(o.delta.notna().sum())
            pdiff = float(p.delta.median()) if valid else np.nan
            odiff = float(o.delta.median()) if valid else np.nan
            rows.append({"mode": mode, "dataset": key[0], "optimizer": key[1], "seed": key[2],
                "entities_total": len(o), "entities_valid": valid,
                "practical_delta": pdiff, "oracle_delta": odiff, "category": category(pdiff, odiff)})
    result = pd.DataFrame(rows)
    if len(result) != 180 * len(modes):
        raise ValueError("Logical grid differs from 180 per mode")
    if not local:
        old = pd.read_csv(controls.ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv")
        check = result[result["mode"] == "raw"].merge(old, on=["dataset", "optimizer", "seed"], validate="one_to_one")
        for arm in ARMS:
            np.testing.assert_allclose(check[f"{arm}_delta"], check[f"corrected_auprc_{arm}_delta"], rtol=0, atol=1e-14)
        if not (check.category == check.corrected_auprc_category).all():
            raise ValueError("Original outcome regression failed")
    return result


def main():
    controls.configure_threads()
    started, cpu_start, load_start = time.perf_counter(), time.process_time(), os.getloadavg()
    out = HERE / "derived"
    if out.exists():
        raise FileExistsError(out)
    validate_causality()
    plan = load_plan()
    records, blocks = [], []
    for index, (base_id, base) in enumerate(sorted(plan["bases"].items()), 1):
        history, source_meta = source_scores(base)
        common_labels = None
        for key, reference in sorted(base["references"].items()):
            optimizer, arm = key.split(":")
            scores = np.load(reference["score_path"], allow_pickle=False).reshape(-1)
            labels = np.load(reference["label_path"], allow_pickle=False).reshape(-1).astype(np.int8)
            if common_labels is None:
                common_labels = labels
            elif not np.array_equal(labels, common_labels):
                raise ValueError("Canonical label alignment changed")
            original = controls.metric(labels, scores)
            if abs(original - reference["auprc"]) > 1e-12:
                raise ValueError("Canonical metric does not match saved scores")
            identity = {"dataset": base["dataset"], "subset": base["subset"], "seed": base["seed"],
                        "optimizer": optimizer, "arm": arm}
            for mode in MODES:
                if mode == "raw":
                    values, floor_fraction, epsilon = scores, 0.0, np.nan
                else:
                    values, _, std, epsilon = standardize(scores, history, int(mode[1:]))
                    floor_fraction = float((std < epsilon).mean())
                if mode in ("raw", "z512"):
                    local, coverage, valid_blocks, rows = local_metrics(values, labels)
                    blocks.extend({**identity, "mode": mode, **r} for r in rows)
                else:
                    local, coverage, valid_blocks = np.nan, np.nan, None
                records.append({**identity, "mode": mode, "auprc": controls.metric(labels, values),
                    "local_auprc": local, "local_point_coverage": coverage, "local_valid_blocks": valid_blocks,
                    "floor_window_fraction": floor_fraction, "epsilon": epsilon, "points": len(labels)})
        print(f"DONE {index}/530 {base_id} CPU_load={os.getloadavg()[0]:.2f}", flush=True)
    physical = pd.DataFrame(records)
    if len(physical) != 4770 * 4:
        raise ValueError("Expected four metrics for every one of 4770 physical flows")
    cells = build_cells(physical)
    local_cells = build_cells(physical, local=True)
    baseline = cells[cells["mode"] == "raw"][["dataset", "optimizer", "seed", "category"]].rename(columns={"category": "original_category"})
    cells = cells.merge(baseline, on=["dataset", "optimizer", "seed"], validate="many_to_one")
    counts = pd.crosstab(cells["mode"], cells.category).reindex(index=MODES, columns=["BI", "OOG", "NI", "POG"], fill_value=0).reset_index()
    transitions = cells.groupby(["mode", "original_category", "category"]).size().rename("cells").reset_index()
    summary = cells.groupby(["dataset", "mode"])[["practical_delta", "oracle_delta"]].median().reset_index()
    local_summary = local_cells.groupby(["dataset", "mode"])[["practical_delta", "oracle_delta", "entities_valid", "entities_total"]].median().reset_index()
    arm_summary = physical.groupby(["dataset", "mode", "arm"])[["auprc", "floor_window_fraction", "local_point_coverage"]].median().reset_index()
    out.mkdir()
    for name, frame in (("physical", physical), ("cells", cells), ("local_cells", local_cells),
                        ("local_blocks", pd.DataFrame(blocks)), ("outcome_counts", counts),
                        ("transitions", transitions), ("dataset_summary", summary),
                        ("local_dataset_summary", local_summary), ("absolute_arm_summary", arm_summary)):
        frame.to_csv(out / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for mode in MODES:
        group = summary[summary["mode"] == mode].sort_values("dataset")
        axes[0].plot(group.dataset, group.oracle_delta * 100, "o-", label=mode, markersize=4)
    for mode in ("raw", "z512"):
        group = local_summary[local_summary["mode"] == mode].sort_values("dataset")
        axes[1].plot(group.dataset, group.oracle_delta * 100, "o-", label=mode, markersize=4,
                     color="tab:blue" if mode == "raw" else "tab:green")
    for ax, title in zip(axes, ("Global PR-AUC", "Local two-class blocks (support reported)")):
        ax.set(title=title, ylabel="Median Oracle minus matched Frozen (pp)")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(.5, color="grey", linestyle="--", linewidth=.7)
        ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"causal_calibration.{suffix}", dpi=180)
    plt.close(fig)
    ni = cells[(cells.original_category == "NI") & (cells["mode"] != "raw")]
    ni_table = ni.groupby("mode").agg(original_NI=("category", "size"), retained_NI=("category", lambda x: int((x == "NI").sum())),
                                     oracle_clear_gain=("oracle_delta", lambda x: int((x > .005).sum()))).reset_index()
    ni_table.to_csv(out / "original_NI_outcomes.csv", index=False)
    lines = ["# Causal historical score standardization: complete results", "", "The fixed 106 entities and 4,770 physical score streams are used, with 180 cells for each of four evaluations. H=512 is primary; H=128/2048 are predefined sensitivity settings.",
        "Historical statistics are computed before each current 12-point window. Initial history uses source validation only. Test labels do not select history or parameters, and calibration is not fed back into updates.", "",
        "| Setting | BI | OOG | NI | POG |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r.mode} | {r.BI} | {r.OOG} | {r.NI} | {r.POG} |" for r in counts.itertuples()]
    table = summary.pivot(index="dataset", columns="mode", values="oracle_delta")
    lines += ["", "| Dataset | Raw Oracle-F (pp) | z128 | z512 (primary) | z2048 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {d} | {100*r['raw']:.4f} | {100*r['z128']:.4f} | {100*r['z512']:.4f} | {100*r['z2048']:.4f} |" for d, r in table.iterrows()]
    lines += ["", "Each gain uses its correspondingly calibrated Frozen baseline. Absolute performance is in absolute_arm_summary.csv. Category changes relative to Frozen cannot all be interpreted as improvements in absolute Oracle performance.",
        "Local metrics use only fixed time blocks containing both classes; support is in local_blocks/local_cells. They are not full-stream metrics. NI does not imply deterioration; +0.5 pp is a descriptive threshold.",
        "This experiment tests z-scores at three fixed history lengths. Recovered gains may support a role for score comparability. Failure to recover gains does not establish that every calibration is ineffective or justify a representation-collapse claim.", "",
        "[Cells](cells.csv), [original NI transitions](original_NI_outcomes.csv), [local support](local_cells.csv), [figure](causal_calibration.png).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    controls.atomic_json(out / "summary.json", {"status": "complete", "finished_at": controls.now(),
        "physical_score_flows": 4770, "logical_cells_per_mode": 180, "modes": list(MODES),
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "load_start": load_start, "load_end": os.getloadavg(),
        "causality_and_canonical_regression": "passed", "counts": counts.to_dict(orient="records"),
        "original_NI": ni_table.to_dict(orient="records")})
    print("COMPLETE causal score calibration", flush=True)


if __name__ == "__main__":
    main()
