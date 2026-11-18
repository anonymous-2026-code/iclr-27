#!/usr/bin/env python3
"""Focused protocol tests and full-stream original-run parity before launch."""
from __future__ import annotations

import json
from pathlib import Path
import time

import run as runner
import aggregate
import numpy as np
import pandas as pd
import torch
from section5_repro.data import PreparedSeries


def toy_checks():
    weight = torch.tensor(2.0, requires_grad=True)
    points = torch.arange(1, 13, dtype=torch.float32).reshape(1, 12) * weight.square()
    mask = torch.arange(12).reshape(1, 12) < 3
    full, k = runner.reduction(points, mask, "full")
    selected, _ = runner.reduction(points, mask, "selected")
    torch.testing.assert_close(full, selected * (3 / 12))
    gf = torch.autograd.grad(full, weight, retain_graph=True)[0]
    gs = torch.autograd.grad(selected, weight, retain_graph=True)[0]
    torch.testing.assert_close(gf, gs * (3 / 12))
    assert k == 3
    all_mask = torch.ones_like(mask)
    torch.testing.assert_close(runner.reduction(points, all_mask, "full")[0],
                               runner.reduction(points, all_mask, "selected")[0])
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    for m in (all_mask, torch.zeros_like(mask)):
        before = parameter.detach().clone()
        optimizer.zero_grad()
        loss, k = runner.reduction(parameter.square().expand(1, 12), m, "selected")
        loss.backward()
        if k == 0:
            assert loss.item() == 0 and parameter.grad.item() == 0
        optimizer.step()
        assert not torch.equal(before, parameter.detach())
    source = np.array([[-1, 3], [0, 3], [1, 3]], dtype=np.float32)
    test = np.array([[2, 7], [4, 9]], dtype=np.float32)
    series = PreparedSeries(train=source[:2], validation=source[2:], test=test,
        test_labels=np.array([0, 1], dtype=np.float32), full_source=source,
        feature_names=("variable", "constant"), scaler_mean=np.zeros(2), scaler_scale=np.ones(2))
    assert runner.transform(series, "identity") is series
    for condition, (scale, offset) in runner.CONDITIONS.items():
        changed = runner.transform(series, condition)
        np.testing.assert_array_equal(changed.test[:, 0], test[:, 0] * np.float32(scale) + np.float32(offset))
        np.testing.assert_array_equal(changed.test[:, 1], test[:, 1])
        assert changed.train is series.train and changed.validation is series.validation
        assert changed.test_labels is series.test_labels
    np.testing.assert_array_equal(series.test, [[2, 7], [4, 9]])


def aggregation_checks(plan):
    simulated = {experiment: [] for experiment in runner.EXPERIMENTS}
    for task in plan["tasks"]:
        base = plan["bases"][task["base_id"]]
        for optimizer, arm in runner.task_runs(task, runner.load_protocol()):
            simulated[task["experiment"]].append({"dataset": base["dataset"], "subset": base["subset"],
                "seed": base["seed"], "optimizer": optimizer, "arm": arm, "condition": task["condition"],
                "auprc": runner.reference(base, arm, optimizer)["auprc"]})
    _, cells = aggregate.normalization_frames(plan, pd.DataFrame(simulated[runner.NORMALIZATION]))
    np.testing.assert_allclose(cells.practical_delta, cells.original_practical_delta, rtol=0, atol=0)
    np.testing.assert_allclose(cells.oracle_delta, cells.original_oracle_delta, rtol=0, atol=0)
    assert (cells.category == cells.original_category).all()
    shifted = aggregate.shift_cells(plan, pd.DataFrame(simulated[runner.SHIFT]))
    assert len(shifted) == 300
    for column in ("frozen_minus_identity", "practical_gain_minus_identity", "oracle_gain_minus_identity"):
        assert (shifted[column] == 0).all()
    assert (shifted.category == shifted.original_category).all()


def main():
    runner.configure_threads()
    torch.cuda.set_device(0)
    output = runner.HERE / "validation"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    plan = json.loads((runner.HERE / "plan.json").read_text())
    toy_checks()
    aggregation_checks(plan)
    print("PASS toy reduction/zero-step/transform and canonical aggregation regression", flush=True)
    base = plan["bases"]["Daphnet/__root__/seed42"]
    series = runner.prepare_series(Path(base["directory"]), runner.load_protocol().train_fraction)
    original_test = series.test.copy()
    configs = ("sgd_ttlr0.001", "adamw_ttlr0.0001")
    task = {"experiment": runner.NORMALIZATION, "condition": "identity"}
    original = {}
    records = []
    for optimizer, arm in [("shared", "frozen")] + [(c, a) for c in configs for a in ("practical", "oracle")]:
        print(f"PARITY {optimizer} {arm}", flush=True)
        result, score, labels = runner.run_one(base, task, optimizer, arm, series,
            output_root=output / "original_full_window", reduction_override="full")
        ref = runner.reference(base, arm, optimizer)
        old_score = np.load(ref["score_path"], allow_pickle=False).reshape(-1)
        np.testing.assert_allclose(score, old_score, rtol=1e-5, atol=1e-6)
        error = abs(result["auprc"] - ref["auprc"])
        if error > 1e-8:
            raise AssertionError(f"Corrected PR-AUC parity failed: {optimizer} {arm} {error}")
        if arm != "frozen":
            old_trace = pd.read_parquet(Path(ref["score_path"]).parent / "tta_pointwise_analysis.parquet",
                                       columns=["simple_used_for_update"])
            new_mask = np.load(Path(result["output_path"]) / "update_mask.npy", allow_pickle=False)
            np.testing.assert_array_equal(new_mask, old_trace.simple_used_for_update.to_numpy().astype(bool))
            tau = json.loads(Path(ref["manifest_path"]).read_text())["result_metadata"]["selector_tau"]
            if abs(result["selector_tau"] - tau) > 1e-10:
                raise AssertionError("Source calibration threshold changed")
        original[optimizer, arm] = result, score
        records.append({"optimizer": optimizer, "arm": arm, "max_score_error": float(np.max(np.abs(score - old_score))),
                        "auprc_error": error, "wall_seconds": result["wall_seconds"]})
    normalized = []
    for optimizer in configs:
        for arm in ("practical", "oracle"):
            print(f"NORMALIZED {optimizer} {arm}", flush=True)
            result, score, _ = runner.run_one(base, task, optimizer, arm, series, output_root=output / "selected_count")
            np.testing.assert_array_equal(score[:12], original[optimizer, arm][1][:12])
            trace = pd.read_parquet(Path(result["output_path"]) / "window_trace.parquet")
            assert (trace.loc[trace.selected_points == 0, "masked_loss"] == 0).all()
            normalized.append({k: result[k] for k in ("optimizer", "arm", "auprc", "optimizer_steps",
                                                       "zero_selected_windows", "wall_seconds")})
    shifted = []
    shifted_task = {"experiment": runner.SHIFT, "condition": "offset_0p5"}
    for optimizer, arm in (("shared", "frozen"), (configs[1], "practical"), (configs[1], "oracle")):
        print(f"SHIFTED {optimizer} {arm}", flush=True)
        result, score, _ = runner.run_one(base, shifted_task, optimizer, arm, series, output_root=output / "shifted")
        if arm != "frozen":
            assert result["selector_tau"] == original[optimizer, arm][0]["selector_tau"]
            np.testing.assert_allclose(score[:12], shifted[0]["first_window"], rtol=1e-5, atol=1e-6)
        shifted.append({**{k: result[k] for k in ("optimizer", "arm", "auprc", "wall_seconds")},
                        "first_window": score[:12].tolist()})
    np.testing.assert_array_equal(series.test, original_test)
    report = {"status": "passed", "finished_at": runner.now(), "wall_seconds": time.perf_counter() - started,
        "toy_reduction_gradient_zero_step_transforms": "passed",
        "canonical_180_cell_and_shift_300_row_aggregation_regression": "passed",
        "validation_dataset": "Daphnet", "seed": 42, "parity": records,
        "selected_count_full_stream": normalized, "controlled_shift_full_stream": shifted,
        "new_experiment_results_are_validation_only": True,
        "cpu_affinity": sorted(runner.os.sched_getaffinity(0))}
    runner.atomic_json(runner.HERE / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
