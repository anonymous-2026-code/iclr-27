#!/usr/bin/env python3
"""Validate the extension against saved ICLR trajectories before full launch."""
import json
from pathlib import Path
import runpy
import time

import run as runner
import aggregate
import numpy as np
import torch


def main():
    runner.configure_threads()
    torch.cuda.set_device(0)
    out = runner.HERE / "validation"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    tests = runpy.run_path(str(runner.ROOT / "rebuttal/validation/supplement_primitives.py"))
    for name in ("test_final_supplement_variant_grids",
                 "test_vectorized_sana_matches_reference_and_zero_gate_is_identity"):
        tests[name]()
        print(f"PASS {name}", flush=True)
    plan = json.loads((runner.HERE / "plan.json").read_text())
    regression = aggregate.validate_existing(plan)
    runner.atomic_json(runner.HERE / "aggregation_validation.json", {"passed": True, **regression})
    base = plan["bases"]["Daphnet/__root__/seed42"]
    series, validation = runner.load_base_inputs(base)
    parity = []
    for arm in runner.ARMS:
        print(f"PARITY {arm}", flush=True)
        record = runner.run_one(base, runner.EXISTING_OPT, arm, series, validation, out / "parity")
        directory = Path(record["output_path"])
        old_dir = Path(base["references"][arm]["score_path"]).parent
        errors = {}
        for name in ("test_energy.npy", "true.npy", "update_mask.npy"):
            actual = np.load(directory / name, allow_pickle=False)
            expected = np.load(old_dir / name, allow_pickle=False)
            if name == "test_energy.npy":
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                errors["max_score_error"] = float(np.max(np.abs(actual - expected)))
            else:
                np.testing.assert_array_equal(actual, expected)
        if runner.ARMS[arm] == "e7":
            np.testing.assert_array_equal(np.load(directory / "selected_window_mask.npy", allow_pickle=False),
                                          np.load(old_dir / "selected_window_mask.npy", allow_pickle=False))
        errors["pr_auc_error"] = abs(record["auprc"] - base["references"][arm]["auprc"])
        if errors["pr_auc_error"] > 1e-8:
            raise AssertionError(f"Saved PR-AUC parity failed: {arm} {errors}")
        parity.append({"arm": arm, **errors})
        print(f"PASS {arm} {errors}", flush=True)

    # Exercise each new optimizer in the real SANA pipeline. Record the actual
    # optimizer and verify that only the two SANA modules enter its parameters.
    original_builder = runner.engine._build_tta_optimizer
    observed = []

    def checked_builder(module, config):
        optimizer = original_builder(module, config)
        if isinstance(module, torch.nn.ModuleDict):
            assert set(module.keys()) == {"sana_in", "sana_out"}
            expected = {id(p) for p in module.parameters()}
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert actual == expected
            group = optimizer.param_groups[0]
            assert group["lr"] == config.learning_rate and group["weight_decay"] == 0
            if config.optimizer == "sgd":
                assert isinstance(optimizer, torch.optim.SGD) and group["momentum"] == 0
            else:
                assert isinstance(optimizer, torch.optim.AdamW) and group["betas"] == (0.9, 0.999)
            observed.append(config.key)
        return optimizer

    runner.engine._build_tta_optimizer = checked_builder
    smoke = []
    try:
        for optimizer in runner.NEW_OPTS:
            for arm in (a for a in runner.ARMS if runner.ARMS[a] == "e7"):
                print(f"OPTIMIZER CHECK {optimizer} {arm}", flush=True)
                record = runner.run_one(base, optimizer, arm, series, validation,
                                        out / "optimizer_checks", max_test_windows=128)
                meta = json.loads((Path(record["output_path"]) / "result.json").read_text())
                if arm == "oracle_normal_window_sana":
                    assert meta["optimizer_steps"] > 0
                    assert meta["sana_changed_parameter_names"]
                smoke.append({"optimizer": optimizer, "arm": arm, "windows": meta["test_windows"],
                              "steps": meta["optimizer_steps"], "backbone_changed": meta["backbone_changed"]})
    finally:
        runner.engine._build_tta_optimizer = original_builder
    assert sorted(observed) == sorted(list(runner.NEW_OPTS) * 3)
    report = {"status": "passed", "finished_at": runner.now(), "wall_seconds": time.perf_counter() - started,
              "existing_protocol_and_sana_tests": "passed", "parity": parity, "new_optimizer_checks": smoke,
              "validation_outputs_excluded_from_experiment_grid": True}
    runner.atomic_json(runner.HERE / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
