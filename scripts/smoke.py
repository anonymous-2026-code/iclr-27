#!/usr/bin/env python3
"""CPU smoke check: no training, optimizer steps, benchmark inference or saved runs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_artifact.aggregate import point_auprc
from section5_repro.engine import _build_model, _point_loss
from section5_repro.protocol import load_protocol
from review_artifact.score_shift import pair_transitions, pair_features
from section5_repro.sana import SANAPipeline
from section5_repro.supplement_selectors import CadenceController, FPMReference, FPMSelector, perturb_oracle_mask
from review_artifact.backbone_engine import build_model, source_loss, point_score, tta_loss, protocol_for
from review_artifact.figure_c1 import _bucket_stats_single_seed
from review_artifact.backbone_analysis import aggregate_block
from review_artifact.cli import entity_records


def run(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def check_primitives(model, x):
    nf, no = np.array([1., 3., 3.]), np.array([.5, 4., 2.])
    af, ao = np.array([2., 3., 5., 5.]), np.array([1., 3., 3., 6.])
    observed = pair_transitions(nf, no, af, ao)
    before, after = af[:, None] > nf[None, :], ao[:, None] > no[None, :]
    assert observed["wrong_to_correct"] == int((~before & after).sum())
    assert observed["correct_to_wrong"] == int((before & ~after).sum())
    assert observed["unchanged_correct"] == int((before & after).sum())
    assert observed["unchanged_wrong"] == int((~before & ~after).sum())
    assert observed["net_pair_gain"] == float(after.mean() - before.mean())
    features = pair_features([0., 2., 1., 3.], [0., 1., 2., 3.], [0, 0, 1, 1])
    assert len(features) == 9
    assert all(r["selective_compression"] == 2 and r["net_pair_gain"] == 1 for r in features)
    labels = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.int8)
    eroded, _ = perturb_oracle_mask(labels, "boundary_erosion", .1, 20260725)
    dilated, _ = perturb_oracle_mask(labels, "boundary_dilation", .1, 20260725)
    np.testing.assert_array_equal(eroded, [1, 1, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(dilated, [0, 0, 0, 0, 0, 0, 0])
    contaminated, _ = perturb_oracle_mask(labels, "fn_contamination", .1, 20260725)
    assert int(((labels == 1) & (contaminated == 1)).sum()) == 1
    # A recording stub checks scheduling and scaling without an optimizer update.
    class OptimizerRecorder:
        def __init__(self): self.steps = 0
        def zero_grad(self, **kwargs): pass
        def step(self): self.steps += 1
    class LossRecorder:
        def __init__(self): self.divisors = []
        def __truediv__(self, divisor): self.divisors.append(divisor); return self
        def backward(self): pass
    optimizer, loss = OptimizerRecorder(), LossRecorder()
    controller = CadenceController(optimizer, 4)
    assert [controller.backward(loss) for _ in range(6)] == [False, False, False, True, False, False]
    controller.discard_tail()
    assert optimizer.steps == 1 and loss.divisors == [4] * 6
    reference = FPMReference(1., torch.tensor([[1., 0.]], dtype=torch.float64),
                             torch.tensor([[0., 1.]], dtype=torch.float64), torch.eye(2, dtype=torch.float64),
                             .5, 2, 1, 2, 1., 2)
    selector = FPMSelector(torch.nn.Identity(), reference)
    decision = selector.select(torch.tensor([[3., 3.], [.5, .5], [3., 3.]]),
                               raw=torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]]]),
                               labels=torch.tensor([999.]), window_index=0)
    assert decision.mask.tolist() == [[True, True], [True, True], [False, False]]
    pipeline = SANAPipeline(model, variables=3, window=12).eval()
    with torch.inference_mode():
        torch.testing.assert_close(pipeline(x), model(x), rtol=0, atol=0)
    assert not any(p.requires_grad for p in pipeline.backbone.parameters())


def compare_csv(actual, reference, keys, columns=None):
    a, b = pd.read_csv(actual), pd.read_csv(reference)
    columns = list(b.columns) if columns is None else columns
    a = a[columns].sort_values(keys).reset_index(drop=True)
    b = b[columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=0, atol=1e-14)


def check_backbones():
    x = torch.arange(72, dtype=torch.float32).reshape(2, 12, 3) / 72
    model = build_model("USAD", 3, "cpu")
    a, b, c = model(x)
    validation = .5 * ((a - x).square().mean(-1) + (b - x).square().mean(-1))
    torch.testing.assert_close(source_loss(model, x, "val"), validation.mean())
    detection = .5 * (a - x).square().mean(-1) + .5 * (c - x).square().mean(-1)
    torch.testing.assert_close(point_score(model, x), detection)
    torch.testing.assert_close(tta_loss(model, x), detection)
    assert not torch.allclose(validation.mean(), detection.mean())
    names = [n for n, _ in model.get_tta_named_trainable_params("reconstruction_path")]
    assert len(names) == 12 and all(n.startswith(("core.decoder1.", "core.decoder2.")) for n in names)
    model = build_model("LSTMED", 3, "cpu")
    model.eval()
    with torch.no_grad():
        autoregressive = point_score(model, x)
    model.train()
    teacher_forced = tta_loss(model, x)
    assert autoregressive.shape == teacher_forced.shape == (2, 12)
    assert not torch.allclose(autoregressive, teacher_forced)
    names = [n for n, _ in model.get_tta_named_trainable_params("reconstruction_path")]
    assert names == ["backbone.decoder.weight_ih_l0", "backbone.decoder.weight_hh_l0",
                     "backbone.decoder.bias_ih_l0", "backbone.decoder.bias_hh_l0",
                     "backbone.output_layer.weight", "backbone.output_layer.bias"]
    assert protocol_for("LSTMED").frozen_test_batch_size == 1
    assert protocol_for("USAD").frozen_test_batch_size == 64
    # Historical percentile ranks include average ties and the upper boundary.
    bins = _bucket_stats_single_seed(np.array([1., 1., 2., 3.]), np.array([1., 3., 5., 7.]), 10)
    assert bins.loc[bins.bin_id == 3, "median_delta"].item() == 2.
    assert bins.loc[bins.bin_id == 7, "count"].item() == 1
    assert bins.loc[bins.bin_id == 9, "count"].item() == 1


def check_backbone_support():
    # Synthetic metrics test the fresh-grid branch without running a model.
    records = []
    for entry in entity_records():
        for seed in range(42, 47):
            for optimizer, arm, value in [("shared", "frozen", .25)] + [
                (c.key, arm, value) for c in load_protocol().tta_configs
                for arm, value in (("practical_static_p95", .25), ("oracle_benchmark_normal_point", .26))
            ]:
                records.append({"dataset": entry["dataset"], "subset": entry["subset"], "seed": seed,
                                "optimizer_id": optimizer, "arm": arm, "corrected_auprc": value,
                                "raw_valid": True, "label_digest": "synthetic-matched-labels"})
    frame = pd.DataFrame(records)
    # One undefined arm excludes its whole entity across all seeds/optimizers.
    bad = frame.index[(frame.dataset == "MSL") & (frame.subset == "C-1") & (frame.arm == "oracle_benchmark_normal_point")][0]
    frame.loc[bad, ["raw_valid", "corrected_auprc"]] = [False, np.nan]
    cells, support = aggregate_block(frame, "USAD", "full_model")
    assert len(cells) == 180 and (cells.category == "OOG").all()
    assert (cells.loc[cells.dataset == "MSL", "retained_entities"] == 26).all()
    assert (support.status == "excluded").sum() == 1
    try:
        aggregate_block(frame.iloc[1:], "USAD", "full_model")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing backbone run was silently accepted")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    torch.set_num_threads(1)
    for script in ("reproduce.py", "make_tables.py", "render_figures.py", "bootstrap_ci.py", "analyze_score_shift.py", "reproduce_variants.py", "reproduce_backbones.py", "render_figure_c1.py"):
        run(script, "--help")
    plan = json.loads(run("reproduce.py", "run", "--dry-run", "--devices", "cpu"))
    assert (plan["source_tasks"], plan["score_runs"], plan["logical_cells"]) == (580, 5220, 180)
    variant_plan = json.loads(run("reproduce_variants.py", "run", "--dry-run", "--devices", "cpu"))
    assert variant_plan["tasks"] == 2900 and variant_plan["new_score_runs"] == 9860
    assert variant_plan["scores_by_group"] == {"m2n2": 1740, "fpm": 580, "fpm_sana": 2320, "labels": 1740, "cadence": 3480}
    backbone_plan = json.loads(run("reproduce_backbones.py", "run", "--dry-run", "--devices", "cpu"))
    assert (backbone_plan["source_tasks"], backbone_plan["unique_score_runs"], backbone_plan["logical_cells"]) == (1160, 19720, 720)
    model = _build_model(load_protocol(), channels=3).eval()
    x = torch.arange(36, dtype=torch.float32).reshape(1, 12, 3) / 36
    with torch.no_grad():
        scores = _point_loss(model, x, stage="test")
    assert scores.shape == (1, 12) and torch.isfinite(scores).all()
    check_primitives(model, x)
    check_backbones()
    check_backbone_support()
    valid, value, _ = point_auprc([0, 1, 0, 1], [.9, .8, .7, .6])
    assert valid and abs(value - 1/3) < 1e-14
    assert not point_auprc([0, 1], [0., float("nan")])[0]
    try:
        point_auprc([0, 1], [0.])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched arrays were accepted")
    print("PASS: entry points, dry-run plans, CPU model forwards, metrics and protocol primitives.")
    print("No training, optimizer steps or benchmark inference was performed.")
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}))


if __name__ == "__main__":
    main()
