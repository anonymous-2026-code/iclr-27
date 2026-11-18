#!/usr/bin/env python3
"""CPU import, protocol-primitive and saved-table checks; no optimizer steps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "rebuttal/experiments"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

CHECKS = {
    "ae_all_normal_window/run.py": """
mask, stats = window_mask(np.array([0]*12 + [0]*11 + [1] + [1]*12))
assert mask.sum() == 12 and stats['mixed_windows'] == 1
assert stats['normal_points_dropped'] == 11
""",
    "gpu_controls/run.py": """
loss = torch.arange(1., 13.).reshape(1,12)
mask = torch.zeros(1,12,dtype=torch.bool); mask[0,:3] = True
assert reduction(loss, mask, 'full')[0].item() == .5
assert reduction(loss, mask, 'selected')[0].item() == 2.
mask.zero_(); assert reduction(loss, mask, 'selected')[0].item() == 0.
""",
    "audit_gap_closure_20261110/experiment.py": """
x = torch.arange(24.).reshape(1,12,2)
mask = torch.ones(1,12,dtype=torch.bool); mask[0,3] = False
reference = torch.tensor([4.,5.])
y = sanitized_input(x, mask, reference)
assert torch.equal(y[mask], x[mask]) and torch.equal(y[0,3], reference)
""",
    "causal_score_calibration/analyze.py": "validate_causality()",
    "causal_score_calibration/cross_window_pair_flips.py": "validate_counter()",
    "remaining_gpu_batch/experiment.py": """
mask = np.ones((3,12),dtype=bool); counts = np.array([0,4,12])
selected = thin_mask(mask, counts, np.random.default_rng(42))
np.testing.assert_array_equal(selected.sum(1), counts)
assert not (selected & ~mask).any()
""",
    "future_probe_mechanism/experiment.py": """
labels = np.array([0,1]*36,dtype=np.int8)
groups = future_indices(labels, 1)
assert all((indices >= 24).all() for indices in groups.values())
assert pair_fraction(np.array([0.,1.]),np.array([1.,2.])) == .75
assert pair_fraction(np.empty(0),np.array([1.])) is None
""",
    "linear_ae_subspace_toy/run.py": """
w = torch.tensor([.7,.2],dtype=torch.float64,requires_grad=True)
x = torch.tensor([1.,.4],dtype=torch.float64)
loss = (x-w*(w@x)).square().sum()
actual, = torch.autograd.grad(loss,w)
np.testing.assert_allclose(actual.numpy(),analytical_gradient(w.detach().numpy(),x.numpy()),rtol=0,atol=1e-12)
""",

}


def check_input_index():
    module = runpy.run_path(str(ROOT / "scripts/prepare_rebuttal_inputs.py"))
    protocol = module["load_protocol"]()
    Entity = module["Entity"]
    with tempfile.TemporaryDirectory(prefix="oracle-rebuttal-index-") as directory:
        temp = Path(directory)
        run = temp / "test_run"
        run.mkdir()
        meta = {"test_rows": 24}
        complete = {"entity": meta, "seed": 42, "protocol": protocol.as_dict(), "setting": run.name}
        (run / "complete.json").write_text(json.dumps(complete))
        (run / "metrics.json").write_text(json.dumps({"selector_tau": .5}))
        labels = np.array([0,1]*12,dtype=np.int8)
        np.save(run / "true.npy",labels)
        np.save(run / "test_energy.npy",labels.astype(float))
        row = module["index_core_run"](run,Entity("Daphnet","",temp),42,"shared","frozen",protocol,meta,"test-checkpoint",temp/"index.json")
        assert row["n_total"] == 24 and row["n_valid"] == 24
        assert row["corrected_pr_curve_auc"] == 1 and row["raw_valid"]
        # Exercise the VUS adapter's file schema as well as its tiny metric
        # computation: the preserved downstream reader requires both hashes.
        core_file = temp / "core.parquet"
        pd.DataFrame([row]).to_parquet(core_file, index=False)
        module["vus"].__globals__.update(OUT=temp, CORE_FILE=core_file)
        module["vus"]()
        vus = pd.read_parquet(temp / "iclr27_main_v1_physical.parquet")
        assert vus["value"].between(0, 1).all()
        assert vus["score_sha256"].str.len().eq(64).all()
        assert vus["base_score_sha256"].str.len().eq(64).all()
        # Bad lengths must fail instead of silently cropping the evaluation.
        np.save(run / "test_energy.npy",np.ones(23))
        try:
            module["index_core_run"](run,Entity("Daphnet","",temp),42,"shared","frozen",protocol,meta,"test-checkpoint",temp/"bad.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Misaligned score/label inputs were accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    scripts = sorted(EXPERIMENTS.rglob("*.py"))
    for script in scripts:
        compile(script.read_text(), str(script), "exec")
        extra = CHECKS.get(script.relative_to(EXPERIMENTS).as_posix(), "")
        command = ("import runpy,sys; from pathlib import Path; "
                   "sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
                   "ns=runpy.run_path(sys.argv[1],run_name='artifact_smoke'); "
                   "exec(sys.argv[2],ns)")
        result = subprocess.run([sys.executable,"-B","-c",command,str(script),extra],cwd=ROOT,env=env,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f"{script.relative_to(ROOT)}\n{result.stdout}\n{result.stderr}")
        print("PASS",script.relative_to(ROOT),flush=True)
    check_input_index()
    preflight = runpy.run_path(str(ROOT / "rebuttal/validation/supplement_primitives.py"))
    preflight["test_final_supplement_variant_grids"]()
    preflight["test_vectorized_sana_matches_reference_and_zero_gate_is_identity"]()
    print("PASS: experiment imports, protocol primitives and synthetic input index; no training, benchmark inference or optimizer steps.")


if __name__ == "__main__":
    main()
