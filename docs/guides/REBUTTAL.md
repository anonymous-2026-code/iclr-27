# Additional experiments: usage

Run these commands after the main AE workflow in the [README](../../README.md).
Only source code is distributed; all required experimental outputs must be
generated locally.

## Prepare inputs from a fresh core reproduction

Complete the main AE reproduction and aggregation in the root README first.
The preserved rebuttal scripts use `data/processed/`, `workspace/ae/` and
an input index under `workspace/rebuttal_inputs/`:

```bash
python scripts/prepare_rebuttal_inputs.py core --derived outputs/ae_derived
```

This reads 5,220 score streams, verifies their identity against the prepared
data and source checkpoints, recomputes trapezoidal PR-AUC and writes the
physical reference manifests expected by the rebuttal readers. It requires
the original fixed 106-entity common support and fails if fresh support
differs. No historical metrics are bundled or substituted for fresh results. Input indexes are generated once; existing indexes are not replaced.

GPU batch scripts retain four workers using GPU IDs 0,1,2,3. Use Linux with
four available GPUs and at least six CPU cores for the complete sequence.
CPU core numbers are selected from the process's allowed affinity, replacing
the original machine's fixed core IDs. This changes resource placement only.
The original small all-normal-window launcher also accepts `--devices`.
The inherited four-worker execution requirement is not a CPU fallback.

## Run the input, scale, shift and count controls

The order matters: later planners reuse earlier measured runtimes, reference
manifests and control results. Each supervisor aggregates its completed batch.

```bash
# 1. Clean windows, retaining scheduled empty-mask optimizer steps.
python rebuttal/experiments/ae_all_normal_window/run.py prepare --devices 0,1
python rebuttal/experiments/ae_all_normal_window/run.py validate
python rebuttal/experiments/ae_all_normal_window/run.py run

# 2. Selected-count normalization and the prespecified affine shifts.
python rebuttal/experiments/gpu_controls/run.py prepare
python rebuttal/experiments/gpu_controls/validate.py
python rebuttal/experiments/gpu_controls/run.py supervise

# 3. Normal shift/headroom and source-validation score histories.
python rebuttal/experiments/normal_shift_headroom/run.py
python rebuttal/experiments/causal_score_calibration/source_scores.py run
python rebuttal/experiments/causal_score_calibration/analyze.py
python rebuttal/experiments/frozen_adaptation_need/run.py

# 4. Strict skip-step/sanitized inputs, plus the three CPU analyses.
python rebuttal/experiments/audit_gap_closure_20261110/experiment.py prepare
python rebuttal/experiments/audit_gap_closure_20261110/validate.py
python rebuttal/experiments/audit_gap_closure_20261110/experiment.py supervise

# 5. Exact H=512 cross-window pair-transition analysis (prints its tables).
python rebuttal/experiments/causal_score_calibration/cross_window_pair_flips.py

# 6. Count matching and the original three-family window experiment.
python rebuttal/experiments/remaining_gpu_batch/experiment.py prepare
python rebuttal/experiments/remaining_gpu_batch/experiment.py validate
python rebuttal/experiments/remaining_gpu_batch/experiment.py supervise

# 7. Extend the same window experiment to all nine families.
python rebuttal/experiments/remaining_gpu_batch/experiment.py window-prepare
python rebuttal/experiments/remaining_gpu_batch/experiment.py window-validate
python rebuttal/experiments/remaining_gpu_batch/experiment.py window-supervise
```

Each experiment writes scores/checkpoints under its own `runs/` or `outputs/`
and CSV/figures under `derived/`. The window extension deliberately expands
its own earlier three-family summary; it does not touch the submission data.
The original fixed seeds, learning rates, masks, loss denominators, prefix
matching and undefined-result rules are retained in the copied implementations.

## Run mechanism diagnostics and geometry

After steps 1–2 above:

```bash
python rebuttal/experiments/future_probe_mechanism/experiment.py prepare
python rebuttal/experiments/future_probe_mechanism/experiment.py validate
python rebuttal/experiments/future_probe_mechanism/experiment.py run
python rebuttal/experiments/future_probe_mechanism/smd_comparison.py prepare
python rebuttal/experiments/future_probe_mechanism/smd_comparison.py run --optimizer sgd
python rebuttal/experiments/future_probe_mechanism/smd_comparison.py run --optimizer adamw
python rebuttal/experiments/future_probe_mechanism/smd_comparison.py aggregate
python rebuttal/experiments/future_probe_mechanism/temporal_dynamics.py
python rebuttal/experiments/future_probe_mechanism/checkpoint_gap_analysis.py
python rebuttal/experiments/future_probe_mechanism/verify_checkpoint_gap.py

# Standalone synthetic experiments; these commands perform updates.
python rebuttal/experiments/linear_ae_subspace_toy/run.py \
  --geometry-output outputs/linear_ae_geometry.csv
python rebuttal/experiments/linear_ae_subspace_toy/run.py \
  --controlled-shift-output outputs/linear_ae_rotation
```

For the full-support extension, first regenerate score-shift features using
the root README raw-score command, then:

```bash
python scripts/prepare_rebuttal_inputs.py score-shift --derived outputs/score_shift
python rebuttal/experiments/future_probe_mechanism/full_grid.py prepare
python rebuttal/experiments/future_probe_mechanism/full_grid.py smoke
python rebuttal/experiments/future_probe_mechanism/full_grid.py launch
python rebuttal/experiments/future_probe_mechanism/full_grid.py status
```

This `full_grid.py smoke` is the original GPU parity replay, not the packaging
smoke check. `launch` is asynchronous. Wait until `status` reports all 2,120
trajectories complete, then run the CPU analysis and independent verifier:

```bash
python rebuttal/experiments/future_probe_mechanism/analyze_full_grid.py
python rebuttal/experiments/future_probe_mechanism/verify_full_grid.py
```

Full-grid outputs go to
`rebuttal/experiments/future_probe_mechanism/checkpoint_gap/`. The analyses
keep one-class future probes undefined, distinguish local state diagnostics
from online full-stream PR-AUC, and retain the original trajectory-parity
checks. Several historical verification scripts assert recorded counts or
counterexample identities. Fresh training deviations may make those checks
fail; their constants must not be changed just to obtain a passing report.

## Re-evaluate VUS-PR and extend the practical methods

```bash
# CPU postprocessing of the core raw scores, followed by paired aggregation.
python scripts/prepare_rebuttal_inputs.py vus
python rebuttal/experiments/vus_pr_robustness/run.py
```

For the method extension, first finish the submitted variant execution and
aggregation from the root README (`outputs/variant_derived/`), then:

```bash
python scripts/prepare_rebuttal_inputs.py variants --derived outputs/variant_derived
python rebuttal/experiments/adaptation_variants_180/run.py prepare
python rebuttal/experiments/adaptation_variants_180/validate.py
python rebuttal/experiments/adaptation_variants_180/run.py supervise
```

The SANA engine accepts an explicit optimizer for this extension; its default
remains the submitted AdamW 1e-4. The validator also emits the aggregation
preflight record consumed by the optional detached launcher. Numeric failures
are terminal records; the aggregation retains the nominal 116-entity support
and marks incomplete cells undefined rather than dropping entities.

After all ten cell-based experiment groups finish, rebuild the navigation
tables from their fresh aggregations:

```bash
python scripts/make_rebuttal_tables.py --experiments rebuttal/experiments \
  --output outputs/rebuttal_tables_fresh
```
