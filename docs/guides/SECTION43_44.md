# Sections 4.3 and 4.4

Run commands from the package root. Use the environment and data preparation
steps in the [README](../../README.md). All output directories must be new.
The submitted protocol uses the same nine datasets, 116 physical entities and
five seeds as the core AE experiment. These commands analyze or execute that
protocol; they do not introduce additional experiment settings.

## Section 4.3: score shifts and pair ordering

After the complete core AE run and its corrected aggregation:

```bash
python scripts/analyze_score_shift.py --workspace workspace/ae \
  --cells outputs/ae_derived/core_cells.csv --output outputs/score_shift
```

The script reads saved pre-update Frozen and Oracle scores, original labels
and completion metadata. It requires exact entity, seed, protocol, dtype and
length matches. It performs no training or inference.

For each entity/optimizer/seed, tails are defined once using **Frozen scores**:

- Hard normals have scores at or above the normal `(1 - alpha)` quantile.
- Boundary anomalies have scores at or below the anomaly `beta` quantile.
- Score drop is Frozen minus Oracle. Selective compression is the median drop
  on hard normals minus the median drop on boundary anomalies.
- A normal/anomaly pair is correctly ordered only when the anomaly score is
  strictly greater. Ties are incorrect. Exact Cartesian pair-transition
  counts use a Fenwick tree, without pair sampling. Net pair gain is
  `(wrong_to_correct - correct_to_wrong) / n_pairs`.

Quantile ties are included. The primary setting is `alpha=0.15, beta=0.30`.
Sensitivity settings are `alpha in {0.10, 0.15, 0.20}` crossed with
`beta in {0.20, 0.30, 0.40}`. An entity with any non-finite Frozen or Oracle
score is undefined for that diagnostic; no finite prefix is substituted.
Logical diagnostic features are entity medians over the finite support of
each cell. This support differs from the globally intersected detection
support. Detection effects come from the matching corrected `core_cells.csv`.

Outputs are `physical_features.csv` (20,880 rows),
`score_shift_full_analysis.csv` (1,620 rows), and
`score_shift_associations.csv` (42 estimates). The last file contains pooled
Spearman correlations for every tail setting; the primary setting also has
dataset-median, pooled within-dataset rank, leave-one-dataset-out and
complete-support-only estimates. Within-dataset ranks use average ties.
These are descriptive associations, not causal estimates.

## Section 4.4: execution

Complete the core AE grid first. Keep its source checkpoints, controls and
variants in the same workspace:

```bash
python scripts/reproduce_variants.py plan
python scripts/reproduce_variants.py run --dry-run --devices cpu
python scripts/reproduce_variants.py run --devices 0,1,2,3 \
  --processed data/processed --workspace workspace/ae
python scripts/reproduce_variants.py aggregate --workspace workspace/ae \
  --core-derived outputs/ae_derived --output outputs/variant_derived
python scripts/make_tables.py --variants-only --evidence outputs/variant_derived \
  --output outputs/variant_tables
```

Use `--devices 0` or `--devices cpu` as needed. `--groups` accepts a
comma-separated subset of the names below; `--datasets` and `--seeds` use the
core launcher's filters. Repeating the run resumes completed outputs.
Aggregation requires the complete grid, not a filtered subset. The inherited
engine can create missing source/control runs, so complete the core stage
first to reuse its matched checkpoints and avoid unintended extra work.

| Group | Submitted arms | Additional score runs |
|---|---|---:|
| `m2n2` | Detrending-only, Practical P95, Oracle point mask | 1,740 |
| `fpm` | FPM-only Practical | 580 |
| `fpm_sana` | Frozen-SANA, Practical-FPM-SANA, normal-window Oracle, clean-FPM Oracle | 2,320 |
| `labels` | False-negative contamination, boundary erosion, boundary dilation at 0.10 | 1,740 |
| `cadence` | Practical and Oracle at B=2/4/8; B=1 reuses core | 3,480 |

There are 2,900 entity/seed/group tasks and 9,860 additional score runs.
Execution delegates to the submitted implementation in
`src/section5_repro/supplement_engine.py`; the portable launcher selects the
exact submitted arms. Hyperparameters are in `configs/supplements.toml`.
The shared implementation/configuration retains unused legacy branches to
preserve its loader and source behavior; this launcher does not select them,
the auxiliary SGD/0.05 label settings, B=16, or rebuttal settings.

All selected adaptations use AdamW at 1e-4 with zero weight decay. Scores are
recorded before each update, using windows/stride 12/12. Specific comparisons:

- **M2N2:** the current window updates the EMA mean before detrending,
  reconstruction and scoring. Gamma is 0.60 for MSL, 0.80 for SMAP, 0.99999
  for SWAT2015 and 0.99 otherwise. The validation P95 threshold remains fixed.
  Both adaptive arms are compared to the matched detrending-only detector.
- **FPM-only:** whole-window selection uses frozen source-encoder normalized
  latents, source-validation score percentile 99.5, hard tail 0.5%, moderate
  reference percentiles 25/75 and chi-square probability 0.05. Covariance uses
  `pinv` with no regularization. The full AE adapts; comparison uses the core
  Frozen and point-mask Oracle.
- **FPM+SANA:** only input/output SANA modules adapt; the AE remains frozen.
  Dimension 512, eight heads, feed-forward dimension 512 and zero gates match
  the submitted source. Initialization uses `seed + 7000`. All four arms
  start from the same modules. Normal-window Oracle selects entirely normal
  windows; clean-FPM Oracle intersects those windows with FPM selection. Each
  comparison uses Frozen-SANA. Updates occur only on selected windows.
- **Cadence:** accumulate each masked point-mean loss divided by B, step every
  B windows and discard any incomplete final accumulation. Empty masks retain
  scheduled step behavior. No score is recomputed after the update.

## Label perturbations and matched aggregation

The fraction is 0.10 and the mask seed is 20260725. False-negative
contamination marks `ceil(0.10 * anomaly_points)` points as eligible for
normal-only updates, traversing randomly ordered anomaly segments and using a
contiguous part of the last selected segment. For each anomaly segment of
length L, boundary width is
`min(max(1, ceil(0.10 * L)), floor((L - 1) / 2))`, or zero when L <= 1.
Erosion removes this width from both anomaly boundaries; dilation extends
them into adjacent normal points. Perturb the full sequence before dropping
the final incomplete window. The original labels remain available.

| Channel | Oracle scores | Evaluation labels for every arm |
|---|---|---|
| `mask-only` | Adaptation with perturbed normal mask | Original |
| `evaluation-only` | Original core Oracle adaptation | Perturbed |
| `joint` | Adaptation with perturbed normal mask | Perturbed |

Frozen and Practical scores are reused, but their metrics are recomputed on
the perturbed labels in the last two channels. Labels are matched across all
three arms before subtraction. Metric computation uses trapezoidal PR-curve
AUC, never the inherited engine's legacy AP field.

Each comparison intersects valid entity support across its three arms and all
five seeds, then takes medians of paired entity effects. The B=1 reference and
label channels additionally use the core grid's global support. Other variant
rows use their own matched support. Non-finite runs exclude whole entities;
missing files, incomplete grids or mismatched labels fail explicitly.

Aggregation writes `variant_physical.csv` (18,560 metric records),
`robustness_outcomes.csv` (360 cells), `robustness_summary.csv`,
`label_perturbation_outcomes.csv` (405 cells) and
`label_perturbation_summary.csv`. Repeated evaluation channels reuse scores,
so the metric-record count is not a count of additional experiments.
The table command writes CSV and LaTeX for Tables B.2/B.3, which support the
Section 4.4 claims. Retention compares the corrected category to the matched
unperturbed B=1 category; the strict clear-gain threshold is 0.005.
