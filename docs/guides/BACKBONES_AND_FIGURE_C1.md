# Table 2 and Figure C.1

These entries complete the submitted cross-backbone experiments and the
representative score-shift figure cited in the main text. They contain no
post-submission experiment settings. Run commands from the package root after
the [environment/data setup](../../README.md).

## Table 2: execute from data

```bash
python scripts/reproduce_backbones.py plan
python scripts/reproduce_backbones.py run --dry-run --devices cpu
python scripts/reproduce_backbones.py run --devices 0,1,2,3 \
  --processed data/processed --workspace workspace/backbones
python scripts/reproduce_backbones.py aggregate --workspace workspace/backbones \
  --output outputs/backbone_derived
python scripts/make_tables.py --table2-only --evidence outputs/ae_derived \
  --backbone-derived outputs/backbone_derived/backbones --output outputs/table2
```

The last command uses the AE `core_cells.csv` produced by the main experiment.
Use `--devices 0` or
`--devices cpu` as needed. Execution filters `--backbones USAD` (or `LSTMED`),
`--datasets` and `--seeds` are available, but aggregation requires both
backbones and the complete dataset/seed grid. Repeating a run resumes completed
matching outputs; incomplete or conflicting outputs fail explicitly.

The full grid has **1,160 source trainings, 19,720 unique score runs and
720 logical cells**. Each backbone/entity/seed task runs one shared Frozen
detector and 16 adaptive arms: two update scopes, four optimizers and
Practical/Oracle. Frozen is reused across scopes. Raw aggregation therefore
has 20,880 metric memberships, not 20,880 unique score runs.

Shared data preparation, source-only StandardScaler, chronological 80/20 split,
12/12 windows, source Adam 1e-4, batch 64, ten epochs, patience three and the
historical type1 LR schedule follow the main data/source contract. Backbone
settings are in `configs/backbones.toml`. The portable loops are in
`src/review_artifact/backbone_engine.py`. The historical USAD and LSTMED model
definitions in `historical_models/` retain their original state-dict names and
are unchanged copies of their submitted generating sources.

| Protocol detail | USAD | LSTMED |
|---|---|---|
| Architecture | Latent dimension 5; two decoders | Hidden dimension 5; one encoder/decoder LSTM layer; effective dropout 0 |
| Source training loss | Historical epoch-dependent `compute_loss`, n = epoch index + 2 | Reconstruction MSE with teacher forcing |
| Source validation | `compute_loss(stage='val')`, n=2; **not detection score** | Autoregressive reconstruction MSE in eval mode |
| Detection/calibration score | 0.5 × decoder-1 MSE + 0.5 × chained decoder-2 MSE | Autoregressive point MSE in eval mode |
| Frozen test batch size | 64 | 1 |
| Decoder-only optimizer parameters | Both decoders, 12 tensors | Decoder LSTM and output layer, 6 tensors |

The original validation function does not pass an epoch to USAD, so n=2 is
fixed for checkpoint selection. Source test-loss logging is omitted; its
DataLoader seed draw is retained to preserve training shuffle progression.
No test labels enter source checkpoint selection.

Every adaptive arm reloads the same entity/seed source. Threshold calibration
uses the fixed validation P95 in eval mode. Each chronological test window is
scored in eval mode **before** adaptation. The model then enters train mode;
Practical selects update-loss points strictly below the threshold and Oracle
selects label-zero points. For LSTMED, this update loss is teacher-forced even
though calibration/detection are autoregressive. The masked loss averages over
all 12 positions; empty masks still receive the scheduled optimizer step.
SGD has zero momentum; AdamW uses betas 0.9/0.999 and epsilon 1e-8; weight decay
is zero. Decoder-only changes the optimizer's parameter list, preserving the
historical gradient path. There is no clipping, reset or alternate cadence.

Outputs retain complete pre-update scores, labels, point indices and update
traces. Non-finite complete score streams are explicitly marked invalid, never
cropped or finite-filtered. Source/calibration failures and structural
mismatches raise errors. Aggregation derives each backbone/scope's common
entity support across all seeds, optimizers and arms, computes trapezoidal
PR-curve AUC and takes medians of paired entity effects. It does not force
historical exclusions or category counts onto fresh runs.

## Figure C.1

```bash
# Recompute all bucket statistics from a completed main AE workspace.
MPLBACKEND=Agg python scripts/render_figure_c1.py --workspace workspace/ae \
  --output outputs/figure_c1_raw
```

The command writes `fig5_3_2.pdf`, `fig5_3_2.png`, and class-specific per-seed
and aggregate CSVs under `PUMP/` and `Daphnet/`. The configuration is AE,
AdamW 1e-4, seeds 42–46. Scores are fully matched pre-update Frozen/Oracle
arrays. Score shift here is **Oracle minus Frozen**, the opposite sign from
the score-drop variable in the systematic Section 4.3 analysis.

The six numerical/drawing functions in `review_artifact/figure_c1.py` are
copied from the submitted working renderer. Within each label class and seed,
bins use average-tie percentile ranks:
`clip(floor(10 * rank(pct=True, method='average')), 0, 9)`. This preserves the
historical boundary convention rather than substituting quantile cuts. Each
bar is the median across five seed-level median score shifts. There are 200
seed/class/bin records and 40 bars; the plot also retains its original shared
axis limits and layout. Its illustrative deciles differ from the 15%/30%
tails used for selective compression and net pair gain.
