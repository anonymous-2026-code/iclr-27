# Submitted core protocol

The executable hyperparameters are in `configs/section5_ae.toml`. The
[Section 4.3/4.4 guide](../guides/SECTION43_44.md) specifies the additional
diagnostics and the submitted subset of `configs/supplements.toml`.

| Item | Setting |
|---|---|
| Datasets | Daphnet, GECCO, Genesis, MSL, PSM, PUMP, SMAP, SMD, SWAT2015 |
| Physical entities before validity exclusions | 116: MSL 27, SMAP 55, SMD 28, six single-entity datasets |
| Seeds | 42, 43, 44, 45, 46 |
| Adaptation | SGD at 1e-3 / 1e-4; AdamW at 1e-4 / 3e-4; weight decay 0 |
| Windows | Length 12, stride 12, no overlap, incomplete tail discarded |
| Source training | Adam 1e-4, batch 64, at most 10 epochs, patience 3, existing type1 LR schedule |
| Source split | Chronological 80/20; validation reconstruction loss selects the checkpoint |
| Online processing | One window at a time, chronological order, full-model update each window |
| Clear gain | Unrounded paired effect strictly greater than 0.005 |

Source normalization uses StandardScaler fitted to the complete official source
training split, before its chronological train/validation division. Benchmark
preprocessing drops a column only if its variance is zero in both source and
test. This existing preprocessing rule is retained. Test values and labels are
not used for checkpoint selection; the entry point disables test-loss logging
during source training and preserves the original random-number cadence.

Frozen keeps its source weights. Practical calibrates a fixed 95th percentile
threshold on source-validation reconstruction scores and selects points with
loss strictly below that threshold. Oracle replaces this mask with `label == 0`.
Both adaptive arms reload the same source checkpoint, use the same optimizer,
learning rate and schedule, and retain optimizer state through the test stream.

Each window is scored before its update. Only those pre-update scores enter
detection evaluation. The masked loss is `(point_loss * mask).mean()` over all
12 positions, including zeros for unselected points. Even an empty mask causes
the scheduled optimizer step. Post-update scores are saved solely for immediate
objective diagnostics. They do not replace the detection scores.

Point scores are mean squared reconstruction error across features. Windows
flatten in chronological order. The submitted AUPRC is
`auc(recall, precision)` after `sklearn.metrics.precision_recall_curve`, without
point adjustment. The copied engine also writes a legacy `auprc` log field that
actually contains AP; the review aggregator ignores it and reads raw arrays.

For each physical entity and seed, subtract Frozen AUPRC from matched Practical
and Oracle AUPRC. Then take the median of those **paired differences** over
entities within each dataset/optimizer/seed cell. Do not subtract separate arm
medians. Use one globally valid entity support across all arms, optimizers and
seeds for each dataset. Missing runs or mismatched lengths fail explicitly;
non-finite score runs exclude their entire entity from common support and are
reported. They are never replaced by zero or counted as NI.

Fresh runs derive and report common support using the rule above. The separate
rebuttal input adapter requires the fixed entity identities specified in
`configs/rebuttal_support.json`; it fails if fresh support differs.

BI means both arms have clear gains; OOG means only Oracle; NI means neither;
POG means only Practical. Table 1 summarizes the 20 Oracle cell effects per
dataset in percentage points. Table 2 counts categories per backbone/scope.

Figure 2's objective is the mean pre-minus-post reconstruction loss on normal
points, followed by an entity median. Its finite diagnostic support is separate
from detection support. Confidence intervals resample nine dataset clusters,
not individual entities or optimizer/seed cells: 10,000 replicates, seeds
2701/2702/2703 for OOG, NI and objective/detection Spearman correlation.
