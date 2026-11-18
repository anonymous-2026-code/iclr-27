# Third-party notices

The inherited research implementation derives from an MIT-licensed
Time-Series-Library-based codebase. The supplied MIT license is retained in
`LICENSE`. Python dependencies retain their respective licenses.

No external checkout is required at runtime, and no third-party benchmark
samples are distributed in this package. Public provider names and URLs in
the acquisition guide identify third-party datasets, not the submission's
authors or affiliations. Consult the original distribution terms for
Telemanom, OmniAnomaly, RANSynCoders, TAB and TSB-AD-M before acquiring or
redistributing their data. A code repository's license does not necessarily
cover the dataset it links to.

The supplemental source includes the inherited M2N2, FPM and SANA
implementations used by the submitted comparisons. FPM is a protocol-specific,
CANDI-inspired implementation; it does not vendor or import CANDI. SANA is
included as local source, with no runtime link to another checkout. The
existing range-volume metric implementation is included for rebuttal VUS-PR.

The Table 2 LSTMED model retains its source attribution to Merlion's LSTM-ED
implementation. Both historical backbone model files are included unchanged;
their upstream derivation remains relevant to attribution/license review.

The inherited source notice records an outstanding public-release review of
the provenance, derivation, attribution, NOTICE and license compatibility of
`sana.py`, the M2N2/FPM/SANA portions of `supplement_engine.py`, and the FPM
portions of `supplement_selectors.py`. Preparing this local review package
does not resolve that review or assert additional third-party permissions.

The rebuttal Affiliation-F analysis includes six unchanged Python evaluation
modules from the TSB-AD distribution under `rebuttal/vendor/affiliation/`.
Its Apache-2.0 distribution license is retained in
`rebuttal/vendor/TSB-AD-LICENSE`; upstream comments and citations are preserved.
These public third-party attributions do not identify the submission authors.
