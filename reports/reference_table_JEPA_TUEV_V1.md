# EB-JEPA — TUEV Reference Table (V1)

Our EB-JEPA runs on **TUEV** (6-class event detection, frozen logistic probe).
Companion to `reference_table_JEPA_V1.pdf` (TUAB) and
`examples/eeg/PROVENANCE_TABLE_TUEV.md` (literature baselines).
Typeset version: `reference_table_JEPA_TUEV_V1.pdf`.

| # | Encoder | Params | Energy (SSL objective) | Eval | TUEV BAcc |
|---|---|---|---|---|---|
| baseline | Conv1D | 0.4M | VICReg — *base aug, no corruption* | frozen | 0.381 |
| 1 | Conv1D | 0.4M | **VICReg + spectral 0.1 + corrupt** | frozen | **0.425** ⭐ best |
| 2 | Conv1D | 0.4M | VICReg + corrupt (s0) | frozen | 0.382 |
| 3 | Conv1D | 0.4M | SIGReg + corrupt | frozen | 0.363 |

**TUEV BAcc** — 6-class macro balanced accuracy on TUEV event detection.
**Chance** = 0.167; **random-encoder floor** = 0.337. All rows use a
**TUAB-pretrained encoder, frozen, never trained on TUEV** — pure
representation-transfer probe.

**Probe protocol** — class-balanced logistic regression on event-centered 5 s
windows (200 Hz, 19 channels, per-channel z-scored), background class
subsampled so the 6 classes (SPSW / GPED / PLED / EYEM / ARTF / BCKG) are
balanced. Subject-disjoint train/eval.

**Fixed for every corruption run** (never swept): views v₁, v₂ made with
Gaussian noise σ = 0.1, scale-jitter ±20 %, channel-drop p = 0.2, plus exact
EB-corruption — 20 % timepoints masked + 20 % ±6σ outliers (≈ 40 %/view). The
**baseline** is the no-corruption reference (legacy contiguous time-mask).

**VICReg\* note** — these runs use the original `inv_coeff = 1` (the 3-seed
ablation showed `inv=1` actually beats `inv=25` by ~3 pp / ~6σ on TUAB;
reframed as a finding rather than a bug in commit `5a13ebd`).

**Reading the table** — best frozen-transfer is **row 1 (VICReg + spectral
0.1 + corrupt → 0.425 BAcc)**, which beats our supervised CNN-Transformer
(0.410) and ST-Transformer (0.339) **without ever seeing a TUEV label**. Every
config lands well above the random-encoder floor (0.337) and the 6-class
chance line (0.167), confirming the TUAB-pretrained representation transfers
to TUEV.
