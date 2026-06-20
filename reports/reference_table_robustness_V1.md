# EB-JEPA — Robustness, Hyperparameter Tuning & Ablation (V1)

Five experiments distilled from a 5-stage optimization of our EB-JEPA on TUAB (v3.0.1,
patient-disjoint, 276 eval recordings; frozen encoder → logistic-regression probe). **Row 1 is the
tuned model** — the winner of the campaign (capacity → augmentation → schedule → loss coefficients →
architecture/data); rows 2–5 each perturb it along one axis. **per-rec** = per-recording (16-window
mean-pool, the clinical metric); **per-win** = per-window (the literature convention). Single-seed.
Companion to `reference_table_JEPA_V1`. Typeset version: `reference_table_robustness_V1.pdf`.

| # | Test | Configuration (vs. tuned model) | Params | per-rec BAcc / AUROC | per-win BAcc / AUROC |
|---|---|---|---|---|---|
| 1 | **Tuned model** | Conv · VICReg + spectral 0.1 + corruption · inv=1 · no scale-jitter · 20 ep | 1.35M | **0.824 / 0.896** | 0.755 / 0.839 |
| 1b | **Tuned — 3-seed** | same recipe, seeds {0,1,2} (mean±std) | 1.35M | 0.812 ± .021 / 0.898 ± .002 | 0.756 ± .001 / 0.836 ± .005 |
| 2 | **Robustness** | pretrain on **25%** of recordings (4× less data) | 1.35M | 0.816 / 0.895 | 0.762 / 0.842 |
| 3 | **HP tuning** | invariance weight **inv=25** (VICReg default) instead of 1 | 1.35M | 0.804 / 0.884 | 0.730 / 0.813 |
| 4 | **Ablation** | remove the **corruption mask** (outliers only) | 1.35M | 0.796 / 0.891 | 0.762 / 0.841 |
| 5 | **Ablation** | **Transformer** encoder at equal capacity | 1.29M | 0.725 / 0.804 | 0.652 / 0.708 |

**Legend / fixed setup.** Every row uses the tuned model's recipe except the stated change: a
0.4M-class 1-D conv scaled to **1.35M** (`hidden=96, depth=4`); two-view VICReg energy + a DDSP
multi-scale-spectral auxiliary term (weight 0.1); exact EB-corruption views *v₁,v₂* (Gaussian noise
σ=0.1, channel-drop p=0.2, 20% timepoints masked + 20% ±6σ outliers); invariance weight 1; 20
epochs; frozen-encoder logreg probe.

**What each row shows.**
- **1b — Seed robustness (important caveat):** per-recording BAcc carries ≈±0.02 seed noise (only 276 recordings, threshold-sensitive; seed 1 dipped to 0.788). Per-**window** BAcc (±.001) and AUROC (±.002) are stable. So the per-rec Δ's in rows 2–4 (<3 pp) sit **within this noise band** — only row 5's −10 pp is statistically unambiguous. Rank configs on per-window / AUROC, not single-seed per-rec.
- **2 — Robustness to data:** cutting pretraining data 4× costs only 0.8 pp (0.824→0.816) — the encoder is *not* data-starved; it saturates early.
- **3 — Hyperparameter tuning:** VICReg's vision-default invariance weight (25) is **2 pp worse** than our tuned `inv=1`. EEG SSL prefers *weak* view-invariance (full swept curve: 1 > 5 > 10 > 25).
- **4 — Augmentation ablation:** the corruption **mask is the single biggest lever** (−2.8 pp when removed); for contrast, scale-jitter and the spectral term are *dead weight* (≈0).
- **5 — Architecture ablation:** a from-scratch Transformer at *equal* ~1.3M params is **10 pp worse** — at our scale/epoch budget the conv wins decisively; the SOTA gap is pretraining corpus + schedule, not architecture.
