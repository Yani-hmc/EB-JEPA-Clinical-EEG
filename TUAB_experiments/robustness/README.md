# TUAB Hyperparameter Tuning + Robustness + Ablation

Five experiments distilled from a 5-stage optimization campaign on EB-JEPA TUAB:
**capacity → augmentation → schedule → loss coefficients → architecture/data**.

**Row 1 is the tuned model** - the winner of the campaign. Rows 2–5 each
perturb that recipe along one axis; row 1b runs the tuned model with 3 seeds
to quantify the seed-noise band.

## Reference table

→ [`reference_table_robustness_V1.md`](reference_table_robustness_V1.md)

| Script | Row | Test | Δ vs tuned (per-rec BAcc) | Expected per-rec BAcc / AUROC |
|---|---|---|---|---|
| [`run_row1_tuned.sh`](run_row1_tuned.sh) | 1 | **TUNED MODEL** (Conv 1.35M, VICReg + spec 0.1 + corrupt, inv=1, no scale-jitter, 20 ep) | - | **0.824 / 0.896** |
| [`run_row1b_tuned_3seed.sh`](run_row1b_tuned_3seed.sh) | 1b | tuned, 3 seeds | (noise band ≈ ±0.02) | 0.812 ± .021 / 0.898 ± .002 |
| [`run_row2_25pct_data.sh`](run_row2_25pct_data.sh) | 2 | Robustness - 25% data | −0.008 | 0.816 / 0.895 |
| [`run_row3_inv25.sh`](run_row3_inv25.sh) | 3 | HP tuning - inv=25 (VICReg default) | −0.020 | 0.804 / 0.884 |
| [`run_row4_no_mask.sh`](run_row4_no_mask.sh) | 4 | Ablation - remove corruption mask | −0.028 | 0.796 / 0.891 |
| [`run_row5_transformer.sh`](run_row5_transformer.sh) | 5 | Ablation - Transformer at ~1.3M params | −0.099 | 0.725 / 0.804 |

## Reading the table - what each row teaches us

- **1b - Seed robustness (the noise band).** Per-recording BAcc carries
  ≈±0.02 seed noise on 276 recordings (threshold-sensitive - seed 1 dipped to
  0.788). Per-**window** BAcc (±.001) and AUROC (±.002) are stable. Rank
  configs on per-window / AUROC, **not** single-seed per-rec.
- **2 - Robustness to data.** Cutting pretraining data 4× costs only
  0.8 pp - the encoder is **not data-starved**; it saturates early.
- **3 - HP tuning.** VICReg's vision-default invariance weight (25) is
  **2 pp worse** than our tuned `inv=1`. EEG SSL prefers **weak**
  view-invariance (full swept curve: 1 > 5 > 10 > 25).
- **4 - Augmentation ablation.** The corruption mask is the **single biggest
  lever** (−2.8 pp when removed); for contrast, scale-jitter and the
  spectral term are ≈0 individually.
- **5 - Architecture ablation.** A from-scratch Transformer at *equal*
  ~1.3M params is **10 pp worse** - at our scale / epoch budget the conv
  wins decisively; the SOTA gap is pretraining corpus + schedule, not
  architecture.

## How each script works

Same pattern as the main TUAB scripts: one `python -m examples.eeg.main`
call with the row-specific dotted overrides, then `eval.py` for both
recording and window-level scoring. The tuned config (`hidden=96 depth=4
inv_coeff=1 spectral_coeff=0.1 aug_exact_corruption=true aug_scale_jitter=0`)
is the **baseline** for this directory; each row script only changes the
ONE knob the experiment is testing.
