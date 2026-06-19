# Runtime Estimates on Dalia (GB200, 185 GB VRAM)

## Reference timings from actual jobs (team vivatech-slightlyunawarefc)

| Job ID | Name | Elapsed | What it did |
|--------|------|---------|-------------|
| 74849 | EEG | 00:03:07 | Conv encoder, 20 epochs, TUAB — lightweight |
| 74861 | eeg_eval | 00:01:26 | TUAB linear probe (logreg + MLP) |
| 74896 | EEG | 00:03:33 | Conv encoder variant, 20 epochs, TUAB |
| **74897** | **EEG** | **00:34:26** | **Transformer encoder (labram/eegpt/biot), 20 epochs, TUAB** |
| 74898 | EEG | 00:03:07 | Conv encoder variant, 20 epochs, TUAB |
| 74900 | EEG | 00:03:41 | Conv encoder variant, 20 epochs, TUAB |
| 74985 | evallvl | 00:04:28 | Extended eval (tvasnier) |
| 75030 | corr_big | ~20 min (still running) | Large correlation experiment |
| 75047 | masked_big | ~15 min (still running) | Masked SSL variant |

**Takeaway:** Conv encoder ≈ 3 min/run. Transformer encoder ≈ 34 min/run on full TUAB (20 epochs).  
TUAB has ~2717 train recordings × ~120 windows = ~326K windows at [B=128, 19ch, 2000t].

---

## All 8 encoder × loss combinations (TUAB pretraining estimate)

| Encoder | SSL loss | Estimated time | Based on |
|---------|----------|---------------|----------|
| conv | vicreg | **~3 min** | Job 74849 actual |
| conv | sigreg | **~3 min** | Job 74898 actual |
| labram | vicreg | **~34 min** | Job 74897 actual |
| labram | sigreg | **~34 min** | Extrapolated |
| eegpt | vicreg | **~34 min** | Extrapolated (same arch scale) |
| eegpt | sigreg | **~34 min** | Extrapolated |
| biot | vicreg | **~34 min** | Extrapolated |
| biot | sigreg | **~34 min** | Extrapolated |

**All 8 runs in parallel:** ~34 min wall time (Dalia has 72 GPUs, 8 fits easily)  
**All 8 runs sequential:** ~3 + 3 + 34 + 34 + 34 + 34 + 34 + 34 = **210 min ≈ 3.5 h**

TUAB linear probe per checkpoint: ~1.5 min → 8 evals = **~12 min total**

---

## Outside-competition datasets — linear probe only (frozen TUAB encoder)

We do NOT retrain on these. We freeze the TUAB encoder, extract features, fit a linear probe.  
Reference: TUAB eval took **1.5 min** for 2717+276 recordings (~326K windows).

### BCI IV 2a — motor imagery 4-class

- 9 subjects × 2 sessions × 144 trials = **~5,200 trials** (already epoched, 4s each)
- 22ch, 250Hz → resample to 200Hz, 800 samples/trial
- Windows compared to TUAB eval: ~5,200 / 326,000 ≈ 1.6%
- **Feature extraction: <1 min**
- **Linear probe: <1 min**
- **Total estimate: ~1–2 min** per encoder checkpoint

### HGD — High-Gamma motor imagery 4-class (Schirrmeister)

- 14 subjects × ~1,000 trials = **~14,000 trials** (already epoched, 4s)
- 128ch, 500Hz → resample to 200Hz, crop to 2000 samples
- Windows: 14,000 / 326,000 ≈ 4.3% but 128ch vs 19ch adds encoder overhead (~6.7×)
- **Feature extraction: ~2–3 min**
- **Linear probe: <1 min**
- **Total estimate: ~3–5 min** per encoder checkpoint

### Sleep-EDF — sleep staging 5-class

- 197 recordings × ~900 epochs/rec × 3 windows/epoch = **~531,000 windows**
- 2ch, 100Hz → resample + broadcast → [B, 19, 2000]
- Windows: 531K / 326K ≈ 1.63× TUAB, but 2ch = very cheap encoder forward pass
- **Feature extraction: ~2–3 min**
- **Linear probe: <1 min**
- **Total estimate: ~3–5 min** per encoder checkpoint
- Note: numbers are inflated by spectral shortcut (see spectral_audit.pdf) — report with caveat

### HMC — Haaglanden sleep staging 5-class

- 151 recordings × ~2,700 windows/rec = **~408,000 windows**
- 4ch, 256Hz → resample + broadcast → [B, 19, 2000]
- Windows: 408K / 326K ≈ 1.25× TUAB, 4ch = still cheap
- **Feature extraction: ~2–3 min**
- **Linear probe: <1 min**
- **Total estimate: ~3–5 min** per encoder checkpoint

---

## Full eval sweep estimate

Freeze all 8 TUAB-pretrained checkpoints, run linear probe on each outside dataset:

| Dataset | Time per checkpoint | 8 checkpoints | Notes |
|---------|--------------------|--------------------|-------|
| TUAB (reference) | 1.5 min | **12 min** | Already done |
| BCI IV 2a | ~2 min | **16 min** | Sharp benchmark, no shortcut |
| HGD | ~4 min | **32 min** | Sharp, 128ch overhead |
| Sleep-EDF | ~4 min | **32 min** | Inflated numbers — caveat |
| HMC | ~4 min | **32 min** | Cleaner sleep staging |

**Full outside-competition eval: ~90–120 min total, or ~20 min if run in parallel**  
(8 checkpoints × 4 datasets = 32 probe jobs; Dalia fits all simultaneously)

---

## Practical recommendation

1. Submit all 8 TUAB pretraining runs in parallel → wait 34 min
2. Submit all 32 probe jobs in parallel → wait ~5 min
3. Total wall time from scratch: **~40 min**
