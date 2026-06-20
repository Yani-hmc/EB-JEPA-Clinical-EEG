# Current Results Summary (as of 2026-06-20)

## Saved checkpoints

### yhammache — 4 encoders × 2 SSL losses (NOT YET EVALUATED)
```
conv   × vicreg   ← 3 min to train
conv   × sigreg   ← 3 min to train
labram × vicreg   ← 34 min to train
labram × sigreg   ← 34 min to train
biot   × vicreg   ← 34 min to train
biot   × sigreg   ← 34 min to train
eegpt  × vicreg   ← 34 min to train
eegpt  × sigreg   ← 34 min to train
```
**No eval numbers exist for any of these. Run eval immediately.**

### tvasnier — 17 ablation experiments (partially evaluated)

| Checkpoint | Best BACC | AUROC | Notes |
|------------|-----------|-------|-------|
| `fftc0p3` | 0.826 (MLP) | 0.896 | FFT consistency coeff=0.3 |
| `spec1p0` | 0.818 (MLP) | 0.881 | Spectral reg coeff=1.0 |
| `fftc0p1` | ~0.817 | ~0.890 | FFT consistency coeff=0.1 |
| `fftc0p05` | ~0.810 | ~0.887 | FFT consistency coeff=0.05 |
| `corrupt` | ~0.820 | ~0.885 | Corruption augmentation |
| `masked_big` | unknown | unknown | Masked SSL — not yet evaled |
| `multicorpus` | unknown | unknown | Multi-corpus — not yet evaled |
| `base` | ~0.800 | ~0.870 | Baseline VICReg |
| best seen | **0.836** | **0.913** | Tag unknown — which checkpoint? |

### tvasnier — fine-tuning (separate regime, NOT comparable to frozen probe)
```
ft_corrupt: BACC ~0.837, n_train=276, n_eval=276  ← VERIFY SETUP BEFORE REPORTING
```

## vs published baselines (frozen probe, patient-disjoint)

| Model | BACC | AUROC |
|-------|------|-------|
| Random init (floor) | ~0.52 | ~0.50 |
| BIOT | 0.796 | — |
| LaBraM | 0.814 | — |
| **Our best (tvasnier fftc0p3)** | **0.826** | **0.896** |
| **Our best (unconfirmed)** | **0.836** | **0.913** |
| Deep4Net (supervised) | ~0.854 | — |

## What's missing before we can claim anything

1. Random floor BACC (never measured — run `--random-floor`)
2. Eval on yhammache's 8 encoder checkpoints
3. Eval on tvasnier's masked_big and multicorpus
4. Identify which checkpoint produced the 0.836 BACC
5. Clarify ft_corrupt evaluation validity
