# ⚠️ URGENT — Fine-Tuning Results Are Not Comparable to Frozen Probes

## The problem

Two different evaluation regimes are being run and their numbers look similar,
but they CANNOT be compared:

| Regime | n_train | n_eval | Valid comparison? |
|--------|---------|--------|-------------------|
| Frozen probe (SSL zero-label) | 2717 | 276 | ✓ Comparable to LaBraM, BIOT |
| Fine-tuning [ft] (ft_corrupt) | 276 | 276 | ✗ Different setup entirely |

The `[ft]` runs (`ft_corrupt` checkpoint) show n_train=276 and n_eval=276.
This means the model is being fine-tuned on the 276-recording eval split and
evaluated on the same 276 recordings. **This is likely data leakage or at
minimum a non-standard evaluation** — it cannot be reported alongside frozen
probe numbers.

## What we think is happening

`ft_corrupt` = the encoder is fine-tuned (weights updated) using only the eval
patients as supervision. This is a "few-shot fine-tuning" scenario:

- Good for: showing adaptation ability with few labeled samples
- Not comparable to: LaBraM/BIOT frozen probe on 2717 train patients
- Possibly invalid if: train and eval patients overlap (same 276 used for both)

## Action needed

**tvasnier**: clarify the ft_corrupt eval setup:
1. Is the fine-tuning training set the same 276 recordings as the eval set? If yes → invalid.
2. Or is it some other split? Document it explicitly.
3. If valid, report separately as "few-shot fine-tuning" — not as main SSL result.

## What numbers to report as main result

Only frozen probe results with n_train=2717, n_eval=276 (patient-disjoint):
- Best so far: **BACC 0.836** (recording logreg) — from tvasnier's ablations
- This beats LaBraM (0.814) if confirmed with random floor and multiple seeds
