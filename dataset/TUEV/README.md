# TUEV - TUH EEG Events Corpus (v2.0.0)

**6-class event classification**: per labeled second, which of {SPSW, GPED,
PLED, EYEM, ARTF, BCKG} is it?

| label | meaning | typical share |
|---|---|---|
| SPSW | spike-and-slow-wave (epileptiform) | rare |
| GPED | generalized periodic epileptiform discharge | rare |
| PLED | periodic lateralized epileptiform discharge | rare |
| EYEM | eye movement artifact | ~1% |
| ARTF | (non-eye) artifact | ~2% |
| BCKG | background (no event) | **~95%** |

The 95% BCKG imbalance is why every literature paper reports **Balanced
Accuracy + Cohen-κ + Weighted-F1** (no accuracy, no AUROC).

## Source + preprocessing

Same TUH-EEG corpus as TUAB. We use the **v2.0.0** event corpus split:

| split | recordings | event windows (5 s, centered on the labeled second) |
|---|---|---|
| train | (subject-disjoint) | 83,932 |
| eval  | (subject-disjoint) | 28,559 |
| total | | 112,491 (matches BIOT Tbl 1) |

**Preprocessing recipe.** We run **BIOT's official
[`process.py`](process.py)** verbatim on the raw EDFs + `.rec` annotation
files. This script:

1. Reads each raw multi-event EDF + its `.rec` (event timestamps).
2. Bandpass-filters 0.5–60 Hz + 60 Hz notch.
3. Builds a 16-channel **bipolar** montage (TCP).
4. Resamples to 200 Hz.
5. Cuts a 5 s window **centered on each labeled second** in `.rec`.
6. Writes one `.pkl` per event window (`{signal: [16, 1000], label: int}`).

After running `process.py`, point `tuev_probe.py --data-root` at the resulting
pkl directory.

Vendored copy: [`process.py`](process.py) (from
[`ycq091044/BIOT/datasets/TUEV/process.py`](https://github.com/ycq091044/BIOT/blob/main/datasets/TUEV/process.py)).

## How EB-JEPA uses TUEV

**Transfer probe only - we never train EB-JEPA on TUEV.** The recipe in
[`../../TUEV_experiments/`](../../TUEV_experiments/) is:

1. Pretrain a Conv1D EB-JEPA encoder on **TUAB** (the SSL energy is whatever
   the row specifies - VICReg, SIGReg, +spectral, +corrupt).
2. Freeze the encoder.
3. Run [`tuev_probe.py`](../../TUEV_experiments/tuev_probe.py) - a 6-class
   logistic regression on event-centered windows from `process.py`'s output.

This is the [JEPA "one representation, many tasks"](https://arxiv.org/abs/2403.00504)
result: a TUAB-pretrained encoder transfers to a different task (TUEV
6-class) with no TUEV labels seen in SSL.

## Probe protocol (matches `tuev_probe.py`)

- Class-balanced logistic regression (`sklearn`, `class_weight="balanced"`)
- BCKG subsampled so the 6 classes are roughly balanced (else 95% is BCKG
  and the task is trivial)
- Per-channel z-score on each window
- Subject-disjoint train/eval split
- Reported metrics: `balanced_accuracy`, `cohen_kappa`, `weighted_f1`,
  `macro_f1`, `accuracy`, plus class counts

## Supervised baselines we compare against

→ See [`supervised_baselines_provenance.md`](supervised_baselines_provenance.md).
10 rows; the 7 BIOT-bundled supervised models (SPaRCNet / ContraWR / CNN-T /
FFCL / ST-T / BIOT scratch / BIOT pretrained) were **re-run by us** on the
exact same `process.py` pkls used by EB-JEPA's probe; the 3 LaBraM rows are
**cited from LaBraM Tbl 2** (different finetuning code path we did not port).
