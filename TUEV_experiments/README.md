# TUEV Experiments - EB-JEPA reference table reproduction

This directory contains **one shell script per row** of our TUEV EB-JEPA
[`reference_table_JEPA_TUEV_V1.md`](reference_table_JEPA_TUEV_V1.md).

Every row is a **transfer probe** - a Conv1D EB-JEPA encoder is **pretrained
on TUAB** (no TUEV labels seen in SSL), then **frozen** and probed on TUEV's
6 event classes with a class-balanced logistic regression.

| Script | Row | SSL energy (on TUAB) | Expected TUEV BAcc |
|---|---|---|---|
| [`run_baseline.sh`](run_baseline.sh) | baseline | VICReg, base aug, no corruption | 0.381 |
| [`run_row1_spec_corrupt.sh`](run_row1_spec_corrupt.sh) | 1 | VICReg + spectral 0.1 + corrupt (inv=1) | **0.425** ⭐ best |
| [`run_row2_vicreg_corrupt.sh`](run_row2_vicreg_corrupt.sh) | 2 | VICReg + corrupt (inv=1, seed 0) | 0.382 |
| [`run_row3_sigreg_corrupt.sh`](run_row3_sigreg_corrupt.sh) | 3 | SIGReg + corrupt | 0.363 |

Reference floor: **random encoder = 0.337**; chance = 0.167.

## How each script works

Two stages:

```bash
# (1) train SSL encoder on TUAB
python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    <row-specific overrides>

# (2) frozen 6-class probe on TUEV
python TUEV_experiments/tuev_probe.py --ckpt "$CKPT_DIR/latest.pth.tar"
```

## The TUEV probe

[`tuev_probe.py`](tuev_probe.py) reads a TUAB-pretrained EB-JEPA checkpoint,
loads BIOT-`process.py`-preprocessed TUEV pkls (one event-window per file),
extracts frozen features, and fits a class-balanced multinomial logistic
regression on the 6 event classes (BCKG subsampled to balance the dataset
- else 95% is BCKG and the task is trivial). Reports the standard literature
metrics: **balanced_accuracy / cohen_kappa / weighted_f1** + a few extras.

### CLI

```bash
python TUEV_experiments/tuev_probe.py \
    --ckpt /path/to/latest.pth.tar \
    [--data-root /path/to/TUEV_PREP200] \
    [--window-sec 5] \
    [--max-files N]      # debug: limit files
    [--random]           # use UNTRAINED encoder (random-feature floor)
```

For the **random-encoder floor** (the 0.337 reference), pass `--random`.

## Headline reading

**Row 1 (0.425 BAcc) beats our supervised CNN-Transformer (0.410) and
ST-Transformer (0.339) without ever seeing a TUEV label** - the JEPA
"one representation, many tasks" result.

→ For the literature/supervised numbers, see
[`../dataset/TUEV/supervised_baselines_provenance.md`](../dataset/TUEV/supervised_baselines_provenance.md).
