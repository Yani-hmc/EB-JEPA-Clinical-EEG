# Deep4Net — Baseline Notes

**Paper:** Schirrmeister et al. 2017 — *Deep learning with convolutional neural networks for EEG decoding and visualization*
**DOI:** 10.1002/hbm.23730 — arxiv 1708.08012
**PDF:** `schirrmeister2017.pdf` (this folder)
**Code:** https://github.com/braindecode/braindecode (same author, `braindecode/models/deep4.py`)

---

## What it is

4-block deep CNN for EEG classification. Supervised — trains directly on labels.

- Block 1: temporal conv → spatial conv (EEG-specific: separate time and channel mixing)
- Blocks 2–4: conv + batchnorm + ELU + maxpool, filters 25→50→100→200
- Final: conv classifier → n_outputs logits

## TUAB result

**85.4% accuracy** on TUH Abnormal corpus, patient-disjoint splits.
This is the supervised ceiling we compare EB-JEPA against.

---

## Adaptation to our data format

**Input:** `[B, 19, 2000]` — works directly, no reshaping.

```python
from braindecode.models import Deep4Net

model = Deep4Net(n_chans=19, n_outputs=2, n_times=2000)
# forward: model(x)  where x is [B, 19, 2000]
```

**One required step:** raw TUAB files are at 250 Hz, EB-JEPA uses 200 Hz.
Resample before windowing:

```python
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import Preprocessor, preprocess

train_set = TUHAbnormal(path="/path/to/TUAB/train", target_name="pathological", preload=False)
eval_set  = TUHAbnormal(path="/path/to/TUAB/eval",  target_name="pathological", preload=False)
preprocess(train_set, [Preprocessor('resample', sfreq=200)])
preprocess(eval_set,  [Preprocessor('resample', sfreq=200)])
```

Then window to 10s (2000 samples), build DataLoader, train with CrossEntropyLoss.

---

## Why this is the right baseline

- Supervised: sees labels during training — EB-JEPA does not
- EEG-native architecture: not a trivial comparison
- Published number on same dataset, same split → directly citeable
- If EB-JEPA frozen probe ≥ 85.4%: world model matches supervised CNN without labels
