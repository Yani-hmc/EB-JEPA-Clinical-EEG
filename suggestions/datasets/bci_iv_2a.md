# BCI Competition IV Dataset 2a — Motor Imagery

## What it is

4-class motor imagery EEG: left hand, right hand, feet, tongue.  
9 subjects × 2 sessions (training + evaluation), ~288 trials per session.

- **Channels:** 22 EEG + 3 EOG → use 22 EEG channels
- **Sampling rate:** 250 Hz
- **Trial length:** 4s (cue on at 2s, end at 6s) → 1000 samples at 250Hz
- **Total:** ~5,184 labeled trials across all subjects

## Why it's the sharpest outside-competition benchmark

Confirmed by spectral audit paper (arxiv 2606.08583): motor imagery EEG shows **zero
spectral shortcut inflation**. Unlike TUAB (0.07–0.13 BACC inflation) and Sleep-EDF
(0.42+ inflation), BCI IV 2a results are genuine — a good BACC here means the encoder
actually learned signal structure, not 1/f frequency bias.

Every SSL-on-EEG paper (LaBraM, BIOT, EEGPT) reports BCI IV 2a. Judges will immediately
recognize the comparison.

## How to download

```bash
# Via MOABB (recommended — no manual registration)
pip install moabb
python -c "
from moabb.datasets import BNCI2014_001
ds = BNCI2014_001()
ds.download()
"

# Via braindecode
from braindecode.datasets import MOABBDataset
ds = MOABBDataset(dataset_name='BNCI2014_001', subject_ids=list(range(1, 10)))
```

## Adapter for our encoder

Our encoder expects `[B, 19, 2000]` (19ch, 200Hz, 10s). BCI IV 2a: 22ch, 250Hz, 4s.

```python
import torch
import torch.nn.functional as F

def adapt_bci2a(x):
    # x: [B, 22, 1000]  (22ch, 250Hz, 4s)
    # resample 250Hz → 200Hz: 1000 → 800 samples
    x = F.interpolate(x, size=800, mode='linear', align_corners=False)
    # pad time 800 → 2000 (zero-pad to 10s)
    x = F.pad(x, (0, 1200))
    # select 19 of 22 channels (drop last 3)
    x = x[:, :19, :]
    return x  # [B, 19, 2000]
```

## Published SSL results (linear probe, for comparison)

| Model | BCI IV 2a accuracy |
|-------|-------------------|
| EEGNet (supervised) | ~72% |
| BENDR | ~68% |
| BIOT | ~74% |
| LaBraM-Base | ~76% |
| **EB-JEPA (target)** | ? |

## Estimated runtime on Dalia

- Feature extraction (frozen TUAB encoder): < 1 min
- Linear probe: < 1 min
- **Total: ~1–2 min** per encoder checkpoint
