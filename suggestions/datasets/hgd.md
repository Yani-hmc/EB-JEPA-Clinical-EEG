# High-Gamma Dataset (HGD) — Schirrmeister et al. 2017

## What it is

4-class motor imagery / execution: left hand, right hand, rest, feet.  
Same first author as the Deep4Net paper on our git — this is the dataset that team
validated Deep4Net on alongside TUAB.

- **Subjects:** 14 healthy subjects
- **Channels:** 128 EEG (high-density cap) + 2 EOG + 1 EMG → use 128 EEG
- **Sampling rate:** 500 Hz
- **Trials:** ~1000 per subject → ~14,000 total, already epoched
- **Trial length:** variable, cropped to 4s = 2000 samples at 500Hz

## Why it matters

1. **Same authors as our baseline** — Schirrmeister validated Deep4Net here (88–92% accuracy).
   Model spread across architectures is documented and real.
2. **128 channels** — high spatial resolution. Encoders that understand channel topology
   (like LaBraM and EEGPT) should advantage here vs flatter architectures.
3. **No spectral shortcut** — motor imagery is not confounded by 1/f aperiodic slope.
4. **Available directly in braindecode** — no registration, auto-download.

## How to download

```bash
# Via braindecode (recommended)
from braindecode.datasets import HGD
ds = HGD(subject_id=1)   # subject_id 1–14

# Or MOABB
from moabb.datasets import Schirrmeister2017
ds = Schirrmeister2017()
ds.download()
```

## Adapter for our encoder

Our encoder expects `[B, 19, 2000]`. HGD: 128ch, 500Hz, 4s.

```python
import torch
import torch.nn.functional as F

def adapt_hgd(x):
    # x: [B, 128, 2000]  (128ch, 500Hz, 4s)
    # resample 500Hz → 200Hz: 2000 → 800 samples
    x = F.interpolate(x, size=800, mode='linear', align_corners=False)
    # select 19 channels (standard 10-20 subset from 128-ch cap)
    idx = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108]
    x = x[:, idx, :]
    # pad time 800 → 2000
    x = F.pad(x, (0, 1200))
    return x  # [B, 19, 2000]
```

Note: the channel index selection above is approximate. A proper mapping from HGD's
128-ch layout to standard 10-20 positions (Fp1, Fp2, F7, F3, Fz, F4, F8...) should
be done using MNE's channel position lookup.

## Estimated runtime on Dalia

- 14,000 trials × 128ch: heavier encoder forward pass than BCI IV 2a
- Feature extraction: ~2–3 min
- Linear probe: < 1 min
- **Total: ~3–5 min** per encoder checkpoint
