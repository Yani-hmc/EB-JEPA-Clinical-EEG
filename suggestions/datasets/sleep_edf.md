# Sleep-EDF — Outside-Competition Benchmark

## What it is

**Sleep-EDF Expanded** (Physionet, Kemp et al. 2000) — 197 whole-night PSG recordings
from 78 healthy subjects. Gold-standard sleep staging benchmark: 5 classes
(Wake / N1 / N2 / N3 / REM) from 30-second epochs.

- **Channels:** 2 EEG (Fpz-Cz, Pz-Oz) + EOG + EMG — we use the 2 EEG channels
- **Sampling rate:** 100 Hz (resample to 200 Hz to match our encoder)
- **Epochs:** 30s × 2ch = [B, 2, 3000] (or crop/pad to match our window)
- **Size:** ~3500 hours total, ~25h usable EEG per subject

## Why it's the right outside-competition dataset

1. **Zero barrier** — direct wget from PhysioNet, no account needed
2. **Different task** — 5-class staging vs 2-class pathology. Generalisation is real
3. **Universal benchmark** — LaBraM, BIOT, BENDR, EEGPT all report Sleep-EDF numbers.
   Judges will immediately recognise the comparison
4. **Fast to run** — 197 recordings, linear probe fits in minutes on CPU
5. **Already in braindecode** — `braindecode.datasets.SleepPhysionet`

## How to download

```bash
# Option A — MNE (no registration, downloads ~20 subjects for a quick test)
python -c "
from mne.datasets.sleep_physionet.age import fetch_data
fetch_data(subjects=range(20), recording=[1])
"

# Option B — full dataset via wget
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

## How to adapt our encoder

Our encoder expects `[B, 19, 2000]` (19ch, 200Hz, 10s). Sleep-EDF has 2ch at 100Hz, 30s epochs.
Two options:

**Option 1 — retrain encoder on Sleep-EDF** (not ideal — defeats the point of SSL generalisation)

**Option 2 — channel broadcast + resample** (correct approach):
```python
import torch
import torch.nn.functional as F

def adapt_sleep_edf(x):
    # x: [B, 2, 3000]  (2ch, 100Hz, 30s)
    # resample 100Hz → 200Hz: 3000 → 6000 samples
    x = F.interpolate(x, size=6000, mode='linear', align_corners=False)
    # crop to 10s window: 6000 → 2000
    x = x[:, :, :2000]
    # broadcast 2ch → 19ch (repeat channels)
    x = x.repeat(1, 10, 1)[:, :19, :]   # [B, 19, 2000]
    return x
```

Then feed through frozen encoder → linear probe → 5-class accuracy.

## Published SSL results on Sleep-EDF (linear probe, for comparison)

| Model | Sleep-EDF accuracy |
|-------|-------------------|
| BENDR | ~78% |
| BIOT | ~82% |
| LaBraM-Base | ~83% |
| **EB-JEPA (target)** | ? |
