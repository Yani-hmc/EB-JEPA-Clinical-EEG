# HMC Sleep Staging Dataset — Haaglanden Medisch Centrum

## What it is

5-class sleep staging: Wake, N1, N2, N3, REM.  
Clinical PSG recordings from Haaglanden Medical Center, Netherlands.

- **Subjects:** 151 patients (mix of healthy and various sleep disorders)
- **Channels:** 4 EEG (F4-M1, C4-M1, O2-M1, C3-M2)
- **Sampling rate:** 256 Hz (often downsampled to 100 Hz in papers)
- **Epoch length:** 30s scored epochs → ~2700 windows per 22.5h recording
- **Total:** ~151 × 2700 ≈ 408,000 windows
- **Access:** Free, PhysioNet, no registration

## Why it's better than Sleep-EDF for this task

Sleep-EDF (Cassette subset) has a known 1/f spectral shortcut:  
the spectral audit paper (arxiv 2606.08583) measured **>0.42 BACC inflation** after
detrending the aperiodic slope. Models look better than they are because sleep staging
cassette EEG has strong low-frequency power differences between stages.

HMC is clinical PSG with a modern amplifier setup — less susceptible to the cassette
recording artifact. Used in the EEG Foundation Models Benchmark paper (arxiv 2601.17883)
as the preferred sleep staging dataset over Sleep-EDF.

## How to download

```bash
wget -r -N -c -np https://physionet.org/files/hmc-sleep-staging/1.1/
```

Or via MNE:
```python
# No direct MNE loader yet — use pyEDFlib to read raw files
import pyedflib
```

## Adapter for our encoder

Our encoder expects `[B, 19, 2000]`. HMC: 4ch, 256Hz, 30s epochs.

```python
import torch
import torch.nn.functional as F

def adapt_hmc(x):
    # x: [B, 4, 7680]  (4ch, 256Hz, 30s)
    # resample 256Hz → 200Hz: 7680 → 6000 samples
    x = F.interpolate(x, size=6000, mode='linear', align_corners=False)
    # crop to 10s: 6000 → 2000
    x = x[:, :, :2000]
    # broadcast 4ch → 19ch
    x = x.repeat(1, 5, 1)[:, :19, :]
    return x  # [B, 19, 2000]
```

## Estimated runtime on Dalia

- ~408,000 windows, 4ch (very cheap encoder forward pass)
- Feature extraction: ~2–3 min
- Linear probe: < 1 min
- **Total: ~3–5 min** per encoder checkpoint
