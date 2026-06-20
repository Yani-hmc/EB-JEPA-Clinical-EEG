# BCI IV 2a Cross-Task Probe

## What it is

Freeze the TUAB-pretrained encoder. Fit a 4-class linear probe on BCI IV 2a
motor imagery (left hand / right hand / feet / rest) with no fine-tuning.

## Why this is the strongest outside-competition claim

- **Data already on Dalia:** `/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/BCICIV_2a_RAW_DATA`
- **Zero spectral shortcut** — motor imagery is confirmed unaffected by 1/f detrending
  (spectral audit paper, arxiv 2606.08583). A good result here is genuine.
- Every SSL-on-EEG paper reports it. Judges will immediately recognize the comparison.
- Different task, different domain — proves generalization, not TUAB overfitting.

**Pitch:** "our encoder, trained with no labels on pathology detection, separates
4-class motor imagery without fine-tuning — a task with no spectral shortcut."

## Published numbers (linear probe, for comparison)

| Model | BCI IV 2a accuracy |
|-------|-------------------|
| EEGNet (supervised) | ~72% |
| BENDR | ~68% |
| BIOT | ~74% |
| LaBraM-Base | ~76% |
| **EB-JEPA (target)** | ? |

## What to implement

1. Dataset adapter (resample + channel selection + zero-pad):

```python
import torch.nn.functional as F

def adapt_bci2a(x):
    # x: [B, 22, 1000]  (22ch, 250Hz, 4s)
    x = F.interpolate(x, size=800, mode='linear', align_corners=False)  # → 200Hz
    x = F.pad(x, (0, 1200))   # pad to 10s = 2000 samples
    x = x[:, :19, :]          # drop 3 channels to match encoder input
    return x                   # [B, 19, 2000]
```

2. Load BCICIV_2a from Dalia path — check the README there for file format.
   braindecode can read GDF files: `braindecode.preprocessing.preprocess`.

3. Re-use existing `probe()` from eval.py — change `pos_label=1` to multiclass
   (use `average='macro'` in f1_score, drop auroc or use OvR).

## Estimated time: 1–2 hours (adapter + loader plumbing)
## Estimated runtime on Dalia: ~2 min per checkpoint
