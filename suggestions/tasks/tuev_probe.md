# TUEV 6-Class Event Probe

## What it is

Freeze the TUAB-pretrained encoder. Fit a 6-class probe on TUEV EEG event detection:
spike/sharp-wave, GPED, PLED, eye movement, artifact, background.

Labels are at **1-second resolution per channel** — much finer than TUAB's per-recording label.

## Why this is the best world-model argument

6-class event detection at 1s resolution requires distinguishing:
- Spike morphology (fast rise, slow recovery, <200ms) vs
- Slowing (prolonged low-frequency waves, >500ms) vs
- Eye movement artifacts (frontally dominant, stereotyped) vs
- Background EEG (no event)

No spectral shortcut can separate all 6 — the model must understand **temporal morphology**.
This is exactly what `frames()` captures and what global mean-pooling loses.

If our frozen TUAB encoder separates TUEV events, we have proved it learned
temporal structure that transfers to a completely different annotation granularity.

## Data

**Already on Dalia:**
```
/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_RAW_DATA/
```
Read `TUEV_README.md` on Dalia for file format details before starting.

TUEV is mentioned as a TODO in `examples/eeg/cfgs/eval.yaml`:
```yaml
# TUEV idea (TODO, optional): swap to the harder 6-class event variant
```

## What to implement

1. `EEGDataset` subclass for TUEV — load 1s windows with event labels
2. 6-class eval probe (swap binary MLP for multiclass, use macro BACC)
3. Compare frame-level probe (`frames()`) vs recording-level probe (`represent()`)
   — TUEV is where the difference will be biggest

## Estimated time: 1 day (new dataset loader is the main work)
## Estimated runtime on Dalia: ~5 min per checkpoint

## Do this after: URGENT_fix_metrics + bci2a_probe
(Build confidence on simpler cross-task probe first)
