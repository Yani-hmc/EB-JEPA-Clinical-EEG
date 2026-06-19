# Golden EEG Datasets

Datasets that appear in virtually every EEG SSL / foundation model paper.
Used as the universal benchmark suite across LaBraM, BIOT, BENDR, EEGPT.

## Dataset table

| Dataset | Task | Subjects / Size | Ch / sfreq | Papers | Access |
|---------|------|-----------------|------------|--------|--------|
| **TUAB** | Normal/Abnormal (2-class) | ~3000 rec, ~1500h | 19ch / 250Hz | All | TUH account (free) |
| **Sleep-EDF** | Sleep staging (5-class) | 197 rec, ~3500h | 2ch / 100Hz | All | **Free, no registration** ← |
| **BCI IV 2a** | Motor imagery (4-class) | 9 subjects | 22ch / 250Hz | BENDR, EEGPT, LaBraM | Free |
| **SEED** | Emotion (3-class) | 15 subjects | 62ch / 200Hz | BIOT, LaBraM | SJTU registration |
| **TUSZ** | Seizure detection (2-class) | ~675 subjects | 19ch / 250Hz | BIOT, LaBraM | TUH account (free) |

## Discriminative dataset ranking

Source: spectral audit paper (arxiv 2606.08583) — measures how much model BACC drops
after removing 1/f spectral slope artifacts (the main shortcut in EEG deep learning).

| Dataset | Shortcut inflation | Discriminative? | Verdict |
|---------|-------------------|----------------|---------|
| **BCI IV 2a** (motor imagery) | None (confirmed) | ✓✓ High | **Best second benchmark** |
| **TUAB** | 0.07–0.13 BACC | ✓ Moderate | Good — already running |
| **Sleep-EDF** | 0.42+ BACC | ✗ Inflated | Universally used but numbers mislead |
| **CHB-MIT** (seizure) | N/A | ✗ Too easy | 99%+ across all models, skip |

## Recommendation: BCI IV 2a as second benchmark

Motor imagery has no spectral shortcut — a good result here genuinely proves the
encoder learned structure, not frequency bias. Every SSL paper reports it, judges
know it, and model spread is real.

**One-line pitch:** our frozen EB-JEPA encoder, pre-trained on TUAB without labels,
reaches competitive linear-probe accuracy on BCI IV 2a motor imagery — a completely
different task — without fine-tuning. That is what a world model representation looks like.

**Sleep-EDF** is still worth running for breadth (universally cited), but its numbers
are inflated by spectral shortcuts — caveat this when reporting.
