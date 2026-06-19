# Suggestions — Datasets & Papers

External references that strengthen our submission beyond the competition scope.

## Why this folder exists

The competition evaluates EB-JEPA on TUAB (abnormality detection). Showing that our
encoder generalises to a **second task on a second dataset** — without retraining —
is a major scoring plus. It proves we learned structure, not just TUAB-specific features.

## Contents

```
suggestions/
  datasets/
    README.md       ← golden dataset table + download instructions
    sleep_edf.md    ← Sleep-EDF deep-dive (recommended outside-competition dataset)
  papers/
    README.md       ← why each paper matters to our work
    labram.pdf      ← LaBraM (ICLR 2024) — masked SSL on EEG, TUAB BACC 0.814
    biot.pdf        ← BIOT (NeurIPS 2023) — frequency tokenization SSL
    bendr.pdf       ← BENDR (2021) — contrastive SSL, collapses on TUAB (cautionary)
    eegpt.pdf       ← EEGPT (NeurIPS 2024) — hierarchical spatio-temporal transformer
```
