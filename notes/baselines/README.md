# Baselines

Supervised and classical baselines we compare EB-JEPA against on TUAB (TUH Abnormal EEG Corpus).

The comparison: these methods train **with labels**. EB-JEPA pre-trains **without labels** then fits a linear probe.

| Baseline | TUAB metric | Folder |
|----------|------------|--------|
| Deep4Net (Schirrmeister 2017) | 85.4% accuracy | `deep4net/` |

## Protocol

- Dataset: TUAB, 19 channels, 200 Hz, 10s windows → `[B, 19, 2000]`
- Splits: patient-disjoint train/eval (shipped with TUAB directory structure)
- Metric: **BACC** (balanced accuracy) for EB-JEPA comparisons; accuracy for citing published numbers
