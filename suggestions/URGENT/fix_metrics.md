# ⚠️ URGENT — eval.py in Repo Missing BACC and AUROC

## The problem

The `examples/eeg/eval.py` currently in the repo only reports:
`accuracy / f1 / recall / precision`

Every published baseline uses **balanced_accuracy (BACC)** and **AUROC**.
Without these our results cannot be compared to:
- LaBraM: 0.814 BACC
- BIOT: 0.796 BACC  
- Deep4Net: reported as accuracy (85.4%) but BACC is standard

tvasnier has a patched version on Dalia that already computes BACC and AUROC.
**That patched version needs to be committed back to the repo.**

## Current best numbers (from tvasnier's patched eval, patient-disjoint)

| Checkpoint | Probe | BACC | AUROC |
|------------|-------|------|-------|
| fftc0p3 | MLP recording | **0.826** | 0.896 |
| fftc0p3 | logreg recording | 0.822 | 0.891 |
| best run (tag unknown) | logreg recording | **0.836** | 0.913 |
| spec1p0 | MLP recording | 0.818 | 0.881 |

vs baselines:
- BIOT SSL frozen: 0.796 BACC
- LaBraM SSL frozen: 0.814 BACC
- **Our best: 0.836 BACC** ← beats LaBraM if confirmed

## Action needed

1. **tvasnier**: commit the patched eval.py to the repo (or open a PR)
2. Add `--random-floor` to every eval sbatch to prove SSL > random init
3. Re-run eval on all saved checkpoints with the patched version to get a complete table

## Exact fix (if patching manually)

In `examples/eeg/eval.py`, `probe()` function:

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)

# In probe(), after clf.fit():
pred = clf.predict(Xev_s)
prob = clf.predict_proba(Xev_s)[:, 1]
return {
    "accuracy":          accuracy_score(yev, pred),
    "balanced_accuracy": balanced_accuracy_score(yev, pred),
    "auroc":             roc_auc_score(yev, prob),
    "f1":                f1_score(yev, pred, pos_label=1, zero_division=0),
    "recall":            recall_score(yev, pred, pos_label=1, zero_division=0),
    "precision":         precision_score(yev, pred, pos_label=1, zero_division=0),
}
```
