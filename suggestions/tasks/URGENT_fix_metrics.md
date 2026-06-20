# ⚠️ URGENT — Add BACC + AUROC to eval.py

## Why this is blocking

Current `eval.py` only reports accuracy / f1 / recall / precision.
Every published baseline uses **balanced accuracy (BACC)** and **AUROC** because
TUAB is class-imbalanced (~54% normal). Without these two numbers our results
cannot be compared to:
- LaBraM: 0.814 BACC
- BIOT: 0.796 BACC
- Deep4Net: 85.4% accuracy (but also reports BACC)

tvasnier already has a patched version that computes them — check his eval logs.

## Where to edit

`examples/eeg/eval.py`, function `probe()`, return dict.

## Exact change (~3 lines)

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)

def probe(Xtr, ytr, Xev, yev):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xev_s = scaler.transform(Xtr), scaler.transform(Xev)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                         max_iter=500, random_state=0)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xev_s)
    prob = clf.predict_proba(Xev_s)[:, 1]
    return {
        "accuracy":          accuracy_score(yev, pred),
        "balanced_accuracy": balanced_accuracy_score(yev, pred),   # ← ADD
        "auroc":             roc_auc_score(yev, prob),             # ← ADD
        "f1":                f1_score(yev, pred, pos_label=1, zero_division=0),
        "recall":            recall_score(yev, pred, pos_label=1, zero_division=0),
        "precision":         precision_score(yev, pred, pos_label=1, zero_division=0),
    }
```

## Also: enable random floor by default

Add `--random-floor` to the sbatch eval script so every run automatically reports
the untrained encoder baseline. Without it we can't show SSL did anything over random.

## Estimated time: 15 minutes
