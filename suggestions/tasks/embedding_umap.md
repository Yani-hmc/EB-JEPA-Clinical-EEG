# UMAP Embedding Visualization

## What it is

Project all recording embeddings to 2D (UMAP or t-SNE) and color by:
1. **normal vs abnormal** — should cluster cleanly if encoder learned pathology
2. **patient ID** — should NOT cluster if encoder learned brain state, not identity

## Why it supports the world-model claim

If the UMAP shows tight normal/abnormal clusters but scattered patient IDs,
the encoder learned the *content* of the EEG (brain state), not *who* produced it.
This is the transferability argument made visual — and it's the kind of figure
that wins presentations.

If patient ID clusters dominate, the encoder learned subject identity (a known failure
mode for EEG SSL) — this is important to catch before claiming generalization.

## Where to run

After `extract_features()` returns `(X, y)` — X is already `[N_recordings, D]`.

## Sketch (~20 lines)

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# X: [N, D], y: [N] (0=normal, 1=abnormal), subj: [N] (patient ID)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
emb = reducer.fit_transform(X)    # [N, 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Color by label
sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=y, cmap='coolwarm', s=8, alpha=0.6)
axes[0].set_title("Colored by normal/abnormal"); plt.colorbar(sc, ax=axes[0])
# Color by patient ID
sc2 = axes[1].scatter(emb[:, 0], emb[:, 1], c=subj % 20, cmap='tab20', s=8, alpha=0.6)
axes[1].set_title("Colored by patient ID")
plt.savefig("umap_embeddings.png", dpi=150)
```

Requires `pip install umap-learn`. Patient IDs can be extracted from the TUAB filename
convention (each file encodes the subject ID).

## Estimated time: 1 hour
