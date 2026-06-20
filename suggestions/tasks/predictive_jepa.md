# Predictive JEPA Objective (actual world model)

## What it is

Replace the current two-view VICReg SSL with a **predictive** objective:
mask a contiguous block of temporal frames, use context frames to PREDICT
the masked embeddings in latent space.

This is the actual JEPA objective (Joint-Embedding Predictive Architecture).
The current SSL is invariance learning — two augmented views, minimize distance.
That is NOT a world model. This is.

## Why it matters

The competition brief uses "world model" and "EB-JEPA" — Energy-Based JEPA.
Our current approach trains a VICReg encoder and calls it JEPA. Technically the
framework is JEPA-compatible, but we never actually predict anything.

Implementing this lets us say: "our encoder predicts future EEG states from context,
learning causal temporal structure — a genuine world model objective."

## Infrastructure already available

`eb_jepa.architectures.RNNPredictor (GRU)` is listed as provided in main.py comments:
```python
# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
```

The GRU predictor takes a context sequence and predicts the next embedding.
`encoder.frames(x)` already returns `[B, F, D]` — the sequence to mask and predict.

## Sketch of the objective

```python
# In EEGSSL.compute_loss():
frames = self.encoder.frames(v1)          # [B, F, D]
ctx_len = F // 2
context = frames[:, :ctx_len, :]          # first half
target  = frames[:, ctx_len:, :].detach()# second half (stop gradient)

pred = self.predictor(context)            # GRU → [B, F//2, D]
loss = F.mse_loss(pred, target)           # predict future frames
# + VICReg variance/covariance on full sequence to prevent collapse
```

## Risk assessment

- Requires full retraining from scratch (34 min per encoder on Dalia)
- New objective may need hyperparameter tuning (mask ratio, predictor depth)
- If representations degrade vs current VICReg, you've lost GPU time
- **Do NOT attempt unless current VICReg BACC is already ≥ 0.78**

## Estimated time: 1–2 days (implement + tune + retrain all encoders)

## Prerequisite: confirm strong baseline first (URGENT_fix_metrics + run eval)
