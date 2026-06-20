# Temporal Attention Probe (use frames() not represent())

## What it is

Replace the global mean-pool in the probe with a **learned attention head** over the
125 temporal frames the encoder already produces.

```
encoder.frames(x) → [B, 125, D]
                           ↓
              learned attention weights (1-layer)
                           ↓
              weighted sum → [B, D] → MLP probe
```

If certain time windows are consistently upweighted for abnormal recordings, you can
visualize WHERE in the recording the pathology lives. This is direct evidence of
temporal structure, not just a global score.

## Why it supports the world-model claim

Mean pooling (`represent()`) is equivalent to treating the 125 frames as an unordered
bag. If the attention probe scores better, temporal structure is load-bearing for the
prediction — the encoder didn't just learn a global spectral fingerprint.

Visualization bonus: plot the attention weights for a normal vs abnormal recording.
A spike in the attention map at the pathological segment is a killer presentation slide.

## Where to edit

`examples/eeg/eval.py` — add an AttentionProbe class, call `encoder.frames()` instead
of `encoder.represent()`.

## Sketch

```python
import torch.nn as nn

class AttentionProbe(nn.Module):
    def __init__(self, d, n_classes=2):
        super().__init__()
        self.attn = nn.Linear(d, 1)          # frame → scalar score
        self.clf  = nn.Linear(d, n_classes)

    def forward(self, frames):               # [B, F, D]
        w = torch.softmax(self.attn(frames), dim=1)   # [B, F, 1]
        z = (w * frames).sum(dim=1)                   # [B, D]
        return self.clf(z), w.squeeze(-1)             # logits + attention weights

# In extract_features: call encoder.frames(flat) instead of encoder.represent(flat)
# Train AttentionProbe with cross-entropy (10 epochs, frozen encoder)
# Save attention weights → plot per recording
```

Note: this probe needs gradient-based training (not sklearn), so it's slightly more
involved than the current MLP probe. Worth it for the visualization.

## Estimated time: 2–3 hours
