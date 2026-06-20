# Recording-Length Robustness Curve

## What it is

Vary how many windows are used per recording (1, 2, 4, 8, 16) and plot probe BACC
vs number of windows. A world model that captured temporal structure should converge
faster — fewer windows needed to reach peak BACC — than a model that only learned
spectral statistics.

This produces a **figure**, not just a number. Strong presentation material.

## Why it supports the world-model claim

A bag-of-frequency-features model needs many windows to average out noise.
A model that learned temporal dynamics should recognize pathology from a single window
because it encoded what normal vs abnormal EEG *evolves like*, not just what it looks like
on average. Fast convergence = temporal structure learned.

## Where to edit

`examples/eeg/eval.py` — add a loop around `extract_features` with varying `n_windows`.

## Sketch (~25 lines added to main())

```python
import matplotlib.pyplot as plt

n_windows_list = [1, 2, 4, 8, 16]
baccs = []
for nw in n_windows_list:
    # extract_features already does mean-pooling over N windows per recording
    # add n_windows param to EEGConfig or slice the loader
    Xtr, ytr = extract_features(encoder, "train", device, n_windows=nw)
    Xev, yev = extract_features(encoder, "eval",  device, n_windows=nw)
    m = probe(Xtr, ytr, Xev, yev)
    baccs.append(m["balanced_accuracy"])
    print(f"n_windows={nw:2d}  BACC={m['balanced_accuracy']:.4f}")

plt.plot(n_windows_list, baccs, marker='o')
plt.xlabel("Windows per recording"); plt.ylabel("BACC")
plt.title("Probe BACC vs recording length used"); plt.savefig("robustness_curve.png")
```

Also run the same curve on a random-init encoder as the floor.

## Estimated time: 1–2 hours (including EEGConfig plumbing)
