# Evaluation Protocol Analysis

The per-window vs per-recording issue is the most important methodological finding
of this hackathon. This document explains it fully.

---

## The two evaluation protocols

### Protocol A — Per-window (what the literature uses)
1. For each recording in the eval split: extract ALL non-overlapping 10s windows
2. Each window is one test sample with the label of its recording
3. Run the MLP probe on each window's individual embedding
4. Compute BACC over all individual window predictions

Test set size: depends on recording duration. Literature (BIOT, LaBraM) uses 2339 recordings
→ ~409,455 windows = ~175 windows per recording on average.

### Protocol B — Per-recording (what our framework uses by default)
1. For each recording: extract exactly N=16 evenly-spaced windows
2. Run encoder on each, get 16 embeddings of size 256
3. Mean-pool the 16 embeddings → one 256-dim vector per recording
4. Run the MLP probe on the pooled embedding
5. Compute BACC over 276 patients (one prediction per patient)

Test set size: exactly 276 samples (our eval split has 276 recordings).

---

## Why they differ by ~5pp

The pooling in Protocol B creates a statistically cleaner signal:

**Noise averaging**: Each 10s window has random noise, artifacts, transient events. 16 windows = 160 seconds of EEG. The mean of 16 embeddings averages out per-window noise, keeping the stable, recording-level signal. The probe sees a more reliable input.

**Empirical magnitude**: For the same encoder (corruption variant, SIGReg):
- Per-window BACC: 0.775
- Per-recording BACC: 0.825
- Gap: **0.050** (50 basis points)

This gap is consistent across all our encoder variants:
| Encoder | WIN | REC | Gap |
|---------|-----|-----|-----|
| EB-JEPA base | 0.756 | 0.796 | 0.040 |
| +corruption | 0.770 | 0.825 | 0.055 |
| +spectral 0.1 | 0.765 | 0.836 | 0.071 |
| +SIGReg+corrupt | 0.775 | 0.825 | 0.050 |

The gap tends to be larger when the spectral regularization helps — suggesting spectral features are more consistent across windows (less noisy per-window) than temporal features.

---

## Why Protocol B is clinically correct but wrong for benchmarking

**Clinically**: a neurologist reviewing an EEG recording makes one decision per patient, not one per 10-second segment. Protocol B mimics this. If we were deploying this system in a hospital, we would pool predictions and make one decision per patient. Protocol B is the right metric for deployment evaluation.

**For benchmarking**: all published TUAB SSL papers use Protocol A. To claim "our method is better than LaBraM," we need to use the same protocol. Otherwise we're comparing different quantities.

**The research community's choice** of Protocol A is debatable. It gives higher $n$ (better statistical power), but the test samples are not independent (windows from the same recording share the same label and largely the same signal). Treating 409,455 correlated windows as i.i.d. test samples inflates effective sample size and underestimates variance.

A rigorous benchmark would:
1. Report per-window BACC as the primary benchmark metric (for comparability)
2. Report per-recording BACC separately (for clinical relevance)
3. Compute confidence intervals properly (accounting for the clustering by recording)

We do (1) and (2). We don't do (3) — error bars require multiple eval seeds or bootstrap, which we didn't compute.

---

## How we discovered the mistake

Timeline:
1. **tvasnier reports BACC 0.836** — team celebrates, thinks we beat LaBraM
2. **tvasnier creates baseline branch** with structured comparison tables
3. **Tables explicitly separate "per-window" and "per-recording"** — tvasnier labeled them correctly
4. **We ran per-window eval** on the same checkpoints → 0.765–0.775
5. **Comparison with literature**: 0.775 < LaBraM 0.814
6. **Retraction**: "we beat LaBraM" was incorrect

The lag between steps 1 and 3 was ~2 days. During that period, the incorrect claim circulated internally.

---

## Impact on each claim we made

| Original claim | Correct assessment |
|---------------|-------------------|
| "We beat LaBraM (0.836 > 0.814)" | False — comparing REC to WIN. WIN: 0.775 < 0.814 |
| "We beat BIOT (0.836 > 0.802)" | False — same issue. WIN: 0.775 < 0.802 |
| "We're competitive with ContraWR" | True — WIN 0.775 = ContraWR 0.775 |
| "SSL beats supervised EEGNet" | False — EEGNet WIN 0.796 > ours WIN 0.775 |
| "SSL matches ShallowConvNet supervised" | True — ShallowConvNet WIN 0.777 ≈ ours 0.775 |
| "Corruption augmentation helps" | True at both levels (+0.014 WIN, +0.029 REC) |
| "SIGReg ≥ VICReg" | True at both levels |
| "EEGPT trained below random" | True |
| "VICReg bug caused EEGPT collapse" | True |

---

## The honest narrative for the presentation

"We initially reported BACC 0.836, which seemed to beat LaBraM (0.814). We then realized
our evaluation used per-recording aggregation (pooling 16 windows per patient) while the
literature uses per-window evaluation. These two metrics differ by ~5pp for the same encoder.
Our fair per-window score is 0.775 — below BIOT and LaBraM, but matching ContraWR and
competitive with supervised ShallowConvNet.

Importantly, per-recording is actually the clinically relevant metric — you classify patients,
not windows. Our frozen encoder reaches 0.836 per-recording, competitive with supervised EEGNet
(0.824 per-recording). The real question is which metric you care about."

---

## Appendix: why 16 windows?

The choice of N=16 in the framework was not explicitly explained in the code. Rationale:
- TUAB recordings are 30–120 minutes; 16 windows of 10s = 160s = ~2.5 minutes sampled
- 16 provides sufficient coverage of the recording without being too expensive
- Evenly spaced (`np.linspace`) avoids boundary effects

With N=1 (single window), per-recording → per-window. As N→∞, per-recording approaches the idealized patient-level score. N=16 is a practical tradeoff.

Had we used N=1, per-recording BACC would equal per-window BACC (same protocol). The discrepancy between our results and the literature is entirely due to N=16 > 1.
