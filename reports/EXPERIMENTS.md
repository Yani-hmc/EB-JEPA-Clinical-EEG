# Experiment Log

Every training run we know about, with outcomes and lessons.
Numbers are recording-level BACC unless otherwise noted.

---

## Notation
- **Probe**: MLP probe (2 hidden layers) on frozen encoder
- **FT**: fine-tuned encoder + probe
- **REC**: per-recording (mean-pool 16 windows)
- **WIN**: per-window (each window is a test point, fair to literature)
- Jobs on Dalia cluster (Slurm); logged to `/lustre/work/.../tcourtois/train_logs/`

---

## tvasnier's conv encoder ablations

All runs: conv encoder (0.4M), 20 epochs, VICReg loss (with OLD bug: inv_coeff=1)

### Base configuration
| Metric | Value |
|--------|-------|
| BACC (REC) | 0.796 |
| AUROC (REC) | 0.888 |
| Notes | Reference baseline, no augmentation extras |

### Spectral auxiliary loss variants
The spectral loss adds a DDSP-inspired auxiliary term penalizing the difference between the predicted and target power spectral density.

| Run name | spectral_coeff | BACC (REC) | AUROC (REC) | Notes |
|----------|---------------|-----------|------------|-------|
| spec (0.05) | 0.05 | 0.821 | 0.884 | Good improvement |
| spec (0.1) = spec_fixed | 0.1 | **0.836** | 0.887 | **Best of all runs; bugfix in freq normalization** |
| spec (0.3) | 0.3 | 0.789 | 0.877 | Spectral loss too strong, hurts |
| spec (1.0) | 1.0 | 0.785 | 0.874 | Same |
| fftc (0.05) | 0.05 | 0.807 | 0.887 | Different formulation of spectral loss |
| fftc (0.1) | 0.1 | 0.798 | 0.881 | Slight regression vs base |
| fftc (0.3) | 0.3 | 0.802 | 0.889 | OK |

**Key lesson**: spectral coeff = 0.1 is the sweet spot; higher values dominate the gradient and hurt. The `spec_fixed` checkpoint was the single best result of the entire hackathon (per-recording), despite using the VICReg bug (inv_coeff=1). The spectral regularization apparently compensated for the underweighted invariance by providing an alternative gradient signal that shaped the frequency features of the representation.

### Corruption augmentation variants
| Run name | Config | BACC (REC) | AUROC (REC) | Notes |
|----------|--------|-----------|------------|-------|
| corrupt (s0) | time_mask + spikes | 0.825 | 0.904 | Seed 0 |
| corrupt_s1 | same, seed 1 | 0.816 | 0.891 | Seed 1 |
| corrupt_s2 | same, seed 2 | 0.818 | 0.906 | Seed 2 |
| corrupt_big | larger encoder | 0.805 | 0.892 | More params didn't help |
| corrupt_spec | corrupt + spectral 0.1 | 0.802 | 0.899 | Combination WORSE than each alone |
| corrupt_sigreg | corrupt + SIGReg | 0.825 | 0.904 | Same as vicreg+corrupt |

**3-seed average** (corrupt, corrupt_s1, corrupt_s2): BACC 0.819 ± 0.004, AUROC 0.900 ± 0.006. Very stable.

**corrupt+spectral regression**: the two auxiliary signals (corruption forcing temporal robustness; spectral loss shaping frequency features) appear to compete. Combining them loses the clean gradient from each. This happens with many auxiliary loss combinations — there's often a "sweet spot" of having one auxiliary and not piling them on.

### Scaling experiments
| Run | Config | BACC (REC) | Notes |
|-----|--------|-----------|-------|
| corrupt_big | depth=5, hidden=96, 150ep | 0.805 | 3× params, worse |
| masked_big | EMA JEPA, large, 150ep | 0.764 | Full JEPA objective, much worse |
| multicorpus | 4× data (TUAB+TUEV+TUSZ+TUEP) | frozen 0.812, FT 0.837 | FT = TUAB-only FT |

**Masked JEPA result**: `EEGPredictiveJEPA` on conv encoder, 150 epochs. BACC 0.764 per-recording — the worst result of all VICReg variants. Possible explanations:
1. Frame-prediction encourages the encoder to represent temporal dynamics, not class-discriminative statistics
2. Global-pool probe averages away the temporal structure the objective learned
3. 150 epochs may not be enough for the EMA target to stabilize

**Multi-corpus**: fine-tuned to 0.837 — identical to corruption fine-tune without multi-corpus. The binding constraint is encoder capacity (256-dim Conv1D), not data amount. More data at the same architecture = zero marginal gain on TUAB eval.

### Fine-tuning
| Run | Init | BACC (REC) | AUROC (REC) |
|-----|------|-----------|------------|
| corruption init → fine-tune | corrupt (s0) | **0.837** | **0.919** |
| multicorpus → fine-tune | multicorpus | 0.837 | — |

Fine-tuning improves BACC from 0.825 → 0.837 and AUROC from 0.904 → 0.919. The SSL pretraining is a good init but not a perfect representation for linear probing. The probe's nonlinearity (MLP) helps somewhat but the full fine-tuning gap suggests the frozen features aren't perfectly class-separable.

---

## yhammache's encoder comparison

All runs: 20 epochs, VICReg (inv_coeff=1 bug), eval with OLD eval.py (accuracy only).

| Encoder | Loss | Accuracy | Random floor | Below random? |
|---------|------|----------|-------------|--------------|
| conv | vicreg | **0.837** | 0.746 | No |
| conv | sigreg | 0.812 | 0.746 | No |
| labram | vicreg | 0.775 | 0.652 | No |
| labram | sigreg | 0.725 | 0.652 | No |
| biot | sigreg | 0.783 | 0.797 | **YES** (-0.014) |
| biot | vicreg | 0.775 | 0.797 | **YES** (-0.022) |
| eegpt | vicreg | 0.641 | 0.674 | **YES** (-0.033) |
| eegpt | sigreg | 0.627 | 0.674 | **YES** (-0.047) |

**EEGPT+SIGReg worse than EEGPT+VICReg**: SIGReg is a stronger anti-collapse mechanism, but with the channel-pool bottleneck, it may have forced even harder diversity constraints that the architecture couldn't satisfy. With the bug, SIGReg made EEGPT worse than VICReg — the opposite of the pattern seen with Conv. This further confirms the issue was architectural + coefficient-related.

**LaBraM+VICReg (0.775) = BIOT+VICReg (0.775)**: both methods reach the same accuracy despite very different tokenisation. BIOT (FFT) loses phase but gains frequency precision; LaBraM (raw) keeps phase but has noisier patches. The net accuracy happens to be the same. Their error modes are likely different.

**LaBraM+SIGReg (0.725) < LaBraM+VICReg (0.775)**: Unusual — SIGReg should be better or equal. Possible cause: with inv_coeff=1 (the bug), VICReg's gentle variance penalty was incidentally better calibrated than SIGReg's harder Gaussianization constraint. This interaction shows how the bug complicated all comparisons.

---

## yhammache's predictive JEPA runs

New objective: `EEGPredictiveJEPA` with EMA target encoder + GRU predictor.

| Encoder | pred_loss at ep 19 | std_loss at ep 19 | Notes |
|---------|-------------------|------------------|-------|
| conv | 1.23 (oscillating) | ~5e-5 (near zero) | Not converged |
| biot | 0.10 (stable) | 0.0 (std > 1.0 = no collapse) | Converged |
| labram | — | — | Likely similar to biot |
| eegpt | — | — | Unknown |

**conv_predictive oscillating**: pred_loss of 1.23 at epoch 19 with high variance. The Conv encoder + GRU predictor combination was unstable. Possibly the Conv representation lacks temporal ordering information (global avg pool collapses time) that the GRU needs to predict future frames. The GRU was trying to predict from a representation that had already thrown away temporal structure.

**biot_predictive converged**: FFT features might be more temporally stable (spectral content changes slowly), making them easier to predict. But ease of prediction ≠ quality for the probe task.

**No probe evals were done on predictive JEPA before end of hackathon** (not enough cluster time). The convergence stats are from training logs only.

---

## tcourtois's EEGPT vicreg fix (job 75490)

Config: EEGPT encoder, VICReg with FIXED coefficients (25,25,1), 20 epochs.

| Epoch | Loss | inv_loss | var_loss | cov_loss |
|-------|------|---------|---------|---------|
| 0 | 2.18 | 0.046 | 0.007 | 0.812 |
| 4 | 1.63 | 0.030 | 0.003 | 0.812 |
| 5 | 1.53 | 0.031 | 0.002 | 0.705 |
| 6 | 1.32 | **0.023** | 0.006 | 0.601 |
| 7 | 1.30 | 0.025 | 0.002 | 0.606 |
| 8 | 1.16 | 0.020 | 0.001 | 0.628 |
| 9 | 1.08 | 0.019 | 0.0003 | 0.609 |
| 10 | 1.12 | 0.020 | 0.002 | 0.547 |
| 11 | 1.11 | **0.018** | 0.005 | 0.524 |

**Old run comparison** (best inv_loss was 0.103 at epoch 19, never improved below that)

**cov_loss is high (0.524–0.812)**: the covariance decorrelation is struggling. This means the 256-dim representations have correlated dimensions — the encoder hasn't fully learned to use the representational space. Possible reasons:
1. With 10 tokens and 4 transformer layers, the EEGPT representation space is inherently low-dimensional despite being projected to 256
2. The projector MLP (256→1024→1024) might need different architecture to spread representations

**The cov_loss pattern was also high for Conv early in training** and decreased over time. At epoch 20+, cov might decrease further for EEGPT. The fixed-coeff EEGPT training was still running at writeup time (epoch 11/20).

**Total loss scale**: 1.1–2.2 vs old run's 0.25–0.47. This is expected — same architecture, but now invariance is weighted 25× more, so the loss magnitude increased proportionally. The numbers are not comparable across the two runs; only the unweighted `inv_loss` (the raw MSE) is comparable.

---

## tvasnier's pending runs (end of hackathon)

These were in the Slurm queue at writeup time:

| Job | Name | Status | Expected result |
|-----|------|--------|----------------|
| 75578 | transf | PD → likely R | New 3.65M transformer encoder; designed to close LaBraM gap |
| 75577 | biot | R | New BIOT variant (unknown config) |
| 75576 | biot | R | Same, different seed or config |
| 75587 | bdnet_s1 | PD | BDNet (brain decoder net?), seed 1 |
| 75588 | bdnet_s2 | PD | BDNet, seed 2 |
| 75586 | bdnet_s0 | R | BDNet, seed 0 |

The `transf` job was the key bet for the end of the hackathon. A 3.65M-param transformer encoder at BIOT-scale parameters would test whether the Conv > LaBraM result is about architecture or parameter count.

---

## Baseline evals (our reimplementations)

### EEGNet (supervised)
Architecture: depthwise separable CNN, temporal + spatial filters, ~9k params.
- Per-window BACC: 0.796 / AUROC: 0.882
- Per-recording BACC: 0.824 / AUROC: 0.913
- Published (Kiessner et al.): 0.804 per-window — our reimplementation slightly below, consistent

### ShallowConvNet (supervised)
Architecture: single conv layer (temporal), depthwise (spatial), square activation, mean pool.
- Per-window BACC: 0.777 / AUROC: 0.857
- Per-recording BACC: 0.803 / AUROC: 0.893
- A simple supervised baseline that our best SSL (per-window 0.775) nearly matches frozen

**Key comparison**: our SSL at 0.775 per-window is just below ShallowConvNet supervised at 0.777. This means our SSL encoder captures almost as much discriminative information as a model specifically trained to classify. This is actually a reasonable result for self-supervised pretraining, framed honestly.

---

## Experiments considered but not run

1. **LogReg probe** (sklearn, L2-regularized): more interpretable than MLP, standard in SSL literature. Would likely give lower absolute numbers but be more directly comparable to published results.

2. **EEGPT without channel-drop augmentation**: ablate the augmentation to see if EEGPT recovers with fixed VICReg but no channel drop. Would isolate whether the architectural bottleneck alone caused collapse.

3. **LaBraM with pretrained tokenizer**: the real LaBraM uses a masked patch prediction pretraining step to learn the patch embeddings before the SSL phase. Would likely improve LaBraM-style encoder significantly.

4. **Longer pretraining (100+ epochs) with correct VICReg**: all our runs were 20 epochs. LaBraM pretrained for ~200 epochs on a much larger dataset. Even with correct coefficients, 20 epochs may be too few.

5. **SimCLR with negative pairs**: contrastive loss with large batch (or memory bank). Not architecturally compatible with the EB-JEPA framework but potentially strong.
