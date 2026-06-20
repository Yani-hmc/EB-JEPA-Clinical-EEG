# Technical Notes — Deep Dives

Architecture, loss functions, and implementation details that didn't fit in the report.

---

## Dataset and preprocessing

### TUAB splits (ours vs literature)
We use a patient-disjoint split derived from TUAB v3.0.1:
- Train: 2717 recordings
- Eval: 276 recordings

The BIOT/LaBraM papers use a different split from the same dataset:
- Train: ~2,715 recordings (very similar)
- Eval: 2,339 recordings (8× larger!)

Their eval split is larger because they use **all eval windows** from each recording as separate test points. They have ~409,455 test windows vs our 276×16 = 4,416. This is the root of the per-window discrepancy.

### z-score normalization
Per-channel, per-window. Applied in `dataset.py`:
```python
mean = x.mean(dim=-1, keepdim=True)
std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
x = (x - mean) / std
```
This removes DC offset and amplitude scale per channel, making the representation amplitude-invariant. Appropriate for EEG where absolute amplitude varies across recording conditions and electrode impedance.

### Data augmentation stack (SSL, two-view)
Both views get independently:
1. `aug_noise_std=0.1`: additive Gaussian noise
2. `aug_scale_jitter=0.2`: random amplitude scaling in [0.8, 1.2]
3. `aug_chan_drop_p=0.2`: zero out each channel with p=0.2 independently per view
4. `aug_time_mask_frac=0.2` (corruption variant): zero out 20% of time steps
5. Spike injection (corruption variant): 20% of windows get ±6σ spikes at random times

The corruption augmentation was designed specifically to force robust temporal representations. Time masking forces the encoder to extrapolate missing segments; spike injection forces robustness to artifact-like contamination (common in clinical EEG).

The **channel-drop + EEGPT channel-pool conflict**: chan_drop_p=0.2 zeroes out channels independently per view. EEGPT's chan_pool compresses 19 channels → 1 token via cross-channel attention. If view1 has channels {1,3,5,...} zeroed and view2 has {2,4,6,...} zeroed, the two summary tokens will look structurally different even though they derive from the same underlying signal. This makes the invariance term much harder to minimize for EEGPT than for Conv or LaBraM (which operate independently per channel).

### Probe evaluation
SSL probe (`eval.py`):
- **N=16 evenly-spaced windows per recording** (frames drawn from `np.linspace(0, duration, 16)`)
- For each window: z-score → encoder → `represent()` → 256-dim embedding
- **Per-recording**: mean of 16 embeddings → single 256-dim vector → MLP probe (2 hidden layers)
- **Per-window** (added later): each embedding → MLP probe separately; majority vote or individual predictions

MLP probe architecture: Linear(256, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 2)
Trained with Adam, 100 epochs, lr=1e-3, weight_decay=1e-4. No LogReg probe was ever added (mentioned in audit but not implemented).

---

## VICReg: the full story

### Paper coefficients
Bardes et al. (ICLR 2022) Table 1 ablation: λ=25, μ=25, ν=1.
This is for 2048-dim projections with batch size 2048. At smaller batch/dim, the covariance term is naturally smaller, so 25/25/1 still makes sense as a starting point.

### What (1,1,1) actually does
With equal weights, the loss is dominated by whichever component has the largest magnitude:
- `sim_loss = MSE(z1, z2)` — can be large early in training when views are dissimilar
- `var_loss = Σ relu(1 - std(z))` — bounded, typically in [0, 2D] where D=256
- `cov_loss = Σ_off_diag(C^2) / (D*(D-1))` — typically small, bounded by 1

In practice with (1,1,1), the optimizer minimizes `sim_loss` first (it's large), but since `sim_loss` weight = 1 and `var_loss` weight = 1, there's no strong push to also maintain variance. Result: encoder gradually collapses to minimizing sim_loss (making z1≈z2) but at the cost of variance — a classic partial collapse.

With (25,25,1), the loss is:
- 25×sim + 25×var dominate equally → encoder must simultaneously match views AND maintain variance
- 1×cov is a gentle decorrelation nudge, not the dominant force

### EEGPT-specific failure mode
With (1,1,1), EEGPT found a local minimum:
- `sim_loss` minimized (z1≈z2) for the temporal transformer's outputs
- But since the channel-pool makes different views structurally different, `sim_loss` couldn't fully converge
- The encoder partially solved this by learning representations that are very low-variance (nearly constant z, making z1≈z2≈constant)
- This explains `acc = 0.641 < random 0.674`: the representations are nearly collapsed to a single point, carrying less information than a random encoder with natural weight variance

With (25,25,1), var is equally penalized — collapse is prevented — so the encoder must find a non-trivial solution to the invariance objective.

---

## Encoder architectures: design notes

### Conv (EEG1DEncoder)
```
Input: [B, 19, 2000]
Block × 4: Conv1d(C→2C, k=7, stride=2, pad=3) → BN → GELU
After 4 blocks: [B, 512, 125]  (T halved 4 times: 2000→1000→500→250→125)
Global avg pool: [B, 512]
Linear(512, 256): [B, 256]
```

Why it works best: 
1. Stride-2 convolutions are proven EEG feature extractors (DeepConvNet, EEGNet ancestry)
2. BN normalizes within-batch, which incidentally aligns with the SSL assumption that batch diversity captures the data manifold
3. Local receptive field (k=7 → 35ms at 200Hz) captures fine temporal structure
4. No inter-channel mixing → each channel processed independently, then aggregated at global pool. This is robustly compatible with channel-drop augmentation.
5. 0.4M params: small enough to overfit minimally, large enough to learn useful features

### LaBraM-style
Patch embedding is the key choice:
- 200 samples = 1 second at 200Hz — chosen to match canonical EEG time scales (alpha rhythm ~10Hz → 10 cycles per patch)
- Linear projection from 200 → 128 dims (learned tokeniser)
- Learnable per-channel embedding: [19, 128] — encodes which electrode this token comes from
- Learnable per-time-position embedding: [10, 128] — encodes temporal order within the window

The patch transformer sees 190 tokens, processes cross-channel and cross-time information simultaneously. This should in principle be more expressive than Conv, but in practice LaBraM performs worse than Conv here, probably because:
1. 190 tokens × 4 attention heads × 4 layers = many parameters per layer that need more pretraining data/epochs to converge
2. Linear patch embedding (raw samples → dim) is a less powerful tokeniser than Conv (which captures local multi-scale features)

LaBraM (the paper) uses a **neural tokenizer** pretrained separately on masked patch prediction before the main SSL. We skipped this step — the raw-sample linear projection is a weak tokeniser by comparison.

### BIOT-style
Same architecture as LaBraM but replaces raw-sample patches with FFT magnitude:
```python
fft = torch.fft.rfft(patch, dim=-1)  # [B, C, P, F] where F = patch_len//2 + 1
x = fft.abs()                         # magnitude only, phase discarded
```

The FFT magnitude patch = power spectral density within the 1-second window.

Why this is suboptimal for TUAB:
- Abnormal EEG includes: spike-wave complexes, burst suppression, rhythmic slowing
- Spike-wave: sharp transient → identifiable by its time course and inter-channel synchrony, not primarily by spectral content
- Burst suppression: alternating high-activity and flat (near-zero) periods → detectable temporally, less so spectrally (both states have distinct spectra but their alternation pattern is key)
- 1/f noise (normal EEG background) has a power-law spectrum — the FFT magnitude alone can't distinguish normal 1/f from pathological spectral shapes without phase information

BIOT performs better on TUEV (event classification) than on TUAB — consistent with the hypothesis that event types (e.g., spike-and-wave at 3Hz, PLED, GPED) have stronger frequency signatures than pathology in general.

### EEGPT-style
The channel-pooling mechanism:
```python
# x: [B, P, C, D] — B=batch, P=patches, C=channels, D=embed
q = self.chan_query.expand(B*P, 1, -1)   # learnable query: [B*P, 1, D]
x_flat = x.view(B*P, C, D)               # reshape channels as keys/values
pooled, _ = self.chan_pool(q, x_flat, x_flat)  # [B*P, 1, D]
pooled = pooled.view(B, P, D)            # [B, P, D] — one token per patch
# then temporal transformer on P=10 tokens
```

The `chan_query` is a learnable query that attends over the 19 channel tokens to produce one summary. In principle this learns which combination of channels is most discriminative. In practice:
- With 10 total tokens, the temporal transformer has very limited context
- Concatenated to 10 patch representations — equivalent to only 5 seconds of EEG at 1-token/second resolution
- Much lower resolution than LaBraM's 190 tokens

The design makes sense for transfer learning (channel-count invariance) but is wasteful for TUAB where we always have 19 channels.

---

## SIGReg (BCS) details

SIGReg = Sliced Gaussianization Regularisation, based on the Batched Characteristic Slicing (BCS) method from the EB-JEPA paper.

Anti-collapse via random projections:
1. For each representation z ∈ R^D: project onto `num_slices=256` random unit vectors
2. For each projection: test if the 1D distribution is Gaussian (Epps-Pulley test)
3. Loss = sum of test statistics (pushes projections toward Gaussian)

Why this works: if z collapses to a point or low-dim submanifold, random 1D projections will be non-Gaussian (degenerate). Penalizing non-Gaussianity forces the representations to spread out.

Compared to VICReg's variance term:
- VICReg: `relu(1 - std_j)` for each dimension separately → hinge on per-dimension variance
- SIGReg: operates on projections (mixtures of dimensions) → captures higher-order statistics

SIGReg should be harder to fool (optimizer can't satisfy it by trading variance across dimensions) and less sensitive to dimension choice.

In practice: SIGReg > VICReg by ~0.005 BACC per-window when combined with corruption augmentation.

---

## Failure modes: taxonomy

### Type 1: Representation collapse
**Symptom**: trained encoder scores below or at random init
**Cause**: SSL loss converges but representations become uninformative
**Seen in**: EEGPT (full collapse), BIOT (partial — collapses to frequency-only features)
**Fix for EEGPT**: correct VICReg coefficients

### Type 2: Wrong objective
**Symptom**: SSL loss converges well, probe score is OK but below expected
**Cause**: SSL objective trains features that don't match probe task
**Seen in**: masked-prediction JEPA (frame-prediction ≠ class-discriminative)
**Fix**: use per-frame probing, or switch to two-view objective

### Type 3: Tokenisation mismatch
**Symptom**: architecture trains fine, probe score low, consistent across SSL objectives
**Cause**: tokenisation discards task-relevant signal
**Seen in**: BIOT (FFT discards phase and temporal dynamics)
**Fix**: switch to raw-sample or learned tokeniser

### Type 4: Probe-encoder mismatch
**Symptom**: fine-tuning >> frozen probe
**Cause**: encoder features are good but not linearly separable in the right way
**Seen in**: all our encoders to varying degrees (frozen 0.775, fine-tuned 0.837 per-recording)
**Implication**: linear probe is a conservative estimate; actual representation quality is higher

---

## Things we considered but didn't try

### LogReg probe
Standard in SSL literature (SimCLR, MoCo, etc.) — scikit-learn LogisticRegression(max_iter=1000, C=0.1). We only had MLP probe. LogReg is more standard for reporting but MLP is more generous. Since both papers and we use different probes, comparisons are even less clean.

### Temperature-scaled SimCLR loss
Not in the framework. Would require negative pairs (larger batch or memory bank). Not applicable with VICReg/SIGReg design.

### EMA target encoder for two-view
Would make the framework closer to BYOL/DINO. Not attempted — predictive JEPA used EMA but in a different regime.

### Multi-scale Conv
Instead of 4×stride-2, use Conv with multiple kernel sizes in parallel (like Inception) to capture EEG patterns at different time scales simultaneously. Could help since EEG phenomena span 0.1s (spike) to 30s (burst suppression).

### Per-subject normalization
We z-score per window. Z-scoring per subject (over all windows) would preserve inter-window amplitude dynamics. Might help for detecting burst suppression (which is amplitude-temporal, not just spectral).

### Contrastive probing
Nearest-neighbour probe in the representation space. Not tried. Would give a proxy for representation quality without training a probe.
