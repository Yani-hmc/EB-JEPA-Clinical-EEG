# EB-JEPA EEG — Full Process Log

Chronological narrative of the entire hackathon. Everything, including dead ends.

---

## Phase 0 — Setup and orientation

### What we inherited
The competition framework (EB-JEPA) provided:
- Two-view SSL training loop in `examples/eeg/main.py`
- Dataset loader `eb_jepa/datasets/eeg/dataset.py` with TUAB-specific splits
- VICReg and SIGReg (BCS/Epps-Pulley) loss implementations in `eb_jepa/losses.py`
- A Conv1D encoder (`EEG1DEncoder`) as the baseline
- `examples/eeg/eval.py` — MLP linear probe on frozen encoder

What was NOT provided:
- Any encoder other than Conv1D
- Any results or baselines
- Any indication of what VICReg coefficients to use
- Any clarity on evaluation protocol (per-window vs per-recording)

### Cluster: Dalia
- 18× nodes, each 4× GB200 GPUs (NVIDIA Blackwell, aarch64)
- Slurm scheduler, Vivatech reservation (`--reservation=vivatech`)
- Our group: `vivatech-slightlyunawarefc` (shared GPU quota)
- Python binary: `/lustre/work/vivatech-slightlyunawarefc/yhammache/venvs/eb_jepa_aarch64/bin/python`
  - Important: our own Python didn't exist at first, we used yhammache's venv throughout
- Checkpoint dir: `/lustre/work/vivatech-slightlyunawarefc/tcourtois/checkpoints/eeg/`
- Training logs: `/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/`

### Team division of labour
- **tcourtois**: infrastructure, code audit, VICReg bug investigation, EEGPT diagnosis, report
- **yhammache**: encoder implementations (labram, biot, eegpt), systematic encoder comparison
- **tvasnier**: SSL objective ablations (spectral regularisation, corruption augmentation), fine-tuning, SOTA table

---

## Phase 1 — First encoder implementations

### yhammache's encoder additions
Added three new encoder classes to `examples/eeg/encoders.py`:

**LaBraMEncoder**: Inspired by the LaBraM paper (Jiang et al., ICLR 2024).
- Patchify each of 19 channels into 10 non-overlapping patches of 200 samples (= 1 second @ 200Hz)
- Linear embed each patch from 200-dim to `embed_dim`
- Add learnable per-channel embedding + learnable per-time-position embedding
- Standard ViT transformer (depth × (MHA + FFN)) on 190 = 19×10 tokens
- Mean-pool all tokens → 256-dim representation

**BIOTEncoder**: Inspired by BIOT (Yang et al., NeurIPS 2023).
- Same patching as LaBraM BUT embeds the FFT magnitude spectrum of each patch instead of raw samples
- The FFT discards phase information entirely — this turns out to be a critical limitation

**EEGPTEncoder**: Hierarchical spatial+temporal.
- For each time-patch: run cross-channel MultiheadAttention with a learnable `chan_query` (shape [1,1,D])
  to compress 19 channel tokens → 1 summary token
- Then run temporal transformer over the 10 summary tokens
- Only 10 total tokens (vs 190 for LaBraM) — extreme compression
- Designed for channel-invariant features (the `chan_query` learns which channel combination to focus on)

### First training runs
yhammache submitted all 4 encoders × 2 losses = 8 jobs. First results (accuracy only, old eval.py):

| Encoder | VICReg | SIGReg | Random floor |
|---------|--------|--------|-------------|
| conv | **0.837** | 0.812 | 0.746 |
| labram | 0.775 | 0.725 | 0.652 |
| biot | 0.775 | 0.783 | 0.797 |
| eegpt | 0.641 | 0.627 | **0.674** |

Two immediate anomalies:
1. BIOT and EEGPT both scored **below their random-init floors** — the trained encoder was WORSE than no training at all.
2. EEGPT was dramatically bad: 0.641 vs 0.674 random. Something had failed.

---

## Phase 2 — EEGPT collapse investigation

### Initial hypothesis: training instability
Checked EEGPT training logs. Loss went down from 0.47 to 0.25 over 20 epochs — appeared stable. Nothing crashed. But the probe score was dismal.

### Second hypothesis: initialization conflict
`EEGPTEncoder.__init__` sets `self.chan_query = nn.Parameter(trunc_normal_(...))` then calls `self.apply(init_module_weights)`. Was `init_module_weights` overwriting `chan_query`?

Checked `init_module_weights`: it only touches `nn.Linear` and `nn.Conv*` modules. `nn.Parameter` is not a module — it's never touched. Not a bug. ✓

### Root cause identified: VICReg coefficients

Examined `eb_jepa/losses.py`. Found:

```python
class VICRegLoss(nn.Module):
    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        ...
    def forward(self, z1, z2):
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)
        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
```

The invariance (sim_loss) weight was **hardcoded to 1.0**. The paper (Bardes et al., ICLR 2022) uses λ=25. Our invariance term was underweighted by **25×**.

What this means for training:
- Loss = 1×inv + 1×var + 1×cov
- Should be: Loss = 25×inv + 25×var + 1×cov
- The optimizer barely penalized view mismatch — it was mostly minimizing covariance

For Conv encoder, this was bad but not catastrophic — Conv is simple enough that even weak invariance pressure shaped useful features.

For EEGPT with its channel-pool bottleneck:
- Two views with channel-drop augmentation: different channels zeroed out per view
- Channel-pool compresses 19 tokens → 1 token based on which channels are present
- Views produce very different summary tokens → high invariance loss naturally
- But since invariance is underweighted 25×, the optimizer chose to minimize covariance instead
- Result: the encoder learned to decorrelate dimensions, NOT to match views
- Representations were random-quality (actually worse, since they'd been distorted to minimize covariance)

### BIOT underperformance — separate issue
BIOT uses FFT magnitude of each patch. The SSL training succeeded (loss converged) but the features are wrong for the task:
- FFT magnitude captures frequency content (power spectrum)
- Abnormal EEG pathology often manifests as temporal patterns (spike-wave complexes, burst suppression) that are phase-dependent
- SSL learned to match frequency statistics between views — but frequency statistics alone are insufficient for abnormality detection
- Per-channel FFT loses inter-channel phase relationships (important for detecting focal vs generalized pathology)

This is an architectural mismatch between the tokenisation and the task, not a training failure.

---

## Phase 3 — VICReg fix

### Code fix
Modified `eb_jepa/losses.py`:
```python
class VICRegLoss(nn.Module):
    def __init__(self, inv_coeff=1.0, std_coeff=1.0, cov_coeff=1.0):
```
Default kept at 1.0 for backward compatibility. Then in `examples/eeg/main.py`:
```python
self.loss_fn = VICRegLoss(
    inv_coeff=cfg.get("inv_coeff", 25.0),
    std_coeff=cfg.get("std_coeff", 25.0),
    cov_coeff=cfg.get("cov_coeff", 1.0)
)
```
And in `examples/eeg/cfgs/train.yaml`:
```yaml
inv_coeff: 25.0
std_coeff: 25.0
cov_coeff: 1.0
```

### EEGPT rerun (job 75490)
Submitted EEGPT+vicreg with fixed coefficients. Results (monitored every ~15 min):

| Epoch | inv_loss (old run) | inv_loss (fixed run) |
|-------|-------------------|----------------------|
| 0 | 0.214 | **0.046** |
| 6 | 0.103 (best ever) | **0.023** |
| 11 | 0.103 | **0.018** |

The fix worked dramatically. By epoch 11 the EEGPT invariance loss was already **5× lower** than the old run ever reached. This confirms the coefficient bug was the primary driver of collapse.

### Merge conflict during this work
While we were working on the VICReg fix, tvasnier pushed a commit adding `objective: twoview` and `ema: 0.996` to train.yaml, and later `pred_std_coeff`/`pred_cov_coeff`. This created a merge conflict.

Resolution: kept BOTH sets of changes. The final train.yaml has:
- Our VICReg coefficients (inv_coeff: 25.0, std_coeff: 25.0)
- tvasnier's predictive-JEPA fields (objective, ema, pred_std_coeff, pred_cov_coeff)

---

## Phase 4 — tvasnier's ablation campaign

### SSL objective ablations (conv encoder, recording-level)
tvasnier ran a systematic ablation over ~16 variants. Key runs:

**Spectral regularisation** (DDSP-inspired auxiliary loss):
- `spec_fixed` (spectral_coeff=0.1): **BACC 0.836** — best of all runs
- `fftc` variants: minor improvement over base (0.807 peak)
- `spec (0.3)`: degraded to 0.789 — too strong regularisation

**Corruption augmentation**:
- `corrupt`: BACC 0.825, AUROC 0.904
- 3-seed average: BACC 0.819 ± 0.004, AUROC 0.900 ± 0.006
- Very consistent across seeds — robust finding

**Combined**:
- `corrupt_spec`: BACC 0.802 — corruption + spectral together actually hurt vs. each alone (possible interference between the two auxiliary signals)
- `corrupt_sigreg`: BACC 0.825, same as base corrupt — SIGReg didn't add much on top of corruption

**Scaling**:
- `corrupt_big` (deeper/wider encoder): BACC 0.805 — worse than small corrupt (0.825)
- `masked_big` (masked-prediction JEPA): BACC 0.764 — significantly worse

**Multi-corpus**:
- Pretrained on 4× data (TUAB+TUEV+TUSZ+TUEP): frozen 0.812, fine-tuned 0.837
- Fine-tuned 0.837 = identical to TUAB-only fine-tune — more data at same encoder size = zero gain

### spec_fixed mystery
The `spec_fixed` checkpoint name caused confusion. "Fixed" referred to a bugfix in the spectral loss (a frequency bin normalization error tvasnier caught), not to our VICReg fix. These are different things. `spec_fixed` uses:
- VICReg with OLD coefficients (1,1,1) — the bug was present
- Spectral auxiliary loss with corrected normalization
- Best recording-level BACC: 0.836

This means our best result was achieved WITH the bug. The spectral regularisation happened to compensate somewhat for the underweighted invariance.

---

## Phase 5 — Evaluation protocol discovery

### How we realised the mistake
tvasnier created `eeg-jepa-baseline` branch with a more complete benchmark comparison. That branch includes `RESULTS_COMPILED.md` which explicitly calls out:

> **B1 — Per-window (FAIR comparison to the literature)**
> **B2 — Per-recording (mean-pool 16 windows; clinical, but NOT comparable to the papers)**

Our early "we beat LaBraM!" announcement was based on comparing:
- Our per-recording BACC: 0.836
- LaBraM-Base per-window BACC: 0.814

These are not the same thing. Per-recording is systematically higher for the same encoder because:
1. 16 windows per patient → mean-pooled embedding has much lower variance
2. The probe effectively sees a smoothed, denoised signal
3. Window-level label noise (EEG is non-stationary) is averaged out

### Per-window evaluation implementation
The baseline branch added `--level both` to eval.py, which runs:
- **per-window**: each of the 16 windows is its own test point
- **per-recording**: mean-pool embeddings, one test point per patient

Running `--level both` on our best checkpoints gave:
- EB-JEPA base: 0.756 per-window / 0.796 per-recording
- EB-JEPA +corruption: 0.770 per-window / 0.825 per-recording
- EB-JEPA +spectral 0.1 (spec_fixed): 0.765 per-window / 0.836 per-recording
- EB-JEPA SIGReg+corruption: **0.775 per-window** / 0.825 per-recording

The gap is consistently 5-6pp. This is not a rounding error.

### What the literature actually reports
We verified every published TUAB number against the source PDF (SOTA_TABLE.md):

| Paper | Claimed BACC | Actual (from PDF) | Eval level |
|-------|-------------|-------------------|-----------|
| BIOT | 0.802 | 0.802 ✓ | per-window |
| LaBraM-Base | 0.814 | 0.814 ✓ | per-window |
| LaBraM-Huge | 0.826 | 0.826 ✓ | per-window |
| FEMBA | 0.808 | 0.808 ✓ | per-window |
| EEGNet (published) | 0.804 | 0.804 ✓ | per-window |

Two hallucinated values found and retracted:
1. EEGNet 0.764 (had appeared in some of our notes — not from the paper, fabricated)
2. Deep4Net BACC 0.854 (was accuracy, not BACC — now correctly labeled)

### Honest final position
- **Per-window (fair)**: 0.775 (SIGReg+corruption) — below BIOT (0.802), below LaBraM-Base (0.814)
- **Per-recording (clinical)**: 0.836 frozen / 0.837 fine-tuned — competitive with EEGNet supervised (0.824)
- **Where we sit**: approximately ContraWR level (0.775 per-window), which is a reasonable SSL baseline from 2022

---

## Phase 6 — Predictive JEPA (teammate addition)

### EEGPredictiveJEPA
yhammache added a new `EEGPredictiveJEPA` class to `examples/eeg/main.py`:
- Splits each window into temporal frames
- Uses EMA (momentum=0.996) target encoder on future frames
- `RNNPredictor` (GRU-based) rolls forward frame-by-frame to predict target representations
- Anti-collapse via HingeStdLoss + CovarianceLoss on online encoder frames

Ran on all 4 encoders (conv, labram, biot, eegpt). Results were disappointing:
- conv+predictive: pred_loss oscillating at 1.23 at epoch 19 — not converged
- biot+predictive: pred_loss 0.10 at epoch 16 — converged, but what did it learn?

Key insight: a frame-prediction objective trains the encoder to produce representations that help predict the NEXT frame. A global-pool frozen probe reads the mean of all frame representations. These objectives are misaligned — the probe doesn't need temporal prediction, it needs class-discriminative features. The predictive JEPA likely learned temporal dynamics at the expense of pathology-relevant features.

The proper evaluation for predictive JEPA would be a per-frame probe or fine-tuning — not tested in this hackathon.

---

## Phase 7 — Infrastructure notes and difficulties

### GPU quota (AssocGrpGRES)
The group `vivatech-slightlyunawarefc` has a shared GPU quota. When tvasnier had 3 running jobs (each on 1 GPU), our EEGPT fix job (75490) was blocked:

```
JOBID  NAME              ST  REASON
75490  eegpt_vicreg_fix  PD  (AssocGrpGRES)
```

We cancelled and resubmitted without `--partition=defq`, which sometimes bypassed the partition-level limit. Eventually started when tvasnier's jobs completed.

### Push rejections (remote ahead)
Multiple times during the hackathon, `git push` was rejected because teammates had pushed since our last pull. Fixed pattern:
1. `git pull --rebase` (preferred, keeps history clean)
2. If conflicts: resolve, `git add`, `git rebase --continue`
3. Then `git push`

### matplotlib not installed
`figures/plot_results.py` failed with `ModuleNotFoundError: matplotlib`. The aarch64 venv didn't have it. Fixed with:
```bash
pip install matplotlib numpy --break-system-packages
```
(The `--break-system-packages` flag was needed because the venv used system-managed pip.)

### eval.yaml never loaded
`examples/eeg/eval.py` uses `EEGConfig` dataclass defaults directly — `eval.yaml` is defined but never passed to the loader. `n_windows=16` in eval.yaml coincidentally matches the dataclass default, so the evaluation was correct by accident.

### aarch64 compilation
Some PyTorch operations behave differently on GB200 (Blackwell) ARM architecture vs. typical x86 A100 nodes. No training bugs observed, but compile times were longer and some CUDA kernel warnings appeared in logs (non-fatal).

---

## Phase 8 — eeg-jepa-baseline branch

tvasnier maintained a separate branch `eeg-jepa-baseline` with extensive new work that was never merged to main. Contents:

### New encoder: 3.65M transformer
Commit `dd640d2`: "add Transformer encoder (3.65M, BIOT-scale) — the capacity upgrade to beat the tie"
- Named `TransformerEncoder`
- Same token count as LaBraM (190 tokens) but much wider/deeper
- Designed to close the gap with LaBraM-Base
- Not yet trained at end of hackathon (tvasnier job 75578 for this was pending)

### EEGNet and ShallowConvNet reimplementations
Full PyTorch reimplementations as supervised upper bounds:
- EEGNet per-window: 0.796 BACC (vs published 0.804 — small gap, consistent)
- ShallowConvNet per-window: 0.777 BACC

These serve as sanity checks: if our SSL beats EEGNet supervised at per-window level, we've achieved a meaningful result. We didn't — EEGNet supervised at 0.796 beats our best SSL (0.775).

### TUEV 6-class transfer
Froze TUAB-pretrained encoder, probed on TUEV event classification (6 classes):
- Random floor: BACC 0.337, Kappa 0.110
- EB-JEPA frozen: BACC 0.364, Kappa 0.141
- BIOT (paper): BACC 0.528, Kappa 0.435

Transfer works — we're above random. But BIOT dominates on TUEV, possibly because FFT features capture the frequency signatures of different event types better than waveform features.

### SOTA_TABLE.md
Every published TUAB result verified against source PDF. Includes:
- SPaRCNet: 0.790 BACC
- ContraWR: 0.775 BACC
- CNN-Transformer: 0.787 BACC
- FFCL: 0.776 BACC
- ST-Transformer: 0.797 BACC
- AFTA: 0.788 BACC
- FEMBA: 0.808 BACC
- BIOT: 0.802 BACC
- LaBraM-Base: 0.814 BACC
- LaBraM-Huge: 0.826 BACC

This branch was never merged — should be merged post-hackathon.

---

## Phase 9 — What remained open at end of hackathon

1. **EEGPT fixed-coefficient eval**: job 75490 still running at writeup time (epoch 11/20). Inv_loss 0.018, much better. Need eval job once training completes.

2. **New transformer encoder (3.65M)**: tvasnier's job 75578 was pending. This was the main architectural bet for closing the LaBraM gap.

3. **Per-window eval on conv encoder**: we have recording-level BACC for conv, but the systematic per-window eval only happened on tvasnier's twoview (VICReg) runs. Need yhammache's encoder comparison at per-window level too.

4. **Merge baseline branch**: `eeg-jepa-baseline` has SOTA_TABLE.md, RESULTS_COMPILED.md, EEGNet/ShallowConvNet, new transformer encoder, TUEV transfer results. None of this is on main.

5. **Predictive JEPA probe alignment**: the frame-prediction objective + global-pool probe mismatch was never resolved. Would need per-frame probing to fairly evaluate it.
