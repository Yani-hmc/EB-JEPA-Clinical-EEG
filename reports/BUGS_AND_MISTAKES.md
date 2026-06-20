# Bugs, Mistakes, and Corrections

A complete record of everything that went wrong, why, and how we fixed it.
This is the document you want when comparing the report's claims to reality.

---

## Bug 0 — Best-epoch selection peeked at the eval set (headline FT number 0.837 → 0.812)

**Severity**: High — it was the exact difference between "SSL **beats** EEGNet" and "SSL **ties** EEGNet."

**What was wrong**: the fine-tuned per-recording result was reported at its *best epoch*, chosen by
the highest accuracy on the **evaluation** set. With no separate validation split, selecting the
epoch on the eval set uses the test data for model selection — a leak. EEGNet, by contrast, was
reported at its final epoch. The comparison was asymmetric and inflated JEPA.

**Evidence (3 seeds, per-recording BACC)**:

| seed | best-epoch (peeked) | final-epoch (fair) |
|---|---|---|
| 0 | 0.837 (ep 5) | 0.812 |
| 1 | 0.848 (ep 4) | 0.807 |
| 2 | 0.839 (ep 2) | 0.817 |
| **mean** | 0.841 | **0.812 ± 0.004** (AUROC 0.908 ± 0.006) |

**Fix**: report final-epoch, seed-averaged for *every* model. JEPA fine-tune = 0.812 ± 0.004 vs
EEGNet = 0.812 ± 0.013 → a statistical tie. The 0.837 / 0.919 figures are retracted; `research_paper.tex`,
`report.tex`, and the result tables now use **0.812 / 0.908**, and it is written up as Methodological
Lesson 4 in the paper. (The internal process logs below retain the original 0.837 as a historical record.)

---

## Bug 1 — VICReg coefficients (1,1,1) instead of (25,25,1)

**Severity**: Critical — caused EEGPT encoder to train below random init

**Location**: `eb_jepa/losses.py`, `VICRegLoss.__init__`

**What was wrong**:
```python
# Before fix:
def __init__(self, std_coeff=1.0, cov_coeff=1.0):
    ...
    total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
    # sim_loss weight = hardcoded 1.0 (no parameter)
```

The paper (Bardes et al., ICLR 2022) specifies λ=25, μ=25, ν=1. The invariance term was underweighted by 25×.

**Effect on Conv encoder**: moderate — Conv is architecturally simple and channel-drop augmentation is less destructive. Conv still learned useful features despite the bug. This is why the bug was hard to catch: the most-used encoder was relatively tolerant.

**Effect on EEGPT**: catastrophic — channel-drop + channel-pool creates structurally incompatible views. With invariance at 1/25th the paper weight, the optimizer never learned to bridge this gap. Instead it learned to decorrelate dimensions (minimize cov_loss), producing near-collapsed representations.

**Effect on BIOT**: likely contributed to underperformance, but the architectural issue (FFT discards phase) was the dominant factor.

**Fix**:
```python
# After fix:
def __init__(self, inv_coeff=1.0, std_coeff=1.0, cov_coeff=1.0):
    self.inv_coeff = inv_coeff
    ...
    total_loss = self.inv_coeff * sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
```
And in train.yaml: `inv_coeff: 25.0, std_coeff: 25.0, cov_coeff: 1.0`

**Evidence the fix worked**:
- EEGPT inv_loss at epoch 0: 0.046 (fixed) vs 0.214 (old)
- EEGPT inv_loss at epoch 11: **0.018** (fixed) vs 0.103 (old, epoch 19 — its all-time best)
- Fix needed 11 epochs to surpass what the old run never reached in 20 epochs

**When discovered**: during code audit of `losses.py`. Triggered by asking "why does EEGPT train worse than random init?"

---

## Bug 2 — eval.yaml never loaded

**Severity**: Low (no actual harm — values matched by coincidence)

**Location**: `examples/eeg/eval.py`

**What was wrong**: `eval.yaml` exists and defines `n_windows: 16`, but `eval.py` instantiates `EEGConfig()` with dataclass defaults, never loading the yaml. The file is dead.

**Why it wasn't harmful**: `EEGConfig` dataclass default for `n_windows` is also 16 — same value. The evaluation was correct by accident.

**Not fixed**: low priority, would require wiring up yaml loading in eval.py.

---

## Bug 3 — Hallucinated EEGNet baseline

**Severity**: Medium (affected our comparison to supervised baseline)

**What was wrong**: At some point in team notes, EEGNet's published BACC on TUAB appeared as 0.764. This is not from any paper — it was fabricated during a conversation.

**The actual published number**: EEGNet achieves **0.804** BACC on TUAB per-window (Kiessner et al., 2022, verified from PDF).

**Discovery**: when tvasnier built `SOTA_TABLE.md` and checked every value against source PDFs.

**Impact**: We claimed "our SSL beats supervised EEGNet" using the 0.764 number. With the correct 0.804, our per-window SSL (0.775) is below supervised EEGNet. Claim retracted.

---

## Bug 4 — Deep4Net "BACC" was actually accuracy

**Severity**: Low (metric label error, value itself is real)

**What was wrong**: Deep4Net "BACC 0.854" from Schirrmeister et al. appeared in early comparisons. The paper reports **accuracy**, not BACC. For a balanced class distribution, acc ≈ BACC, but they're not the same metric. TUAB eval split is 45.6% abnormal (near-balanced), so the difference is small but the label was wrong.

**Fix**: relabeled as "accuracy 0.854" in SOTA_TABLE.md and the report's backup slide.

---

## Mistake 1 — Per-recording vs per-window comparison

**Severity**: Critical for external claims, moderate for internal understanding

**What happened**: We compared our per-recording BACC (0.836) to published per-window BAcc (LaBraM-Base: 0.814) and announced "we match LaBraM."

**Why this is wrong**: Per-recording BACC is systematically ~5pp higher than per-window for the same encoder (noise averaging over 16 windows). The two metrics are not comparable. LaBraM achieves 0.814 per-window; our per-window best is 0.775. We do not match LaBraM.

**Timeline**:
- During ablation campaign: tvasnier noticed per-recording scores were high, first reports say "BACC 0.836"
- Comparison to LaBraM: made without checking evaluation protocol
- Discovery: tvasnier's baseline branch explicitly labels per-window vs per-recording tables
- Correction: all claims revised; report separates the two tables clearly

**Why the mistake happened**: 
1. The framework's default eval was per-recording (single embedding per patient). This naturally produced higher numbers.
2. Literature numbers from BIOT/LaBraM were copied without checking their evaluation section carefully.
3. BACC 0.836 > 0.814 is emotionally satisfying — confirmation bias led us to not question it.

**What we learned**:
1. Always check evaluation protocol before comparing to literature
2. Per-recording is the clinically correct metric (classify patients), per-window is the benchmark metric. Both are valid, but they must be labeled clearly.
3. Numbers higher than expected should trigger a verification step, not celebration.

---

## Mistake 2 — BIOT labeled "below random" as "collapse"

**Severity**: Low — diagnostic precision issue

**What happened**: BIOT scored below random init on some metrics and we initially called it "SSL collapse" alongside EEGPT.

**Why this is imprecise**:
- EEGPT collapsed because the VICReg bug + architectural bottleneck caused near-zero-variance representations. The SSL didn't learn at all.
- BIOT collapsed because it learned the WRONG features (frequency statistics) rather than no features. The SSL converged fine; the tokenisation was the problem.

These are different failure modes (see `TECHNICAL_NOTES.md`, taxonomy of failure modes). The word "collapse" is appropriate for EEGPT but misleading for BIOT — "architectural mismatch" is more accurate.

**In the report and slides**: we use "trained below random" for both and explain EEGPT specifically. The BIOT explanation ("FFT discards phase") is mentioned in the main report but not in the slides due to time.

---

## Mistake 3 — Initial EEGPT debugging went in wrong direction

**Severity**: Time lost only

**What happened**: First hypothesis was initialization conflict — `self.apply(init_module_weights)` might overwrite the `chan_query` parameter. We spent time reading through `init_module_weights` to verify it only touches `nn.Linear`/`nn.Conv*`.

**Reality**: This was not the bug. The real bug was the VICReg coefficients, which we found independently by examining `losses.py`.

**Why it was a reasonable first guess**: `trunc_normal_` init followed immediately by `self.apply(...)` is a suspicious pattern. It would be easy for the init function to accidentally reset a recently set Parameter.

---

## Mistake 4 — Claimed "multi-corpus helps" before verifying per-window

**Severity**: Low

**What happened**: Multi-corpus pretraining showed frozen BACC 0.812 per-recording (vs base 0.796 per-recording) — +0.016. Early note: "multi-corpus pretraining improves SSL."

**After per-window eval**: the improvement is still there but smaller (+0.012 per-recording), and per-window numbers were not separately computed. The fine-tuned model (0.837 per-recording) was identical to TUAB-only fine-tuned (0.837) — no benefit. The binding constraint is encoder capacity, not data volume.

---

## Things we got right under pressure

**Worth recording** — these worked on first try:

1. **EEGPT training loss didn't crash** even with the bug — it converged stably to a bad solution, which made diagnosing harder but also meant we had real data to analyze.

2. **3-seed corruption average** was close to single-seed (0.819±0.004 vs 0.825) — corruption augmentation is stable, not a lucky seed.

3. **VICReg fix diff was minimal** — adding one parameter to `__init__`, one multiplication in `forward`. The fix took 5 minutes to write; diagnosis took much longer.

4. **merge conflict resolution** — merged tvasnier's objective fields AND our VICReg coefficients without breaking either training mode.

5. **Audit of published numbers** — every SOTA number checked against source PDF before going into any comparison. Caught two errors before they made it into the presentation.

---

## Bug 3 — Predictive JEPA eval loads the same checkpoint for all encoders

**Severity**: High — all reported predictive JEPA numbers are invalid.

**What was wrong**: `eval_all_full_74861.out` is byte-identical across the four checkpoint dirs
(`conv_predictive_seed0`, `labram_predictive_seed0`, `biot_predictive_seed0`,
`eegpt_predictive_seed0`). The eval script resolves the checkpoint path from a hardcoded or
shared config rather than from the checkpoint directory it is launched from, so it evaluates the
same model four times and produces identical numbers regardless of encoder type.

**Evidence**: the four `eval_all_full_74861.out` files all report the same four blocks of results
(accuracy 0.837, 0.775, 0.641, 0.775) in the same order — impossible if different checkpoints
were loaded.

**Second symptom**: on the small-scale eval (`eval_all_74839.out`), conv and eegpt predictive
score *below their own random-init floors*, confirming collapsed representations. The anti-collapse
term (`std_loss`) was near-zero throughout training for all four runs, which is the training-time
signature of collapse.

**Fix**:
```python
# In eval.py / the eval launch script, replace any hardcoded ckpt_dir with:
ckpt_path = os.path.join(args.ckpt_dir, "latest.pth.tar")
ckpt = torch.load(ckpt_path, weights_only=False)
encoder = build_encoder(OmegaConf.create(ckpt["cfg"]["model"]))
encoder.load_state_dict(ckpt["encoder"])
```
The key is that `args.ckpt_dir` must be passed explicitly when launching each eval job, not
read from a shared config file that all four jobs happen to share.

**Re-run command** (once the eval script is fixed):
```bash
for enc in conv labram biot eegpt; do
  sbatch --reservation=Vivatech --account=vivatech-slightlyunawarefc \
    eval_predictive.sh \
    /lustre/work/vivatech-slightlyunawarefc/yhammache/checkpoints/eeg/dev_2026-06-20_03-33/${enc}_predictive_seed0
done
```

---

## Suggestion — What a proper naive JEPA baseline looks like (and why ours is not it)

### What we actually ran (`EEGPredictiveJEPA`)

The current `objective=predictive` mode is **not** a standard JEPA baseline. It is:
- Two **augmented views** (v1, v2) of the **same 10-second window**
- Online encoder encodes v1 frame-by-frame; EMA target encoder encodes v2 frame-by-frame
- GRU predictor rolls forward inside v1 and is asked to match v2's frame embeddings at each step
- Anti-collapse via hinge-std + covariance loss (VICReg primitives)

This conflates two ideas: **temporal prediction** (GRU unrolling) and **two-view invariance**
(matching v1 to v2). Because v1 and v2 span the *same time*, the "prediction" is really
augmentation-invariance in disguise, and the GRU has no real temporal task to solve. The
near-zero `std_loss` throughout training suggests the encoder found a degenerate solution: output
near-constant vectors that the GRU can easily "predict" with low MSE.

### What a genuine naive JEPA baseline would be

A minimal, honest JEPA baseline follows I-JEPA (Assran et al., 2023) applied to EEG:

1. **No augmentations.** Take a single raw EEG window. Split it temporally: frames 0..K-1
   are the context, frames K..F-1 are the target. The asymmetry between context and target
   is what prevents collapse — no VICReg-style terms needed.
2. **Online encoder** sees only the context frames and produces context embeddings.
3. **EMA target encoder** (momentum ≈ 0.996, same as now) sees only the target frames and
   produces target embeddings. No gradient flows into the target encoder.
4. **Predictor** (can be a simple MLP or shallow Transformer, not a GRU) takes context
   embeddings + positional information about *which* target frames to predict, and outputs
   predicted target embeddings.
5. **Loss**: MSE between predicted and EMA-target embeddings. Nothing else.

```python
class NaiveJEPA(nn.Module):
    """Minimal I-JEPA-style baseline for EEG. No augmentations, no anti-collapse term."""

    def __init__(self, encoder, context_ratio=0.5, ema=0.996):
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.ema = ema
        D = encoder.out_dim
        # Predictor: takes context avg-pool + target position embedding → target embedding
        self.predictor = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.context_ratio = context_ratio

    @torch.no_grad()
    def _update_target(self):
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.mul_(self.ema).add_(op, alpha=1 - self.ema)

    def compute_loss(self, x):
        # x: [B, C, T] raw EEG window
        self._update_target()
        F = x.shape[-1]
        split = int(F * self.context_ratio)
        context_frames = self.online.frames(x[..., :split])   # [B, K, D]
        with torch.no_grad():
            target_frames = self.target.frames(x[..., split:]) # [B, F-K, D]
        ctx = context_frames.mean(1)                            # [B, D] global context
        preds = self.predictor(ctx).unsqueeze(1).expand_as(target_frames)
        return F.mse_loss(preds, target_frames)
```

### Why this is more "basic" than everything else we ran

| Property | Two-view VICReg | Two-view SIGReg | `EEGPredictiveJEPA` (ours) | Naive JEPA |
|---|---|---|---|---|
| Augmentation design | required | required | required | **none** |
| Anti-collapse term | VICReg (3 coefficients) | SIGReg (2 coefficients) | hinge-std + cov | **none needed** |
| Collapse mechanism | regularizer | regularizer | regularizer (failed) | masking asymmetry |
| Prediction task | invariance to corruption | invariance to corruption | forward in time (but same span) | **future frames** |
| Hyperparameters | 3 loss weights | 4 loss weights | 3 loss weights + EMA | **1 (EMA)** |
| Failure mode | coeff bugs (Bug 1) | coeff bugs | collapse (Bug 3) | encoder ignores context |

The naive JEPA has the fewest design choices and the fewest ways to silently misconfigure.
Collapse is prevented architecturally (context and target never see the same frames), so the
`std_loss ≈ 0` failure mode of our current predictive objective cannot occur. It is the honest
"can a predictive objective even work here?" baseline before adding any of the complexity we
already have.

**Why we did not start here**: the EEGPredictiveJEPA was written to reuse the existing
two-view data pipeline (which already produces (v1, v2) pairs). A masking-based JEPA
requires a different data path (single-view, split by time index), which is a small but
non-trivial change to `EEGConfig` and `make_loader`.
