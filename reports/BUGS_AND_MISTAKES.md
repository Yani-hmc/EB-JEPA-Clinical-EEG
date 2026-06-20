# Bugs, Mistakes, and Corrections

A complete record of everything that went wrong, why, and how we fixed it.
This is the document you want when comparing the report's claims to reality.

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
