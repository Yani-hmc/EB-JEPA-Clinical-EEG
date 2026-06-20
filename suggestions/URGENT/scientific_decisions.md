# Open Scientific Decisions

**Status**: these are not bugs to fix or observations to record — they are genuine
forks in the road where the right answer is not known in advance and where choosing
wrong wastes a training run and corrupts a claim in the report.

Use `/science-check` before acting on any of these.

---

## D1 — EEGPT channel-drop ablation ← ACTIVE (jobs running)

**What's running**:
- Job 75752 `eegpt_fix2`: VICReg(25,25,1) + chan_drop=0.2 (fix only)
- Job 75806 `eegpt_nodrop`: VICReg(25,25,1) + chan_drop=0.0 (fix + remove conflict)

**The decision**: when results arrive, you must decide whether the channel-drop/channel-pool
conflict was a real second cause of collapse, or whether the VICReg coefficient bug was
the whole story.

**Why it's hard**: three outcomes are possible, each implying a different next step:

| Outcome | Interpretation | Next action |
|---------|---------------|-------------|
| nodrop >> fix2 (>0.02 BACC) | Conflict is real and major | Disable chan_drop per-encoder; redesign augmentation stack |
| nodrop > fix2 (0.005–0.02) | Conflict is real but minor | Note it; probably not worth redesigning augmentation |
| nodrop ≈ fix2 (<0.005) | VICReg bug was the whole story | EEGPT is compatible with chan_drop; no redesign needed |

**Pre-registration** (fill in before reading results):
```
I predict outcome: ___
My falsification threshold: if gap < ___ BACC, I conclude the conflict is not meaningful.
If outcome 3 (nodrop ≈ fix2), I will: ___
```

**What goes in the report**: the conclusion of this ablation directly determines
whether Section 6.2 (EEGPT Collapse) claims one cause or two.

---

## D2 — What explains the 3.9pp gap to LaBraM-Base?

**The gap**: our best per-window BACC is 0.775; LaBraM-Base is 0.814.

**Three candidate causes** (fully confounded — we have no data to separate them):
1. **Tokenizer quality**: LaBraM uses a neural tokenizer (masked patch prediction pretraining). We use raw linear projection.
2. **Model scale**: LaBraM-Base is 5.8M params; our LaBraM-style encoder is 1.2M.
3. **Pretraining duration**: LaBraM pretrained ~200 epochs on a larger dataset. We used 20 epochs.

**The decision**: which variable to isolate first?

Each requires a separate 20–40 epoch training run. Testing all three would take the rest of the hackathon. **You must choose one.** The choice reveals your theory of the gap.

| If you test... | You believe... | Risk if wrong |
|---------------|---------------|--------------|
| Tokenizer | Quality of patch embeddings is the bottleneck | Neural tokenizer training is expensive; if it doesn't help, you've spent a run on the wrong variable |
| Scale | More parameters = better features | If scale doesn't help, the claim "capacity is the constraint" (already in the report) is weakened |
| Duration | 20 epochs is too few | If longer training doesn't help, the report's "open questions" section loses its main hope |

**Not deciding is a decision**: accepting 0.775 as our ceiling and not explaining why we're
below LaBraM. That's defensible if framed honestly.

---

## D3 — Probe design: MLP vs LogReg

**Current state**: every result in the report used an MLP probe. The SSL community standard
(SimCLR, MoCo, BYOL) uses scikit-learn LogisticRegression(C=0.316, max_iter=1000).

**Why this matters**: MLP probe is more generous (it learns nonlinear separability).
LogReg is a stricter test of whether features are linearly separable — the theoretically
correct test for SSL quality. If you switch and rankings change, you have a problem.

**The decision**: do you add LogReg as a second probe and report both? Or commit to MLP?

| Choice | Implication |
|--------|------------|
| Report MLP only | Cannot directly compare to SimCLR/BYOL numbers; must caveat |
| Report LogReg only | Retract all existing numbers and rerun everything |
| Report both | Need to explain why they differ; adds complexity to the report |
| Report MLP, note the discrepancy | Honest and practical; weakens claims slightly |

**Stakes**: if LogReg shows conv_corrupt BACC drops from 0.770 to 0.750, our comparison
to ContraWR (0.775) would flip from "tied" to "below."

---

## D4 — Spectral regularisation + VICReg fix: submit or accept confound?

**The confound**: `spec_fixed` (per-recording BACC 0.836, our best checkpoint) was trained
with VICReg inv_coeff=1 (the bug). The spectral loss appears to have partially compensated
for the underweighted invariance.

**The question**: does `spec_fixed` + inv_coeff=25 produce a better encoder, or does the
spectral loss conflict with the stronger invariance signal?

**Two hypotheses**:
- H1: spectral loss adds independent signal → spec_fixed + fix should beat both alone
- H2: spectral loss was compensating for the bug → removing the bug makes it redundant or harmful

**The decision**: submit the run? If H2 is true and spec + fix is *worse* than fix alone,
this is a negative result that should go in the report but complicates our "best checkpoint"
narrative.

**If you don't submit**: `spec_fixed` remains permanently confounded and you cannot
claim the spectral regularisation is a genuine contribution independent of the bug.

---

## D5 — EEGPT rehabilitation threshold

**Triggered by**: results from jobs 75752 and 75806.

**The claim to make or not make**: "EEGPT encoder is now competitive after fixing the
VICReg coefficients (and optionally removing channel-drop)."

**You must define "competitive" before reading results.** Options:
- Competitive = above random floor (>0.674 accuracy) — low bar, almost certainly met
- Competitive = within 0.02 of conv encoder — medium bar
- Competitive = above LaBraM-style encoder — high bar

**Why this matters for the report**: Section 6.2 currently says EEGPT "trained below random
init." If the fix brings EEGPT to 0.80 per-recording, that sentence becomes the setup for
a much stronger result. If it only reaches 0.65, the fix helped but EEGPT is still architecturally
limited — a different conclusion.

---

## D6 — What mc_transf tells us about the capacity claim

**Job**: tvasnier's 75792 `mc_transf` (3.65M transformer, multi-corpus pretraining)

**Current claim in the report** (Section 5, ablation):
> "Scaling did not help (150 ep, deeper encoder): per-window flat at 0.768;
> the bottleneck is encoder capacity, not training time."

**Problem**: that conclusion was based on scaling a Conv encoder. `mc_transf` is a different
architecture at a different scale. Its result either:

- Confirms the claim (mc_transf ≈ conv) → "architecture doesn't matter at this scale"
- Breaks the claim (mc_transf >> conv) → "scaling DOES help, but only with better architecture"
- Partially breaks it (mc_transf > conv but < LaBraM) → "architecture matters but capacity
  still isn't the only factor"

**The report implication**: the ablation table and conclusion section may need to be revised
depending on mc_transf's result. Do not finalize those sections before this run completes.

---

## How to use this document

1. Before running any experiment related to D1–D6: use `/science-check`
2. Before writing conclusions about any of these in the report: answer questions 1–4 from the skill
3. After results arrive: fill in the pre-registration fields above, then document actual outcome
4. If actual outcome ≠ predicted: write down why in `BUGS_AND_MISTAKES.md`, section "Things we got wrong in our predictions"
