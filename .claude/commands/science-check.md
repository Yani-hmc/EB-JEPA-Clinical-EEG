---
description: Force explicit scientific decision-making before running experiments, drawing conclusions, or writing claims. Stops reactive "notice and fix" mode and requires the user to own the reasoning.
---

A scientific decision is about to be made. Do NOT proceed — do not submit jobs, write conclusions, update the report, or implement anything — until the user has answered all four questions below for the specific decision at hand.

# Protocol

Identify the decision being requested. Then ask the user these four questions explicitly, one at a time, waiting for a real answer to each:

**1. What exactly are you deciding?**
State it as a binary or multi-way choice between concrete alternatives. "Run EEGPT without channel-drop" is not a decision — "I am choosing between: (A) disable channel-drop for EEGPT only, (B) disable it for all encoders, (C) keep it and accept EEGPT is architecturally incompatible" IS a decision.

**2. What assumption does this rest on?**
Every experiment tests an assumption. Make it explicit. If the assumption is wrong, the result is uninterpretable. Example: "I assume the channel-drop/channel-pool conflict is the SECOND cause of EEGPT collapse after the VICReg bug. If EEGPT still collapses without channel-drop, this assumption was wrong."

**3. What result would falsify your hypothesis?**
Before seeing results, define what "this didn't work" means numerically. Example: "If eegpt_nodrop BACC < eegpt_fix2 BACC + 0.01, the channel-drop conflict is not a meaningful cause." Pre-registration prevents post-hoc rationalization.

**4. What will you do if the result is unexpected?**
Unexpected = outcome that doesn't fit either your hypothesis OR its negation cleanly. Example: "If eegpt_nodrop is better on per-window but worse on per-recording, I will..." — have an answer before running.

---

# Open scientific decisions in this project (as of June 2026)

These are the decisions that remain unresolved. Reference them when relevant:

## D1 — EEGPT channel-drop ablation
**Jobs running**: 75752 (fix2, drop=0.2) and 75806 (nodrop, drop=0.0)
**Decision triggered when results arrive**: interpret the gap and decide whether to standardize augmentation per-encoder.
**Pre-registration required**: define the minimum BACC difference that constitutes "the conflict matters" before reading the logs.

## D2 — Tokenisation: learned vs raw vs FFT
**Status**: untested. Our 3.9pp gap to LaBraM-Base could be tokenizer quality, model scale, or pretraining length — these are confounded.
**Decision**: which variable to isolate first? Testing all three requires 3 separate training runs and 3 weeks. Choose one.
**What the choice reveals about your theory of the gap.**

## D3 — Probe design: MLP vs LogReg
**Status**: we always used MLP. LogReg is the SSL community standard.
**Decision**: if you switch to LogReg and rankings change, which results do you report? You cannot report both without explaining why they differ.
**Stakes**: changes whether you can directly compare to SimCLR/MoCo literature.

## D4 — Spectral + VICReg fix combination
**Status**: spec_fixed (0.836 per-recording) was trained WITH the VICReg bug. Never retrained with inv_coeff=25.
**Decision**: submit the run or accept that spec_fixed's number is permanently confounded?
**If you submit and it's worse**: the spectral loss was compensating for the bug, not adding independent signal. That's a negative result worth reporting.

## D5 — EEGPT rehabilitation claim
**Triggered when**: eegpt_fix2 and eegpt_nodrop results arrive.
**Decision**: at what BACC does EEGPT become "rehabilitated"? If it reaches 0.75 per-recording, do you claim success? What about 0.70? Define the threshold before reading results.

## D6 — What the mc_transf result means for the capacity claim
**Job**: tvasnier's 75792 mc_transf (3.65M transformer, multi-corpus)
**Current claim**: "encoder capacity is the binding constraint" — based on Conv scaling experiments.
**Decision**: if mc_transf < conv_corrupt (0.825 per-recording), is the claim "capacity doesn't matter" or "this specific architecture at this scale doesn't help"? These have different implications for future work.

---

# What this skill is NOT for

- Noticing bugs (no decision, just fix them)
- Fixing implementation errors (VICReg coefficients, eval protocol)
- Running evals on already-trained checkpoints (no scientific decision, just execute)
- Writing code that implements an already-decided design

# Ground rule

If the user cannot answer question 3 (falsification criterion) before seeing results, the experiment should not be run yet. Premature experiments produce results you can rationalize in any direction.
