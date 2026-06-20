# Scientific Decision Log

Records every genuine scientific decision: what was decided, why, what assumption it
rested on, and what would have falsified it. Distinct from `BUGS_AND_MISTAKES.md`
(which records errors) and `EXPERIMENTS.md` (which records outcomes).

A decision is genuine if: two reasonable people could disagree on the right answer
before seeing results, and the wrong answer wastes a training run or corrupts a claim.

Use `/science-check` before adding a new entry.

---

## Decisions made

### DM1 — Corruption augmentation design
**Date**: early hackathon
**Decision**: add time masking (20%) + spike injection (±6σ, 20% of windows) on top of the existing noise/jitter/channel-drop stack.
**Assumption**: EEG artifact robustness transfers to pathology detection. Making the encoder survive corrupted views forces it to learn stable, low-frequency structure.
**Falsification criterion**: if corrupted views trained no better than noise/jitter alone, the artifact hypothesis was wrong.
**Actual outcome**: +0.014 BACC per-window, +0.029 per-recording. 3-seed std 0.004.
**What we didn't do**: never ablated spike vs masking separately. Don't know which component drove the gain.
**Verdict**: decision was correct, but reasoning was never verified.

### DM2 — Conv as primary ablation encoder
**Date**: early hackathon
**Decision**: run tvasnier's entire ablation campaign (spectral, corruption, scaling, fine-tuning) on the conv encoder only.
**Assumption**: conv is the best encoder, so ablating on it gives the most signal.
**Falsification criterion**: if another encoder was systematically better across all ablations, the ablation campaign would be encoder-specific rather than general.
**Actual outcome**: conv was indeed the best encoder. Assumption was correct.
**What we didn't do**: never ran corruption or spectral ablations on LaBraM-style. Don't know if the gains transfer.

### DM3 — VICReg fix: default inv_coeff to 1.0 (backward compatible)
**Date**: mid hackathon
**Decision**: when adding `inv_coeff` parameter to `VICRegLoss`, default it to 1.0 (old behavior) rather than 25.0 (correct paper value). Explicit override in train.yaml.
**Assumption**: backward compatibility matters — other code or configs might rely on the old (1,1,1) behavior.
**Falsification criterion**: if nothing uses VICRegLoss without explicit yaml overrides, the default is irrelevant.
**Verdict**: safe choice. No downstream breakage. Arguably the default should be 25.0 to prevent the bug recurring, but that would silently change existing run behavior.

### DM4 — Accept per-recording as clinical metric, not retract it
**Date**: after per-window discovery
**Decision**: keep per-recording numbers in the report (labeled clearly) rather than retracting them entirely.
**Assumption**: per-recording is a valid metric for clinical deployment evaluation, even if not the benchmark convention.
**Falsification criterion**: if the paper reviewers or jury considers per-recording dishonest or misleading even when labeled, the decision was wrong.
**Reasoning**: per-recording is arguably the correct clinical metric (classify patients, not 10s windows). Hiding it would make the report less complete.
**Verdict**: correct — the dual-table presentation (per-window + per-recording, clearly labeled) is more informative than either alone.

---

## Decisions pending (open)

These require `/science-check` before proceeding. See `suggestions/URGENT/scientific_decisions.md` for full context.

| ID | Decision | Triggered by | Status |
|----|---------|-------------|--------|
| D1 | EEGPT channel-drop conflict — one cause or two? | jobs 75752 + 75806 results | Waiting |
| D2 | Which variable explains the LaBraM gap? (tokenizer / scale / duration) | — | Open |
| D3 | Probe design: MLP vs LogReg — which to report? | — | Open |
| D4 | Submit spectral + VICReg fix combination? | — | Open |
| D5 | Define "EEGPT rehabilitated" threshold before reading results | jobs 75752 + 75806 | Waiting |
| D6 | Revise capacity claim based on mc_transf result | job 75792 result | Waiting |

---

## The meta-decision: what this project actually did

Most of the work was **diagnostic, not creative**:
- Noticed VICReg bug → fixed it (no decision, just correct the error)
- Noticed per-window vs per-recording discrepancy → corrected it (no decision)
- Noticed EEGPT below random → diagnosed cause (no decision, just tracing)
- Noticed BIOT is bad for temporal tasks → explained it (no decision, just reasoning)

**The only genuine creative decisions** were:
1. Corruption augmentation design (DM1) — a real bet
2. Spectral regularisation sweep (tvasnier) — methodical search
3. Accepting per-recording as a valid secondary metric (DM4)

**The pattern**: we were good at noticing things, accurate in diagnosing causes, and honest in correcting mistakes. We were not particularly bold in making architectural or methodological bets. The remaining open decisions (D1–D6) are the first places where real scientific choice — not just error-correction — is required.
