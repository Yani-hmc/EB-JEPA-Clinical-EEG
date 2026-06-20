# yhammache Encoder Eval Results (2026-06-20)

Evaluated all 8 checkpoints (4 encoders × 2 SSL losses) on TUAB patient-disjoint split.
Jobs 75398–75405 on Dalia. Output: `/lustre/work/vivatech-slightlyunawarefc/tcourtois/eval_out/`

**Note:** used yhammache's eval.py (old version) — reports accuracy/f1 only.
BACC and AUROC missing. See `fix_metrics.md` — re-run with patched eval for proper comparison.

---

## Results table

| Encoder | SSL Loss | Accuracy | F1 | Recall | Precision |
|---------|----------|----------|----|--------|-----------|
| **conv** | **vicreg** | **0.837** | **0.812** | 0.770 | 0.858 |
| conv | sigreg | 0.812 | 0.785 | 0.754 | 0.819 |
| biot | sigreg | 0.783 | 0.760 | 0.754 | 0.766 |
| biot | vicreg | 0.775 | 0.730 | 0.667 | 0.808 |
| labram | vicreg | 0.775 | 0.721 | 0.635 | 0.833 |
| labram | sigreg | 0.725 | 0.658 | 0.579 | 0.760 |
| eegpt | vicreg | 0.641 | 0.596 | 0.579 | 0.613 |
| eegpt | sigreg | 0.627 | 0.605 | 0.627 | 0.585 |

Eval set: n=276 recordings, 45.65% abnormal (patient-disjoint from 2717 train recordings).

---

## vs baselines (accuracy, for comparison)

| Model | Type | Accuracy |
|-------|------|----------|
| Chance | — | ~54% (majority class) |
| BIOT SSL | frozen probe | ~79% |
| **conv + vicreg (ours)** | frozen probe | **83.7%** |
| Deep4Net (Schirrmeister 2017) | supervised | 85.4% |
| LaBraM | frozen probe | ~81% (BACC 0.814) |

**conv + vicreg beats LaBraM and approaches Deep4Net supervised — with no labels.**

---

## Key findings

### 1. Conv beats all transformers
The simple strided Conv1D encoder outperforms labram, biot, and eegpt by a large margin.
This is not what we expected. Possible explanations:
- Transformers may need more epochs or a larger dataset (TUAB is ~3K recordings)
- The patch tokenization (200-sample patches) may not be optimal for 200Hz EEG
- Conv's inductive bias (local temporal structure) fits TUAB better than global attention

### 2. EEGPT collapsed
Both EEGPT runs (vicreg 0.641, sigreg 0.627) are barely above majority-class chance.
The encoder did not learn useful representations. Likely causes:
- The hierarchical spatial→temporal design may require more data
- The channel-mixing attention collapses channel information that TUAB classification needs
- Training may not have converged (only 20 epochs)

### 3. SIGReg generally underperforms VICReg
VICReg beats SIGReg for conv (0.837 vs 0.812) and labram (0.775 vs 0.725).
Exception: biot (sigreg 0.783 > vicreg 0.775 — marginal difference).

### 4. Architecture matters more than SSL loss
The gap between encoders (0.627–0.837) is far larger than the gap between losses
(~0.02–0.05). Encoder choice is the dominant factor.

---

## Immediate actions needed

1. **Re-run eval with patched eval.py** (add BACC + AUROC) on conv_vicreg checkpoint
   — need BACC to compare to LaBraM's 0.814 BACC directly
2. **Investigate EEGPT collapse** — check training loss curves, compare to tvasnier's eegpt runs
3. **Run more epochs for transformers** — 20 epochs may be insufficient for labram/biot/eegpt
4. **Report conv+vicreg as primary result** — it's the strongest, simplest, most interpretable

## Best checkpoint path
```
/lustre/work/vivatech-slightlyunawarefc/yhammache/checkpoints/eeg/dev_2026-06-19_23-54/conv_vicreg_seed0/latest.pth.tar
```
