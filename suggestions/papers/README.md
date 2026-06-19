# Reference Papers

Four SSL-on-EEG papers whose architectural ideas directly shape our encoder choices.
All PDFs are in this folder.

---

---

## EEGNet — `eegnet.pdf`
**EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces**
Lawhern et al., 2018 — arxiv 1611.08024

**Why it matters:** The compact supervised baseline. Tests across 4 BCI paradigms (P300, ERN,
MRCP, SMR) — each reveals different architectural weaknesses. Sharp benchmark because small
per-paradigm N makes overfitting visible and model spread is clear. Our Deep4Net baseline is
from the same family; EEGNet is the leaner alternative. If EB-JEPA frozen probe beats EEGNet
on BCI IV 2a (no labels during pretraining), the argument is complete.

---

## Spectral Audit — `spectral_audit.pdf`
**A spectral audit framework reveals task-dependent aperiodic reliance across EEG deep learning**
Bindra & Panwar, 2026 — arxiv 2606.08583

**Why it matters:** Directly shows which datasets are shortcuts. After removing 1/f spectral
slope artifacts: TUAB drops 0.07–0.13 BACC (moderately inflated), Sleep-EDF drops 0.42+
(severely inflated — largely a spectral slope task), motor imagery (BCI IV 2a) is unaffected.
6 of 7 EEG foundation models show significant aperiodic reliance. **Implication for us:**
BCI IV 2a is the sharpest second benchmark. EB-JEPA's VICReg/SIGReg regularization forces
use of the full latent space — it is architecturally designed to avoid the spectral shortcuts
this paper exposes in BENDR and masked SSL models.

---

## EEG Foundation Models Benchmark — `eeg_benchmark.pdf`
**EEG Foundation Models: Progresses, Benchmarking, and Open Problems**
Liu et al., 2026 — arxiv 2601.17883

**Why it matters:** Reviews 50 models, evaluates 12 open-source ones across 13 datasets and
9 BCI paradigms. Key finding: weak model spread — specialist models remain competitive,
larger foundation models don't clearly win, linear probing often fails. Identifies *which*
datasets separate models (BCI IV 2a, TUAB) vs which don't (CHB-MIT at 99%+ is trivial).
Meta-reference for understanding why our benchmark choices matter.

---

## LaBraM — `labram.pdf`
**Large Brain Model for Learning Generic Representations with Tremendous EEG Data**
Jiang et al., ICLR 2024 — arxiv 2405.18765

**Why it matters:** State-of-the-art frozen-probe BACC on TUAB (0.814). Two-stage masked SSL:
VQ-VAE neural tokenizer → BERT-style masked prediction. Our `LaBraMEncoder` borrows its
(channel, time-patch) tokenization + ViT backbone, but replaces the VQ codebook with EB-JEPA's
two-view VICReg/SIGReg objective. If our simpler SSL matches LaBraM's BACC, the tokenizer
complexity is not what mattered — the representation objective is.

**TUAB BACC:** 0.814 (frozen probe) | **Sleep-EDF:** ~83%

---

## BIOT — `biot.pdf`
**BIOT: Biosignal Foundation Model for Cross-Data Transfer Learning**
Yang et al., NeurIPS 2023 — arxiv 2305.10351

**Why it matters:** Introduces frequency-domain tokenization (FFT magnitude per patch)
as an alternative to raw-signal tokens. Our `BIOTEncoder` implements this idea directly.
Cross-modal transfer across EEG, ECG, EMG signals. Good TUAB BACC (0.796) despite
being a general biosignal model, not EEG-specific.

**TUAB BACC:** 0.796 (frozen probe) | **Sleep-EDF:** ~82%

---

## BENDR — `bendr.pdf`
**BENDR: Using Transformers and a Contrastive Self-Supervised Objective to Learn from Massive Amounts of EEG Data**
Kostas et al., Frontiers in Human Neuroscience 2021 — arxiv 2101.12037

**Why it matters:** The cautionary baseline. CPC-style contrastive SSL on EEG.
Collapses near chance (~55% BACC) on TUAB — the model learns 1/f spectral artifacts
rather than pathology. This failure is the core motivation for EB-JEPA's energy-based
regularization (VICReg/SIGReg): the regularizer forces the encoder to use the full
latent space, preventing the spectral shortcut that kills BENDR.

**TUAB BACC:** ~0.55 (collapses) | Teaches us what NOT to do

---

## EEGPT — `eegpt.pdf`
**EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals**
NeurIPS 2024 — no arxiv (proceedings only)

**Why it matters:** Hierarchical spatial → temporal encoding. Channel-mixing attention
first collapses 19 channel tokens per time-patch into one spatial summary, then
a temporal transformer runs over those summary tokens. Our `EEGPTEncoder` implements
this exact design. Reduces sequence length from C×P to P tokens, cutting compute
and improving temporal modelling at the cost of explicit channel information.

**Architecture:** the one most different from a flat ViT — worth comparing against LaBraM-style
