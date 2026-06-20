# EEG - self-supervised representation learning on TUH EEG (abnormality detection)

**Question.** Can *two-view invariance learning* on **unlabeled** EEG learn features
that linearly separate **normal vs abnormal** recordings, and generalize to
**held-out (patient-disjoint) subjects**?

## Data
TUH Abnormal EEG corpus, preprocessed: raw `.edf`, **19 channels** (standard 10-20
montage, fixed order) **@ 200 Hz**, per-channel z-scored. Patient-disjoint
`train` / `eval` splits, each with `normal` / `abnormal` recordings. Lives at
`/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED`.
Read directly from EDF with `pyedflib` (partial window reads) - no prep step.

## Layout
```
eb_jepa/datasets/eeg/   dataset.py (provided EDF loader) + data_config.yaml
examples/eeg/
  main.py     SSL pretraining - TODO: build_encoder() + build_ssl()
  eval.py     patient-disjoint probe - TODO: probe() + metric
  cfgs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` - a 1D encoder over `[B, 19, T]` (`represent() -> [B, D]`,
   and `frames()` if you go predictive).
2. `main.py:build_ssl` - the SSL objective: two-view VICReg (`Projector` +
   `VICRegLoss`, the natural choice - the dataset already returns two views)
   **or** predictive JEPA (eb_jepa `RNNPredictor` + EMA target + `VCLoss`).
3. `eval.py:probe` - the **patient-disjoint** frozen-feature probe + metric
   (`LogisticRegression`; accuracy / balanced-acc / AUROC), compared to a
   random-encoder floor and a supervised end-to-end baseline.

Everything else (EDF loading, two-view training loop, recording-level feature
extraction) is provided.

## Run
```bash
python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
```

## Extension - TUEV (the "hard" one)
TUAB is recording-level binary (normal vs abnormal). The harder variant is **TUEV**
(TUH EEG Events): **6-class**, **second-level** event labels
(`SPSW, GPED, PLED, EYEM, ARTF, BCKG`), a tiny + massively imbalanced corpus. The
same per-frame encoder feeds a temporal model; the probe becomes a patient-disjoint
**6-class** classifier (macro-F1 / macro-AUROC, fighting the background imbalance).
A natural follow-up once the binary TUAB probe works.
