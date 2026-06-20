# TUAB - TUH Abnormal EEG Corpus (v3.0.1)

**Binary classification**: is the recording **normal** or **abnormal**?

## Source

[Temple University Hospital EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/) - register
to obtain the v3.0.1 dump. We use the canonical split:

| split | recordings | patients (disjoint!) |
|---|---|---|
| train | 2,717 | 2,717 |
| eval  | 276 | 276 |

`train` and `eval` share **no patient** - this is what makes the eval BAcc a
real generalization number. Every recording has a single label
(`normal` / `abnormal`); the prevalence in eval is 43% abnormal / 57% normal.

## Preprocessing (we apply on-the-fly)

| step | value |
|---|---|
| Channels | 19 (10-20 montage), in fixed order |
| Sampling rate | 200 Hz |
| Window length | 10 s (= 2000 samples) |
| Windows per recording (eval) | 16 evenly-spaced, mean-pooled at probe time |
| Windows per recording (SSL train) | random crop, `epoch_size=20000` random windows / epoch |
| Per-channel z-score | yes (computed on the window) |

All of this lives in [`eb_jepa/datasets/eeg/dataset.py`](../../eb_jepa/datasets/eeg/dataset.py)
(class `EEGDataset` + `make_loader`). There is **no separate preprocessing
script** - the EDFs are read directly via `pyedflib`.

Set the data root to your local TUAB tree (e.g. on DALIA:
`/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED`).
This path is configured in [`examples/eeg/cfgs/train.yaml`](../../examples/eeg/cfgs/train.yaml).

## Two-view corruption (for SSL)

When training EB-JEPA, each window is loaded twice with **two independent
corruptions** to make the views `v₁, v₂` that VICReg / SIGReg pulls together:

| corruption | strength | knob |
|---|---|---|
| Gaussian noise | σ = 0.1 | `data.aug_noise_std` |
| Per-channel scale-jitter | ±20% (set to 0 in the robustness-table tuned model) | `data.aug_scale_jitter` |
| Channel-drop | p = 0.2 | `data.aug_chan_drop_p` |
| Time-mask (legacy contiguous) | frac = 0.2 | `data.aug_time_mask_frac` |
| **Exact EB-corruption** (opt-in) | 20% timepoints masked + 20% ±6σ outliers (~40%/view) | `data.aug_exact_corruption=true` + `aug_mask_frac` / `aug_outlier_frac` / `aug_outlier_scale` |

The "baseline" rows use only the legacy contiguous time-mask; the "+corrupt"
rows turn on `aug_exact_corruption=true`.

## Scoring conventions

- **per-recording BAcc / AUROC** - 16 windows mean-pooled to one prediction per
  recording. This is the **clinical metric** (what an OEM would report).
- **per-window BAcc / AUROC** - predict each 10s window independently. This is
  the **TUAB-literature convention** (BIOT, LaBraM, SPaRCNet tables).

Our [`reference_table_JEPA_V1.md`](../../TUAB_experiments/reference_table_JEPA_V1.md)
reports BOTH; the headline number in talks is per-recording.

## Supervised baselines we compare against

→ See [`supervised_baselines_provenance.md`](supervised_baselines_provenance.md).
10 rows: LaBraM-{Base,Large,Huge}, BIOT vanilla, BIOT pretrained, SPaRCNet,
ContraWR, CNN-T, FFCL, ST-T, EEGNet, ShallowConvNet - each with paper title,
GitHub URL, pretrained-weights provenance, and our-pipeline scores.
