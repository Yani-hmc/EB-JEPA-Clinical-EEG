# Datasets - TUAB + TUEV (TUH EEG Corpus)

Both datasets come from the **Temple University Hospital EEG Corpus (TUH-EEG)**,
a publicly available collection of clinical scalp-EEG recordings. They share
the same 19-channel 10-20 montage and the same per-subject train/eval split,
but differ in task:

| | **[TUAB](TUAB/)** | **[TUEV](TUEV/)** |
|---|---|---|
| Full name | TUH Abnormal EEG Corpus (TUAB v3.0.1) | TUH EEG Events Corpus (TUEV v2.0.0) |
| Task | Binary: normal vs abnormal recording | 6-class events: SPSW, GPED, PLED, EYEM, ARTF, BCKG |
| Sample size | 2717 train + 276 eval recordings (patient-disjoint) | 83,932 + 28,559 event windows (subject-disjoint) |
| Granularity | 1 label per recording (~30 min) | 1 label per labeled second (event-centered 5 s window) |
| Headline metric | Balanced Accuracy / AUROC, per-recording (16-window mean-pool) | Balanced Accuracy / Cohen-κ / Weighted-F1, per-window |

Obtaining the raw data requires a data-use agreement with TUH (`isip.piconepress.com/projects/tuh_eeg/`).

Once you have the raw EDFs, the per-dataset README explains the preprocessing
recipe we used. **TUAB** is read directly from the EDFs by
[`eb_jepa/datasets/eeg/dataset.py`](../eb_jepa/datasets/eeg/dataset.py)
(no separate preprocessing pickle). **TUEV** is preprocessed with BIOT's
official [`process.py`](TUEV/process.py) (vendored here) into one `.pkl`
event-window per labeled second.

---

## Where the literature baseline numbers come from

For each dataset, the corresponding `supervised_baselines_provenance.md` lists
**every literature baseline we cite**, with exact paper title + GitHub URL +
"did we re-run it on OUR pkls or cite paper?":

- [`TUAB/supervised_baselines_provenance.md`](TUAB/supervised_baselines_provenance.md) - 10 baselines on TUAB
- [`TUEV/supervised_baselines_provenance.md`](TUEV/supervised_baselines_provenance.md) - 10 baselines on TUEV (7 re-run, 3 cited)
