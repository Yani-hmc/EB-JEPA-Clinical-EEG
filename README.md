# EB-JEPA on EEG - Clean Reproduction Bundle

This repository organises everything you need to **reproduce the EB-JEPA results**
on the two TUH EEG benchmarks (TUAB normal/abnormal binary, TUEV 6-class event
detection). It is divided into **four sections**:

| Section | What's there |
|---|---|
| [`dataset/`](dataset/) | TUH dataset prep - what TUAB and TUEV are, where to obtain them, and how we preprocess each one. |
| [`TUAB_experiments/`](TUAB_experiments/) | One shell script per row of [`reference_table_JEPA_V1.md`](TUAB_experiments/reference_table_JEPA_V1.md). Plus a [`robustness/`](TUAB_experiments/robustness/) subdir for hyperparameter / ablation experiments matching [`reference_table_robustness_V1.md`](TUAB_experiments/robustness/reference_table_robustness_V1.md). |
| [`TUEV_experiments/`](TUEV_experiments/) | One shell script per row of [`reference_table_JEPA_TUEV_V1.md`](TUEV_experiments/reference_table_JEPA_TUEV_V1.md) - TUAB-pretrained encoder, frozen, probed on TUEV. |
| [`eb_jepa/`](eb_jepa/) + [`examples/eeg/`](examples/eeg/) | The actual EB-JEPA codebase - Python package, encoder, losses, training loop. The shell scripts above call into it. |

---

## How we used external GitHub code to get our scores

For the EB-JEPA results, **the code in this repo is ours** - Conv1D / Transformer
encoders, VICReg / SIGReg losses, two-view corruption, the spectral term, all
written in [`examples/eeg/main.py`](examples/eeg/main.py) and
[`eb_jepa/`](eb_jepa/). It is descended from the EB-JEPA reference implementation
at [`github.com/Trick5t3r/eb_jepa`](https://github.com/Trick5t3r/eb_jepa)
(commit imported in `ec9a0f6`); we kept the package layout and added the EEG
example. **No pretrained EB-JEPA weights** - every row was trained from scratch
on TUAB SSL.

For the **literature baselines we compare against** (SPaRCNet, ContraWR,
CNN-Transformer, FFCL, ST-Transformer, BIOT, LaBraM, EEGNet, ShallowConvNet -
see [`dataset/TUAB/supervised_baselines_provenance.md`](dataset/TUAB/supervised_baselines_provenance.md)
and [`dataset/TUEV/supervised_baselines_provenance.md`](dataset/TUEV/supervised_baselines_provenance.md)),
**we used the official authors' code** unmodified:

| Baseline group | Official GitHub | What we used |
|---|---|---|
| SPaRCNet / ContraWR / CNN-T / FFCL / ST-T / BIOT (`run_multiclass_supervised.py` for TUEV; supervised TUAB recipe) | [`ycq091044/BIOT`](https://github.com/ycq091044/BIOT) | Model classes from `model/*.py` + official train script + official `datasets/TUEV/process.py` preprocessing. |
| BIOT pretrained | same | Official `pretrained-models/EEG-PREST-16-channels.ckpt` from the BIOT repo's git tree (no external Drive / OneDrive). |
| LaBraM-Base (literature only - we didn't re-run) | [`935963004/LaBraM`](https://github.com/935963004/LaBraM) | Cited from LaBraM Tbl 2; we did not re-run LaBraM because it needs a separate finetuning script path. |
| EEGNet / ShallowConvNet | [`braindecode/braindecode`](https://github.com/braindecode/braindecode) (PyTorch port of Lawhern / Schirrmeister) | `EEGNetv4` / `ShallowFBCSPNet` classes, our standard TUAB training recipe (20 ep AdamW lr=1e-3). |

The exact baseline scripts (BIOT/LaBraM/EEGNet/ShallowConvNet adapters + sbatch
drivers) live on a sibling branch `eeg-jepa-baseline` for historical reasons -
this `clean-v1` branch focuses on the EB-JEPA reproduction. The provenance
tables in [`dataset/{TUAB,TUEV}/supervised_baselines_provenance.md`](dataset/)
give per-row source + per-model parameter count + per-model "did we run it
on our pkls, or cite paper?".

---

## Quickstart

```bash
# 1. install the EB-JEPA package + EEG examples
uv sync

# 2. point env.sh at your dataset roots and ckpt dir (see env.sh comments)
source env.sh

# 3. reproduce the TUAB reference-table baseline
./TUAB_experiments/run_baseline.sh

# 4. reproduce a TUEV row (transfer probe)
./TUEV_experiments/run_row1_spec_corrupt.sh    # best TUEV row, 0.425 BAcc
```

Each shell script in [`TUAB_experiments/`](TUAB_experiments/) and
[`TUEV_experiments/`](TUEV_experiments/) is a 5–15 line wrapper that calls
`python -m examples.eeg.main` with the row-specific overrides + the
corresponding probe. Read the comment header in each script for the expected
metric value.

---

## Compute environment

Designed for the **DALIA GB200 cluster** (slurm); each row takes 20–60 min on
1× GB200 for SSL pretraining + a few minutes for the probe. Adapt the
`source env.sh` line and `CKPT_DIR` env var for your own setup.

---

## License

This work is released under the same license as the upstream EB-JEPA package
- see [`LICENSE.md`](LICENSE.md).

  
