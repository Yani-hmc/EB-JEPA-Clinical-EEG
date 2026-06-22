# TUAB Experiments - EB-JEPA reference table reproduction

<p align="center">
  <img src="../assets/EEG_EB_JEPA.png" alt="EB-JEPA Architecture for EEG" width="85%">
</p>

This directory contains **one shell script per row** of our TUAB EB-JEPA
[`reference_table_JEPA_V1.md`](reference_table_JEPA_V1.md). Each script
reproduces a single row end-to-end: SSL pretraining + frozen-encoder probe.

| Script | Row | Energy / Config | Expected per-rec BAcc / AUROC |
|---|---|---|---|
| [`run_baseline.sh`](run_baseline.sh) | baseline | VICReg, base aug, no corruption | 0.796 / 0.888 |
| [`run_row1_spectral.sh`](run_row1_spectral.sh) | 1 | VICReg + spectral 0.1 + corrupt | 0.836 / 0.887 |
| [`run_row2_sigreg.sh`](run_row2_sigreg.sh) | 2 | SIGReg + corrupt | 0.825 / 0.913 |
| [`run_row3_vicreg_3seed.sh`](run_row3_vicreg_3seed.sh) | 3 | VICReg + corrupt, 3 seeds | 0.819 ± .004 / 0.900 ± .006 |
| [`run_row4_vicreg_ft_3seed.sh`](run_row4_vicreg_ft_3seed.sh) | 4 | row 3 + end-to-end FT | 0.812 ± .004 / 0.908 ± .006 |
| [`run_row5_transformer_multicorpus.sh`](run_row5_transformer_multicorpus.sh) | 5 | Transformer 3.65M + multi-corpus | 0.798 / 0.872 |
| [`run_row6_stylized_facts.sh`](run_row6_stylized_facts.sh) | 6 | row 1 + Δz/Δz²/ACF/sACF | 0.823 ± .007 / 0.900 ± .010 |

## How each script works

A script is a 5–15 line bash wrapper:

```bash
python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    <row-specific dotted overrides>

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both
```

All shared SSL knobs (data path, encoder dims, optimizer) come from
[`examples/eeg/cfgs/train.yaml`](../examples/eeg/cfgs/train.yaml). The
**only thing each script changes** is the row-specific overrides (e.g.
`model.spectral_coeff=0.1 data.aug_exact_corruption=true` for row 1).

## Common usage

```bash
# from the repo root
source env.sh
./TUAB_experiments/run_row1_spectral.sh

# or override the checkpoint dir
CKPT_DIR=/scratch/$USER/my_row1 ./TUAB_experiments/run_row1_spectral.sh
```

For SLURM on DALIA, wrap the script in a sbatch with:

```
#SBATCH --job-name=tuab_row1
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
```

## Hyperparameter tuning + ablations

→ See [`robustness/`](robustness/) for the 5+1 robustness/ablation
experiments matching
[`robustness/reference_table_robustness_V1.md`](robustness/reference_table_robustness_V1.md).

## Caveats - rows that don't fully reproduce from this branch alone

- **Row 4 (FT)** needs a pretrained row-3 checkpoint as `--init`. Run
  `run_row3_vicreg_3seed.sh` first to produce the three seed checkpoints.
- **Row 5 (multi-corpus)** needs a TUAB+TUEV+TUSZ+TUEP union preprocessing root.
  We point `MULTI_ROOT` at our DALIA path; adapt for your environment.
- **Row 6 (stylized facts)** is reproduced **without** the four path-statistics
  blocks (Δz/Δz²/ACF/sACF) - those need a Phase-6 patch on top of the loss
  function (see commit `fc5041e` for the canonical version). The script
  reproduces the closest configuration (row 1 with 3 seeds).
