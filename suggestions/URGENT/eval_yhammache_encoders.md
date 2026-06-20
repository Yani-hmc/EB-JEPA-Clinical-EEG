# ⚠️ URGENT — Run Eval on yhammache's 8 Checkpoints

## The gap

yhammache ran all 8 encoder × SSL loss combinations. **None of them have eval results.**
These are the main competition encoders (labram, eegpt, biot, conv × vicreg/sigreg)
and we have no BACC numbers for them. We are flying blind.

## Checkpoints to evaluate

```
/lustre/work/vivatech-slightlyunawarefc/yhammache/checkpoints/eeg/

dev_2026-06-20_00-17/labram_sigreg_seed0/latest.pth.tar
dev_2026-06-20_00-17/biot_sigreg_seed0/latest.pth.tar
dev_2026-06-20_00-17/eegpt_sigreg_seed0/latest.pth.tar
dev_2026-06-20_00-16/conv_sigreg_seed0/latest.pth.tar

dev_2026-06-19_23-54/labram_vicreg_seed0/latest.pth.tar   ← use this session (latest)
dev_2026-06-19_23-54/biot_vicreg_seed0/latest.pth.tar
dev_2026-06-19_23-54/eegpt_vicreg_seed0/latest.pth.tar
dev_2026-06-19_23-54/conv_vicreg_seed0/latest.pth.tar
```

(dev_2026-06-19_23-35 is an earlier session, prefer dev_2026-06-19_23-54 for vicreg)

## How to run (submit all 8 in parallel)

```bash
BASE=/lustre/work/vivatech-slightlyunawarefc/yhammache/checkpoints/eeg

for ckpt in \
  dev_2026-06-20_00-17/labram_sigreg_seed0/latest.pth.tar \
  dev_2026-06-20_00-17/biot_sigreg_seed0/latest.pth.tar \
  dev_2026-06-20_00-17/eegpt_sigreg_seed0/latest.pth.tar \
  dev_2026-06-20_00-16/conv_sigreg_seed0/latest.pth.tar \
  dev_2026-06-19_23-54/labram_vicreg_seed0/latest.pth.tar \
  dev_2026-06-19_23-54/biot_vicreg_seed0/latest.pth.tar \
  dev_2026-06-19_23-54/eegpt_vicreg_seed0/latest.pth.tar \
  dev_2026-06-19_23-54/conv_vicreg_seed0/latest.pth.tar; do
    tag=$(echo $ckpt | grep -oP '(labram|biot|eegpt|conv)_(vicreg|sigreg)')
    sbatch --reservation=Vivatech --gres=gpu:1 -J "eval_$tag" \
      --wrap="python -m examples.eeg.eval --ckpt $BASE/$ckpt --random-floor"
done
```

## Expected runtime: ~2 min per eval, all 8 in parallel = ~2 min total

## What to report

For each checkpoint: BACC (logreg), BACC (MLP), AUROC, random floor BACC.
Then we have a 4×2 table of encoder × SSL loss — the core competition result.

## Also run tvasnier's best (fftc0p3)

```bash
sbatch --reservation=Vivatech --gres=gpu:1 -J "eval_fftc0p3" \
  --wrap="python -m examples.eeg.eval \
    --ckpt /lustre/work/vivatech-slightlyunawarefc/tvasnier/checkpoints/eeg/fftc0p3/latest.pth.tar \
    --random-floor"
```

This is the checkpoint that produced BACC 0.826 (MLP) — confirm it with random floor.
