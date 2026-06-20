#!/bin/bash
#SBATCH --job-name=viz_latents
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-slightlyunawarefc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/viz_latents_%j.out

PYTHON=/lustre/work/vivatech-slightlyunawarefc/tvasnier/venvs/eb_jepa_x86_64/bin/python
REPO=/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa

# Checkpoint à visualiser — modifier ici ou passer CKPT= en env
CKPT=${CKPT:-/lustre/work/vivatech-slightlyunawarefc/tvasnier/checkpoints/eeg/corrupt/latest.pth.tar}
OUT=${OUT:-/lustre/work/vivatech-slightlyunawarefc/tcourtois/viz_out/$(basename $(dirname $CKPT))}

echo "=== viz_latents | ckpt=$CKPT | out=$OUT ==="
date

$PYTHON $REPO/examples/eeg/viz_latents.py \
    --ckpt "$CKPT" \
    --out  "$OUT"  \
    --tuev

echo "=== done ===" && date
