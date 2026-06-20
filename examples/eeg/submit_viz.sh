#!/bin/bash
#SBATCH --job-name=viz_latents
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-slightlyunawarefc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/viz_latents_%j.out

set -e

PYTHON=/lustre/work/vivatech-slightlyunawarefc/tvasnier/venvs/eb_jepa_x86_64/bin/python
REPO=/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa
SCRIPT=$REPO/examples/eeg/viz_latents.py

# ── s'assurer que les librairies sont là ─────────────────────────────────────
echo "=== vérification / install des dépendances ==="
$PYTHON -m pip install --quiet --upgrade \
    scikit-learn \
    umap-learn \
    matplotlib

# ── meilleur modèle : Conv1D + VICReg (λ=1) + corruption  (BACC 0.819 ± .004) ──
# Changer CKPT pour tester un autre checkpoint
CKPT=${CKPT:-/lustre/work/vivatech-slightlyunawarefc/tvasnier/checkpoints/eeg/corrupt/latest.pth.tar}
OUT=${OUT:-/lustre/work/vivatech-slightlyunawarefc/tcourtois/viz_out/$(basename $(dirname $CKPT))}

echo ""
echo "=== viz_latents ==="
echo "ckpt : $CKPT"
echo "out  : $OUT"
date

$PYTHON $SCRIPT \
    --ckpt "$CKPT" \
    --out  "$OUT"  \
    --split eval   \
    --n-windows 16

echo ""
echo "=== done ===" && date

# Copier les figures dans reports/figures/ pour git
REPORT_FIG=/lustre/work/vivatech-slightlyunawarefc/tcourtois/eb_jepa/reports/figures
mkdir -p $REPORT_FIG
cp $OUT/*.png $REPORT_FIG/ 2>/dev/null || true
echo "=== figures copiées dans $REPORT_FIG ==="
