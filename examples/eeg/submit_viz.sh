#!/bin/bash
#SBATCH --job-name=viz_latents
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-slightlyunawarefc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/viz_latents_%j.out

set -e

PYTHON=/lustre/work/vivatech-slightlyunawarefc/yhammache/venvs/eb_jepa_aarch64/bin/python
REPO=/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa
SCRIPT=$REPO/examples/eeg/viz_latents.py

# ── installer libs depuis wheels pré-téléchargés (pas de réseau sur compute) ─
WHEELS=/lustre/work/vivatech-slightlyunawarefc/tcourtois/wheels
echo "=== install deps offline ==="
$PYTHON -m ensurepip --upgrade 2>/dev/null || true
$PYTHON -m pip install --quiet --no-index --find-links=$WHEELS \
    umap-learn matplotlib

echo "=== check deps ==="
$PYTHON -c "import sklearn; print('sklearn', sklearn.__version__)"
$PYTHON -c "import umap; print('umap ok')"
$PYTHON -c "import matplotlib; print('matplotlib', matplotlib.__version__)"

# ── meilleur modèle : Conv1D + VICReg (λ=1) + corruption  (BACC 0.819 ± .004) ──
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
