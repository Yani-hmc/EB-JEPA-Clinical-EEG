#!/bin/bash
#SBATCH --job-name=viz_latents
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-slightlyunawarefc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/viz_latents_%j.out

set -e

# système python3.12 (a pip) — sert à installer dans EXTRA seulement
SYS_PY=/cm/local/apps/python312/bin/python3.12
# yhammache venv — a torch, sklearn, omegaconf, etc.
PYTHON=/lustre/work/vivatech-slightlyunawarefc/yhammache/venvs/eb_jepa_aarch64/bin/python
WHEELS=/lustre/work/vivatech-slightlyunawarefc/tcourtois/wheels
EXTRA=/lustre/work/vivatech-slightlyunawarefc/tcourtois/venv_extra
REPO=/lustre/work/vivatech-slightlyunawarefc/tcourtois/eb_jepa
SCRIPT=$REPO/examples/eeg/viz_latents.py

# ── installer umap-learn + matplotlib dans EXTRA (tcourtois-owned) ───────────
echo "=== install umap-learn + matplotlib → $EXTRA ==="
mkdir -p $EXTRA
$SYS_PY -m pip install --quiet --no-index --find-links=$WHEELS --target=$EXTRA \
    umap-learn matplotlib

# ── vérifier que tout est disponible ─────────────────────────────────────────
echo "=== check deps ==="
PYTHONPATH=$EXTRA $PYTHON -c "import sklearn; print('sklearn', sklearn.__version__)"
PYTHONPATH=$EXTRA $PYTHON -c "import umap; print('umap ok')"
PYTHONPATH=$EXTRA $PYTHON -c "import matplotlib; print('matplotlib ok')"
PYTHONPATH=$EXTRA $PYTHON -c "import torch; print('torch', torch.__version__)"

# ── meilleur modèle : Conv1D + VICReg (λ=1) + corruption  (BACC 0.819 ± .004) ──
CKPT=${CKPT:-/lustre/work/vivatech-slightlyunawarefc/tvasnier/checkpoints/eeg/corrupt/latest.pth.tar}
OUT=${OUT:-/lustre/work/vivatech-slightlyunawarefc/tcourtois/viz_out/$(basename $(dirname $CKPT))}

echo ""
echo "=== viz_latents ==="
echo "ckpt : $CKPT"
echo "out  : $OUT"
date

PYTHONPATH=$EXTRA $PYTHON $SCRIPT \
    --ckpt "$CKPT" \
    --out  "$OUT"  \
    --split eval   \
    --n-windows 16

echo ""
echo "=== done ===" && date

# ── copier les figures dans reports/figures/ pour git ─────────────────────────
REPORT_FIG=/lustre/work/vivatech-slightlyunawarefc/tcourtois/eb_jepa/reports/figures
mkdir -p $REPORT_FIG
cp $OUT/*.png $REPORT_FIG/ 2>/dev/null || true
echo "=== figures copiées dans $REPORT_FIG ==="
