#!/bin/bash
#SBATCH --job-name=viz_biot
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-slightlyunawarefc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:35:00
#SBATCH --output=/lustre/work/vivatech-slightlyunawarefc/tcourtois/train_logs/viz_biot_%j.out

set -e

SYS_PY=/cm/local/apps/python312/bin/python3.12
PYTHON=/lustre/work/vivatech-slightlyunawarefc/yhammache/venvs/eb_jepa_aarch64/bin/python
WHEELS=/lustre/work/vivatech-slightlyunawarefc/tcourtois/wheels
EXTRA=/lustre/work/vivatech-slightlyunawarefc/tcourtois/venv_extra
REPO=/lustre/work/vivatech-slightlyunawarefc/tcourtois/eb_jepa
SCRIPT=$REPO/examples/eeg/viz_biot.py
CKPT=/lustre/work/vivatech-slightlyunawarefc/tvasnier/external/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt
OUT=/lustre/work/vivatech-slightlyunawarefc/tcourtois/viz_out/biot

# ── vérifier que tout est là (préinstallé manuellement via unzip) ─────────────
echo "=== check deps ==="
PYTHONPATH=$EXTRA $PYTHON -c "
import sklearn, umap, matplotlib
import linear_attention_transformer
import axial_positional_embedding, linformer, local_attention, product_key_memory
print('all deps ok')
"

echo ""
echo "=== viz_biot ==="
echo "ckpt : $CKPT"
echo "out  : $OUT"
date

PYTHONPATH=$EXTRA $PYTHON $SCRIPT \
    --ckpt "$CKPT" \
    --out  "$OUT"  \
    --n-windows 16

echo ""
echo "=== done ===" && date

# Copier dans reports/figures/
REPORT_FIG=$REPO/reports/figures
mkdir -p $REPORT_FIG
cp $OUT/*.png $REPORT_FIG/ 2>/dev/null || true
echo "=== figures copiées dans $REPORT_FIG ==="
