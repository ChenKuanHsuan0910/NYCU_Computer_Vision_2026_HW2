#!/bin/bash
# =============================================================================
# NYCU Visual Recognition 2026 Spring — HW2
# DETR + ResNet-50 Digit Detection
#
# Usage (submit):
#   sbatch slurm_train_hw2.sh
#
# Switch between smoke test and full training by toggling the MODE variable
# below.  Everything else is handled automatically.
# =============================================================================

#SBATCH -J vr_hw2_detr
#SBATCH -o /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw2/outputs/slurm_%j.out
#SBATCH -e /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw2/outputs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=h200q

# =============================================================================
# MODE: set to "smoke" for quick sanity check (~5-10 min)
#       set to "full"  for leaderboard training  (~8-12 h on H200)
# =============================================================================
MODE="full"      # <-- change to "smoke" to do a quick test first

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load anaconda
module load slurm
module load nvidia-hpc
module load nvhpc-hpcx-cuda12
module list

eval "$(conda shell.bash hook)"
conda deactivate || true
conda activate Visual_Recognition

export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Paths  (data is already extracted — no unzip needed)
# ---------------------------------------------------------------------------
PROJECT_DIR="/home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw2"
DATA_ROOT="${PROJECT_DIR}/data/nycu-hw2-data"
OUTPUT_BASE="${PROJECT_DIR}/outputs"
RUN_DIR="${OUTPUT_BASE}/run_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"

echo "======================================================"
echo " PROJECT_DIR : ${PROJECT_DIR}"
echo " DATA_ROOT   : ${DATA_ROOT}"
echo " RUN_DIR     : ${RUN_DIR}"
echo " MODE        : ${MODE}"
echo "======================================================"

# Verify data exists
if [ ! -d "${DATA_ROOT}/train" ] || [ ! -f "${DATA_ROOT}/train.json" ]; then
    echo "[ERROR] Data not found at ${DATA_ROOT}. Check the path."
    exit 1
fi
echo "[INFO] Data check passed. train images: $(ls ${DATA_ROOT}/train | wc -l)"

# ---------------------------------------------------------------------------
# Install missing Python dependencies (first run only; cached afterward)
# ---------------------------------------------------------------------------
echo "[INFO] Checking / installing Python dependencies..."
pip install --quiet \
    "transformers>=4.35.0" \
    "albumentations>=1.3.0" \
    "pycocotools" \
    "timm"

echo "[INFO] Dependencies OK."

# ---------------------------------------------------------------------------
# Hyper-parameters
#
# SMOKE TEST  →  quick sanity check (2 epochs, eval every epoch)
#   Goal: verify the code runs end-to-end without errors.
#   Expected time: ~5-10 min on H200
#
# FULL TRAINING  →  leaderboard submission
#   Goal: maximise mAP@[0.5:0.95] on test set
#   Expected time: ~8-12 h on H200
# ---------------------------------------------------------------------------
if [ "${MODE}" = "smoke" ]; then
    echo "[INFO] Running SMOKE TEST"
    CHECKPOINT_RESUME=""
    EPOCHS=2
    BATCH_SIZE=4
    GRAD_ACCUM=1
    LR=1e-4
    LR_BACKBONE=1e-5
    WEIGHT_DECAY=1e-4
    GRAD_CLIP=0.1
    WARMUP_RATIO=0.05
    SHORTEST_EDGE=800
    LONGEST_EDGE=1333
    CONF_THRESHOLD_TRAIN=0.05
    CONF_THRESHOLD_INFER=0.05
    NMS_IOU=0.0
    MAX_DETS=20
    EVAL_EVERY=1
    NUM_WORKERS=4
    AMP_FLAG="--amp"
    COMBINE_TRAINVAL_FLAG=""
    TTA_FLAG=""
else
    echo "[INFO] Running FULL TRAINING (fresh from COCO pretrained + category_id fix + TTA)"
    # KEY FIX: Train from scratch - previous checkpoints (49331/49615) had wrong
    # class mappings: category_id=10 (digit '9') was mapped to the DETR no-object
    # class index (=num_labels=10), causing digit '9' to be NEVER detected.
    # Fix: id2label now uses 0-indexed keys (0-9), dataset remaps cat_id-1.
    EPOCHS=100
    BATCH_SIZE=4
    GRAD_ACCUM=2           # effective batch = 4 * 2 = 8
    LR=1e-4                # standard DETR LR (fresh training from COCO pretrained)
    LR_BACKBONE=1e-5       # standard backbone LR
    WEIGHT_DECAY=1e-4
    GRAD_CLIP=0.1
    WARMUP_RATIO=0.05
    SHORTEST_EDGE=800
    LONGEST_EDGE=1333
    CONF_THRESHOLD_TRAIN=0.3   # threshold for COCO eval during training
    CONF_THRESHOLD_INFER=0.05  # lower threshold at inference -> better recall
    NMS_IOU=0.5                # per-class NMS IoU
    MAX_DETS=30
    EVAL_EVERY=5               # eval every 5 epochs (100 epochs total)
    NUM_WORKERS=8
    AMP_FLAG="--amp"
    COMBINE_TRAINVAL_FLAG="--combine_trainval"   # 11% more data (train+valid)
    TTA_FLAG="--tta --tta_scales 0.8,1.0,1.2"   # multi-scale TTA at inference
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Starting training..."

python train_hw2_detr.py \
    --data_root      "${DATA_ROOT}" \
    --output_dir     "${RUN_DIR}" \
    ${COMBINE_TRAINVAL_FLAG} \
    --epochs         "${EPOCHS}" \
    --batch_size     "${BATCH_SIZE}" \
    --num_workers    "${NUM_WORKERS}" \
    --lr             "${LR}" \
    --lr_backbone    "${LR_BACKBONE}" \
    --weight_decay   "${WEIGHT_DECAY}" \
    --grad_clip      "${GRAD_CLIP}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --warmup_ratio   "${WARMUP_RATIO}" \
    --shortest_edge  "${SHORTEST_EDGE}" \
    --longest_edge   "${LONGEST_EDGE}" \
    --conf_threshold "${CONF_THRESHOLD_TRAIN}" \
    --max_dets_per_image "${MAX_DETS}" \
    --eval_every     "${EVAL_EVERY}" \
    ${AMP_FLAG} \
    --do_train

echo "[INFO] Training finished."

# ---------------------------------------------------------------------------
# Inference using best checkpoint  →  pred.json
# ---------------------------------------------------------------------------
BEST_CKPT="${RUN_DIR}/checkpoints/best.pt"

# Fallback to latest if best does not exist (e.g. mAP was never evaluated)
if [ ! -f "${BEST_CKPT}" ]; then
    echo "[WARN] best.pt not found, falling back to latest.pt"
    BEST_CKPT="${RUN_DIR}/checkpoints/latest.pt"
fi

echo ""
echo "[INFO] Running inference with checkpoint: ${BEST_CKPT}"
echo "[INFO] conf_threshold=${CONF_THRESHOLD_INFER}  nms_iou=${NMS_IOU}  max_dets=${MAX_DETS}"
python train_hw2_detr.py \
    --data_root      "${DATA_ROOT}" \
    --output_dir     "${RUN_DIR}" \
    --shortest_edge  "${SHORTEST_EDGE}" \
    --longest_edge   "${LONGEST_EDGE}" \
    --conf_threshold "${CONF_THRESHOLD_INFER}" \
    --nms_iou        "${NMS_IOU}" \
    --max_dets_per_image "${MAX_DETS}" \
    ${TTA_FLAG} \
    --checkpoint     "${BEST_CKPT}" \
    --do_infer

echo "[INFO] Inference done. pred.json written to: ${RUN_DIR}/pred.json"

# ---------------------------------------------------------------------------
# Validate pred.json format
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Validating pred.json format..."
python train_hw2_detr.py --validate_pred "${RUN_DIR}/pred.json"

# ---------------------------------------------------------------------------
# Create submission.zip  (use python zipfile — zip binary may not be in PATH)
# ---------------------------------------------------------------------------
cd "${RUN_DIR}"
python3 -c "
import zipfile, os
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('pred.json', 'pred.json')
print('submission.zip size:', os.path.getsize('submission.zip'), 'bytes')
"
echo ""
echo "======================================================"
echo " submission.zip ready: ${RUN_DIR}/submission.zip"
echo " pred.json entries   : $(python3 -c \"import json; d=json.load(open('pred.json')); print(len(d))\")"
echo "======================================================"
