#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA}"
GPU_ID="${1:-0}"
EVAL_BATCH_SIZE="${ORI_XFI_EVAL_BATCH_SIZE:-}"

cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
LATEST=""
if [ -f pre-trained_weights/best.pth ]; then
  LATEST="pre-trained_weights/best.pth"
elif [ -f pre-trained_weights/last.pth ]; then
  LATEST="pre-trained_weights/last.pth"
else
  LATEST=$(ls -t pre-trained_weights/checkpoint_*.pth 2>/dev/null | head -n 1 || true)
fi
if [ -z "${LATEST}" ]; then
  echo "[Ori-HPE] No main-model checkpoint found."
  exit 1
fi
echo "[Ori-HPE] Use checkpoint: $LATEST"
CMD=(python -u validate_all.py --dataset "$DATASET" --pt_weights "$LATEST")
if [ -n "$EVAL_BATCH_SIZE" ]; then
  CMD+=(--batch-size "$EVAL_BATCH_SIZE")
  echo "[Ori-HPE] Override eval batch size: $EVAL_BATCH_SIZE"
fi
"${CMD[@]}"
