#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA}"
GPU_ID="${1:-0}"
RESUME_MODE="${ORI_XFI_RESUME:-0}"

cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
CMD=(python -u run.py --dataset "$DATASET")
if [ "$RESUME_MODE" = "1" ]; then
  CMD+=(--resume)
  echo "[Ori-HPE] Start main training | mode=resume | dataset=$DATASET | gpu=$GPU_ID"
else
  echo "[Ori-HPE] Start main training | mode=fresh_start | dataset=$DATASET | gpu=$GPU_ID"
fi
"${CMD[@]}"
