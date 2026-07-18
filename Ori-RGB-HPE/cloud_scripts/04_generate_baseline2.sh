#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA}"
GPU_ID="${1:-0}"

cd "$ROOT_DIR/baseline2"
mkdir -p baseline_results
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[Ori-HPE] Generate baseline2 single-modality results | dataset=$DATASET | gpu=$GPU_ID"
python -u generate_single_result.py \
  --dataset "$DATASET" \
  --device cuda:0 \
  --weights-dir ../baseline1/baseline_weights \
  --results-dir ./baseline_results
