#!/bin/bash
set -euo pipefail

ROOT_DIR="/apps/users/icps_intelligence/data/lyg/X-Fi/Ori-HAR"
DATASET="${DATASET:-/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA}"
GPU_ID="${1:-0}"

cd "$ROOT_DIR/baseline1"
mkdir -p baseline_weights
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[Ori-HAR] 开始训练 baseline1 | dataset=$DATASET | gpu=$GPU_ID"
python -u run.py --dataset "$DATASET" --device cuda:0 --weights-dir ./baseline_weights
