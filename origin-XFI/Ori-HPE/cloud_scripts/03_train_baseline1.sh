#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA}"
GPU_ID="${1:-0}"

cd "$ROOT_DIR/baseline1"
mkdir -p baseline_weights
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[Ori-HPE] Start baseline1 training | dataset=$DATASET | gpu=$GPU_ID"
python -u run.py --dataset "$DATASET" --device cuda:0 --weights-dir ./baseline_weights
