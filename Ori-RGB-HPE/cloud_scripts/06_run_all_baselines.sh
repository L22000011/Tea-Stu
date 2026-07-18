#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_ID="${1:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/outputs"
RUN_LOG="$OUTPUT_DIR/baseline_pipeline_${TIMESTAMP}.log"
LATEST_LOG="$OUTPUT_DIR/baseline_pipeline_latest.log"

mkdir -p "$OUTPUT_DIR"
exec > >(tee "$RUN_LOG") 2>&1

cd "$ROOT_DIR"
echo "[Ori-HPE] Run baseline1 + baseline2 pipeline | gpu=$GPU_ID"
bash cloud_scripts/03_train_baseline1.sh "$GPU_ID"
bash cloud_scripts/04_generate_baseline2.sh "$GPU_ID"
bash cloud_scripts/05_run_baseline2_fusion.sh
cp "$RUN_LOG" "$LATEST_LOG"
echo "[Ori-HPE] Baseline pipeline finished"
echo "[Ori-HPE] Saved log: $RUN_LOG"
echo "[Ori-HPE] Latest log: $LATEST_LOG"
