#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_ID="${1:-0}"

cd "$ROOT_DIR"
echo "[Ori-HPE] Run main training and evaluation | gpu=$GPU_ID"
bash cloud_scripts/01_train_main.sh "$GPU_ID"
bash cloud_scripts/02_validate_main_latest.sh "$GPU_ID"
echo "[Ori-HPE] Main training and evaluation finished"
