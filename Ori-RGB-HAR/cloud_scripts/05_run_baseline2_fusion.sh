#!/bin/bash
set -euo pipefail

ROOT_DIR="/apps/users/icps_intelligence/data/lyg/X-Fi/Ori-HAR"

cd "$ROOT_DIR/baseline2"
echo "[Ori-HAR] 开始运行 baseline2 late fusion"
python -u run.py
