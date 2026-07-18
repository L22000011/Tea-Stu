# Dual Readout Ablation

隔离评估 `chunk_only`、`global_only` 和正式 `dual` 三种读出。脚本不修改 HPE/HAR 正式模型文件。

云端快速实验：

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi

CUDA_VISIBLE_DEVICES=0 python -u Dual-Readout-Ablation/run_dual_readout_ablation.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0
```

默认三种模式都使用相同的10 epoch、每epoch最多300个训练batch、每组合100个验证batch。结果固定写入：

```text
outputs/dual_readout_ablation/
  HPE/{chunk_only,global_only,dual}/
  HAR/{chunk_only,global_only,dual}/
  dual_readout_ablation_summary.csv
  runs/<timestamp>/{manifest.json,status.csv,logs/}
```

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python -u Dual-Readout-Ablation/run_dual_readout_ablation.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --tasks HPE \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --force
```

Dry-run：

```bash
python -u Dual-Readout-Ablation/run_dual_readout_ablation.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --dry-run
```
