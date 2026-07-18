# RGB-Subset Fairness Baseline

本目录只服务一个实验目的：

在同一批 RGB 可用样本上，公平比较：

- `RGB-XFi`: X-Fi 使用 Raw/Defaced RGB + 其他传感器。
- `VK-RMD`: 我们的方法使用同一样本对应的 VK + 其他传感器，推理不使用 RGB。

覆盖任务：

- HPE
- HAR

覆盖协议：

- cross-scene: train `E01+E02+E03`, test `E04`
- cross-subject: 沿用论文中的 subject split，仅在 RGB 可用样本内过滤

## 文件说明

- `run_xfi_rgb_pipeline.py`: 复现 RGB-XFi baseline。
- `run_vkrmd_vk_pipeline.py`: 跑同一 RGB 子集上的 VK-RMD teacher/student。
- `rgb_subset_common.py`: manifest、配置生成、任务执行、日志与 resume/skip 逻辑。
- `summarize_rgb_subset_fairness.py`: 两条 pipeline 都跑完后汇总论文表格。

脚本会自动兼容两套工程命名：

- 本地：`HPE/`, `HAR/`, `origin-XFI/Ori-HPE/`, `origin-XFI/Ori-HAR/`
- 云端：`MMFi_HPE/`, `MMFi_HAR/`, `Ori-HPE/`, `Ori-HAR/`

## 云端路径

假设代码根目录：

```bash
/apps/users/icps_intelligence/data/lyg/X-Fi
```

RGB 子集：

```bash
/apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB
```

MMFi 全模态数据：

```bash
/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA
```

## 先做 dry-run

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi

python -u RGB-Subset-Fairness/run_xfi_rgb_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --dry-run
```

```bash
python -u RGB-Subset-Fairness/run_vkrmd_vk_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --dry-run
```

dry-run 会生成/复用 manifest，并打印每个任务、checkpoint、输出路径，不训练。

## Smoke Test

正式长跑前建议只跑 1 epoch、2 个 train batch、2 个 eval batch：

```bash
CUDA_VISIBLE_DEVICES=0 python -u RGB-Subset-Fairness/run_xfi_rgb_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --force
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u RGB-Subset-Fairness/run_vkrmd_vk_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --force
```

注意：`CUDA_VISIBLE_DEVICES=1` 后，进程内部仍然使用 `--device cuda:0`，这是正确的。

## 正式两卡并行

当前 GPU 状态显示 GPU0 已占用约 32GB，GPU1 占用约 8GB。

建议：

- VK-RMD 更重，放 GPU1。
- RGB-XFi 放 GPU0；如果 GPU0 OOM，需要先清理 GPU0 上其它进程，或改成等 GPU1 空闲后单独跑。

三天内完成的推荐最小设置使用 `--quick`：

- `epochs=10`
- `max_train_batches=300`
- `max_eval_batches=300`
- HPE + HAR
- cross-scene + cross-subject

这是投稿前公平对比的 quick adaptation setting，不是 full convergence benchmark。

开两个 tmux：

```bash
tmux new -s rgb_xfi
cd /apps/users/icps_intelligence/data/lyg/X-Fi
CUDA_VISIBLE_DEVICES=0 python -u RGB-Subset-Fairness/run_xfi_rgb_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --quick
```

另一个窗口：

```bash
tmux new -s vkrmd_vk
cd /apps/users/icps_intelligence/data/lyg/X-Fi
CUDA_VISIBLE_DEVICES=1 python -u RGB-Subset-Fairness/run_vkrmd_vk_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --quick
```

如果时间更紧，只先跑 HPE：

```bash
CUDA_VISIBLE_DEVICES=0 python -u RGB-Subset-Fairness/run_xfi_rgb_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --quick \
  --tasks hpe
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u RGB-Subset-Fairness/run_vkrmd_vk_pipeline.py \
  --rgb-root /apps/users/icps_intelligence/data/lyg/code/MMFi_RGB_SUBSET/MMFi_Defaced_RGB \
  --mmfi-root /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 \
  --quick \
  --tasks hpe
```

## 中断恢复与保存

每个 pipeline 都有：

- 已有 `best.pth`：自动跳过该训练任务。
- 有 `last.pth` 且没有 `best.pth`：自动 resume。
- 每个任务结束立即写 `status.csv`。
- X-Fi: 每 epoch 覆盖 `last.pth`，每 5 epoch 保存 `epoch_005.pth / epoch_010.pth ...`。
- VK-RMD: 沿用原训练引擎，每 epoch 覆盖 `last.pth`，每 5 epoch 保存历史 checkpoint。

## 输出目录

Manifest:

```bash
outputs/rgb_subset_manifest/manifest.csv
outputs/rgb_subset_manifest/split_summary.md
```

RGB-XFi:

```bash
Ori-RGB-HPE/outputs_rgb_subset/
Ori-RGB-HAR/outputs_rgb_subset/
```

VK-RMD:

```bash
HPE/outputs_rgb_subset/
HAR/outputs_rgb_subset/
```

Pipeline 日志:

```bash
outputs/rgb_subset_runs/<timestamp>/manifest.json
outputs/rgb_subset_runs/<timestamp>/status.csv
outputs/rgb_subset_runs/<timestamp>/logs/
```

## 汇总论文表格

两条 pipeline 都跑完后：

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi
python -u RGB-Subset-Fairness/summarize_rgb_subset_fairness.py
```

输出：

```bash
tables/rgb_subset_fairness_hpe.csv
tables/rgb_subset_fairness_har.csv
```

论文中只把这组结果写成：

`Fair comparison on RGB-available subset`

不要和 full MMFi 主结果直接混表比较。
