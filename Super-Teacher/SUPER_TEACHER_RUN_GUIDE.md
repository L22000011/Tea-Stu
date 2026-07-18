# Super-Teacher 执行指南

本文档记录 Super-Teacher 实验的云端执行方式。默认使用 **0 号显卡**。

## 1. 准备 Backbones

在云端 `Super-Teacher` 目录下创建软链接，复用 HPE 工程中的 `backbones`：

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/Super-Teacher

ln -s /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HPE/backbones backbones
```

检查软链接是否创建成功：

```bash
ls -l backbones
```

正确结果类似：

```bash
backbones -> /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HPE/backbones
```

如果当前目录下已经存在错误的空 `backbones` 文件夹，确认无用后再删除并重新创建：

```bash
rm -r backbones
ln -s /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HPE/backbones backbones
```

## 2. Dry-Run 检查

Dry-run 只打印任务列表并检查命令拼接是否正确，不会训练模型。

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/Super-Teacher

CUDA_VISIBLE_DEVICES=1 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000 \
  --dry-run
```

## 3. 正式一体化执行

使用 0 号显卡一次性运行全部 Super-Teacher 实验：

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/Super-Teacher

CUDA_VISIBLE_DEVICES=1 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000
```

默认情况下，训练每个 epoch 只跑 `1000` 个 batch；测试为完整验证集测试。除非只是快速调试，否则不要传入 `--max-eval-batches`。

## 4. 断点恢复与跳过规则

- `last.pth` 每个 epoch 覆盖保存一次，用于中断恢复。
- 如果发生 OOM 或其他异常，脚本会额外尝试保存 `failed.pth`。
- 历史 checkpoint 每 10 个 epoch 保存一次，例如 `epoch_010.pth`、`epoch_020.pth`。
- `best*.pth` 只在对应指标变好时替换保存。
- pipeline 会检测每个任务的预期输出。
- 如果某个任务的预期输出已经全部存在，该任务会自动跳过。
- 如果任务没有完成，但存在 `last.pth`，训练会自动从 `last.pth` 恢复。
- 只有明确想重新跑已经完成的任务时，才使用 `--force`。

## 5. 输出目录

主要训练输出：

```bash
outputs/super_teacher/
outputs/super_teacher_cross_scene/
outputs/super_teacher_cross_subject/
outputs/super_hpe_student_vk/
outputs/super_hpe_student_nv/
outputs/super_har_student_vk/
outputs/super_har_student_nv/
```

所有评估 CSV：

```bash
outputs/eval/
```

pipeline 总控记录：

```bash
outputs/super_runs/<timestamp>/
  manifest.json
  status.csv
  logs/
```

其中：

- `manifest.json` 记录本次 pipeline 的全部任务和命令。
- `status.csv` 记录每个任务的完成、跳过或失败状态。
- `logs/` 保存每个任务的终端输出日志。

## 6. 一体化脚本执行内容

`scripts/run_super_pipeline.py` 一共执行 **37 个任务**。

每个 Student variant 都是独立任务。例如 HPE Student-VK 和 HPE Student-NV 分开训练、分开检测输出、分开恢复，避免其中一个失败导致两个都被绑在同一个大任务里反复重进。

### 6.1 Random Split

- Super Teacher 训练。
- Super Teacher 在 HPE 上进行 31 种模态组合测试。
- Super Teacher 在 HAR 上进行全模态组合测试。
- 使用 Super Teacher 蒸馏训练 HPE Student-VK。
- HPE Student-VK 进行全部组合测试。
- 使用 Super Teacher 蒸馏训练 HPE Student-NV。
- HPE Student-NV 进行非视觉组合测试。
- 使用 Super Teacher 蒸馏训练 HAR Student-VK。
- HAR Student-VK 进行全部组合测试。
- 使用 Super Teacher 蒸馏训练 HAR Student-NV。
- HAR Student-NV 进行非视觉组合测试。

### 6.2 Cross-Scene Split

- 使用跨场景划分重新训练 Super Teacher。
- 测试跨场景 Super Teacher 的 HPE/HAR 组合性能。
- 训练跨场景 HPE Student-VK 和 HPE Student-NV。
- 测试跨场景 HPE Student 的全组合和非视觉组合性能。
- 训练跨场景 HAR Student-VK 和 HAR Student-NV。
- 测试跨场景 HAR Student 的全组合和非视觉组合性能。

### 6.3 Cross-Subject Split

- 使用跨主体划分重新训练 Super Teacher。
- 测试跨主体 Super Teacher 的 HPE/HAR 组合性能。
- 训练跨主体 HPE Student-VK 和 HPE Student-NV。
- 测试跨主体 HPE Student 的全组合和非视觉组合性能。
- 训练跨主体 HAR Student-VK 和 HAR Student-NV。
- 测试跨主体 HAR Student 的全组合和非视觉组合性能。

### 6.4 必要消融实验

- HPE-only Super Teacher：
  - 只保留 HPE 监督训练 teacher。
  - 使用 `best_hpe.pth` 做 HPE 组合测试。

- HAR-only Super Teacher：
  - 只保留 HAR 监督训练 teacher。
  - 使用 `best_har.pth` 做 HAR 组合测试。

## 7. 推荐执行顺序

第一次上云端运行时，建议先执行：

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/Super-Teacher

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000 \
  --dry-run
```

确认路径没有问题后，再执行正式版本：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000
```
