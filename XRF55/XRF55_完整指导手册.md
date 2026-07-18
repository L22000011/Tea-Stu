# XRF55 项目完整指导手册

## 1. 项目定位

本目录对应我们在 **XRF55** 数据集上的完整复现实验与蒸馏增强版本实现。

当前主目录下保留了两套代码：

- `origin-XRF55/`：官方原始代码，不改动，作为官方参考实现。
- `XRF55/`：我们的蒸馏增强版本，统一采用和当前 HPE、HAR 项目一致的工程化训练、评估、消融和 pipeline 逻辑。

本项目只做 **与官方 XRF55 一致的默认 train/test 任务**，不额外虚构 cross-scene 或 cross-subject 协议。

---

## 2. 我们的方法思路

本项目的方法可以概括为：

1. 以 `WiFi + RFID + mmWave` 三模态全输入训练一个 **Teacher**。
2. 构造一个支持 **任意模态缺失** 的 **Student**，训练时随机缺失 1 或 2 个模态。
3. 通过 **logit distillation + token distillation + reliability distillation + semantic alignment**，让 Student 在模态缺失场景下仍保持稳定识别能力。
4. 使用 **可靠性融合** 动态分配不同模态在当前样本上的决策权，而不是简单平均。

---

## 3. 目录结构

```text
Tea/
├── origin-XRF55/                 # 官方代码
└── XRF55/                        # 我们的完整工程
    ├── configs/
    ├── data/
    ├── docs/
    ├── losses/
    ├── models/
    ├── outputs/
    ├── scripts/
    ├── training/
    ├── utils/
    └── XRF55_完整指导手册.md
```

---

## 4. 每个文件夹和文件的作用

### 4.1 `origin-XRF55/`

- `dml_train.py`：官方三模态 mutual learning 训练脚本。
- `dml_eval.py`：官方单模态测试脚本。
- `XRFDataset.py`：官方数据读取。
- `model/resnet1d.py`：官方 WiFi backbone。
- `model/resnet1d_rfid.py`：官方 RFID backbone。
- `model/resnet2d.py`：官方 mmWave backbone。
- `word2vec/bert_new_sentence_large_uncased.npy`：官方语义向量。

这部分保持官方原样，用于：

- 官方基线对照；
- 我们初始化 / 兼容官方预训练权重；
- 语义向量监督。

### 4.2 `XRF55/configs/`

- `teacher_full.yaml`：Teacher 全模态训练配置。
- `student_missing.yaml`：Student 任意模态缺失训练配置。
- `baseline_full.yaml`：不做蒸馏的全模态 baseline 配置。
- `ablation_no_distill.yaml`：去蒸馏消融配置。
- `ablation_uniform_fusion.yaml`：去可靠性融合，改成均匀融合的消融配置。
- `ablation_no_semantic.yaml`：去语义监督的消融配置。

### 4.3 `XRF55/data/`

- `__init__.py`：导出 dataloader 构建接口。
- `xrf55_dataset.py`：我们的 XRF55 数据集实现。

作用：

- 读取官方 `dml_train.txt` / `dml_val.txt`；
- 读取 `WiFi / RFID / mmWave` 三模态 `.npy`；
- 自动读取官方 `word2vec` 语义向量；
- 对异常 shape、NaN、Inf 做基本容错和修正；
- 构建训练集和验证集 DataLoader。

### 4.4 `XRF55/models/`

- `__init__.py`：导出模型构建接口。
- `official_backbones.py`：兼容官方 XRF55 backbone 的安全实现，支持加载官方参数。
- `reliability_fusion.py`：模态 token 编码器和可靠性融合模块。
- `xrf_rcd.py`：我们的主模型 `XRFRCDModel`。

作用：

- 复用官方三种 backbone 的结构风格；
- 将每个模态 backbone 输出的 1024 维特征映射为统一 token；
- 对不同模态 token 做统一编码；
- 使用 uncertainty / attention / uniform 三种融合方式完成分类；
- 额外输出语义向量预测，用于语义监督和蒸馏。

### 4.5 `XRF55/losses/`

- `__init__.py`：导出 loss 接口。
- `distill_losses.py`：Teacher / Student 损失定义。

作用：

- Teacher：分类损失 + 语义损失 + 不确定性正则。
- Student：分类损失 + 语义损失 + logit KD + token KD + reliability KD + semantic KD。

### 4.6 `XRF55/training/`

- `__init__.py`
- `engine.py`

作用：

- 训练 Teacher；
- 训练 Student；
- 保存 `best.pth / last.pth / interrupted.pth / epoch_xxx.pth`；
- 自动记录 `epoch_history.csv / final_eval.csv / final_summary.json / training_curve.png`；
- 支持恢复训练；
- 支持 7 个模态组合评估。

### 4.7 `XRF55/utils/`

- `__init__.py`
- `config.py`：读取 yaml 配置。
- `metrics.py`：参数量、accuracy、macro-F1 等指标计算。
- `modality.py`：模态名称规范化、缺失模态采样、组合生成。
- `reporting.py`：CSV、Markdown、依赖记录、训练曲线输出。

### 4.8 `XRF55/scripts/`

- `_shared.py`：所有脚本共享的参数解析、配置解析、模型加载逻辑。
- `train_teacher_full.py`：训练全模态 Teacher。
- `train_student_missing.py`：训练支持任意模态缺失的 Student。
- `train_baseline_full.py`：训练不做蒸馏的全模态 baseline。
- `ablate_no_distillation.py`：去蒸馏消融训练。
- `ablate_uniform_fusion.py`：均匀融合消融训练。
- `ablate_no_semantic.py`：去语义监督消融训练。
- `eval_all_combinations.py`：测试全部 7 个非空模态组合。
- `summarize_missing_counts.py`：把组合结果按“缺失模态数量”汇总。
- `export_complexity.py`：导出参数量、速度、显存等复杂度指标。
- `run_full_pipeline.py`：主实验一键脚本。
- `run_supplemental_pipeline.py`：补充实验一键脚本。

### 4.9 `XRF55/outputs/`

用于保存所有训练与测试产物，典型结构如下：

- `outputs/teacher_full/`
- `outputs/student_missing/`
- `outputs/baseline_full/`
- `outputs/ablation_no_distill/`
- `outputs/ablation_uniform_fusion/`
- `outputs/ablation_no_semantic/`
- `outputs/eval/`
- `outputs/pipeline_runs/`
- `outputs/supplemental_runs/`

---

## 5. 数据集放置方式

云端推荐直接把官方生成后的 `XRF_dataset` 目录作为 `--dataset` 输入。

期望结构如下：

```text
XRF_dataset/
├── dml_train.txt
├── dml_val.txt
└── dml_new_data/
    ├── train_data/
    │   ├── WiFi/
    │   ├── RFID/
    │   └── mmWave/
    └── test_data/
        ├── WiFi/
        ├── RFID/
        └── mmWave/
```

如果你传入的是更上层目录，只要里面存在 `XRF_dataset/` 子目录，代码也会自动识别。

---

## 6. 官方代码如何执行

如果你想先跑官方代码：

```bash
cd /path/to/Tea/origin-XRF55
CUDA_VISIBLE_DEVICES=0 python dml_train.py
CUDA_VISIBLE_DEVICES=0 python dml_eval.py
```

说明：

- 官方训练和测试脚本只覆盖原始任务；
- 官方测试主要是单模态结果；
- 我们的新工程在 `XRF55/` 中，不依赖你手工改官方代码。

---

## 7. 我们的单独脚本怎么执行

以下命令都在：

```bash
cd /path/to/Tea/XRF55
```

下执行。

### 7.1 训练 Teacher

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_teacher_full.py \
  --dataset /path/to/XRF_dataset \
  --config configs/teacher_full.yaml \
  --device cuda:0 \
  --max-train-batches 1000
```

### 7.2 训练 Student

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_student_missing.py \
  --dataset /path/to/XRF_dataset \
  --config configs/student_missing.yaml \
  --teacher outputs/teacher_full/best.pth \
  --device cuda:0 \
  --max-train-batches 1000
```

### 7.3 训练 Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_baseline_full.py \
  --dataset /path/to/XRF_dataset \
  --config configs/baseline_full.yaml \
  --device cuda:0 \
  --max-train-batches 1000
```

### 7.4 评估 7 个组合

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_all_combinations.py \
  --dataset /path/to/XRF_dataset \
  --config configs/student_missing.yaml \
  --checkpoint outputs/student_missing/best.pth \
  --device cuda:0 \
  --output-csv outputs/eval/student_all_combinations.csv \
  --output-md outputs/eval/student_all_combinations.md
```

---

## 8. 主实验 pipeline 覆盖哪些任务

主实验一键脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_full_pipeline.py \
  --dataset /path/to/XRF_dataset \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000
```

它一共执行 **12 个任务**：

1. Teacher 全模态训练
2. Teacher 全部 7 组合测试
3. Student 任意缺失训练
4. Student 全部 7 组合测试
5. Baseline 全模态训练
6. Baseline 全部 7 组合测试
7. 无蒸馏消融训练
8. 无蒸馏消融测试
9. 均匀融合消融训练
10. 均匀融合消融测试
11. 去语义监督消融训练
12. 去语义监督消融测试

特点：

- 默认不覆盖已存在结果；
- 只要输出已经存在，就自动跳过；
- 支持 `--force` 强制重跑；
- 每个任务都有单独 log；
- 每次运行会生成独立的 `outputs/pipeline_runs/<timestamp>/status.csv`。

---

## 9. 补充实验 pipeline 覆盖哪些任务

补充实验一键脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_supplemental_pipeline.py \
  --dataset /path/to/XRF_dataset \
  --gpu 0 \
  --device cuda:0
```

它一共执行 **9 个任务**：

1. Teacher 结果按缺失模态数量汇总
2. Student 结果按缺失模态数量汇总
3. Baseline 结果按缺失模态数量汇总
4. 无蒸馏消融结果按缺失模态数量汇总
5. 均匀融合消融结果按缺失模态数量汇总
6. 去语义监督消融结果按缺失模态数量汇总
7. Teacher 复杂度导出
8. Student 复杂度导出
9. Baseline 复杂度导出

补充实验的作用：

- 支撑论文中的鲁棒性趋势分析；
- 给出 efficiency / complexity 结果；
- 让实验包更完整。

---

## 10. 输出文件如何理解

每个训练目录一般包含：

- `best.pth`：验证集最优模型。
- `last.pth`：最新 checkpoint。
- `interrupted.pth`：中断保存。
- `epoch_005.pth`、`epoch_010.pth` ...：周期性 checkpoint。
- `epoch_history.csv`：每个 epoch 的训练记录。
- `final_eval.csv`：训练完成后自动测试一次的结果。
- `final_eval.md`：测试结果 Markdown 版。
- `final_summary.json`：最终结果摘要。
- `training_curve.png`：训练曲线。
- `config_used.yaml`：本次真实使用的配置。
- `dependencies.txt`：依赖快照。
- `experiment_info.json`：实验信息。

---

## 11. 建议的云端执行顺序

建议先跑主实验，再跑补充实验：

```bash
cd /path/to/Tea/XRF55

tmux new -s xrf55

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_full_pipeline.py \
  --dataset /path/to/XRF_dataset \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_supplemental_pipeline.py \
  --dataset /path/to/XRF_dataset \
  --gpu 0 \
  --device cuda:0
```

如果要退出 tmux：

```bash
Ctrl + B，然后按 D
```

重新进入：

```bash
tmux attach -t xrf55
```

---

## 12. 论文实验建议写法

建议正文至少包含下面几块：

1. 官方 XRF55 默认任务复现说明。
2. 我们的 Teacher / Student / Baseline 主结果对比。
3. 7 个模态组合结果表。
4. 缺失模态数量汇总结果。
5. 无蒸馏、无可靠性融合、无语义监督三组消融。
6. 参数量、速度、显存复杂度表。

---

## 13. 这套代码当前能覆盖什么

这套代码已经覆盖：

- 官方默认任务下的完整训练流程；
- 全模态 Teacher；
- 任意模态缺失 Student；
- 全模态 Baseline；
- 7 个模态组合评估；
- 3 组关键消融；
- 缺失模态数量汇总；
- 复杂度导出；
- 一键 pipeline 与日志管理。

也就是说，对 XRF55 这个项目而言，当前已经是 **完整论文实验包** 版本。

---

## 14. 最后说明

- 所有新增代码注释均采用英文。
- 官方代码与我们的代码彻底分离，方便对照和后续复现。
- 如果云端运行报错，把报错日志直接发回来，我可以继续基于这套结构增量修复。

