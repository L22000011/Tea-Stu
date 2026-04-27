# HAR VK-RCD Guide

## Overview

This folder contains the HAR adaptation of the VK-RCD pipeline:

- `Teacher`: full-modality structural classifier using `VK + Depth + LiDAR + mmWave`
- `Student-VK`: arbitrary-missing-modality classifier distilled from the teacher
- `Student-NV`: no-VK inference classifier distilled from the teacher

Official RGB image processing is removed. The `rgb/` folder is treated as `VK` and must store per-frame `17x2` keypoint `.npy` files.

## Folder Layout

- `legacy_xfi/`: archived official HAR code
- `configs/`: training and ablation configs
- `data/`: MM-Fi HAR dataset reader and collate logic
- `models/`: VK encoder, legacy backbones, fusion model
- `losses/`: classification and distillation losses
- `training/`: training, evaluation, checkpointing, reporting
- `scripts/`: train and evaluation entrypoints
- `outputs/`: checkpoints and reports

## Training

### 1. Teacher

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_teacher_full.py \
  --dataset /path/to/MMFi_DATA \
  --config configs/teacher_full.yaml \
  --device cuda:0
```

### 2. Student with VK

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_student_vk_missing.py \
  --dataset /path/to/MMFi_DATA \
  --config configs/student_vk_missing.yaml \
  --teacher outputs/teacher_full/best.pth \
  --device cuda:0
```

### 3. Student without VK

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_student_nv_missing.py \
  --dataset /path/to/MMFi_DATA \
  --config configs/student_nv_missing.yaml \
  --teacher outputs/teacher_full/best.pth \
  --device cuda:0
```

## Evaluation

### All 15 combinations

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_all_combinations.py \
  --dataset /path/to/MMFi_DATA \
  --config configs/student_vk_missing.yaml \
  --checkpoint outputs/student_vk_missing/best.pth \
  --device cuda:0
```

### Non-visual 7 combinations

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_nonvisual_combinations.py \
  --dataset /path/to/MMFi_DATA \
  --config configs/student_nv_missing.yaml \
  --checkpoint outputs/student_nv_missing/best.pth \
  --device cuda:0
```

## Ablation

- `train_baseline_full.py`: full-modality baseline without KD
- `ablate_no_distillation.py`: random missing modality training without teacher KD
- `ablate_uniform_fusion.py`: replace reliability fusion with uniform weighting

## Output Files

Each training run writes:

- `best.pth`
- `last.pth`
- `epoch_005.pth`, `epoch_010.pth`, ...
- `epoch_history.csv`
- `final_eval.csv`
- `final_eval.md`
- `final_summary.json`
- `training_curve.png`
- `config_used.yaml`
- `dependencies.txt`
- `experiment_info.json`
