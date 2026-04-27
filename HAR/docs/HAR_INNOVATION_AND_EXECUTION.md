# HAR Innovation and Execution Guide

## Research Positioning

This HAR project adapts the VK-RCD idea from HPE to human action recognition. It tests whether a privacy-first VK teacher-student framework also works for action classification under arbitrary modality loss.

The HAR task uses VK, Depth, LiDAR, and mmWave. Raw RGB images are not used. The `rgb/` directory is treated as VK and is expected to contain `17 x 2` keypoint `.npy` files.

## Core Innovations

1. RGB-free action recognition: the official RGB ResNet branch is removed.
2. VK-guided action teacher: full-modality action knowledge is learned from `VK + Depth + LiDAR + mmWave`.
3. Arbitrary-missing student: Student-VK supports all `2^4 - 1 = 15` non-empty modality combinations.
4. Reliability-aware classification fusion: modality logits are fused by dynamic reliability weights.
5. Non-VK deployment: Student-NV removes VK at inference and uses only Depth, LiDAR, and mmWave.

## Completed Experiments

| Experiment | Output file | Status |
|---|---|---|
| Teacher all 15 combinations | `outputs/eval/all_combinations.csv` | Completed |
| Teacher all 15 combinations Markdown | `outputs/eval/all_combinations.md` | Completed |
| Student-VK all 15 combinations | `outputs/eval/student_vk_all_combinations.csv` | Completed |
| Student-VK all 15 combinations Markdown | `outputs/eval/student_vk_all_combinations.md` | Completed |

Current random-split results show that the teacher reaches about `95.6%` full-modality accuracy, while Student-VK reaches about `96.5%` and improves many missing-modality cases.

## Main Commands

Train teacher:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_teacher_full.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/teacher_full.yaml --device cuda:0 --max-train-batches 1000
```

Train Student-VK:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_student_vk_missing.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_vk_missing.yaml --teacher outputs/teacher_full/best.pth --device cuda:0 --max-train-batches 1000
```

Train Student-NV:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_student_nv_missing.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_nv_missing.yaml --teacher outputs/teacher_full/best.pth --device cuda:0 --max-train-batches 1000
```

Evaluate Student-VK 15 combinations:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_all_combinations.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_vk_missing.yaml --checkpoint outputs/student_vk_missing/best.pth --device cuda:0 --output-csv outputs/eval/student_vk_all_combinations.csv --output-md outputs/eval/student_vk_all_combinations.md
```

Evaluate Student-NV non-visual combinations:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_nonvisual_combinations.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_nv_missing.yaml --checkpoint outputs/student_nv_missing/best.pth --device cuda:0 --output-csv outputs/eval/student_nv_nonvisual_combinations.csv --output-md outputs/eval/student_nv_nonvisual_combinations.md
```

## Remaining Work

| Priority | Experiment | Script |
|---|---|---|
| High | Student-NV training | `scripts/train_student_nv_missing.py` |
| High | Student-NV non-visual combinations | `scripts/eval_nonvisual_combinations.py` |
| High | Cross-subject teacher and student | set `split_to_use: cross_subject_split` |
| High | Cross-scene teacher and student | set `split_to_use: cross_scene_split` |
| Medium | Full baseline | `scripts/train_baseline_full.py` |
| Medium | No-distillation ablation | `scripts/ablate_no_distillation.py` |
| Medium | Uniform-fusion ablation | `scripts/ablate_uniform_fusion.py` |

## Paper Statement

We extend the VK-based privileged teacher-student design to action recognition, showing that sparse structural visual cues and non-RGB sensors can support robust HAR under arbitrary modality loss.

HAR should be presented as an important cross-task validation task. HPE remains the stronger primary task because pose estimation directly benefits from structural VK priors.
