# HPE Innovation and Execution Guide

## Research Positioning

This HPE project is built on MM-Fi/X-Fi. It removes raw RGB input and replaces the visual branch with visual keypoints (VK). The goal is a privacy-first multimodal pose estimator that remains reliable under arbitrary modality loss.

## Core Innovations

1. Privacy-first VK prior: raw RGB is not used. The `rgb/*.npy` files are treated as `17 x 2` skeleton keypoints.
2. Lightweight skeleton encoder: ResNet18 is removed and replaced by a joint-wise keypoint encoder with graph-aware skeleton mixing.
3. Full-modality teacher: VK, Depth, LiDAR, mmWave, and WiFi-CSI are fused to learn privileged structural knowledge.
4. Arbitrary-missing student: Student-VK is trained with random modality dropping and teacher distillation.
5. Reliability-aware fusion: modality weights are generated dynamically from estimated reliability.
6. Non-visual deployment: Student-NV removes VK at inference and uses only non-RGB sensing modalities.

## Completed Experiments

| Experiment | Output file | Status |
|---|---|---|
| Teacher all 31 combinations | `outputs/eval/teacher_all_combinations.csv` | Completed |
| Student-VK all 31 combinations | `outputs/eval/all_combinations.csv` | Completed |
| Student-VK missing-count summary | `outputs/eval/student_vk_missing_modality_summary.csv` | Completed |
| Student-VK VK-noise robustness | `outputs/eval/student_vk_noise_robustness.csv` | Completed |

Current random-split results show that Student-VK stays close to the full-modality teacher and is much more stable than the teacher under missing modality combinations.

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

Evaluate Student-VK 31 combinations:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_all_combinations.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_vk_missing.yaml --checkpoint outputs/student_vk_missing/best.pth --device cuda:0 --output-csv outputs/eval/student_vk_all_combinations.csv
```

Evaluate Student-NV non-visual combinations:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_nonvisual_combinations.py --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA --config configs/student_nv_missing.yaml --checkpoint outputs/student_nv_missing/best.pth --device cuda:0 --output-csv outputs/eval/student_nv_nonvisual_combinations.csv
```

## Remaining Work

| Priority | Experiment | Script |
|---|---|---|
| High | Student-NV training | `scripts/train_student_nv_missing.py` |
| High | Student-NV non-visual combinations | `scripts/eval_nonvisual_combinations.py` |
| High | Cross-subject teacher and student | set `split_to_use: cross_subject_split` |
| High | Cross-scene teacher and student | set `split_to_use: cross_scene_split` |
| Medium | Full baseline | `scripts/train_baseline_full.py` |
| Medium | Distillation ablation | `scripts/ablate_distillation.py` |
| Medium | Reliability ablation | `scripts/ablate_reliability.py` |
| Medium | VK encoder ablation | `scripts/ablate_encoder.py` |

## Paper Statement

We propose a privacy-first multimodal HPE framework that converts raw visual input into sparse visual keypoints, learns full-modality structural knowledge with a privileged teacher, and distills this knowledge into students that remain robust under arbitrary modality loss.
