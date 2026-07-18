# Super-Teacher for MMFi HPE and HAR

This project is an independent exploration branch. It does not modify the existing `HPE` or `HAR` folders.

## Goal

Train one privacy-friendly non-RGB multimodal Super Teacher on MMFi, using a shared body representation for:

- HPE pose regression.
- HAR action classification.
- Distillation to HPE students.
- Distillation to HAR students.
- Random missing-modality training and arbitrary modality-combination evaluation.

RGB images are not used. The visual branch is replaced by VK keypoints.

## Required Cloud Layout

Place the official backbone folder here:

```bash
Super-Teacher/backbones/
```

The default configs expect:

```bash
backbones/depth_benchmark/depth_Resnet18.pt
backbones/lidar_benchmark/lidar_all_random.pt
backbones/mmwave_benchmark/mmwave_all_random_TD.pt
backbones/CSI_benchmark/protocol3_random_1.pkl
```

## One-Command Pipeline

Run from the project root:

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/Super-Teacher

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000
```

Use dry-run first if you only want to inspect the task list:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_super_pipeline.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --gpu 0 \
  --device cuda:0 \
  --max-train-batches 1000 \
  --dry-run
```

## What The Pipeline Runs

For each protocol:

- `random_split`
- `cross_scene_split`
- `cross_subject_split`

It runs:

- Super Teacher training.
- Super Teacher HPE all-combination evaluation.
- Super Teacher HAR all-combination evaluation.
- HPE Student-VK training from the Super Teacher.
- HPE Student-VK all-combination evaluation.
- HPE Student-NV training from the Super Teacher.
- HPE Student-NV non-visual-combination evaluation.
- HAR Student-VK training from the Super Teacher.
- HAR Student-VK all-combination evaluation.
- HAR Student-NV training from the Super Teacher.
- HAR Student-NV non-visual-combination evaluation.

It also runs two necessary teacher ablations:

- HPE-only teacher.
- HAR-only teacher.

## Resume And Skip Rules

The pipeline skips a task when all expected outputs already exist, unless `--force` is passed.

Training scripts save:

```bash
best.pth
last.pth
epoch_010.pth
epoch_history.csv
config_used.yaml
dependencies.txt
experiment_info.json
final_eval_*.csv
final_eval_*.md
training_curve.png
```

Super Teacher additionally saves:

```bash
best_joint.pth
best_hpe.pth
best_har.pth
last.pth
failed.pth
```

Pipeline records are saved in:

```bash
outputs/super_runs/<timestamp>/
  manifest.json
  status.csv
  logs/
```

## Single Scripts

Train the Super Teacher:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_super_teacher.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config configs/super_teacher.yaml \
  --device cuda:0 \
  --max-train-batches 1000
```

Train HPE students:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_hpe_student_from_super.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config configs/hpe_student_from_super.yaml \
  --teacher outputs/super_teacher/best_joint.pth \
  --device cuda:0 \
  --max-train-batches 1000
```

Train HAR students:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_har_student_from_super.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config configs/har_student_from_super.yaml \
  --teacher outputs/super_teacher/best_joint.pth \
  --device cuda:0 \
  --max-train-batches 1000
```

## Notes

- Training uses `max_train_batches=1000` by default in the pipeline.
- Evaluation is full validation unless `--max-eval-batches` is explicitly passed.
- HAR currently uses `vk, depth, lidar, mmwave`.
- HPE uses `vk, depth, lidar, mmwave, wifi-csi`.
- The Super Teacher is a supplemental exploration. It should be compared against the already trained independent HPE and HAR teachers/students.
