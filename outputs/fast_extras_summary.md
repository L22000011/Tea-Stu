# Fast Submission Extras Summary

This report summarizes lightweight experiments and result reorganization for VK-RMD.

## Fair Missing-Modality Baselines

The fair-baseline table compares official/reproduced X-Fi, full-modality baselines under missing tests, uniform fusion, w/o KD, Student-VK, and Student-NV.

- HPE | Official X-Fi/VK table | avg=103.70 | full=83.70 | MPJPE mm lower-better
- HPE | Full-modality baseline under missing test | avg=617.17 | full=49.14 | MPJPE mm lower-better
- HPE | Uniform-fusion ablation | avg=82.77 | full=52.28 | MPJPE mm lower-better
- HPE | w/o KD ablation | avg=72.68 | full=48.83 | MPJPE mm lower-better
- HPE | VK-RMD Student-VK | avg=75.18 | full=50.17 | MPJPE mm lower-better
- HPE | VK-RMD Student-NV | avg=84.68 | full=51.03 | MPJPE mm lower-better
- HAR | Official/reproduced X-Fi/VK | avg=NA | full=NA | Accuracy % higher-better
- HAR | Full-modality teacher under missing test | avg=66.25 | full=95.57 | Accuracy % higher-better
- HAR | Uniform-fusion ablation | avg=67.46 | full=95.39 | Accuracy % higher-better
- HAR | w/o KD ablation | avg=82.86 | full=95.99 | Accuracy % higher-better
- HAR | VK-RMD Student-VK | avg=85.15 | full=96.56 | Accuracy % higher-better
- HAR | VK-RMD Student-NV | avg=80.62 | full=96.12 | Accuracy % higher-better

## Subset Severity

- HAR missing=0 | avg=96.56 | combos=1 | Accuracy %
- HAR missing=1 | avg=93.83 | combos=4 | Accuracy %
- HAR missing=2 | avg=87.55 | combos=6 | Accuracy %
- HAR missing=3 | avg=70.00 | combos=4 | Accuracy %
- HPE missing=0 | avg=50.17 | combos=1 | MPJPE mm
- HPE missing=1 | avg=53.55 | combos=5 | MPJPE mm
- HPE missing=2 | avg=60.78 | combos=10 | MPJPE mm
- HPE missing=3 | avg=78.10 | combos=10 | MPJPE mm
- HPE missing=4 | avg=124.76 | combos=5 | MPJPE mm

## Leave-One-Modality Degradation

- HPE missing vk: degradation=1.61 (MPJPE mm)
- HPE missing depth: degradation=14.07 (MPJPE mm)
- HPE missing lidar: degradation=0.24 (MPJPE mm)
- HPE missing mmwave: degradation=0.88 (MPJPE mm)
- HPE missing wifi-csi: degradation=0.08 (MPJPE mm)
- HAR missing vk: degradation=0.12 (Accuracy %)
- HAR missing depth: degradation=3.83 (Accuracy %)
- HAR missing lidar: degradation=0.00 (Accuracy %)
- HAR missing mmwave: degradation=6.94 (Accuracy %)

## Reliability-Performance Correlation

- HPE: Pearson=0.9396, Spearman=0.9000
- HAR: Pearson=0.8833, Spearman=0.8000

## Reliability Corruption Outputs

- /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HPE/outputs/eval/reliability_corruption_hpe.csv
- /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HAR/outputs/eval/reliability_corruption_har.csv

## Paper Usage

- Use the fair-baseline table to address the concern that VK-RMD only beats weak baselines.
- Use subset severity and leave-one-out analyses to support subset-invariant missing-modality learning.
- Use reliability corruption and correlation as evidence that reliability weights are learned task-level contribution proxies, not physical sensor calibration.
- Keep privacy claims bounded to reduced visual exposure and RGB-free inference.