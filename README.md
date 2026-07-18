# VK-RMD

VK-RMD is a multimodal human-sensing research project for HPE and HAR under dynamic sensor availability. The repository contains the model code, experiment pipelines, lightweight result artifacts, analysis scripts, figures, and manuscript sources used in the project.

## Repository contents

- `HPE/`, `HAR/`: task-specific models, losses, training code, configurations, and lightweight evaluation outputs.
- `scripts/`: shared evaluation, corruption, reliability, privacy, and submission-analysis utilities.
- `Dual-Readout-Ablation/`, `SP-SFC-Ablation/`, `Matched-XFi-VK/`: focused experiment pipelines.
- `RGB-Subset-Fairness/`, `privacy_leakage_probe/`, `IoT-Deployment-Benchmark/`: supporting feasibility, exposure, and deployment analyses.
- `outputs/`, `tables/`, `figures/`, `supplement/`, `reports/`: CSV/JSON summaries, plots, and diagnostic reports.
- `Paper/`, `Writing/`, `books/`: LaTeX manuscripts, bibliography files, paper figures, and research notes.

## Large files and data

Model checkpoints, pretrained weights, raw MMFi data, per-sample generated arrays, caches, logs, archives, and downloadable upstream repositories are intentionally excluded from Git. These files include `*.pth`, `*.pt`, `*.ckpt`, `*.npy`, raw RGB subsets, and local pretrained-weight directories.

The committed result artifacts are intended to preserve the reported numerical evidence without requiring multi-gigabyte model files. Checkpoints and datasets should be obtained or regenerated separately according to the relevant experiment configuration.

## Scope

The project studies a low-appearance Visual Keypoint interface, full-observation-to-subset consistency learning, and visible-modality chunk evidence fusion. Claims and limitations are documented in the manuscript and audit files; the repository does not claim formal privacy guarantees.
