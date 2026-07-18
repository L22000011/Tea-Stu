# IoT Deployment Benchmark for VK-RMD

This folder contains a deployment-oriented supplementary experiment for the IoTJ manuscript.

The goal is to measure whether VK-RMD is practical for sensor-rich IoT inference under missing modalities. This is **not a training experiment**. It only loads existing checkpoints and measures forward-pass runtime.

## What This Experiment Measures

- Parameters: model size.
- Model size: checkpoint-level parameter/buffer footprint in MB.
- Latency: mean/std inference time in milliseconds.
- FPS and samples/s: real-time sensing capability.
- Peak CUDA memory: GPU memory required during inference.
- Missing-modality inference cost: runtime under different available modality subsets.

The key claim supported by this experiment is:

> When fewer sensors are available, VK-RMD can skip unavailable modality encoders, so missing-modality inference changes not only accuracy but also runtime cost.

## Do We Need 100 Epoch + 1000 Iter?

No.

For IoT deployment benchmarking, we do **not** retrain any model. We use the already trained checkpoints:

- HPE Student-VK: `HPE/outputs/student_vk_missing/best.pth`
- HAR Student-VK: `HAR/outputs/student_vk_missing/best.pth`

Recommended benchmark setting:

- warmup: 30 iterations
- measured iterations: 200
- one validation batch

This is enough to report latency/FPS/memory on one NVIDIA L40 GPU. If time is very tight, use `--iters 100`.

## Cloud Commands

Run from the cloud repo root:

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi
```

HPE on GPU 0. If your cloud project uses `MMFi_HPE` instead of `HPE`, replace every `HPE/` below with `MMFi_HPE/`.

```bash
CUDA_VISIBLE_DEVICES=0 python -u IoT-Deployment-Benchmark/benchmark_vkrmd_deployment.py \
  --project HPE \
  --project-root HPE \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config HPE/configs/student_vk_missing.yaml \
  --checkpoint HPE/outputs/student_vk_missing/best.pth \
  --device cuda:0 \
  --warmup 30 \
  --iters 200 \
  --output-csv IoT-Deployment-Benchmark/results/deployment_efficiency_hpe_l40.csv \
  --manifest-json IoT-Deployment-Benchmark/results/deployment_efficiency_hpe_l40_manifest.json
```

HAR on GPU 1. If your cloud project uses `MMFi_HAR` instead of `HAR`, replace every `HAR/` below with `MMFi_HAR/`.

```bash
CUDA_VISIBLE_DEVICES=1 python -u IoT-Deployment-Benchmark/benchmark_vkrmd_deployment.py \
  --project HAR \
  --project-root HAR \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config HAR/configs/student_vk_missing.yaml \
  --checkpoint HAR/outputs/student_vk_missing/best.pth \
  --device cuda:0 \
  --warmup 30 \
  --iters 200 \
  --output-csv IoT-Deployment-Benchmark/results/deployment_efficiency_har_l40.csv \
  --manifest-json IoT-Deployment-Benchmark/results/deployment_efficiency_har_l40_manifest.json
```

Important: with `CUDA_VISIBLE_DEVICES=1`, the script still uses `--device cuda:0`, because the process sees the selected physical GPU as logical GPU 0.

## Default Modality Subsets

HPE:

- V
- D
- R
- V+D
- D+R
- V+D+L+R+W

HAR:

- V
- D
- R
- V+D
- D+R
- V+D+L+R

Notation:

- V = Visual Keypoints
- D = Depth
- L = LiDAR
- R = mmWave radar
- W = WiFi-CSI

## Paper Usage

Use this as a compact table in the IoTJ paper:

`Task | Model | Modalities | Params | Model size | Latency | FPS | Samples/s | Peak memory`

Recommended wording:

> To assess deployment feasibility in sensor-rich IoT environments, we report measured latency, FPS and peak GPU memory under representative modality subsets on one NVIDIA L40 GPU. Unlike theoretical FLOPs, measured latency better reflects heterogeneous modality encoders and missing-modality execution.

Do not claim Jetson or real edge deployment unless you also test on that device.
