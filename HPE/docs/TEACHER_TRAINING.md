# Teacher Training Notes

## What the Teacher Loads

The VK-RCD Teacher uses the original X-Fi pretrained sensor backbones under `backbones/`.

Expected cloud layout:

```text
MMFi_HPE/
  backbones/
    depth_benchmark/depth_Resnet18.pt
    lidar_benchmark/lidar_all_random.pt
    mmwave_benchmark/mmwave_all_random_TD.pt
    CSI_benchmark/protocol3_random_1.pkl
```

These four pretrained files are loaded:

| Modality | File |
|---|---|
| Depth | `backbones/depth_benchmark/depth_Resnet18.pt` |
| LiDAR | `backbones/lidar_benchmark/lidar_all_random.pt` |
| mmWave | `backbones/mmwave_benchmark/mmwave_all_random_TD.pt` |
| WiFi-CSI | `backbones/CSI_benchmark/protocol3_random_1.pkl` |

The RGB pretrained file is not used:

```text
backbones/RGB_benchmark/RGB_Resnet18.pt
```

Reason: our visual branch input is not a real RGB image. MM-Fi provides `rgb/*.npy` visual keypoints with shape `17 x 2`, so the VK branch is a new `SkeletonPromptEncoder` initialized from scratch.

## What Is Trained

By default, pretrained Depth/LiDAR/mmWave/WiFi-CSI backbones are frozen and kept in eval mode.

The Teacher trains:

1. `SkeletonPromptEncoder` for VK.
2. Token projectors.
3. Cross-modal token encoder.
4. Reliability fusion and pose heads.

This follows the original X-Fi idea of reusing pretrained sensor feature extractors, while replacing RGB ResNet with a keypoint-based skeleton prompt branch.

## Teacher Command

Run from the project root that contains `backbones/`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_teacher_full.py \
  --dataset /path/to/MMFi_Dataset \
  --config configs/teacher_full.yaml \
  --device cuda:0
```

If `backbones/` is not under the project root:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_teacher_full.py \
  --dataset /path/to/MMFi_Dataset \
  --config configs/teacher_full.yaml \
  --backbone-root /path/to/MMFi_HPE/backbones \
  --device cuda:0
```

Resume from interruption:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_teacher_full.py \
  --dataset /path/to/MMFi_Dataset \
  --config configs/teacher_full.yaml \
  --resume outputs/teacher_full/last.pth \
  --device cuda:0
```

## Output

The Teacher writes:

```text
outputs/teacher_full/best.pth
outputs/teacher_full/last.pth
outputs/teacher_full/epoch_005.pth
outputs/teacher_full/final_eval.csv
outputs/teacher_full/final_eval.md
outputs/teacher_full/epoch_history.csv
```

