# origin-XFI

This folder is a fairness-oriented baseline rebuilt from the official X-Fi repository structure:

- Source repository: `https://github.com/NTUMARS/X-Fi`
- Tasks included:
  - `HPE/`
  - `HAR/`

## Purpose

The original official X-Fi code uses `RGB` as the visual modality.
This baseline keeps the official training and fusion structure as intact as possible, and only replaces the first modality from `RGB` to `VK`.

## What Changed

For both `HPE` and `HAR`, the following changes were applied:

- `config.yaml`: first modality changed from `rgb` to `vk`
- dataset loader:
  - logical modality name is now `vk`
  - on-disk folder still maps to `rgb`
  - `rgb/*.npy` is loaded as a `17x2` VK array
- visual encoder:
  - the original RGB-ResNet branch is replaced by a lightweight VK MLP
  - the output is still projected to the official `49 x 512` token shape
  - downstream projector and fusion modules stay in the official format
- optimizer:
  - the new VK encoder parameters are included in training
- validation output:
  - printed modality names use `VK` instead of `RGB`

## What Stayed Official

The following were intentionally kept in the official X-Fi style:

- folder split: `HPE/` and `HAR/`
- entry scripts:
  - `run.py`
  - `validate_all.py`
  - `evaluate.py` for HPE
- fusion backbone and modality ordering
- protocol and split configuration format
- checkpoint and loader entry patterns
- official `baseline1/` and `baseline2/` directory layout

## Backbone Placeholders

Empty backbone folders were created so you can place the official support files on the server later.

- `HPE/backbones/CSI_benchmark`
- `HPE/backbones/RGB_benchmark`
- `HPE/backbones/depth_benchmark`
- `HPE/backbones/lidar_benchmark`
- `HPE/backbones/mmwave_benchmark`
- `HAR/backbones/RGB_benchmark`
- `HAR/backbones/depth_benchmark`
- `HAR/backbones/lidar_benchmark`
- `HAR/backbones/mmwave_benchmark`

## Baseline Subfolders

Both tasks now also include the official baseline subfolders with the same `RGB -> VK` adaptation:

- `baseline1/`: train modality-specific or combination-specific official baselines
- `baseline2/`: generate single-modality outputs and fuse them with the official late-fusion logic

For `baseline2/`, the single-modality checkpoint names were aligned with `baseline1/` outputs:

- `vk_.pt`
- `depth_.pt`
- `mmwave_.pt`
- `lidar_.pt`
- `wifi-csi_.pt` for HPE

## Expected VK Data Location

The code expects the logical modality to be `vk`, but it reads data from the on-disk `rgb` folder.
This is intentional, because the current MMFi dataset layout stores the VK `.npy` files under `rgb/`.

## Run Entry

HPE:

```bash
cd origin-XFI/HPE
python run.py --dataset /path/to/MMFi_DATA
python validate_all.py --dataset /path/to/MMFi_DATA --pt_weights /path/to/checkpoint.pth
```

HAR:

```bash
cd origin-XFI/HAR
python run.py --dataset /path/to/MMFi_DATA
python validate_all.py --dataset /path/to/MMFi_DATA --pt_weights /path/to/checkpoint.pth
```
