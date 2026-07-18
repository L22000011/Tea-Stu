# Adapted X-Fi-VK baseline

This is a matched, protocol-controlled baseline. It uses the formal VK and
non-visual inputs, the same frozen modality backbones, the same random subset
training policy, and the same all-combination evaluation as VK-RMD. Its fusion
module keeps the X-Fi-style modality-specific KV projections, cross-modal
pooling, and cross-attention injection, while omitting SP-SFC and the VK-RMD
dual readout.

The result must be called `Adapted X-Fi-VK`, not official X-Fi and not a claim
of an exact reproduction of the original RGB implementation.

Cloud command (the process-local `cuda:0` is physical GPU 1 after
`CUDA_VISIBLE_DEVICES=1`):

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi
CUDA_VISIBLE_DEVICES=1 python -u Matched-XFi-VK/run_matched_xfi_vk.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 --tasks HPE HAR --protocols random \
  --epochs 10 --max-train-batches 300 --max-eval-batches 100
```

Outputs are written to `outputs/adapted_xfi_vk_baseline/` and can be resumed
from `last.pth`. Use `--dry-run` to print commands without starting training.
