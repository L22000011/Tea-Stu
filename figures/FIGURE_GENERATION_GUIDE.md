# VK-RMD Paper Figure Generation Guide

This guide records how to regenerate the six review-support figures used by `Paper/main.tex`.

## Generated Figures

The current figure files are:

- `fig1_overall_framework.png`: overall method framework.
- `fig2_missing_modality_setting.png`: missing-modality protocol.
- `fig3_main_results.png`: HPE/HAR main results.
- `fig4_missing_modality_robustness.png`: robustness grouped by missing-modality count.
- `fig5_ablation.png`: ablation summary.
- `fig6_reliability_weights.png`: reliability-weight visualization.

## Regenerate All Figures Locally

```bash
python E:/Deskbook/Tea/scripts/make_paper_figures.py --root E:/Deskbook/Tea
```

## Export Reliability Weights on the Server

The fusion modules already return reliability weights as `alphas`.

### HPE Student-VK

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HPE

CUDA_VISIBLE_DEVICES=0 python -u scripts/export_reliability_weights.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config configs/student_vk_missing.yaml \
  --checkpoint outputs/student_vk_missing/best.pth \
  --device cuda:0 \
  --output-csv outputs/eval/student_vk_reliability_weights.csv \
  --max-batches 100
```

### HAR Student-VK

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi/MMFi_HAR

CUDA_VISIBLE_DEVICES=0 python -u scripts/export_reliability_weights.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --config configs/student_vk_missing.yaml \
  --checkpoint outputs/student_vk_missing/best.pth \
  --device cuda:0 \
  --output-csv outputs/eval/student_vk_reliability_weights.csv \
  --max-batches 100
```

## Redraw Reliability Heatmap After Syncing CSVs Locally

After syncing the exported CSVs into:

- `E:/Deskbook/Tea/HPE/outputs/eval/student_vk_reliability_weights.csv`
- `E:/Deskbook/Tea/HAR/outputs/eval/student_vk_reliability_weights.csv`

run:

```bash
python E:/Deskbook/Tea/scripts/make_paper_figures.py --root E:/Deskbook/Tea
```

`fig6_reliability_weights.png` will automatically change from the placeholder workflow figure to a heatmap.
