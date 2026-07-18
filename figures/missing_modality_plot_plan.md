# Figure 2 Plan: Missing-Modality Robustness

**Caption:** Performance under different available modality combinations. The proposed VK-RMD Student maintains lower HPE error and higher HAR accuracy than reproduced X-Fi/VK baselines and fixed-fusion variants when sensor modalities are missing.

**Recommended plots:**

1. **HPE bar plot:** average MPJPE for reproduced X-Fi/VK baseline, Student-VK, Student-NV, uniform fusion, and no-distillation variant.
2. **HAR bar plot:** average accuracy for reproduced X-Fi/VK baseline, Student-VK, Student-NV, uniform fusion, and no-distillation variant.
3. **Protocol plot:** random split vs cross-scene vs cross-subject for Student-VK and Student-NV.

**Use files:**

- `tables/main_hpe_results.csv`
- `tables/main_har_results.csv`
- `tables/ablation_results.csv`
- `tables/baseline_comparison.csv`

**Do not plot in main figure:**

- SuperTeacher missing-combination averages.
- Full-modality teacher missing-combination averages with meter-level HPE errors.
- Single-modality collapse cases unless used in a dedicated failure-case panel.

