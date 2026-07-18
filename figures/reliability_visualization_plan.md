# Figure 3 Plan: Reliability-Guided Fusion Visualization

**Caption:** Reliability weights estimated by VK-RMD under different sensor availability patterns. The visualization should show that fusion weights adapt to the currently available modalities rather than using fixed averaging.

**Required data not yet confirmed:**

- Per-sample or per-combination reliability weights from the trained Student.
- The corresponding modality set for each sample.
- HPE/HAR output confidence or error for each modality-wise candidate, if available.

**Minimum acceptable implementation:**

1. Extract reliability weights from the model during evaluation.
2. Average weights by modality combination.
3. Plot a heatmap with rows as modality combinations and columns as modalities.
4. Compare against uniform fusion using `ablation_uniform_fusion_all_combinations.csv`.

**If weights are unavailable:** keep this figure as `[TODO: add reliability-weight visualization]` and do not fabricate values.

