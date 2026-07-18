# Missing Data Report

## 可用且已进入主文的结果
- X-Fi 风格 HPE/HAR 逐组合表：来自 `HPE/HAR/legacy_xfi` 与 `student_vk_random_all_combinations.csv`。
- FSG 缺失强度表：来自 Student-VK 与 Student-NV all-combination CSV。
- Leave-one-out 表：由 full-modality 结果与移除单一模态组合重算。
- 跨协议表：来自 `tables/main_hpe_results.csv` 与 `tables/main_har_results.csv`。
- 消融表：来自 `tables/ablation_results.csv`，保留 uniform fusion 与 w/o distillation 的谨慎解释。
- 隐私泄露探针：来自已整理结果，主文只声称 reduced visual exposure，不声称匿名。

## 缺失或不建议强写的结果
- 缺少 formal privacy guarantee、identity attack 的完整协议；因此不能写 privacy-preserving 或 anonymous。
- body-latent alignment 诊断已进入正文，但只能解释为弱对应性证据，不应作为严格语义对齐证明。
- reliability corruption 结果已进入正文，用于说明 learned contribution proxy 的响应边界，不是校准物理可靠性。
- SuperTeacher 结果已进入正文作为探索边界，但不进入主贡献。
- Ori-XFI 244 mm 异常复现结果不进入主表，避免误导性比较。

