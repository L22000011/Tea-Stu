# 隐私泄露 Probe 结果分析

## 1. 实验目的

该实验使用轻量 subject-ID linear probe 评估不同表示是否携带身份相关信息。它不是形式化隐私证明，也不是匿名性评估；它用于限制论文中的隐私表述边界。

实验采用 `action-holdout` 划分，即同一 subject 的训练动作和测试动作不同，比随机 frame split 更严格。

## 2. 关键结果

- Chance accuracy: 2.50%。
- Defaced RGB subject-ID accuracy: 67.53%。
- VK subject-ID accuracy: 15.43%。
- Depth subject-ID accuracy: 43.80%。
- LiDAR subject-ID accuracy: 36.57%。
- mmWave subject-ID accuracy: 9.72%。
- WiFi-CSI subject-ID accuracy: 8.65%。

## 3. 结论

1. Defaced RGB 仍然具有最高身份泄露风险。即使去脸后，RGB 仍可能保留衣着、体型、背景、场景和外观线索。

2. VK 的身份泄露明显低于 defaced RGB：VK 为 15.43%，defaced RGB 为 67.53%。这支持 `reduced visual exposure`，但不支持 `privacy-preserving`。

3. VK 仍显著高于 chance：15.43% vs 2.50%。这说明 VK 保留身体比例、骨架结构或动作习惯等身份相关线索，不能声称匿名或 identity-free。

4. Depth 和 LiDAR 也存在明显身份泄露，分别为 43.80% 和 36.57%。因此非 RGB 不等于隐私安全。

5. mmWave 和 WiFi-CSI 的 subject-ID accuracy 较低但仍高于 chance，说明无线/雷达信号也可能携带身份相关行为或空间线索。

## 4. 论文推荐写法

> We evaluate identity leakage using a lightweight subject-identification probe under an action-holdout split. Defaced RGB shows the highest leakage, while VK substantially reduces identity predictability compared with defaced RGB. However, VK remains above chance, indicating that keypoint geometry still carries body-structure and motion-style cues. Therefore, our claim is limited to reduced visual exposure and RGB-free downstream inference, not formal privacy preservation or anonymity.

中文：

> 我们使用 action-holdout subject-ID probe 评估不同表示的身份泄露风险。结果显示，defaced RGB 仍然具有最高泄露风险，而 VK 相比 defaced RGB 明显降低了身份可识别性。然而，VK 仍显著高于随机猜测，说明关键点几何仍包含身体结构和动作风格线索。因此，本文只主张降低视觉外观暴露和 RGB-free downstream inference，不主张形式化隐私保护或匿名性。

## 5. 不应写的结论

- 不写 VK is privacy-preserving。
- 不写 VK is anonymous / identity-free。
- 不写 non-RGB modalities are privacy-safe。
- 不写该实验证明了隐私安全。

## 6. 输出文件

- `privacy_leakage_probe_subject_id_summary.csv`
- `fig_privacy_leakage_subject_id.png`
- `fig_privacy_leakage_subject_id.svg`

## 7. 汇总表

| Modality | Subject-ID acc (%) | Chance (%) | Times chance | Reduction vs defaced RGB (%) |
|---|---:|---:|---:|---:|
| rgb | 67.53 | 2.50 | 27.01 | 0.00 |
| vk | 15.43 | 2.50 | 6.17 | 77.15 |
| depth | 43.80 | 2.50 | 17.52 | 35.14 |
| lidar | 36.57 | 2.50 | 14.63 | 45.84 |
| mmwave | 9.72 | 2.50 | 3.89 | 85.60 |
| wifi-csi | 8.65 | 2.50 | 3.46 | 87.19 |