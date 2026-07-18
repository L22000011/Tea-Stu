# 两个最终 baseline 的公平复现方案

## 结论

最终主实验应采用两个 **adapted baselines**，而不是把不同视觉输入的官方表格包装成完全公平比较：

1. **Adapted X-Fi-VK + matched subset training**：最接近的多模态人体感知方法。
2. **Adapted MMANet full-to-subset baseline**：最接近的完整观测到不完整观测蒸馏方法。

现有 X-Fi official table 仍可作为 `official result`；PTA/COMPASS 只可作为同协议外部参照，不能替代下面的逐样本公平复现。

## Baseline A：Adapted X-Fi-VK

### 选择理由

X-Fi 是 MM-Fi 上同时覆盖 HPE/HAR、动态模态组合和模态专属编码器的最近邻方法。它比通用 missing-modality 网络更能检验 VK-RMD 的提升是否来自新融合/FSG，而不是只来自 backbone 或数据处理。

### 公平协议

- 数据：与 VK-RMD 使用完全相同的 MM-Fi dataloader 和逐样本 manifest。
- 输入：把 X-Fi 的 RGB 分支替换为同一份 17点 VK 和同一 `SkeletonPromptEncoder`；禁止拿 RGB-X-Fi 声称完全公平。
- 非视觉编码器：使用同一 X-Fi pretrained Depth/LiDAR/mmWave/WiFi backbone，并保持相同冻结策略。
- token：每模态32个、维度512。
- 训练：相同 seed、batch size、epoch、optimizer、最大 batch 数和 `drop_counts`；同一 batch 使用同一个采样子集。
- 协议：random、cross-subject、cross-scene 的训练/验证列表逐项相同。
- 指标：HPE 报 MPJPE/PA-MPJPE，HAR 报 Accuracy/Macro-F1；评估全部31/15组合。
- 唯一变化：融合器保留 X-Fi 的 X-Fusion，不使用 VK-RMD 的 uncertainty dual readout、FSG loss 或 reliability KL。

### 结果标签

`Adapted baseline (our reproduction under matched VK/subset protocol)`，不得称 official X-Fi result。

### 修改成本与价值

成本中等：已有 `HPE/legacy_xfi`、`HAR/legacy_xfi` 和统一 dataloader，但需要接入 SkeletonPromptEncoder 与当前 checkpoint/reporting。审稿价值最高，因为它直接回答“是不是只因输入和训练协议不同而胜过 X-Fi”。**最终采用。**

## Baseline B：Adapted MMANet

### 选择理由

MMANet（CVPR 2023）明确处理 incomplete multimodal learning，并使用蒸馏与模态感知正则，构成 FSG 最接近的正式发表参照。它比普通 no-KD 或 uniform fusion 更能回答“FSG 是否只是已有完整到不完整蒸馏的改名”。

### 公平协议

- 使用与 VK-RMD 完全相同的 VK、Depth、LiDAR、mmWave、WiFi 输入及冻结 backbone。
- 将 MMANet 的图像任务头替换成相同容量的 HPE regression head 或 HAR classification head；其 margin-aware distillation 和 modality-aware regularization保持原定义。
- teacher 与 student 参数规模和 VK-RMD 相同；teacher使用完整模态，student使用完全相同的随机子集序列。
- 不允许使用额外预训练数据、额外增强或更长训练。
- random、cross-subject、cross-scene 使用同一 manifest；报告同一全部组合与缺失强度分组。
- 若原损失无法定义到 HPE regression，应只在 HAR 中列为 adapted MMANet，HPE 不制造不可比数字。

### 结果标签

`Adapted MMANet (matched MM-Fi inputs, splits and budget)`，与论文原始数据集上的 official result 分开。

### 修改成本与价值

成本中高：需移植其蒸馏/正则而不是只抄名称；但只训练一组 random split 即可先形成 P0 比较，cross协议可作为下一阶段。审稿价值高，因为它直接检验 FSG 的增量。**最终采用；若HPE损失无法忠实迁移，则仅采用HAR版本并明确限制。**

## 未采用候选

| 候选 | 判定 | 原因 |
|---|---|---|
| Official X-Fi | 保留作外部参考，不是公平 adapted baseline | RGB/VK输入、训练过程和缺失采样不完全一致 |
| PTA 2026 | 补充比较 | HPE和官方random协议可比，但没有逐样本manifest与cross协议，且为近期工作 |
| COMPASS 2026 | 补充比较 | HAR非视觉子集可比，但没有逐样本manifest与cross协议 |
| ActionMAE | 不作为最终两项 | 主要面向动作识别与缺失重建，无法无歧义覆盖HPE回归 |
| SMIL | 不采用 | 设定偏严重缺失训练样本，不是当前同步传感器子集部署的最近邻 |
| Uniform fusion / no-KD | 继续作为内部消融 | 不是已发表外部方法，不能替代外部baseline |

## 相关正式论文

- X-Fi, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html
- MMANet, CVPR 2023: https://doi.org/10.1109/CVPR52729.2023.01919
- Towards Good Practices for Missing Modality Robust Action Recognition, AAAI 2023: https://doi.org/10.1609/aaai.v37i3.25378
- ShaSpec, CVPR 2023: https://doi.org/10.1109/CVPR52729.2023.01524
