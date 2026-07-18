# 面向 SCI 二区的论文冲刺指导书

日期：2026-06-19  
项目：MMFi RGB-free 任意模态缺失鲁棒人体感知  
任务：HPE + HAR  
核心方法：VK-RCD / full-modality teacher distillation / missing-modality training / reliability-calibrated fusion  

---

## 1. 直接结论

当前工作**不建议继续主打 Super-Teacher**。已有实验显示，简单 HPE+HAR 多任务 Super-Teacher 并没有稳定超过单任务 teacher 蒸馏结果，因此它不适合作为主贡献。

当前最适合投稿 SCI 二区的主线是：

> 面向隐私保护和传感器缺失场景，提出一个 RGB-free 的任意模态鲁棒人体感知框架，使 HPE 与 HAR 在没有原始 RGB 的情况下，仍能利用 VK、Depth、LiDAR、mmWave、WiFi-CSI 等模态完成稳定推理。

一句话主张：

> We propose a reliability-calibrated missing-modality distillation framework for RGB-free multimodal human perception, enabling robust HPE and HAR under arbitrary modality availability.

当前工作如果直接冲 CCF-A，风险很高；如果整理成传感器融合、隐私友好、缺失模态鲁棒的完整系统论文，**SCI 二区有现实机会**。

---

## 2. 当前工作等级判断

### 2.1 CCF-A 判断

当前版本距离 CCF-A 仍有明显差距。

主要原因：

1. 方法创新更像模块组合：VK 替代 RGB、蒸馏、随机模态缺失、可靠性融合都不是全新概念。
2. Super-Teacher 实验没有形成正向结果，不能作为强贡献。
3. 理论机制还不够强，缺少明确的任务定义、风险建模或泛化理论。
4. 目前主要依赖 MMFi 单一数据集，跨数据集与真实部署证据不足。
5. 隐私性目前更多是动机，缺少定量隐私泄露实验。

判断：

| 目标 | 当前概率 | 补齐实验后概率 |
|---|---:|---:|
| CCF-A | < 3% | 5%-8% |
| CCF-B | 8%-15% | 15%-25% |
| SCI 二区 | 30%-40% | 50%-65% |
| SCI 三区 / CCF-C | 50%-65% | 70%+ |

### 2.2 当前最像什么级别文章

当前最像：

- SCI 二区传感器融合 / 智能感知方向论文；
- 或 CCF-C / SCI 三区较稳论文；
- 如果实验完整、故事线干净，有机会冲 SCI 二区。

不建议现在把主目标定为 CCF-A。

---

## 3. 最强主线选择

候选主线里，最适合当前工作的不是 Super-Teacher，也不是“VK 替代 RGB”本身，而是：

> **RGB-free arbitrary-modality robust human perception**

中文表述：

> 面向隐私保护的 RGB-free 任意模态缺失鲁棒人体感知。

这个主线最好，因为它把所有模块串成一个因果链：

1. 原始 RGB 有隐私风险；
2. VK 保留人体结构信息，同时弱化身份、纹理、背景等敏感信息；
3. 现实部署中传感器会缺失或质量不稳定；
4. full-modality teacher 提供稳定上界知识；
5. student 在训练时随机采样模态子集；
6. 每个模态产生候选预测；
7. reliability head 根据当前模态组合动态分配权重；
8. 最终支持任意模态组合下的 HPE 和 HAR。

不建议作为主线的方向：

| 方向 | 是否建议主打 | 原因 |
|---|---|---|
| RGB-VK Super-Teacher | 不建议 | 当前结果不支持，且简单多任务 teacher 易被认为普通 |
| 单纯 VK 替代 RGB | 不建议 | 创新强度不够 |
| 单纯随机缺失训练 | 不建议 | 已有 missing modality learning 工作较多 |
| 单纯可靠性融合 | 不建议 | 容易被认为 attention/uncertainty variant |
| HPE+HAR 统一框架 | 辅助贡献 | 当前 Super-Teacher 不够强，不能主打 |
| RGB-free 任意模态鲁棒感知 | 建议主打 | 最能覆盖隐私、鲁棒性、传感器融合和部署价值 |

---

## 4. 方法应如何重新命名

建议方法名：

**RC-MMD: Reliability-Calibrated Missing-Modality Distillation for RGB-Free Human Perception**

中文：

**可靠性校准的缺失模态蒸馏框架**

核心模块：

1. **VK Privacy-Preserving Representation**
   - 用 VK 替代 RGB 图像输入；
   - 保留人体结构；
   - 减少身份、纹理、背景隐私泄露。

2. **Full-to-Partial Modality Distillation**
   - teacher 使用全模态；
   - student 使用随机缺失模态；
   - student 学习从任意子集逼近 full-modality teacher。

3. **Modality-wise Candidate Prediction**
   - 每个可用模态输出候选 pose/logits；
   - 避免简单拼接后黑盒融合。

4. **Reliability-Calibrated Dynamic Fusion**
   - 根据当前模态质量估计 reliability；
   - 动态分配融合权重；
   - 支持不同模态组合。

5. **Arbitrary-Modality Inference**
   - HPE 支持 31 种组合；
   - HAR 支持 15 种组合；
   - 推理阶段不要求固定模态输入。

---

## 5. 当前代码是否支撑该主线

结论：**基本支撑。**

代码中已经实现：

1. 训练时随机采样模态子集；
2. full-modality teacher 监督 partial-modality student；
3. 每个模态产生候选预测；
4. reliability head 输出动态融合权重；
5. HPE 31 组合、HAR 15 组合评估。

需要注意：

- HPE-NV 的 `drop_counts` 包含 `0`，论文中不能写“每次都缺失”，应写“随机采样完整或部分非视觉模态组合”。
- Super-Teacher 结果不应作为主线证据。
- 当前论文应围绕 HPE/HAR 主线代码，而不是 Super-Teacher 代码。

---

## 6. 三个最强贡献点

建议最终贡献写成：

### Contribution 1: RGB-free multimodal human perception

We introduce a privacy-preserving RGB-free multimodal human perception framework that replaces raw RGB images with visual keypoints while integrating heterogeneous sensing modalities including depth, LiDAR, mmWave, and WiFi-CSI.

### Contribution 2: Full-to-part missing-modality distillation

We propose a full-to-part distillation strategy where a full-modality teacher guides a student trained with randomly sampled modality subsets, enabling robust inference under arbitrary modality availability.

### Contribution 3: Reliability-calibrated dynamic fusion

We design a reliability-calibrated fusion module that predicts modality-wise candidates and dynamically weights available modalities according to their estimated uncertainty, improving robustness under sensor degradation and modality missingness.

---

## 7. 必须保留的实验

### 7.1 HPE 主实验

必须包含：

- full-modality teacher；
- Student-VK 31 组合；
- Student-NV 非视觉组合；
- baseline full model；
- no distillation；
- no reliability fusion；
- simple VK encoder / skeleton-prompt VK encoder；
- official X-Fi 或 MMFi baseline 对比。

主文不建议放完整 31 行大表。主文放汇总：

| 设置 | 单模态平均 | 双模态平均 | 三模态平均 | 四模态平均 | 全模态 | 全组合平均 |
|---|---:|---:|---:|---:|---:|---:|

完整 31 组合放 appendix。

### 7.2 HAR 主实验

必须包含：

- full-modality teacher；
- Student-VK 15 组合；
- Student-NV 非视觉组合；
- baseline full model；
- no distillation；
- no reliability fusion；
- official baseline 对比。

同样主文放汇总，完整 15 组合放 appendix。

### 7.3 消融实验

最必要的消融：

1. w/o distillation；
2. w/o reliability fusion；
3. w/o random modality dropping；
4. simple VK encoder vs skeleton-prompt VK encoder；
5. Student-VK vs Student-NV；
6. full-modality teacher vs baseline full model。

不建议继续大规模做 Super-Teacher 消融。

### 7.4 鲁棒性实验

建议至少保留：

1. random missing modality；
2. fixed strong modality missing；
3. VK noise robustness；
4. cross-scene；
5. cross-subject 如果已有结果完整则加入。

### 7.5 隐私分析

SCI 二区非常需要把“隐私友好”做实。

建议补：

1. RGB 与 VK 可视化对比；
2. 身份信息保留程度分析；
3. 背景信息泄露分析；
4. VK 下游性能保持；
5. 隐私-性能 trade-off 图。

如果没有时间做复杂攻击实验，至少做定性可视化 + 讨论，不要过度声称“严格隐私保护”。

---

## 8. 审稿人最可能攻击的问题与回应

### 1. 这是不是模块堆叠？

风险：高。  
回应方式：

- 不按模块罗列贡献；
- 强调统一问题：RGB-free arbitrary-modality robust perception；
- 用 full-to-part distillation + reliability fusion 的因果链解释每个模块必要性。

### 2. VK 替代 RGB 是否有创新？

风险：中高。  
回应方式：

- 不把 VK 本身当唯一创新；
- 把 VK 作为隐私友好中间表示；
- 展示 VK 在隐私和性能之间的平衡。

### 3. 随机模态缺失训练是否已有？

风险：高。  
回应方式：

- 承认已有 missing modality learning；
- 强调你的设定是 MMFi 多传感器人体感知 + HPE/HAR + RGB-free + reliability-calibrated distillation。

### 4. reliability fusion 是否只是 attention？

风险：中。  
回应方式：

- 明确 reliability 基于每模态预测不确定性；
- 与 uniform fusion、attention fusion 做消融；
- 展示不同场景下权重变化。

### 5. 为什么不用 RGB？

风险：中。  
回应方式：

- 目标不是追求 RGB 最强，而是隐私友好部署；
- 用 VK 保留人体结构，降低原始视觉隐私泄露。

### 6. 只有 MMFi 一个数据集够吗？

风险：中高。  
回应方式：

- 加 cross-scene、cross-subject；
- 强调 MMFi 是多模态人体感知标准数据集；
- 如果不能跨数据集，需在 limitation 中诚实说明。

### 7. HPE 和 HAR 是否真的统一？

风险：中。  
回应方式：

- 不强推 Super-Teacher；
- 写成同一框架在两个任务上的验证，而不是一个 foundation model。

### 8. Super-Teacher 为什么不用？

风险：低。  
回应方式：

- 不在主文强推；
- 可在 discussion 中说明 naive cross-task teacher 未稳定提升，未来需要更强跨任务对齐机制。

---

## 9. 论文结构建议

### Title 候选

1. Reliability-Calibrated Missing-Modality Distillation for RGB-Free Multimodal Human Perception
2. RGB-Free Multimodal Human Perception under Arbitrary Modality Missingness
3. Robust Human Pose Estimation and Activity Recognition with Privacy-Preserving Multimodal Sensing
4. Full-to-Partial Modality Distillation for RGB-Free Human Perception
5. Privacy-Preserving Multimodal Human Perception via Reliability-Calibrated Distillation

最推荐：

**Reliability-Calibrated Missing-Modality Distillation for RGB-Free Multimodal Human Perception**

### Abstract 逻辑

1. 背景：多模态人体感知依赖 RGB 与固定模态组合；
2. 问题：RGB 有隐私风险，真实部署中传感器会缺失；
3. 方法：提出 RC-MMD，使用 VK 替代 RGB，full-modality teacher 蒸馏 partial-modality student，reliability 动态融合；
4. 实验：在 MMFi 上验证 HPE 和 HAR，支持 31/15 种组合；
5. 结果：缺失模态下优于 baseline 和消融方法。

### Introduction 四段式

1. 多模态人体感知的重要性与 MMFi 场景；
2. RGB 隐私风险与固定模态假设不现实；
3. 现有方法不足：要么依赖 RGB，要么不支持任意缺失，要么融合不可靠；
4. 本文贡献：RGB-free、full-to-part distillation、reliability-calibrated fusion、HPE/HAR 验证。

### Method 章节

1. Problem Formulation；
2. RGB-free VK Representation；
3. Modality-specific Token Encoding；
4. Full-to-Partial Missing-Modality Distillation；
5. Reliability-Calibrated Fusion；
6. Training Objective；
7. Inference under Arbitrary Modality Availability。

### Experiments 章节

1. Dataset and Protocols；
2. Implementation Details；
3. Main Results on HPE；
4. Main Results on HAR；
5. Missing-Modality Robustness；
6. Ablation Studies；
7. Cross-Scene / Cross-Subject Generalization；
8. Privacy and Interpretability Analysis；
9. Failure Cases。

---

## 10. 未来 4 周冲刺路线

### 第 1 周：结果清点与主表重做

目标：确认哪些结果可用，哪些必须补。

必须做：

- 汇总 HPE 31 组合；
- 汇总 HAR 15 组合；
- 汇总 Student-VK / Student-NV；
- 汇总 no distillation / no reliability；
- 生成主文摘要表。

不要做：

- 不要继续 Super-Teacher；
- 不要重训已有可用 teacher；
- 不要做大而全的新 pipeline。

### 第 2 周：补关键缺口

目标：补齐 SCI 二区最低实验闭环。

优先级：

1. HPE Student-NV 非视觉组合；
2. HAR Student-VK 组合；
3. HAR Student-NV 非视觉组合；
4. no reliability fusion；
5. no distillation。

### 第 3 周：泛化与隐私分析

目标：提高论文可信度。

必须做：

- cross-scene 至少一组；
- VK vs RGB 可视化；
- reliability 权重可视化；
- 失败案例分析。

可选：

- cross-subject；
- 噪声扰动；
- 传感器质量下降实验。

### 第 4 周：写论文

目标：形成可投版本。

必须完成：

- Abstract；
- Introduction；
- Method；
- Experiments；
- Discussion；
- Limitations；
- 所有表格和图。

---

## 11. 投稿策略

### 首选方向

SCI 二区传感器融合 / 智能感知 / 多媒体应用方向。

推荐定位：

1. IEEE Sensors Journal；
2. Sensors；
3. Measurement；
4. Neural Computing and Applications；
5. Multimedia Tools and Applications；
6. Engineering Applications of Artificial Intelligence。

### 不建议当前直接冲

- CVPR / ICCV / ECCV；
- NeurIPS / ICLR；
- ACM MM；
- SenSys / UbiComp。

原因：

- 理论新意不足；
- 缺少真实部署系统；
- 单数据集风险较高；
- Super-Teacher 没有正结果支撑。

### CCF-B 可作为冲刺备选

可以尝试：

- ICASSP；
- ICME；
- PerCom。

但不建议押宝。

---

## 12. 最终行动建议

### 当前是否值得冲 CCF-A？

不建议。

### 当前是否值得写 SCI 二区？

值得。

### 当前最强卖点是什么？

不是 Super-Teacher，而是：

> RGB-free + 任意模态缺失 + full-to-part distillation + reliability-calibrated fusion + HPE/HAR 双任务验证。

### 当前最大硬伤是什么？

1. 创新容易被认为模块组合；
2. Super-Teacher 结果不支持主贡献；
3. 隐私分析还不够定量；
4. 泛化实验需要进一步整理；
5. 需要把实验表从“堆结果”重组为“鲁棒性证据链”。

### 哪些模块保留？

保留：

- VK 替代 RGB；
- full-modality teacher；
- random modality subset training；
- reliability fusion；
- HPE/HAR 双任务验证；
- Student-VK / Student-NV。

弱化：

- Super-Teacher；
- 跨任务统一 teacher；
- 过多复杂消融。

删除或后置：

- Super-Teacher 作为主贡献；
- CCF-A 级 foundation framework 叙事；
- 没有结果支撑的“统一多任务 teacher”表述。

### 最优实验优先级

1. HPE 31 组合主表；
2. HAR 15 组合主表；
3. Student-NV 非视觉鲁棒性；
4. no distillation；
5. no reliability；
6. cross-scene；
7. privacy visualization；
8. reliability visualization；
9. failure cases。

---

## 13. 我的最终指导

你现在最需要的不是继续加新模型，而是**收缩主线、清理实验、马上写论文**。

正确方向：

> 把工作写成一篇面向真实传感器部署问题的 RGB-free 任意模态缺失鲁棒人体感知论文。

不要再把论文写成：

> 我们做了 Super-Teacher、VK、蒸馏、缺失训练、可靠性融合。

而要写成：

> 真实人体感知部署中，RGB 有隐私风险，传感器会缺失。我们提出一个统一的 RGB-free full-to-part distillation 框架，使模型能根据当前可用模态动态选择可靠信息，在 HPE 和 HAR 中都保持稳定性能。

如果你的结果表整理后能证明：

1. Student 在缺失模态下优于 baseline；
2. reliability fusion 优于 uniform/attention；
3. distillation 明显有效；
4. VK 在隐私友好前提下保持性能；
5. HPE 和 HAR 都有一致趋势；

那么 SCI 二区是现实目标。

当前不要再追 CCF-A。  
先把 SCI 二区写出来，拿到一篇稳定论文，再考虑后续把 Super-Teacher 或跨任务统一 teacher 做成第二篇增强工作。

