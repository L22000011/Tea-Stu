# 三个核心审稿质疑的回应与论文降风险策略

本文档针对 VK-RMD 当前创新布局中最可能被审稿人抓住的三个问题进行预审式分析。目标不是回避问题，而是把质疑转化为论文中可防守的叙事、证据边界、必要降调和可选补强。

总体判断：

> 当前工作有完整的系统逻辑，但不是强理论证明型创新。最稳的创新定位是：**VK/Depth 低外观结构锚点 + task-supervised body-latent tokenization + subset-conditioned missing-modality learning + full-to-subset guidance**。论文必须避免把诊断性证据写成严格证明，也不能把已有模块重新命名后声称为完全原创算法。

---

## 1. 质疑一：Body-latent token 对齐是否只是普通 Transformer 融合的重新命名？

### 1.1 审稿人可能怎么问

审稿人很可能会质疑：

> 你所谓的 body-latent token alignment，本质上是不是就是 modality-specific encoder + projector + modality embedding + Transformer fusion？这和常见多模态 Transformer 融合有什么区别？

更尖锐的问题是：

> 你如何证明 LiDAR、mmWave、WiFi-CSI token 不是普通任务特征，而是被 VK/Depth 锚定到了 body-latent space？

### 1.2 这个质疑为什么成立

这个质疑有道理。当前代码层面确实主要实现了：

- 每个模态通过自己的 encoder 提取特征；
- 通过 projector 映射到统一 token 维度；
- 加 modality embedding；
- 进入 Transformer encoder；
- 通过 HPE/HAR 任务监督训练。

这可以证明模型进行了 **token-space fusion**，但不能自动证明形成了严格意义上的“人体语义对齐”。如果论文直接写成“我们证明所有异构传感器都被对齐到人体语义空间”，会被认为证据不足。

### 1.3 当前已有证据能支撑什么

当前已有证据可以支撑较稳的结论：

1. **代码证据**  
   `TokenProjector + modality embedding + TransformerEncoder` 说明不同模态被映射到统一维度，并在 token 空间发生交互。

2. **任务监督证据**  
   HPE 和 HAR 共同监督说明 token 表示被优化为人体状态相关，而不是无意义的统一维度特征。

3. **VK 语义扰动诊断证据**  
   `vk_semantic_anchor_conclusion.md` 中显示：
   - HPE 中 VK joint shuffle 会带来约 `+10.10 mm` MPJPE 退化；
   - HAR 中 VK joint shuffle 会带来约 `1.19 pp` accuracy drop。

   这说明模型利用了 VK 的 joint identity 和 skeleton organization，而不是只把 VK 当作普通坐标表。

4. **任务相关物理模态证据**  
   同一诊断也显示：
   - HPE 对 Depth corruption 更敏感；
   - HAR 对 mmWave corruption 更敏感。

   这说明 VK 不是唯一中心，Depth/mmWave 等物理模态提供关键互补证据。

### 1.4 当前不能声称什么

不能声称：

- “严格证明所有模态都语义对齐到人体空间”；
- “WiFi/LiDAR/mmWave token 已经被 VK 完全锚定”；
- “VK 是唯一 semantic center”；
- “body-latent alignment 是一个可验证的显式几何对齐过程”。

这些说法都超出了当前代码和结果能证明的范围。

### 1.5 推荐论文写法

更安全的中文表述：

> 本文不在原始信号空间强行对齐 WiFi-CSI、LiDAR、mmWave、Depth 和 VK，而是将各模态映射到统一 token 维度，并在 HPE/HAR 任务监督下诱导其形成与人体状态相关的 body-latent 表示。该空间不是显式几何坐标系，而是任务监督形成的潜在表示空间。

更安全的英文表述：

> We do not claim explicit raw-space or geometry-level alignment among heterogeneous sensors. Instead, VK-RMD induces a task-supervised body-latent token space, where modality-specific features are projected into a shared dimension and optimized toward common HPE/HAR targets.

### 1.6 如果要进一步补强，最小补强是什么

不建议再做大规模训练。最小补强是做分析型图表：

- `subset severity`：不同缺失程度下模型性能变化；
- `VK semantic perturbation`：VK joint shuffle / sample mismatch 对 HPE/HAR 的影响；
- `reliability proxy heatmap`：不同模态组合下权重变化。

这些不能证明严格语义对齐，但可以支撑：

> 模型学到的是任务相关的人体状态表示，而不是简单拼接后的普通融合特征。

---

## 2. 质疑二：Full-to-Subset Guidance 是否只是 modality dropout + KD 换名？

### 2.1 审稿人可能怎么问

审稿人可能会问：

> 你所谓 Full-to-Subset Guidance，本质上是不是就是在 modality dropout 训练时加了 teacher loss？

或者更尖锐地问：

> 如果 teacher 和 student 架构相同，甚至参数量相近，为什么这还叫蒸馏？为什么需要一个新名字？

### 2.2 这个质疑为什么成立

这个质疑也成立。当前实现确实包含两个已有思想：

- random missing-modality sampling；
- teacher-student guidance / KD loss。

如果论文只说“我们提出 FSG”，但没有解释它和传统 KD 的区别，就容易被认为是旧模块换名字。

尤其需要注意：如果 HPE 的 no-KD 变体表现很强，甚至在某些统计上优于 full model，那么不能声称：

> FSG 是 HPE 提升的决定性原因。

否则审稿人会抓住这个矛盾。

### 2.3 FSG 真正可以成立的点

FSG 的合理性不在于“teacher 更大”，而在于 **输入条件不对称**。

传统 KD 是：

$$
F_T^{large}(x)\rightarrow F_S^{small}(x).
$$

FSG 是：

$$
F_T(\mathcal{M})\rightarrow F_S(S),\qquad S\subseteq\mathcal{M}.
$$

也就是说：

- teacher 的优势是看到完整模态；
- student 的困难是只能看到随机子集；
- student 学习的是 full-modality sensing behavior 在 missing-modality 条件下的近似。

因此 FSG 应该被写成：

> full-modality behavior regularization for subset-conditioned learning

而不是：

> 大模型压缩小模型的传统蒸馏。

### 2.4 当前已有代码能支撑什么

代码中 FSG 的证据是明确的：

1. Teacher 使用全模态：

```text
teacher_output = teacher(batch["inputs"], teacher_modalities)
```

2. Student 使用随机模态子集：

```text
selected = sample_missing_modalities(...)
student_output = student(batch["inputs"], selected)
```

3. HPE guidance 包含：

- output pose guidance；
- token guidance；
- bone structure guidance；
- reliability/fusion weight guidance；
- uncertainty regularization。

4. HAR guidance 包含：

- logit KD；
- token guidance；
- reliability/fusion weight guidance；
- uncertainty regularization。

这说明 FSG 不只是一个单一 KL loss，而是 full-to-subset 的多层行为约束。

### 2.5 当前不能声称什么

不能声称：

- “FSG 是所有性能提升的唯一来源”；
- “FSG 在 HPE/HAR 中都稳定优于 no-KD”；
- “FSG 是全新蒸馏理论”；
- “没有 FSG 模型就不能工作”。

尤其如果 HPE no-KD 强，论文必须诚实降调：

> FSG 对 subset-conditioned learning 起到辅助正则和行为约束作用，但其收益具有任务相关性。

### 2.6 推荐论文写法

中文安全写法：

> Full-to-Subset Guidance 并不是传统的大模型到小模型压缩蒸馏。本文中的 teacher 和 student 可以具有相同结构，二者的关键差异在于模态可用性：teacher 观察完整模态集合，student 在随机缺失模态子集上训练。FSG 的作用是将 full-modality teacher 的输出、token 表示、结构关系和融合行为作为辅助约束，引导 student 在不同模态可用条件下学习更稳定的感知行为。

英文安全写法：

> Full-to-Subset Guidance is not conventional large-to-small knowledge distillation. The asymmetry lies in modality availability rather than model capacity: the teacher observes the full modality set, whereas the student is optimized under randomly sampled modality subsets. FSG therefore acts as full-modality behavior regularization for subset-conditioned learning.

### 2.7 如果要进一步补强，最小补强是什么

最小补强不是重训大模型，而是整理已有消融：

- full VK-RMD；
- w/o KD；
- uniform fusion；
- Student-VK；
- Student-NV。

如果 HPE no-KD 强，论文应该这样写：

> 在 HPE 中，随机缺失训练和结构融合是主要收益来源，FSG 更多承担辅助稳定作用；在 HAR 中，logit/token guidance 对分类边界可能更有帮助。

这样不会被 no-KD 结果反杀。

---

## 3. 质疑三：低外观结构锚点是否足以支撑 privacy-friendly 叙事？

### 3.1 审稿人可能怎么问

审稿人可能会问：

> VK 和 Depth 仍然包含身体形态、姿态、步态、动作习惯，这些也可能泄露身份或行为隐私。为什么还能称为 privacy-friendly？

更严格的审稿人会问：

> 你有没有 identity leakage、attribute leakage、re-identification attack 或 privacy metric？

### 3.2 这个质疑为什么成立

这个质疑非常重要。只去掉 RGB 并不等于隐私安全。

VK 仍可能泄露：

- 身高比例；
- 体型；
- 步态；
- 动作习惯；
- 行为类别；
- 姿态风格。

Depth 也可能泄露：

- 身体轮廓；
- 空间位置；
- 动作轨迹；
- 体态特征。

因此，如果标题或摘要写成 privacy-preserving，就会被认为是概念偷换。

### 3.3 当前工作真正能支撑什么

当前工作能支撑的是：

1. **不使用 raw RGB 推理**  
   主模型以 VK/Depth/LiDAR/mmWave/WiFi-CSI 为输入，不使用原始 RGB 图像。

2. **降低外观暴露**  
   VK 去除了 RGB 中的人脸、纹理、衣着颜色和背景细节；Depth 相比 RGB 也减少了纹理和颜色暴露。

3. **仍保留人体结构和行为信息**  
   这不是缺点，而是任务所必需的信息。但它意味着不能声称匿名或隐私保证。

所以正确表述是：

> reduced visual exposure

而不是：

> privacy guarantee

### 3.4 当前不能声称什么

不能声称：

- privacy-preserving；
- anonymous；
- identity-free；
- formal privacy protection；
- secure sensing；
- VK/Depth 不泄露隐私。

这些都需要额外攻击实验或隐私理论支持。

### 3.5 推荐论文写法

中文安全写法：

> 本文中的 privacy-friendly 指的是 reduced visual exposure，而不是形式化隐私保护。模型在推理阶段不使用原始 RGB 图像，从而减少面部、纹理、衣着颜色和背景外观暴露。然而，VK 和 Depth 仍然包含身体几何、姿态和动作线索，可能泄露身份相关或行为相关信息。因此，本文不主张匿名性或差分隐私保证。

英文安全写法：

> In this work, privacy-friendly refers to reduced visual exposure rather than formal privacy preservation. By removing raw RGB from inference, the model suppresses facial appearance, texture, clothing color, and background cues. However, VK and depth still contain body geometry, pose, and motion patterns that may reveal identity- or behavior-related information. We therefore do not claim anonymity, differential privacy, or cryptographic protection.

### 3.6 推荐加入的表格

论文中建议加入一张小表：

| Representation | Face/texture exposure | Background exposure | Body geometry | Gait/action leakage | Formal privacy guarantee | Safe claim |
|---|---|---|---|---|---|---|
| RGB | High | High | High | High | No | Full visual exposure |
| VK | Low | Low | Medium | Medium/High | No | Reduced visual exposure |
| Depth | Low/Medium | Medium | High | Medium | No | Low-appearance geometry |
| LiDAR/mmWave/WiFi | Low | Low/Medium | Task-dependent | Task-dependent | No | Non-RGB sensing |

这张表的作用不是证明隐私安全，而是让审稿人看到作者没有偷换概念。

---

## 4. 三个质疑合并后的论文主线修正

最稳的主线不是：

> 我们提出了全新的 VK 对齐理论、全新的蒸馏理论和隐私保护方法。

而应该是：

> 本文提出一种面向 reduced-visual-exposure multimodal human perception 的 missing-modality learning framework。该框架使用 VK/Depth 作为低外观结构锚点，通过 task-supervised body-latent tokenization 组织异构传感器表示，并通过 subset-conditioned training 与 Full-to-Subset Guidance，使模型在任意模态子集下完成 HPE 和 HAR。

这条主线的优势是：

- 承认单个模块不是完全原创；
- 强调问题定义和训练目标；
- 把 VK/Depth、缺失模态、融合、FSG 串成一个完整系统；
- 不夸大 privacy；
- 不把诊断证据写成严格理论证明。

---

## 5. 建议改写后的三条贡献

### Contribution 1：问题定义与低外观结构锚点

中文：

> 我们将多模态 HPE/HAR 重新表述为 reduced-visual-exposure 条件下的任意模态缺失学习问题，并使用 VK/Depth 作为低外观结构锚点，在不依赖 raw RGB 推理的情况下保留人体结构和几何信息。

英文：

> We formulate multimodal HPE and HAR as missing-modality learning under reduced visual exposure, using VK and depth as low-appearance structural anchors to preserve body structure without raw RGB inference.

### Contribution 2：Task-supervised body-latent tokenization

中文：

> 我们不在原始空间对齐异构传感器，而是通过模态专属编码器、token projector 和任务监督，将 VK、Depth、LiDAR、mmWave 和 WiFi-CSI 组织到共享 body-latent token 表示中。

英文：

> Instead of enforcing raw-space alignment, we organize heterogeneous sensors into a task-supervised body-latent token space through modality-specific encoders, token projectors, and HPE/HAR supervision.

### Contribution 3：Subset-conditioned learning 与 FSG

中文：

> 我们提出 subset-conditioned missing-modality learning，并通过 Full-to-Subset Guidance 使用 full-modality teacher 的输出、token、结构和融合行为对随机缺失模态 student 进行辅助约束，从而提升任意模态组合下的鲁棒性。

英文：

> We introduce subset-conditioned missing-modality learning with Full-to-Subset Guidance, where a full-modality teacher regularizes randomly missing-modality students through output-, token-, structure-, and fusion-level behavior.

---

## 6. 最终推荐写法

如果只用一段话概括全文，建议写成：

> VK-RMD is not intended as a formal privacy-preserving method or a new Transformer fusion mechanism. Instead, it addresses reduced-visual-exposure multimodal human perception under arbitrary sensor availability. VK and depth provide low-appearance structural anchors, heterogeneous sensors are projected into a task-supervised body-latent token space, and students are trained under randomly sampled modality subsets with full-to-subset guidance from a full-modality teacher. This design makes the framework suitable for HPE and HAR when raw RGB is unavailable and sensor modalities may be missing, while its privacy and alignment claims are intentionally bounded to reduced visual exposure and diagnostic evidence.

中文对应：

> VK-RMD 不是形式化隐私保护方法，也不是单纯提出一个新的 Transformer 融合模块。它解决的是 reduced-visual-exposure 条件下任意传感器可用性的多模态人体感知问题。VK 和 Depth 提供低外观结构锚点，异构传感器被映射到任务监督的 body-latent token 空间，student 在随机模态子集下训练，并通过 full-modality teacher 的 Full-to-Subset Guidance 学习完整感知行为。该设计使模型能够在不使用 raw RGB 且传感器可能缺失的情况下完成 HPE 和 HAR，同时其隐私和对齐主张被严格限定为视觉暴露降低和诊断性证据。

---

## 7. 当前工作是否有疑问？

有，但不是致命问题。更准确地说：

1. **body-latent alignment 的理论证明不足**  
   目前只能说是 task-supervised latent representation，不能说严格语义对齐。

2. **FSG 的独立贡献需要谨慎**  
   它可以作为 full-to-subset regularization，但不能被写成所有性能提升的决定性来源。

3. **privacy-friendly 必须降调**  
   只能说 reduced visual exposure，不能说 privacy-preserving。

如果论文按上述边界写，当前工作仍然可以形成一个完整、诚实、可投稿的系统型创新故事。真正的核心不是“每个模块都是首创”，而是：

> 把低外观结构锚点、异构传感器 token 化、任意模态缺失训练和 full-to-subset guidance 组织成一个面向 HPE/HAR 的完整 reduced-visual-exposure missing-modality perception framework。
