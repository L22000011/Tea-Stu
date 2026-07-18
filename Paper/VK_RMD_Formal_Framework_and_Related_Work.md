# VK-RMD 形式化框架、模块机制与相关工作差异化说明

## 1. 总体定位

VK-RMD 应被表述为一种 **VK-centered privacy-friendly multimodal perception paradigm**。它的核心不是简单组合 VK、modality dropout、attention fusion 和 distillation，而是以 VK 为结构中间件重新组织多模态人体感知流程：

1. 用 VK 替代 raw RGB 参与主模型训练与推理，降低视觉外观暴露。
2. 将 VK、Depth、LiDAR、mmWave、WiFi-CSI 等异构模态映射到共享 body-latent token 空间。
3. 训练 full-modality VK teacher，再让随机缺失模态 student 学习 teacher 的输出、结构 token 和融合行为。

因此，VK-RMD 解决的是三个耦合问题：

- **Reduced visual exposure**：不依赖 raw RGB 主模型输入。
- **Heterogeneous sensor alignment**：不在原始信号空间对齐，而在 body-latent token 空间对齐。
- **Arbitrary modality availability**：student 在随机模态子集下训练，支持任意可用模态组合。

---

## 2. 模态集合与输入符号

HPE 模态集合定义为：

```math
\mathcal{M}_{hpe}=\{v,d,l,r,w\}
```

其中：

- \(v\)：VK，17 个视觉关键点；
- \(d\)：Depth；
- \(l\)：LiDAR；
- \(r\)：mmWave radar；
- \(w\)：WiFi-CSI。

HAR 模态集合定义为：

```math
\mathcal{M}_{har}=\{v,d,l,r\}
```

一个样本表示为：

```math
\{x_m\}_{m\in\mathcal{M}}, y
```

其中 \(y\) 在 HPE 中是 3D pose，在 HAR 中是 action label。Student 训练时随机采样可用模态子集：

```math
S \sim q(S),\quad S\subseteq\mathcal{M},\quad S\neq\emptyset
```

---

## 3. 模块内部机制与形式化描述

### 3.1 VK Structural Middleware

VK 输入为：

```math
x_v\in\mathbb{R}^{17\times 2}
```

VK 不是普通坐标表，而是带有关节身份和骨架拓扑的人体结构图：

```math
G_v=(V,E,X)
```

其中 \(V\) 是 17 个关节节点，\(E\) 是骨架边，\(X\) 是坐标。代码对应：

- `SkeletonPromptEncoder`
- `BoneGraphMixer`
- `COCO17_BONES`

处理流程：

```math
\tilde{x}_v=\mathrm{Normalize}(x_v)
```

```math
J_v=\phi_c(\tilde{x}_v)+\phi_p(2\tilde{x}_v-1)+e_{joint}
```

```math
J_v^{(k+1)}=\mathrm{BoneGraphMixer}(J_v^{(k)},A_{bone})
```

```math
Z_v=P_v(J_v)\in\mathbb{R}^{K\times D}
```

当前代码默认 \(K=32,D=512\)。

论文表达建议：

> VK is treated as a structural middleware rather than a raw visual modality. It converts RGB-derived visual information into an appearance-suppressed body graph and anchors the body-latent space through joint identities and skeleton topology.

---

### 3.2 非 VK 模态编码与 Token Projection

对任意非 VK 模态：

```math
m\in\mathcal{M}\setminus\{v\}
```

先用模态专属 encoder 提取特征：

```math
H_m=E_m(x_m)
```

再用 projector 映射到统一 token 空间：

```math
Z_m=P_m(H_m)\in\mathbb{R}^{K\times D}
```

其中：

- \(E_m\)：X-Fi-style pretrained modality backbone；
- \(P_m\)：trainable TokenProjector；
- \(Z_m\)：统一维度 body-latent token。

当前代码使用的 X-Fi backbone：

- Depth：`Depth_ResNet18`
- LiDAR：`lidar_PointTransformer`
- mmWave：`mmwave_PointTransformer`
- WiFi-CSI：CSI benchmark model

注意：当前正式主线没有启用 X-Fi 的 RGB ResNet branch；raw RGB branch 被 VK structural encoder 替代。

---

### 3.3 Body-Latent Token Encoding

对当前可用模态子集 \(S\)，每个模态 token 加上 modality embedding：

```math
\bar{Z}_m=Z_m+e_m
```

拼接后输入 Transformer encoder：

```math
U_S=\mathrm{Concat}_{m\in S}(\bar{Z}_m)
```

```math
T_S=\mathrm{TransformerEncoder}(U_S)
```

代码对应：

- `ModalityTokenEncoder`
- modality embedding
- Transformer encoder

这一步实现的是 **token-space semantic alignment**，不是 raw-space numerical alignment。WiFi-CSI、LiDAR、Depth、mmWave 不需要变成 VK 坐标，而是被优化为解释同一个 latent human state。

---

### 3.4 Per-Modality Expert Prediction

将编码后的 token 按模态切分：

```math
T_S=\{T_m\}_{m\in S}
```

HPE 中每个模态 chunk 产生 pose expert：

```math
\hat{p}_m,s_m=H_{pose}(T_m)
```

HAR 中每个模态 chunk 产生 classification expert：

```math
\hat{c}_m,s_m=H_{cls}(T_m)
```

其中 \(s_m\) 是 log-variance / uncertainty score。代码对应：

- HPE：`PoseExpertHead`
- HAR：`ClassificationExpertHead`

---

### 3.5 Reliability-Guided Subset Fusion

默认 fusion mode 是 uncertainty-based fusion：

```math
\alpha_m=
\frac{\exp(-s_m/\tau)}
{\sum_{j\in S}\exp(-s_j/\tau)}
```

其中 \(\alpha_m\) 是 **learned reliability proxy**，不是物理传感器真实可靠性。

HPE 融合：

```math
\hat{p}_{exp}=\sum_{m\in S}\alpha_m\hat{p}_m
```

```math
z_S=\sum_{m\in S}\alpha_m\mathrm{Pool}(T_m)
```

```math
\hat{p}_{res}=H_{res}(z_S)
```

```math
\hat{p}=0.5\hat{p}_{exp}+0.5\hat{p}_{res}
```

HAR 融合：

```math
\hat{c}_{exp}=\sum_{m\in S}\alpha_m\hat{c}_m
```

```math
z_S=\sum_{m\in S}\alpha_m\mathrm{Pool}(T_m)
```

```math
\hat{c}_{res}=H_{res}(z_S)
```

```math
\hat{c}=0.5\hat{c}_{exp}+0.5\hat{c}_{res}
```

代码对应：

- `ReliabilityFusion`
- `ClassificationReliabilityFusion`

---

## 4. Teacher-Student 机制与总体优化目标

### 4.1 Teacher

Teacher 使用 full modalities：

```math
S_T=\mathcal{M}
```

训练时不做随机缺失。HPE teacher 目标：

```math
\mathcal{L}^{T}_{hpe}
=
\mathcal{L}_{pose}(\hat{p}_T,y)
+
\lambda_{bone}\mathcal{L}_{bone}(\hat{p}_T,y)
```

HAR teacher 目标：

```math
\mathcal{L}^{T}_{har}
=
CE(\hat{c}_T,y)
+
\lambda_{unc}\mathcal{L}_{unc}
```

Teacher 学到的是 full-modality VK/non-RGB sensing behavior，而不是 raw RGB teacher behavior。

---

### 4.2 Student

Student 每个 batch 随机采样模态子集：

```math
S\sim q(S)
```

并学习 full-modality teacher 的监督信号。

HPE student 总损失：

```math
\mathcal{L}^{S}_{hpe}
=
\mathcal{L}_{gt}
+
\lambda_{out}\mathcal{L}_{out}
+
\lambda_{tok}\mathcal{L}_{tok}
+
\lambda_{bone}\mathcal{L}_{bone}^{KD}
+
\lambda_{rel}\mathcal{L}_{rel}
+
\lambda_{unc}\mathcal{L}_{unc}
```

其中：

```math
\mathcal{L}_{gt}=\mathrm{SmoothL1}(\hat{p}_S,y)
```

```math
\mathcal{L}_{out}=\mathrm{SmoothL1}(\hat{p}_S,\hat{p}_T)
```

```math
\mathcal{L}_{tok}=||z_S-z_T||_2^2
```

```math
\mathcal{L}_{bone}^{KD}=||B(\hat{p}_S)-B(\hat{p}_T)||_1
```

```math
\mathcal{L}_{rel}=KL(\alpha_S\;||\;\mathrm{Renorm}(\alpha_T|_S))
```

HAR student 总损失：

```math
\mathcal{L}^{S}_{har}
=
CE(\hat{c}_S,y)
+
\lambda_{kd}T^2 KL(
\sigma(\hat{c}_T/T)
\;||\;
\sigma(\hat{c}_S/T)
)
+
\lambda_{tok}\mathcal{L}_{tok}
+
\lambda_{rel}\mathcal{L}_{rel}
+
\lambda_{unc}\mathcal{L}_{unc}
```

总体优化目标：

```math
\min_{\theta_T}
\mathbb{E}_{(x,y)}
[
\mathcal{L}^{T}(F_T(\{x_m\}_{m\in\mathcal{M}}),y)
]
```

固定 teacher 后优化 student：

```math
\min_{\theta_S}
\mathbb{E}_{(x,y)}
\mathbb{E}_{S\sim q(S)}
[
\mathcal{L}^{S}
(
F_S(\{x_m\}_{m\in S}),
y,
F_T(\{x_m\}_{m\in\mathcal{M}})
)
]
```

---

## 5. 模块间运作机制

整体流程：

```math
x_m
\rightarrow
E_m
\rightarrow
P_m
\rightarrow
Z_m
\rightarrow
\text{Body-Latent Token Space}
\rightarrow
\text{Reliability Fusion}
\rightarrow
\hat{y}
```

Teacher-student 关系：

```math
F_T(\mathcal{M})
\rightarrow
\{
\hat{y}_T,
z_T,
\alpha_T,
\text{structure}_T
\}
\rightarrow
F_S(S)
```

其中：

```math
S\subseteq\mathcal{M},\quad S\neq\emptyset
```

核心机制：

1. VK 提供显式人体结构锚点。
2. 不同传感器通过各自 encoder + projector 进入共享 body-latent token space。
3. Transformer 建模模态 token 间关系。
4. Reliability fusion 对当前可用模态动态加权。
5. Full-modality teacher 提供完整感知行为，student 学习在缺失模态下逼近它。

---

## 6. 当前代码框架与理论叙事的对应关系

当前代码实际实现的是：

```math
\text{VK full-modality teacher}
\rightarrow
\text{missing-modality VK/non-RGB student}
```

不是：

```math
\text{raw RGB teacher}
\rightarrow
\text{VK student}
```

因此论文中应避免说：

> RGB teacher distills dense visual knowledge into VK student.

应该写：

> VK serves as a structural middleware throughout the teacher-student pipeline. The teacher and student are both RGB-free after offline VK extraction, and the full-modality VK teacher transfers body-structural and fusion-level knowledge to missing-modality students.

代码层面的事实：

- X-Fi 的 RGB backbone 存在于 legacy code，但当前主线没有启用。
- 当前主线复用了 X-Fi 的 Depth/LiDAR/mmWave/WiFi-CSI pretrained backbones。
- VK 分支是我们的 `SkeletonPromptEncoder`。
- Teacher 训练 full modalities，不随机缺失。
- Student 训练随机缺失模态。
- HPE 和 HAR 是两个独立工程，但共享同一 VK-centered 范式。

---

## 7. 核心参考文献与差异性

### 7.1 X-Fi: Modality-Invariant Multimodal Human Sensing

X-Fi 提出 modality-invariant multimodal human sensing，使用 Transformer 和 X-fusion 支持独立或组合模态输入，并在 MM-Fi/XRF55 上做 HPE/HAR。

差异：

- X-Fi 目标是 modality-invariant foundation model。
- VK-RMD 目标是 VK-centered reduced-visual-exposure missing-modality learning。
- X-Fi 保留 raw RGB 体系；VK-RMD 主线用 VK 替代 raw RGB。
- X-Fi 关注任意组合使用；VK-RMD 进一步加入 full-modality teacher 到 missing-modality student 的监督，并系统报告 Student-VK / Student-NV。

为什么重要：

- 我们不是简单复现 X-Fi，而是在 X-Fi backbone 基础上把视觉中心从 RGB-centered 改为 VK-centered。
- 贡献集中在 structural middleware、body-latent alignment 和 missing-modality robustness。

---

### 7.2 MM-Fi Dataset

MM-Fi 是多模态非侵入 4D human sensing 数据集，包含同步多模态数据、40 subjects、27 actions，并支持 HPE/HAR 等任务。

差异：

- MM-Fi 是数据集与基准。
- VK-RMD 是面向该类数据的 VK-centered missing-modality learning framework。

为什么重要：

- MM-Fi 的同步采集使 VK、Depth、LiDAR、mmWave、WiFi-CSI 可以通过同一 latent human state 建立训练对齐。
- VK-RMD 利用同步性、共同标签和 VK 结构锚点实现 body-latent alignment。

---

### 7.3 Missing Modality Learning

现有 missing-modality learning 通常使用 modality dropout、generation、hallucination、reconstruction 或 knowledge transfer 处理缺失输入。

差异：

- 现有方法多为通用缺失模态学习。
- VK-RMD 将 missing modality 具体化到多传感器人体感知，并把每个子集视为 deployable sensing configuration。
- VK-RMD 不是只做 dropout regularization，而是优化 modality-availability space。

为什么重要：

- 真实部署不是“全模态输入 + 偶尔扰动”，而是传感器可用性持续变化。
- 31/15 全组合评估直接覆盖这个可用性空间。

---

### 7.4 Missing-Modality Robust Action Recognition

相关工作研究缺失模态动作识别实践，尤其关注 action recognition 中的缺失模态训练与测试。

差异：

- 该方向主要聚焦 action recognition。
- VK-RMD 同时覆盖 HPE 和 HAR。
- VK-RMD 引入 VK structural middleware 和 body-latent token alignment，而不是只做动作识别层面的缺失模态处理。

为什么重要：

- HPE 更依赖细粒度身体几何。
- 同一范式支持 pose regression 和 action classification，说明该方法不是单一任务技巧。

---

### 7.5 Knowledge Distillation / FitNets / Privileged Information

相关工作包括知识蒸馏、中间层 hint 蒸馏和 learning using privileged information。

差异：

- 标准 KD 多关注模型压缩或输出分布。
- FitNets 关注中间特征 hint。
- VK-RMD 的 teacher-student 不是单纯压缩，而是 full-modality-to-subset supervision。
- 蒸馏项包含 output、token、body structure 和 fusion weights。

为什么重要：

- Student 的困难不是模型小，而是输入模态缺失。
- Teacher 传递的是完整模态感知行为，使 student 在部分模态下学习完整系统的判断方式。

---

### 7.6 Skeleton-Based Recognition / ST-GCN

ST-GCN 等 skeleton-based 方法将人体骨架建模为图并学习空间-时间模式。

差异：

- ST-GCN 主要把 skeleton 当作动作识别输入。
- VK-RMD 把 VK skeleton 当作跨模态 structural middleware。
- VK 不只是输入特征，而是 body-latent space 的结构锚点。

为什么重要：

- VK 从“单模态骨架识别输入”提升为“多模态对齐中间件”。
- LiDAR/mmWave/WiFi token 不是直接对齐坐标，而是被训练去解释 VK 显式参数化的人体状态。

---

## 8. 我们解决了什么

VK-RMD 可以明确声称解决：

1. **Raw RGB dependence**  
   用 VK 替代 raw RGB 贯穿 teacher-student 主流程，降低视觉外观暴露。

2. **Raw-space heterogeneity**  
   WiFi-CSI、LiDAR、mmWave、Depth 和 VK 不在原始空间对齐，而是通过 body-latent token space 对齐。

3. **Arbitrary modality availability**  
   Student 在随机模态子集下训练，支持任意传感器组合。

4. **Full-to-subset supervision gap**  
   Full-modality teacher 将完整模态下的输出、结构 token 和融合行为传递给 missing-modality student。

5. **Task generality across HPE/HAR**  
   同一 VK-centered 范式同时用于 pose regression 和 action classification，证明其不是单一任务技巧。

---

## 9. 可直接写进论文的方法段

> VK-RMD reorganizes multimodal human perception around visual keypoints as a structural middleware. Instead of aligning heterogeneous raw signals such as WiFi-CSI tensors, LiDAR point clouds, radar reflections and keypoint coordinates directly, VK-RMD maps each modality into a shared body-latent token space. VK anchors this space through explicit joint identities and skeleton topology, while sample-level synchronization and common HPE/HAR supervision induce semantic alignment across non-visual sensors. A full-modality VK teacher learns complete sensing behavior without raw RGB input, and missing-modality students are optimized over randomly sampled modality subsets to imitate the teacher through output-, token-, structure- and fusion-level supervision. Therefore, the framework addresses reduced visual exposure, heterogeneous sensor alignment and arbitrary modality availability in a unified objective.

中文解释：

> VK-RMD 以视觉关键点作为结构中间件重新组织多模态人体感知。它不在原始空间强行对齐 WiFi-CSI、LiDAR、mmWave、Depth 和 VK，而是将每种模态映射到共享 body-latent token 空间。VK 通过关节语义和骨架拓扑锚定该空间，同步样本和共同 HPE/HAR 监督诱导非视觉传感器的语义对齐。全模态 VK teacher 在不使用 raw RGB 的条件下学习完整感知行为，随机缺失模态 student 则通过输出、token、结构和融合层面监督学习 teacher 的完整判断方式。因此，该框架统一处理了视觉暴露降低、异构传感器对齐和任意模态可用性问题。

---

## 10. 写作边界

必须避免：

- “raw RGB teacher distills dense visual knowledge into VK student”
- “VK guarantees privacy”
- “reliability weight is physical sensor reliability”
- “distillation is the only source of improvement”
- “cross-scene generalization is solved”

推荐使用：

- “VK-centered structural middleware”
- “appearance-suppressed body graph”
- “body-latent token alignment”
- “learned reliability proxy”
- “full-modality-to-subset supervision”
- “reduced visual exposure”

