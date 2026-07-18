# 科学问题与理论形式化表述

本文档用于把 VK-RMD 的科学问题、理论动机、训练机制和代码证据统一起来。它不是宣传稿，而是面向论文写作和审稿质疑的“底层逻辑说明”。

本文最稳妥的核心定位是：

> VK-RMD 研究的是在不依赖原始 RGB 推理的条件下，如何利用低外观暴露的人体结构锚点，让多模态人体感知模型在任意传感器模态缺失时仍能完成 HPE 姿态估计和 HAR 动作识别。

更具体地说，本文使用 VK 和 Depth 作为低外观结构锚点，将 LiDAR、mmWave、WiFi-CSI 等异构传感器映射到共享 body-latent token 空间，并通过 full-modality teacher 对 random missing-modality student 进行 Full-to-Subset Guidance。

需要特别注意：本文不应被写成严格隐私保护方法、raw RGB teacher 方法，或者传统“大模型蒸馏小模型”的模型压缩方法。

---

## 1. 核心科学问题

### 1.1 一句话科学问题

如何在不使用原始 RGB 推理的情况下，利用低外观暴露的人体结构表示，把异构传感器数据对齐到统一的人体潜在空间，并使模型在任意模态子集可用时仍能完成 HPE 和 HAR？

### 1.2 为什么这是一个科学问题

多模态人体感知的难点不只是“把多个传感器拼起来”，而是存在三个本质矛盾。

第一，原始数据空间完全异构。VK 是稀疏关节坐标，Depth 是稠密几何图，LiDAR 和 mmWave 是点云或反射点，WiFi-CSI 是无线信道矩阵。它们在数学形式上没有天然共同空间，因此不能在 raw space 中强行对齐。

第二，不同模态对不同任务的作用不同。VK 和 Depth 更接近人体结构和几何信息；LiDAR、mmWave 提供物理空间和运动反射证据；WiFi-CSI 是间接无线传播信息。HPE 更依赖细粒度几何，HAR 更可能依赖动作模式和动态线索。因此不能简单地说某一个模态永远是中心。

第三，真实部署中不能假设所有传感器永远可用。某些模态可能缺失、损坏、遮挡、噪声过强或成本受限。固定全模态输入模型无法自然适配这种情况。因此，真正的问题不是“全模态融合能不能更准”，而是“模型能否覆盖整个模态可用性空间”。

### 1.3 问题定义

HPE 中的模态集合为：

$$
\mathcal{M}_{hpe}=\{v,d,l,r,w\},
$$

其中：

- \(v\)：VK，视觉关键点；
- \(d\)：Depth；
- \(l\)：LiDAR；
- \(r\)：mmWave radar；
- \(w\)：WiFi-CSI。

HAR 中的模态集合为：

$$
\mathcal{M}_{har}=\{v,d,l,r\}.
$$

对任务 \(t\in\{\mathrm{hpe},\mathrm{har}\}\)，一个样本可以写成：

$$
(\{x_m\}_{m\in\mathcal{M}_t}, y^t),
$$

其中 \(y^{hpe}\) 是 3D pose 标签，\(y^{har}\) 是 action label。

推理时模型可能只接收到任意非空模态子集：

$$
S\subseteq\mathcal{M}_t,\qquad S\neq\emptyset.
$$

目标是学习：

$$
\hat{y}^t_S=F^t_\theta(\{x_m\}_{m\in S}, S),
$$

使模型不仅在全模态 \(S=\mathcal{M}_t\) 下有效，也能在不同缺失模态组合下保持可用。

当前代码中，HPE 和 HAR 是两个独立任务工程，但共享同一套范式：VK/非 RGB 模态编码、body-latent token 对齐、reliability-guided fusion、full-to-subset student 训练。当前主线不是一个真正的 HPE+HAR 联合多任务网络。

### 1.4 本文不主张什么

为了降低审稿风险，论文中必须明确以下边界：

- 本文不提供差分隐私、加密、匿名性或身份泄露防护等形式化隐私保证。
- 本文不证明 VK 是唯一语义中心。现有结果更稳妥的结论是：VK 提供显式关节语义和骨架结构，Depth、mmWave 等模态提供任务相关物理证据。
- 本文不在原始空间对齐 WiFi-CSI、LiDAR、mmWave、Depth 和 VK，而是在 body-latent token 空间中进行任务驱动对齐。
- 当前主线代码没有实现 raw RGB teacher 到 VK student 的蒸馏。
- 本文不是传统大模型压缩小模型的知识蒸馏。teacher 和 student 可以是同一结构，核心差异是 teacher 看全模态，student 看随机缺失模态。
- reliability 权重不是物理传感器真实可靠性，而是 learned contribution proxy，即学习到的任务贡献代理权重。

---

## 2. 方法论原则

### 2.1 VK 和 Depth 作为低外观结构锚点

本文更稳的说法不是“VK 等于隐私保护”，而是：

> VK 和 Depth 是低外观暴露的人体结构锚点。

VK 可以表示为人体骨架图：

$$
G_v=(V,E,X),
$$

其中 \(V\) 是 17 个关节节点，\(E\) 是骨架连接关系，\(X\in\mathbb{R}^{17\times 2}\) 是关键点坐标。

VK 编码过程不是 RGB 图像编码，而是：

$$
J_v=\phi_c(\mathrm{Norm}(X))+\phi_p(2\mathrm{Norm}(X)-1)+e_{joint},
$$

其中 \(\phi_c\) 编码关节坐标，\(\phi_p\) 编码中心化位置，\(e_{joint}\) 是关节身份嵌入。

随后通过骨架图混合：

$$
J_v^{(k+1)}=\mathrm{BoneGraphMixer}(J_v^{(k)}, A_{bone}),
$$

最后投影成固定数量的 token：

$$
Z_v=P_v(J_v)\in\mathbb{R}^{K\times D}.
$$

当前代码默认 \(K=32\)，\(D=512\)。

Depth 的角色是补充 VK：Depth 不是关节语义，但它保留空间几何，同时比 RGB 更少暴露纹理、衣着、面部和背景细节。因此论文可以把 VK 和 Depth 写成低外观结构锚点族：VK 提供显式骨架语义，Depth 提供稠密几何证据。

### 2.2 异构模态的 Body-Latent Token 化

对任意模态 \(m\)，先使用模态专属编码器提取特征：

$$
H_m=E_m(x_m).
$$

然后通过 projector 映射到统一 token 维度：

$$
Z_m=P_m(H_m)\in\mathbb{R}^{K\times D}.
$$

对于 VK，\(E_v\) 是 skeleton encoder，\(P_v\) 是 VK token projector。对于非 VK 模态，\(E_m\) 是对应传感器 backbone，\(P_m\) 是可训练 token projector。

每个模态 token 再加上 modality embedding：

$$
\bar{Z}_m=Z_m+e_m.
$$

对当前可用模态子集 \(S\)，只拼接当前存在的模态：

$$
U_S=\mathrm{Concat}_{m\in S}(\bar{Z}_m).
$$

然后进入 Transformer encoder：

$$
T_S=\Phi(U_S).
$$

这就是本文中“对齐”的真实含义：不是把 WiFi-CSI、LiDAR、mmWave 和 VK 变成同一种原始数据，而是将它们投影到同一 token 维度，在共享任务监督下学习人体状态相关的 latent 表示。

### 2.3 Reliability-Guided Fusion 作为学习到的贡献估计

Transformer 输出 \(T_S\) 会按照模态切分为：

$$
T_S=\{T_m\}_{m\in S}.
$$

HPE 中，每个模态 chunk 产生一个 pose expert 和一个 log-variance：

$$
(\hat{p}_m,s_m)=H_{pose}(T_m).
$$

HAR 中，每个模态 chunk 产生一个分类 logit expert 和一个 log-variance：

$$
(\hat{c}_m,s_m)=H_{cls}(T_m).
$$

默认 uncertainty fusion 下，权重计算为：

$$
\alpha_m=
\frac{\exp(-s_m/\tau)}
{\sum_{j\in S}\exp(-s_j/\tau)}.
$$

这里的 \(\alpha_m\) 只能写成 learned reliability proxy 或 learned contribution weight，不能写成真实传感器可靠性。

融合 token 为：

$$
z_S=\sum_{m\in S}\alpha_m\,\mathrm{Pool}(T_m).
$$

HPE 最终预测由专家加权输出和 residual head 共同组成：

$$
\hat{p}_{exp}=\sum_{m\in S}\alpha_m\hat{p}_m,
$$

$$
\hat{p}_{res}=H_{res}(z_S),
$$

$$
\hat{p}_S=0.5\hat{p}_{exp}+0.5\hat{p}_{res}.
$$

HAR 类似：

$$
\hat{c}_{exp}=\sum_{m\in S}\alpha_m\hat{c}_m,
$$

$$
\hat{c}_{res}=H_{res}(z_S),
$$

$$
\hat{c}_S=0.5\hat{c}_{exp}+0.5\hat{c}_{res}.
$$

---

## 3. Full-to-Subset Guidance

### 3.1 为什么这不是传统大模型到小模型蒸馏

传统知识蒸馏通常是：

$$
F_T^{large}(x)\rightarrow F_S^{small}(x),
$$

即大 teacher 压缩到小 student。

本文更准确的机制是：

$$
F_T(\mathcal{M})\rightarrow F_S(S),\qquad S\subseteq \mathcal{M}.
$$

teacher 和 student 可以是同构模型。差异不在模型大小，而在输入条件：

- Teacher：固定使用完整模态集合 \(\mathcal{M}\)；
- Student：每个 batch 随机采样可用模态子集 \(S\)；
- Student 学习在缺失模态下逼近 teacher 的完整感知行为。

因此更合适的名字是 Full-to-Subset Guidance，简称 FSG。它不是传统模型压缩蒸馏，而是完整模态感知行为对缺失模态感知行为的指导。

### 3.2 Full-Modality Teacher

Teacher 使用全模态：

$$
S_T=\mathcal{M}_t.
$$

HPE teacher 输出：

$$
(\hat{p}_T,z_T,\alpha_T)=F_T^{hpe}(\{x_m\}_{m\in\mathcal{M}_{hpe}}).
$$

HPE teacher 损失为：

$$
\mathcal{L}^{T}_{hpe}
=
\mathcal{L}_{pose}(\hat{p}_T,y)
+\lambda_{bone}\mathcal{L}_{bone}(\hat{p}_T,y).
$$

HAR teacher 输出：

$$
(\hat{c}_T,z_T,\alpha_T)=F_T^{har}(\{x_m\}_{m\in\mathcal{M}_{har}}).
$$

HAR teacher 损失为：

$$
\mathcal{L}^{T}_{har}
=
\mathrm{CE}(\hat{c}_T,y)
+\lambda_{unc}\mathcal{L}_{unc}.
$$

在 student 训练中，teacher 被设为 eval 模式，并在 `torch.no_grad()` 下产生指导信号。

### 3.3 Random Missing-Modality Student

Student 每个 batch 随机采样模态子集：

$$
S\sim q(S),\qquad S\subseteq\mathcal{M}_t,\qquad S\neq\emptyset.
$$

代码中通过 `drop_counts` 控制每次随机丢弃几个模态，并保证至少保留一个模态。

student 输出：

$$
(\hat{y}_S,z_S,\alpha_S)=F_S^t(\{x_m\}_{m\in S},S).
$$

这意味着缺失模态不是只在测试时模拟，而是直接进入训练目标。

### 3.4 Guidance Loss

HPE student 的实现损失为：

$$
\mathcal{L}^{S}_{hpe}
=
\mathcal{L}_{gt}
+\lambda_{out}\mathcal{L}_{out}
+\lambda_{tok}\mathcal{L}_{tok}
+\lambda_{bone}\mathcal{L}_{bone}^{KD}
+\lambda_{rel}\mathcal{L}_{rel}
+\lambda_{unc}\mathcal{L}_{unc}.
$$

其中：

$$
\mathcal{L}_{gt}=\mathcal{L}_{pose}(\hat{p}_S,y),
$$

$$
\mathcal{L}_{out}=\mathcal{L}_{pose}(\hat{p}_S,\mathrm{stopgrad}(\hat{p}_T)),
$$

$$
\mathcal{L}_{tok}=\|z_S-\mathrm{stopgrad}(z_T)\|_2^2,
$$

$$
\mathcal{L}_{bone}^{KD}
=
\mathcal{L}_{bone}(\hat{p}_S,\mathrm{stopgrad}(\hat{p}_T)),
$$

$$
\mathcal{L}_{rel}
=
\mathrm{KL}\left(
\alpha_S\;||\;\mathrm{Renorm}(\alpha_T|_S)
\right).
$$

这里 \(\alpha_T|_S\) 表示只取 teacher 中 student 当前可见模态对应的权重，并重新归一化。

HAR student 的实现损失为：

$$
\mathcal{L}^{S}_{har}
=
\mathrm{CE}(\hat{c}_S,y)
+\lambda_{kd}T^2
\mathrm{KL}\left(
\sigma(\hat{c}_T/T)\;||\;\sigma(\hat{c}_S/T)
\right)
+\lambda_{tok}\mathcal{L}_{tok}
+\lambda_{rel}\mathcal{L}_{rel}
+\lambda_{unc}\mathcal{L}_{unc}.
$$

当前代码实现了输出/logit 指导、token 指导、融合权重指导和不确定性正则。当前主线没有实现 raw RGB teacher，也没有实现单一模型同时输出 HPE 和 HAR。

### 3.5 总体优化目标

Teacher 训练目标：

$$
\min_{\theta_T}
\mathbb{E}_{(x,y)}
\left[
\mathcal{L}^{T}
\left(
F_{\theta_T}(\{x_m\}_{m\in\mathcal{M}_t}),y
\right)
\right].
$$

Teacher 训练完成后固定，student 训练目标为：

$$
\min_{\theta_S}
\mathbb{E}_{(x,y)}
\mathbb{E}_{S\sim q(S)}
\left[
\mathcal{L}^{S}
\left(
F_{\theta_S}(\{x_m\}_{m\in S},S),
y,
F_{\theta_T}(\{x_m\}_{m\in\mathcal{M}_t})
\right)
\right].
$$

这就是本文最核心的形式化目标：完整模态 teacher 指导任意子集 student。

---

## 4. 训练与推理流程

### 4.1 训练流程

当前实现流程为：

1. 离线提取或读取 VK keypoints。
2. 主模型不使用原始 RGB 图像。
3. 训练 full-modality teacher。
4. student 训练时冻结 teacher。
5. 每个 batch 随机采样模态子集 \(S\)。
6. teacher 用全模态前向，student 用子集模态前向。
7. student 同时优化任务监督和 FSG 指导损失。
8. 按脚本保存 `best.pth`、`last.pth`、历史 checkpoint 和评估结果。

### 4.2 推理流程

推理时模型只需要当前可用模态：

$$
\hat{y}_S=F_\theta(\{x_m\}_{m\in S},S).
$$

不需要 raw RGB。如果 VK 也不可用，可以使用 Student-NV 版本，只输入非视觉模态。

### 4.3 缺失模态评估

完整缺失模态评估空间为：

$$
\mathcal{P}(\mathcal{M}_t)\setminus \{\emptyset\}.
$$

HPE 有 31 种非空组合，HAR 有 15 种非空组合。论文应强调模型优化的不是单一 full-modality peak performance，而是整个 modality-availability space 的可用性。

cross-subject 和 cross-scene 应作为泛化诊断。尤其 cross-scene 性能下降必须诚实写成 limitation，而不能包装成已经解决跨环境部署问题。

---

## 5. 为什么这个方法应该有效

第一，VK 和 Depth 降低了原始视觉外观依赖。它们保留人体结构和几何信息，但减少了 RGB 中的人脸、纹理、衣着和背景暴露。

第二，body-latent tokenization 避免了不可能的 raw-space 对齐。WiFi-CSI 不需要变成点云，LiDAR 也不需要变成骨架坐标。所有模态只需要在任务监督下进入共享 token 空间。

第三，随机模态子集训练把“传感器缺失”变成训练条件，而不是测试时才出现的异常。

第四，Full-to-Subset Guidance 让 student 在部分模态下学习 full-modality teacher 的完整判断方式。这里 teacher 的优势不是参数更大，而是信息更完整。

第五，learned contribution weights 让模型可以根据当前模态组合调整融合方式。这些权重不能解释为真实物理可靠性，但可以作为任务层面的贡献代理。

---

## 6. 安全的贡献表述

推荐论文贡献写成：

1. 本文将 reduced-visual-exposure multimodal HPE/HAR 形式化为任意模态子集下的 missing-modality learning 问题。
2. 本文构建了 VK-centered body-latent tokenization 框架，将 VK、Depth、LiDAR、mmWave、WiFi-CSI 映射到共享人体潜在 token 空间。
3. 本文提出 Full-to-Subset Guidance，使 full-modality teacher 从输出、token、结构和融合行为层面指导 random missing-modality student。
4. 本文在 HPE 和 HAR 上进行 all-combination、Student-VK、Student-NV、cross-subject 和 cross-scene 评估。
5. 本文通过诊断实验说明 VK 提供关节结构语义，但 Depth 和 mmWave 等物理模态也提供任务相关证据。

更稳的标题方向：

> Learning with Low-Appearance Structural Anchors for Missing-Modality Multimodal Human Perception

或：

> VK-RMD: Visual-Keypoint Anchored Missing-Modality Learning for Reduced-Exposure Multimodal Human Perception

---

## 7. 高风险说法与建议降调

| 高风险说法 | 风险原因 | 建议写法 |
|---|---|---|
| VK 保证隐私 | VK 仍可能泄露身高、步态、动作习惯 | VK 降低视觉暴露，不提供形式化隐私保证 |
| VK 是唯一语义中心 | 结果显示 Depth/mmWave 对特定任务影响更大 | VK 是结构显式语义锚点，与物理传感器互补 |
| reliability 权重是真实传感器可靠性 | 代码学习的是任务贡献代理，不是物理传感器质量 | learned reliability proxy / learned contribution weight |
| 蒸馏是所有提升的来源 | HPE 中 no-KD 可能竞争力很强 | FSG 是辅助且任务相关的 full-to-subset supervision |
| raw RGB teacher 蒸馏给 VK student | 当前主线代码没有实现 | 当前主线是 VK/non-RGB full teacher 指导 missing-modality student |
| 一个模型同时解决 HPE 和 HAR | 当前 HPE/HAR 是两个工程实例 | 同一范式分别实例化到 HPE 和 HAR |
| 已解决跨场景部署 | cross-scene 退化明显 | 缺失模态鲁棒性不等于跨场景泛化鲁棒性 |
| 提出全新 attention/dropout/KD | 单个组件并非原创 | 创新在 VK/body-latent 问题定义、FSG 训练目标和系统化验证 |

---

## 8. 代码证据映射

| 论文说法 | 代码证据 | 支撑内容 | 强度 |
|---|---|---|---|
| VK 是骨架结构编码，不是 RGB 图像编码 | `HPE/models/skeleton_prompt_encoder.py`, `HAR/models/skeleton_prompt_encoder.py` | `SkeletonPromptEncoder`, `BoneGraphMixer`, joint embedding, bone graph | 强 |
| HPE 使用 VK、Depth、LiDAR、mmWave、WiFi-CSI | `HPE/utils/modality.py` | `ALL_MODALITIES` | 强 |
| HAR 使用 VK、Depth、LiDAR、mmWave | `HAR/utils/modality.py` | `ALL_MODALITIES` | 强 |
| 当前主线没有 active raw RGB | `LEGACY_TO_CANONICAL` 将 `rgb` 映射到 `vk`；模型使用 `vk_encoder` | RGB-free active model after VK extraction | 强 |
| 非 VK 传感器投影到统一 token 维度 | `TokenProjector` in `HPE/models/vk_rcd.py`, `HAR/models/vk_rcd_har.py` | body-latent tokenization | 强 |
| 模态身份被编码 | `ModalityTokenEncoder.modality_embedding` | modality embedding | 强 |
| token-space 跨模态交互 | `nn.TransformerEncoder` in `ModalityTokenEncoder` | learned token-space alignment | 强 |
| reliability fusion 产生 learned alphas | `ReliabilityFusion.forward`, `ClassificationReliabilityFusion.forward` | uncertainty/attention/uniform fusion | 强 |
| teacher 使用全模态 | `teacher_output = teacher(batch["inputs"], teacher_modalities)` | full-modality teacher query | 强 |
| student 使用随机缺失模态 | `sample_missing_modalities(...)` | random subset training | 强 |
| teacher 在 student 训练中冻结 | `teacher.eval()` and `torch.no_grad()` | no teacher update during student training | 强 |
| HPE FSG 包含 output/token/bone/reliability | `HPE/losses/distill_losses.py` | HPE guidance terms | 强 |
| HAR FSG 包含 logit/token/reliability | `HAR/losses/distill_losses.py` | HAR guidance terms | 强 |
| VK 结构语义是诊断性证据，不是唯一中心证明 | `supplement/vk_semantic_anchor_summary.csv`, `supplement/vk_semantic_anchor_analysis/vk_semantic_anchor_conclusion.md` | VK perturbation analysis | 中 |
| raw RGB teacher 到 VK student | 当前 HPE/HAR 主线未实现 | 只能作为 conceptual/future 或历史设想 | 未实现 |
| 单一 SuperTeacher 同时处理 HPE/HAR | `Super-Teacher` 分支 | 探索性实验，不是主线证据 | 探索性 |

---

## 可直接写入论文的核心段落

VK-RMD 以低外观结构锚点重新组织多模态人体感知，而不是依赖原始 RGB。VK 提供显式关节身份和骨架拓扑，Depth 提供纹理抑制的空间几何，LiDAR、mmWave 和 WiFi-CSI 提供互补的物理传感证据。不同模态并不在原始空间强行对齐，而是通过模态专属编码器和 token projector 映射到共享 body-latent token 空间，并由 Transformer 建模可用模态之间的关系。在训练阶段，full-modality teacher 学习完整传感器组合下的感知行为，random missing-modality student 则通过 Full-to-Subset Guidance 在输出、token、结构和融合权重层面学习 teacher 的完整判断方式。该框架统一处理了视觉暴露降低、异构传感器对齐和任意模态可用性问题，同时避免将 VK 误写为形式化隐私保证或将 reliability 权重误写为真实物理可靠性。
