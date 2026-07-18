# VK-RMD 算法创新叙事：从模块堆叠到连贯机制

## 1. 一句话算法定位

VK-RMD 不应被写成“VK + modality dropout + attention + distillation”的模块拼装，而应被写成一个统一的缺失模态学习算法：

> **VK-RMD is a VK-anchored subset-invariant multimodal learning algorithm for reduced-visual-exposure human perception.**

中文主线：

> **VK-RMD 的算法创新不在于单独发明 dropout、attention 或 distillation，而在于提出一种以 VK 人体结构为锚点的缺失模态学习机制：先将异构传感器映射到同一身体结构 token 空间，再在随机模态子集下学习子集不变表示，并通过 learned reliability proxy 动态选择当前可用模态的决策贡献。**

因此，本文的算法贡献应写成：

> VK-RMD formulates multimodal HPE and HAR under missing modalities as a VK-anchored subset-invariant representation learning problem.

而不是：

> We use modality dropout, attention fusion, and distillation.

---

## 2. 算法名字与标题降风险

当前 `VK-RMD` 可以保留，但不建议继续把 `D` 强绑定为 `Distillation`，因为 HPE 中 no-distillation 结果很强，审稿人会质疑 distillation 是否真是核心。

更稳的解释：

- `VK-RMD`: **Visual-Keypoint Anchored Reliability-guided Missing-modality Dynamics**
- 或正文中写成：
  **VK-RMD: Visual-Keypoint Anchored Reliability-guided Missing-modality Learning**

推荐标题方向：

> **VK-RMD: Visual-Keypoint Anchored Reliability-Guided Missing-Modality Learning for Multimodal HPE and HAR**

如果必须保留 privacy 方向，建议写：

> **VK-RMD: Visual-Keypoint Anchored Missing-Modality Learning for Reduced-Visual-Exposure Multimodal Human Perception**

不建议标题继续主打：

> Reliability-Guided Missing-Modality Distillation

原因：distillation 是辅助机制，不是所有任务的稳定收益来源。

---

## 3. 统一算法命题

论文 Method 开头建议直接给出算法命题：

> The key idea of VK-RMD is to learn a body-structure-centered representation that remains usable under arbitrary modality subsets. VK provides the structural anchor, modality-specific encoders map heterogeneous sensor streams into a shared body-token space, and a reliability-aware fusion module selects the contribution of each available modality according to the current subset.

中文解释：

> VK-RMD 的核心思想是学习一种以人体结构为中心、对模态子集变化鲁棒的表示。VK 提供结构锚点；各模态编码器将异构传感器输入映射到共享身体 token 空间；可靠性融合模块根据当前可用模态组合动态分配决策权。

这个命题把所有模块串成一条算法链：

```text
RGB privileged knowledge
  -> VK structural anchor
  -> shared body-token space
  -> subset-invariant missing-modality training
  -> reliability-aware subset fusion
  -> HPE / HAR prediction under arbitrary modality availability
```

---

## 4. 三阶段算法链路

### Stage 1: VK-Anchored Structural Tokenization

#### 问题

RGB 是 dense image，VK 是人体关键点，Depth / LiDAR / mmWave / WiFi-CSI 又分别是深度图、点云、雷达点和信道矩阵。它们不能在 raw space 中直接对齐。

因此，论文不能写成“把 RGB 换成 VK 后继续用原来的 backbone”。正确写法是：

> RGB-to-VK changes the input from dense appearance to sparse body geometry. Therefore, VK should be encoded by a joint-level structural encoder rather than by an image backbone.

#### 算法处理

每个模态先通过自己的 encoder，再通过 projector 进入统一 body-token 空间：

```text
x_m -> encoder E_m -> projector P_m -> body token z_m
```

公式：

```text
z_m = P_m(E_m(x_m)) + e_m
```

其中：

- `x_m` 是模态 `m` 的输入；
- `E_m` 是模态专属编码器；
- `P_m` 是 token projector；
- `e_m` 是 modality embedding；
- `z_m` 是统一维度的 body token。

VK 的特殊路径：

```text
RGB frame
  -> RGB-derived VK/keypoints
  -> lightweight skeleton encoder
  -> VK body token z_vk
```

#### 与原 X-Fi/RGB 分支的区别

原 RGB 分支通常用 ResNet18 等图像 backbone 提取外观纹理和空间视觉特征。现在的 VK 分支输入不是图像，而是 17 个关节的结构坐标，因此应采用 joint-level encoder：

```text
17 keypoints
  -> joint-wise coordinate lifting
  -> skeleton/global joint interaction
  -> VK structural token
```

这说明 VK 不是简单替换 RGB 文件，而是改变了表征空间：

```text
image appearance space -> body structure space
```

推荐正文句子：

> We do not force heterogeneous sensors into a common raw representation. Instead, VK-RMD aligns them through a VK-anchored body-token space, where RGB-derived keypoints provide the structural coordinate system and non-RGB modalities are projected into the same semantic space.

---

### Stage 2: Subset-Invariant Missing-Modality Training

#### 问题

真实部署中不能假设所有传感器都稳定可用。Depth 可能被遮挡，LiDAR / mmWave 可能稀疏或丢点，WiFi-CSI 可能受环境扰动，某些模态也可能因为设备故障直接缺失。

因此，训练目标不能只优化 full-modality input。

#### 算法处理

每次训练随机采样一个可用模态子集：

```text
S ~ q(S), S subset M
```

模型只使用 `S` 中的模态预测：

```text
y_hat_S = F({z_m | m in S})
```

#### 这不是普通 modality dropout

普通 modality dropout 往往被当作正则化手段：训练时随机丢一些模态，让模型不要过拟合。

VK-RMD 中的 subset learning 要写成部署条件建模：

> In VK-RMD, modality dropping is not used as a generic regularizer. It defines the deployment condition during training: every sampled subset is treated as a valid sensing configuration that must produce a pose or action prediction.

中文：

> VK-RMD 中的随机模态缺失不是普通正则化，而是把每个模态子集都视为部署时可能出现的合法输入条件。模型训练的目标不是 full-modality 最优，而是任意可用模态组合都可用。

这就是算法上的连贯创新点之一：  
**从 full-modality optimization 转为 subset-invariant optimization。**

---

### Stage 3: Reliability-Aware Subset Fusion

#### 问题

不同模态的有效性随组合、场景和任务变化：

- Depth 对 HPE 强，但可能受遮挡影响；
- LiDAR 和 mmWave 稀疏，但能提供几何/运动线索；
- WiFi-CSI 噪声更高，但在非视觉场景有部署价值；
- VK 结构强，但仍来自视觉关键点，不等于完全非视觉。

固定拼接或平均融合假设每个模态同等可靠，这在缺失模态条件下不成立。

#### 算法处理

对当前可用模态计算 learned reliability proxy：

```text
alpha_m = softmax(g(z_m, e_m, S))
```

融合表示：

```text
z_S = sum_{m in S} alpha_m z_m
```

输出：

```text
HPE: y_hat_pose = Head_hpe(z_S)
HAR: y_hat_action = Head_har(z_S)
```

#### 表述边界

这里不能把 `alpha_m` 写成真实物理传感器可靠性。它不是传感器标定值，也不是概率意义上的可靠性保证。

应写成：

> The reliability score is a learned proxy rather than a physical sensor-quality measurement. It is optimized to improve task prediction under each available subset.

中文：

> 可靠性权重是任务驱动学习得到的可靠性代理，而不是物理传感器质量测量。它表示模型在当前模态组合下对各模态决策贡献的估计。

这能避免审稿人说“这不就是 attention”时抓住过度声称。

---

## 5. Loss Design Narrative

不要把 loss 设计写得过于复杂，也不要强行把 distillation 写成所有提升的来源。

稳妥统一形式：

```text
L_total = L_task + lambda_kd L_kd + lambda_struct L_struct
```

其中：

- `L_task`：任务监督；
  - HPE 使用 pose regression / MPJPE-related loss；
  - HAR 使用 cross-entropy；
- `L_kd`：student 向 full-modality teacher 学习；
- `L_struct`：如果代码中已有 bone / token / structural KD，则作为结构一致性约束；如果没有，就不要硬写。

更保守写法：

```text
L_total = L_task + lambda_kd L_kd
```

然后正文明确：

> Distillation is not the sole source of improvement. It is used as an auxiliary mechanism to stabilize subset learning, while the central mechanism is VK-anchored missing-modality representation learning.

中文：

> 蒸馏不是本文所有提升的唯一来源，而是用于稳定子集学习的辅助机制。VK-RMD 的核心是 VK 锚定的缺失模态表示学习。

这样可以解释 HPE 中 no-KD 变体很强的问题。

---

## 6. How This Is Not Simple Module Stacking

### Reviewer question

> Is this just modality dropout + attention + KD?

### Response

不是简单堆叠，原因有三层。

#### 1. 表示层不是普通输入拼接

VK 作为结构锚点，把 RGB privileged knowledge 压缩成身体结构 token。其他模态不是和 RGB 图像像素对齐，而是和 VK 身体结构空间对齐。

推荐句：

> The key representational step is not feature concatenation, but VK-anchored body-token alignment.

#### 2. 训练目标不是普通 dropout 正则化

随机模态子集不是为了防过拟合，而是把每个模态组合都当作部署时可能出现的合法输入条件。

推荐句：

> Subset sampling defines the inference-time availability space and forces the student to learn a valid predictor for each modality subset.

#### 3. 融合不是普通 attention 可视化

Reliability-aware fusion 在每个模态子集内产生动态权重，并通过 uniform fusion ablation 证明它对缺失模态鲁棒性有实际作用。

推荐句：

> Reliability-aware fusion is evaluated as a functional component rather than presented only as an attention visualization.

### 推荐贡献表述

> We formulate missing-modality multimodal human perception as VK-anchored subset-invariant learning, where heterogeneous sensors are aligned in a body-token space and optimized under all possible modality-availability conditions.

中文：

> 我们将缺失模态多模态人体感知表述为 VK 锚定的子集不变学习问题，使异构传感器在身体结构 token 空间中对齐，并在所有可能的模态可用条件下进行优化。

---

## 7. Recommended Method Section Structure

### 3.1 Problem Formulation

定义模态集合：

```text
M_HPE = {VK, D, L, R, W}
M_HAR = {VK, D, L, R}
S subset M
```

目标：

```text
Given any available subset S, predict HPE pose or HAR label without raw RGB inference.
```

强调：

- RGB 只作为训练期 privileged source；
- 推理期不使用 raw RGB；
- 模型必须支持任意可用模态组合。

---

### 3.2 VK-Anchored Body Tokenization

写 RGB 到 VK 的变化，以及为什么 ResNet18 不适合 VK。

必须包含：

- RGB 是训练期 privileged source；
- VK 是结构锚点；
- VK encoder 是 joint-level encoder；
- 其他模态经各自 backbone + projector 到统一 token 维度；
- modality embedding 保留模态身份信息；
- 对齐发生在 body-token space，而不是 raw sensor space。

---

### 3.3 Subset-Invariant Missing-Modality Training

写随机采样模态子集。

强调：

- 不是普通正则化；
- 是部署条件模拟；
- 训练和测试都覆盖任意模态组合；
- full modality 只是所有子集中的一个，不是唯一目标。

---

### 3.4 Reliability-Aware Fusion

写 learned reliability proxy。

强调：

- 当前子集内动态融合；
- 对不同模态组合生成不同权重；
- 通过 uniform fusion ablation 验证；
- 不声称是真实传感器可靠性。

---

### 3.5 Optional Teacher-Student Supervision

写 distillation，但降级为辅助。

强调：

- full-modality teacher 提供稳定监督；
- distillation 对 HPE / HAR 的收益可能不同；
- 不写 KD always improves；
- KD 是辅助，核心机制是 VK-anchored subset-invariant learning。

---

## 8. Suggested Algorithm Box

论文中建议增加一个 Algorithm 1：

```text
Algorithm 1: VK-RMD Training

Input:
  synchronized multimodal sample {x_m},
  task label y,
  modality set M,
  subset sampling distribution q(S),
  optional full-modality teacher T

1. Extract VK structural input from RGB-derived keypoints.
2. For each modality m in M:
     z_m = P_m(E_m(x_m)) + e_m
3. Sample available subset:
     S ~ q(S)
4. Keep only tokens from the available subset:
     Z_S = {z_m | m in S}
5. Estimate learned reliability scores:
     alpha_m = softmax(g(z_m, e_m, S))
6. Fuse subset representation:
     z_S = sum_{m in S} alpha_m z_m
7. Predict task output:
     HPE: y_hat = Head_hpe(z_S)
     HAR: y_hat = Head_har(z_S)
8. Optimize:
     L_total = L_task(y_hat, y) + lambda_kd L_kd(optional)
9. Repeat over random subsets and tasks.

Output:
  student model supporting arbitrary modality subsets.
```

这个算法框能让审稿人看到一个完整流程，而不是散模块列表。

---

## 9. Experiments That Support This Algorithm Story

主表和消融要对应算法三部分。

### 9.1 VK structural anchor

支持证据：

- Student-VK vs Student-NV；
- VK encoder ablation；
- official X-Fi/VK baseline 对比；
- HPE/HAR 同时验证。

对应结论：

> VK provides a useful structural anchor, while Student-NV shows that the framework can also operate without VK at inference.

---

### 9.2 Subset-invariant learning

支持证据：

- HPE 31 组合；
- HAR 15 组合；
- grouped missing-modality robustness；
- random / cross-subject / cross-scene。

对应结论：

> The method is not optimized only for full modality; it is explicitly evaluated across the modality availability space.

---

### 9.3 Reliability-aware fusion

支持证据：

- uniform fusion ablation；
- reliability weight visualization；
- HPE/HAR missing combination performance。

对应结论：

> Learned reliability-aware fusion provides a functional advantage over fixed uniform fusion under missing modalities.

---

### 9.4 Distillation as auxiliary

支持证据：

- w/o KD；
- KD variants；
- HPE 与 HAR 的差异分析。

对应结论：

> Distillation is task-dependent. It can help stabilize learning, especially in HAR, but should not be claimed as the universal source of performance gain.

---

## 10. What To Avoid

不要写：

- “我们提出全新的 attention 机制”；
- “我们首次解决任意模态缺失”；
- “VK 保证隐私安全”；
- “KD 是所有提升的来源”；
- “跨场景已经解决”；
- “reliability weight 等于真实传感器可靠性”。

要写：

- “VK-anchored body-token alignment”；
- “subset-invariant missing-modality learning”；
- “learned reliability proxy”；
- “RGB-free inference / reduced visual exposure”；
- “distillation is auxiliary and task-dependent”；
- “cross-scene remains challenging and is reported as a limitation”。

---

## 11. 正文可直接使用的 Contribution 版本

建议把 contribution 改成三条：

### Contribution 1

> We formulate reduced-visual-exposure multimodal human perception as VK-anchored body-token learning, where RGB-derived keypoints serve as structural privileged information and raw RGB is removed from inference.

### Contribution 2

> We propose a subset-invariant missing-modality learning strategy that trains the student under randomly sampled modality subsets and enables HPE/HAR prediction under arbitrary sensor availability.

### Contribution 3

> We introduce a reliability-aware fusion mechanism that estimates learned modality-contribution weights for each available subset, and validate its effect through all-combination evaluation and uniform-fusion ablation.

中文版本：

1. 将 reduced-visual-exposure 多模态人体感知表述为 VK 锚定的身体 token 学习问题，使 RGB-derived keypoints 作为训练期结构特权知识，而原始 RGB 不进入推理阶段。
2. 提出子集不变的缺失模态学习策略，在随机模态子集下训练 student，使其支持任意传感器可用组合下的 HPE/HAR 推理。
3. 引入 reliability-aware fusion，为当前可用模态子集估计动态模态贡献权重，并通过全组合评估和 uniform-fusion 消融验证其作用。

---

## 12. Final One-Sentence Story

英文：

> VK-RMD converts RGB-derived visual knowledge into a VK-anchored body-token space, trains the student under randomly sampled modality subsets, and learns reliability-aware fusion for each available subset, enabling reduced-visual-exposure HPE and HAR under arbitrary sensor availability.

中文：

> VK-RMD 将 RGB 视觉知识压缩到以 VK 为锚点的人体结构 token 空间，在随机模态子集下训练 student，并为每种可用模态组合学习可靠性融合权重，从而在不使用原始 RGB 推理的条件下实现 HPE 和 HAR 的缺失模态鲁棒感知。

---

## 13. 给审稿人的底层回答

如果审稿人问：

> 你的工作是不是只是现有模块组合？

回答逻辑：

> 单个构件确实不是本文的独立首创。本文的贡献在于将这些机制组织成一个面向缺失模态人体感知的结构化算法：以 VK 作为身体结构锚点，将异构传感器投影到共享 body-token space，在所有模态子集条件下训练，并用 learned reliability proxy 完成子集级动态融合。实验不是只报告 full modality，而是系统覆盖 HPE 31 种组合和 HAR 15 种组合，并通过 uniform fusion、no-KD、encoder ablation 等实验定位各机制作用。

这比强行声称“我们提出全新 attention / KD”更稳，也更符合现有实验结果。
