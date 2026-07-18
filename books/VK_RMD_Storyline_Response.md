# VK-RMD 主线回应：VK 桥梁、RGB-to-VK 对齐与 P0 风险处理

## 1. 文档目的

本文档用于统一回答 VK-RMD 论文当前最容易被审稿人质疑的几个核心问题：

1. VK 为什么不是简单替换 RGB，而是 RGB 与非 RGB 感知之间的结构桥梁。
2. RGB 转换成 VK 后，如何与 Depth、LiDAR、mmWave、WiFi-CSI 等异构模态对齐。
3. 本文的方法论创新应该如何表述，避免被认为只是 modality dropout、attention/gating 和 KD 的堆叠。
4. 2025/2026 相关 baseline 应该如何处理，哪些需要做，哪些不建议强行复现。
5. 两个 P0 风险如何处理：
   - P0-1：创新性不足，方法像已有模块组合。
   - P0-2：privacy claim 不够成立，VK 并不等同于正式隐私保护。

本文档不是实验记录，而是论文主线和审稿风险处理手册。它应与另一个文件 `VK_RMD_Algorithmic_Story.md` 配合使用：

- `VK_RMD_Storyline_Response.md`：回答故事线、定位、对齐、baseline、风险。
- `VK_RMD_Algorithmic_Story.md`：回答算法机制如何连贯成一个方法。

---

## 2. 论文真实定位

论文不应写成：

> We propose a completely new attention / dropout / distillation algorithm.

更稳的定位是：

> **VK-RMD is a visual-keypoint bridged missing-modality learning framework for reduced-visual-exposure multimodal human perception.**

中文表述：

> 本文提出的不是一个孤立的新模块，而是一个面向真实部署约束的完整学习框架：训练期允许 RGB/VK 提供强结构知识，推理期不使用原始 RGB；模型通过 VK 结构桥梁、随机模态缺失训练和 reliability-aware fusion，在 HPE 与 HAR 中支持任意可用模态组合。

因此，论文核心不应是“我发明了某个模块”，而应是：

```text
RGB privileged knowledge
  -> VK structural bridge
  -> heterogeneous modality token alignment
  -> subset-invariant missing-modality learning
  -> reliability-aware fusion
  -> HPE/HAR under arbitrary sensor availability
```

---

## 3. VK 为什么是桥梁，而不是普通替换 RGB

### 3.1 RGB 的问题

RGB 图像信息量强，但包含大量隐私敏感内容：

- 脸部外观；
- 身份特征；
- 衣着纹理；
- 背景环境；
- 物体上下文；
- 场景布局。

因此，RGB 在长期部署的人体感知系统中容易带来视觉暴露风险。

### 3.2 VK 的角色

VK 是从 RGB 中抽取的人体结构中间表示。它保留：

- 关节拓扑；
- 姿态几何；
- 身体运动结构；
- 动作相关的骨架变化。

同时弱化或去除：

- 原始纹理；
- 背景；
- 人脸；
- 衣着颜色；
- 图像级外观细节。

因此，VK 的角色不是“第五个普通模态”，而是：

> **RGB privileged knowledge 与 non-RGB sensing 之间的结构翻译层。**

或者：

> **appearance-suppressed structural bottleneck.**

推荐正文句子：

> VK acts as a structural bottleneck between RGB privileged supervision and non-RGB sensing. Unlike raw RGB, it suppresses appearance, texture, and background information, while preserving body topology and motion-relevant geometry. Therefore, VK is used as a bridge for structural supervision rather than as a claim of formal privacy protection.

### 3.3 为什么不是简单替换

如果只是把 RGB 文件替换成 VK 文件，那确实没有充分创新。

VK-RMD 必须强调的是：

1. RGB 是训练期 privileged source。
2. VK 是从 RGB 中压缩出来的 body-structure representation。
3. VK 不走图像 backbone，而走 joint-level structural encoder。
4. 非 RGB 模态不是和 RGB 图像像素对齐，而是和 VK body-token space 对齐。
5. 最终 student 在缺失模态条件下学习任意模态组合到 HPE/HAR 的映射。

因此，VK 的贡献不在于“多一个输入”，而在于提供统一身体结构坐标系。

---

## 4. RGB-to-VK 后如何与其他模态对齐

这是论文 Method 中必须补清楚的内容。建议新增小节：

> **VK-bridged modality alignment**

对齐需要分成四层讲。

---

### 4.1 时间对齐：sample-level synchronization

MMFi-style 数据本身是多模态同步采集的。RGB、Depth、LiDAR、mmWave、WiFi-CSI 对应同一动作序列和同一 frame index。

因此，VK 不是单独生成的孤立数据，而是从同一时刻 RGB frame 得到的 keypoint-level representation。它和其他模态共享：

- 同一 subject；
- 同一 action；
- 同一 environment；
- 同一 frame index；
- 同一 HPE / HAR label。

这解决的问题是：

> VK 与其他模态是否描述的是同一时刻、同一个人的同一个动作？

回答：

> 是。VK 由同步 RGB frame 生成，并与其他传感器流通过 frame-level sample index 对齐。

推荐正文句子：

> Since MMFi-style data are synchronously collected, VK is generated from the RGB frame corresponding to the same subject, action, environment, and frame index as the non-RGB sensor streams. The alignment is therefore established at the sample level before feature learning.

---

### 4.2 表征对齐：from dense appearance to sparse structure

原始 RGB 是 dense image，VK 是稀疏关键点。两者不能共用 ResNet18。

旧 RGB branch：

```text
RGB image -> ResNet18 -> dense visual feature
```

新 VK branch：

```text
17 keypoints
  -> joint-wise coordinate lifting
  -> skeleton/global joint interaction
  -> VK structural token
```

按现有代码逻辑可以解释为：

```text
nn.Linear(2, 64) 对每个关节独立升维
nn.Linear(17, 32) 或 skeleton projector 做全局关节交互
得到 VK structural representation
```

核心句：

> RGB-to-VK changes the input from dense appearance to sparse body geometry. Therefore, we replace the ResNet image encoder with a joint-level structural encoder rather than forcing keypoints into an image backbone.

这回答了：

> 之前是 RGB，现在是 VK，数据不一样怎么办？

答案：

> 正因为数据形态不同，所以不能沿用 ResNet18。我们将 RGB 图像分支替换为轻量级骨架结构编码器，使模型从外观建模转向身体结构建模。

---

### 4.3 Token 维度对齐：shared body-token space

不同模态的 raw data 完全不同：

- VK：17 个 2D/3D 关节点；
- Depth：深度图；
- LiDAR：点云；
- mmWave：稀疏雷达点云；
- WiFi-CSI：无线信道矩阵。

不能在 raw space 对齐，只能在 feature/token space 对齐。

统一写法：

```text
x_m -> modality-specific encoder f_m -> projector P_m -> shared token z_m
```

公式：

```text
z_m = P_m(f_m(x_m)) + e_m
```

其中：

- `f_m` 是每个模态自己的 backbone / encoder；
- `P_m` 把不同维度特征投影到统一 token 维度；
- `e_m` 是 modality embedding，用来告诉模型这个 token 来自哪种模态；
- `z_m` 是进入 fusion 模块的 shared body token；
- 缺失模态时，只输入当前可用 token。

这就是 VK 和其他模态的对齐方式：

> **不是像素级对齐，而是身体语义 token 空间对齐。**

推荐正文句子：

> We do not require raw-space correspondence between heterogeneous sensors. Instead, each modality is encoded by its own backbone and projected into a shared body-token space, with modality embeddings preserving the sensor identity.

---

### 4.4 任务监督对齐：label- and structure-driven alignment

最终对齐不是靠强行让所有模态的原始特征一样，而是靠共同任务监督。

HPE：

```text
all modality subsets -> same 3D pose label
```

HAR：

```text
all modality subsets -> same action label
```

Teacher/student：

```text
missing-modality student -> full-modality teacher output / structure
```

Reliability fusion：

```text
available modality tokens -> learned modality contribution weights
```

推荐正文句子：

> The alignment is label- and structure-driven. All modalities are synchronized at the sample level, projected into a shared token dimension, and optimized toward the same pose or action target. This avoids requiring raw-space correspondence between heterogeneous sensors.

这层对齐解决：

> Depth、LiDAR、mmWave、WiFi-CSI 与 VK 的数据完全不同，为什么能一起融合？

答案：

> 因为它们描述同一时刻同一人体状态，并通过共同 HPE/HAR 监督被投影到同一身体结构语义空间。

---

## 5. 方法论创新怎么写

不要说每个模块都是首创。应该把创新收缩成一个新机制组合：

> **Structure-mediated missing-modality learning.**

或：

> **VK-anchored subset-invariant multimodal learning.**

它包含三件事。

### 5.1 VK structural bottleneck

VK 把 RGB privileged knowledge 压缩成结构监督。

写法：

> VK serves as a structural bottleneck that transfers RGB-derived body knowledge without exposing raw visual appearance at inference.

### 5.2 Shared body-token alignment

不同模态通过 projector 进入统一 token 空间，并由 modality embedding 保留来源信息。

写法：

> Heterogeneous sensors are not aligned in the raw input space. They are aligned after modality-specific encoding and projection into a shared body-token space.

### 5.3 Reliability-aware subset fusion

对任意可用模态组合，模型学习当前组合下的融合权重。

写法：

> The model estimates a learned reliability proxy for each available modality and fuses the subset representation according to the current sensing condition.

### 5.4 推荐贡献句

> The methodological contribution is not modality dropout or attention alone, but a structure-mediated missing-modality learning framework that uses VK as a body-structure bridge, aligns heterogeneous sensors in a shared token space, and performs reliability-aware fusion under arbitrary modality availability.

中文：

> 本文的方法论贡献不是单独的 modality dropout 或 attention，而是一种结构介导的缺失模态学习框架：以 VK 作为身体结构桥梁，将异构传感器对齐到共享 token 空间，并在任意模态可用条件下进行可靠性感知融合。

---

## 6. 2025/2026 Baseline 怎么处理

不能盲目说：

> 我们超过所有 2025/2026 方法。

更稳的处理方式是把近期方法分成三类。

### 6.1 Missing-modality learning

近期 missing-modality 方法通常包括：

- modality dropout；
- missing modality generation；
- knowledge distillation；
- masked modeling；
- parameter-efficient adaptation；
- modality-invariant representation learning。

这些方法可以在 Related Work 中讨论，但不一定能直接作为 MMFi HPE/HAR baseline。

原因：

- 数据集不同；
- 任务不同；
- 输入模态不同；
- 缺失设置不同；
- 代码未必公开；
- 适配 MMFi 成本高。

### 6.2 X-Fi / modality-invariant human sensing

X-Fi 是最直接相关 baseline，因为它同样关注多模态人体感知，并涉及 VK / MMFi / XRF55 等设置。

本文应使用：

- official X-Fi/VK table；
- reproduced X-Fi/VK baseline；
- full-modality baseline under missing test。

注意：

> `244.91 mm` 这类本地复现弱结果不能作为主结论的唯一 baseline。HPE 主表应使用 official X-Fi/VK table 中的 `103.70 mm` average 和 `83.70 mm` full。

### 6.3 Selective / adaptive multimodal HPE or HAR

近期一些方法会强调：

- 根据模态质量选择传感器；
- LiDAR / mmWave / camera selective fusion；
- sensor reliability estimation；
- adaptive multimodal fusion。

这些适合在 Related Work 中作为方法背景，但如果没有同 MMFi、同 HPE/HAR、同模态缺失设置，不建议强行数值对比。

### 6.4 最稳 baseline 策略

正文主表使用三类 baseline：

1. **Official external baseline**
   - official X-Fi/VK table；
   - reproduced X-Fi/VK HAR result。

2. **Fair internal baseline**
   - full-modality baseline under missing test；
   - uniform fusion；
   - w/o KD；
   - simple VK encoder；
   - Student-NV。

3. **Related-work comparison table**
   - 2025/2026 方法只做能力维度比较：
     - 是否支持 HPE；
     - 是否支持 HAR；
     - 是否支持 arbitrary missing modality；
     - 是否不使用 RGB inference；
     - 是否有 reliability-aware fusion；
     - 是否有 all-combination evaluation。

这比强行复现多个最新 baseline 更稳。

---

## 7. P0-1：创新性不足，像模块堆叠

### 7.1 审稿人可能质疑

> 这不就是 modality dropout + attention/gating + KD 吗？

### 7.2 不能这样回应

不要说：

- 我们首次提出 modality dropout；
- 我们首次提出 attention fusion；
- 我们首次提出 KD；
- 我们完全解决任意模态缺失。

这些都容易被反驳。

### 7.3 正确回应

承认单个模块不是首创，但强调本文贡献在于：

1. **Problem formulation**
   - 将 reduced-visual-exposure multimodal HPE/HAR 建模为 missing-modality learning 问题。

2. **VK structural bridge**
   - 用 VK 作为 RGB privileged knowledge 与非 RGB sensing 的结构中间层。

3. **Shared body-token alignment**
   - 将异构传感器统一到身体结构 token 空间。

4. **Subset-invariant training**
   - 训练目标覆盖任意模态子集，而不是只优化 full modality。

5. **Systematic validation**
   - HPE 31 组合、HAR 15 组合、random / cross-scene / cross-subject、Student-VK / Student-NV、消融和可靠性权重。

### 7.4 建议加入对比表

```text
Method type              | Missing modality | VK bridge | Non-RGB inference | HPE+HAR | All-combination eval | Reliability-aware fusion
X-Fi                     | partial           | partial   | yes               | yes     | limited/reproduced   | X-fusion
Modality dropout          | yes               | no        | task-dependent    | no/var  | often partial         | no/var
KD missing modality       | yes               | no        | task-dependent    | no/var  | often partial         | no/var
Adaptive fusion           | partial           | no/var    | task-dependent    | no/var  | often partial         | yes/var
VK-RMD                    | yes               | yes       | yes               | yes     | full                  | yes
```

### 7.5 推荐正文句

> We do not claim novelty from modality dropout, gating, or distillation in isolation. The novelty lies in using VK as a structural bridge for reduced-visual-exposure human sensing and validating the resulting framework under exhaustive missing-modality HPE/HAR protocols.

中文：

> 本文不声称 modality dropout、gating 或 distillation 本身是首次提出。本文的贡献在于将 VK 作为 reduced-visual-exposure 人体感知的结构桥梁，并在 HPE/HAR 全组合缺失模态协议下系统验证该框架。

---

## 8. P0-2：Privacy Claim 不够成立

### 8.1 审稿人可能质疑

> VK/keypoints 仍可能泄露身份、步态、身高、性别或动作习惯，凭什么说 privacy-friendly？

这个质疑是成立的。VK 不等于匿名，也不等于正式隐私保护。

### 8.2 必须降调

不能写：

- privacy-preserving；
- anonymous；
- privacy guarantee；
- secure；
- identity-free；
- formal privacy protection。

可以写：

- privacy-friendly；
- RGB-free inference；
- reduced visual exposure；
- appearance-suppressed representation；
- reduced access to raw visual appearance。

### 8.3 必须明确边界

推荐正文句子：

> VK still contains body geometry and motion cues, which may reveal identity-related information such as gait or body shape. Therefore, this work does not provide formal privacy guarantees such as differential privacy or cryptographic protection.

然后接：

> The privacy-related contribution is to remove raw RGB from inference and reduce visual exposure, not to guarantee anonymity.

### 8.4 建议加入 Privacy Exposure Table

```text
Representation | Raw appearance | Face/background | Body geometry | Motion/gait leakage | Formal privacy guarantee | Our usage
RGB            | High           | High            | High          | High                | No                       | Training privileged source only
VK             | Low            | Low             | Medium/High   | Medium/High         | No                       | Structural bridge / Student-VK
Depth          | Medium         | Low/Medium      | High          | Medium              | No                       | Non-RGB sensing
LiDAR/mmWave   | Low            | Low             | Medium        | Medium              | No                       | Non-RGB sensing
WiFi-CSI       | Very low visual| Low             | Indirect      | Possible            | No                       | Non-visual sensing
```

表格结论：

> VK and non-RGB modalities reduce visual exposure but do not provide formal privacy guarantees.

这样写反而更可信。

---

## 9. 主线收缩方案

当前论文不要再包装成“全新单点算法”或“严格隐私保护方法”。最稳主线应改为：

> **VK-bridged missing-modality learning for reduced-visual-exposure multimodal human perception.**

也就是：

- VK 不是隐私保证，而是 RGB 外观信息与非 RGB 传感器之间的结构桥梁；
- 方法核心不是单纯 KD，而是随机模态子集训练 + reliability-aware fusion + HPE/HAR 全组合缺失模态验证；
- 贡献边界从“全新算法”调整为“面向真实部署约束的结构化学习框架”。

---

## 10. Key Changes for Manuscript

### 10.1 标题与贡献降调

建议从 `Distillation` 主标题中移除，改为：

- `VK-RML: Reliability-Guided Missing-Modality Learning for Reduced-Visual-Exposure Multimodal Human Perception`
- `Visual-Keypoint Bridged Missing-Modality Learning for Multimodal HPE and HAR`
- `VK-RMD: Visual-Keypoint Anchored Missing-Modality Learning for Multimodal HPE and HAR`

Distillation 写成辅助训练信号：

- HAR 中有帮助；
- HPE 中不稳定；
- 不能作为主创新；
- 不能写成所有提升来源。

---

### 10.2 VK 作为桥梁的故事线

正文应明确：

- RGB 在训练期提供强结构监督，但推理期不使用 RGB；
- VK 将 RGB 外观丰富信息压缩为人体结构瓶颈；
- VK 去除或弱化纹理、背景、脸部外观等原始视觉暴露；
- VK 统一 HPE 和 HAR 的身体结构语义；
- Depth、LiDAR、mmWave、WiFi-CSI 在同一身体结构 token 空间中对齐；
- VK 不是匿名保证，只是 reduced visual exposure。

---

### 10.3 方法论创新重新定义

不说：

> We invent modality dropout / attention / KD.

而说：

> We formulate reduced-visual-exposure multimodal human perception as VK-bridged subset-invariant missing-modality learning.

具体包含：

- 训练时随机采样模态子集；
- 每个模态产生 body token；
- reliability-aware gate 学习当前组合下的融合权重；
- teacher/KD 作为可选结构监督；
- 测试时覆盖所有模态组合，而不是只测 full modality。

创新点放在：

- 问题设定；
- 系统化协议；
- VK 桥梁；
- 全组合验证；
- HPE/HAR 双任务一致性；
- fairness-aware internal baselines。

---

### 10.4 公平 baseline 处理

主表不能再只用 `244.91 mm` reproduced baseline 来证明巨大提升。

HPE 主 baseline 使用：

```text
Official X-Fi/VK table:
Average MPJPE = 103.70 mm
Full-modality MPJPE = 83.70 mm
```

同时加入已有内部强 baseline：

- full-modality baseline under missing test；
- uniform fusion；
- w/o KD；
- simple VK encoder；
- Student-NV。

这样回答审稿人：

> 不是因为对手没训练缺失模态所以我们赢。我们还与 missing-trained uniform fusion、w/o KD 和其他内部变体对比。

---

### 10.5 Reliability claim 修正

将：

```text
reliability weights
```

改写为：

```text
learned reliability proxy
reliability-aware modality weights
learned modality-contribution scores
```

不要说：

> 模型真实感知了传感器可靠性。

要说：

> 模型根据当前模态组合自适应调整各模态对任务预测的贡献。

支撑证据：

- uniform fusion ablation；
- reliability weight visualization；
- missing-modality grouped results。

---

### 10.6 Cross-scene 处理

不回避 cross-scene 性能下降。

写成：

> Missing-modality robustness is not equivalent to scene generalization.

论文主结论限制在：

- fixed environment；
- calibrated sensor setup；
- known deployment domain；
- controlled smart-space sensing。

cross-scene 作为 limitation 和 future work。

---

## 11. Minimal Experiments

### P0：不新增大训练

不建议继续跑大规模 no-random 训练，时间成本太高。

P0 必须补的是论文组织和表格，而不是新模型：

1. Fair Missing-Modality Baseline Table；
2. Privacy Exposure and Residual Risk Table；
3. Reliability-aware Fusion Evidence Table；
4. VK-bridged modality alignment subsection。

---

### P1：只用已有结果整理

强烈建议补但不需要新训练：

1. HPE/HAR missing-modality 主表：
   - full；
   - single-missing；
   - multi-missing；
   - all-combination average。

2. Reliability fusion 对比表：
   - ours vs uniform fusion。

3. KD 诚实表：
   - ours vs w/o KD；
   - 说明 HPE 中 KD 不是稳定收益；
   - HAR 中 KD 有辅助收益。

---

### P2：不建议现在做

不建议继续补：

- 2025/2026 大 baseline 复现；
- 完整 3 次随机种子；
- SuperTeacher 主线扩展；
- 差分隐私、加密、身份反推实验；
- no-random 100 epoch 大训练；
- 新数据集迁移实验。

原因：

> 当前目标是尽快投稿工程型 SCI。继续扩大战线会拖慢论文完成，并且可能让主线更加分散。

---

## 12. Paper Logic

正文贡献建议改成三条。

### Contribution 1: VK-bridged reduced-visual-exposure formulation

用 VK 作为 RGB privileged knowledge 与非 RGB 传感器之间的结构桥梁，支持 RGB-free inference。

推荐表述：

> We formulate reduced-visual-exposure multimodal human perception as VK-bridged body-token learning, where RGB-derived keypoints provide structural privileged information while raw RGB is removed from inference.

---

### Contribution 2: Reliability-aware missing-modality learning

通过随机模态子集训练和 learned reliability proxy，使模型支持任意模态组合推理。

推荐表述：

> We train the student under randomly sampled modality subsets and introduce a learned reliability proxy to adaptively fuse the currently available modalities.

---

### Contribution 3: Systematic HPE/HAR validation

在 HPE 和 HAR 上进行 random、cross-subject、cross-scene、all-combination missing-modality 测试，展示适用边界和失败场景。

推荐表述：

> We validate the framework on both HPE and HAR with exhaustive modality-combination evaluation, internal ablations, and cross-domain protocols, while explicitly reporting its cross-scene limitations.

---

## 13. Manuscript Test Plan

修改论文后必须检查以下内容。

### 13.1 删除或弱化高风险词

全文搜索并删除或降调：

- `privacy-preserving`
- `privacy guarantee`
- `anonymous`
- `identity-free`
- `distillation consistently improves`
- `solve missing modality`
- `robust cross-scene deployment`
- `first`
- `universal`

---

### 13.2 主表必须包含

HPE / HAR 主结果中应包含：

- official X-Fi/VK baseline；
- full-modality baseline；
- uniform fusion；
- w/o KD；
- ours；
- Student-VK；
- Student-NV。

---

### 13.3 SuperTeacher 位置检查

SuperTeacher：

- 不出现在 Abstract；
- 不作为 Contribution 主语；
- 不进入 Main Result 主表；
- 只放 Discussion / Appendix；
- 作为探索实验，不作为主线支柱。

---

### 13.4 Reliability 表述检查

所有 reliability 相关文字应使用：

- learned reliability proxy；
- reliability-aware fusion；
- learned modality-contribution weights；
- adaptive subset fusion。

避免：

- true sensor reliability；
- physical reliability measurement；
- calibrated sensor quality；
- causal reliability explanation。

---

## 14. Final Recommended Story

最终全文主线可压缩成：

> VK-RMD converts RGB-derived visual knowledge into a VK-anchored body-token space, trains the student under randomly sampled modality subsets, and learns reliability-aware fusion for each available subset, enabling reduced-visual-exposure HPE and HAR under arbitrary sensor availability.

中文：

> VK-RMD 将 RGB 视觉知识压缩到以 VK 为锚点的人体结构 token 空间，在随机模态子集下训练 student，并为每种可用模态组合学习可靠性融合权重，从而在不使用原始 RGB 推理的条件下实现 HPE 和 HAR 的缺失模态鲁棒感知。

---

## 15. Final Assumptions

- 当前目标是尽快投稿工程型 SCI，而不是重做算法型顶会论文。
- 不新增大规模实验。
- SuperTeacher 只作为探索或附录，不再作为主贡献中心。
- VK 的核心价值是结构桥梁和视觉暴露降低，而不是严格匿名。
- Distillation 是辅助机制，不是全文唯一核心。
- Reliability 是 learned proxy，不是真实物理可靠性测量。
- Cross-scene 性能下降必须作为 limitation 诚实呈现。

