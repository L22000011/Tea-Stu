# X-Fi 与 VK-RMD 的代码机制差异分析

本文档用于内部论文写作、结果解释和 reviewer response。它不是论文正文，而是把 X-Fi 官方实现与当前 VK-RMD 实现之间的本质差异讲清楚，尤其解释为什么我们的 Depth 单模态结果显著强于 X-Fi 时，不能简单归因于 VK 或 KD。

## 1. 总体定位差异

**X-Fi** 可以概括为：

> RGB-centered modality-invariant fusion model.

它以 `rgb/depth/lidar/mmwave/wifi-csi` 为输入模态，通过各自 backbone 抽取特征，投影到统一 token 数量后，用 `X_Fusion` 进行 cross-modal transformer 融合，最后由一个 regression head 输出 HPE pose。

**VK-RMD** 可以概括为：

> VK/non-RGB missing-modality expert model.

它不使用 raw RGB 作为 downstream 输入，而是使用 `vk/depth/lidar/mmwave/wifi-csi`。VK 分支把 17×2 视觉关键点编码成 skeleton-aware tokens；非 VK 模态复用 X-Fi-style 预训练 sensor backbones；融合阶段不是只输出一个 fused pose，而是为每个模态 chunk 生成 pose expert 和 uncertainty，再进行 reliability-weighted decision fusion。Student 训练时随机采样模态子集，并接受 full-modality teacher 的 full-to-subset guidance。

最关键的结论是：

> VK-RMD 的 Depth 单模态提升主要来自 subset-conditioned training 和 per-modality expert supervision，而不是 VK 或 KD 单独造成。

---

## 2. X-Fi 官方 HPE 代码机制

本节依据 GitHub `NTUMARS/X-Fi` 当前 main 分支中的 `MMFi_HPE/X_Fi.py`、`MMFi_HPE/utils.py`、`MMFi_HPE/syn_DI_dataset.py` 和 `MMFi_HPE/config.yaml`。

### 2.1 输入模态

X-Fi 的 HPE 配置中使用：

```yaml
modality: ['rgb', 'depth', 'lidar', 'mmwave', 'wifi-csi']
```

代码中没有 VK branch。视觉分支是 raw RGB image branch。

### 2.2 模态特征抽取

X-Fi 的 `feature_extrator` 对不同模态使用不同 backbone：

| 模态 | X-Fi 处理方式 |
|---|---|
| RGB | `RGB_ResNet18`，去掉最后层，输出 image feature map |
| Depth | `Depth_ResNet18`，去掉最后层，输出 image feature map |
| mmWave | `mmwave_PointTransformerReg` 的 backbone 部分 |
| LiDAR | `lidar_PointTransformerReg` 的 backbone 部分 |
| WiFi-CSI | CSI benchmark encoder |

在 `feature_extrator.forward()` 中，根据 `modality_list` 选择当前可用模态，将可用模态的 feature list 返回。

### 2.3 统一 token 投影

X-Fi 的 `linear_projector` 将不同模态特征投影到统一 token 表达：

```text
RGB / Depth: 49 tokens -> 32 tokens
mmWave / LiDAR: 32 tokens -> 32 tokens
WiFi-CSI: 17*4 tokens -> 32 tokens
```

投影后的特征大致为：

```text
B × (32 × number_of_modalities) × 512
```

LiDAR 分支还会通过 `selective_pos_enc()` 生成位置编码并加到 `projected_feature` 上。

### 2.4 X-Fusion 融合机制

X-Fi 的核心融合模块是 `X_Fusion`，由以下部分组成：

1. `cross_modal_transformer`
   - 对 concat 后的多模态 token 做 self-attention。
   - 输出一个固定 32-token 的 shared feature embedding。

2. `kv_projection`
   - 对每个可用模态生成自己的 K/V。

3. `fusion_transformer_block`
   - 使用 shared feature embedding 作为 Q。
   - 使用每个模态的 K/V 进行 cross-attention。
   - 将每个模态交互后的结果 concat。

4. 迭代融合
   - 重复执行 cross-attention fusion 和 cross-modal transformer。

5. `regression_Head`
   - 对最终 feature embedding 做 mean pooling 和 linear projection。
   - 输出 `17 × 3` pose。

可以概括为：

```text
selected modality features
-> linear projector
-> cross-modal transformer
-> per-modality KV cross-attention
-> cross-modal transformer
-> single regression head
-> pose
```

### 2.5 X-Fi 训练机制

X-Fi 的 `utils.py` 中有 `generate_none_empth_modality_list()`，训练时随机生成当前 batch 的可用模态：

```text
RGB / Depth / mmWave: 50% selected
LiDAR / WiFi: 70% selected
至少保留一个模态
```

训练时使用：

```python
train_criterion = nn.MSELoss()
outputs = model(..., modality_list)
loss = MSE(outputs, labels)
```

这说明 X-Fi 确实也有随机模态组合训练，但它的监督对象是 **最终 output pose**。代码中没有看到 per-modality pose expert，也没有对每个单模态分支的独立 pose 输出进行 GT 监督。

---

## 3. VK-RMD 当前 HPE 代码机制

本节依据本地实现：

- `HPE/models/vk_rcd.py`
- `HPE/models/skeleton_prompt_encoder.py`
- `HPE/models/reliability_fusion.py`
- `HPE/losses/distill_losses.py`
- `HPE/training/engine.py`

### 3.1 输入模态

VK-RMD HPE 使用：

```text
vk, depth, lidar, mmwave, wifi-csi
```

HAR 使用：

```text
vk, depth, lidar, mmwave
```

主线 downstream model 不使用 raw RGB。VK 原始输入是从 `rgb` 文件夹读取的 `.npy` 关键点文件，shape 为：

```text
17 × 2
```

这不是图像，也不是预先存好的 skeleton graph。模型内部再把它编码成 skeleton-aware tokens。

### 3.2 VK skeleton-aware branch

VK 分支由 `SkeletonPromptEncoder` 实现：

```text
17×2 keypoints
-> normalize_keypoints
-> coord_mlp + position_mlp + joint_embedding
-> BoneGraphMixer with COCO17_BONES
-> token_projector
-> 32 × 512 VK tokens
```

其中 `BoneGraphMixer` 使用 COCO17 骨架边构建邻接矩阵，并执行邻居聚合：

```python
neighbor_tokens = torch.einsum("ij,bjd->bid", adjacency, tokens)
mixed = MLP([tokens, neighbor_tokens])
tokens = LayerNorm(tokens + mixed)
```

需要注意：图邻接、邻居聚合、残差 MLP 不是概念上全新的骨架网络理论。本文更稳妥的说法是：

> VK-RMD introduces a lightweight skeleton-aware VK branch to replace the raw RGB branch in the downstream multimodal perception pipeline.

不要写成：

> 我们提出了全新的 skeleton graph learning 方法。

### 3.3 非 VK 模态编码

非 VK 模态仍然复用 X-Fi-style sensor backbones：

| 模态 | VK-RMD 处理方式 |
|---|---|
| Depth | `DepthFeatureExtractor`，加载 depth ResNet18 pretrained weights |
| LiDAR | `LidarFeatureExtractor`，加载 lidar PointTransformer pretrained weights |
| mmWave | `MMwaveFeatureExtractor`，加载 mmWave PointTransformer pretrained weights |
| WiFi-CSI | `CSIFeatureExtractor`，加载 CSI benchmark model |

随后每个非 VK 模态通过 `TokenProjector` 映射为：

```text
32 × 512 tokens
```

所以 VK-RMD 不是完全重写全部 backbone，而是在 X-Fi-style backbone 之上重构了视觉入口、融合机制和训练目标。

### 3.4 Modality token interaction

VK-RMD 的 `ModalityTokenEncoder` 对当前可用模态 tokens 加 modality embedding，然后输入 TransformerEncoder：

```text
projected[name] + modality_embedding[name]
-> concatenate selected modality tokens
-> TransformerEncoder
```

这一步实现的是 task-supervised token interaction。论文中应该谨慎写作：

> task-supervised body-latent tokenization

而不是过度声称：

> 严格证明所有模态已语义对齐到人体空间。

### 3.5 ReliabilityFusion 与 per-modality expert

VK-RMD 的最大结构差异在 `ReliabilityFusion`：

1. 将 encoded tokens 按模态切分。
2. 每个模态 chunk 经过同一个 `PoseExpertHead`。
3. 每个 chunk 输出：
   - 一个 `pose`
   - 一个 `logvar`
4. 用 `softmax(-logvar / tau)` 得到 modality weights。
5. weighted sum 得到 `fused_pose`。
6. 同时从 weighted pooled token 得到 `residual_pose`。
7. 最终输出：

```text
pose = 0.5 * fused_pose + 0.5 * residual_pose
```

也就是说，VK-RMD 不是只在 feature 层融合一次，而是显式形成：

```text
Depth expert pose
LiDAR expert pose
mmWave expert pose
WiFi expert pose
VK expert pose
```

然后再由 learned reliability proxy 进行决策级融合。

#### 3.5.1 residual pose 是怎么得到的

`residual_pose` 是 VK-RMD fusion 里的第二条预测路径。它不是某一个单模态 expert 的输出，而是从 reliability-weighted fused token 中再预测出来的一个补充 pose。

代码逻辑位于 `HPE/models/reliability_fusion.py`：

```python
pooled = torch.stack(pooled_tokens, dim=1)
fused_token = torch.sum(alphas[:, :, None] * pooled, dim=1)
residual_pose = self.final_head(self.final_norm(fused_token)).view(...)
pose = 0.5 * fused_pose + 0.5 * residual_pose
```

具体含义如下：

1. 每个模态 chunk 先做 mean pooling，得到一个模态级 token：

```text
VK chunk       -> pooled_vk
Depth chunk    -> pooled_depth
LiDAR chunk    -> pooled_lidar
```

2. 根据 learned reliability weights 做加权求和：

```text
fused_token =
  alpha_vk    * pooled_vk
+ alpha_depth * pooled_depth
+ alpha_lidar * pooled_lidar
```

3. 把这个 `fused_token` 送入一个线性预测头：

```text
fused_token -> LayerNorm -> Linear(512, 17*3) -> residual_pose
```

4. 最终 pose 是两条路径的平均：

```text
final_pose = 0.5 * fused_pose + 0.5 * residual_pose
```

其中：

- `fused_pose` 来自每个模态 expert 预测结果的加权和；
- `residual_pose` 来自融合后的 latent token 表示；
- `final_pose` 同时利用了 decision-level expert prediction 和 token-level residual prediction。

可以把它理解为：

> per-modality experts 先给出每个传感器自己的姿态判断；residual head 再基于融合后的全局 token 做一次补充修正。

这和 X-Fi 的单一 regression head 不同。X-Fi 是融合后的 feature 直接输出一个 pose；VK-RMD 是 `expert pose path + residual pose path` 两条路径共同输出。

---

## 3.6 一个完整前向例子：VK + Depth + LiDAR

为了画图和理解，下面用一个 batch 中当前可用模态为：

```text
S = {VK, Depth, LiDAR}
```

来走完整流程。

### Step 1: 输入数据

当前 batch 中包含所有原始输入，但 student 这一次只选择 `VK+Depth+LiDAR`：

```text
VK:       17 × 2 keypoint coordinates
Depth:    depth image
LiDAR:    point cloud
mmWave:   exists in batch but not selected
WiFi-CSI: exists in batch but not selected
```

未被选中的 mmWave 和 WiFi-CSI 不进入 student forward。

### Step 2: 每个模态各自编码

VK 走我们自己的 skeleton-aware branch：

```text
VK 17×2
-> normalize_keypoints
-> coord_mlp + position_mlp + joint_embedding
-> BoneGraphMixer
-> token_projector
-> Z_vk: 32 × 512
```

Depth 走 X-Fi-style depth backbone 和 projector：

```text
Depth image
-> DepthFeatureExtractor
-> raw depth features
-> TokenProjector
-> Z_depth: 32 × 512
```

LiDAR 走 X-Fi-style LiDAR backbone 和 projector：

```text
LiDAR point cloud
-> LidarFeatureExtractor
-> raw lidar features
-> TokenProjector
-> Z_lidar: 32 × 512
```

此时得到三个模态 token block：

```text
Z_vk     = 32 × 512
Z_depth  = 32 × 512
Z_lidar  = 32 × 512
```

### Step 3: 加 modality embedding

模型给每个模态加一个可学习身份标签：

```text
Z_vk     + e_vk
Z_depth  + e_depth
Z_lidar  + e_lidar
```

这里的 `e_vk/e_depth/e_lidar` 来自 `nn.Embedding(max_modalities, 512)`。它告诉模型：

```text
这些 token 来自 VK
这些 token 来自 Depth
这些 token 来自 LiDAR
```

所以 VK-RMD 不只是靠 token 所在位置来识别模态，而是显式学习模态身份。

### Step 4: 拼接 tokens 并做 token interaction

将三个 token block 按当前选择顺序拼接：

```text
[VK 32 tokens][Depth 32 tokens][LiDAR 32 tokens]
-> total 96 tokens
```

送入 `TransformerEncoder`：

```text
96 × 512 tokens
-> TransformerEncoder
-> 96 × 512 encoded tokens
```

这一步会让 VK、Depth、LiDAR tokens 之间发生 self-attention 交互。

需要注意：

> 这里使用的是标准 TransformerEncoder 内部的 self-attention。它当然也会在内部计算 Q/K/V，但这不是 X-Fi 的 X-Fusion 机制。

X-Fi 是：

```text
shared feature as Q
each modality produces K/V
shared Q cross-attends to modality-specific K/V
```

VK-RMD 是：

```text
selected modality tokens + modality embeddings
-> standard TransformerEncoder self-attention
-> split chunks
-> per-modality experts
-> reliability-weighted decision fusion
```

因此，VK-RMD 没有采用 X-Fi 那套 explicit shared-Q / modality-specific-KV cross-attention fusion。

### Step 5: 按模态切回 chunk

TransformerEncoder 输出仍然是 96 个 tokens。由于每个模态固定 32 个 tokens，所以按顺序切回：

```text
encoded tokens 1-32   -> VK chunk
encoded tokens 33-64  -> Depth chunk
encoded tokens 65-96  -> LiDAR chunk
```

得到：

```text
T_vk     = 32 × 512
T_depth  = 32 × 512
T_lidar  = 32 × 512
```

### Step 6: 每个 chunk 通过 pose expert

每个模态 chunk 经过 `PoseExpertHead`：

```text
T_vk     -> pose_vk,    logvar_vk
T_depth  -> pose_depth, logvar_depth
T_lidar  -> pose_lidar, logvar_lidar
```

其中：

- `pose_*` 是该模态 expert 自己预测的 17×3 pose；
- `logvar_*` 是该模态 expert 对自己预测不确定性的估计。

例如模型可能得到：

```text
logvar_vk     = 0.7
logvar_depth  = 0.1
logvar_lidar  = 0.5
```

logvar 越小，表示模型认为该模态当前更可靠。

### Step 7: 从 logvar 得到 reliability weights

VK-RMD 使用：

```text
alpha = softmax(-logvar / tau)
```

假设得到：

```text
alpha_vk     = 0.25
alpha_depth  = 0.50
alpha_lidar  = 0.25
```

这表示当前样本/当前模态组合下，模型更相信 Depth expert。

注意：

> 这里的 alpha 是 learned reliability proxy，不是物理传感器真实可靠性。

### Step 8: 生成 fused_pose

对每个 expert pose 加权：

```text
fused_pose =
  0.25 * pose_vk
+ 0.50 * pose_depth
+ 0.25 * pose_lidar
```

这是 decision-level fusion，因为它融合的是每个模态已经预测出来的 pose。

### Step 9: 生成 residual_pose

同时，模型还会对每个 chunk 做 mean pooling：

```text
T_vk     -> pooled_vk
T_depth  -> pooled_depth
T_lidar  -> pooled_lidar
```

再用同一组 alpha 做 token-level 加权：

```text
fused_token =
  0.25 * pooled_vk
+ 0.50 * pooled_depth
+ 0.25 * pooled_lidar
```

然后：

```text
fused_token
-> LayerNorm
-> Linear(512, 17*3)
-> residual_pose
```

这个 residual pose 是从融合后的 latent representation 中预测的补充姿态。

### Step 10: 最终输出

最终：

```text
final_pose = 0.5 * fused_pose + 0.5 * residual_pose
```

如果是 Depth-only：

```text
S = {Depth}
alpha_depth = 1
fused_pose = pose_depth
fused_token = pooled_depth
residual_pose = Linear(pooled_depth)
final_pose = 0.5 * pose_depth + 0.5 * residual_pose
```

这就是为什么 Depth-only 可以很强：Depth expert 和 Depth residual path 都在训练中被优化过，而不是临时用一个 full-modality fusion model 去硬适配单模态输入。

### Step 11: 训练时为什么每个 expert 会变强

Student loss 不只监督 `final_pose`，还监督每个 selected modality 的 expert pose：

```text
pose_vk     vs GT
pose_depth  vs GT
pose_lidar  vs GT
```

所以只要 Depth 在某个训练 batch 中被选中，它的 expert pose 就会被 GT 直接训练。长期下来，Depth expert 本身会成为一个强单模态 pose estimator。

这正是 VK-RMD 与 X-Fi 的核心区别之一。

### 3.6 Student training 与 full-to-subset guidance

Student 训练时每个 batch 采样一个模态子集：

```python
selected = sample_missing_modalities(available_modalities, drop_counts, rng)
teacher_output = teacher(batch["inputs"], teacher_modalities)
student_output = student(batch["inputs"], selected)
loss = compute_student_loss(student_output, target, weights, teacher_output)
```

HPE student loss 包括：

| Loss 项 | 含义 |
|---|---|
| GT pose loss | Student final pose 对 GT |
| output KD | Student final pose 对 teacher pose |
| token KD | Student token 对 teacher token |
| bone KD | Student pose bone length 对 teacher pose bone length |
| reliability KD | Student alpha 对 restricted teacher alpha |
| uncertainty / expert loss | 每个 modality expert pose 对 GT |

其中最关键的是 `distill_losses.py` 中：

```python
for idx, pose in enumerate(student["modality_poses"]):
    err = smooth_l1_loss(pose, target)
    unc_terms.append(exp(-logvar) * err + logvar)
```

这意味着每个可用模态 expert 都会被 GT 直接监督。它是 Depth 单模态显著增强的核心机制之一。

### 3.7 FSG：Full-to-Subset Guidance 完整训练流程

为了避免把当前方法误写成传统“大 teacher 压缩小 student”的知识蒸馏，本文更适合把训练机制称为 **Full-to-Subset Guidance, FSG**。

FSG 的核心不是参数量压缩，而是输入可用性不对称：

```text
Teacher: 使用完整模态集合 M
Student: 使用随机采样模态子集 S, S ⊂ M
```

对于 HPE：

```text
M = {VK, Depth, LiDAR, mmWave, WiFi-CSI}
```

对于 HAR：

```text
M = {VK, Depth, LiDAR, mmWave}
```

训练目标是让 student 在只看到部分模态时，尽量接近 full-modality teacher 的完整感知行为，同时仍然被 GT 直接监督。

#### 3.7.1 FSG 的单个 batch 流程

以 HPE 为例，一个 training batch 的 FSG 流程如下：

```text
Step 1: 读取完整 batch
  inputs = {VK, Depth, LiDAR, mmWave, WiFi-CSI}
  target = 3D pose GT

Step 2: teacher full-modality forward
  teacher_input = {VK, Depth, LiDAR, mmWave, WiFi-CSI}
  teacher_output = Teacher(teacher_input, M)

Step 3: student 随机采样一个可用子集
  S ~ q(S)
  例如 S = {VK, Depth, LiDAR}

Step 4: student subset forward
  student_output = Student(inputs, S)

Step 5: 计算 student loss
  L = GT supervision
    + output guidance
    + token guidance
    + bone guidance
    + reliability guidance
    + per-modality expert uncertainty supervision

Step 6: 只更新 student
  teacher frozen
  student backward + optimizer step
```

代码对应：

```python
selected = sample_missing_modalities(available_modalities, drop_counts, rng)

with torch.no_grad():
    teacher_output = teacher(batch["inputs"], teacher_modalities)

student_output = student(batch["inputs"], selected)
loss, _ = compute_student_loss(student_output, batch["target"], loss_weights, teacher_output)
```

这里的 teacher 使用 full modalities；student 使用 `selected` 子集。teacher 不更新，只提供 guidance。

#### 3.7.2 FSG 指导了什么

HPE 中 FSG 不只是学习 teacher 的最终输出，还包含多层信号：

| Guidance 信号 | 代码变量 | 含义 |
|---|---|---|
| GT supervision | `student["pose"]` vs `target` | Student final pose 必须接近真实 3D pose |
| Output guidance | `student["pose"]` vs `teacher["pose"]` | subset student 学 full teacher 的最终预测 |
| Token guidance | `student["tokens"]` vs `teacher["tokens"]` | subset student 的全局 token 表示接近 full teacher |
| Bone guidance | `bone_length(student["pose"])` vs `bone_length(teacher["pose"])` | 保持 teacher 的骨架结构关系 |
| Reliability guidance | `student["alphas"]` vs restricted `teacher["alphas"]` | student 学 teacher 在对应模态上的融合偏好 |
| Per-modality expert supervision | 每个 `modality_pose` vs `target` | 每个可用模态 expert 自己也要能预测 GT |

因此 FSG 的目的不是单纯让小模型模仿大模型，而是让：

> partial-modality student learns the behavior of full-modality sensing under subset input conditions.

中文：

> 缺失模态 student 在只看到部分传感器的情况下，学习完整模态系统的输出、结构、token 和融合行为。

#### 3.7.3 FSG 公式化表达

定义完整模态集合为：

```text
M = {m_1, m_2, ..., m_K}
```

随机采样 student 可用模态子集：

```text
S ~ q(S),  S ⊆ M,  S ≠ ∅
```

teacher 使用完整模态：

```text
y_T, z_T, alpha_T = F_T({x_m | m ∈ M})
```

student 使用子集模态：

```text
y_S, z_S, alpha_S, {y_m, s_m}_{m∈S} = F_S({x_m | m ∈ S})
```

HPE student 总体目标可以写成：

```text
L_student =
  L_gt(y_S, y)
+ λ_out  L_out(y_S, y_T)
+ λ_tok  L_tok(z_S, z_T)
+ λ_bone L_bone(y_S, y_T)
+ λ_rel  L_rel(alpha_S, Renorm(alpha_T|S))
+ λ_unc  Σ_{m∈S} L_unc(y_m, y, s_m)
```

其中：

- `L_gt`：student final pose 对 GT；
- `L_out`：student final pose 对 teacher pose；
- `L_tok`：student token 对 teacher token；
- `L_bone`：student 与 teacher 的骨长关系；
- `L_rel`：student reliability weights 对 teacher restricted weights；
- `L_unc`：每个模态 expert 的 GT-supervised uncertainty loss。

注意：

> 如果实验中 no-KD 变体表现很强，论文不能声称 FSG 是所有性能提升的唯一来源。更稳妥的说法是：FSG 是 full-to-subset regularization，而 subset-conditioned expert supervision 是性能提升的核心机制之一。

#### 3.7.4 继续使用 VK + Depth + LiDAR 的 FSG 示例

沿用前面的 inference 示例：

```text
Student selected subset S = {VK, Depth, LiDAR}
```

但训练时 teacher 仍然看完整模态：

```text
Teacher input:
  {VK, Depth, LiDAR, mmWave, WiFi-CSI}

Student input:
  {VK, Depth, LiDAR}
```

Teacher forward 后得到：

```text
teacher_pose
teacher_tokens
teacher_alphas over [VK, Depth, LiDAR, mmWave, WiFi-CSI]
```

假设 teacher 的 reliability weights 是：

```text
alpha_T =
  VK       0.20
  Depth    0.35
  LiDAR    0.15
  mmWave   0.20
  WiFi     0.10
```

Student 只使用 `{VK, Depth, LiDAR}`，所以 reliability guidance 会先取 teacher 在这三个模态上的权重，并重新归一化：

```text
teacher alpha restricted to S:
  VK       0.20
  Depth    0.35
  LiDAR    0.15

sum = 0.70

renormalized:
  VK       0.20 / 0.70 = 0.286
  Depth    0.35 / 0.70 = 0.500
  LiDAR    0.15 / 0.70 = 0.214
```

Student 自己预测：

```text
student_pose
student_tokens
student_alphas over [VK, Depth, LiDAR]

pose_vk,    logvar_vk
pose_depth, logvar_depth
pose_lidar, logvar_lidar
```

Student 的 loss 包含：

```text
1. student_pose vs GT
2. student_pose vs teacher_pose
3. student_tokens vs teacher_tokens
4. bone(student_pose) vs bone(teacher_pose)
5. student_alphas vs [0.286, 0.500, 0.214]
6. pose_vk vs GT
7. pose_depth vs GT
8. pose_lidar vs GT
```

这样训练后，Depth branch 会被两类信号强化：

```text
Depth expert 自己要预测 GT
+ Depth 所在 subset 的 final pose 要接近 GT/teacher
```

这就是为什么 Depth-only 测试时可能非常强。它不是临时从融合模型里抽出一个 Depth token，而是在训练中已经被反复要求成为可部署的 pose expert。

#### 3.7.5 FSG 与传统 KD 的区别

传统 KD 常见形式是：

```text
Large teacher -> small student
full input teacher -> same input student
student 学 teacher logits / features
```

VK-RMD 的 FSG 是：

```text
full-modality teacher -> subset-modality student
teacher/student 参数量可以接近
差异主要来自输入可用性，而不是模型大小
student 学 teacher output/token/structure/reliability behavior
```

因此，FSG 更适合被描述为：

> full-to-subset behavior regularization

而不是简单写成：

> model compression distillation.

论文中的安全写法：

> We use full-to-subset guidance to regularize the missing-modality student with the behavior of a full-modality teacher. This differs from conventional compression-oriented distillation because the main gap lies in modality availability rather than model capacity.

中文：

> 我们使用 full-to-subset guidance，让缺失模态 student 在部分输入条件下学习完整模态 teacher 的感知行为。这不同于传统模型压缩蒸馏，因为 teacher 和 student 的主要差距不是参数规模，而是可用模态集合。

---

## 4. 本质区别对照表

| 维度 | X-Fi | VK-RMD |
|---|---|---|
| 视觉入口 | raw RGB branch | VK skeleton-aware branch |
| 输入模态 | RGB, Depth, LiDAR, mmWave, WiFi-CSI | VK, Depth, LiDAR, mmWave, WiFi-CSI |
| 隐私定位 | 保留 raw RGB | downstream RGB-free / reduced visual exposure |
| VK 处理 | 无 VK branch | 17×2 keypoints -> joint embedding + bone topology -> skeleton-aware tokens |
| 非视觉 backbone | 预训练 sensor backbones | 复用 X-Fi-style sensor backbones |
| token 投影 | 每模态投影到 32 tokens | 每模态投影到 32 tokens |
| 融合机制 | X-Fusion cross-modal transformer + per-modality KV attention | ModalityTokenEncoder + per-modality pose expert + reliability-weighted fusion |
| 输出头 | 一个 regression head 输出 pose | 每模态 expert pose + fused pose + residual pose |
| 训练监督 | 最终 output pose MSE | final pose + teacher guidance + per-modality expert GT supervision |
| 缺失模态角色 | 支持组合输入，随机 modality list 训练 | 把 subset availability 作为 Student 训练目标中心 |
| Teacher-student | 无 full-to-subset guidance | full-modality teacher 指导 missing-modality student |
| 单模态能力来源 | 由共享融合模型泛化到单模态 | 每个 selected modality expert 被直接训练为可用 predictor |

---

## 5. 为什么 VK-RMD Depth-only 可能显著强于 X-Fi

现有结果：

| 方法 | Depth-only MPJPE |
|---|---:|
| X-Fi official Table 1 | 101.8 mm |
| VK-RMD Teacher | 234.5 mm |
| VK-RMD full baseline | 172.8 mm |
| VK-RMD Student-VK | 53.88 mm |
| VK-RMD Student-NV | 52.34 mm |
| VK-RMD w/o KD | 52.46 mm |
| VK-RMD uniform fusion | 55.72 mm |
| VK-RMD Student-VK cross-subject | 53.23 mm |
| VK-RMD Student-VK cross-scene | 71.91 mm |

这些结果说明：

1. **Depth 提升不是 VK 导致的。**  
   Student-NV 不使用 VK，但 Depth-only 仍为约 52.34 mm。

2. **Depth 提升不是 KD 单独导致的。**  
   `w/o KD` 的 Depth-only 约 52.46 mm，甚至略优于 full Student-VK。

3. **Depth 提升不是 reliability fusion 单独导致的。**  
   uniform fusion 的 Depth-only 仍为约 55.72 mm。单模态下本来也几乎没有融合选择问题。

4. **最合理解释是 subset-conditioned expert training。**  
   Student 训练时持续在缺失模态子集下优化，并且每个模态 expert 的 pose 都被 GT 直接监督。Depth 是强几何模态，因此它从这种训练中获益最大。

5. **不是纯 random split 泄漏即可解释。**  
   random split 中同一 subject 会跨 action 同时出现在 train/val，因此存在 subject-level leakage 风险。但 cross-subject Depth-only 仍约 53.23 mm，cross-scene 约 71.91 mm，说明 Depth 强结果不只是 random split 幻觉。

最安全的论文解释是：

> The strong Depth-only result is mainly attributed to subset-conditioned student training with per-modality expert supervision. Depth already preserves dense body geometry, and VK-RMD explicitly optimizes each available modality to serve as a deployable pose expert under missing-modality conditions.

不应写成：

> VK 让 Depth 变强。

也不应写成：

> 我们的 Depth backbone 比 X-Fi 更强。

---

## 6. 审稿风险与安全写法

### 6.1 审稿风险 1：是否只是 X-Fi 小改？

如果只说“我们换了 VK，加了 KD 和 reliability”，审稿人容易认为是 X-Fi 上的小修补。

更稳写法：

> Compared with X-Fi, VK-RMD changes the downstream sensing interface from RGB-centered fusion to VK/non-RGB missing-modality expert learning. The key difference is not only the replacement of RGB by VK, but also the introduction of per-modality expert prediction, reliability-weighted decision fusion, and full-to-subset guidance for arbitrary modality availability.

中文解释：

> VK-RMD 不是只把 X-Fi 的 RGB branch 换成 VK branch，而是把模型从“统一融合后单头预测”改成“每个模态都能形成可部署 expert，再由 reliability proxy 动态组合”的缺失模态学习框架。

### 6.2 审稿风险 2：Depth 结果过强

Depth-only 约 53 mm 显著优于 X-Fi 101.8 mm，也优于近期 missing-modality 方法中常见的 Depth 报告值。这是亮点，也是红旗。

安全写法：

> We do not attribute the Depth-only gain to VK. Instead, the result indicates that subset-conditioned expert supervision can substantially strengthen standalone geometric sensing. We therefore report cross-subject and cross-scene results to avoid relying solely on the random protocol.

中文解释：

> 不要把 Depth 的强结果解释成 VK 的功劳，而要解释成缺失模态训练和单模态 expert 强监督释放了 Depth 的几何感知能力。

### 6.3 审稿风险 3：fusion 是否全面优于 X-Fi fusion

不要直接说“我们的 fusion 全面优于 X-Fi fusion”。因为 X-Fi 的 X-Fusion 是一个完整 cross-attention 机制，VK-RMD 的 reliability fusion 是另一个目标下的机制。

安全写法：

> X-Fi focuses on modality-invariant fusion, whereas VK-RMD focuses on subset-conditioned expert learning. The two fusion designs solve related but different problems.

中文解释：

> X-Fi 追求统一融合；VK-RMD 追求每个可用模态子集都能稳定预测。二者问题设定不同。

### 6.4 审稿风险 4：privacy claim

X-Fi 使用 raw RGB；VK-RMD downstream 不使用 raw RGB。但 VK 仍可能泄露体型、步态和动作习惯。

安全写法：

> VK-RMD reduces visual exposure by removing raw RGB from downstream inference. It does not claim formal privacy protection or anonymity.

---

## 7. 可以写入论文的凝练版本

英文安全段落：

> X-Fi maps RGB, Depth, LiDAR, mmWave and WiFi-CSI features into a common token sequence and applies X-Fusion to obtain a modality-invariant pose representation. VK-RMD differs in both sensing interface and training objective. It replaces the raw RGB branch with a skeleton-aware VK encoder, keeps non-RGB sensor backbones as feature extractors, and introduces per-modality pose experts with learned reliability proxies. During student training, each batch is optimized under a sampled modality subset, while a full-modality teacher provides full-to-subset guidance. Therefore, the strong standalone Depth performance should be interpreted as the effect of subset-conditioned expert supervision on a geometry-rich modality, rather than as evidence that VK alone improves Depth sensing.

中文安全段落：

> X-Fi 将 RGB、Depth、LiDAR、mmWave 和 WiFi-CSI 特征映射为统一 token 序列，并通过 X-Fusion 获得模态不变的姿态表示。VK-RMD 的差异不仅在于将 raw RGB branch 替换为 skeleton-aware VK encoder，更在于训练目标和融合方式的改变：非 RGB backbone 被保留为特征抽取器，每个模态 chunk 都产生自己的 pose expert 和 learned reliability proxy；Student 训练时在随机模态子集下优化，并由 full-modality teacher 提供 full-to-subset guidance。因此，Depth 单模态的强结果应解释为 subset-conditioned expert supervision 对强几何模态的强化，而不是 VK 单独提升了 Depth。

---

## 8. 结论

从代码层面看，VK-RMD 与 X-Fi 的本质差异主要有三点：

1. **输入接口不同**  
   X-Fi 是 raw RGB-centered；VK-RMD 是 VK/non-RGB downstream RGB-free。

2. **融合范式不同**  
   X-Fi 是 feature-level cross-attention fusion；VK-RMD 是 per-modality expert prediction + reliability-weighted decision fusion。

3. **训练目标不同**  
   X-Fi 主要监督最终 output pose；VK-RMD 同时监督 final pose、teacher guidance 和每个 modality expert。

因此，VK-RMD 不是简单的 X-Fi 小改，但也不能把所有提升都归因于 VK 或 KD。最可信的主线是：

> VK-RMD reorganizes X-Fi-style multimodal sensing into a reduced-visual-exposure missing-modality expert framework, where each available sensor is explicitly trained to be deployable under subset conditions and then combined through a learned reliability proxy.
