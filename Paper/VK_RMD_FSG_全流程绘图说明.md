# VK-RMD + FSG 全流程绘图说明

本文档用于绘制论文主框架图。目标不是生成实验图，而是给绘图人员一个完整、可执行的结构说明：应该画哪些模块、箭头如何连接、每个模块表达什么科学含义，以及哪些表述需要避免过度声称。

## 1. 图的核心结论

这张总图要让读者一眼看懂：

> VK-RMD 将 raw RGB 从下游推理链路中移除，用 VK 作为低外观人体结构接口；每个可用传感器模态被编码成 body-related tokens，并在缺失模态子集下形成自己的 pose/action expert；FSG 训练机制用完整模态 teacher 的输出、token、结构和融合行为指导随机缺失模态 student，使 student 能在任意可用模态组合下完成 HPE/HAR。

图中不要强调：

- VK 保证隐私；
- 每个 token 已被严格语义对齐；
- reliability weight 是真实物理传感器可靠性；
- FSG 是传统大模型压缩蒸馏。

图中应该强调：

- downstream RGB-free / reduced visual exposure；
- VK 是 structural interface；
- subset-conditioned missing-modality learning；
- per-modality expert prediction；
- learned reliability proxy；
- full-to-subset guidance。

## 2. 推荐图类型

推荐绘制为 **schematic-led composite**，即一个大流程图为主，少量局部放大框辅助解释。

建议分成 4 个视觉区域：

```text
Panel A: Input and VK structural interface
Panel B: VK-RMD subset forward / reliability fusion
Panel C: Full-to-Subset Guidance training loop
Panel D: Inference under arbitrary modality availability
```

如果版面只能放一张大图，可以将 A/B/C/D 融合成一个横向流程：

```text
Inputs -> Modality encoders -> Body-related tokens -> Subset fusion -> Output
                           ↘ Teacher full branch -> FSG losses -> Student update
```

## 3. Panel A：输入模态与 VK 结构接口

### 3.1 画面内容

左侧画多模态输入：

```text
VK: 17×2 keypoints
Depth: depth image
LiDAR: point cloud
mmWave: radar point cloud / sparse reflection
WiFi-CSI: CSI amplitude tensor
```

视觉上可以这样画：

- VK：17 个节点 + 骨架连线，不画人脸/衣服/背景；
- Depth：灰度人体深度轮廓；
- LiDAR：稀疏点云；
- mmWave：更稀疏的点云或雷达反射点；
- WiFi-CSI：矩阵热图或时频块。

### 3.2 VK 的表达方式

VK 不要画成普通第五个传感器就结束。建议给 VK 单独加一个标注框：

```text
VK structural interface
appearance-suppressed, joint-level body structure
```

可以画：

```text
offline / upstream RGB-to-VK or pose source
        ↓
17×2 keypoints
        ↓
downstream VK-RMD
```

这里要避免让读者误解为本文训练了 raw RGB teacher。安全标注是：

```text
Raw RGB is not used in downstream inference.
VK can be obtained by an upstream keypoint extractor or other pose source.
```

### 3.3 该 panel 的科学含义

这一块回答：

> 为什么 VK 不是 raw RGB？为什么它可以作为下游结构接口？

建议图注写法：

> VK compresses image-level visual information into joint-level body geometry and removes raw appearance from the downstream model. It is used as a structural interface rather than a formal privacy guarantee.

中文解释：

> VK 将图像级视觉信息压缩为关节级人体几何，下游模型不再处理 raw RGB。VK 是结构接口，不是匿名保证。

## 4. Panel B：VK-RMD subset forward 与 reliability fusion

这是图中最重要的主体流程。

### 4.1 编码流程

从每个输入模态画箭头到对应 encoder：

```text
VK -> SkeletonPromptEncoder
Depth -> Depth backbone + TokenProjector
LiDAR -> LiDAR backbone + TokenProjector
mmWave -> mmWave backbone + TokenProjector
WiFi-CSI -> CSI backbone + TokenProjector
```

每个 encoder 输出统一 token：

```text
Z_vk     : 32 × 512
Z_depth  : 32 × 512
Z_lidar  : 32 × 512
Z_mmwave : 32 × 512
Z_wifi   : 32 × 512
```

图里可以把每个 token block 画成 32 个小矩形条。

### 4.2 VK branch 的局部放大

VK 分支建议放一个小 inset：

```text
17×2 keypoints
-> coordinate lifting
-> joint identity embedding
-> COCO17 bone topology mixing
-> 32 skeleton-aware tokens
```

不要写“new GCN theory”。更稳的图中文字：

```text
Skeleton-aware VK tokenization
```

### 4.3 Modality embedding

每个 token block 进入 TransformerEncoder 前，加上模态身份标签：

```text
Z_m + e_m
```

图中可以画成每个 token block 旁边有一个小色块：

```text
e_vk, e_depth, e_lidar, e_mmwave, e_wifi
```

解释文字：

> A learnable modality embedding marks the source identity of each token block.

中文：

> 可学习的 modality embedding 给每组 token 标注来源身份。

### 4.4 Token interaction

将当前可用模态 tokens 拼接后输入 TransformerEncoder：

```text
[selected tokens + modality embeddings]
-> TransformerEncoder
-> encoded subset tokens
```

图中建议画成：

```text
Concatenate selected modality tokens
        ↓
Modality-token Transformer
```

注意：这里不要画成 X-Fi 的 Q-K-V cross-attention 结构。我们的机制不是：

```text
shared Q + modality-specific K/V
```

而是：

```text
standard self-attention over selected modality tokens
```

图中可用注释：

```text
Self-attention over selected modality tokens
not X-Fi-style shared-Q / modality-KV fusion
```

### 4.5 Chunk split

Transformer 输出后，按每个模态 32 tokens 切回 chunk：

```text
Encoded tokens
-> VK chunk
-> Depth chunk
-> LiDAR chunk
-> ...
```

如果当前 subset 是 `VK+Depth+LiDAR`，图中可以画：

```text
96 encoded tokens
-> 32 VK tokens
-> 32 Depth tokens
-> 32 LiDAR tokens
```

### 4.6 Per-modality expert

每个 chunk 进入同一个 pose expert head，产生：

```text
pose_m
logvar_m
```

图中建议画成多个并行分支：

```text
VK chunk     -> Pose expert -> pose_vk     + logvar_vk
Depth chunk  -> Pose expert -> pose_depth  + logvar_depth
LiDAR chunk  -> Pose expert -> pose_lidar  + logvar_lidar
```

虽然代码中 `PoseExpertHead` 是共享参数，但图上可以写：

```text
shared pose expert applied to each modality chunk
```

避免误解为每个模态都有完全独立的一套参数。

### 4.7 Learned reliability proxy

由 logvar 得到模态权重：

```text
alpha_m = softmax(-logvar_m / tau)
```

图中画：

```text
logvar_vk, logvar_depth, logvar_lidar
        ↓
softmax(-logvar)
        ↓
alpha_vk, alpha_depth, alpha_lidar
```

标注必须谨慎：

```text
learned reliability proxy
```

不要写：

```text
true sensor reliability
```

### 4.8 fused_pose

各模态 expert pose 按 alpha 加权：

```text
fused_pose =
  alpha_vk * pose_vk
+ alpha_depth * pose_depth
+ alpha_lidar * pose_lidar
```

图中可以画成“专家投票”：

```text
pose_vk
pose_depth    -> reliability-weighted pose fusion -> fused_pose
pose_lidar
```

### 4.9 residual_pose

这是最容易画错的部分。建议单独画一条下方支路。

每个 chunk 先 mean pooling：

```text
VK chunk    -> mean pooling -> pooled_vk
Depth chunk -> mean pooling -> pooled_depth
LiDAR chunk -> mean pooling -> pooled_lidar
```

再用同一组 alpha 融合 token：

```text
fused_token =
  alpha_vk * pooled_vk
+ alpha_depth * pooled_depth
+ alpha_lidar * pooled_lidar
```

然后：

```text
fused_token -> LayerNorm + Linear -> residual_pose
```

图中可以画成：

```text
modality chunks
-> mean pooling
-> reliability-weighted token summary
-> residual pose head
-> residual_pose
```

解释文字：

> residual_pose is a supplementary prediction from the reliability-fused latent token, not a ground-truth residual.

中文：

> residual_pose 是从融合后的 latent token 再预测出的补充姿态，不是 `GT - fused_pose` 的真实残差。

### 4.10 final pose

最终：

```text
final_pose = 0.5 × fused_pose + 0.5 × residual_pose
```

图中画成两个箭头汇合：

```text
fused_pose      \
                 -> final HPE pose
residual_pose   /
```

HAR 分支可以简化画：

```text
replace pose experts with classification experts
output action logits
```

如果图空间有限，主图以 HPE 为例，旁边小标签写：

```text
Same subset-fusion principle is used for HAR with classification heads.
```

## 5. Panel C：FSG 训练机制

Panel C 要画清楚：

```text
Teacher 是 full-modality branch
Student 是 subset-modality branch
FSG 是训练连接，不是 inference 模块
```

### 5.1 Teacher branch

上方画 teacher：

```text
Full modality set M
{VK, Depth, LiDAR, mmWave, WiFi-CSI}
        ↓
Teacher VK-RMD forward
        ↓
teacher_pose
teacher_token_summary
teacher_reliability_weights
teacher_structure / bone relation
```

标注：

```text
Full-modality teacher
frozen during student training
```

### 5.2 Student branch

下方画 student：

```text
Sampled subset S ⊂ M
example: {VK, Depth, LiDAR}
        ↓
Student VK-RMD forward
        ↓
student_pose
student_token_summary
student_reliability_weights
student_modality_poses
```

标注：

```text
Subset student
updated by FSG losses
```

### 5.3 FSG loss links

用虚线箭头从 teacher 输出连接到 student 输出：

```text
teacher_pose -> student_pose
teacher_token_summary -> student_token_summary
teacher_bone_structure -> student_bone_structure
teacher_reliability_weights restricted to S -> student_reliability_weights
```

同时用实线从 GT 连接 student：

```text
GT pose -> student final pose
GT pose -> each student modality pose
```

图中建议用颜色区分：

- 蓝色实线：GT supervision；
- 紫色虚线：FSG teacher guidance；
- 灰色箭头：forward path；
- 橙色权重箭头：reliability fusion。

### 5.4 FSG loss 的图中文字

可以在 Panel C 右侧放一个小公式块：

```text
L_student =
  L_gt
+ λ_out L_out
+ λ_tok L_tok
+ λ_bone L_bone
+ λ_rel L_rel
+ λ_unc L_unc
```

旁边写一句：

```text
Fixed loss weights; FSG regularizes full-to-subset behavior.
```

不要画成“自动学习 loss 权重”。当前权重是固定超参数。

### 5.5 token loss 维度说明

如果图里需要解释 token guidance，可以画一个小注释：

```text
Teacher: 5 modalities × 32 tokens -> mean -> [B, 512]
Student: |S| modalities × 32 tokens -> mean -> [B, 512]
Token-summary MSE
```

这能避免读者质疑：

> teacher 和 student 模态数不同，token 数不同，怎么计算 token loss？

图中要强调：

```text
global token-summary guidance
not token-to-token matching
```

### 5.6 reliability guidance 维度说明

Teacher 有完整模态权重：

```text
alpha_T over M
```

Student 只有子集权重：

```text
alpha_S over S
```

画法：

```text
alpha_T over [VK, Depth, LiDAR, mmWave, WiFi]
        ↓ restrict to selected S
        ↓ renormalize
alpha_T|S
        ↓ KL
alpha_S
```

图中文字：

```text
restrict and renormalize teacher weights to the selected subset
```

## 6. Panel D：任意模态缺失推理

这一 panel 展示测试阶段不需要 teacher，也不需要 raw RGB。

### 6.1 画面内容

画 3 个例子：

```text
Case 1: Full subset
VK + Depth + LiDAR + mmWave + WiFi

Case 2: Missing visual keypoints
Depth + LiDAR + mmWave + WiFi

Case 3: Severe missing modality
Depth only / mmWave only / VK+Depth
```

每个 case 都进入同一个 Student VK-RMD：

```text
Available subset S
-> Student VK-RMD
-> HPE pose / HAR label
```

标注：

```text
No teacher at inference
No raw RGB at downstream inference
```

### 6.2 图中应体现的结论

Panel D 不是展示性能数字的地方，而是展示部署逻辑：

> 训练时 student 见过随机子集条件，因此测试时可以接受任意非空模态组合。

图中不要写：

> all missing-modality cases solved

应写：

```text
supports arbitrary non-empty modality subsets
```

## 7. 推荐整体布局

### 7.1 横向主流程版

适合 IEEE 双栏跨栏大图。

```text
Left: Inputs and VK interface
Middle-left: Encoders and tokenization
Middle-right: Subset fusion with experts
Right: HPE/HAR outputs
Top branch: Full-modality teacher
Bottom branch: Subset student
Dashed vertical links: FSG losses
```

视觉层级：

```text
Hero: Student subset forward
Support: Teacher full branch and FSG loss links
Inset: VK skeleton-aware tokenization
Inset: reliability/residual fusion detail
```

### 7.2 上下训练/推理版

适合方法 section 细图。

```text
Top row: Training with FSG
  Full teacher branch
  Subset student branch
  FSG losses

Bottom row: Inference
  Available subset
  Student only
  HPE/HAR output
```

优点是能清楚区分：

```text
训练时：teacher + student + losses
推理时：student only
```

## 8. 图中术语建议

建议统一使用以下英文标签：

| 中文含义 | 图中英文标签 |
|---|---|
| VK 结构接口 | VK structural interface |
| 骨架感知 VK 编码 | Skeleton-aware VK tokenization |
| 模态身份嵌入 | Modality embedding |
| 模态 token 交互 | Modality-token interaction |
| 每模态专家 | Per-modality expert |
| 不确定性分数 | Log-variance score |
| 可靠性代理 | Learned reliability proxy |
| 决策级融合 | Reliability-weighted decision fusion |
| 补充姿态分支 | Residual pose branch |
| 完整到子集指导 | Full-to-Subset Guidance, FSG |
| 全局 token 摘要指导 | Global token-summary guidance |
| 下游无 RGB 推理 | RGB-free downstream inference |

不建议使用：

```text
privacy guarantee
true sensor reliability
semantic proof
strict alignment
RGB teacher
```

## 9. 推荐图注草稿

英文图注：

> Overview of VK-RMD with Full-to-Subset Guidance. Raw RGB is removed from the downstream perception pipeline and represented by visual keypoints as a structural interface. VK and non-RGB sensing modalities are encoded into modality-specific token blocks and marked by learnable modality embeddings. For each sampled modality subset, the student performs modality-token interaction, splits the encoded tokens into modality chunks, and predicts per-modality pose or action experts with log-variance scores. The scores define learned reliability proxies for decision-level fusion, while a residual branch predicts a supplementary output from the reliability-fused token summary. During training, a full-modality teacher provides output-, token-summary-, structure- and reliability-level guidance to the subset student, while ground truth supervises the final output and each available modality expert. At inference, only the student is used and can operate on arbitrary non-empty modality subsets without raw RGB input.

中文图注：

> VK-RMD 与 Full-to-Subset Guidance 的整体流程。下游感知链路不直接使用 raw RGB，而是以 VK 作为人体结构接口。VK 与非 RGB 传感器模态分别编码为模态 token block，并加入可学习的 modality embedding。对于每个随机采样的可用模态子集，student 先进行模态 token 交互，再按模态切分 encoded tokens，并为每个模态生成 pose/action expert 与 log-variance score。log-variance 形成 learned reliability proxy，用于决策级 expert 融合；同时 residual branch 从 reliability-fused token summary 预测补充输出。训练时，full-modality teacher 从输出、全局 token 摘要、结构和可靠性权重层面对 subset student 进行指导，GT 同时监督最终输出和每个可用模态 expert。推理时只使用 student，支持任意非空模态子集，并且下游不输入 raw RGB。

## 10. 绘图时必须避免的误读

1. **不要把 VK 画成严格隐私保护模块。**  
   它是 reduced visual exposure / appearance-suppressed structural interface。

2. **不要把 reliability proxy 画成真实传感器质量检测器。**  
   它是 task-learned contribution proxy。

3. **不要把 FSG 画成大模型压缩小模型。**  
   它是 full-modality behavior regularization for subset student。

4. **不要把 token guidance 画成一一 token 对齐。**  
   它是 mean-pooled global token-summary matching。

5. **不要把我们画成复用了 X-Fi 的 Q-K-V fusion。**  
   我们使用标准 TransformerEncoder 做 selected token interaction，然后通过 per-modality expert + reliability fusion 输出。

## 11. 一句话给绘图人员

这张图要画出的不是“多模态特征拼接”，而是：

> 一个下游无 RGB 的 VK/non-RGB 缺失模态学习系统：每个可用模态先成为自己的 expert，再由 reliability proxy 融合；训练时 full teacher 教 subset student，推理时 student 独立处理任意模态组合。

