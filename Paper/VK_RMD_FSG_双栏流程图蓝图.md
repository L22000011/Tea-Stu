# VK-RMD + FSG 双栏流程图蓝图

目标：画一张**左右双栏架构图**。  
左栏展示 **VK-RMD 前向：多模态输入 -> token 对齐 -> expert 预测 -> reliability 加权 -> HPE/HAR 输出**。  
右栏展示 **FSG 训练：Full Teacher 与 Subset Student 的 loss 对齐关系**。

整体风格参考：

- 少文字；
- 强流程箭头；
- 模块块状化；
- token 用小矩形堆叠；
- teacher/student 用上下双分支；
- loss 用虚线或彩色连线；
- VK 单独特殊处理，其他传感器并排处理。

---

## 1. 总体版式

```text
┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│ Left: VK-RMD Forward / Fusion                 │  │ Right: Full-to-Subset Guidance, FSG           │
│                                              │  │                                              │
│ Inputs -> Encoders -> Tokens -> Fusion -> y  │  │ Full Teacher                                  │
│                                              │  │      ↓                                        │
│ VK special branch                            │  │ Loss links                                    │
│ Non-RGB parallel branches                     │  │      ↓                                        │
│ Reliability weighted experts                 │  │ Subset Student                                │
└──────────────────────────────────────────────┘  └──────────────────────────────────────────────┘
```

建议比例：

```text
Left 60%
Right 40%
```

左边是模型机制主图；右边是训练监督主图。

---

## 2. 左栏：VK-RMD Forward / Fusion

### 2.1 左栏总流程

```text
Input modalities
      ↓
Modality encoders
      ↓
32×512 modality tokens
      ↓
+ modality embedding
      ↓
TransformerEncoder
      ↓
Split by modality chunks
      ↓
Per-modality expert
      ↓
pose/logits + logvar
      ↓
Reliability weights
      ↓
Final HPE / HAR output
```

---

## 3. 左栏输入区

### 3.1 VK 特殊分支

VK 单独放在最上方或左上角，视觉上和其他模态区分开。

```text
VK
17×2 keypoints
      ↓
Skeleton-aware VK Encoder
      ↓
VK tokens
32×512
```

VK encoder 内部只保留 3 个核心标签：

```text
Coord lifting
Joint ID
Bone topology
```

图形元素：

```text
17 个点 + 骨架线
    ↓
小模块：Skeleton-aware VK Encoder
    ↓
绿色 token block: Z_vk
```

推荐文字：

```text
VK structural interface
```

不要写：

```text
privacy guarantee
RGB teacher
```

### 3.2 其他模态并排分支

Depth、LiDAR、mmWave、WiFi-CSI 并排，结构一致。

```text
Depth image     -> Depth Encoder     -> Z_depth
LiDAR points    -> LiDAR Encoder     -> Z_lidar
mmWave points   -> mmWave Encoder    -> Z_mmwave
WiFi-CSI matrix -> CSI Encoder       -> Z_wifi
```

每个输出统一画成：

```text
32×512 tokens
```

图中可以统一标注：

```text
X-Fi-style sensor backbones + TokenProjector
```

但不要让这句话抢主图。可以放在底部小字。

---

## 4. 左栏 token 对齐区

### 4.1 Token blocks

把每个模态 token 画成一组小矩形：

```text
Z_vk       [□□□□ ... □] 32×512
Z_depth    [□□□□ ... □] 32×512
Z_lidar    [□□□□ ... □] 32×512
Z_mmwave   [□□□□ ... □] 32×512
Z_wifi     [□□□□ ... □] 32×512
```

### 4.2 Modality embedding

每个 token block 旁边加一个小标签：

```text
+ e_vk
+ e_depth
+ e_lidar
+ e_mmwave
+ e_wifi
```

推荐画法：

```text
Z_m + e_m
```

非常短即可。

### 4.3 Selected subset

用虚线框标出当前可用子集，例如：

```text
Selected subset S = {VK, Depth, LiDAR}
```

未选择模态用灰色虚线或淡化：

```text
mmWave, WiFi-CSI faded
```

---

## 5. 左栏 Transformer 区

### 5.1 拼接

```text
[Z_vk+e_vk | Z_depth+e_depth | Z_lidar+e_lidar]
              ↓
        TransformerEncoder
```

模块名：

```text
Modality-token interaction
```

### 5.2 注意不要画成 X-Fi 的 QKV

这里不要画：

```text
Q from shared feature
K/V from each modality
```

因为这不是我们的 fusion。

可以在图边角加小注释：

```text
Self-attention over selected tokens
```

---

## 6. 左栏 chunk 区

Transformer 输出后按模态切回。

```text
Encoded tokens
      ↓ split by 32-token blocks

H_vk
H_depth
H_lidar
```

画法：

```text
一条长 token bar
用竖线切成 3 段
每段标注 H_vk / H_depth / H_lidar
```

核心标签：

```text
Modality chunks
```

---

## 7. 左栏 Expert + Reliability 区

### 7.1 每个 chunk 进入 expert

```text
H_vk     -> Expert -> y_vk     + s_vk
H_depth  -> Expert -> y_depth  + s_depth
H_lidar  -> Expert -> y_lidar  + s_lidar
```

其中：

```text
y_m = pose or logits
s_m = logvar score
```

图上写：

```text
Per-modality expert
```

如果 HPE 图：

```text
y_m = pose_m
```

如果兼顾 HAR：

```text
y_m = pose/logits
```

### 7.2 logvar 到 alpha

```text
s_vk, s_depth, s_lidar
        ↓
softmax(-s)
        ↓
α_vk, α_depth, α_lidar
```

模块名：

```text
Learned reliability proxy
```

不要写：

```text
sensor reliability
```

### 7.3 expert decision fusion

```text
ŷ_exp =
  α_vk y_vk
+ α_depth y_depth
+ α_lidar y_lidar
```

图中可以写：

```text
Reliability-weighted expert fusion
```

---

## 8. 左栏 residual branch

这一块要画得简单，但必须有。

### 8.1 mean pooling

每个 chunk 下面拉一条支路：

```text
H_vk     -> mean -> h_vk
H_depth  -> mean -> h_depth
H_lidar  -> mean -> h_lidar
```

### 8.2 reliability-weighted token fusion

```text
h_S =
  α_vk h_vk
+ α_depth h_depth
+ α_lidar h_lidar
```

### 8.3 residual output

```text
h_S -> Linear head -> ŷ_res
```

图上写：

```text
Residual prediction branch
```

非常重要：不要画成误差残差。

可以用小字：

```text
supplementary prediction from fused token
```

### 8.4 final output

两条线合并：

```text
ŷ_exp  ─┐
        ├─ 0.5 / 0.5 -> ŷ
ŷ_res  ─┘
```

最终输出：

```text
HPE: 3D pose
HAR: action logits
```

---

## 9. 右栏：FSG Training

右栏画成上下双分支：

```text
Full Teacher
===================
Subset Student
```

### 9.1 Teacher 分支

Teacher 输入是完整模态。

```text
Full M
{VK, D, L, R, W}
      ↓
Teacher VK-RMD
      ↓
ŷ_T, z_T, α_T, b_T
```

图中模块：

```text
Full-modality Teacher
```

输出只保留四个核心：

```text
ŷ_T     output
z_T     token summary
α_T     reliability weights
b_T     bone / structure
```

### 9.2 Student 分支

Student 输入是随机子集。

```text
Subset S
example: {VK, D, L}
      ↓
Student VK-RMD
      ↓
ŷ_S, z_S, α_S, {ŷ_m}
```

模块名：

```text
Subset Student
```

输出：

```text
ŷ_S       final output
z_S       token summary
α_S       reliability weights
ŷ_m       modality expert outputs
```

---

## 10. 右栏 Loss 对齐关系

FSG 右栏重点画 loss 怎么比。

### 10.1 GT supervision

GT 单独从右侧或下方进入 Student。

```text
GT y
 ├── L_gt:      y vs ŷ_S
 └── L_unc:     y vs each ŷ_m
```

画成蓝色实线。

### 10.2 Output guidance

```text
ŷ_T ── L_out ── ŷ_S
```

画成紫色虚线。

### 10.3 Token-summary guidance

```text
z_T ── L_tok ── z_S
```

旁边小字：

```text
mean-pooled token summary
same dim: 512
```

不要画成 token-to-token 一一对齐。

### 10.4 Structure / bone guidance

```text
b_T ── L_bone ── b_S
```

如果图太拥挤，可以把它合并到 output guidance 旁边：

```text
Output / structure guidance
```

### 10.5 Reliability guidance

Teacher 的 α 是完整模态：

```text
α_T over M
```

Student 的 α 是子集：

```text
α_S over S
```

画一个小模块：

```text
α_T
 ↓ restrict to S
 ↓ renormalize
α_T|S ── L_rel ── α_S
```

这是右栏最重要的细节之一。

---

## 11. 右栏 Loss 汇总块

右下角放一个简洁公式块：

```text
L =
  L_gt
+ λ_out L_out
+ λ_tok L_tok
+ λ_bone L_bone
+ λ_rel L_rel
+ λ_unc L_unc
```

旁边写：

```text
fixed λ
```

不要写：

```text
learned loss weights
```

---

## 12. 训练与推理区别

图底部用一条横条表示：

```text
Training:
Full Teacher + Subset Student + FSG losses

Inference:
Student only + arbitrary non-empty subset S
```

推理时右边 teacher 分支可以用灰色虚线标注：

```text
Teacher not used at inference
```

---

## 13. 推荐颜色

```text
VK: green
Depth: blue
LiDAR: teal
mmWave: orange
WiFi-CSI: purple
Teacher: light red / pink
Student: light green / cyan
GT loss: blue solid line
FSG loss: purple dashed line
Reliability α: orange arrows
Unused modalities: gray faded
```

颜色不宜太多，token block 用浅色，loss 箭头用高饱和色。

---

## 14. 推荐最终图标题

```text
Figure 1. VK-RMD framework with Full-to-Subset Guidance.
```

或者：

```text
Figure 1. VK-centered missing-modality expert learning with FSG.
```

---

## 15. 图上保留的最少文字

左栏只保留：

```text
VK structural interface
Sensor backbones
TokenProjector
Modality embedding
TransformerEncoder
Modality chunks
Per-modality expert
Learned reliability proxy
Expert fusion
Residual branch
HPE pose / HAR logits
```

右栏只保留：

```text
Full Teacher
Subset Student
FSG losses
L_gt
L_out
L_tok
L_bone
L_rel
L_unc
restrict + renormalize
Student only at inference
```

---

## 16. 一句话绘图原则

> 左边画“数据如何变成 expert 并加权预测”，右边画“完整 teacher 如何通过 FSG loss 指导缺失 student”；VK 单独突出为结构接口，其他传感器并排处理，FSG 重点画 loss 对齐而不是重复模型结构。

