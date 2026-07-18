# Fig. 1 VK-RMD 整体处理架构流程图重新设计说明

这张图不要画成公式图，也不要画成简单的“多个模块堆叠图”。它应该是一张完整的 **数据处理与模型推理流程架构图**，核心是让读者一眼看懂：

> RGB 只在训练阶段提供结构知识；RGB 被转换成 VK 后成为人体结构桥梁；VK、Depth、LiDAR、mmWave、WiFi-CSI 分别进入各自编码分支；模型根据当前可用模态选择输入；再通过 reliability-aware fusion 得到统一人体表征；最后同时服务 HPE 和 HAR。

---

## 图的核心标题

建议图标题：

**VK-RMD architecture: from RGB-derived VK bridge to missing-modality robust HPE/HAR**

或者中文理解为：

**VK-RMD 整体流程：从 RGB 结构知识到缺失模态鲁棒 HPE/HAR**

---

## 整体图布局

建议采用 **从左到右的横向流程图**，分成 6 个大区域：

```text
Training RGB source
        ↓
RGB-to-VK structural bridge
        ↓
Multimodal sensor branches
        ↓
Available-modality subset selection
        ↓
Reliability-aware fusion
        ↓
HPE / HAR outputs
```

图中不要放数学公式。所有模块都用短标签、箭头、图标和颜色表达。

---

## 区域 1：Training-only RGB Source

### 画什么

最左侧画一个 RGB 图像输入框，可以画成人体 RGB frame 的小图标。

上方或框内标注：

```text
RGB frame
training only
```

### 想表达什么

RGB 不是最终推理输入，只在训练阶段用于生成 VK 或提供结构监督。

### 图中必须避免

不要让 RGB 箭头直接连到 final fusion 或 HPE/HAR output。

否则审稿人会误解为：

> 你的模型推理时仍然使用 RGB。

### 推荐图中文字

```text
RGB is used only to obtain structural keypoints during training.
```

如果图太拥挤，可以简写为：

```text
RGB: training only
```

---

## 区域 2：RGB-to-VK Structural Bridge

### 画什么

从 RGB frame 引出箭头，连接到一个 VK skeleton 图。

VK skeleton 应画成 17 个关节点和骨架连接线。

模块名称建议：

```text
RGB-to-VK extraction
```

或者：

```text
VK structural bridge
```

### 想表达什么

RGB 被压缩成 VK。VK 去掉外观、纹理、背景、人脸和衣着，只保留人体结构。

这里 VK 不是普通视觉图像特征，而是连接 RGB 视觉知识与其他非 RGB 传感器的人体结构桥梁。

### 推荐图中文字

```text
Appearance-suppressed body structure
```

或者更短：

```text
Body structure bridge
```

### 视觉重点

VK 这一块应该比其他普通模态更突出，例如：

- 用绿色或青色高亮；
- 放在 RGB 和其他传感器之间；
- 标注 `structural bridge`；
- 不要和 Depth/LiDAR/mmWave/WiFi 平行地画成普通输入。

---

## 区域 3：Multimodal Sensor Branches

### 画什么

从 VK bridge 后面开始，画出所有可用模态分支：

```text
VK
Depth
LiDAR
mmWave
WiFi-CSI
```

每个模态分支建议包含两个小模块：

```text
modality input → modality encoder
```

然后所有分支都指向同一个区域：

```text
Body-latent representation
```

### 具体画法

可以这样画：

```text
VK skeleton        → VK encoder        ┐
Depth map          → Depth encoder     │
LiDAR point cloud  → LiDAR encoder     │
mmWave points      → mmWave encoder    │ → Body-latent representation
WiFi-CSI signal    → WiFi encoder      │
```

### 想表达什么

不同模态的数据形态不同，所以不在原始空间对齐：

- VK 是关节点；
- Depth 是深度图；
- LiDAR 是点云；
- mmWave 是雷达点；
- WiFi-CSI 是信道信号。

它们先经过各自 encoder，再进入统一的人体潜在表示空间。

### 图中不要写公式

不要在图里写：

```text
z_m = P_m(E_m(x_m)) + e_m
```

这会让图变成公式解释图，不适合整体架构图。

可以改成流程标签：

```text
Modality-specific encoding
```

和：

```text
Shared body-latent space
```

---

## 区域 4：Available-Modality Subset Selection

### 画什么

在 body-latent representation 前后画一个“模态可用性选择”模块。

模块名称建议：

```text
Available modality subset
```

或者：

```text
Missing-modality sampling / availability mask
```

### 视觉表达

画 5 个小 token：

```text
VK | D | L | R | W
```

其中部分 token 正常亮起，部分 token 变灰。

例如：

```text
VK  Depth  LiDAR  mmWave  WiFi
ON  ON     OFF    ON      OFF
```

这样读者能看出：

模型不是固定输入所有模态，而是根据当前可用模态工作。

### 训练和推理如何体现

可以在这个区域旁边放两个小标签：

```text
Training: randomly sampled subsets
Inference: whatever sensors are available
```

这句话非常重要。

它能说明：

- 训练时主动模拟缺失模态；
- 推理时接受真实可用模态组合。

---

## 区域 5：Reliability-Aware Fusion

### 画什么

所有可用模态的 body-latent 表征进入一个融合模块。

模块名称建议：

```text
Reliability-aware fusion
```

下面加一行小字：

```text
learned modality-contribution weights
```

### 想表达什么

模型不是简单平均所有模态，也不是固定拼接，而是根据当前可用模态组合动态分配权重。

### 视觉表达

可以画一个融合模块，里面有几个不同粗细的箭头：

```text
Depth  ━━━━━
VK     ━━━
mmWave ━━━━
LiDAR  ━━
WiFi   ━
```

箭头粗细表示权重大小。

### 必须避免

不要写成：

```text
true sensor reliability
```

应该写成：

```text
learned reliability proxy
```

或：

```text
modality contribution weights
```

因为我们的权重是任务学习出来的贡献权重，不是物理传感器标定可靠性。

---

## 区域 6：Task Outputs

### 画什么

融合后的 unified body representation 分成两个输出头：

```text
HPE head → 3D human pose
HAR head → action class
```

HPE 输出可以画 3D skeleton。

HAR 输出可以画 action label，例如：

```text
walking / sitting / waving
```

### 想表达什么

同一套 VK-RMD 缺失模态学习机制同时服务：

- 细粒度身体结构估计 HPE；
- 高层动作语义识别 HAR。

这不是两个分散任务，而是同一个 robust human perception framework 的两个验证任务。

---

## 推荐最终图形结构

可以直接按照下面这个结构画：

```text
┌──────────────────────┐
│ RGB frame            │
│ training only        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ RGB-to-VK extraction │
│ VK structural bridge │
│ body keypoints       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Multimodal sensor branches                           │
│                                                      │
│ VK skeleton       → VK encoder                       │
│ Depth map         → Depth encoder                    │
│ LiDAR point cloud → LiDAR encoder                    │
│ mmWave points     → mmWave encoder                   │
│ WiFi-CSI signal   → WiFi encoder                     │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Shared body-latent representation                    │
│ All available modalities are represented as          │
│ task-relevant body-structure features                │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Available-modality subset                            │
│ Training: random subset sampling                     │
│ Inference: use whatever sensors are available        │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Reliability-aware fusion                             │
│ learned modality-contribution weights                │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┬──────────────────────┐
│ HPE head             │ HAR head             │
│ 3D pose              │ action class          │
└──────────────────────┴──────────────────────┘
```

---

## 更适合放在图中的短标签

建议图里只放短标签，不放长句：

```text
RGB frame
training only

RGB-to-VK extraction
VK structural bridge

VK encoder
Depth encoder
LiDAR encoder
mmWave encoder
WiFi encoder

Shared body-latent representation

Available-modality subset
random during training
available during inference

Reliability-aware fusion
learned contribution weights

HPE: 3D pose
HAR: action class
```

---

## 图注推荐版本

可以直接放入论文图注：

```text
Overall architecture of VK-RMD. RGB is used only during training to provide VK-based structural knowledge. The RGB frame is converted into VK keypoints, which serve as an appearance-suppressed body-structure bridge. VK and non-RGB sensor streams are processed by modality-specific encoders and represented in a shared body-latent space. During training, random modality subsets are sampled to simulate missing sensors; during inference, the model accepts whatever modalities are available. A reliability-aware fusion module assigns learned contribution weights to the available modalities and produces task-specific outputs for HPE and HAR.
```

中文理解：

```text
VK-RMD 整体架构。RGB 只在训练阶段用于提供 VK 结构知识。RGB 帧先被转换为 VK 关键点，VK 作为去外观化的人体结构桥梁。随后 VK 与非 RGB 传感器分别经过各自编码器，被表示到共享 body-latent 空间。训练时随机采样模态子集以模拟传感器缺失，推理时模型使用当前实际可用模态。可靠性融合模块为可用模态分配学习得到的贡献权重，并输出 HPE 姿态和 HAR 动作识别结果。
```

---

## 这张图要回答的审稿问题

### Q1：RGB 到底有没有参与推理？

图中回答：

RGB 只在最左侧 training-only 区域出现，不进入最终 inference flow。

### Q2：VK 为什么不是普通第五模态？

图中回答：

VK 放在 RGB-to-structure bridge 位置，而不是简单和 Depth/LiDAR/mmWave/WiFi 平行排列。

### Q3：不同模态怎么对齐？

图中回答：

不做 raw-space 对齐，而是通过 modality-specific encoders 进入 shared body-latent representation。

### Q4：缺失模态怎么处理？

图中回答：

通过 available-modality subset 模块显示训练随机缺失、推理按实际可用模态输入。

### Q5：reliability fusion 是什么？

图中回答：

不是物理可靠性标定，而是 learned modality-contribution weights。

---

## 画图时最容易犯的错误

1. 把 RGB 直接连到 fusion 或 output。
2. 把 VK 和 Depth/LiDAR/mmWave/WiFi 画成完全平级，弱化 VK bridge。
3. 在图中塞公式，导致整体架构不直观。
4. 把 reliability 写成真实传感器可靠性。
5. 没有画 missing-modality subset，导致看不出本文处理缺失模态。
6. 只画 HPE，不画 HAR，削弱双任务统一验证。

