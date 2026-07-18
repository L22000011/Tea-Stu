# VK-RCD 三个最大创新与“已有工作”应对口径

日期：2026-05-28

## 一句话总定位

我们的创新不是“蒸馏本身”，而是提出一个面向真实部署的隐私友好、任意模态缺失、多任务人体感知框架：

> 在不使用原始 RGB 的前提下，将 VK 结构化视觉输入、全模态特权 Teacher、随机缺失 Student、非视觉部署和可靠性感知融合统一起来，并在 HPE/HAR、多协议、多模态组合下系统验证。

---

## 创新一：无原始 RGB 的隐私友好多模态人体感知

### 我们做了什么

官方 X-Fi 使用 RGB 作为视觉输入之一，而我们不使用原始 RGB 图像，改用 VK 关键点作为结构化视觉表示，并结合 Depth、LiDAR、mmWave、WiFi-CSI 等非 RGB 模态完成 HPE/HAR。

### 为什么是创新

这个创新点不在于“关键点编码器是一个 MLP”，而在于重新定义了输入约束：

- 去除原始 RGB 中的面部、衣着、背景、家庭环境等隐私信息。
- 保留人体结构信息，使模型仍能完成姿态估计和动作识别。
- 进一步支持 Student-NV，即完全不使用视觉输入，只用非视觉传感器部署。

### 是不是从没人做过？

不能说“从没人做过”。隐私友好人体感知、骨架动作识别、非视觉传感器感知都有人做过。

我们应该这样表述：

> 已有工作分别研究过骨架表示、非视觉传感器和隐私友好感知，但我们关注的是在 X-Fi/MM-Fi 这种多传感器人体感知框架下，系统性移除原始 RGB，并验证 VK 与非视觉输入在 HPE/HAR、任意模态组合和跨协议部署中的可用性。

### 如果有人说别人做过怎么办？

回答：

> 我们不声称“VK 或非视觉感知”本身首次提出。我们的贡献是把它放到多模态人体感知的真实部署问题中，形成无 RGB 输入约束下的完整训练、推理和评估框架。

---

## 创新二：面向任意模态缺失的 Teacher-Student 部署框架

### 我们做了什么

我们训练全模态 Teacher 学习完整传感器条件下的人体结构知识；再训练 Student-VK，在训练阶段随机缺失模态，使其适应任意可用模态组合；同时训练 Student-NV，验证无视觉输入时的部署能力。

### 为什么是创新

蒸馏不是创新本体，创新在于蒸馏服务的问题不同：

- 不是普通模型压缩。
- 不是简单 teacher-student 复现。
- 而是把全模态知识迁移到“缺失模态、无 RGB、可部署”的 Student 中。

我们的实验也不是只测全模态，而是测：

- HPE 31 种模态组合。
- HAR 15 种模态组合。
- Student-NV 非视觉组合。
- random、cross-scene、cross-subject。
- NoKD、Uniform、VK 噪声、关节缺失等消融。

### 是不是从没人做过？

不能说从没人做过。缺失模态学习和知识蒸馏已有 B 类及以上工作，例如：

- KDD 2020: *Multimodal Learning with Incomplete Modalities by Knowledge Distillation*。
- CVPR 2022: *Are Multimodal Transformers Robust to Missing Modality?*。
- CVPR 2023/2024 也有多模态缺失、跨模态蒸馏相关工作。

### 如果有人说蒸馏早就有了怎么办？

回答：

> 是的，蒸馏是已有技术，我们不把“蒸馏”本身作为创新。我们的创新是问题设置和框架组合：在无原始 RGB 的多传感器人体感知中，把全模态 Teacher 的知识迁移到任意模态缺失的 Student-VK 和非视觉 Student-NV，并用完整的 HPE/HAR 实验验证真实部署能力。

更短汇报版：

> 蒸馏是工具，不是卖点。卖点是“无 RGB + 任意模态缺失 + HPE/HAR 双任务部署”。

---

## 创新三：从单任务 Teacher 走向跨任务 Super-Teacher

### 我们要做什么

当前 HPE 和 HAR 主要是分别训练 Teacher。下一步 Super-Teacher 计划训练一个统一 Teacher，同时学习：

- HPE 的人体几何结构知识。
- HAR 的动作语义知识。
- 多模态共享人体表征。

然后这个 Super-Teacher 再分别蒸馏给：

- HPE Student-VK / Student-NV。
- HAR Student-VK / Student-NV。

### 实现原理

Super-Teacher 的核心结构：

1. 使用 Shared Multimodal Encoder 编码 VK、Depth、LiDAR、mmWave、WiFi-CSI。
2. 加入 modality embedding，显式标识不同模态。
3. 使用 Transformer fusion 学习统一人体 token 表征。
4. 接两个任务头：
   - HPE head 输出 3D pose。
   - HAR head 输出 action logits。
5. 使用联合损失训练：
   - HPE: pose loss、MPJPE、bone loss。
   - HAR: cross entropy。
   - reliability regularization。
6. Student 从 Super-Teacher 学习：
   - HPE pose knowledge。
   - HAR logit knowledge。
   - shared token knowledge。
   - reliability weight knowledge。

### 为什么是更强创新

Super-Teacher 可以把论文从“系统整合”提升到“跨任务人体表征学习”：

- 普通 Teacher：只服务一个任务。
- Super-Teacher：一个 Teacher 同时理解“人在哪里”和“人在做什么”。
- 普通蒸馏：单任务内迁移。
- Super-Teacher 蒸馏：跨任务共享人体结构与动作语义。

### 是不是从没人做过？

不能直接说从没人做过。多任务学习、共享编码器、跨任务蒸馏都有人研究过。

但我们可以这样说：

> 现有工作通常在通用视觉、图文、多媒体或单任务人体感知中研究多任务学习和蒸馏；我们的 Super-Teacher 聚焦 MM-Fi/X-Fi 多传感器人体感知场景，在无 RGB、任意模态缺失、HPE/HAR 双任务部署下学习统一人体表征。

### 如果有人说多任务 Teacher 也有人做过怎么办？

回答：

> 我们不声称多任务学习本身首次提出。我们的贡献是把多任务 Teacher 具体落到多传感器人体感知中，让同一个 Super-Teacher 同时学习 HPE 的几何结构和 HAR 的动作语义，并验证它能否提升缺失模态 Student 的跨场景、跨主体和非视觉部署性能。

---

## 最推荐的汇报说法

如果导师问“你到底创新在哪里”，直接这样回答：

> 老师，我们不把蒸馏本身作为创新，因为蒸馏和缺失模态学习确实已有很多工作。我们的创新有三个层次：第一，面向隐私保护，我们把 X-Fi 中的原始 RGB 替换为 VK 结构化视觉表示，并进一步验证无视觉 Student-NV；第二，面向真实部署，我们把全模态 Teacher 的知识迁移到任意模态缺失 Student，而不是只做全模态训练；第三，下一步我们做 Super-Teacher，让一个 Teacher 同时学习 HPE 的人体几何和 HAR 的动作语义，形成跨任务共享人体表征。也就是说，我们不是发明蒸馏，而是把蒸馏用于无 RGB、任意模态缺失、多任务人体感知这个具体且完整的部署问题。

---

## 不能说的话

不要说：

- “我们第一个做蒸馏。”
- “我们第一个做缺失模态。”
- “我们第一个做骨架替代 RGB。”
- “我们的方法完全没人做过。”

这些话很容易被反驳。

---

## 应该说的话

应该说：

- “蒸馏是工具，不是我们的唯一创新。”
- “我们的创新是问题设置、框架组合和系统验证。”
- “我们关注的是无原始 RGB、任意模态缺失、多传感器人体感知部署。”
- “已有工作做过蒸馏和缺失模态，但很少在 MM-Fi/X-Fi 多传感器 HPE/HAR 框架下做完整闭环。”
- “Super-Teacher 是下一步把方法从系统整合提升到跨任务统一人体表征学习的关键。”

---

## 投稿口径

当前 VK-RCD 的合理定位：

> 隐私友好缺失模态人体感知框架。

加入 Super-Teacher 后的更强定位：

> 面向隐私友好与任意模态缺失部署的跨任务多模态人体表征蒸馏框架。

这两个标题的差别在于：

- VK-RCD：强调系统完整性。
- Super-Teacher：强调方法创新性。

