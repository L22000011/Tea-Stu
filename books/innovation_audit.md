# VK-RMD 创新与证据审计

## 1. 一句话客观结论

**VK-RMD 已形成一条可投稿的“低外观结构接口 + 完整观测到随机子集约束 + 子集条件化证据融合”工程算法链，但三个组成方向均有明确前作，项目独有性主要来自它们在同一 HPE/HAR 缺失传感器问题中的具体机制关系和验证闭环；目前属于中等方法创新、较强实验系统性，尚不足以稳妥支撑中科院一区。**

## 2. 代码与实验审查范围

本审计先建立了 `E:/Deskbook/Tea` 下 8,762 个文件的递归索引，再重点审查：

- 正式模型：`HPE/models`、`HAR/models`；
- 损失与训练：`HPE/losses`、`HAR/losses`、两个任务的 `training/engine.py`；
- 数据与子集：两个任务的 configs、`utils/modality.py` 和 dataloader/split 代码；
- 正式结果：`HPE/outputs`、`HAR/outputs`、`outputs/tables`、`tables`、`supplement`；
- 探索结果：`Super-Teacher`、Depth/mmWave-generated VK、corruption、body-latent 与 privacy probe；
- 写作材料：`故事线.md`、`Paper/EAAI.tex`、`Paper/IoTJ_VK_RMD*.tex`、现有图表与报告。

审查发现旧《故事线.md》描述的“RGB-VK Super-Teacher → 非RGB Student”与正式代码不一致。正式 HPE/HAR teacher 和 student 的模态集合分别在配置中均为 VK + 非RGB 模态；训练代码没有调用 raw RGB backbone。`utils/modality.py` 将字符串 `rgb` 映射为 `vk` 只是兼容别名，容易误导写作，但不等于模型实际读取 RGB。下文以正式代码和结果为准，SuperTeacher 仅视为探索。

## 3. 三个最大创新

### 创新一：VK结构化低外观接口与异构人体状态token化

**代码证据。** `SkeletonPromptEncoder` 不是把17点直接展平：`HPE/models/skeleton_prompt_encoder.py:11-31` 用 COCO17 骨邻接混合关节 token，`:51-67` 叠加坐标、中心化位置和 joint identity embedding，`:68-74` 投影为32个 token。`HPE/models/vk_rcd.py:58-90` 将它与冻结的 Depth、LiDAR、mmWave、WiFi backbones及 projectors接入同一 512维 token encoder；HAR使用同构实现。

**共同部分。** 骨架图建模、模态专属编码器、线性投影和 Transformer 均已有。X-Fi 已研究可变模态人体感知；STPrivacy 已说明“动作效用与隐私信息抑制”需要显式评价；MM-Fi 提供同步异构人体感知数据。

**项目独有部分。** 当前项目把关节身份与骨拓扑编码的 VK 作为可插拔的低外观结构接口，同时保留 Student-VK 与 Student-NV 两种部署模式，并在相同 HPE/HAR 模型族中覆盖全部子集。这里的独有性是接口定义与任务闭环，不是“首次使用骨架”。

**类型与强度。** 问题设定创新 + 系统组合创新，**中**。

**实验依据。** 身份探针中 defaced RGB、VK 的 subject-ID accuracy 分别为 67.53% 和15.43%，但VK仍高于2.5% chance。配对边际分析控制同一非VK子集后，加入VK使HPE 15/15、HAR 7/7配对均改善。Student-NV仍能工作，说明VK是有价值的结构证据而不是唯一中心。

**安全表述。** “VK-RMD将17点人体结构编码为低外观结构接口，并与异构传感器共同进行任务监督的人体状态token化。”

**审稿风险。** 不能写“严格body-latent对齐”：`tables/body_latent_alignment_from_logs.csv` 中 same/shuffled cosine gap仅0.0001–0.001，检索top-1接近随机。也不能写“隐私保护”：Depth、LiDAR、VK身份探针仍为43.80%、36.57%、15.43%。

**最近的三篇正式工作。**

1. [X-Fi, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)：同样处理动态多模态人体感知；区别是本项目显式使用骨拓扑VK接口、full-to-subset约束和不确定性双读出。
2. [STPrivacy, ICCV 2023](https://doi.org/10.1109/ICCV51070.2023.00471)：显式处理动作识别的隐私—效用；区别是其重点是隐私表征学习，本项目只降低在线外观暴露且处理异构传感器缺失。
3. [MM-Fi, NeurIPS 2023](https://doi.org/10.52202/075280-0822)：提供同步多模态HPE/HAR基准；本项目是在该同步任务空间上的学习框架，不应把数据集贡献据为方法创新。

### 创新二：完整观测到随机子集的多层行为约束（FSG）

**代码证据。** `HPE/training/engine.py:400-409` 与 `HAR/training/engine.py:348-357` 每个batch先随机采样student可见子集，同时让teacher读取固定完整模态。HPE损失在 `HPE/losses/distill_losses.py:40-91` 组合 GT、输出、token、骨长、受限并重归一的teacher融合权重与不确定性项；HAR在 `HAR/losses/distill_losses.py:47-96` 使用CE、温度KL、token、受限权重KL和专家不确定性。

**共同部分。** 完整模态teacher指导不完整student、modality dropout和多层蒸馏均已有，FSG不能被描述成全新的蒸馏类别。

**项目独有部分。** teacher/student参数规模相近，差异是观测条件；teacher权重被截取到student可见模态后重新归一；HPE与HAR分别注入几何和分类约束。这比普通 logits KD 更具体，但属于任务化机制组合。

**类型与强度。** 训练范式创新 + 机制创新，**弱到中**。

**实验依据。** HAR full为85.15%，no-KD为82.86%，支持辅助收益；但HPE no-KD 72.68 mm优于full 75.18 mm，否定“FSG稳定提升两任务”的强命题。

**安全表述。** “本文将已有完整到不完整蒸馏思想具体化为面向传感器可用子集的多层行为约束；其主要作用是正则化随机子集学习，而非压缩模型。”

**审稿风险。** 名称容易被认为给 modality dropout + KD 换名；没有 matched MMANet 等外部adapted baseline；HPE消融不支持其成为标题级贡献。

**最近的三篇正式工作。**

1. [MMANet, CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.01919)：不完整多模态学习中的margin-aware distillation与modality-aware regularization，是FSG最直接前作。
2. [Towards Good Practices for Missing Modality Robust Action Recognition, AAAI 2023](https://doi.org/10.1609/aaai.v37i3.25378)：系统研究缺失模态动作识别训练实践；区别是本项目覆盖HPE并加入任务结构/融合行为约束。
3. [Unbiased Missing-Modality Multimodal Learning, ICCV 2025](https://doi.org/10.1109/ICCV51701.2025.02272)：处理缺失模态偏置；区别是其理论目标更一般，本项目针对完整观测到传感器子集的行为迁移。

### 创新三：子集条件化的chunk证据保留与不确定性双读出

**代码证据。** `HPE/models/reliability_fusion.py:76-109` 将Transformer输出按32-token模态chunk切分，共享专家头输出pose和log-variance，以 `softmax(-logvar/tau)` 融合模态级pose；同一权重再融合pool后的token并产生全局pose，最终两种预测0.5/0.5组合。HAR在 `HAR/models/reliability_fusion.py:74-106` 对logits执行同构机制。

**共同部分。** Transformer融合、per-modality experts、uncertainty weighting和局部/全局辅助读出均有前作。

**项目独有部分。** 同一alpha耦合“专家预测融合”和“融合token全局读出”，只处理当前可见chunk且不生成缺失代理token；同一计算关系同时用于HPE回归和HAR分类。

**类型与强度。** 机制创新 + 系统组合创新，**中**。

**实验依据。** uniform fusion使HPE由75.18恶化至82.77 mm、HAR由85.15%降至67.46%，说明自适应融合重要。但corruption中HPE Depth受扰后其权重反而上升，证明alpha只是learned contribution proxy。expert compensation也显示融合不总优于最佳单专家。

**安全表述。** “本文提出面向动态可见子集的chunk-preserving dual-readout fusion，以共享任务不确定性代理联合组织模态级和子集级证据。”

**审稿风险。** 双分支与专家并非此前无人使用；`residual_pose/logits` 实际是第二个完整预测后等权平均，并不是严格的残差校正，论文宜称 global readout 而不是 residual branch。尚缺 chunk-only/global-only/dual 的直接消融，因此不能把双读出收益与uncertainty weighting收益分离。

**最近的三篇正式工作。**

1. [X-Fi, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)：跨模态Transformer与模态信息注入是最近邻；本项目区别在chunk专家预测、任务不确定性与双读出的耦合。
2. [ShaSpec, CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.01524)：共享/专属特征处理缺失模态；区别是本项目保留可观测chunk并输出显式模态任务证据。
3. [FuseMoE, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d62a85ebfed2f680eb5544beae93191-Paper-Conference.pdf)：缺失模态下使用专家和路由；区别是本项目不路由到多个网络专家，而以共享头和log-variance对可见chunk进行双层读出。

## 4. 与已发表工作的差异

三个创新都不是“零前作”。真正差异可压缩为：X-Fi解决可变模态融合，但没有当前代码中的完整到随机子集多层行为约束与任务不确定性双读出；MMANet等解决不完整多模态蒸馏，但不围绕同步人体传感器、VK低外观接口及HPE/HAR全子集建立闭环；隐私动作识别工作更直接优化隐私表征，而本项目只做可测的身份泄露降低和raw-RGB-free在线推理。因此，本文应定位为**具有具体机制增量的应用驱动算法框架**，不是新理论或通用基础模型。

## 5. 唯一核心科学问题

**中文：** 在在线推理不依赖原始RGB且传感器可能任意缺失时，如何利用低外观人体结构接口组织异构传感器证据，并使随机可见子集尽可能保持完整观测下的HPE/HAR行为？

**English:** How can a low-appearance human-structure interface organize heterogeneous sensor evidence and preserve full-observation HPE/HAR behavior when raw RGB is unavailable online and arbitrary sensor subsets may be missing?

## 6. 主线逻辑及三个创新的关系

1. **问题：** raw RGB身份暴露高，非视觉传感器又异构且会缺失。
2. **矛盾：** 压缩视觉外观会损失结构线索；只训练完整模态无法覆盖动态可用性；简单平均会让弱/坏模态支配结果。
3. **表示层（创新一）：** 将VK编码为低外观结构接口，并把各传感器投影为任务监督的人体状态token。这里不是证明所有模态严格对齐到VK坐标。
4. **训练层（创新二）：** 完整观测teacher提供行为参照，student在随机子集下接受任务特定的多层约束；FSG是辅助正则，不是两任务一致增益来源。
5. **决策层（创新三）：** 保留每个可见chunk的任务证据，再以learned uncertainty proxy联合形成模态级和子集级预测。
6. **验证：** HPE/HAR、Student-VK/NV、31/15全组合、missing severity、leave-one-out、random/cross-subject/cross-scene、隐私探针和消融共同覆盖效用、缺失与边界。

三项可以统一，但应把 **FSG从标题级核心降为训练机制**。若论文篇幅必须再收缩，删除SuperTeacher、严格body-latent alignment和生成VK故事；它们不是这条链成立的必要条件。

## 7. 已完成工作取舍表

完整逐项清单见 `work_relevance_inventory.csv`。正文优先级如下：

| 层级 | 内容 |
|---|---|
| 主线核心 | VK配对边际贡献；身份泄露探针；Student-VK/NV；HPE 31/HAR 15组合；缺失强度；leave-one-out；uniform fusion；random/cross-subject/cross-scene |
| 支撑证据 | reliability heatmap；HAR FSG/no-KD；复杂度；official X-Fi与近期共同非视觉子集比较 |
| 补充材料 | VK semantic perturbation；expert compensation；KD variants端点；Depth/mmWave-generated VK可行性 |
| 删除主线 | SuperTeacher；严格body-latent alignment图；Ori-XFi 244.91 mm主比较；极端encoder崩溃表 |
| 谨慎使用 | HPE FSG；corruption weights；generated VK；cross-scene HPE；Depth/LiDAR隐私泄露 |

## 8. 两个最终 baseline 及公平协议

### 8.1 Adapted X-Fi-VK

这是最接近的多模态人体感知baseline。使用相同VK、相同冻结backbones、相同32×512 tokens、同一逐样本split、同一subset sampler、epoch/batch/optimizer和全部组合评估；仅保留X-Fi融合器，移除FSG和当前双读出。标记为 **adapted reproduction**，与 official RGB-X-Fi 分开。

### 8.2 Adapted MMANet

这是最接近的完整到不完整蒸馏baseline。接入相同输入、backbones、teacher/student容量、随机子集序列、训练预算和cross协议，保留MMANet原始margin-aware distillation/modality-aware regularization。若HPE回归无法忠实定义其损失，只在HAR使用，不能硬造HPE比较。

详细计划见 `baseline_fairness_plan.md`。现有PTA/COMPASS表只达到同published protocol，不具备对方逐样本manifest，故只作补充。

## 9. 创新度与一区潜力评分

| 维度 | 分数 | 判断 |
|---|---:|---|
| 科学问题价值 | 8.0/10 | 在线低外观与动态传感器缺失具有明确工程意义 |
| 方法创新 | 6.0/10 | 有具体机制关系，但基础思想均有前作 |
| 独有性 | 5.5/10 | 独有在整条计算/训练链，不在单模块 |
| 实验完整性 | 8.0/10 | 双任务、全子集、跨协议和多种诊断较完整，但单seed与公平外部baseline不足 |
| 与前人区分度 | 6.0/10 | 与X-Fi/MMANet可区分，但尚缺matched adapted结果证明差异来源 |
| 中科院一区潜力 | **中偏弱** | 有送审基础，但目前更像强工程增量而非一区稳稿 |

**当前最可能档次：** EAAI / Sensors Journal / Information Fusion下游工程方向中的中等偏强稿件；IoT Journal可尝试但风险较高；KBS若突出不完整多模态学习机制并补强baseline，匹配度可能高于纯IoT包装。

**达到一区仍缺少的最少改动：**

1. 完成一个 matched Adapted X-Fi-VK baseline；
2. 完成一个 matched MMANet baseline，至少random split，最好增加cross-scene；
3. 补 `chunk-only / global-only / dual` 消融，分离双读出与uncertainty weighting的贡献；
4. 用3个seed只重复核心random结果或关键消融，而不是重跑全部31/15组合。

## 10. 最大拒稿风险

1. **创新边界不清：** FSG和双读出若写成首次提出，容易被MMANet、X-Fi、ShaSpec和专家融合文献击穿。
2. **公平性：** official RGB-X-Fi与VK-RMD不是完全公平训练对照；异常弱的244.91 mm复现不能进主结论。
3. **隐私过度承诺：** 当前只是轻量subject probe和raw-RGB-free在线推理，Depth/LiDAR/VK仍泄露身份。
4. **机制证据不完全：** uniform消融支持adaptive fusion，但不单独证明双读出；body-latent diagnostic不支持严格对齐；reliability proxy不校准。
5. **FSG不稳定：** HPE no-KD更好，使“完整到子集指导是核心性能来源”不能成立。
6. **场景泛化：** HPE cross-scene平均142.02 mm，部署表述必须限定为已校准或固定布设环境。

## 11. 最少修改方案

1. 把标题和摘要中心放在 `low-appearance structural interface + arbitrary sensor subsets`，不把distillation放标题。
2. 方法中把“body-latent alignment”改为“task-supervised human-state tokenization”；把“residual branch”改为“global fused-token readout”。
3. 主实验保留 VK paired marginal、31/15组合、Student-NV、uniform、cross协议和privacy probe。
4. FSG仅作为第二层训练机制，公开HPE task-dependent结果；SuperTeacher、generated VK和alignment diagnostic移至补充或删除。
5. 补两个adapted baselines和一个三分支读出消融，这是最小的拒稿风险修复组合。

## 12. 最终推荐的标题和三条贡献

### 标题

**VK-RMD: Low-Appearance Structural Tokenization for Missing-Modality Multimodal Human Perception**

中文：**VK-RMD：面向缺失模态多模态人体感知的低外观结构token化方法**

### 三条贡献

1. 提出一种面向在线raw-RGB-free人体感知的低外观结构接口，将带有关节身份和骨拓扑的VK与异构非视觉传感器统一组织为任务监督的人体状态tokens；身份探针和配对边际分析分别量化其残余泄露与条件效用。
2. 将完整观测到不完整观测的蒸馏具体化为传感器子集条件下的多层行为约束，使完整模态teacher从输出、token、几何/分类结构和融合行为层面正则化随机子集student；该机制被明确限定为任务相关辅助监督。
3. 设计保留可见模态chunk证据的不确定性双读出机制，并在HPE/HAR、Student-VK/NV、全部31/15组合及random/cross-subject/cross-scene协议下系统验证效用、缺失鲁棒性和适用边界。

---

**总评：** 这不是三个互不相关的模块，也不是强理论原创。它是一条可以成立的应用驱动算法链。能否从“较完整的工程组合”上升为“一区可竞争的方法”，关键不再是继续增加生成VK或可视化，而是用两个严格matched baselines和双读出直接消融证明：提升确实来自项目独有的机制关系。
