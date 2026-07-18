# VK-RMD 现有创新的文献查重与全文主线定稿

> 定稿日期：2026-07-15  
> 范围：仅依据当前代码、现有实验结果及已发表文献形成贡献表述；不修改模型，不新增实验，不把未实现机制写入论文。

## 1. 一句话客观结论

截至本轮检索，骨架表示、完整模态教师指导缺失模态学生、随机模态丢弃、专家融合和不确定性加权均有前人基础，但尚未发现一篇正式论文同时实现 VK-RMD 的完整计算链：**以带关节身份和骨拓扑的 VK 作为 raw-RGB-free 在线结构接口，以任务相关的完整到子集结构—融合一致性约束训练随机模态子集 student，并只对当前可见 modality chunks 进行局部预测与全局 token 的耦合双层聚合，同时统一服务 HPE 与 HAR。** 因此，本项目可以突出“面向特定科学问题形成的新机制关系和任务化实现”，但不应声称首次提出骨架、蒸馏、Transformer、专家或不确定性融合。

## 2. 唯一核心科学问题

### 中文

> 在原始 RGB 不进入在线推理、且传感器可用子集随机变化的条件下，如何利用低外观人体结构接口组织异构传感器证据，并使有限模态子集尽可能保持完整观测下的 HPE/HAR 感知行为？

### English

> How can a low-appearance human-structure interface organize heterogeneous sensor evidence and enable a limited modality subset to preserve full-observation HPE/HAR behavior when raw RGB is excluded from online inference and sensor availability varies dynamically?

### 一句话论点

VK-RMD 通过低外观 VK 结构接口、完整到子集的结构—融合一致性约束，以及只聚合可见 modality chunks 的局部—全局双层读出，使单一模型在在线不处理原始 RGB 的条件下适应动态传感器子集；身份探针、VK 配对边际贡献、全组合缺失评估和机制消融支持其效用，而残余身份泄露、HPE 中 teacher guidance 的任务依赖性和 cross-scene 退化限定其适用边界。

## 3. 全文因果链

```text
原始 RGB 携带较强身份外观信息
  -> 以 VK 替代在线原始 RGB，保留显式关节结构
  -> Depth / LiDAR / mmWave / WiFi-CSI 与 VK 经专属编码器形成同维任务 token
  -> 真实部署中的随机传感器缺失造成 full observation 与 subset observation 行为差异
  -> FS-SFC 用完整观测 teacher 的输出、表示、结构及子集融合行为约束随机子集 student
  -> 不同子集会改变证据优势，因而保留可见 chunk 的局部任务判断，并形成全局 fused-token 判断
  -> learned contribution proxy 同时组织局部预测和全局 token 证据
  -> HPE/HAR、31/15 个组合、VK/NV、跨主体与跨场景实验验证收益和边界
```

这条主线中，三个创新不是并列插件：创新一规定在线模型处理什么样的人体信息；创新二规定完整观测知识如何约束随机子集；创新三规定当前子集中的证据如何被读取和聚合。

## 4. 术语与声明边界

| 推荐术语 | 准确定义 | 不应替换成 |
|---|---|---|
| raw-RGB-free online inference | 原始 RGB 不进入当前 teacher/student 在线感知主流程；VK 可由离线上游获得 | formal privacy protection |
| reduced visual exposure | 相比 defaced RGB，VK 的主体识别探针显著下降，但仍高于随机水平 | anonymous / identity-free |
| task-supervised human-state tokenization | 各模态被投影为同维 token，并由同步样本和共同 HPE/HAR 标签学习任务相关人体状态 | strict semantic alignment |
| Full-to-Subset Structural-Fusion Consistency, FS-SFC | 完整 teacher 对随机子集 student 的任务输出、表示、结构及子集融合行为约束体系 | 全新蒸馏类别 |
| learned contribution proxy | 由每个可见 chunk 的预测误差相关 log-variance 产生的任务贡献权重 | calibrated physical sensor reliability |
| visible-chunk dual evidence readout | 同一可见 chunk 同时参与模态级预测融合和全局 fused-token 预测 | independent parameter experts / missing-token reconstruction |

需要特别说明：当前实现只有 teacher 融合权重会按 student 可见子集筛选并重新归一化；teacher 的完整输出、token 和 HPE 骨结构目标仍被直接用于对齐。代码没有单独执行 teacher-to-student log-variance 蒸馏，student uncertainty 由真实任务标签监督，teacher uncertainty 仅通过其融合权重间接进入一致性约束。因此名称固定为 **FS-SFC**，不使用 `subset-projected`。

## 5. 三个最终创新

### 5.1 创新一：VK 驱动的低外观结构化人体感知接口

#### 论文级定义

本文不是把“使用骨架”本身作为创新，而是将 Visual Keypoints 重新组织为贯穿 full-observation teacher、random-subset student、HPE 和 HAR 的在线结构接口。VK 通过关节身份、二维位置和 COCO-17 骨拓扑编码人体结构，替代在线原始 RGB；其余异构传感器通过各自编码器和 projector 形成同维任务 token。这里的 VK 是一种结构显式的输入接口，不是 attention 中查询其他模态的 query，也不是保证其他模态已被严格对齐到骨架坐标系的“中心”。

#### 代码依据

- `HPE/models/skeleton_prompt_encoder.py:11-31`：`BoneGraphMixer` 依据 `COCO17_BONES` 聚合骨邻接信息。
- `HPE/models/skeleton_prompt_encoder.py:34-74`：`SkeletonPromptEncoder` 联合坐标 MLP、位置 MLP、joint embedding、骨图混合和 token projector，将 17 个关节变为 32 个结构 token。
- `HPE/models/vk_rcd.py:58-84`：VK 编码器、非 VK modality projectors、`ModalityTokenEncoder` 和 fusion 被接入同一 HPE 模型。
- `HAR/models/vk_rcd_har.py:60-84`：HAR 使用同一结构接口范式。
- `HPE/training/engine.py:369-409` 与 `HAR/training/engine.py:318-357`：teacher 和 student 均在当前 VK/non-RGB 模态集合上运行，主线不是 raw-RGB teacher 到 VK student。

#### 实验依据

- `privacy_leakage_probe/privacy_leakage_probe_subject_id_summary.csv`：defaced RGB 主体识别率为 67.53%，VK 为 15.43%，随机水平为 2.50%。这支持“降低视觉/身份暴露”，不支持匿名或隐私保证。
- `supplement/vk_paired_marginal_contribution/vk_paired_marginal_summary.csv`：在相同非 VK 子集上加入 VK，HPE 的 15/15 对 MPJPE 与 PA-MPJPE 均改善，平均分别改善 28.56 mm 和 16.70 mm；HAR 的 7/7 对 Accuracy 与 Macro-F1 均改善，平均分别提高 5.81 和 5.97 个百分点。
- `tables/main_hpe_results.csv` 与 `tables/main_har_results.csv`：Student-VK 和 Student-NV 的配套结果说明 VK 在严格非视觉系统之上提供条件性任务增益，而不是系统唯一可用证据。

#### 三篇最近邻工作与差异

1. [X-Fi, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)：同样研究多传感器人体感知和任意模态组合，也使用专属编码器与 Transformer；但原始体系包含 RGB，并以 X-Fusion 的跨注意力注入为核心，没有把带骨拓扑的 VK 设为 raw-RGB-free 在线结构接口，也没有本文的 full-to-subset 多层约束。
2. [SCoPLe, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Semantic-guided_Cross-Modal_Prompt_Learning_for_Skeleton-based_Zero-shot_Action_Recognition_CVPR_2025_paper.html)：使用 skeleton 并强调其紧凑、低外观特征，但目标是 skeleton-text prompt 的零样本动作识别，不处理多传感器任意缺失、HPE 或完整到子集行为约束。
3. [STPrivacy, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_STPrivacy_Spatio-Temporal_Privacy-Preserving_Action_Recognition_ICCV_2023_paper.html)：通过视频 token 稀疏化与对抗匿名化优化动作—隐私权衡，提供更直接的隐私机制；VK-RMD 不执行视频匿名化或正式隐私优化，其差异是以结构接口移除在线原始 RGB，并在异构传感器缺失条件下服务 HPE/HAR。

#### 创新判断

- 类型：问题设定创新 + 结构接口机制创新。
- 强度：中等。
- 独有内容：不是 VK 数据本身，而是 VK 结构编码在 raw-RGB-free 在线推理、完整/子集 teacher-student 和 HPE/HAR 动态模态系统中的贯穿式角色。
- 安全表述：**本文提出一种以 VK 为低外观结构接口的异构人体感知组织方式。**
- 禁止表述：首次提出 skeleton；VK 保证隐私；所有非视觉模态被严格对齐到 VK 坐标空间；任意隐私设备均能稳定产生同等质量 VK。

### 5.2 创新二：FS-SFC 完整到子集的结构—融合一致性约束

#### 论文级定义

FS-SFC 将已有的 complete-to-incomplete teacher/student 思想任务化为一套面向 HPE/HAR 的多层一致性目标。完整观测 teacher 学习所有传感器可用时的任务行为；student 每个 batch 只接收随机可用子集。HPE 同时约束真实姿态、teacher 输出、token、骨长关系和当前可见模态的融合分布；HAR 同时约束真实类别、teacher soft logits、token 和当前可见模态的融合分布。每个可见 chunk 的 uncertainty 还由真实任务误差监督。

#### 形式化描述

对完整模态集合 \(\mathcal M\) 和随机可见子集 \(S\subseteq\mathcal M\)，训练过程为：

\[
T_{\mathcal M}=F_T(\{x_m\}_{m\in\mathcal M}),\qquad
S_S=F_S(\{x_m\}_{m\in S}),\qquad S\sim q(S).
\]

teacher 的融合分布只在 student 可见模态上重归一化：

\[
\bar\alpha^T_m(S)=
\frac{\alpha^T_m}{\sum_{j\in S}\alpha^T_j},\quad m\in S.
\]

HPE student 的实际目标可凝练为：

\[
\mathcal L_{\mathrm{HPE}}=
\mathcal L_{\mathrm{GT}}
+\lambda_o\mathcal L_{\mathrm{output}}
+\lambda_t\mathcal L_{\mathrm{token}}
+\lambda_b\mathcal L_{\mathrm{bone}}
+\lambda_f\,D_{\mathrm{KL}}(\bar\alpha^T(S)\|\alpha^S)
+\lambda_u\mathcal L_{\mathrm{unc}}.
\]

HAR student 的实际目标为：

\[
\mathcal L_{\mathrm{HAR}}=
\mathcal L_{\mathrm{CE}}
+\lambda_k T^2D_{\mathrm{KL}}(p_T^T\|p_S^T)
+\lambda_t\mathcal L_{\mathrm{token}}
+\lambda_f\,D_{\mathrm{KL}}(\bar\alpha^T(S)\|\alpha^S)
+\lambda_u\mathcal L_{\mathrm{unc}}.
\]

这些式子是对当前代码联合目标的形式化凝练，不意味着每个基础损失项均为新损失。可主张的是：面向人体传感器子集变化，将任务输出、HPE 几何结构、任务 token、子集融合行为和 per-chunk uncertainty 组织成统一约束体系。

#### 代码依据

- `HPE/training/engine.py:400-409`：每个 batch 采样 student 子集，而 teacher 始终处理 `teacher_modalities`。
- `HAR/training/engine.py:348-357`：HAR 执行相同的 full-observation-to-random-subset 训练关系。
- `HPE/losses/distill_losses.py:27-37`：teacher alpha 按 student 可见模态筛选并重归一化。
- `HPE/losses/distill_losses.py:40-89`：GT、output、token、bone、fusion-alpha 和 uncertainty 目标。
- `HAR/losses/distill_losses.py:35-44,47-97`：CE、soft-logit、token、fusion-alpha 和 uncertainty 目标。

#### 实验依据与边界

- `tables/fair_missing_modality_baseline.csv`：HAR 的 w/o-KD 为 82.86%，完整 FS-SFC Student-VK 为 85.15%，支持 teacher guidance 对 HAR 的辅助收益。
- 同一文件中 HPE w/o-KD 为 72.68 mm，完整 FS-SFC 为 75.18 mm；因此 FS-SFC 不能被写成 HPE/HAR 均稳定提升的唯一性能来源。对 HPE，它更适合被解释为完整到子集的结构行为正则化，当前权重配置可能存在负迁移。
- 31/15 个组合与缺失强度分析证明 student 确实覆盖 modality-availability space，但不能单独证明每个 FS-SFC 分量均必要。

#### 三篇最近邻工作与差异

1. [MMANet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_MMANet_Margin-Aware_Distillation_and_Modality-Aware_Regularization_for_Incomplete_Multimodal_Learning_CVPR_2023_paper.pdf)：已有完整 teacher 和随机缺失 student，并提出 margin-aware distillation 与 modality-aware regularization。VK-RMD 的不同点是 HPE/HAR 任务化的输出/骨结构或分类分布、token、可见子集融合分布和 per-chunk uncertainty 的联合约束。
2. [CMAD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf)：同样采用完整 teacher 和随机缺失 student，但核心是情感分析中的 correlation-aware feature distillation 与组合难度重加权，不包含人体骨结构约束、可见模态 alpha 重归一化或 HPE/HAR 双任务实例化。
3. [UMDF, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28871)：通过统一 self-distillation、多粒度跨模态交互和动态融合处理缺失模态情感分析；它不是独立完整 teacher 到随机子集 student，也没有本文的姿态结构和可见融合行为约束。

补充边界：[ActionMAE, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25378/25150) 已通过随机模态丢弃和特征重建处理缺失模态动作识别；VK-RMD 不生成缺失 token，而是约束当前可见子集保持完整观测行为。

#### 创新判断

- 类型：训练范式的任务化机制创新。
- 强度：中等；在 HAR 上证据更强，在 HPE 上为任务相关而非普遍增益。
- 独有内容：不是“full teacher + missing student”形式，而是 HPE/HAR 特定的结构—融合多层约束组合，尤其是 teacher 融合行为向当前可见子集的限制与重归一化。
- 安全表述：**本文将完整到不完整学习具体化为面向 HPE/HAR 的 Full-to-Subset Structural-Fusion Consistency。**
- 禁止表述：首次提出完整 teacher 指导缺失 student；全新的蒸馏类别；FS-SFC 在所有任务上稳定提升。

### 5.3 创新三：子集自适应的可见 chunk 双层证据读出

#### 论文级定义

Transformer 交互后，VK-RMD 不直接把全部 token 压缩成单一预测，也不生成缺失模态代理。编码结果仍按当前可见模态来源切回固定长度 chunks；同一个共享任务头读取每个 chunk，形成模态级任务预测和 log-variance。由 log-variance 得到的 contribution proxy 同时用于两条耦合路径：一条融合 chunk 级 pose/logits，另一条融合 pooled chunk tokens 并产生全局预测；最终组合局部专家证据和全局互补证据。

需要准确称为“chunk 级专家读出”，因为代码中的 `shared_head` 在不同 chunk 之间共享参数，并不是每个模态拥有独立参数专家。

#### 形式化描述

对于可见模态 \(m\in S\)，共享任务头得到预测 \(\hat y_m\) 和 log-variance \(s_m\)：

\[
(\hat y_m,s_m)=H_{\mathrm{shared}}(C_m),\qquad
\alpha_m=\frac{\exp(-s_m/\tau)}{\sum_{j\in S}\exp(-s_j/\tau)}.
\]

局部证据与全局 token 证据分别为：

\[
\hat y_{\mathrm{chunk}}=\sum_{m\in S}\alpha_m\hat y_m,qquad
z_S=\sum_{m\in S}\alpha_m\operatorname{Pool}(C_m),qquad
\hat y_{\mathrm{global}}=G(z_S).
\]

当前实现采用等权联合读出：

\[
\hat y=0.5\hat y_{\mathrm{chunk}}+0.5\hat y_{\mathrm{global}}.
\]

其机制独特性在于：只处理可见 chunks、共享任务头产生 chunk 证据、同一 contribution proxy 同时耦合预测级和 token 级聚合，并由 FS-SFC 约束其子集融合行为。

#### 代码依据

- `HPE/models/reliability_fusion.py:70-101`：共享 pose head、chunk split、log-variance alpha、pose-level 聚合、token-level 聚合及双读出。
- `HAR/models/reliability_fusion.py:69-99`：HAR 对 logits 和 pooled tokens 执行同构双层聚合。
- `HPE/losses/distill_losses.py:81-89` 与 `HAR/losses/distill_losses.py:89-97`：每个 chunk 的预测误差监督 log-variance，使 alpha 具有任务贡献含义。
- `HPE/losses/distill_losses.py:64-77` 与 `HAR/losses/distill_losses.py:72-85`：student 当前子集的 alpha 对齐 teacher 在同一可见子集上的重归一化分布。

#### 实验依据

- `dual_readout_ablation/dual_readout_ablation_summary.csv`：HPE 中 chunk-only 最优（95.27 mm），dual 为 99.44 mm，global-only 为 109.81 mm；HAR 中 dual 最优（82.43% Accuracy、43.40% Macro-F1）。这支持“任务相关的局部/全局证据平衡”，不支持 dual 在所有任务上普遍最优。
- `tables/fair_missing_modality_baseline.csv`：uniform fusion 的 HPE 为 82.77 mm、HAR 为 67.46%，当前 learned fusion 分别为 75.18 mm 和 85.15%，支持动态贡献代理优于固定平均的系统级作用。
- `tables/reliability_performance_correlation.csv`：权重与任务表现的 Spearman 相关在 HPE/HAR 分别为 0.90/0.80；这是趋势证据，不是物理可靠性校准证明。

#### 三篇最近邻工作与差异

1. [X-Fi, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)：使用 cross-modal Transformer 和 modality-specific cross-attention 将模态信息注入统一表示；VK-RMD 则在交互后保留可见 chunk 的任务预测，并用同一 alpha 耦合预测级与 token 级双层读出。
2. [FuseMoE, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html)：通过稀疏门控 MoE 路由模态到参数专家以处理 fleximodal 和不规则数据；VK-RMD 不路由多个参数专家，而是用共享任务头读取观察到的 modality chunks，并融合它们的任务证据。
3. [MCMoE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38104)：使用自适应 gated generator 补全缺失模态，再混合 unimodal experts 提取联合表示；VK-RMD 不重建缺失模态，仅聚合当前可见 chunks，并将同一 contribution proxy 用于局部预测和全局 token 两层。

#### 创新判断

- 类型：融合机制创新。
- 强度：中等。
- 独有内容：不是一般 MoE 或 uncertainty fusion，而是“可见 chunk 保留 + 共享头任务证据 + 同权重双层聚合 + 子集融合一致性”的耦合关系。
- 安全表述：**本文提出一种面向动态可用子集的可见 chunk 双层证据读出机制。**
- 禁止表述：首次提出 expert/uncertainty/local-global fusion；每个模态具有独立参数专家；权重是真实传感器可靠性；dual 对所有任务均最优。

## 6. 文献机制查重矩阵

| 方法 | 主要问题 | 缺失模态处理 | teacher/student | 融合/专家机制 | 与 VK-RMD 的关键差异 |
|---|---|---|---|---|---|
| X-Fi, ICLR 2025 | 任意模态组合的人体感知 | 随机模态丢弃 | 无本文式 full-to-subset 多层约束 | Transformer + modality-specific cross-attention | 包含 RGB；无 VK 结构接口、FS-SFC 和 chunk 任务双读出 |
| SCoPLe, CVPR 2025 | skeleton 零样本 HAR | 非缺失模态问题 | prompt/语义迁移 | skeleton-text prompt | 单任务骨架识别，不是异构传感器动态可用系统 |
| STPrivacy, ICCV 2023 | 视频动作识别隐私—效用权衡 | 非缺失模态问题 | 对抗隐私学习 | tubelet 稀疏化与匿名化 | 隐私机制更强，但仍处理视频；无传感器子集学习与 HPE |
| MMANet, CVPR 2023 | incomplete multimodal learning | 随机 modality dropout | full teacher -> incomplete student | margin/relation distillation + modality regularization | 已覆盖训练形式；无 VK、人体骨结构和可见 alpha 一致性 |
| ActionMAE, AAAI 2023 | missing-modality action recognition | 重建缺失特征 | reconstruction guidance | masked autoencoder | 生成缺失表征；VK-RMD 只利用可见证据 |
| UMDF, AAAI 2024 | incomplete sentiment analysis | unified missing-modality learning | self-distillation | 多粒度交互 + dynamic fusion | 无独立 full teacher、人体结构约束和 HPE |
| CMAD, ICCV 2025 | incomplete sentiment analysis | 随机丢弃模态 | full teacher -> missing student | correlation-aware distillation + difficulty weighting | 已覆盖 full/missing 形式；目标和一致性层级不同 |
| FuseMoE, NeurIPS 2024 | fleximodal/irregular data | 稀疏专家路由 | 非本文式 FS-SFC | parameter MoE + Laplace gate | 路由参数专家，而非共享头读取 observed chunks |
| MCMoE, AAAI 2026 | incomplete multimodal AQA | 生成并细化缺失模态 | 完整特征/专家知识指导生成 | gated generator + modality experts | 显式补全缺失模态；VK-RMD 不生成缺失 token |

本矩阵支持的结论是“在本轮正式论文检索范围内没有发现完全相同的端到端计算链”，而不是对全部已发表论文作绝对不存在声明。

## 7. 贡献—代码—实验闭环

| 贡献 | 必须引用的代码 | 必须引用的实验 | 论文能够得出的结论 |
|---|---|---|---|
| VK 低外观结构接口 | `SkeletonPromptEncoder`、`BoneGraphMixer`、HPE/HAR `vk_rcd` | identity probe、paired marginal contribution、Student-VK/NV | VK 显著降低相对 defaced RGB 的主体可预测性，并对匹配的非 VK 子集提供一致任务增益 |
| FS-SFC | HPE/HAR `engine.py` 的 full teacher/random subset；两套 `distill_losses.py` | w/o KD、all-combination、missing severity | 完整观测多层 guidance 对 HAR 有明确辅助收益；HPE 中属于任务相关正则化而非稳定增益来源 |
| 可见 chunk 双层证据读出 | 两套 `reliability_fusion.py` | uniform fusion、chunk/global/dual、weight-performance correlation | 动态子集需要任务相关证据聚合；HPE 偏向 chunk 几何，HAR 更受益于 dual 语义互补 |

## 8. 可直接写入论文的三条贡献

1. **低外观结构接口。** 本文提出 VK-RMD，将带有关节身份与骨拓扑的 Visual Keypoints 组织为 raw-RGB-free 在线人体感知接口，并与 Depth、LiDAR、mmWave 和 WiFi-CSI 的异构表征共同形成任务监督的人体状态 token。该设计降低了在线原始视觉暴露，同时保留 HPE/HAR 所需的显式人体结构。

2. **完整到子集的结构—融合一致性。** 本文将 complete-to-incomplete guidance 具体化为 FS-SFC，以完整观测 teacher 的任务输出、token、HPE 骨结构或 HAR 分类分布，以及在 student 可见子集上重归一化的融合行为约束随机子集 student，从而学习动态传感器可用条件下的任务行为。该机制是针对人体感知的多层一致性设计，不被表述为全新的蒸馏类别。

3. **子集自适应的可见 chunk 双层证据读出。** 本文在跨模态交互后保留当前可见模态的 chunk 级任务证据，并以 learned contribution proxy 同时组织局部预测聚合和全局 fused-token 读出。HPE/HAR 全组合、uniform fusion 与 chunk/global/dual 消融揭示了几何回归和语义分类对局部—全局证据的不同需求。

## 9. 可直接写入 Introduction 的核心段落

现有多模态人体感知通常分别处理视觉暴露或传感器缺失：前者通过匿名化、骨架或非视觉信号减少原始图像信息，后者通过模态丢弃、特征重建、蒸馏或专家路由适应不完整输入。然而，在在线不处理原始 RGB 且传感器可用性持续变化的部署条件下，仍需同时回答三个相互关联的问题：以何种低外观表示保留人体结构，如何缩小完整观测与有限子集之间的行为差异，以及如何避免不同子集中的模态证据被单一全局表示掩盖。为此，VK-RMD 以 Visual Keypoints 构建带关节身份和骨拓扑的在线结构接口，通过 FS-SFC 将完整观测下的任务输出、结构表示和融合行为传递给随机子集 student，并采用可见 chunk 双层证据读出联合模态局部判断与跨模态全局互补信息。三个机制围绕同一目标协同工作，而非将关键点、蒸馏和 Transformer 作为彼此独立的模块叠加。

## 10. 最终总结段

> VK-RMD 面向低视觉暴露人体感知中结构信息保留与动态传感器可用性并存的科学问题。该框架首先将 Visual Keypoints 编码为带有关节身份和骨架拓扑的低外观结构接口，并与 Depth、LiDAR、mmWave 和 WiFi-CSI 的异构特征共同组织为任务相关 token；随后，FS-SFC 利用完整模态 teacher 的任务输出、结构表示及当前可见子集上的融合行为约束随机模态子集 student，使部分观测模型学习保持完整观测下的人体感知行为；最后，子集自适应的可见 chunk 双层证据读出同时保留当前模态的局部任务证据和跨模态全局互补信息。因而，VK-RMD 的创新并非来自孤立的关键点、蒸馏或 Transformer，而在于围绕 raw-RGB-free 在线推理和随机传感器缺失，建立了低外观结构接口、完整到子集结构—融合一致性以及可见证据双层聚合之间的统一机制关系。

## 11. 审稿风险与固定回答

### “这仍然是模块组合。”

回答重点不是否认基础模块已有，而是指出三个模块之间存在可追溯的因果和接口约束：VK 定义在线人体结构输入，FS-SFC 定义 full/subset 观测关系，visible-chunk readout 定义当前子集证据如何被任务化读取。每一项都有独立代码接口和配套实验，完整计算链与检索到的最近邻方法不同。

### “完整 teacher 指导缺失 student 已经有人做过。”

应直接承认 MMANet、CMAD 等已有该训练形式。本文贡献是 HPE/HAR 场景中的结构—融合多层一致性具体化，尤其是 HPE 骨结构和可见模态融合分布约束，而非重新命名传统蒸馏。

### “VK 并不等于隐私。”

应同意。身份探针显示 VK 为 15.43%，仍显著高于 2.50% chance；Depth、LiDAR 等也有残余泄露。本文只主张在线不处理原始 RGB和降低视觉暴露，不主张匿名、形式化隐私或整个多传感器系统无身份信息。

### “为什么 dual 不是 HPE 最优？”

该结果揭示任务差异而非推翻机制：HPE 的精细几何回归更依赖 chunk 局部结构证据；HAR 的动作语义更需要局部证据和全局交互。统一架构提供两层证据，但最优平衡具有任务相关性。因此不能把 dual 写成普遍最优，只能写成跨任务统一读出及其任务相关证据平衡。

### “是否能宣称严格 body-latent alignment？”

不能。代码证明的是 modality-specific encoder/projector、同维 token、Transformer 交互和共同任务监督。最准确表述是 `task-supervised human-state tokenization`，而不是严格对齐或 VK 作为 query 强制映射其他模态。

## 12. 最终标题建议

**VK-RMD: Availability-Conditioned Structural Evidence Learning for Raw-RGB-Free Multimodal Human Perception**

中文：

**VK-RMD：面向无在线原始 RGB 多模态人体感知的可用性条件结构证据学习**

该标题把论文中心放在“动态可用性条件下的结构证据学习”，而不是把 distillation、privacy guarantee 或单一 fusion 模块抬成唯一创新。

## 13. 已核验的核心文献

1. [X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)
2. [MMANet: Margin-Aware Distillation and Modality-Aware Regularization for Incomplete Multimodal Learning, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_MMANet_Margin-Aware_Distillation_and_Modality-Aware_Regularization_for_Incomplete_Multimodal_Learning_CVPR_2023_paper.pdf)
3. [ActionMAE: Missing Modality Robust Action Recognition, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25378/25150)
4. [Unified Multimodal Self-Distillation for Incomplete Multimodal Sentiment Analysis, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28871)
5. [CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf)
6. [SCoPLe: Semantic-guided Cross-Modal Prompt Learning for Skeleton-based Zero-shot Action Recognition, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Semantic-guided_Cross-Modal_Prompt_Learning_for_Skeleton-based_Zero-shot_Action_Recognition_CVPR_2025_paper.html)
7. [STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_STPrivacy_Spatio-Temporal_Privacy-Preserving_Action_Recognition_ICCV_2023_paper.html)
8. [FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html)
9. [MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38104)

## 14. 最终创新强度判断

| 维度 | 判断 |
|---|---|
| 是否存在项目独有内容 | 是。三项均包含当前代码中特定的任务接口或机制耦合，不只是调用现成模块 |
| 是否存在完全同构的已发表工作 | 本轮 2021-2026 正式论文检索未发现；这是范围限定结论，不是绝对不存在证明 |
| 是否可称为算法创新 | 可以称为面向特定科学问题的机制和训练目标创新；不宜称为全新基础算子或理论 |
| 最强创新 | FS-SFC 与 visible-chunk dual evidence readout 的耦合关系 |
| 最稳应用创新 | VK 低外观结构接口 + raw-RGB-free 在线 HPE/HAR + arbitrary-subset validation |
| 最大证据短板 | HPE 中 full FS-SFC 不优于 w/o KD；dual 在 HPE 不优于 chunk-only；cross-scene 明显退化 |
| 推荐总体定位 | 结构证据可用性条件学习框架，而不是 VK/KD/attention 的并列组合 |
