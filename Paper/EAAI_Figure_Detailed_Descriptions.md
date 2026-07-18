# EAAI 论文图件详细设计说明

本文图件必须服务同一条主线：

> VK-RMD 将 RGB 视觉知识压缩为 VK 人体结构桥梁，将 VK、Depth、LiDAR、mmWave、WiFi-CSI 等异构模态映射到统一 body-latent token 空间，并在随机模态子集训练和 reliability-aware fusion 下实现 reduced-visual-exposure 的 HPE/HAR 缺失模态鲁棒感知。

因此，所有图都不应把论文画成“VK + dropout + attention + KD 的模块堆叠”，而要围绕三个问题展开：

1. VK 如何作为 RGB 与非 RGB 传感器之间的结构桥梁？
2. 模型如何在任意模态子集下完成 HPE/HAR？
3. 结果是否证明该框架在缺失模态、非视觉设置和公平 baseline 下有效？

---

## Fig. 1. Overall Framework of VK-RMD

对应文件：

`figures/fig1_overall_framework.png`

论文位置：

方法部分，`Overview of VK-RMD`

### 核心结论

VK-RMD 不是直接使用 RGB 推理，而是将 RGB-derived visual knowledge 压缩成 VK structural bottleneck，再把 VK 与其他非 RGB 模态映射到 shared body-latent token space，最后通过 reliability-aware fusion 输出 HPE/HAR。

### 图的类型

Schematic-led composite。

这是全文最重要的“机制总图”，应该是 hero figure。

### 推荐图结构

建议分为 4 个横向阶段：

1. **Training-time RGB privileged source**
   - 左侧放 RGB frame 图标。
   - 箭头指向 `RGB-to-VK keypoint extraction`。
   - 标注：`RGB is used only to obtain VK / privileged structural knowledge during training`。
   - 不要画成 RGB 参与最终推理。

2. **VK structural bridge**
   - 中左侧放 17-joint skeleton / VK 图。
   - 标注：
     - `appearance-suppressed structural bottleneck`
     - `body topology`
     - `pose geometry`
   - 视觉上应明显区别于 RGB 图像，例如用骨架点线图，不用照片。

3. **Heterogeneous sensor encoding into body-latent tokens**
   - 中间放五条模态分支：
     - VK
     - Depth
     - LiDAR
     - mmWave
     - WiFi-CSI
   - 每条分支结构：
     `input x_m -> modality-specific encoder E_m -> projector P_m -> body-latent token z_m`
   - 在所有 token 之后画一个共享区域：
     `Shared body-latent token space`
   - 关键公式可以放在图中：
     `z_m = P_m(E_m(x_m)) + e_m`

4. **Subset-invariant fusion and task heads**
   - 右侧画 `available subset S`。
   - 将缺失模态用灰色虚线或灰掉的 token 表示。
   - 进入 `learned reliability proxy / reliability-aware fusion`。
   - 输出两个 head：
     - HPE head -> 17 x 3 pose
     - HAR head -> action class

### 需要强调的视觉信息

- RGB 分支必须明确标为 `training only`。
- Inference 部分不能出现 raw RGB。
- VK 不应画成普通第五模态，而应画在 RGB 与 body-latent token space 中间，作为 structural bridge。
- Reliability 模块不要写成 physical sensor reliability，建议写：
  `learned reliability proxy`
  或
  `modality-contribution weights`

### 可放入图注的句子

`RGB is not used as an inference modality. Instead, RGB-derived VK provides an appearance-suppressed structural bridge. Each available modality is encoded and projected into a shared body-latent token space, where reliability-aware fusion produces task-specific representations for HPE and HAR.`

### 数据来源

机制图，不依赖具体 CSV。

### 审稿风险

- 如果 RGB 画得太靠近 inference，审稿人会质疑“你到底有没有用 RGB 推理”。
- 如果 VK 画成普通输入模态，会削弱“VK bridge”的创新叙事。
- 如果 reliability 写成“true sensor reliability”，会被质疑没有物理可靠性标定。

### 绘图建议

- 用蓝色或灰色表示 training-only RGB。
- 用绿色或青色强调 VK structural bridge。
- 用统一的 token 小方块表示 body-latent space。
- 缺失模态用灰色半透明 token 或断开的虚线表示。

---

## Fig. 2. Missing-Modality Training and Evaluation Protocol

对应文件：

`figures/fig2_missing_modality_setting.png`

论文位置：

方法部分，`Random missing-modality training`

### 核心结论

VK-RMD 的缺失模态处理不是普通 dropout 正则化，而是把每个 sampled modality subset 当成部署时可能出现的合法 sensing configuration；训练时随机采样子集，测试时评估所有有效组合。

### 图的类型

Schematic-led composite / protocol diagram。

### 推荐图结构

建议分为上下两行：

#### 上半部分：Training

显示完整模态集合：

`M_HPE = {VK, D, L, R, W}`

然后画随机采样：

`S ~ q(S), S subset M`

示例子集：

- `VK + Depth + mmWave`
- `Depth + LiDAR + WiFi`
- `VK + LiDAR`
- `mmWave + WiFi`

每个 batch 只输入当前 subset。

标注：

`Each sampled subset is trained as a valid input condition.`

#### 下半部分：Evaluation

画两个 exhaustive testing blocks：

- HPE: `31 non-empty combinations`
- HAR: `15 non-empty combinations`

可以用组合网格表示：

- full modality
- missing 1
- missing 2
- missing 3
- single modality

### 需要强调的视觉信息

- Training 是 random subset sampling。
- Evaluation 是 all-combination testing。
- 不能让读者误解为“训练只做一次 dropout，测试只测 full modality”。

### 可放入图注的句子

`During training, modality subsets are randomly sampled to simulate deployment-time sensor availability. During evaluation, all valid non-empty modality combinations are tested, yielding 31 HPE combinations and 15 HAR combinations.`

### 数据来源

机制图，不直接依赖数值。

但可以在图角落标注：

- HPE: 5 modalities -> 31 combinations
- HAR: 4 modalities -> 15 combinations

### 审稿风险

- 如果只画“dropout”，审稿人会说这只是普通 modality dropout。
- 需要明确这是 deployment-condition simulation。

### 绘图建议

- 使用矩阵/网格表达组合空间。
- 用颜色表示 available modality，用灰色表示 missing modality。
- 在训练部分放随机骰子或 sampling icon，但不要显得太卡通。

---

## Fig. 3. Main HPE/HAR Results

对应文件：

`figures/fig3_main_results.png`

论文位置：

结果部分，`Main HPE results` 与 `Main HAR results` 后。

### 核心结论

VK-RMD 在 HPE 和 HAR 两个任务上均提升 all-combination missing-modality robustness，同时 Student-NV 支持更严格的非视觉推理设置。

### 图的类型

Quantitative grid。

### 推荐 panel 设计

建议做成 2 x 2 或 1 x 2。

#### Panel A: HPE random split

柱状图或点图：

- Official X-Fi/VK table: 103.70 mm
- VK-RMD Student-VK: 75.18 mm
- VK-RMD Student-NV: 84.68 mm

Y 轴：

`Average MPJPE over modality combinations (mm, lower is better)`

必须标注 lower is better。

#### Panel B: HAR random split

柱状图或点图：

- Official/reproduced X-Fi/VK: 62.46%
- VK-RMD Student-VK: 85.15%
- VK-RMD Student-NV: 80.62%

Y 轴：

`Average accuracy over modality combinations (%, higher is better)`

必须标注 higher is better。

#### Panel C: HPE protocol comparison 可选

展示 Student-VK / Student-NV 在：

- random
- cross-subject
- cross-scene

的平均 MPJPE。

用于说明 cross-scene 是 limitation。

#### Panel D: HAR protocol comparison 可选

展示 Student-VK / Student-NV 在：

- random
- cross-subject
- cross-scene

的平均 accuracy。

### 推荐数值

HPE：

- Official X-Fi/VK: 103.70 mm
- Student-VK random: 75.18 mm
- Student-NV random: 84.68 mm
- Student-VK cross-scene: 142.02 mm
- Student-NV cross-scene: 153.89 mm
- Student-VK cross-subject: 78.57 mm
- Student-NV cross-subject: 90.74 mm

HAR：

- Official/reproduced X-Fi/VK: 62.46%
- Student-VK random: 85.15%
- Student-NV random: 80.62%
- Student-VK cross-scene: 80.61%
- Student-NV cross-scene: 76.70%
- Student-VK cross-subject: 85.30%
- Student-NV cross-subject: 81.26%

### 图注重点

不要写成“全面解决跨场景”。

推荐图注：

`VK-RMD improves average all-combination robustness in random and cross-subject settings, while cross-scene HPE remains substantially more difficult, indicating that missing-modality robustness and scene generalization are distinct challenges.`

### 数据来源

- `HPE/outputs/eval/student_vk_random_all_combinations.csv`
- `HPE/outputs/eval/student_nv_random_nonvisual_combinations.csv`
- HPE cross-scene/cross-subject eval CSV
- `HAR/outputs/eval/student_vk_random_all_combinations.csv`
- `HAR/outputs/eval/student_nv_random_nonvisual_combinations.csv`
- HAR cross-scene/cross-subject eval CSV
- `HPE/legacy_xfi/xfi_hpe_table1.csv`
- `HAR/legacy_xfi/xfi_har_table6.csv`

### 审稿风险

- HPE 和 HAR 指标方向相反，必须标清：
  - HPE lower is better
  - HAR higher is better
- 不要把 244.91 mm reproduced Ori-HPE baseline 作为主图核心 baseline。
- Cross-scene 结果较弱，应画出来并解释为 limitation，不要藏。

---

## Fig. 4. Missing-Modality Robustness by Missing Severity

对应文件：

`figures/fig4_missing_modality_robustness.png`

辅助已有图：

`figures/subset_severity_curve.png`

论文位置：

结果部分，`Robustness under missing modalities`

### 核心结论

VK-RMD 在轻中度缺失下退化较平滑，但 severe missing / single-modality setting 仍然困难。这说明方法增强的是 modality-availability space 的整体可用性，而不是保证每个单模态都强。

### 图的类型

Quantitative trend / grouped robustness curve。

### 推荐 panel 设计

建议 2 个 panel：

#### Panel A: HPE missing severity

X 轴：

`Number of missing modalities`

Y 轴：

`Average MPJPE (mm, lower is better)`

曲线：

- HPE Student-VK
- HPE Student-NV

数值：

Student-VK：

- missing 0: 50.17
- missing 1: 53.55
- missing 2: 60.78
- missing 3: 78.10
- missing 4: 124.76

Student-NV：

- missing 0: 51.03
- missing 1: 61.32
- missing 2: 79.43
- missing 3: 124.32

#### Panel B: HAR missing severity

X 轴：

`Number of missing modalities`

Y 轴：

`Average accuracy (%, higher is better)`

曲线：

- HAR Student-VK
- HAR Student-NV

数值：

Student-VK：

- missing 0: 96.56
- missing 1: 93.83
- missing 2: 87.55
- missing 3: 70.00

Student-NV：

- missing 0: 96.12
- missing 1: 90.05
- missing 2: 66.02

### 图注重点

推荐图注：

`Moderate missingness is handled well, whereas severe missingness and single-modality inference remain difficult. This supports the paper's claim of missing-modality robustness while defining the method's practical boundary.`

### 数据来源

`tables/subset_severity_summary.csv`

### 审稿风险

- 单模态结果不强，不能写成“arbitrary missing modalities are solved”。
- 要强调 severe missing 是 remaining limitation。

### 绘图建议

- HPE 和 HAR 不要共用同一个 y 轴。
- HPE 使用向上代表变差，HAR 使用向下代表变差。可以在标题中显式写：
  - HPE: lower is better
  - HAR: higher is better
- 建议用点线图，不建议只用柱状图，因为趋势更重要。

---

## Fig. 5. Ablation and Fair Baseline Evidence

对应文件：

`figures/fig5_ablation.png`

论文位置：

结果部分，`Ablation study`

### 核心结论

VK-RMD 的主要稳定证据来自 missing-modality training 和 reliability-aware fusion；distillation 是辅助且任务相关，不能作为唯一主贡献。

### 图的类型

Quantitative grid / ablation comparison。

### 推荐 panel 设计

建议做成 2 x 2：

#### Panel A: HPE fair baselines

显示：

- Official X-Fi/VK: 103.70 mm
- Uniform fusion: 82.77 mm
- w/o KD: 72.68 mm
- VK-RMD Student-VK: 75.18 mm
- Student-NV: 84.68 mm

Y 轴：

`Average MPJPE (mm, lower is better)`

注意：

HPE w/o KD 比 full VK-RMD 略好，这必须保留，不能画错。

#### Panel B: HAR fair baselines

显示：

- Official/reproduced X-Fi/VK: 62.46%
- Uniform fusion: 67.46%
- w/o KD: 82.86%
- VK-RMD Student-VK: 85.15%
- Student-NV: 80.62%

Y 轴：

`Average accuracy (%, higher is better)`

#### Panel C: Reliability fusion effect

用两组差值显示：

- HPE: Uniform 82.77 mm vs VK-RMD 75.18 mm
- HAR: Uniform 67.46% vs VK-RMD 85.15%

可以用 connected dot plot，突出 uniform -> learned reliability proxy 的变化。

#### Panel D: KD effect

用两组差值显示：

- HPE: w/o KD 72.68 mm vs VK-RMD 75.18 mm
- HAR: w/o KD 82.86% vs VK-RMD 85.15%

图上应标注：

`KD is auxiliary and task-dependent`

### 图注重点

推荐图注：

`Fair ablations show that reliability-aware fusion improves over uniform fusion in both HPE and HAR, while distillation has task-dependent effects. Therefore, the main mechanism should be interpreted as VK-anchored missing-modality learning rather than distillation alone.`

### 数据来源

- `tables/fair_missing_modality_baseline.csv`
- `HPE/outputs/eval/ablation_uniform_fusion_all_combinations.csv`
- `HPE/outputs/eval/ablation_no_distill_all_combinations.csv`
- `HAR/outputs/eval/ablation_uniform_fusion_all_combinations.csv`
- `HAR/outputs/eval/ablation_no_distill_all_combinations.csv`

### 审稿风险

- 如果只画 full VK-RMD 最好，会被质疑选择性展示。
- HPE no-KD 更好必须诚实展示。
- 图注要明确 distillation 不是所有提升来源。

### 绘图建议

- HPE 用误差条方向 lower better。
- HAR 用准确率方向 higher better。
- 对 KD panel 加灰色提示框：
  `HPE: no-KD competitive`
  `HAR: KD modestly helpful`

---

## Fig. 6. Reliability-Aware Fusion and Modality Contribution Proxy

对应文件：

`figures/fig6_reliability_weights.png`

辅助已有图：

- `figures/fig6a_hpe_reliability_weights.png`
- `figures/fig6b_har_reliability_weights.png`
- `figures/reliability_weight_heatmap.png`

论文位置：

结果部分，`Reliability-aware fusion analysis`

### 核心结论

Reliability-aware fusion 不是固定平均融合。模型会根据当前可用模态组合重新分配 modality-contribution weights；这些权重是 learned reliability proxy，不是物理传感器可靠性的直接测量。

### 图的类型

Heatmap + diagnostic table / quantitative heatmap。

### 推荐 panel 设计

建议 3 个 panel：

#### Panel A: HPE reliability weight heatmap

行：

若干代表性 modality subsets，例如：

- VK+Depth
- Depth+LiDAR
- LiDAR+mmWave
- VK+Depth+LiDAR
- VK+Depth+LiDAR+mmWave+WiFi-CSI

列：

- VK
- Depth
- LiDAR
- mmWave
- WiFi-CSI

颜色：

权重大小，0 到 1。

Unavailable modality 用空白或灰色。

#### Panel B: HAR reliability weight heatmap

行：

- VK+mmWave
- LiDAR+mmWave
- VK+Depth+mmWave
- VK+Depth+LiDAR+mmWave

列：

- VK
- Depth
- LiDAR
- mmWave

#### Panel C: reliability-performance correlation

小柱图或文本框：

- HPE Pearson: 0.9396
- HPE Spearman: 0.9000
- HAR Pearson: 0.8833
- HAR Spearman: 0.8000

标题：

`Weights correlate with subset-level performance`

### 图中可标注的代表性数值

HPE：

- VK+Depth: VK 0.396, Depth 0.604
- Depth+LiDAR: Depth 0.622, LiDAR 0.378
- LiDAR+mmWave: LiDAR 0.489, mmWave 0.511
- Full five-modalities: VK 0.189, Depth 0.270, LiDAR 0.178, mmWave 0.183, WiFi 0.179

HAR：

- VK+mmWave: VK 0.414, mmWave 0.586
- LiDAR+mmWave: LiDAR 0.410, mmWave 0.590
- VK+Depth+mmWave: VK 0.317, Depth 0.331, mmWave 0.352
- VK+Depth+LiDAR+mmWave: VK 0.238, Depth 0.249, LiDAR 0.245, mmWave 0.268

### 图注重点

推荐图注：

`The heatmaps visualize learned reliability-proxy weights rather than calibrated physical sensor reliability. The weights vary across modality subsets and correlate positively with subset-level task performance, supporting their interpretation as task-level modality-contribution proxies.`

### 数据来源

- `HPE/outputs/eval/student_vk_reliability_weights.csv`
- `HAR/outputs/eval/student_vk_reliability_weights.csv`
- `tables/reliability_performance_correlation.csv`

### 审稿风险

- 不能说“模型真实感知了传感器可靠性”。
- 应写成：
  `learned reliability proxy`
  `task-level modality contribution`
- 如果没有 corruption/noise weight response 结果，不要声称 causal reliability。

### 绘图建议

- 使用同一个 colorbar。
- 空白格表示 unavailable modality，不要填 0，否则读者会误解成模型主动给了 0 权重。
- 如果空间有限，HPE 和 HAR 分成 Fig.6a 和 Fig.6b 也可以。

---

## 建议新增 Fig. 7. VK-to-Body-Latent Alignment Diagnostic

当前是否在 tex 中：

尚未作为主图插入。

建议状态：

如果 `run_body_latent_alignment.py` 已经跑出结果，可以作为补充图或主文机制诊断图加入。

### 核心结论

不同模态不是在 raw space 对齐，而是在 shared body-latent space 中通过 task supervision 和 projector 学到结构语义对齐。

### 图的类型

Asymmetric mixed-modality figure。

### 推荐 panel 设计

#### Panel A: alignment schematic

画出：

`RGB -> VK -> z_vk`

以及：

`Depth/LiDAR/mmWave/WiFi -> encoder/projector -> z_m`

所有 `z_m` 汇入：

`shared body-latent space`

#### Panel B: representation similarity matrix

如果有输出，可以画模态间 latent cosine similarity：

行列：

- VK
- Depth
- LiDAR
- mmWave
- WiFi

颜色：

cosine similarity 或 CKA similarity。

#### Panel C: same-label vs different-label distance

展示同一动作/同一姿态样本的跨模态 latent 距离是否小于不同标签样本。

### 图注重点

`The diagnostic does not claim perfect geometric correspondence between raw sensors. It tests whether modality-specific projectors produce a shared task-relevant body-latent space.`

### 数据来源

如果已运行：

- `outputs/eval/body_latent_alignment_hpe.csv`
- `outputs/eval/body_latent_alignment_har.csv`

如果未运行：

该图先不要放主文，只放在绘图计划或 supplementary TODO。

### 审稿风险

- 如果相似度不明显，不要强行放主文。
- 这张图用于解释机制，不是必须主结果。

---

## 建议新增 Fig. 8. Reliability Response under Modality Corruption

当前是否在 tex 中：

尚未作为主图插入；tex 中已有 corruption diagnostic 的文字说明，但没有正式结果。

建议状态：

如果 `reliability_corruption_hpe.csv` 和 `reliability_corruption_har.csv` 已经生成，可以作为 reliability claim 的强补充。

### 核心结论

当某个模态被人为污染时，该模态的 learned reliability-proxy weight 应下降，其他模态权重应有补偿趋势；任务性能同时下降。

### 图的类型

Quantitative grid。

### 推荐 panel 设计

#### Panel A: HPE corruption performance

X 轴：

- corrupt VK
- corrupt Depth
- corrupt LiDAR
- corrupt mmWave
- corrupt WiFi

Y 轴：

`Delta MPJPE (mm, higher means worse)`

#### Panel B: HPE weight response

Y 轴：

`Delta corrupted-modality weight`

期望多数为负值。

#### Panel C: HAR corruption performance

X 轴：

- corrupt VK
- corrupt Depth
- corrupt LiDAR
- corrupt mmWave

Y 轴：

`Delta accuracy (pp, lower means worse)`

#### Panel D: HAR weight response

Y 轴：

`Delta corrupted-modality weight`

### 图注重点

`The corruption test evaluates whether learned reliability-proxy weights respond to degraded inputs. It should be interpreted as a diagnostic of adaptive fusion, not as calibration of physical sensor quality.`

### 数据来源

- `HPE/outputs/eval/reliability_corruption_hpe.csv`
- `HAR/outputs/eval/reliability_corruption_har.csv`

### 审稿风险

- 如果权重不下降，不要放这张图。
- 如果只跑了 100/200 batches，应写清楚是 fast diagnostic，不是 full validation benchmark。

---

## 当前主图顺序建议

推荐正文主图：

1. Fig. 1 Overall Framework
2. Fig. 2 Missing-Modality Protocol
3. Fig. 3 Main HPE/HAR Results
4. Fig. 4 Missing-Modality Robustness
5. Fig. 5 Fair Baseline and Ablation
6. Fig. 6 Reliability-Aware Fusion

可选补充图：

7. VK-to-Body-Latent Alignment Diagnostic
8. Reliability Response under Corruption

如果版面紧张，Fig. 7 和 Fig. 8 放 Supplementary / Appendix，不要强行挤进正文。

---

## 每张图对应的审稿问题

| 图 | 回答的审稿问题 | 作用 |
|---|---|---|
| Fig. 1 | 你的方法是不是模块堆叠？ | 说明 VK bridge + body-latent alignment + subset fusion 是完整流程 |
| Fig. 2 | 你的 missing-modality training 和普通 dropout 有什么区别？ | 说明训练模拟部署条件，测试覆盖所有组合 |
| Fig. 3 | 主结果是否有效？ | 展示 HPE/HAR 的平均缺失模态性能 |
| Fig. 4 | 任意缺失下是否稳定？ | 展示 moderate missing 强，severe missing 是 limitation |
| Fig. 5 | baseline 是否公平？KD 是否真的必要？ | 展示 fair baseline、uniform fusion、w/o KD |
| Fig. 6 | reliability 是否只是普通 attention？ | 展示 learned reliability proxy 的动态权重和性能相关性 |
| Fig. 7 | RGB/VK 与其他模态怎么对齐？ | 展示 body-latent 对齐机制 |
| Fig. 8 | reliability 权重是否响应模态退化？ | 展示 corruption diagnostic |

---

## 全文图件总原则

1. 不要把 VK 画成普通第五模态，要画成 RGB 与非 RGB 之间的 structural bridge。
2. 不要把 reliability 权重画成物理传感器可靠性，要写成 learned reliability proxy。
3. 不要隐藏 cross-scene 下降，要在 Fig. 3 或正文表中诚实呈现。
4. 不要把 KD 画成唯一核心模块，Fig. 5 必须体现 HPE no-KD competitive。
5. HPE 与 HAR 指标方向必须永远标清：
   - HPE: MPJPE lower is better
   - HAR: accuracy higher is better
6. 主图要强调系统流程和证据链，而不是堆满所有实验细节。

