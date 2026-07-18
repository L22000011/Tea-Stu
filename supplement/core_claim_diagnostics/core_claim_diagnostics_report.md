# 核心理论质疑的小实验/诊断证据包

本报告由 `scripts/build_core_claim_diagnostics.py` 生成，只汇总已有结果，不训练、不修改模型。

## 总结判断

- body-latent token 对齐目前只能写成 **task-supervised body-latent tokenization**，不能写成严格语义对齐证明。
- FSG 可以写成 **full-modality behavior regularization under modality-availability asymmetry**，不能写成传统大模型蒸馏或唯一性能来源。
- privacy-friendly 必须降调为 **reduced visual exposure / RGB-free inference**，不能写成 anonymity 或 privacy guarantee。

## 1. Body-latent alignment 证据边界

- HPE body-latent diagnostic rows: 4.
- HAR body-latent diagnostic rows: 3.
- 当前 cosine gap 普遍较小，因此 Fig.7/相关表格只能作为弱诊断证据。
- 推荐写法：模型在任务监督下诱导人体状态相关 token 表示，而不是显式证明异构传感器严格语义对齐。

## 2. FSG 证据边界

### HPE
- Full method avg: 75.18 (MPJPE mm lower-better)
- No-KD avg: 72.68
- Interpretation: No-KD is competitive or stronger; present FSG as auxiliary.
- Fusion interpretation: Learned fusion improves over uniform.

### HAR
- Full method avg: 85.15 (Accuracy % higher-better)
- No-KD avg: 82.86
- Interpretation: FSG/KD improves over no-KD.
- Fusion interpretation: Learned fusion improves over uniform.

## 3. Privacy-friendly 证据边界

- 已生成 5 类 representation 的 visual exposure / residual risk 定性汇总。
- 这不是隐私攻击实验，也不是形式化隐私指标，不能作为论文中的量化风险结果。
- 系统性评价应使用 `scripts/run_privacy_leakage_probe.py` 的 subject-identity leakage probe。
- 推荐写法：VK/Depth 减少 raw RGB 外观暴露，但仍保留身体几何和动作线索。

## 4. 三条审稿质疑回应矩阵

### Body-latent alignment may be ordinary Transformer fusion.
- Evidence strength: limited
- Available evidence: Token projection, modality embedding, Transformer interaction, HPE/HAR supervision, VK semantic perturbation, body-latent diagnostic.
- Safe response: Call it task-supervised body-latent tokenization, not strict semantic alignment.
- Paper action: Use Fig.7/diagnostic only as weak support; emphasize all-combination performance and perturbation evidence.

### FSG may be modality dropout plus KD under a new name.
- Evidence strength: bounded
- Available evidence: Full teacher sees all modalities; student samples subsets; HPE/HAR losses include output/logit, token, reliability and structure guidance where implemented.
- Safe response: Define FSG as modality-availability asymmetry and full-modality behavior regularization.
- Paper action: Do not present FSG as the only performance source; keep no-KD out of main table if it distracts, but discuss task-dependent auxiliary role if needed.

### Privacy-friendly may be overstated because VK/Depth still leak body and behavior cues.
- Evidence strength: conceptual/diagnostic
- Available evidence: Qualitative exposure table for 5 representations; run privacy leakage probe for identity-leakage evidence.
- Safe response: Use reduced visual exposure and RGB-free inference; explicitly state no anonymity or formal privacy guarantee.
- Paper action: Add privacy exposure and residual risk table; avoid privacy-preserving wording.

## 5. 可以直接写入论文的安全表述

> VK-RMD does not claim explicit raw-space alignment or formal privacy preservation. Instead, it learns a task-supervised body-latent token space under reduced visual exposure. VK and depth provide low-appearance structural anchors, while LiDAR, mmWave and WiFi-CSI provide complementary physical evidence. Full-to-Subset Guidance regularizes randomly missing-modality students with full-modality teacher behavior, but its role is auxiliary and task-dependent rather than the sole source of performance gains.

中文：

> VK-RMD 不主张显式 raw-space 对齐或形式化隐私保护。它学习的是 reduced visual exposure 条件下的任务监督 body-latent token 空间。VK 和 Depth 提供低外观结构锚点，LiDAR、mmWave 和 WiFi-CSI 提供互补物理证据。Full-to-Subset Guidance 使用 full-modality teacher 行为约束随机缺失模态 student，但其作用是辅助且任务相关的，而不是唯一性能来源。
