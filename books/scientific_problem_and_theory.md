# Scientific Problem and Theoretical Formulation

This document formulates the scientific problem behind VK-RMD and maps the formulation to the current codebase. It is written for manuscript use, but it deliberately separates implemented mechanisms from conceptual extensions.

The central position of this work should be stated carefully:

> VK-RMD studies reduced-visual-exposure multimodal human perception under arbitrary missing modalities. It uses visual keypoints (VK) and depth as low-appearance structural anchors, maps heterogeneous sensors into a shared body-latent token space, and trains missing-modality students through full-to-subset guidance from a full-modality teacher.

It should not be described as formal privacy protection, a raw RGB teacher, or a conventional large-teacher-to-small-student compression method.

---

## 1. Core Scientific Question

### 1.1 One-sentence Scientific Question

How can a multimodal human perception model use low-appearance structural anchors to align heterogeneous sensor streams and remain effective when any subset of modalities is available, without relying on raw RGB at inference?

### 1.2 Why This Is a Scientific Problem

Multimodal human perception is difficult because the available signals are heterogeneous at three levels.

First, the raw spaces are incompatible. VK is a sparse joint coordinate set, depth is a dense geometric map, LiDAR and mmWave are point-based reflections, and WiFi-CSI is a wireless channel tensor. These signals cannot be aligned by forcing them into a common raw representation.

Second, the information content differs by task. VK and depth carry low-appearance body structure, LiDAR and mmWave provide physical spatial evidence, and WiFi-CSI provides indirect propagation cues. HPE is geometry-sensitive, while HAR can rely more on motion and coarse body dynamics. Therefore, one modality cannot be assumed to dominate all tasks.

Third, real deployment cannot assume full sensor availability. A fixed full-modality model can fail when one or more sensors are unavailable. The scientific question is therefore not only how to fuse modalities, but how to learn a representation that remains usable over the modality-availability space.

### 1.3 Problem Definition

For HPE, the implemented modality set is

$$
\mathcal{M}_{hpe}=\{v,d,l,r,w\},
$$

where \(v\) is VK, \(d\) is depth, \(l\) is LiDAR, \(r\) is mmWave radar, and \(w\) is WiFi-CSI.

For HAR, the implemented modality set is

$$
\mathcal{M}_{har}=\{v,d,l,r\}.
$$

For a task \(t\in\{\mathrm{hpe},\mathrm{har}\}\), a sample is

$$
(\{x_m\}_{m\in\mathcal{M}_t}, y^t),
$$

where \(y^{hpe}\) is a 3D pose target and \(y^{har}\) is an action class label.

At inference, the model may receive any non-empty subset

$$
S\subseteq\mathcal{M}_t,\qquad S\neq\emptyset.
$$

The objective is to learn a predictor

$$
\hat{y}^t_S=F^t_\theta(\{x_m\}_{m\in S}, S),
$$

such that it remains accurate over many possible \(S\), rather than only under \(S=\mathcal{M}_t\).

In the current implementation, HPE and HAR are implemented as separate task-specific models that share the same paradigm. A single joint HPE+HAR network is conceptual/future extension unless the Super-Teacher branch is explicitly used as exploratory material.

### 1.4 What This Work Does Not Claim

This work should not claim the following:

- It does not provide formal privacy guarantees such as differential privacy, cryptographic protection, anonymity, or identity leakage prevention.
- It does not prove that VK is the only semantic center. The current diagnostic evidence supports VK as a structure-explicit anchor, while depth and mmWave can be more task-sensitive for HPE and HAR.
- It does not align WiFi-CSI, LiDAR, mmWave, depth, and VK in raw space. Alignment is induced in a learned body-latent token space.
- It does not implement a raw RGB teacher in the current mainline. The active HPE/HAR models are RGB-free after offline VK extraction.
- It is not conventional large-to-small model compression. Teacher and student can share architecture; the key asymmetry is modality availability.
- It should not claim that reliability weights are calibrated physical sensor reliabilities. They are learned contribution proxies.

---

## 2. Methodological Principle

### 2.1 VK and Depth as Low-Appearance Structural Anchors

The current method should be framed around low-appearance structural anchoring rather than strict privacy.

VK is represented as a structured skeleton:

$$
G_v=(V,E,X),
$$

where \(V\) is the set of 17 joints, \(E\) is the COCO-style bone graph, and \(X\in\mathbb{R}^{17\times 2}\) contains 2D keypoint coordinates.

The VK branch is not an RGB image backbone. The implemented VK encoder computes

$$
J_v=\phi_c(\mathrm{Norm}(X))+\phi_p(2\mathrm{Norm}(X)-1)+e_{joint},
$$

then applies bone-graph mixing:

$$
J_v^{(k+1)}=\mathrm{BoneGraphMixer}(J_v^{(k)}, A_{bone}),
$$

and projects joint tokens into a fixed number of body-latent tokens:

$$
Z_v=P_v(J_v)\in\mathbb{R}^{K\times D}.
$$

In the current code, \(K=32\) and \(D=512\) by default.

Depth should be described as a complementary low-appearance structural modality: it preserves spatial geometry while suppressing texture, clothing color, face appearance, and background details more than RGB. The paper can therefore argue for a low-appearance structural anchor family, with VK providing explicit joint semantics and depth providing dense geometric evidence.

### 2.2 Body-Latent Tokenization of Heterogeneous Modalities

For every modality \(m\), the model first extracts modality-specific features:

$$
H_m=E_m(x_m).
$$

The features are then projected into a common token dimension:

$$
Z_m=P_m(H_m)\in\mathbb{R}^{K\times D}.
$$

For VK, \(E_v\) is the skeleton encoder and \(P_v\) is its token projector. For non-VK sensors, \(E_m\) is the corresponding pretrained sensor backbone and \(P_m\) is a trainable token projector.

Each modality token receives a modality embedding:

$$
\bar{Z}_m=Z_m+e_m.
$$

For an available subset \(S\), the model concatenates only the available modality tokens:

$$
U_S=\mathrm{Concat}_{m\in S}(\bar{Z}_m),
$$

and applies a Transformer encoder:

$$
T_S=\Phi(U_S).
$$

This is the implemented body-latent alignment mechanism. It is not raw-space alignment. The shared token dimension, modality embeddings, Transformer interaction, and common HPE/HAR supervision induce a task-oriented latent alignment.

### 2.3 Reliability-Guided Fusion as Learned Contribution Estimation

The encoded tokens are split back into modality chunks:

$$
T_S=\{T_m\}_{m\in S}.
$$

For HPE, each modality chunk produces a pose expert and a log-variance score:

$$
(\hat{p}_m,s_m)=H_{pose}(T_m).
$$

For HAR, each chunk produces a logit expert and a log-variance score:

$$
(\hat{c}_m,s_m)=H_{cls}(T_m).
$$

In the default uncertainty mode, contribution weights are computed as

$$
\alpha_m=
\frac{\exp(-s_m/\tau)}
{\sum_{j\in S}\exp(-s_j/\tau)}.
$$

The model also supports uniform and attention modes in code. The manuscript should call \(\alpha_m\) a learned reliability proxy or learned contribution weight, not true physical sensor reliability.

The fused body token is

$$
z_S=\sum_{m\in S}\alpha_m\,\mathrm{Pool}(T_m).
$$

HPE uses a weighted expert pose and a residual pose:

$$
\hat{p}_{exp}=\sum_{m\in S}\alpha_m\hat{p}_m,\qquad
\hat{p}_{res}=H_{res}(z_S),
$$

$$
\hat{p}_S=0.5\hat{p}_{exp}+0.5\hat{p}_{res}.
$$

HAR analogously uses weighted expert logits and residual logits:

$$
\hat{c}_{exp}=\sum_{m\in S}\alpha_m\hat{c}_m,\qquad
\hat{c}_{res}=H_{res}(z_S),
$$

$$
\hat{c}_S=0.5\hat{c}_{exp}+0.5\hat{c}_{res}.
$$

---

## 3. Full-to-Subset Guidance

### 3.1 Why This Is Not Conventional Large-to-Small Distillation

Conventional knowledge distillation usually transfers knowledge from a larger, stronger teacher to a smaller student:

$$
F_T^{large}(x)\rightarrow F_S^{small}(x).
$$

The current method is different. Teacher and student may use the same architecture. The asymmetry is not primarily parameter count, but input availability:

$$
F_T(\mathcal{M})\rightarrow F_S(S),\qquad S\subseteq \mathcal{M}.
$$

The teacher observes the full modality set and learns complete sensing behavior. The student observes randomly sampled subsets and learns to approximate the teacher when sensors are missing. Therefore, a more accurate name is Full-to-Subset Guidance (FSG), not conventional model-compression distillation.

### 3.2 Full-Modality Teacher

The teacher receives all configured modalities:

$$
S_T=\mathcal{M}_t.
$$

For HPE:

$$
(\hat{p}_T,z_T,\alpha_T)=F_T^{hpe}(\{x_m\}_{m\in\mathcal{M}_{hpe}}).
$$

The implemented teacher objective is

$$
\mathcal{L}^{T}_{hpe}
=
\mathcal{L}_{pose}(\hat{p}_T,y)
+\lambda_{bone}\mathcal{L}_{bone}(\hat{p}_T,y).
$$

For HAR:

$$
(\hat{c}_T,z_T,\alpha_T)=F_T^{har}(\{x_m\}_{m\in\mathcal{M}_{har}}).
$$

The implemented teacher objective is

$$
\mathcal{L}^{T}_{har}
=
\mathrm{CE}(\hat{c}_T,y)
+\lambda_{unc}\mathcal{L}_{unc}.
$$

During student training, the teacher is set to evaluation mode and is queried under `torch.no_grad()`.

### 3.3 Random Missing-Modality Student

At each training step, the student samples an available subset:

$$
S\sim q(S),\qquad S\subseteq\mathcal{M}_t,\qquad S\neq\emptyset.
$$

The implemented sampler chooses a random number of dropped modalities from `drop_counts`, removes that many modalities, and guarantees at least one remaining modality. Thus, missing-modality training is not merely evaluation-time stress testing; it is part of the training objective.

The student prediction is

$$
(\hat{y}_S,z_S,\alpha_S)=F_S^t(\{x_m\}_{m\in S},S).
$$

### 3.4 Guidance Loss

For HPE, the implemented student loss is

$$
\mathcal{L}^{S}_{hpe}
=
\mathcal{L}_{gt}
+\lambda_{out}\mathcal{L}_{out}
+\lambda_{tok}\mathcal{L}_{tok}
+\lambda_{bone}\mathcal{L}_{bone}^{KD}
+\lambda_{rel}\mathcal{L}_{rel}
+\lambda_{unc}\mathcal{L}_{unc}.
$$

The components are:

$$
\mathcal{L}_{gt}=\mathcal{L}_{pose}(\hat{p}_S,y),
$$

$$
\mathcal{L}_{out}=\mathcal{L}_{pose}(\hat{p}_S,\mathrm{stopgrad}(\hat{p}_T)),
$$

$$
\mathcal{L}_{tok}=\|z_S-\mathrm{stopgrad}(z_T)\|_2^2,
$$

$$
\mathcal{L}_{bone}^{KD}
=
\mathcal{L}_{bone}(\hat{p}_S,\mathrm{stopgrad}(\hat{p}_T)),
$$

and

$$
\mathcal{L}_{rel}
=
\mathrm{KL}\left(
\alpha_S\;||\;\mathrm{Renorm}(\alpha_T|_S)
\right).
$$

Here, \(\alpha_T|_S\) means the teacher's full-modality contribution vector restricted to the student-visible modalities and renormalized.

For HAR, the implemented student loss is

$$
\mathcal{L}^{S}_{har}
=
\mathrm{CE}(\hat{c}_S,y)
+\lambda_{kd}T^2
\mathrm{KL}\left(
\sigma(\hat{c}_T/T)\;||\;\sigma(\hat{c}_S/T)
\right)
+\lambda_{tok}\mathcal{L}_{tok}
+\lambda_{rel}\mathcal{L}_{rel}
+\lambda_{unc}\mathcal{L}_{unc}.
$$

The code implements output/logit guidance, token guidance, fusion-weight guidance, and uncertainty regularization. It does not implement a raw RGB teacher, nor does it implement a single unified HPE+HAR multitask loss in the current mainline.

### 3.5 Overall Objective

The teacher objective is

$$
\min_{\theta_T}
\mathbb{E}_{(x,y)}
\left[
\mathcal{L}^{T}
\left(
F_{\theta_T}(\{x_m\}_{m\in\mathcal{M}_t}),y
\right)
\right].
$$

After the teacher is trained, student optimization is

$$
\min_{\theta_S}
\mathbb{E}_{(x,y)}
\mathbb{E}_{S\sim q(S)}
\left[
\mathcal{L}^{S}
\left(
F_{\theta_S}(\{x_m\}_{m\in S},S),
y,
F_{\theta_T}(\{x_m\}_{m\in\mathcal{M}_t})
\right)
\right].
$$

This is the most compact formal statement of Full-to-Subset Guidance.

---

## 4. Training and Inference Pipeline

### 4.1 Training Procedure

The implemented procedure is:

1. Extract or load VK keypoints offline. Raw RGB is not used by the active HPE/HAR model.
2. Train a full-modality teacher with all configured modalities.
3. Freeze/evaluate the teacher during student training.
4. For each student batch, sample a modality subset \(S\).
5. Query the teacher with full modalities and the student with subset modalities.
6. Optimize task loss plus implemented guidance losses.
7. Save `best.pth`, `last.pth`, periodic checkpoints, final summaries, and evaluation CSVs according to each project script.

### 4.2 Inference Procedure

At inference, the model receives only the available subset \(S\). The same forward interface is used:

$$
\hat{y}_S=F_\theta(\{x_m\}_{m\in S},S).
$$

The model does not need raw RGB at inference. If VK is unavailable, non-visual student variants can be evaluated with depth/LiDAR/mmWave/WiFi-CSI for HPE and depth/LiDAR/mmWave for HAR.

### 4.3 Missing-Modality Evaluation

The intended evaluation protocol is all non-empty modality subsets:

$$
\mathcal{P}(\mathcal{M}_t)\setminus \{\emptyset\}.
$$

For HPE, this gives 31 combinations. For HAR, this gives 15 combinations. Student-VK evaluates combinations containing or not containing VK depending on the script/configuration, while Student-NV focuses on non-visual subsets.

Cross-subject and cross-scene results should be reported as generalization diagnostics. Cross-scene degradation should be treated as a limitation, not as solved deployment robustness.

---

## 5. Why the Method Should Work

The method is expected to work for five reasons.

First, VK and depth reduce appearance dependence. They preserve body structure and geometry while removing much of the raw RGB texture, face, clothing, and background information.

Second, body-latent tokenization avoids impossible raw-space alignment. The model does not require WiFi-CSI to become point clouds or LiDAR to become skeleton coordinates. Instead, each modality is projected into a common token dimension and trained toward the same human state.

Third, random subset training turns modality absence into a training condition. Each sampled subset is treated as a valid sensing configuration, not only as regularization noise.

Fourth, full-to-subset guidance transfers full-modality behavior to partial-modality inputs. This matters because the teacher can learn a more complete sensor-fusion behavior, while the student must approximate it under missing evidence.

Fifth, learned contribution weights allow the model to adapt to different subsets. These weights should be interpreted as task-level contribution proxies. They support robustness when paired with all-combination evaluation and uniform-fusion ablations, but they should not be overinterpreted as calibrated sensor reliability.

---

## 6. Safe Contribution Wording

Recommended contribution wording:

1. We formulate reduced-visual-exposure multimodal HPE/HAR as a missing-modality learning problem over arbitrary sensor subsets.
2. We introduce a VK-centered body-latent tokenization framework that maps VK, depth, LiDAR, mmWave, and WiFi-CSI into a shared task-oriented token space.
3. We propose Full-to-Subset Guidance, where a full-modality teacher guides a randomly missing-modality student through output, token, structure, and fusion-level supervision.
4. We evaluate the framework under all modality combinations, Student-VK and Student-NV settings, and cross-subject/cross-scene protocols.
5. We provide diagnostic analysis showing that VK contributes joint-structure semantics, while depth and mmWave provide task-specific physical evidence.

Safer title direction:

> Learning with Low-Appearance Structural Anchors for Missing-Modality Multimodal Human Perception

or

> VK-RMD: Visual-Keypoint Anchored Missing-Modality Learning for Reduced-Exposure Multimodal Human Perception

---

## 7. Risky Claims and Recommended Downgrades

| Risky claim | Why risky | Recommended downgrade |
|---|---|---|
| VK guarantees privacy | VK can still leak body shape, gait, and action habits | VK reduces visual exposure; no formal privacy guarantee |
| VK is the only semantic center | Diagnostics show depth/mmWave can dominate task-specific degradation | VK is a structure-explicit semantic anchor, complemented by physical sensors |
| Reliability weights are true sensor reliability | The code learns task contribution proxies, not calibrated physical quality | Learned reliability proxy / learned contribution weight |
| Distillation is the main reason for all gains | HPE no-KD can be competitive in existing results | Full-to-Subset Guidance is auxiliary and task-dependent |
| Raw RGB teacher distills image knowledge into VK student | Not implemented in current mainline | Current mainline uses VK/non-RGB teacher and student after offline VK extraction |
| One model jointly solves HPE and HAR | Current HPE/HAR implementations are separate | Same paradigm is instantiated on HPE and HAR |
| Cross-scene deployment is solved | Cross-scene HPE degrades substantially | Missing-modality robustness does not imply full domain generalization |
| New attention/dropout/KD algorithm | Components have precedents | Novelty lies in VK/body-latent formulation, full-to-subset training, and systematic HPE/HAR validation |

---

## 8. Code Evidence Map

| Claim | Code evidence | What it supports | Evidence strength |
|---|---|---|---|
| VK is encoded as a structured skeleton, not an RGB image | `HPE/models/skeleton_prompt_encoder.py`, `HAR/models/skeleton_prompt_encoder.py`; `SkeletonPromptEncoder`, `BoneGraphMixer`, `COCO17_BONES` | VK uses joint embeddings and bone graph mixing | Strong |
| HPE uses VK, depth, LiDAR, mmWave, WiFi-CSI | `HPE/utils/modality.py`, `ALL_MODALITIES` | Implemented HPE modality set | Strong |
| HAR uses VK, depth, LiDAR, mmWave | `HAR/utils/modality.py`, `ALL_MODALITIES` | Implemented HAR modality set | Strong |
| Raw RGB is not an active mainline modality | `LEGACY_TO_CANONICAL` maps `rgb` to `vk`; active model builders use VK encoder and sensor extractors | RGB-free active HPE/HAR model after VK extraction | Strong |
| Non-VK sensors are projected into common token size | `TokenProjector` in `HPE/models/vk_rcd.py` and `HAR/models/vk_rcd_har.py` | Body-latent tokenization | Strong |
| Modality identity is encoded | `ModalityTokenEncoder.modality_embedding` in reliability fusion modules | Tokens retain modality source after projection | Strong |
| Cross-modal interaction is token-space Transformer interaction | `nn.TransformerEncoder` in `ModalityTokenEncoder` | Learned token-space alignment | Strong |
| Reliability fusion produces learned contribution weights | `ReliabilityFusion.forward`, `ClassificationReliabilityFusion.forward` | `alphas` from uncertainty, attention, or uniform modes | Strong |
| Teacher sees full modalities during student training | `teacher_output = teacher(batch["inputs"], teacher_modalities)` in HPE/HAR `training/engine.py` | Full-modality teacher query | Strong |
| Student sees random missing subsets | `sample_missing_modalities(...)` in HPE/HAR `training/engine.py` and `utils/modality.py` | Subset training | Strong |
| Teacher is frozen/eval during student training | `teacher.eval()` and `with torch.no_grad()` in training engines | Guidance without teacher update | Strong |
| HPE Full-to-Subset Guidance includes output, token, bone, reliability losses | `HPE/losses/distill_losses.py` | Implemented HPE FSG terms | Strong |
| HAR Full-to-Subset Guidance includes logit, token, reliability losses | `HAR/losses/distill_losses.py` | Implemented HAR FSG terms | Strong |
| VK semantic-anchor evidence is diagnostic, not proof | `supplement/vk_semantic_anchor_summary.csv`, `supplement/vk_semantic_anchor_analysis/vk_semantic_anchor_conclusion.md` | VK perturbation affects performance but is not sole center | Moderate |
| Raw RGB teacher to VK student | Not implemented in current HPE/HAR mainline | Historical/conceptual only unless a separate branch is used | Not implemented |
| Single SuperTeacher for HPE+HAR | `Super-Teacher` branch exists as exploratory work | Not part of mainline evidence | Exploratory |

---

## Manuscript-ready Core Paragraph

VK-RMD reorganizes multimodal human perception around low-appearance structural anchors rather than raw RGB. VK provides explicit joint identity and skeleton topology, while depth contributes texture-suppressed spatial geometry. Heterogeneous sensors such as LiDAR, mmWave, and WiFi-CSI are not aligned in their raw spaces; instead, each modality is encoded by a modality-specific backbone, projected into a shared body-latent token space, and fused through learned contribution weights under the currently available subset. A full-modality teacher learns complete sensing behavior, and a randomly missing-modality student is optimized through Full-to-Subset Guidance, including task, output/logit, token, structure, and fusion-level terms where implemented. This formulation addresses reduced visual exposure, heterogeneous sensor alignment, and arbitrary modality availability in a single training objective, while avoiding claims of formal privacy guarantees or calibrated physical sensor reliability.
