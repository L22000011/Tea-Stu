# Reviewer Risk Notes

## 1. VK 创新与 X-Fi 的区分
风险：审稿人可能认为只是将 RGB 换成 VK 并重新训练。
应对：主文强调 V 替代下游 RGB、身份泄露探针、FSG full-to-subset 可用性约束，以及逐组合缺失评估。

## 2. FSG 证据边界
风险：HPE w/o distillation 结果竞争力强，不能声称 teacher guidance 是唯一收益来源。
应对：写成完整模态行为约束和缺失子集训练范式；对 HAR 有辅助收益，对 HPE 不做统一提升承诺。

## 3. Reliability proxy 解释
风险：权重被误读为真实传感器可靠性。
应对：统一写 learned modality contribution proxy；corruption 只作为诊断，不作为强证明。

## 4. 隐私友好表述
风险：VK/Depth 仍可能泄露体型、步态、动作习惯。
应对：仅使用 reduced visual exposure / RGB-free inference；明确非 formal privacy。

## 5. 表格密度
风险：正文表格多，可能显得结果堆叠。
应对：每张表绑定一条主线：V 替代 RGB 或 FSG 缺失模态，而不是散放。
