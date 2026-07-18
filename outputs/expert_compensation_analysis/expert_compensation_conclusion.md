# Expert Compensation Analysis

This diagnostic evaluates whether degraded VK is compensated by non-VK modality experts.
It does not claim calibrated physical sensor reliability.

## How to read

- `fusion_gain_vs_vk > 0` means the final fused output is better than the VK expert.
- For HPE, lower MPJPE is better; gains are `VK expert MPJPE - fused MPJPE`.
- For HAR, higher accuracy is better; gains are `fused Acc - VK expert Acc`.
- `alpha_quality_pearson` measures whether larger weights tend to align with better expert quality.

## Key rows

- HAR `vk+depth` under `vk_noise_0.6`: final=0.896875, vk_expert=0.896875, best_non_vk=0.896875, fusion_gain_vs_vk=0.000000, alpha_vk=0.4790, alpha_quality_pearson=0.0040.
- HAR `vk+depth` under `vk_joint_shuffle`: final=0.841250, vk_expert=0.806875, best_non_vk=0.857500, fusion_gain_vs_vk=0.034375, alpha_vk=0.4643, alpha_quality_pearson=0.0110.
- HAR `vk+mmwave` under `vk_noise_0.6`: final=0.829375, vk_expert=0.791250, best_non_vk=0.843125, fusion_gain_vs_vk=0.038125, alpha_vk=0.3927, alpha_quality_pearson=0.0122.
- HAR `vk+mmwave` under `vk_joint_shuffle`: final=0.609375, vk_expert=0.524375, best_non_vk=0.725625, fusion_gain_vs_vk=0.085000, alpha_vk=0.4972, alpha_quality_pearson=-0.0795.
- HAR `vk+depth+mmwave` under `vk_noise_0.6`: final=0.996250, vk_expert=0.995000, best_non_vk=0.996250, fusion_gain_vs_vk=0.001250, alpha_vk=0.3184, alpha_quality_pearson=0.0122.
- HAR `vk+depth+mmwave` under `vk_joint_shuffle`: final=0.986250, vk_expert=0.973125, best_non_vk=0.990625, fusion_gain_vs_vk=0.013125, alpha_vk=0.3130, alpha_quality_pearson=0.0194.
- HAR `vk+depth+lidar+mmwave` under `vk_noise_0.6`: final=0.994375, vk_expert=0.995000, best_non_vk=0.994375, fusion_gain_vs_vk=-0.000625, alpha_vk=0.2417, alpha_quality_pearson=0.0245.
- HAR `vk+depth+lidar+mmwave` under `vk_joint_shuffle`: final=0.985000, vk_expert=0.973125, best_non_vk=0.991875, fusion_gain_vs_vk=0.011875, alpha_vk=0.2347, alpha_quality_pearson=0.0169.

## Safe paper statement

When VK is perturbed, the framework should be interpreted as compensating degraded structural cues through complementary physical modalities if the fused output remains better than the VK expert. The learned weights are task-level contribution proxies, not calibrated sensor-quality measurements.
