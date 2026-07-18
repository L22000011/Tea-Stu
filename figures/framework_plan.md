# Figure 1 Plan: VK-RMD Framework for IoT Smart Spaces

**Caption:** Overview of the proposed VK-RMD framework for privacy-friendly and missing-modality robust human sensing in IoT smart spaces. RGB is used only as training-time privileged information. Visual keypoints provide a structured privacy-friendly bridge, while the deployed Student receives available non-RGB sensing modalities and performs reliability-guided fusion for HPE and HAR.

**Panel design:**

1. **Training-only privileged branch:** RGB image -> VK structured representation -> RGB-VK privileged teacher.
2. **Deployable IoT sensing branch:** Depth, LiDAR, mmWave, WiFi-CSI, and optional VK stream enter modality-specific encoders.
3. **Missing-modality mask:** indicate unavailable sensors using gray blocks.
4. **Reliability-guided fusion:** show modality-wise candidate predictions and adaptive weights.
5. **Task heads:** HPE head and HAR head.
6. **Deployment statement:** inference uses no raw RGB.

**Data required:** architectural diagram only; no additional experimental data required.

