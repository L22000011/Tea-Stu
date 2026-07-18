# VK Semantic-Anchor Diagnostic

## Purpose

This diagnostic evaluates whether VK contributes structure-explicit semantics rather than acting as a replaceable coordinate feature. It should be used as supplementary evidence, not as a claim that VK is the only semantic center.

## Key Numbers

- HPE clean MPJPE: 45.05 mm.
- HPE VK joint shuffle: +10.10 mm MPJPE.
- HPE VK coordinate noise at 0.3: +4.90 mm MPJPE.
- HPE Depth noise at 0.3: +160.54 mm MPJPE.
- HAR clean accuracy: 99.75%.
- HAR VK joint shuffle: 1.19 percentage-point accuracy drop.
- HAR mmWave noise at 0.3: 10.31 percentage-point accuracy drop.

## Conclusion

VK joint shuffling degrades both HPE and HAR, indicating that the model uses VK joint identity and skeleton organization as structure-explicit cues. However, the strongest degradation is caused by Depth corruption for HPE and mmWave corruption for HAR. Therefore, the evidence supports the safer conclusion that VK acts as a structure-explicit semantic anchor, while task-specific physical sensors provide critical complementary evidence.

## Recommended Paper Wording

The semantic perturbation diagnostic shows that disrupting VK joint identity degrades both HPE and HAR, suggesting that VK is not used merely as raw coordinates but contributes structure-explicit body semantics. At the same time, the stronger degradation under Depth corruption for HPE and mmWave corruption for HAR indicates task-dependent sensor complementarity. We therefore describe VK as a semantic anchor in the body-latent space, not as the sole semantic center.

## Claims to Avoid

- Do not claim that VK is the only semantic center.
- Do not claim that VK is always the dominant modality.
- Do not claim that VK is noise-free or privacy-preserving.
- Do not claim that this diagnostic provides a formal proof of semantic alignment.
