# Recent MMFi Missing-Modality Comparison

## Eligibility

- HPE compares VK-RMD with PTA on all seven shared non-visual subsets of Depth, LiDAR and WiFi-CSI.
- HAR compares VK-RMD with COMPASS on all seven shared non-visual subsets of Depth, LiDAR and mmWave.
- Both comparisons use the published MM-Fi/X-Fi random protocol, common tasks and metrics. RGB/VK-containing rows are excluded because the visual inputs are not equivalent.
- The competing papers do not publish sample-level manifests; therefore this is a published-protocol comparison, not a paired identical-manifest experiment.

## Results

- HPE versus PTA: VK-RMD wins 4/7 subsets on MPJPE and 5/7 on PA-MPJPE.
- HAR versus COMPASS: VK-RMD wins 5/7 subsets on top-1 accuracy.
- These are mixed rather than universal gains. LiDAR-dominant cases remain a visible weakness and must not be hidden.

## Paper-safe conclusion

Under the shared official MM-Fi random-split protocol and matched non-visual modality subsets, VK-RMD is competitive with recent missing-modality methods and achieves stronger results on most geometry-complementary subsets. The comparison does not establish universal superiority or exact sample-paired equivalence.
