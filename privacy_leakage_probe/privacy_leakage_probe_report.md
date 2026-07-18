# Privacy Leakage Probe

This lightweight probe evaluates whether subject identity can be predicted from each representation.
It is a diagnostic privacy-leakage test, not a formal privacy guarantee.

## Interpretation

- Accuracy near chance suggests weak subject leakage under this probe.
- Accuracy clearly above chance means the representation still carries identity-related cues.
- Even low leakage does not prove anonymity; it only supports reduced visual exposure.

## Results

- rgb: status=done, split=action-holdout, subject_test_acc=0.675290, chance=0.025000 (2.50%). High subject leakage; only reduced visual exposure can be claimed.
- vk: status=done, split=action-holdout, subject_test_acc=0.154326, chance=0.025000 (2.50%). Measurable subject leakage; avoid privacy guarantee claims.
- depth: status=done, split=action-holdout, subject_test_acc=0.438002, chance=0.025000 (2.50%). High subject leakage; only reduced visual exposure can be claimed.
- lidar: status=done, split=action-holdout, subject_test_acc=0.365745, chance=0.025000 (2.50%). High subject leakage; only reduced visual exposure can be claimed.
- mmwave: status=done, split=action-holdout, subject_test_acc=0.097235, chance=0.025000 (2.50%). Measurable subject leakage; avoid privacy guarantee claims.
- wifi-csi: status=done, split=action-holdout, subject_test_acc=0.086530, chance=0.025000 (2.50%). Measurable subject leakage; avoid privacy guarantee claims.