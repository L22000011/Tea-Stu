# Paired Marginal Contribution of VK

## Definition

For every non-VK modality subset `S`, this analysis compares the same trained model under `S` and the exactly matched input set `S+VK`. Positive gain always means that adding VK improves the task metric.

## Main result

- HPE MPJPE: VK improves 15/15 matched subsets; mean reduction = 28.56 mm, median reduction = 2.22 mm.
- HAR Accuracy: VK improves 7/7 matched subsets; mean gain = 5.81 percentage points, median gain = 1.49 percentage points.

## Interpretation

The paired design controls the identity of the non-VK sensor subset: the only input-set difference within each pair is the addition of VK. Therefore, it is stronger evidence for the conditional task value of VK than comparing unrelated modality combinations or reporting an all-combination average.

The result does not prove that VK is a unique semantic center, that the pairs are statistically independent, or that VK provides formal privacy protection. The exact sign-consistency value in the summary CSV is descriptive because modality subsets share sensors and model parameters.

## Inputs

- HPE: `E:\Deskbook\Tea\HPE\outputs\eval\student_vk_random_all_combinations.csv`
- HAR: `E:\Deskbook\Tea\HAR\outputs\eval\student_vk_random_all_combinations.csv`
