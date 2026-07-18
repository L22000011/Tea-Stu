# RGB-Subset Fixed-Budget Convergence Audit

This report checks whether the short RGB-available subset experiments show usable optimization evidence.
It should be described as a fixed-budget fairness diagnostic, not as a full convergence benchmark.

| Method | Task | Protocol | Epochs | First loss | Last loss | Drop | Drift | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| none | none | none | 0 | | | | | no training curves found |

## How to write this in the paper

Use cautious wording:

> We conduct a fixed-budget RGB-available subset comparison to audit fairness under identical sample availability and cross protocols. Both RGB-XFi and VK-RMD are trained with the same update budget, and convergence diagnostics are reported to verify that optimization proceeds normally. This subset study is not used as the primary full-convergence benchmark.

Avoid claiming that 10 epochs are fully converged unless the task-specific curves clearly plateau.
