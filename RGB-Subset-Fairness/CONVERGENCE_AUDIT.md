# Convergence Audit for RGB-Subset Quick Runs

The `--quick` setting is a fixed-budget fairness diagnostic, not a full
convergence benchmark.

After the two pipelines finish, run:

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi
python -u RGB-Subset-Fairness/audit_rgb_subset_convergence.py
```

Outputs:

```bash
tables/rgb_subset_convergence_audit.csv
reports/rgb_subset_convergence_audit.md
```

Recommended paper wording:

> We conduct a fixed-budget RGB-available subset comparison under identical
> sample availability, cross protocols, and update budgets. The convergence
> audit is reported to verify that the short adaptation runs optimize normally.
> This subset study is not used as the primary full-convergence benchmark.

Reviewer-facing interpretation:

- Do not claim that 10 epochs are fully converged.
- Claim that both RGB-XFi and VK-RMD are compared under the same limited
  adaptation budget.
- Use the audit table to show whether losses decrease and whether the last
  epochs are stable or still improving.
- If a curve is still improving, describe the experiment as a time-limited
  fairness check, while keeping the full-MMFi VK-RMD experiments as the main
  evidence.
