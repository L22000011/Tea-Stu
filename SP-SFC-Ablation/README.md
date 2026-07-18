# SP-SFC loss ablation

The four profiles isolate the full-to-subset objective without changing the
formal VK-RMD architecture:

* `task_only`: ground-truth task loss only;
* `output`: output/logit guidance only;
* `output_token`: output/logit plus token guidance;
* `full`: the current structural, fusion, and uncertainty terms.

The default quick setting is 10 epochs, 300 training batches, and 100
evaluation batches per profile. Results are written under
`outputs/sp_sfc_ablation/` and do not overwrite formal outputs.

Cloud command:

```bash
cd /apps/users/icps_intelligence/data/lyg/X-Fi
CUDA_VISIBLE_DEVICES=1 python -u SP-SFC-Ablation/run_sp_sfc_ablation.py \
  --dataset /apps/users/icps_intelligence/data/lyg/code/MMFi_DATA \
  --device cuda:0 --tasks HPE HAR \
  --epochs 10 --max-train-batches 300 --max-eval-batches 100
```

Each profile is isolated, resumable from `last.pth`, and skipped when its
`all_combinations.csv` already exists. `--dry-run` prints all eight commands
without running them.
