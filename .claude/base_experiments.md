# Experiments

All CheXpert, NIH14, and DeepLesion numbers are macro AUROC (linear probe, frozen encoder).
PMC-QA MC is teacher-text-space multiple-choice accuracy (proxy for VLM alignment).

---

## Layer 0 — Baseline Probes

Run: `python tools/baseline_probe.py`

| Model | Type | Params | CheXpert AUROC | NIH14 AUROC | DeepLesion AUROC | Date |
|-------|------|--------|---------------|-------------|-----------------|------|
| ViT-S/16 (ImageNet, no distill) | student | 22M | - | - | - | - |
| BiomedCLIP ViT-B/16 | teacher | 86M | - | - | - | - |
| RadCLIP | teacher | - | - | - | - | - |
| OpenPMC-CLIP | teacher | - | - | - | - | - |
| PLIP | teacher | - | - | - | - | - |
| QuiltNet-B-32 | teacher | - | - | - | - | - |

---

## Layer 1 — Hypothesis Experiments

| Run ID | Script | Hypothesis | CheXpert AUROC | NIH14 AUROC | DeepLesion AUROC | PMC-QA MC | vs Baseline | Date |
|--------|--------|------------|---------------|-------------|-----------------|-----------|-------------|------|
| H1 | distill_multiteacher_vits16.py | BiomedCLIP + RadCLIP dual-teacher | - | - | - | - | - | - |
| H2 | distill_hardneg_vits16.py | MoCo text-queue hard negatives (K=4096) | - | - | - | - | - | - |
| H3 | distill_barlow_vits16.py | Barlow Twins cross-covariance KD | - | - | - | - | - | - |

---

## Notes

- Baseline = best distilled ViT-S/16 result found in `results/` for each dataset.
- Teacher probes are the upper-bound reference; surpassing them at ViT-S/16 scale is the goal.
- All hypothesis runs checkpoint to `results/<run_name>/best.pt` (CheXpert-tracked) and `last.pt`.

### Baseline Run — 2026-05-11 23:23
Run dir: `baseline_20260511_224812`

| Model | Type | CheXpert AUROC | NIH14 AUROC | DeepLesion AUROC |
|-------|------|---------------|-------------|-----------------|
| ViT-S/16 (ImageNet, no distill) | student | 70.68 | 72.81 | 92.60 |
| BiomedCLIP ViT-B/16 | teacher | 72.86 | 75.20 | 94.58 |
| RadCLIP | teacher | 75.94 | 74.18 | 93.98 |
| PLIP | teacher | 68.21 | 67.63 | 91.77 |
| QuiltNet-B-32 | teacher | 66.89 | 65.67 | 90.75 |

### Baseline Run — 2026-05-19 15:16
Run dir: `baseline_20260519_130708`

| Model | Type | Chex AUROC | Chex F1 | Chex Recall | NIH14 AUROC | NIH F1 | NIH Recall | DL AUROC | DL F1 | DL Recall | CM AUROC | CM F1 | CM Recall |
|-------|------|------------|---------|-------------|-------------|--------|------------|----------|-------|-----------|----------|-------|-----------|
| ViT-S/16 (ImageNet, no distill) | student | 65.88 | 11.76 | 10.96 | 66.66 | 0.00 | 0.00 | 91.95 | 33.94 | 28.68 | 66.41 | 0.00 | 0.00 |
| BiomedCLIP ViT-B/16 | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| PubMedCLIP ViT-B/32 | teacher | 65.77 | 10.25 | 9.01 | 66.38 | 0.00 | 0.00 | 92.45 | 40.80 | 38.12 | 66.11 | 0.00 | 0.00 |
| RadCLIP | teacher | 73.99 | 19.79 | 18.92 | 68.74 | 0.45 | 0.24 | 92.50 | 46.23 | 41.54 | 66.84 | 0.15 | 0.07 |
| PLIP | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| QuiltNet-B-32 | teacher | 63.81 | 6.02 | 5.80 | 64.28 | 0.00 | 0.00 | 88.70 | 11.60 | 10.37 | 64.05 | 0.00 | 0.00 |

### Baseline Run — 2026-05-22 18:06
Run dir: `baseline_20260522_180629`

| Model | Type | Chex AUROC | Chex F1 | Chex Recall | NIH14 AUROC | NIH F1 | NIH Recall | DL AUROC | DL F1 | DL Recall | CM AUROC | CM F1 | CM Recall |
|-------|------|------------|---------|-------------|-------------|--------|------------|----------|-------|-----------|----------|-------|-----------|
| ViT-S/16 (ImageNet, no distill) | student | - | - | - | - | - | - | - | - | - | - | - | - |
| BiomedCLIP ViT-B/16 | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| PubMedCLIP ViT-B/32 | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| RadCLIP | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| PLIP | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
| QuiltNet-B-32 | teacher | - | - | - | - | - | - | - | - | - | - | - | - |
