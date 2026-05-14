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
