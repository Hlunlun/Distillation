# Results

## Experiments

|time|run|method|student|teacher|dataset|macro_auroc|
|---|---|---|---|---|---:|---:|
|2026-05-11 23:29:46|distill_multiteacher_vits16_20260511_232946|KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|-|CheXpert|67.43|
|2026-05-11 23:29:46|distill_multiteacher_vits16_20260511_232946|KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|-|NIH14|67.37|
|2026-05-12 07:22:59|distill_hardneg_vits16_20260512_072259|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|68.12|
|2026-05-12 07:22:59|distill_hardneg_vits16_20260512_072259|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|70.08|
|2026-05-12 08:03:54|distill_barlow_vits16_20260512_080354|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.57|
|2026-05-12 08:03:54|distill_barlow_vits16_20260512_080354|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.07|
|2026-05-12 09:50:57|distill_pmc_vits16_20260512_095057|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|67.38|

## SOTA Comparison (Linear Probe AUROC)

|資料集|我用的模型|BiomedCLIP-PubMedBERT_256-|RadCLIP|open-pmc-clip|medclip-vit-base-patch16|llava-med|
|---|---|---|---|---|---|---|
|NIH14|71.07 (distill_barlow_vits16_20260512_080354)|-|-|-|-|-|
|CheXpert|69.57 (distill_barlow_vits16_20260512_080354)|-|-|-|-|-|

## Notes

- Baseline columns stay `-` until you run probes for those teacher models (separate baseline eval runs).
- `我用的模型` is the best distilled run found under `results/` for each dataset.
