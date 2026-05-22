# Results

## Experiments

|time|run|method|student|teacher|dataset|macro_auroc|acc|macro_f1|macro_recall|
|---|---|---|---|---|---|---:|---:|---:|---:|
|2026-05-11 23:29:46|distill_multiteacher_vits16_20260511_232946|KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|-|CheXpert|67.43|-|-|-|
|2026-05-11 23:29:46|distill_multiteacher_vits16_20260511_232946|KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|-|DeepLesion|93.22|-|-|-|
|2026-05-11 23:29:46|distill_multiteacher_vits16_20260511_232946|KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|-|NIH14|67.37|-|-|-|
|2026-05-12 07:22:59|distill_hardneg_vits16_20260512_072259|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|68.12|-|-|-|
|2026-05-12 07:22:59|distill_hardneg_vits16_20260512_072259|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.20|-|-|-|
|2026-05-12 07:22:59|distill_hardneg_vits16_20260512_072259|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|70.08|-|-|-|
|2026-05-12 08:03:54|distill_barlow_vits16_20260512_080354|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.57|-|-|-|
|2026-05-12 08:03:54|distill_barlow_vits16_20260512_080354|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.01|-|-|-|
|2026-05-12 08:03:54|distill_barlow_vits16_20260512_080354|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.07|-|-|-|
|2026-05-12 09:50:57|distill_pmc_vits16_20260512_095057|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|67.38|-|-|-|
|2026-05-12 09:50:57|distill_pmc_vits16_20260512_095057|KD-IMG+KD-OA+KD-MC (qa=0.3, warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|92.76|-|-|-|
|2026-05-14 12:32:55|barlow_20260514_123255|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.84|-|-|-|
|2026-05-14 12:32:55|barlow_20260514_123255|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.36|-|-|-|
|2026-05-14 12:32:55|barlow_20260514_123255|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.77|-|-|-|
|2026-05-14 12:54:46|multiteacher_20260514_125446|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|68.99|-|-|-|
|2026-05-14 12:54:46|multiteacher_20260514_125446|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|70.67|-|-|-|
|2026-05-14 23:06:55|hardneg_20260514_230655|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.55|-|-|-|
|2026-05-14 23:06:55|hardneg_20260514_230655|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.50|-|-|-|
|2026-05-14 23:06:55|hardneg_20260514_230655|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.89|-|-|-|
|2026-05-16 15:41:21|barlow_20260516_154121|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.84|86.41|13.40|12.64|
|2026-05-16 15:41:21|barlow_20260516_154121|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|70.42|94.75|0.15|0.08|
|2026-05-16 15:41:21|barlow_20260516_154121|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.36|93.33|50.33|46.51|
|2026-05-16 15:41:21|barlow_20260516_154121|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.77|94.79|0.31|0.16|
|2026-05-17 23:27:04|hardneg_20260517_232704|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.48|86.34|13.06|12.23|
|2026-05-17 23:27:04|hardneg_20260517_232704|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|70.07|94.75|0.06|0.03|
|2026-05-17 23:27:04|hardneg_20260517_232704|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.80|94.79|0.21|0.11|
|2026-05-17 23:28:51|multiteacher_20260517_232851|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|68.99|86.31|13.09|12.40|
|2026-05-17 23:28:51|multiteacher_20260517_232851|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|69.10|94.74|0.03|0.01|
|2026-05-17 23:28:51|multiteacher_20260517_232851|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|70.67|94.80|0.29|0.15|
|2026-05-18 10:31:22|barlow_patch_20260518_103122|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.87|86.38|13.34|12.59|
|2026-05-18 10:31:22|barlow_patch_20260518_103122|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|70.47|94.75|0.18|0.09|
|2026-05-18 10:31:22|barlow_patch_20260518_103122|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.34|93.31|50.13|46.28|
|2026-05-18 10:31:22|barlow_patch_20260518_103122|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.81|94.79|0.33|0.17|
|2026-05-18 13:19:40|barlow_20260518_131940|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|CheXpert|69.00|86.24|14.13|14.23|
|2026-05-18 13:19:40|barlow_20260518_131940|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|ChestMNIST|63.42|94.74|0.18|0.09|
|2026-05-18 13:19:40|barlow_20260518_131940|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|DeepLesion|92.60|92.77|44.82|42.31|
|2026-05-18 13:19:40|barlow_20260518_131940|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|NIH14|64.98|94.78|0.52|0.28|
|2026-05-18 14:49:59|barlow_patch_20260518_144959|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|CheXpert|69.00|86.24|14.13|14.23|
|2026-05-18 14:49:59|barlow_patch_20260518_144959|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|ChestMNIST|63.42|94.74|0.18|0.09|
|2026-05-18 14:49:59|barlow_patch_20260518_144959|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|DeepLesion|92.61|92.77|44.78|42.23|
|2026-05-18 14:49:59|barlow_patch_20260518_144959|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-pth-clip:zluvolyote/RadCLIP|NIH14|64.98|94.78|0.52|0.28|
|2026-05-18 22:07:19|barlow_patch_20260518_220719|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.87|86.38|13.34|12.59|
|2026-05-18 22:07:19|barlow_patch_20260518_220719|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|70.47|94.75|0.18|0.09|
|2026-05-18 22:07:19|barlow_patch_20260518_220719|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.34|93.31|50.13|46.28|
|2026-05-18 22:07:19|barlow_patch_20260518_220719|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.81|94.79|0.33|0.17|
|2026-05-18 23:24:52|barlow_patch_20260518_232452|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|CheXpert|69.87|86.38|13.34|12.59|
|2026-05-18 23:24:52|barlow_patch_20260518_232452|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|ChestMNIST|70.47|94.75|0.18|0.09|
|2026-05-18 23:24:52|barlow_patch_20260518_232452|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|DeepLesion|93.34|93.31|50.13|46.28|
|2026-05-18 23:24:52|barlow_patch_20260518_232452|KD-IMG+KD-OA+KD-MC (warmup=0.15)|vit_small_patch16_224|hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224|NIH14|71.81|94.79|0.33|0.17|

## SOTA Comparison (Linear Probe AUROC)

|資料集|我用的模型|BiomedCLIP-PubMedBERT_256-|RadCLIP|open-pmc-clip|medclip-vit-base-patch16|llava-med|
|---|---|---|---|---|---|---|
|NIH14|71.89 (hardneg_20260514_230655)|-|-|-|-|-|
|CheXpert|69.87 (barlow_patch_20260518_103122)|-|-|-|-|-|
|DeepLesion|93.50 (hardneg_20260514_230655)|-|-|-|-|-|
|ChestMNIST|70.47 (barlow_patch_20260518_103122)|-|-|-|-|-|

## Notes

- Baseline columns stay `-` until you run probes for those teacher models.
- `我用的模型` is the best distilled run found under `results/` for each dataset.
