# AutoResearch — Visual/Text Encoder Distillation

Self-driving research tool. Claude Code searches ArXiv, analyzes papers, generates hypotheses,
and writes PyTorch experiment code to surpass SOTA in vision-language distillation.
No Gemini dependency — all steps performed directly by Claude Code via WebSearch/WebFetch.

## Skills
- karpathy-guidelines
- 不要我問個問題就直接改code，先跟我講完我確認再改
- 講中文 Speak Chinese!


## Project Structure

```
config/
  paths.py               # Local data paths (PMC-OA, CheXpert, NIH14, DeepLesion)
tools/
  arxiv_search.py        # ArXiv API + PDF fetcher + SOTA model registry (for auto research)
  baseline_probe.py      # Standalone baseline linear probe evaluation
  prepare_chexpert_hf.py # Download and prepare CheXpert from HuggingFace
  summarize_results.py   # Summarize all runs under results/ into one table
  update_results_md.py   # Update results Markdown report
experiments/
  data_loaders.py        # Local dataset loaders + run_linear_probe()
  generated/             # Auto-generated experiment_*.py scripts
memory/
  papers/                # Cached paper JSON (auto-populated)
  hypotheses.json        # Saved hypotheses
  last_analysis.txt      # Cached literature analysis
reports/                 # Timestamped Markdown research reports
results/                 # Experiment run outputs (tb/, metrics.json, summary.json, best.pt)
```

## Pipeline (Claude Code-native, no Gemini)

1. **Literature Analysis** — Search ArXiv with WebSearch/WebFetch, read cached papers in `memory/papers/`
2. **Hypothesis Generation** — Claude reasons over papers, proposes novel approaches scored on novelty/feasibility/impact
3. **Experiment Generation** — Claude writes full PyTorch training scripts (see code rules below)
4. **Report** — Structured Markdown report with SOTA table, roadmap, risk assessment

## Key Models / Benchmarks Tracked

| Benchmark | What it measures |
|-----------|-----------------|
| ImageNet zero-shot Top-1 | Zero-shot visual classification |
| PMC-OA i2t/t2i R@1 | Cross-modal retrieval quality (student visual vs teacher text) |
| PathMNIST linear probe acc | Pathology linear probe |
| CheXpert linear probe macro AUROC | Chest X-ray multi-label AUROC (frozen visual encoder) |
| NIH ChestX-ray14 linear probe macro AUROC | Chest X-ray multi-label AUROC (frozen visual encoder) |
| DeepLesion linear probe macro AUROC | CT lesion type (8-class) AUROC (frozen visual encoder) |
| VQA-RAD closed accuracy | Radiology VQA |
| Params (M) | Efficiency / model size |

## Evaluation (all real, no placeholders)

- **Linear probe**: frozen image features + sklearn (single-label acc or multi-label macro AUROC) via `run_linear_probe()` in `experiments.data_loaders`
- **Retrieval**: ROCO image-text R@1 via cosine similarity on real embeddings
- **VQA**: zero-shot closed-ended on VQA-RAD (image vs answer-option text similarity)
- **Training curves**: TensorBoard SummaryWriter — loss per step, LR, eval metrics per epoch

## Comparison Set (Medical Visual Encoder Distillation)

### Teachers (visual+text space providers)

- BiomedCLIP: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- BioViL-T (CXR-focused VLM): `microsoft/BiomedVLP-BioViL-T`
- RadCLIP (radiology CLIP-style): `zluvolyote/RadCLIP`
- OpenPMC-CLIP (PMC-trained CLIP-style): `vector-institute/open-pmc-clip`
- MedCLIP (CXR-focused CLIP-style): `microsoft/medclip-vit-base-patch16`
- PLIP (pathology CLIP-style): `vinid/plip`
- QuiltNet (pathology CLIP-style): `wisdomik/QuiltNet-B-32`
- CONCH (pathology VLM, license-gated): `MahmoodLab/conch`
- UNI (pathology visual-only teacher): `MahmoodLab/uni`

### Students (visual encoder to deploy)

- ViT-S/16 (baseline student): `timm:vit_small_patch16_224`

### Methods To Compare (same student, same data, fair compute)

- Baseline: ImageNet-pretrained student, no distillation
- KD-IMG: image embedding distillation only (`loss_img`)
- KD-OA: KD-IMG + in-batch image-text logit distillation on PMC-OA (`loss_oa`)
- KD-MC: KD-IMG + multiple-choice distillation on PMC-QA (`loss_mc`)
- KD-OA+MC: KD-IMG + KD-OA + KD-MC (default starting point)
- KD-OA+MC+CE: add small supervised CE on PMC-QA answers (`w_ce > 0`)

## Current Experiment Scripts

- Distill (PMC-OA + PMC-QA) + eval probes — KD-OA+MC baseline:
  - `experiments/generated/distill_pmc_vits16.py`
- Barlow Twins-style cross-correlation distillation:
  - `experiments/generated/distill_barlow_vits16.py`
- Hard negative mining distillation:
  - `experiments/generated/distill_hardneg_vits16.py`
- Multi-teacher distillation (ensemble of medical teachers):
  - `experiments/generated/distill_multiteacher_vits16.py`
- Prepare CheXpert from Hugging Face into `images/ + csv` for linear probe:
  - `tools/prepare_chexpert_hf.py`

## Experiment Logging (always on)

Each experiment run writes to `results/<run_name>/`:

- `tb/` TensorBoard logs
- `metrics.json` per-epoch metrics (including probe outputs)
- `summary.json` best checkpoint pointer and best metric
- `best.pt` best student checkpoint (based on probe metric)

To summarize all runs under `results/` into one table, use:

```bash
python tools/summarize_results.py
```

CheXpert is prepared to `/mnt/data/vlmdata/CheXpert/` by default:

```bash
python tools/prepare_chexpert_hf.py --split train
```

## Code Generation Rules (STRICT — apply to all files in experiments/generated/)

1. No dummy/fake data — always use real datasets from `experiments.data_loaders`
2. No placeholders — if evaluation logic is not implemented, implement it; never use `random.uniform`
3. All imports at the top of the file, no inline imports
4. No try/except to hide errors — if something fails, fix the root cause
5. Minimal comments — only when WHY is non-obvious; no docstring blocks, no emojis
6. TensorBoard — every experiment uses `SummaryWriter` for loss, LR, and eval metrics
7. Checkpoint saving — save best model by eval metric after each epoch
8. Real evaluation — run `run_linear_probe()` + ROCO retrieval R@1 after training



