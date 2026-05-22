# Hypothesis Log

Each entry records the reasoning, goal, and outcome of one distillation hypothesis.
Update the Results section after each run completes.

---

## H1 — Multi-Teacher Distillation (BiomedCLIP + RadCLIP)

**Date**: 2026-04-28
**Script**: `experiments/generated/distill_multiteacher_vits16.py`

**Hypothesis**:
RadCLIP is trained on radiology report–image pairs and encodes domain structure
that BiomedCLIP (trained on broad biomedical figures) does not emphasise.
Forcing the student to simultaneously match both teachers' image embedding spaces
gives it richer radiology-specific supervision, which should improve CheXpert/NIH14 AUROC.

**Method**:
- Student has two projection heads: one to BiomedCLIP space, one to RadCLIP space.
- Loss = w_img_primary * cosine(BiomedCLIP) + w_img_secondary * cosine(RadCLIP) + w_oa + w_mc
- OA/MC losses use BiomedCLIP text encoder only (RadCLIP text encoder not loaded).
- Probe eval uses primary (BiomedCLIP-space) head.

**Goal**: CheXpert macro AUROC > single-teacher baseline

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC: 
- NIH14 AUROC: 
- DeepLesion: 
- PMC-QA MC acc: 
- Run dir: 

---

## H2 — Hard Negative Queue (MoCo-style)

**Date**: 2026-04-28
**Script**: `experiments/generated/distill_hardneg_vits16.py`

**Hypothesis**:
In-batch negatives (B=64) are too few and too easy for medical images, where many
images share similar visual appearance. A FIFO queue of K=4096 teacher text embeddings
from previous steps provides much harder negatives for the OA KL loss, forcing the
student to learn more discriminative features.

**Method**:
- Maintains a 4096-slot circular text embedding queue.
- OA KL loss computed on expanded (B+K, D) negative set instead of (B, D).
- Queue enqueued after each OA backward pass.

**Goal**: CheXpert macro AUROC > single-teacher baseline (H0)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC: 
- NIH14 AUROC: 
- DeepLesion: 
- PMC-QA MC acc: 
- Run dir: 

---

## H4 — COSMOS-Med: Text-Guided Adaptive Crop Distillation

**Date**: 2026-05-19
**Script**: `experiments/generated/distill_cosmos_vits16.py`

**Hypothesis**:
Medical image-text pairs vary in alignment quality (precise captions vs. boilerplate). Three
complementary signals can improve the student beyond L_itc + L_mc:
(1) EMA self-distillation (L_cosmos) stabilises training by providing a slowly-evolving target;
(2) local-global patch distillation (L_lg) forces the student to match teacher representations of
    random crops from its patch tokens, closing the DINO-style global→local gap;
(3) text-guided adaptive crop (L_crop + TGAC) selects the most text-relevant student patches via
    cross-modal scoring and aligns them with sentence-level teacher text embeddings.
Uncertainty-aware λ (teacher image-text cosine similarity, stop-grad) down-weights L_lg and
L_crop for noisy/weakly-aligned pairs.

**Method**:
- Student: ViT-S/16 + proj_cls + proj_patch, both Linear(384→512)
- `forward_full()` exposes (cls B×512, patch_tok B×196×512)
- EMA teacher: deepcopy(student), momentum=0.999; updated at start of each step
- TGAC: top-K patches by cross-modal score (patch_proj · t_txt_cls); stop-grad on indices
- λ = (t_img · t_txt).sum(dim=-1).detach() — per-sample scalar, stop-grad
- L_itc: KL(student img-text logits, teacher img-text logits)
- L_cosmos: InfoNCE(s_cls, ema_cls)
- L_lg: num_crops random crops (scale 0.4–0.8); per crop: spatial_crop(student patch_tok) →
  InfoNCE vs teacher patch mean-pool of that crop; mean over crops
- L_crop: TGAC top-K regions vs K nltk-sentences of caption; λ-weighted cosine loss
- Total: L = L_itc + L_mc + L_cosmos + λ_mean·(w_lg·L_lg) + w_crop·L_crop_λ

**Goal**: NIH14 AUROC > 71.89, CheXpert AUROC > 69.87

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC: 
- NIH14 AUROC: 
- DeepLesion: 
- PMC-QA MC acc: 
- Run dir: 

---

## H5 — AlignKD-COSMOS: Layer-Selective Attention Map Distillation

**Date**: 2026-05-21
**Script**: `experiments/generated/distill_alignkd_cosmos_vits16.py`

**Hypothesis**:
Align-KD (CVPR 2025) shows that in VLMs, adjacent-layer feature cosine similarity
is much higher in middle layers than in first/last layers — only a few layers undergo
significant representation change. We adapt this insight to medical ViT encoder
distillation: (1) compute three layer-change metrics on medical images to identify
high-change layers; (2) distil attention maps (not features) at those layers, which
avoids dimension mismatch (student 384-dim, teacher 768-dim) and aligns with Align-KD's
attention-level distillation philosophy.

**Method**:
- Analysis module (`experiments/analysis/layer_sim.py`) computes three metrics:
  (a) Per-layer adjacent CLS cosine similarity (same as Align-KD Fig 2a)
  (b) Linear CKA between student and teacher CLS at each layer (cross-model alignment)
  (c) CLS vs mean-patch normalised Euclidean distance (local/global divergence)
  → top_k layers with lowest (a) score are selected dynamically; figure saved to results/
- Persistent forward hooks on selected student + teacher blocks at training start
- L_layer = mean KL(teacher attn_mean_heads || student attn_mean_heads) over selected layers
  student attn: [B, 6, 197, 197] → mean heads → [B*N, N]
  teacher attn: [B, 12, 197, 197] → mean heads → [B*N, N]
- All other losses identical to H4 COSMOS-Med (L_itc, L_cosmos, L_lg, L_crop)
- Total: L = L_itc + L_cosmos + λ_mean·w_lg·L_lg + w_crop·L_crop + w_layer·L_layer

**Goal**: NIH14 AUROC > 71.89, CheXpert AUROC > 69.87 (surpass H4 COSMOS-Med)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC:
- NIH14 AUROC:
- DeepLesion:
- PMC-QA MC acc:
- Run dir:

---

## H6 — Semantic Slot Student (SemanticSlotStudent)

**Date**: 2026-05-21
**Script**: `experiments/generated/distill_slot_vits16.py`
**Experiment key**: `slot`

**Hypothesis**:
Medical images contain localised findings (nodules, infiltrates, lesions) that correspond
to distinct clinical concepts rather than a single holistic representation. K learnable
semantic slot tokens, each attending to different patches via cross-attention, can capture
these disentangled concepts. During training a text-conditioned sigmoid gate up-weights
slots relevant to the caption, forcing slots to specialise. At inference the gate is
removed and slots are mean-pooled, yielding a purely visual encoder.

**Method**:
- `SemanticSlotStudent`: ViT-S/16 + K slot tokens (nn.Parameter [K, D_in]) + MHA + gate_proj
- `forward_full(images, text_cls)` → (cls [B,D], patch_proj [B,N,D], slot_out [B,K,D_in], weights [B,K,N])
- `forward(images)` → uniform mean-pool of slots (no text, pure visual)
- L_img (cosine) + L_itc (KL logits) + w_slot_div × L_div (slot orthogonality loss)
- L_div = mean of off-diagonal entries in normalised slot gram matrix → prevents slot collapse
- Default: num_slots=8, num_heads=4, w_slot_div=0.1

**Goal**: NIH14 / CheXpert AUROC > barlow_patch baseline (71.81 / 69.87)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC:
- NIH14 AUROC:
- DeepLesion:
- PMC-QA MC acc:
- Run dir:

---

## H7 — Hierarchical Patch Pyramid Student (HierarchicalStudent)

**Date**: 2026-05-21
**Script**: `experiments/generated/distill_hier_vits16.py`
**Experiment key**: `hier`

**Hypothesis**:
Pathology manifests at multiple spatial scales: organ-level context (coarse 4×4 grid),
region-level structure (mid 7×7), and fine texture (full 14×14). A single global CLS
token discards scale information. Distilling teacher representations at each scale
independently, then fusing via learnable softmax weights, gives the student a richer
multi-scale feature that generalises better to diverse probe tasks.

**Method**:
- `HierarchicalStudent`: ViT-S/16 + proj_cls + proj_fine + proj_mid + proj_coarse + scale_logits [4]
- `forward_pyramid(images)` → (fused [B,D], cls_emb, fine_emb, mid_emb, coarse_emb)
- fused = softmax(scale_logits) · [cls, fine, mid, coarse]
- L_img on fused + L_itc on fused + w_scale × L_scale (mean cosine loss across three patch scales)
- L_scale = (L_fine + L_mid + L_coarse) / 3, each is cosine(scale_emb, t_img)
- Default: w_scale=1.0

**Goal**: NIH14 / CheXpert AUROC > barlow_patch baseline (71.81 / 69.87)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC:
- NIH14 AUROC:
- DeepLesion:
- PMC-QA MC acc:
- Run dir:

---

## H8 — Text-Gated Bottleneck Adapter (BottleneckStudent)

**Date**: 2026-05-21
**Script**: `experiments/generated/distill_tgba_vits16.py`
**Experiment key**: `tgba`

**Hypothesis**:
A compact bottleneck between the ViT backbone and the projection head acts as an
information filter. During training, a text-conditioned sigmoid gate selects which
bottleneck units activate for each caption, forcing the bottleneck to learn
text-relevant visual features. At inference the gate is absent and all units are
active, yielding a pure visual encoder with stronger semantic alignment in the
bottleneck space than a direct projection.

**Method**:
- `BottleneckStudent`: ViT-S/16 → bottleneck MLP (in_dim→neck_dim, GELU, LayerNorm) → proj_cls → [B,D]
- gate_proj: Linear(out_dim → neck_dim); gate = sigmoid(gate_proj(t_txt))
- `forward_train(images, text_cls)` → (cls_gated, patch_proj, gate); cls_gated uses neck * gate
- `forward(images)` → full bottleneck, no gate (inference)
- L_img + L_itc + w_sparsity × L_sparsity; L_sparsity = gate.mean() (L1 on gate)
- Default: bottleneck_dim=256, w_sparsity=0.01

**Goal**: NIH14 / CheXpert AUROC > barlow_patch baseline (71.81 / 69.87)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC:
- NIH14 AUROC:
- DeepLesion:
- PMC-QA MC acc:
- Run dir:

---

## H3 — Barlow Twins Cross-Covariance KD

**Date**: 2026-04-28
**Script**: `experiments/generated/distill_barlow_vits16.py`

**Hypothesis**:
Cosine distillation only aligns the first-order mean embedding direction. The (D×D)
cross-correlation matrix between student and teacher batch embeddings encodes feature
covariance structure — e.g., which embedding dimensions co-activate for the same pathology.
Enforcing that this matrix equals the identity (Barlow Twins style) forces the student to
preserve the teacher's feature diversity and reduces redundancy, improving linear probe AUROC.

**Method**:
- barlow_twins_kd(s_emb, t_emb, lam=5e-3) added on every batch.
- On-diagonal: each student dim aligns with corresponding teacher dim.
- Off-diagonal: student's extra inter-dim correlations are penalised.
- w_barlow=1.0 default; lam_barlow=5e-3.

**Goal**: CheXpert macro AUROC > single-teacher baseline (H0)

**Status**: Pending (not yet run)

**Results**:
- CheXpert AUROC: 
- NIH14 AUROC: 
- DeepLesion: 
- PMC-QA MC acc: 
- Run dir: 

