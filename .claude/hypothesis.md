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
