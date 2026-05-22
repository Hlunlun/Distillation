from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from experiments.data_loaders import (
    BatchOA, BatchQA, PMCOADataset, collate_oa,
)
from experiments.models import FrozenTeacher, kl_logits, CosmosStudent, TGAC
from experiments.generated.distill_cosmos_vits16 import (
    infonce, _teacher_patch_features, _make_crop_params, _spatial_crop, _pad_sentences,
)
from experiments.analysis.layer_sim import analyze_layer_similarity, make_attn_hook


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_teacher_trunk(teacher: FrozenTeacher):
    visual = getattr(teacher.model, "visual", None)
    return getattr(visual, "trunk", None) if visual is not None else None


def _attn_kl_loss(
    s_buf: dict[int, torch.Tensor],
    t_buf: dict[int, torch.Tensor],
    selected_layers: list[int],
    device: str,
) -> torch.Tensor:
    """Mean KL(teacher_attn || student_attn) over selected layers, heads averaged."""
    if not selected_layers or not t_buf:
        return torch.zeros((), device=device)
    losses = []
    for idx in selected_layers:
        if idx not in s_buf or idx not in t_buf:
            continue
        s_a = s_buf[idx].float()                              # [B, H_s, N, N]
        t_a = t_buf[idx].float()                              # [B, H_t, N, N]
        s_mean = s_a.mean(dim=1).reshape(-1, s_a.shape[-1])  # [B*N, N]
        t_mean = t_a.mean(dim=1).reshape(-1, t_a.shape[-1])  # [B*N, N]
        losses.append(
            F.kl_div(
                (s_mean + 1e-8).log(),
                t_mean.detach(),
                reduction="batchmean",
            )
        )
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def _tqva_loss(
    t_txt:   torch.Tensor,   # [B, D] — text query, L2-normalised, CLIP space, no grad
    t_patch: torch.Tensor,   # [B, N, D] — teacher patches, CLIP space, no grad
    s_patch: torch.Tensor,   # [B, N, D] — student patches, CLIP space, has grad
) -> torch.Tensor:
    """Text-Query-Vision Attention distillation: KL(teacher cross-attn map || student cross-attn map).
    Both maps computed in shared CLIP space — no projection adapter needed."""
    D = s_patch.shape[-1]
    # cross-attention score: text query × patch keys / sqrt(D)
    attn_t = (t_txt.unsqueeze(1) @ t_patch.float().transpose(1, 2)).squeeze(1).div(D ** 0.5).softmax(dim=-1)  # [B, N]
    attn_s = (t_txt.unsqueeze(1) @ s_patch.float().transpose(1, 2)).squeeze(1).div(D ** 0.5).softmax(dim=-1)  # [B, N]
    return F.kl_div((attn_s + 1e-8).log(), attn_t.detach(), reduction="batchmean")


# ── probe model ───────────────────────────────────────────────────────────────

def get_probe_model(student: CosmosStudent) -> CosmosStudent:
    return student


# ── init ─────────────────────────────────────────────────────────────────────

def init(shared_args, exp_args, device):
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    teacher = FrozenTeacher(
        shared_args.teacher_model,
        pretrained=shared_args.teacher_pretrained,
        load_tokenizer=True,
    )
    teacher.model.to(device)

    student = CosmosStudent(shared_args.timm_student, out_dim=teacher.embed_dim).to(device)

    ema = copy.deepcopy(student)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.to(device)

    tgac = TGAC(top_k=exp_args.tgac_k)

    # ── layer-change analysis ─────────────────────────────────────────────────
    teacher_trunk = _get_teacher_trunk(teacher)
    n_blocks = len(student.backbone.blocks)
    selected_layers: list[int] = [0, n_blocks - 1]   # default: first and last

    if teacher_trunk is not None:
        ds = PMCOADataset(
            image_dir=shared_args.pmc_oa_image_dir,
            jsonl_path=shared_args.pmc_oa_jsonl,
            max_samples=exp_args.analysis_n_images,
        )
        analysis_loader = DataLoader(
            ds,
            batch_size=min(exp_args.analysis_n_images, 64),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda ex: collate_oa(ex, teacher.preprocess),
        )
        analysis_images = next(iter(analysis_loader)).images.to(device)

        save_path = Path(shared_args.log_dir) / "analysis" / "layer_sim.png"
        selected_layers, fig = analyze_layer_similarity(
            student.backbone,
            teacher_trunk,
            analysis_images,
            top_k=exp_args.top_k_layers,
            save_path=save_path,
        )
        plt.close(fig)
        print(f"[AlignKD-COSMOS] selected layers: {selected_layers}  |  figure: {save_path}")
    else:
        print(f"[AlignKD-COSMOS] No visual.trunk found; using default layers: {selected_layers}")

    # ── register persistent attention hooks for training ──────────────────────
    s_attn_buf: dict[int, torch.Tensor] = {}
    t_attn_buf: dict[int, torch.Tensor] = {}

    for idx in selected_layers:
        student.backbone.blocks[idx].attn.register_forward_hook(
            make_attn_hook(s_attn_buf, idx, detach=False)
        )
        if teacher_trunk is not None:
            teacher_trunk.blocks[idx].attn.register_forward_hook(
                make_attn_hook(t_attn_buf, idx, detach=True)
            )

    # ── EMA update ────────────────────────────────────────────────────────────
    def _ema_update() -> None:
        m = exp_args.ema_momentum
        for ep, sp in zip(ema.parameters(), student.parameters()):
            ep.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    # ── compute_loss ──────────────────────────────────────────────────────────
    def compute_loss(student, teacher, batch, args, exp_args, device):
        _ema_update()

        images = batch.images.to(device, non_blocking=True)
        B = images.shape[0]

        # forward calls trigger the registered hooks → fills s_attn_buf, t_attn_buf
        s_cls, s_patch = student.forward_full(images)    # [B, D], [B, 196, D]
        t_img          = teacher.encode_image(images)    # [B, D]

        with torch.no_grad():
            ema_cls, _ = ema.forward_full(images)        # [B, D]

        l_img    = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()
        l_cosmos = infonce(s_cls, ema_cls, temp=args.temp)
        l_layer  = _attn_kl_loss(s_attn_buf, t_attn_buf, selected_layers, device)

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)  # [B, D]

            logits_t = (t_img @ t_txt.t()) / args.temp
            logits_s = (s_cls @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)

            lam      = (t_img * t_txt).sum(dim=-1).detach().clamp(0.0, 1.0)  # [B]
            lam_mean = lam.mean()

            # L_tqva: text-query-vision attention distillation (CLIP space, no adapter)
            t_patch_clip = _teacher_patch_features(teacher, images)  # [B, 196, D] or None
            l_tqva = (
                _tqva_loss(t_txt, t_patch_clip, s_patch)
                if t_patch_clip is not None
                else torch.zeros((), device=device)
            )

            # L_lg: local-global patch distillation
            l_lg_list = []
            for _ in range(exp_args.num_crops):
                top, left, h, w = _make_crop_params(224, 224)
                s_region   = _spatial_crop(s_patch, top, left, h, w)
                local_imgs = TF.resized_crop(images, top, left, h, w, [224, 224])
                t_patches  = _teacher_patch_features(teacher, local_imgs)
                t_local    = (
                    F.normalize(t_patches.mean(dim=1), dim=-1)
                    if t_patches is not None
                    else teacher.encode_image(local_imgs)
                )
                l_lg_list.append(infonce(s_region, t_local, temp=args.temp))
            l_lg = torch.stack(l_lg_list).mean()

            # L_crop: TGAC top-K regions vs sentence-level teacher text
            K = exp_args.tgac_k
            sentences_per_sample = [
                _pad_sentences(nltk.sent_tokenize(cap), K)
                for cap in batch.captions
            ]
            flat_sents = [sentences_per_sample[b][k] for k in range(K) for b in range(B)]
            t_sents    = teacher.encode_text(flat_sents).view(K, B, -1)     # [K, B, D]

            region, _  = tgac(s_patch, t_txt)                               # [B, D]
            cos_kb     = (t_sents.detach() * region.unsqueeze(0)).sum(dim=-1)  # [K, B]
            l_crop_raw = (1.0 - cos_kb).mean(dim=0)                         # [B]
            l_crop     = (lam * l_crop_raw).mean()

            total = (
                args.w_img * l_img
                + l_itc
                + l_cosmos
                + lam_mean * exp_args.w_lg * l_lg
                + exp_args.w_crop * l_crop
                + exp_args.w_layer * l_layer
                + lam_mean * exp_args.w_tqva * l_tqva
            )
            return {
                "loss/total":       total,
                "loss/img":         l_img,
                "loss/itc":         l_itc,
                "loss/cosmos":      l_cosmos,
                "loss/lg":          l_lg,
                "loss/crop":        l_crop,
                "loss/layer":       l_layer,
                "loss/tqva":        l_tqva,
                "data/lambda_mean": lam_mean,
                "data/is_qa":       torch.zeros(()),
            }

        else:
            option_texts = []
            for q, ch in zip(batch.questions, batch.choices):
                option_texts.extend([f"{q} {c}" for c in ch])
            t_opt    = teacher.encode_text(option_texts).view(B, 4, -1)
            logits_t = torch.einsum("bd,bkd->bk", t_img, t_opt) / args.tau_mc
            logits_s = torch.einsum("bd,bkd->bk", s_cls, t_opt) / args.tau_mc
            l_mc     = kl_logits(logits_s, logits_t)
            l_ce     = (
                F.cross_entropy(logits_s, batch.labels.to(device), ignore_index=-1)
                if args.w_ce > 0.0 else torch.zeros((), device=device)
            )
            total = args.w_img * l_img + l_mc + args.w_ce * l_ce
            return {
                "loss/total": total,
                "loss/img":   l_img,
                "loss/mc":    l_mc,
                "loss/ce":    l_ce,
                "data/is_qa": torch.ones(()),
            }

    return student, teacher, compute_loss
