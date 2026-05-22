from __future__ import annotations

import copy
import math
import random

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.data_loaders import BatchOA, BatchQA
from experiments.models import FrozenTeacher, kl_logits, CosmosStudent, TGAC


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_PATCH_SIZE = 16
_GRID_SIZE  = 14


# ── helpers ───────────────────────────────────────────────────────────────────

def infonce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    B = a.shape[0]
    logits = a @ b.T / temp
    labels = torch.arange(B, device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


@torch.no_grad()
def _find_proj_linear(visual: nn.Module) -> nn.Linear | None:
    """Traverse open_clip TimmModel to find the final Linear projection layer."""
    for attr in ("proj", "head"):
        m = getattr(visual, attr, None)
        if isinstance(m, nn.Linear):
            return m
        if isinstance(m, nn.Sequential):
            for child in m.modules():
                if isinstance(child, nn.Linear):
                    return child
    return None


@torch.no_grad()
def _teacher_patch_features(
    teacher: FrozenTeacher, images: torch.Tensor
) -> torch.Tensor | None:
    """[B, 196, D_clip] patch tokens in CLIP space; None if unsupported."""
    visual = getattr(teacher.model, "visual", None)
    trunk  = getattr(visual, "trunk", None)
    if trunk is None or not hasattr(trunk, "forward_features"):
        return None
    feats      = trunk.forward_features(images)    # [B, N+1, D_trunk]
    patches    = feats[:, 1:, :].float()           # [B, 196, D_trunk]
    proj_layer = _find_proj_linear(visual)
    if proj_layer is None:
        return None                                # cannot project; caller falls back
    patches = F.linear(patches, proj_layer.weight, proj_layer.bias)  # [B, 196, D_clip]
    return patches


def _make_crop_params(
    height: int,
    width: int,
    scale: tuple[float, float] = (0.4, 0.8),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> tuple[int, int, int, int]:
    """Sample (top, left, h, w) crop in [0, height] × [0, width] pixel space."""
    area = height * width
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        log_r = random.uniform(math.log(ratio[0]), math.log(ratio[1]))
        w_c = int(round(math.sqrt(target_area * math.exp(log_r))))
        h_c = int(round(math.sqrt(target_area / math.exp(log_r))))
        if 0 < w_c <= width and 0 < h_c <= height:
            top  = random.randint(0, height - h_c)
            left = random.randint(0, width  - w_c)
            return top, left, h_c, w_c
    h_c = int(height * 0.6)
    w_c = int(width  * 0.6)
    return (height - h_c) // 2, (width - w_c) // 2, h_c, w_c


def _spatial_crop(
    patch_proj: torch.Tensor,
    top: int, left: int, h: int, w: int,
) -> torch.Tensor:
    """Mean-pool student patch tokens that fall inside the crop region.
    patch_proj: [B, 196, D] normalised; returns [B, D] normalised."""
    row_min = top  // _PATCH_SIZE
    row_max = min((top  + h - 1) // _PATCH_SIZE, _GRID_SIZE - 1)
    col_min = left // _PATCH_SIZE
    col_max = min((left + w - 1) // _PATCH_SIZE, _GRID_SIZE - 1)
    indices = [
        r * _GRID_SIZE + c
        for r in range(row_min, row_max + 1)
        for c in range(col_min, col_max + 1)
    ]
    if not indices:
        return patch_proj.mean(dim=1)
    selected = patch_proj[:, indices, :]        # [B, N_sel, D]
    return F.normalize(selected.mean(dim=1), dim=-1)   # [B, D]


def _pad_sentences(sents: list[str], k: int) -> list[str]:
    if not sents:
        return [""] * k
    while len(sents) < k:
        sents = sents + [sents[-1]]
    return sents[:k]


# ── init ─────────────────────────────────────────────────────────────────────

def get_probe_model(student: CosmosStudent) -> CosmosStudent:
    return student


def init(shared_args, exp_args, device):
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

    def _ema_update() -> None:
        m = exp_args.ema_momentum
        for ep, sp in zip(ema.parameters(), student.parameters()):
            ep.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        _ema_update()

        images = batch.images.to(device, non_blocking=True)
        B      = images.shape[0]

        s_cls, s_patch = student.forward_full(images)    # [B, D], [B, 196, D]
        t_img          = teacher.encode_image(images)    # [B, D]

        with torch.no_grad():
            ema_cls, _ = ema.forward_full(images)        # [B, D]

        l_img    = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()
        l_cosmos = infonce(s_cls, ema_cls, temp=args.temp)

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)  # [B, D]

            logits_t = (t_img @ t_txt.t()) / args.temp
            logits_s = (s_cls @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)

            # λ: teacher image-text alignment as per-sample confidence (stop-grad)
            lam     = (t_img * t_txt).sum(dim=-1).detach().clamp(0.0, 1.0)  # [B]
            lam_mean = lam.mean()

            # L_lg: local-global patch distillation over num_crops random crops
            l_lg_list = []
            for _ in range(exp_args.num_crops):
                top, left, h, w = _make_crop_params(224, 224)
                s_region   = _spatial_crop(s_patch, top, left, h, w)        # [B, D]
                local_imgs = F.interpolate(
                    images[:, :, top:top+h, left:left+w],
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False,
                    antialias=False,
                )
                t_patches  = _teacher_patch_features(teacher, local_imgs)
                if t_patches is not None:
                    t_local = F.normalize(t_patches.mean(dim=1), dim=-1)
                else:
                    t_local = teacher.encode_image(local_imgs)
                l_lg_list.append(infonce(s_region, t_local, temp=args.temp))
            l_lg = torch.stack(l_lg_list).mean()

            # L_crop: TGAC top-K regions vs sentence-level teacher text
            K = exp_args.tgac_k
            sentences_per_sample = [
                _pad_sentences(_SENT_SPLIT.split(cap.strip()) or [cap], K)
                for cap in batch.captions
            ]
            # encode all K*B texts in one call
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
            )
            return {
                "loss/total":       total,
                "loss/img":         l_img,
                "loss/itc":         l_itc,
                "loss/cosmos":      l_cosmos,
                "loss/lg":          l_lg,
                "loss/crop":        l_crop,
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
                "loss/total":  total,
                "loss/img":    l_img,
                "loss/mc":     l_mc,
                "loss/ce":     l_ce,
                "data/is_qa":  torch.ones(()),
            }

    return student, teacher, compute_loss
