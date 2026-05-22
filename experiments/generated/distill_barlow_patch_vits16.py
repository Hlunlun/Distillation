from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, kl_logits


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def barlow_twins_kd(s_emb: torch.Tensor, t_emb: torch.Tensor, lam: float) -> torch.Tensor:
    B = s_emb.shape[0]
    eps = 1e-5
    Z_s = (s_emb - s_emb.mean(0)) / (s_emb.std(0) + eps)
    Z_t = (t_emb - t_emb.mean(0)) / (t_emb.std(0) + eps)
    C = Z_s.T @ Z_t / B
    on_diag = (C.diagonal() - 1.0).pow(2).sum()
    off_diag = C.pow(2).sum() - C.diagonal().pow(2).sum()
    return on_diag + lam * off_diag


class BarlowPatchStudent(nn.Module):
    """ViT student with CLS projection + patch projection for spatial distillation."""

    def __init__(self, timm_name: str, out_dim: int, patch_teacher_dim: int):
        super().__init__()
        import timm
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = int(self.backbone.num_features)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_patch = nn.Linear(in_dim, patch_teacher_dim, bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(images)), dim=-1)

    def forward_with_patches(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_emb [B, out_dim], patch_proj [B, N, patch_teacher_dim])."""
        all_tokens = self.backbone.forward_features(images)  # [B, N+1, in_dim]
        cls_emb = F.normalize(self.proj(all_tokens[:, 0, :]), dim=-1)
        patch_proj = self.proj_patch(all_tokens[:, 1:, :])   # [B, N, patch_teacher_dim]
        return cls_emb, patch_proj


def _get_teacher_patch_dim(teacher: FrozenTeacher) -> int:
    visual = getattr(teacher.model, "visual", None)
    trunk = getattr(visual, "trunk", None)
    if trunk is not None:
        return int(getattr(trunk, "num_features", teacher.embed_dim))
    return teacher.embed_dim


@torch.no_grad()
def _teacher_patch_features(teacher: FrozenTeacher, images: torch.Tensor) -> torch.Tensor | None:
    """Returns [B, N, D] patch token features from last ViT block of teacher.
    Only supported for open_clip TimmModel teachers (e.g., BiomedCLIP).
    """
    visual = getattr(teacher.model, "visual", None)
    trunk = getattr(visual, "trunk", None)
    if trunk is not None and hasattr(trunk, "forward_features"):
        feats = trunk.forward_features(images)  # [B, N+1, D]
        return feats[:, 1:, :]
    return None


def init(shared_args, exp_args, device):
    teacher = FrozenTeacher(shared_args.teacher_model, pretrained=shared_args.teacher_pretrained)
    teacher.model.to(device)

    patch_dim = _get_teacher_patch_dim(teacher)
    student = BarlowPatchStudent(
        shared_args.timm_student,
        out_dim=teacher.embed_dim,
        patch_teacher_dim=patch_dim,
    ).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)

        s_img, s_patches = student.forward_with_patches(images)
        t_img = teacher.encode_image(images)

        loss_img = (1.0 - (s_img * t_img).sum(dim=-1)).mean()
        loss_barlow = barlow_twins_kd(s_img, t_img.float(), exp_args.lam_barlow)

        t_patches = _teacher_patch_features(teacher, images)
        if t_patches is not None:
            s_patch_n = F.normalize(s_patches, dim=-1)
            t_patch_n = F.normalize(t_patches.float(), dim=-1)
            loss_patch = F.mse_loss(s_patch_n, t_patch_n)
        else:
            loss_patch = torch.zeros((), device=device)

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)
            logits_t = (t_img @ t_txt.t()) / args.temp
            logits_s = (s_img @ t_txt.t()) / args.temp
            loss_oa = kl_logits(logits_s, logits_t)
            total = (args.w_img * loss_img + args.w_oa * loss_oa
                     + exp_args.w_barlow * loss_barlow + exp_args.w_patch * loss_patch)
            return {
                "loss/total": total, "loss/img": loss_img, "loss/oa": loss_oa,
                "loss/barlow": loss_barlow, "loss/patch": loss_patch,
                "data/is_qa": torch.zeros(()),
            }
        else:
            option_texts = []
            for q, ch in zip(batch.questions, batch.choices):
                option_texts.extend([f"{q} {c}" for c in ch])
            t_opt = teacher.encode_text(option_texts).view(images.shape[0], 4, -1)
            logits_t = torch.einsum("bd,bkd->bk", t_img, t_opt) / args.tau_mc
            logits_s = torch.einsum("bd,bkd->bk", s_img, t_opt) / args.tau_mc
            loss_mc = kl_logits(logits_s, logits_t)
            loss_ce = (
                F.cross_entropy(logits_s, batch.labels.to(device), ignore_index=-1)
                if args.w_ce > 0.0 else torch.zeros((), device=device)
            )
            total = (args.w_img * loss_img + args.w_mc * loss_mc + args.w_ce * loss_ce
                     + exp_args.w_barlow * loss_barlow + exp_args.w_patch * loss_patch)
            return {
                "loss/total": total, "loss/img": loss_img, "loss/mc": loss_mc,
                "loss/ce": loss_ce, "loss/barlow": loss_barlow, "loss/patch": loss_patch,
                "data/is_qa": torch.ones(()),
            }

    return student, teacher, compute_loss
