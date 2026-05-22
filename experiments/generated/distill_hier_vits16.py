from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, kl_logits, HierarchicalStudent


def infonce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    B = a.shape[0]
    logits = a @ b.T / temp
    labels = torch.arange(B, device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def get_probe_model(student: HierarchicalStudent) -> HierarchicalStudent:
    return student


def init(shared_args, exp_args, device):
    teacher = FrozenTeacher(
        shared_args.teacher_model,
        pretrained=shared_args.teacher_pretrained,
        load_tokenizer=True,
    )
    teacher.model.to(device)

    student = HierarchicalStudent(
        shared_args.timm_student,
        out_dim=teacher.embed_dim,
    ).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        B      = images.shape[0]

        fused, cls_emb, fine_emb, mid_emb, coarse_emb = student.forward_pyramid(images)
        t_img = teacher.encode_image(images)   # [B, D]

        # Direct cosine distillation for each scale + fused
        l_img    = (1.0 - (fused     * t_img).sum(dim=-1)).mean()
        l_fine   = (1.0 - (fine_emb  * t_img).sum(dim=-1)).mean()
        l_mid    = (1.0 - (mid_emb   * t_img).sum(dim=-1)).mean()
        l_coarse = (1.0 - (coarse_emb * t_img).sum(dim=-1)).mean()
        l_scale  = (l_fine + l_mid + l_coarse) / 3.0

        if isinstance(batch, BatchOA):
            t_txt    = teacher.encode_text(batch.captions)
            logits_t = (t_img  @ t_txt.t()) / args.temp
            logits_s = (fused  @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)

            total = args.w_img * l_img + l_itc + exp_args.w_scale * l_scale
            return {
                "loss/total":   total,
                "loss/img":     l_img,
                "loss/itc":     l_itc,
                "loss/scale":   l_scale,
                "loss/fine":    l_fine,
                "loss/mid":     l_mid,
                "loss/coarse":  l_coarse,
                "data/is_qa":   torch.zeros(()),
            }

        else:
            option_texts = []
            for q, ch in zip(batch.questions, batch.choices):
                option_texts.extend([f"{q} {c}" for c in ch])
            t_opt    = teacher.encode_text(option_texts).view(B, 4, -1)
            logits_t = torch.einsum("bd,bkd->bk", t_img,  t_opt) / args.tau_mc
            logits_s = torch.einsum("bd,bkd->bk", fused,  t_opt) / args.tau_mc
            l_mc     = kl_logits(logits_s, logits_t)
            l_ce     = (
                F.cross_entropy(logits_s, batch.labels.to(device), ignore_index=-1)
                if args.w_ce > 0.0 else torch.zeros((), device=device)
            )
            total = args.w_img * l_img + exp_args.w_scale * l_scale + l_mc + args.w_ce * l_ce
            return {
                "loss/total":  total,
                "loss/img":    l_img,
                "loss/scale":  l_scale,
                "loss/mc":     l_mc,
                "loss/ce":     l_ce,
                "data/is_qa":  torch.ones(()),
            }

    return student, teacher, compute_loss
