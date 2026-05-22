from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, kl_logits, SemanticSlotStudent


def infonce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    B = a.shape[0]
    logits = a @ b.T / temp
    labels = torch.arange(B, device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    """Penalise slot collapse: off-diagonal entries of the K×K gram matrix should be 0."""
    normed = F.normalize(slots, dim=-1)        # [K, D]
    gram   = normed @ normed.T                 # [K, K]
    K = gram.shape[0]
    mask   = ~torch.eye(K, dtype=torch.bool, device=gram.device)
    return (gram[mask] ** 2).mean()


def get_probe_model(student: SemanticSlotStudent) -> SemanticSlotStudent:
    return student


def init(shared_args, exp_args, device):
    teacher = FrozenTeacher(
        shared_args.teacher_model,
        pretrained=shared_args.teacher_pretrained,
        load_tokenizer=True,
    )
    teacher.model.to(device)

    student = SemanticSlotStudent(
        shared_args.timm_student,
        out_dim=teacher.embed_dim,
        num_slots=exp_args.num_slots,
        num_heads=exp_args.num_heads,
    ).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        B      = images.shape[0]

        t_img  = teacher.encode_image(images)    # [B, D]

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)   # [B, D]

            # Text-gated slot aggregation during OA training
            s_cls, s_patch, slot_out, _ = student.forward_full(images, t_txt)

            l_img  = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()
            logits_t = (t_img  @ t_txt.t()) / args.temp
            logits_s = (s_cls  @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)
            l_div    = slot_diversity_loss(student.slots)

            total = args.w_img * l_img + l_itc + exp_args.w_slot_div * l_div
            return {
                "loss/total":    total,
                "loss/img":      l_img,
                "loss/itc":      l_itc,
                "loss/slot_div": l_div,
                "data/is_qa":    torch.zeros(()),
            }

        else:
            s_cls, s_patch, _, _ = student.forward_full(images)

            l_img  = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()

            option_texts = []
            for q, ch in zip(batch.questions, batch.choices):
                option_texts.extend([f"{q} {c}" for c in ch])
            t_opt    = teacher.encode_text(option_texts).view(B, 4, -1)
            logits_t = torch.einsum("bd,bkd->bk", t_img,  t_opt) / args.tau_mc
            logits_s = torch.einsum("bd,bkd->bk", s_cls,  t_opt) / args.tau_mc
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
