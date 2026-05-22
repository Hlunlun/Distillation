from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, StudentDualHead, kl_logits


class PrimaryWrapper(nn.Module):
    def __init__(self, student: StudentDualHead):
        super().__init__()
        self._student = student

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._student.encode_primary(images)


def get_probe_model(student: StudentDualHead) -> PrimaryWrapper:
    return PrimaryWrapper(student)


def init(shared_args, exp_args, device):
    teacher_pri = FrozenTeacher(exp_args.teacher_primary, pretrained=exp_args.teacher_primary_pretrained, load_tokenizer=True)
    teacher_pri.model.to(device)
    teacher_sec = FrozenTeacher(exp_args.teacher_secondary, pretrained=exp_args.teacher_secondary_pretrained, load_tokenizer=False)
    teacher_sec.model.to(device)

    student = StudentDualHead(
        shared_args.timm_student,
        out_dim_primary=teacher_pri.embed_dim,
        out_dim_secondary=teacher_sec.embed_dim,
    ).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        t_img_pri = teacher.encode_image(images)
        t_img_sec = teacher_sec.encode_image(images)
        s_img_pri, s_img_sec = student(images)

        loss_img_pri = (1.0 - (s_img_pri * t_img_pri).sum(dim=-1)).mean()
        loss_img_sec = (1.0 - (s_img_sec * t_img_sec).sum(dim=-1)).mean()

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)
            logits_t = (t_img_pri @ t_txt.t()) / args.temp
            logits_s = (s_img_pri @ t_txt.t()) / args.temp
            loss_oa = kl_logits(logits_s, logits_t)
            total = exp_args.w_img_primary * loss_img_pri + exp_args.w_img_secondary * loss_img_sec + args.w_oa * loss_oa
            return {"loss/total": total, "loss/img_primary": loss_img_pri, "loss/img_secondary": loss_img_sec, "loss/oa": loss_oa, "data/is_qa": torch.zeros(())}
        else:
            option_texts = []
            for q, ch in zip(batch.questions, batch.choices):
                option_texts.extend([f"{q} {c}" for c in ch])
            t_opt = teacher.encode_text(option_texts).view(images.shape[0], 4, -1)
            logits_t = torch.einsum("bd,bkd->bk", t_img_pri, t_opt) / args.tau_mc
            logits_s = torch.einsum("bd,bkd->bk", s_img_pri, t_opt) / args.tau_mc
            loss_mc = kl_logits(logits_s, logits_t)
            loss_ce = (
                F.cross_entropy(logits_s, batch.labels.to(device), ignore_index=-1)
                if args.w_ce > 0.0 else torch.zeros((), device=device)
            )
            total = exp_args.w_img_primary * loss_img_pri + exp_args.w_img_secondary * loss_img_sec + args.w_mc * loss_mc + args.w_ce * loss_ce
            return {"loss/total": total, "loss/img_primary": loss_img_pri, "loss/img_secondary": loss_img_sec, "loss/mc": loss_mc, "loss/ce": loss_ce, "data/is_qa": torch.ones(())}

    return student, teacher_pri, compute_loss
