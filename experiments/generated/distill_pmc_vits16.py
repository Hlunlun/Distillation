from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, StudentVisualEncoder, kl_logits


def relational_kd(student_emb: torch.Tensor, teacher_emb: torch.Tensor, tau: float) -> torch.Tensor:
    # Match within-batch image-image similarity distributions row-wise.
    sim_s = (student_emb @ student_emb.t()) / tau
    sim_t = (teacher_emb @ teacher_emb.t()) / tau
    return kl_logits(sim_s, sim_t)


def init(shared_args, _exp_args, device):
    teacher = FrozenTeacher(shared_args.teacher_model, pretrained=shared_args.teacher_pretrained)
    teacher.model.to(device)
    student = StudentVisualEncoder(shared_args.timm_student, out_dim=teacher.embed_dim).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        t_img = teacher.encode_image(images)
        s_img = student(images)
        loss_img = (1.0 - (s_img * t_img).sum(dim=-1)).mean()

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)
            logits_t = (t_img @ t_txt.t()) / args.temp
            logits_s = (s_img @ t_txt.t()) / args.temp
            loss_oa = kl_logits(logits_s, logits_t)
            loss_rel = relational_kd(s_img, t_img, exp_args.tau_rel) if exp_args.w_rel > 0.0 else torch.zeros((), device=device)
            total = args.w_img * loss_img + args.w_oa * loss_oa + exp_args.w_rel * loss_rel
            return {"loss/total": total, "loss/img": loss_img, "loss/oa": loss_oa, "loss/rel": loss_rel, "data/is_qa": torch.zeros(())}
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
            loss_rel = relational_kd(s_img, t_img, exp_args.tau_rel) if exp_args.w_rel > 0.0 else torch.zeros((), device=device)
            total = args.w_img * loss_img + args.w_mc * loss_mc + args.w_ce * loss_ce + exp_args.w_rel * loss_rel
            return {"loss/total": total, "loss/img": loss_img, "loss/mc": loss_mc, "loss/ce": loss_ce, "loss/rel": loss_rel, "data/is_qa": torch.ones(())}

    return student, teacher, compute_loss
