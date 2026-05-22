from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, kl_logits, BottleneckStudent


def get_probe_model(student: BottleneckStudent) -> BottleneckStudent:
    return student


def init(shared_args, exp_args, device):
    teacher = FrozenTeacher(
        shared_args.teacher_model,
        pretrained=shared_args.teacher_pretrained,
        load_tokenizer=True,
    )
    teacher.model.to(device)

    student = BottleneckStudent(
        shared_args.timm_student,
        out_dim=teacher.embed_dim,
        bottleneck_dim=exp_args.bottleneck_dim,
    ).to(device)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        B      = images.shape[0]

        t_img  = teacher.encode_image(images)    # [B, D]

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)   # [B, D]

            # Text-gated forward: gate selects bottleneck units relevant to caption
            s_cls, s_patch, gate = student.forward_train(images, t_txt)

            l_img = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()

            logits_t = (t_img  @ t_txt.t()) / args.temp
            logits_s = (s_cls  @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)

            # Sparsity: encourage most gates to stay near 0 (L1 on gate activations)
            l_sparsity = gate.mean()

            total = args.w_img * l_img + l_itc + exp_args.w_sparsity * l_sparsity
            return {
                "loss/total":     total,
                "loss/img":       l_img,
                "loss/itc":       l_itc,
                "loss/sparsity":  l_sparsity,
                "data/gate_mean": l_sparsity.detach(),
                "data/is_qa":     torch.zeros(()),
            }

        else:
            # QA: no caption available, use visual-only bottleneck
            s_cls = student(images)

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
