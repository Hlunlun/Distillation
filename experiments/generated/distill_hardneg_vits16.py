from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.data_loaders import BatchOA
from experiments.models import FrozenTeacher, StudentVisualEncoder, kl_logits


def _enqueue(queue: torch.Tensor, ptr: list[int], new_embs: torch.Tensor) -> None:
    K = queue.shape[0]
    B = new_embs.shape[0]
    idx = torch.arange(ptr[0], ptr[0] + B) % K
    queue[idx] = new_embs.detach()
    ptr[0] = (ptr[0] + B) % K


def init(shared_args, exp_args, device):
    teacher = FrozenTeacher(shared_args.teacher_model, pretrained=shared_args.teacher_pretrained)
    teacher.model.to(device)
    student = StudentVisualEncoder(shared_args.timm_student, out_dim=teacher.embed_dim).to(device)

    # Queue starts random-initialized; early steps have noisier hard negatives but converges quickly.
    txt_queue = F.normalize(torch.randn(exp_args.queue_size, teacher.embed_dim), dim=-1).to(device)
    txt_queue_ptr = [0]

    def compute_loss(student, teacher, batch, args, _exp_args, device):
        images = batch.images.to(device, non_blocking=True)
        t_img = teacher.encode_image(images)
        s_img = student(images)
        loss_img = (1.0 - (s_img * t_img).sum(dim=-1)).mean()

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)
            neg_keys = torch.cat([t_txt.detach(), txt_queue], dim=0)
            logits_s = (s_img @ neg_keys.T) / args.temp
            logits_t = (t_img @ neg_keys.T) / args.temp
            loss_oa = kl_logits(logits_s, logits_t)
            _enqueue(txt_queue, txt_queue_ptr, t_txt)
            total = args.w_img * loss_img + args.w_oa * loss_oa
            return {"loss/total": total, "loss/img": loss_img, "loss/oa": loss_oa, "data/is_qa": torch.zeros(())}
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
            total = args.w_img * loss_img + args.w_mc * loss_mc + args.w_ce * loss_ce
            return {"loss/total": total, "loss/img": loss_img, "loss/mc": loss_mc, "loss/ce": loss_ce, "data/is_qa": torch.ones(())}

    return student, teacher, compute_loss
