from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from experiments.data_loaders import BatchOA, BatchQA
from experiments.models import FrozenTeacher, kl_logits, CosmosStudent
from experiments.generated.distill_cosmos_vits16 import infonce

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


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

    # asymmetric augmentation for student only; teacher/EMA always see clean image
    student_rrc = T.RandomResizedCrop(
        224,
        scale=(exp_args.rrc_scale_min, exp_args.rrc_scale_max),
        ratio=(3 / 4, 4 / 3),
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )

    def _ema_update() -> None:
        m = exp_args.ema_momentum
        for ep, sp in zip(ema.parameters(), student.parameters()):
            ep.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    def compute_loss(student, teacher, batch, args, exp_args, device):
        _ema_update()

        images = batch.images.to(device, non_blocking=True)
        B = images.shape[0]

        # apply RRC per sample (each crop is independent)
        images_crop = torch.stack([student_rrc(img) for img in images])

        t_img = teacher.encode_image(images)              # [B, D]  — clean

        s_cls, _ = student.forward_full(images_crop)      # [B, D]

        with torch.no_grad():
            ema_cls, _ = ema.forward_full(images)         # [B, D]  — clean

        l_img    = (1.0 - (s_cls * t_img).sum(dim=-1)).mean()
        l_cosmos = infonce(s_cls, ema_cls, temp=args.temp)

        if isinstance(batch, BatchOA):
            t_txt = teacher.encode_text(batch.captions)   # [B, D]

            logits_t = (t_img @ t_txt.t()) / args.temp
            logits_s = (s_cls @ t_txt.t()) / args.temp
            l_itc    = kl_logits(logits_s, logits_t)

            total = args.w_img * l_img + l_itc + l_cosmos
            return {
                "loss/total":  total,
                "loss/img":    l_img,
                "loss/itc":    l_itc,
                "loss/cosmos": l_cosmos,
                "data/is_qa":  torch.zeros(()),
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
