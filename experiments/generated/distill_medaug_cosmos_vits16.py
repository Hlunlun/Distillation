from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
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
from experiments.generated.distill_alignkd_cosmos_vits16 import (
    _get_teacher_trunk, _attn_kl_loss, _tqva_loss,
)
from experiments.analysis.layer_sim import analyze_layer_similarity, make_attn_hook

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


# ── augmentation ──────────────────────────────────────────────────────────────

def _student_augment(
    images: torch.Tensor,
    intensity_scale: float,
    intensity_shift: float,
    p_channel_cutmix: float,
) -> torch.Tensor:
    """Apply asymmetric augmentation to student inputs (teacher always sees clean images).

    intensity_scale: max multiplicative perturbation (e.g. 0.1 → scale ∈ [0.9, 1.1])
    intensity_shift: max additive perturbation in normalised units
    p_channel_cutmix: per-channel probability of swapping from a random other image.
                      0.0 disables channel cutmix entirely.
    """
    B = images.shape[0]
    out = images.clone()

    # per-sample intensity scale and shift
    scale = 1.0 + (torch.rand(B, 1, 1, 1, device=images.device) * 2 - 1) * intensity_scale
    shift = (torch.rand(B, 1, 1, 1, device=images.device) * 2 - 1) * intensity_shift
    out = out * scale + shift

    if p_channel_cutmix > 0.0:
        # for each (sample, channel) independently: swap that channel from a random other image
        perm = torch.randperm(B, device=images.device)
        # mask shape [B, C, 1, 1]: True → take channel from perm[b]-th image
        mask = torch.rand(B, images.shape[1], 1, 1, device=images.device) < p_channel_cutmix
        out = torch.where(mask, images[perm], out)

    return out


# ── probe model ───────────────────────────────────────────────────────────────

def _attention_mask(
    t_attn_buf: dict[int, torch.Tensor],
    layer_idx: int,
    H: int,
    W: int,
    mask_floor: float,
) -> torch.Tensor:
    attn = t_attn_buf[layer_idx]                      # [B, heads, N, N]
    cls_attn = attn[:, :, 0, 1:].mean(dim=1)          # [B, 196] — CLS→patch, avg heads
    mn = cls_attn.amin(dim=1, keepdim=True)
    mx = cls_attn.amax(dim=1, keepdim=True)
    cls_attn = (cls_attn - mn) / (mx - mn + 1e-6)    # [B, 196] in [0, 1]
    B = cls_attn.shape[0]
    n = int(cls_attn.shape[1] ** 0.5)                 # 14 for ViT-S/16 on 224px
    spatial = cls_attn.view(B, 1, n, n)
    mask = F.interpolate(spatial, size=(H, W), mode="bilinear", align_corners=False)
    return mask_floor + (1.0 - mask_floor) * mask     # [mask_floor, 1.0]


def get_probe_model(student: CosmosStudent) -> CosmosStudent:
    return student


# ── init ──────────────────────────────────────────────────────────────────────

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

    # ── layer-change analysis ─────────────────────────────────────────────────
    teacher_trunk = _get_teacher_trunk(teacher)
    n_blocks = len(student.backbone.blocks)
    selected_layers: list[int] = [0, n_blocks - 1]

    _has_oa = getattr(shared_args, "data_sources", "both") != "qa"
    if teacher_trunk is not None and _has_oa:
        ds = PMCOADataset(
            image_dir=shared_args.pmc_oa_image_dir,
            jsonl_path=shared_args.pmc_oa_train_jsonl,
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
        selected_layers = [0, 11]
        plt.close(fig)
        print(f"[MedAug-COSMOS] selected layers: {selected_layers}  |  figure: {save_path}")
    else:
        print(f"[MedAug-COSMOS] No visual.trunk found or OA data unavailable; using default layers: {selected_layers}")

    # ── register attention hooks ──────────────────────────────────────────────
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

        # asymmetric: student sees augmented+masked, teacher and EMA see clean
        images_aug = _student_augment(
            images,
            intensity_scale=exp_args.intensity_scale,
            intensity_shift=exp_args.intensity_shift,
            p_channel_cutmix=exp_args.p_channel_cutmix,
        )

        # teacher runs first so t_attn_buf is populated before student forward
        t_img = teacher.encode_image(images)                 # [B, D]  — clean

        # suppress background: use teacher's deepest-layer CLS attention as spatial mask
        if t_attn_buf:
            attn_mask = _attention_mask(
                t_attn_buf, selected_layers[-1],
                images.shape[-2], images.shape[-1],
                exp_args.mask_floor,
            ).to(device)
            images_aug = images_aug * attn_mask

        # forward calls trigger the registered hooks
        s_cls, s_patch = student.forward_full(images_aug)   # [B, D], [B, 196, D]

        with torch.no_grad():
            ema_cls, _ = ema.forward_full(images)            # [B, D]  — clean

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

            # L_tqva: s_patch already from augmented images; t_patch_clip from clean
            t_patch_clip = _teacher_patch_features(teacher, images)
            l_tqva = (
                _tqva_loss(t_txt, t_patch_clip, s_patch)
                if t_patch_clip is not None
                else torch.zeros((), device=device)
            )

            # L_lg: s_patch from augmented forward; teacher crops from clean images
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
                _pad_sentences(_SENT_SPLIT.split(cap.strip()) or [cap], K)
                for cap in batch.captions
            ]
            flat_sents = [sentences_per_sample[b][k] for k in range(K) for b in range(B)]
            t_sents    = teacher.encode_text(flat_sents).view(K, B, -1)

            region, _  = tgac(s_patch, t_txt)
            cos_kb     = (t_sents.detach() * region.unsqueeze(0)).sum(dim=-1)
            l_crop_raw = (1.0 - cos_kb).mean(dim=0)
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
