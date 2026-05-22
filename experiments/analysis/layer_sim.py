"""
Layer-change analysis for ViT distillation.

Three analyses (adapted from Align-KD CVPR 2025 for encoder-only setting):
  (a) Adjacent-layer CLS cosine similarity — where features change most
  (b) Linear CKA between student and teacher at each layer — cross-model alignment
  (c) CLS vs mean-patch normalised Euclidean distance — local/global divergence

Standalone:
    python experiments/analysis/layer_sim.py

Importable:
    from experiments.analysis.layer_sim import analyze_layer_similarity, make_attn_hook
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── shared hook factory (also used by training experiments) ───────────────────

def make_attn_hook(captured: dict, idx: int, detach: bool = True):
    """Forward hook that captures softmax attention weights → captured[idx] = [B, H, N, N]."""
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, module.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        w = (q @ k.transpose(-2, -1)) * module.scale
        attn = w.softmax(dim=-1)
        captured[idx] = attn.detach() if detach else attn
    return hook


# ── CKA helper ────────────────────────────────────────────────────────────────

def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between [B, D1] and [B, D2] (dimension-free representational similarity)."""
    X = X.float()
    Y = Y.float()
    n = X.shape[0]
    K = X @ X.T                               # [B, B]
    L = Y @ Y.T                               # [B, B]
    H = torch.eye(n, device=K.device) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_kl = (Kc * Lc).sum()
    denom = ((Kc * Kc).sum().sqrt() * (Lc * Lc).sum().sqrt()).clamp_min(1e-8)
    return (hsic_kl / denom).item()


# ── main analysis ──────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_layer_similarity(
    student_backbone: nn.Module,
    teacher_trunk: nn.Module,
    images: torch.Tensor,
    top_k: int = 2,
    writer: "SummaryWriter | None" = None,
    writer_step: int = 0,
    save_path: "Path | str | None" = None,
) -> tuple[list[int], plt.Figure]:
    """
    Returns (selected_layer_indices, figure).
    selected_layer_indices: top_k student layers with lowest adjacent cosine similarity.
    Green dashed lines on figure mark the selected layers.
    """
    student_backbone.eval()
    teacher_trunk.eval()

    n_s = len(student_backbone.blocks)
    n_t = len(teacher_trunk.blocks)

    # ── capture input embeddings (before block 0) via pre-hook ───────────────
    _s_pre: dict = {}
    _t_pre: dict = {}

    def _pre_hook(store: dict):
        def hook(_, inp):
            store["x"] = inp[0].detach()
        return hook

    _hs = student_backbone.blocks[0].register_forward_pre_hook(_pre_hook(_s_pre))
    _ht = teacher_trunk.blocks[0].register_forward_pre_hook(_pre_hook(_t_pre))

    # ── get per-layer intermediate features ───────────────────────────────────
    s_feats = list(student_backbone.get_intermediate_layers(
        images, n=list(range(n_s)), return_prefix_tokens=True,
    ))
    t_feats = list(teacher_trunk.get_intermediate_layers(
        images, n=list(range(n_t)), return_prefix_tokens=True,
    ))

    _hs.remove()
    _ht.remove()

    # prepend input embedding as layer 0 (patch [B,N,D], cls [B,1,D])
    def _split_input(x: torch.Tensor):
        return (x[:, 1:, :], x[:, :1, :])

    s_feats = [_split_input(_s_pre["x"])] + s_feats
    t_feats = [_split_input(_t_pre["x"])] + t_feats

    # Each element: (patch [B, 196, D], cls [B, 1, D])
    s_cls  = [F.normalize(f[1][:, 0, :].float(), dim=-1) for f in s_feats]  # n_s × [B, D_s]
    t_cls  = [F.normalize(f[1][:, 0, :].float(), dim=-1) for f in t_feats]  # n_t × [B, D_t]
    s_pat  = [F.normalize(f[0].float().mean(dim=1), dim=-1) for f in s_feats]
    t_pat  = [F.normalize(f[0].float().mean(dim=1), dim=-1) for f in t_feats]
    # mean of all tokens (CLS + patch) for adjacent-layer similarity
    s_all  = [F.normalize(torch.cat([f[1], f[0]], dim=1).float().mean(dim=1), dim=-1) for f in s_feats]
    t_all  = [F.normalize(torch.cat([f[1], f[0]], dim=1).float().mean(dim=1), dim=-1) for f in t_feats]

    # ── (a) adjacent-layer cosine similarity (all tokens) ─────────────────────
    def _adj_sim(feat_list: list[torch.Tensor]) -> list[float]:
        return [
            (feat_list[i - 1] * feat_list[i]).sum(dim=-1).mean().item()
            for i in range(1, len(feat_list))
        ]

    s_adj = _adj_sim(s_all)   # length n_s - 1
    t_adj = _adj_sim(t_all)   # length n_t - 1

    # ── (d) per-token patch adjacent cosine similarity ────────────────────────
    def _patch_adj_sim(feats: list) -> list[float]:
        return [
            F.normalize(feats[i - 1][0].float(), dim=-1)
            .mul(F.normalize(feats[i][0].float(), dim=-1))
            .sum(dim=-1).mean().item()
            for i in range(1, len(feats))
        ]

    s_patch_adj = _patch_adj_sim(s_feats)
    t_patch_adj = _patch_adj_sim(t_feats)

    # ── (b) linear CKA between student and teacher CLS at each shared layer ───
    n_shared = min(n_s, n_t)
    cka_vals = [_linear_cka(s_cls[k], t_cls[k]) for k in range(n_shared)]

    # ── (c) CLS vs mean-patch normalised Euclidean distance (unit vectors) ────
    def _cls_patch_dist(cls_list: list[torch.Tensor], pat_list: list[torch.Tensor]) -> list[float]:
        return [
            torch.norm(cls_list[k] - pat_list[k], dim=-1).mean().item()
            for k in range(len(cls_list))
        ]

    s_dist = _cls_patch_dist(s_cls, s_pat)
    t_dist = _cls_patch_dist(t_cls, t_pat)

    # ── select top_k layers by lowest adjacent sim ────────────────────────────
    # layer 0 has no prior → score 0.0 (treated as most changed)
    layer_scores = [0.0] + s_adj   # length n_s
    selected_layers = sorted(
        sorted(range(n_s), key=lambda i: layer_scores[i])[:top_k]
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    ax = axes[0]
    ax.plot(range(len(s_adj)), s_adj, "b-o", ms=4, label="Student (ViT-S)")
    ax.plot(range(len(t_adj)), t_adj, "r-s", ms=4, label="Teacher (BiomedCLIP)")
    for sl in selected_layers:
        ax.axvline(sl, color="g", ls="--", alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean-token cosine similarity")
    ax.set_title("(a) Adjacent-layer cosine similarity (all tokens)\n(↓ = more change)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(n_shared), cka_vals, "m-^", ms=4, label="Student–Teacher CKA")
    for sl in selected_layers:
        ax.axvline(sl, color="g", ls="--", alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title("(b) Cross-model representational\nalignment (CKA, ↑ = more aligned)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(range(len(s_dist)), s_dist, "b-o", ms=4, label="Student")
    ax.plot(range(len(t_dist)), t_dist, "r-s", ms=4, label="Teacher")
    for sl in selected_layers:
        ax.axvline(sl, color="g", ls="--", alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Euclidean distance (unit vectors)")
    ax.set_title("(c) CLS vs mean-patch distance\n(↑ = local/global divergence)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(range(len(s_patch_adj)), s_patch_adj, "b-o", ms=4, label="Student (ViT-S)")
    ax.plot(range(len(t_patch_adj)), t_patch_adj, "r-s", ms=4, label="Teacher (BiomedCLIP)")
    for sl in selected_layers:
        ax.axvline(sl, color="g", ls="--", alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Per-token cosine similarity")
    ax.set_title("(d) Adjacent-layer patch similarity\n(per-token, ↓ = more change)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Layer-change analysis — selected layers: {selected_layers}", fontsize=12)
    fig.tight_layout()

    if writer is not None:
        writer.add_figure("analysis/layer_sim", fig, global_step=writer_step)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return selected_layers, fig


# ── standalone CLI ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from config import paths_cfg
    p = argparse.ArgumentParser()
    p.add_argument("--pmc_oa_image_dir", default=paths_cfg.pmc_oa_image_dir)
    p.add_argument("--pmc_oa_jsonl", default=paths_cfg.pmc_oa_jsonl)
    p.add_argument("--log_dir", default="results/analysis")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--n_images", type=int, default=128)
    p.add_argument("--teacher_model",
                   default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--timm_student", default="vit_small_patch16_224")
    args = p.parse_args()

    import timm
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from experiments.data_loaders import PMCOADataset, collate_oa
    from experiments.models import FrozenTeacher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = FrozenTeacher(args.teacher_model, load_tokenizer=False)
    teacher.model.to(device)

    teacher_trunk = getattr(getattr(teacher.model, "visual", None), "trunk", None)
    if teacher_trunk is None:
        raise RuntimeError("Teacher has no visual.trunk; expected open_clip TimmModel.")

    student_backbone = timm.create_model(
        args.timm_student, pretrained=True, num_classes=0
    ).to(device)

    ds = PMCOADataset(
        image_dir=args.pmc_oa_image_dir,
        jsonl_path=args.pmc_oa_jsonl,
        max_samples=args.n_images,
    )
    loader = DataLoader(
        ds, batch_size=args.n_images, shuffle=True, num_workers=0,
        collate_fn=lambda ex: collate_oa(ex, teacher.preprocess),
    )
    images = next(iter(loader)).images.to(device)

    out_dir = Path(args.log_dir)
    writer = SummaryWriter(str(out_dir / "tb"))
    save_path = out_dir / "layer_sim.png"

    selected, fig = analyze_layer_similarity(
        student_backbone, teacher_trunk, images,
        top_k=args.top_k,
        writer=writer,
        writer_step=0,
        save_path=save_path,
    )
    print(f"Selected layers: {selected}")
    print(f"Figure saved:    {save_path}")
    writer.close()
    plt.close(fig)


if __name__ == "__main__":
    main()
