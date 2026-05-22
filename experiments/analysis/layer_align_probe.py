"""
Layer-alignment probe analysis.

Trains one Linear(D_teacher -> D_student) per shared layer to align teacher
features into student space, then measures cosine similarity between projected
teacher and student features at each layer.

Standalone:
    python experiments/analysis/layer_align_probe.py

Importable:
    from experiments.analysis.layer_align_probe import train_alignment_probes, aligned_cosine_sim
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def train_alignment_probes(
    student_backbone: nn.Module,
    teacher_trunk: nn.Module,
    images: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[nn.Linear]:
    """
    Train one Linear(D_t -> D_s, bias=False) per shared layer minimising
    1 - cosine_similarity(W @ t_k, s_k).
    Returns list of trained probes (one per shared layer).
    """
    student_backbone.eval()
    teacher_trunk.eval()

    n_s = len(student_backbone.blocks)
    n_t = len(teacher_trunk.blocks)
    n_shared = min(n_s, n_t)

    with torch.no_grad():
        s_feats = student_backbone.get_intermediate_layers(
            images, n=list(range(n_s)), return_prefix_tokens=True,
        )
        t_feats = teacher_trunk.get_intermediate_layers(
            images, n=list(range(n_t)), return_prefix_tokens=True,
        )

    s_cls = [f[1][:, 0, :].float() for f in s_feats]
    t_cls = [f[1][:, 0, :].float() for f in t_feats]

    D_s = s_cls[0].shape[-1]
    D_t = t_cls[0].shape[-1]

    probes: list[nn.Linear] = []
    for k in range(n_shared):
        probe = nn.Linear(D_t, D_s, bias=False).to(device)
        nn.init.orthogonal_(probe.weight)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)

        s_k = F.normalize(s_cls[k].to(device), dim=-1)
        t_k = t_cls[k].to(device)

        for _ in range(n_epochs):
            opt.zero_grad()
            proj = F.normalize(probe(t_k), dim=-1)
            loss = (1.0 - (proj * s_k).sum(dim=-1)).mean()
            loss.backward()
            opt.step()

        probes.append(probe)

    return probes


@torch.no_grad()
def aligned_cosine_sim(
    student_backbone: nn.Module,
    teacher_trunk: nn.Module,
    images: torch.Tensor,
    probes: list[nn.Linear],
) -> list[float]:
    """
    Compute per-layer cosine similarity between W_k(teacher_cls_k) and student_cls_k.
    probes[k] is the Linear(D_t -> D_s) trained for layer k.
    """
    student_backbone.eval()
    teacher_trunk.eval()

    n_s = len(student_backbone.blocks)
    n_t = len(teacher_trunk.blocks)
    n_shared = min(len(probes), n_s, n_t)
    device = next(probes[0].parameters()).device

    s_feats = student_backbone.get_intermediate_layers(
        images, n=list(range(n_s)), return_prefix_tokens=True,
    )
    t_feats = teacher_trunk.get_intermediate_layers(
        images, n=list(range(n_t)), return_prefix_tokens=True,
    )

    s_cls = [f[1][:, 0, :].float().to(device) for f in s_feats]
    t_cls = [f[1][:, 0, :].float().to(device) for f in t_feats]

    sims: list[float] = []
    for k in range(n_shared):
        s_k = F.normalize(s_cls[k], dim=-1)
        t_proj = F.normalize(probes[k](t_cls[k]), dim=-1)
        sims.append((s_k * t_proj).sum(dim=-1).mean().item())

    return sims


def plot_aligned_cosine_sim(
    sims_before: list[float],
    sims_after: list[float] | None = None,
    selected_layers: list[int] | None = None,
    save_path: "Path | str | None" = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(sims_before)), sims_before, "b-o", ms=4, label="Before distillation")
    if sims_after is not None:
        ax.plot(range(len(sims_after)), sims_after, "r-s", ms=4, label="After distillation")
    if selected_layers:
        for sl in selected_layers:
            ax.axvline(sl, color="g", ls="--", alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Teacher→student aligned cosine similarity per layer\n(trained linear probe)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    import argparse
    from config import paths_cfg
    p = argparse.ArgumentParser()
    p.add_argument("--pmc_oa_image_dir", default=paths_cfg.pmc_oa_image_dir)
    p.add_argument("--pmc_oa_jsonl", default=paths_cfg.pmc_oa_jsonl)
    p.add_argument("--log_dir", default="results/analysis")
    p.add_argument("--n_images", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--teacher_model",
                   default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--timm_student", default="vit_small_patch16_224")
    args = p.parse_args()

    import timm
    from torch.utils.data import DataLoader
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
        max_samples=args.n_images * 2,
    )
    loader = DataLoader(
        ds, batch_size=args.n_images * 2, shuffle=True, num_workers=0,
        collate_fn=lambda ex: collate_oa(ex, teacher.preprocess),
    )
    all_images = next(iter(loader)).images.to(device)
    train_images = all_images[:args.n_images]
    eval_images  = all_images[args.n_images:]

    print("Training alignment probes...")
    probes = train_alignment_probes(
        student_backbone, teacher_trunk, train_images,
        n_epochs=args.n_epochs, device=device,
    )

    print("Computing aligned cosine similarity...")
    sims = aligned_cosine_sim(student_backbone, teacher_trunk, eval_images, probes)
    for k, s in enumerate(sims):
        print(f"  Layer {k:2d}: {s:.4f}")

    out_dir = Path(args.log_dir)
    save_path = out_dir / "layer_align_probe.png"
    fig = plot_aligned_cosine_sim(sims, save_path=save_path)
    print(f"Figure saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
