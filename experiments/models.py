from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── dependency guards ──────────────────────────────────────────────────────────

def require_open_clip():
    try:
        import open_clip  # type: ignore
        return open_clip
    except Exception as e:
        raise RuntimeError("Missing dependency `open_clip_torch`.") from e


def require_timm():
    try:
        import timm  # type: ignore
        return timm
    except Exception as e:
        raise RuntimeError("Missing dependency `timm`.") from e


# ── student ────────────────────────────────────────────────────────────────────

class StudentVisualEncoder(nn.Module):
    def __init__(self, timm_name: str, out_dim: int):
        super().__init__()
        timm = require_timm()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            raise RuntimeError(f"Unsupported timm model (missing num_features): {timm_name}")
        self.proj = nn.Linear(int(in_dim), int(out_dim), bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(images)), dim=-1)


class StudentDualHead(nn.Module):
    def __init__(self, timm_name: str, out_dim_primary: int, out_dim_secondary: int):
        super().__init__()
        timm = require_timm()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            raise RuntimeError(f"Unsupported timm model (missing num_features): {timm_name}")
        self.proj_primary = nn.Linear(int(in_dim), int(out_dim_primary), bias=False)
        self.proj_secondary = nn.Linear(int(in_dim), int(out_dim_secondary), bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(images)
        return (
            F.normalize(self.proj_primary(feats), dim=-1),
            F.normalize(self.proj_secondary(feats), dim=-1),
        )

    def encode_primary(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_primary(self.backbone(images)), dim=-1)


# ── teacher ────────────────────────────────────────────────────────────────────

class _HFCLIPPthAdapter(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self._clip = clip_model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self._clip.vision_model(pixel_values=images).pooler_output
        return self._clip.visual_projection(pooled)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = self._clip.text_model(input_ids=tokens).pooler_output
        return self._clip.text_projection(pooled)


class FrozenTeacher(nn.Module):
    def __init__(self, model_name: str, pretrained: Optional[str] = None, load_tokenizer: bool = True):
        super().__init__()
        _pth_base_hf_id: Optional[str] = None
        if model_name.startswith("hf-pth-clip:"):
            model, preprocess, _pth_base_hf_id = self._load_hf_clip_pth(model_name[len("hf-pth-clip:"):])
        else:
            open_clip = require_open_clip()
            model, preprocess = self._load_open_clip(open_clip, model_name, pretrained)

        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = preprocess

        if load_tokenizer:
            if _pth_base_hf_id is not None:
                _clip_arch = "ViT-L-14" if _pth_base_hf_id == "openai/clip-vit-large-patch14" else "ViT-B-32"
                self.tokenizer = require_open_clip().get_tokenizer(_clip_arch)
            else:
                self.tokenizer = require_open_clip().get_tokenizer(model_name)
        else:
            self.tokenizer = None

        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224))
            img = self.model.encode_image(dummy)
        self.embed_dim = int(img.shape[-1])

    @staticmethod
    def _load_open_clip(open_clip, model_name: str, pretrained: Optional[str]):
        if model_name.startswith("hf-hub:") and hasattr(open_clip, "create_model_from_pretrained"):
            out = open_clip.create_model_from_pretrained(model_name)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return out[0], out[-1]
            raise RuntimeError("create_model_from_pretrained returned unexpected value.")
        if hasattr(open_clip, "create_model_and_transforms"):
            out = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                return out[0], out[2]
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return out[0], out[-1]
        raise RuntimeError("Unsupported open_clip version.")

    @staticmethod
    def _load_hf_clip_pth(repo_id: str):
        import huggingface_hub
        from transformers import CLIPModel
        files = list(huggingface_hub.list_repo_files(repo_id))
        pth_files = [f for f in files if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in HF repo {repo_id}")
        pth_path = huggingface_hub.hf_hub_download(repo_id, pth_files[0])
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
        ln_weight = state_dict.get("vision_model.post_layernorm.weight")
        if ln_weight is None:
            raise RuntimeError(f"Cannot determine vision architecture for {repo_id}")
        base_model = "openai/clip-vit-large-patch14" if ln_weight.shape[0] >= 1024 else "openai/clip-vit-base-patch32"
        clip = CLIPModel.from_pretrained(base_model)
        clip.load_state_dict(state_dict, strict=False)
        clip.eval()
        preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        return _HFCLIPPthAdapter(clip), preprocess, base_model

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("This teacher was loaded without a tokenizer.")
        tokens = self.tokenizer(texts)
        emb = self.model.encode_text(tokens.to(next(self.model.parameters()).device))
        return F.normalize(emb, dim=-1)


# ── shared loss ────────────────────────────────────────────────────────────────

def kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )


# ── Encoder classes ────────────────────────────────────────────────────────────

class OpenCLIPEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: Optional[str] = None):
        super().__init__()
        import open_clip  # type: ignore
        if model_name.startswith("hf-hub:") and hasattr(open_clip, "create_model_from_pretrained"):
            out = open_clip.create_model_from_pretrained(model_name)
            if not (isinstance(out, (list, tuple)) and len(out) >= 2):
                raise RuntimeError(f"create_model_from_pretrained returned unexpected value for {model_name}")
            model, preprocess = out[0], out[-1]
        elif hasattr(open_clip, "create_model_and_transforms"):
            out = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                model, preprocess = out[0], out[2]
            elif isinstance(out, (list, tuple)) and len(out) >= 2:
                model, preprocess = out[0], out[-1]
            else:
                raise RuntimeError(f"create_model_and_transforms returned unexpected value for {model_name}")
        else:
            raise RuntimeError("Unsupported open_clip version.")
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.preprocess = preprocess
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224))
            emb = model.encode_image(dummy)
        self.embed_dim = int(emb.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)


class PthOpenCLIPEncoder(nn.Module):
    def __init__(self, hf_repo: str, pth_filename: str, arch: str):
        super().__init__()
        import open_clip
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(hf_repo, pth_filename)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        model, _, preprocess = open_clip.create_model_and_transforms(arch)
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [{arch}] missing keys: {len(missing)}")
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.preprocess = preprocess
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224))
            emb = model.encode_image(dummy)
        self.embed_dim = int(emb.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)


class PthHFCLIPEncoder(nn.Module):
    def __init__(self, hf_repo: str, pth_filename: str, base_model_id: str):
        super().__init__()
        from transformers import CLIPModel, CLIPImageProcessor
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(hf_repo, pth_filename)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        model = CLIPModel.from_pretrained(base_model_id)
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [{hf_repo}] missing keys: {len(missing)}")
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        try:
            proc = CLIPImageProcessor.from_pretrained(base_model_id)
            size = proc.size
            if hasattr(size, 'get'):
                image_size = size.get("shortest_edge", size.get("height", 224))
            else:
                image_size = int(size) if size else 224
            mean, std = proc.image_mean, proc.image_std
        except Exception:
            image_size = 224
            mean, std = _CLIP_MEAN, _CLIP_STD
        self.preprocess = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        with torch.no_grad():
            dummy = torch.zeros((1, 3, image_size, image_size))
            emb = model.vision_model(pixel_values=dummy).pooler_output
        self.embed_dim = int(emb.shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        emb = self.model.vision_model(pixel_values=images).pooler_output
        return F.normalize(emb, dim=-1)


class HFCLIPEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import CLIPModel, CLIPImageProcessor
        try:
            proc = CLIPImageProcessor.from_pretrained(model_id)
            size = proc.size
            if hasattr(size, 'get'):
                image_size = size.get("shortest_edge", size.get("height", 224))
            else:
                image_size = int(size) if size else 224
            mean, std = proc.image_mean, proc.image_std
        except (OSError, EnvironmentError):
            image_size = 224
            mean, std = _CLIP_MEAN, _CLIP_STD
        model = CLIPModel.from_pretrained(model_id).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        with torch.no_grad():
            dummy = torch.zeros((1, 3, image_size, image_size))
            emb = model.vision_model(pixel_values=dummy).pooler_output
        self.embed_dim = int(emb.shape[-1])
        self.preprocess = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        emb = self.model.vision_model(pixel_values=images).pooler_output
        return F.normalize(emb, dim=-1)


class HFAutoVisionEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        try:
            proc = AutoImageProcessor.from_pretrained(model_id)
            size = getattr(proc, "size", {}) or {}
            if isinstance(size, dict):
                image_size = size.get("shortest_edge", size.get("height", 224))
            else:
                image_size = 224
            mean = getattr(proc, "image_mean", _CLIP_MEAN)
            std  = getattr(proc, "image_std",  _CLIP_STD)
        except (OSError, EnvironmentError):
            image_size = 224
            mean, std = _CLIP_MEAN, _CLIP_STD
        model = AutoModel.from_pretrained(model_id).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        with torch.no_grad():
            dummy = torch.zeros((1, 3, image_size, image_size))
            emb = self._embed(dummy)
        self.embed_dim = int(emb.shape[-1])
        self.preprocess = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        if callable(getattr(self.model, "get_image_features", None)):
            out = self.model.get_image_features(pixel_values=images)
            if isinstance(out, torch.Tensor):
                return out
        return self.model.vision_model(pixel_values=images).pooler_output

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._embed(images), dim=-1)


class TimmEncoder(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = True):
        super().__init__()
        import timm  # type: ignore
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


# ── cosmos student ─────────────────────────────────────────────────────────────

class CosmosStudent(nn.Module):
    """ViT-S/16 student with CLS and patch projection heads for COSMOS-style distillation."""

    def __init__(self, timm_name: str, out_dim: int):
        super().__init__()
        timm = require_timm()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = int(self.backbone.num_features)
        with torch.no_grad():
            _feats = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        if _feats.dim() != 3:
            raise RuntimeError(
                f"CosmosStudent requires a ViT backbone (forward_features → [B, N+1, D]), "
                f"but '{timm_name}' returned shape {tuple(_feats.shape)}. "
                "Pass --timm_student vit_small_patch16_224"
            )
        self.proj       = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_patch = nn.Linear(in_dim, out_dim, bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(images)), dim=-1)

    def forward_full(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls [B, D], patch_proj [B, 196, D]) both L2-normalised."""
        all_tok    = self.backbone.forward_features(images)   # [B, N+1, in_dim]
        cls_       = F.normalize(self.proj(all_tok[:, 0, :]), dim=-1)
        patch_proj = F.normalize(self.proj_patch(all_tok[:, 1:, :]), dim=-1)
        return cls_, patch_proj


# ── TGAC ──────────────────────────────────────────────────────────────────────

class TGAC(nn.Module):
    """Text-Guided Adaptive Crop: soft-weighted student patch aggregation.

    Scores each patch by dot product with text_cls, applies softmax(/ temp),
    and returns a single weighted-sum region — fully differentiable, so the
    backbone learns to produce text-responsive patch features.
    """

    def __init__(self, top_k: int, temp: float = 0.1):
        super().__init__()
        self.top_k = top_k   # kept for config compatibility; not used in forward
        self.temp  = temp

    def forward(
        self,
        patch_proj: torch.Tensor,   # [B, N, D] normalised
        text_cls:   torch.Tensor,   # [B, D] normalised
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores  = (patch_proj * text_cls.unsqueeze(1)).sum(dim=-1)          # [B, N]
        weights = F.softmax(scores / self.temp, dim=-1)                      # [B, N]
        region  = F.normalize((weights.unsqueeze(-1) * patch_proj).sum(dim=1), dim=-1)  # [B, D]
        return region, weights


# ── SemanticSlotStudent ────────────────────────────────────────────────────────

class SemanticSlotStudent(nn.Module):
    """ViT student with K learnable semantic slot tokens.

    Slots aggregate patch features via cross-attention.  During training a
    text-conditioned sigmoid gate up-weights slots relevant to the caption;
    at inference (no text) slots are mean-pooled uniformly so the student
    remains a pure visual encoder.
    """

    def __init__(self, timm_name: str, out_dim: int, num_slots: int = 8, num_heads: int = 4):
        super().__init__()
        timm = require_timm()
        self.backbone  = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = int(self.backbone.num_features)
        with torch.no_grad():
            _feats = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        if _feats.dim() != 3:
            raise RuntimeError(
                f"SemanticSlotStudent requires a ViT backbone, got shape {tuple(_feats.shape)}"
            )
        self.num_slots = num_slots
        self.slots     = nn.Parameter(torch.randn(num_slots, in_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(in_dim)
        self.gate_proj = nn.Linear(out_dim, num_slots)    # text emb → slot gate
        self.proj_cls   = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_patch = nn.Linear(in_dim, out_dim, bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def _aggregate(
        self, all_tok: torch.Tensor, text_cls: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (agg [B,D_in], slot_out [B,K,D_in], attn_weights [B,K,N])."""
        patches = all_tok[:, 1:, :]                                      # [B, N, D_in]
        B = patches.shape[0]
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)               # [B, K, D_in]
        slot_out, weights = self.slot_attn(slots, patches, patches)      # [B, K, D_in]
        slot_out = self.slot_norm(slot_out + slots)                       # residual
        if text_cls is not None:
            gate = torch.sigmoid(self.gate_proj(text_cls))               # [B, K]
            denom = gate.sum(-1, keepdim=True).clamp(min=1e-6)
            agg = (slot_out * gate.unsqueeze(-1)).sum(1) / denom         # [B, D_in]
        else:
            agg = slot_out.mean(1)                                        # [B, D_in]
        return agg, slot_out, weights

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        all_tok = self.backbone.forward_features(images)
        agg, _, _ = self._aggregate(all_tok)
        return F.normalize(self.proj_cls(agg), dim=-1)

    def forward_full(
        self, images: torch.Tensor, text_cls: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cls [B,D], patch_proj [B,N,D], slot_out [B,K,D_in], attn_weights [B,K,N])."""
        all_tok = self.backbone.forward_features(images)
        agg, slot_out, weights = self._aggregate(all_tok, text_cls)
        cls_       = F.normalize(self.proj_cls(agg), dim=-1)
        patch_proj = F.normalize(self.proj_patch(all_tok[:, 1:, :]), dim=-1)
        return cls_, patch_proj, slot_out, weights


# ── HierarchicalStudent ────────────────────────────────────────────────────────

class HierarchicalStudent(nn.Module):
    """ViT student with 3-scale patch pyramid: fine (14×14), mid (7×7), coarse (4×4).

    Each scale has its own projection head; a learnable softmax weight vector
    fuses the four representations (cls + fine + mid + coarse) into a single
    embedding for downstream distillation and probing.
    """

    def __init__(self, timm_name: str, out_dim: int):
        super().__init__()
        timm = require_timm()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = int(self.backbone.num_features)
        with torch.no_grad():
            _feats = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        if _feats.dim() != 3:
            raise RuntimeError(
                f"HierarchicalStudent requires a ViT backbone, got shape {tuple(_feats.shape)}"
            )
        self.proj_cls    = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_fine   = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_mid    = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_coarse = nn.Linear(in_dim, out_dim, bias=False)
        # Learnable scale-fusion weights (log-uniform init → equal start)
        self.scale_logits = nn.Parameter(torch.zeros(4))   # cls, fine, mid, coarse
        self.preprocess   = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward_pyramid(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (fused [B,D], cls_emb [B,D], fine_emb [B,D], mid_emb [B,D], coarse_emb [B,D])."""
        all_tok = self.backbone.forward_features(images)  # [B, N+1, in_dim]
        cls_raw = all_tok[:, 0, :]                         # [B, in_dim]
        patches = all_tok[:, 1:, :]                        # [B, 196, in_dim]
        B, N, C = patches.shape
        G = int(N ** 0.5)                                  # 14 for ViT-S/16

        grid = patches.reshape(B, G, G, C).permute(0, 3, 1, 2)          # [B, C, G, G]
        mid_raw    = F.adaptive_avg_pool2d(grid, 7).flatten(2).transpose(1, 2)   # [B, 49, C]
        coarse_raw = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)   # [B, 16, C]

        cls_emb    = F.normalize(self.proj_cls(cls_raw),            dim=-1)
        fine_emb   = F.normalize(self.proj_fine(patches.mean(1)),   dim=-1)
        mid_emb    = F.normalize(self.proj_mid(mid_raw.mean(1)),     dim=-1)
        coarse_emb = F.normalize(self.proj_coarse(coarse_raw.mean(1)), dim=-1)

        w = torch.softmax(self.scale_logits, dim=0)        # [4]
        fused = w[0]*cls_emb + w[1]*fine_emb + w[2]*mid_emb + w[3]*coarse_emb
        fused = F.normalize(fused, dim=-1)
        return fused, cls_emb, fine_emb, mid_emb, coarse_emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fused, *_ = self.forward_pyramid(images)
        return fused


# ── BottleneckStudent ──────────────────────────────────────────────────────────

class BottleneckStudent(nn.Module):
    """ViT student with a text-gated bottleneck adapter between backbone and projection.

    During training the text embedding produces a sigmoid gate that selects which
    bottleneck units are active, forcing the bottleneck to learn text-relevant
    visual features.  At inference no gate is applied — the bottleneck runs in
    full, yielding a pure visual encoder.
    """

    def __init__(self, timm_name: str, out_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        timm = require_timm()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        in_dim = int(self.backbone.num_features)
        with torch.no_grad():
            _feats = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        if _feats.dim() != 3:
            raise RuntimeError(
                f"BottleneckStudent requires a ViT backbone, got shape {tuple(_feats.shape)}"
            )
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
        )
        self.gate_proj  = nn.Linear(out_dim, bottleneck_dim)   # text emb → gate
        self.proj_cls   = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.proj_patch = nn.Linear(in_dim, out_dim, bias=False)
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        all_tok = self.backbone.forward_features(images)
        neck = self.bottleneck(all_tok[:, 0, :])             # [B, neck_dim]
        return F.normalize(self.proj_cls(neck), dim=-1)

    def forward_train(
        self, images: torch.Tensor, text_cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cls_gated [B,D], patch_proj [B,N,D], gate [B,neck_dim])."""
        all_tok = self.backbone.forward_features(images)
        neck       = self.bottleneck(all_tok[:, 0, :])       # [B, neck_dim]
        gate       = torch.sigmoid(self.gate_proj(text_cls)) # [B, neck_dim]
        neck_gated = neck * gate
        cls_       = F.normalize(self.proj_cls(neck_gated), dim=-1)
        patch_proj = F.normalize(self.proj_patch(all_tok[:, 1:, :]), dim=-1)
        return cls_, patch_proj, gate
