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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(images)), dim=-1)


# ── teacher ────────────────────────────────────────────────────────────────────

class _HFCLIPPthAdapter(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self._clip = clip_model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self._clip.vision_model(pixel_values=images).pooler_output
        return self._clip.visual_projection(pooled)


class FrozenTeacher(nn.Module):
    def __init__(self, model_name: str, pretrained: Optional[str] = None, load_tokenizer: bool = True):
        super().__init__()
        if model_name.startswith("hf-pth-clip:"):
            model, preprocess = self._load_hf_clip_pth(model_name[len("hf-pth-clip:"):])
        else:
            open_clip = require_open_clip()
            model, preprocess = self._load_open_clip(open_clip, model_name, pretrained)

        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = preprocess

        if load_tokenizer and not model_name.startswith("hf-pth-clip:"):
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
        return _HFCLIPPthAdapter(clip), preprocess

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
