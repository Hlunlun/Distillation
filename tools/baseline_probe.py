from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.data_loaders import run_linear_probe
from config.baseline_cfg import TEACHERS, STUDENT, add_baseline_args

# ── Model definitions ─────────────────────────────────────────────────────────

class OpenCLIPEncoder(nn.Module):
    """Wraps an open_clip teacher to expose a single forward(images) → embeddings."""

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


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def _clip_preprocess(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


class PthOpenCLIPEncoder(nn.Module):
    """Loads an open_clip-compatible model from a HF-hosted raw .pth checkpoint.

    RadCLIP ships only RadCLIP.pth (no config); we instantiate the matching
    open_clip arch and transplant the weights.
    """

    def __init__(self, hf_repo: str, pth_filename: str, arch: str):
        super().__init__()
        import open_clip
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(hf_repo, pth_filename)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            # Common checkpoint wrappers
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break

        model, _, preprocess = open_clip.create_model_and_transforms(arch)
        missing, unexpected = model.load_state_dict(state, strict=False)
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
    """Loads a HF-CLIPModel-format .pth from a HuggingFace repo.

    RadCLIP.pth uses HF key names (vision_model.*, text_model.*) but ships without
    the config/tokenizer files needed for from_pretrained(), so we borrow the
    architecture from a matching HF base model and transplant the weights.
    """

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
            if isinstance(size, dict):
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
    """Wraps a HuggingFace transformers CLIPModel (vinid/plip, zluvolyote/RadCLIP, etc.)."""

    def __init__(self, model_id: str):
        super().__init__()
        from transformers import CLIPModel, CLIPImageProcessor

        # Some repos (RadCLIP) omit preprocessor_config.json; fall back to CLIP defaults.
        try:
            proc = CLIPImageProcessor.from_pretrained(model_id)
            size = proc.size
            if isinstance(size, dict):
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

        # Use vision_model directly — get_image_features() may return a dataclass
        # on some HF model versions rather than a plain tensor.
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
    """Wraps a HuggingFace AutoModel with a vision tower (VisionTextDualEncoder, etc.)."""

    def __init__(self, model_id: str):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor

        # Some repos (open-pmc-clip) omit preprocessor_config.json; fall back to CLIP defaults.
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
    """Wraps a timm backbone (num_classes=0) for frozen linear probe evaluation."""

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


# ── Probe runner ──────────────────────────────────────────────────────────────

def probe_model(
    model: nn.Module,
    preprocess,
    device: str,
    args: argparse.Namespace,
) -> dict:
    """Run CheXpert and NIH14 probes for one model. Returns result dict."""
    model = model.to(device)
    result: dict = {}

    chexpert_ok = (
        args.chexpert_images_dir
        and args.chexpert_csv
        and Path(args.chexpert_images_dir).exists()
        and Path(args.chexpert_csv).exists()
    )
    if chexpert_ok:
        result["chexpert"] = run_linear_probe(
            model=model,
            dataset_name="chexpert_auroc",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            chexpert_images_dir=args.chexpert_images_dir,
            chexpert_csv_path=args.chexpert_csv,
            chexpert_uncertain_policy=args.chexpert_uncertain_policy,
            image_transform=preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
        )
    else:
        result["chexpert"] = None

    if args.run_nih14_probe:
        nih14_ok = (args.nih14_csv and args.nih14_images_dir)
        if nih14_ok:
            result["nih14"] = run_linear_probe(
                model=model,
                dataset_name="nih14_auroc",
                image_size=224,
                device=device,
                batch_size=args.batch_size,
                nih_data_dir=args.nih14_dir,
                nih_csv_path=args.nih14_csv,
                nih_images_dir=args.nih14_images_dir,
                image_transform=preprocess,
                max_samples=args.probe_max_samples,
                seed=args.seed,
            )
        else:
            result["nih14"] = None
    else:
        result["nih14"] = None

    if args.run_deeplesion_probe and args.deeplesion_dir and Path(args.deeplesion_dir).exists():
        result["deeplesion"] = run_linear_probe(
            model=model,
            dataset_name="deeplesion_auroc",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            deeplesion_data_dir=args.deeplesion_dir,
            deeplesion_csv_path=args.deeplesion_csv,
            image_transform=preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
        )
    else:
        result["deeplesion"] = None

    return result


# ── Experiments.md updater ────────────────────────────────────────────────────

def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def append_baseline_to_experiments_md(rows: list[dict], run_dir: Path) -> None:
    exp_md = _REPO_ROOT / ".claude" / "base_experiments.md"
    if not exp_md.exists():
        return

    lines = [
        f"\n### Baseline Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Run dir: `{run_dir.name}`\n",
        "| Model | Type | CheXpert AUROC | NIH14 AUROC | DeepLesion AUROC |",
        "|-------|------|---------------|-------------|-----------------|",
    ]
    for r in rows:
        chex = _fmt((r.get("chexpert") or {}).get("macro_auroc"))
        nih = _fmt((r.get("nih14") or {}).get("macro_auroc"))
        dl = _fmt((r.get("deeplesion") or {}).get("macro_auroc"))
        lines.append(f"| {r['name']} | {r['type']} | {chex} | {nih} | {dl} |")

    with exp_md.open("a") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    add_baseline_args(p)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / f"baseline_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    only_filter = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    models_to_run = [STUDENT] + TEACHERS
    if only_filter:
        models_to_run = [m for m in models_to_run if any(f in m["name"].lower() for f in only_filter)]
        if not models_to_run:
            raise ValueError(f"--only filter '{args.only}' matched no models. Available: {[m['name'] for m in [STUDENT] + TEACHERS]}")

    all_results = []
    col_w = 36

    print(f"\n{'Model':<{col_w}} {'CheXpert AUROC':>14} {'NIH14 AUROC':>11} {'DeepLesion AUROC':>16}")
    print("-" * (col_w + 45))

    for spec in models_to_run:
        name = spec["name"]
        print(f"  Loading {name} ...", flush=True)

        model_result = {"name": name, "type": spec["type"], "model_id": spec["model_id"]}

        try:
            if spec["loader"] == "open_clip":
                enc = OpenCLIPEncoder(spec["model_id"])
            elif spec["loader"] == "hf_clip":
                enc = HFCLIPEncoder(spec["model_id"])
            elif spec["loader"] == "hf_auto":
                enc = HFAutoVisionEncoder(spec["model_id"])
            elif spec["loader"] == "pth_open_clip":
                enc = PthOpenCLIPEncoder(spec["model_id"], spec["pth_filename"], spec["arch"])
            elif spec["loader"] == "pth_hf_clip":
                enc = PthHFCLIPEncoder(spec["model_id"], spec["pth_filename"], spec["base_model_id"])
            else:
                enc = TimmEncoder(spec["model_id"])
            preprocess = enc.preprocess

            probes = probe_model(enc, preprocess, device, args)
            model_result.update(probes)
            model_result["error"] = None

        except Exception as e:
            print(f"  FAILED: {e}")
            model_result["error"] = str(e)
            model_result["chexpert"] = None
            model_result["nih14"] = None

        all_results.append(model_result)

        chex = _fmt((model_result.get("chexpert") or {}).get("macro_auroc"))
        nih = _fmt((model_result.get("nih14") or {}).get("macro_auroc"))
        dl = _fmt((model_result.get("deeplesion") or {}).get("macro_auroc"))
        print(f"  {name:<{col_w-2}} {chex:>14} {nih:>11} {dl:>16}")

        # Save incrementally so partial results survive interruptions.
        (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))

    (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {run_dir / 'metrics.json'}")

    append_baseline_to_experiments_md(all_results, run_dir)
    print(f"Appended to: .claude/base_experiments.md")


if __name__ == "__main__":
    main()
