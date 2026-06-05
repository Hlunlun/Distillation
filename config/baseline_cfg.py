from __future__ import annotations

import argparse

from . import paths_cfg as PATHS


TEACHERS: list[dict] = [
    {
        "name": "BiomedCLIP ViT-B/16",
        "type": "teacher",
        "loader": "open_clip",
        "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    },
    {
        "name": "PubMedCLIP ViT-B/32",
        "type": "teacher",
        "loader": "hf_clip",
        "model_id": "flaviagiammarino/pubmed-clip-vit-base-patch32",
    },
    {
        "name": "RadCLIP",
        "type": "teacher",
        "loader": "pth_hf_clip",
        "model_id": "zluvolyote/RadCLIP",
        "pth_filename": "RadCLIP.pth",
        "base_model_id": "openai/clip-vit-large-patch14",
    },
    # OpenPMC-CLIP ships only a PyTorch Lightning .ckpt using mmlearn — not loadable without that dep.
    {
        "name": "PLIP",
        "type": "teacher",
        "loader": "hf_clip",
        "model_id": "vinid/plip",
    },
    {
        "name": "QuiltNet-B-32",
        "type": "teacher",
        "loader": "open_clip",
        "model_id": "hf-hub:wisdomik/QuiltNet-B-32",
    },
    {
        "name": "CLIP ViT-B/16 (OpenAI)",
        "type": "teacher",
        "loader": "hf_clip",
        "model_id": "openai/clip-vit-base-patch16",
    },
    {
        "name": "CLIP ViT-L/14 (OpenAI)",
        "type": "teacher",
        "loader": "hf_clip",
        "model_id": "openai/clip-vit-large-patch14",
    },
    {
        "name": "CLIP ViT-B/32 (LAION-2B)",
        "type": "teacher",
        "loader": "open_clip",
        "model_id": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    },
    {
        "name": "ResNet-50 (ImageNet)",
        "type": "student",
        "loader": "timm",
        "model_id": "resnet50",
    },
    {
        "name": "ResNet-18 (ImageNet)",
        "type": "student",
        "loader": "timm",
        "model_id": "resnet18",
    },
]

STUDENT: dict = {
    "name": "ViT-S/16 (ImageNet, no distill)",
    "type": "student",
    "loader": "timm",
    "model_id": "vit_small_patch16_224",
}


def add_baseline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chexpert_images_dir", default=PATHS.chexpert_images_dir)
    p.add_argument("--chexpert_train_csv", default=PATHS.chexpert_train_csv)
    p.add_argument("--chexpert_test_csv", default=PATHS.chexpert_test_csv)
    p.add_argument("--chexpert_uncertain_policy", default="zeros")
    p.add_argument("--run_nih14_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--deeplesion_dir", default=PATHS.deeplesion_dir)
    p.add_argument("--deeplesion_csv", default=None)
    p.add_argument("--run_deeplesion_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_lc25000_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--lc25000_dir", default=PATHS.lc25000_dir)
    p.add_argument("--run_chestmnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_pathmnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_dermamnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_octmnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_pneumoniamnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_organamnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--nih14_dir", default=None)
    p.add_argument("--nih14_images_dir", default=PATHS.nih14_images_dir)
    p.add_argument("--nih14_csv", default=PATHS.nih14_csv)
    p.add_argument("--nih14_train_val_list", default=PATHS.nih14_train_val_list)
    p.add_argument("--nih14_test_list", default=PATHS.nih14_test_list)
    p.add_argument("--pcam_dir", default=PATHS.pcam_dir)
    p.add_argument("--run_pcam_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--probe_max_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_dir", default="results")
    p.add_argument("--only", default="", help="Comma-separated model names to run (substring match).")
    p.add_argument("--tb_dir", default=None, help="TensorBoard log dir (default: <log_dir>/baseline_<ts>/tb)")
    p.add_argument("--attn_samples", type=int, default=4, help="Number of sample images for attention grid")
