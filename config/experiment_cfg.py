from __future__ import annotations

import argparse

from . import paths_cfg as PATHS


EXPERIMENTS: dict[str, dict] = {
    "pmc": {
        "module": "experiments.generated.distill_pmc_vits16",
        "args": [
            {"name": "tau_rel", "type": float, "default": 0.07},
            {"name": "w_rel",   "type": float, "default": 0.0},
        ],
    },
    "barlow": {
        "module": "experiments.generated.distill_barlow_vits16",
        "args": [
            {"name": "w_barlow",   "type": float, "default": 1.0},
            {"name": "lam_barlow", "type": float, "default": 5e-3},
        ],
    },
    "hardneg": {
        "module": "experiments.generated.distill_hardneg_vits16",
        "args": [
            {"name": "queue_size", "type": int, "default": 4096},
        ],
    },
    "multiteacher": {
        "module": "experiments.generated.distill_multiteacher_vits16",
        "args": [
            {"name": "teacher_primary",              "type": str,   "default": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"},
            {"name": "teacher_primary_pretrained",   "type": str,   "default": None},
            {"name": "teacher_secondary",            "type": str,   "default": "hf-pth-clip:zluvolyote/RadCLIP"},
            {"name": "teacher_secondary_pretrained", "type": str,   "default": None},
            {"name": "w_img_primary",                "type": float, "default": 1.0},
            {"name": "w_img_secondary",              "type": float, "default": 0.5},
        ],
    },
}


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pmc_oa_image_dir", default=PATHS.pmc_oa_image_dir)
    p.add_argument("--pmc_oa_jsonl", default=PATHS.pmc_oa_jsonl)
    p.add_argument("--pmc_qa_image_dir", default=PATHS.pmc_qa_image_dir)
    p.add_argument("--pmc_qa_train_csv", default=PATHS.pmc_qa_train_csv)
    p.add_argument("--pmc_qa_test_csv", default=PATHS.pmc_qa_test_csv)
    p.add_argument("--run_nih14_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_deeplesion_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--deeplesion_dir", default=PATHS.deeplesion_dir)
    p.add_argument("--nih14_dir", default=None)
    p.add_argument("--nih14_images_dir", default=PATHS.nih14_images_dir)
    p.add_argument("--nih14_csv", default=PATHS.nih14_csv)
    p.add_argument("--chexpert_images_dir", default=PATHS.chexpert_images_dir)
    p.add_argument("--chexpert_csv", default=PATHS.chexpert_csv)
    p.add_argument("--chexpert_uncertain_policy", default="zeros")
    p.add_argument("--probe_max_samples", type=int, default=50000)
    p.add_argument("--timm_student", default="vit_small_patch16_224")
    p.add_argument("--teacher_model", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--teacher_pretrained", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.15)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--tau_mc", type=float, default=0.07)
    p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--w_oa", type=float, default=1.0)
    p.add_argument("--w_mc", type=float, default=1.0)
    p.add_argument("--w_ce", type=float, default=0.0)
    p.add_argument("--log_dir", default="results")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--results_md", default="experiments.md")
    p.add_argument("--run_vlm_eval", action="store_true")
    p.add_argument("--pmcqa_eval_max_samples", type=int, default=5000)
    p.add_argument("--pmcoa_eval_samples", type=int, default=2000)
