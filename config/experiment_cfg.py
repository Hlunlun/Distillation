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
    "barlow_patch": {
        "module": "experiments.generated.distill_barlow_patch_vits16",
        "args": [
            {"name": "w_barlow",   "type": float, "default": 1.0},
            {"name": "lam_barlow", "type": float, "default": 5e-3},
            {"name": "w_patch",    "type": float, "default": 1.0},
        ],
    },
    "rrc_cosmos": {
        "module": "experiments.generated.distill_rrc_cosmos_vits16",
        "args": [
            {"name": "ema_momentum",  "type": float, "default": 0.999},
            {"name": "rrc_scale_min", "type": float, "default": 0.2},
            {"name": "rrc_scale_max", "type": float, "default": 1.0},
        ],
    },
    "cosmos": {
        "module": "experiments.generated.distill_cosmos_vits16",
        "args": [
            {"name": "tgac_k",        "type": int,   "default": 4},
            {"name": "num_crops",     "type": int,   "default": 2},
            {"name": "ema_momentum",  "type": float, "default": 0.999},
            {"name": "w_lg",          "type": float, "default": 1.0},
            {"name": "w_crop",        "type": float, "default": 0.5},
            {"name": "timm_student",  "type": str,   "default": "vit_small_patch16_224"},
        ],
    },
    "alignkd_cosmos": {
        "module": "experiments.generated.distill_alignkd_cosmos_vits16",
        "args": [
            {"name": "tgac_k",            "type": int,   "default": 4},
            {"name": "num_crops",         "type": int,   "default": 2},
            {"name": "ema_momentum",      "type": float, "default": 0.999},
            {"name": "w_lg",              "type": float, "default": 1.0},
            {"name": "w_crop",            "type": float, "default": 0.5},
            {"name": "w_layer",           "type": float, "default": 1.0},
            {"name": "top_k_layers",      "type": int,   "default": 2},
            {"name": "analysis_n_images", "type": int,   "default": 128},
            {"name": "w_tqva",            "type": float, "default": 1.0},
        ],
    },
    "medaug_cosmos": {
        "module": "experiments.generated.distill_medaug_cosmos_vits16",
        "args": [
            {"name": "tgac_k",            "type": int,   "default": 4},
            {"name": "num_crops",         "type": int,   "default": 2},
            {"name": "ema_momentum",      "type": float, "default": 0.999},
            {"name": "w_lg",              "type": float, "default": 1.0},
            {"name": "w_crop",            "type": float, "default": 0.5},
            {"name": "w_layer",           "type": float, "default": 1.0},
            {"name": "top_k_layers",      "type": int,   "default": 2},
            {"name": "analysis_n_images", "type": int,   "default": 128},
            {"name": "w_tqva",            "type": float, "default": 1.0},
            # ── medaug-specific ──────────────────────────────────────────────
            {"name": "intensity_scale",   "type": float, "default": 0.1},
            {"name": "intensity_shift",   "type": float, "default": 0.05},
            {"name": "p_channel_cutmix",  "type": float, "default": 0.0},
            {"name": "mask_floor",        "type": float, "default": 0.1},
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
    # ── thesis contribution experiments ──────────────────────────────────────
    "slot": {
        "module": "experiments.generated.distill_slot_vits16",
        "args": [
            {"name": "num_slots",    "type": int,   "default": 8},
            {"name": "num_heads",    "type": int,   "default": 4},
            {"name": "w_slot_div",   "type": float, "default": 0.1},
            {"name": "timm_student", "type": str,   "default": "vit_small_patch16_224"},
        ],
    },
    "hier": {
        "module": "experiments.generated.distill_hier_vits16",
        "args": [
            {"name": "w_scale",      "type": float, "default": 1.0},
            {"name": "timm_student", "type": str,   "default": "vit_small_patch16_224"},
        ],
    },
    "tgba": {
        "module": "experiments.generated.distill_tgba_vits16",
        "args": [
            {"name": "bottleneck_dim", "type": int,   "default": 256},
            {"name": "w_sparsity",     "type": float, "default": 0.01},
            {"name": "timm_student",   "type": str,   "default": "vit_small_patch16_224"},
        ],
    },
}


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pmc_oa_image_dir", default=PATHS.pmc_oa_image_dir)
    p.add_argument("--pmc_oa_train_jsonl", default=PATHS.pmc_oa_train_jsonl)
    p.add_argument("--pmc_oa_test_jsonl",  default=PATHS.pmc_oa_test_jsonl)
    p.add_argument("--pmc_qa_image_dir", default=PATHS.pmc_qa_image_dir)
    p.add_argument("--pmc_qa_train_csv", default=PATHS.pmc_qa_train_csv)
    p.add_argument("--pmc_qa_test_csv", default=PATHS.pmc_qa_test_csv)
    p.add_argument("--run_nih14_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_deeplesion_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_chestmnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_pathmnist_probe", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_dermamnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_octmnist_probe", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_pneumoniamnist_probe", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--run_organamnist_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--deeplesion_dir", default=PATHS.deeplesion_dir)
    p.add_argument("--run_lc25000_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--lc25000_dir", default=PATHS.lc25000_dir)
    p.add_argument("--run_pcam_probe", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--pcam_dir", default=PATHS.pcam_dir)
    p.add_argument("--nih14_dir", default=None)
    p.add_argument("--nih14_images_dir", default=PATHS.nih14_images_dir)
    p.add_argument("--nih14_csv", default=PATHS.nih14_csv)
    p.add_argument("--nih14_train_val_list", default=PATHS.nih14_train_val_list)
    p.add_argument("--nih14_test_list", default=PATHS.nih14_test_list)
    p.add_argument("--chexpert_images_dir", default=PATHS.chexpert_images_dir)
    p.add_argument("--chexpert_train_csv", default=PATHS.chexpert_train_csv)
    p.add_argument("--chexpert_test_csv", default=PATHS.chexpert_test_csv)
    p.add_argument("--chexpert_uncertain_policy", default="zeros")
    p.add_argument("--probe_max_samples", type=int, default=50000)
    p.add_argument("--timm_student", default="vit_small_patch16_224", choices=["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224", "resnet50"])
    p.add_argument("--teacher_model", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        choices=[
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "hf-hub:wisdomik/QuiltNet-B-32",
            "hf-hub:MahmoodLab/conch",       # license-gated: hf.co/MahmoodLab/conch
            "hf-pth-clip:zluvolyote/RadCLIP", # image-only, no OA loss
        ])
    p.add_argument("--teacher_pretrained", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=280)
    p.add_argument("--num_workers", type=int, default=4)
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
    p.add_argument("--data_sources", default="both", choices=["oa", "qa", "both"],
        help="Training data: 'oa' (PMC-OA only), 'qa' (PMC-QA only), 'both' (default).")
    p.add_argument("--run_vlm_eval", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--pmcqa_eval_max_samples", type=int, default=5000)
