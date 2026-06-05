from __future__ import annotations

import os
# os.environ.setdefault("MKL_THREADING_LAYER", "GNU")


import faulthandler
faulthandler.enable()
import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.utils import (
    load_encoder,
    run_linear_probe,
    log_attention_grid,
    sample_probe_images,
)
from config.baseline_cfg import TEACHERS, STUDENT, add_baseline_args


# ── Probe runner ──────────────────────────────────────────────────────────────

def probe_model(model, preprocess, device, args, writer=None, model_name="model") -> dict:
    result: dict = {}
    attn_kw = dict(
        writer=writer,
        attn_encoder_specs=[],
        attn_step=0,
    ) if writer is not None else {}
    chexpert_ok = (
        args.chexpert_images_dir
        and args.chexpert_train_csv
        and args.chexpert_test_csv
        and Path(args.chexpert_images_dir).exists()
        and Path(args.chexpert_train_csv).exists()
        and Path(args.chexpert_test_csv).exists()
    )
    if chexpert_ok:
        print("  Running CheXpert probe ...", flush=True)
        result["chexpert"] = run_linear_probe(
            model=model,
            dataset_name="chexpert_auroc",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            chexpert_images_dir=args.chexpert_images_dir,
            chexpert_train_csv_path=args.chexpert_train_csv,
            chexpert_test_csv_path=args.chexpert_test_csv,
            chexpert_uncertain_policy=args.chexpert_uncertain_policy,
            image_transform=preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
            attn_tag=f"attention/{model_name}/chexpert",
            **attn_kw,
        )
    else:
        result["chexpert"] = None

    if args.run_nih14_probe and args.nih14_csv and args.nih14_images_dir:
        print("  Running NIH14 probe ...", flush=True)
        result["nih14"] = run_linear_probe(
            model=model,
            dataset_name="nih14_auroc",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            nih_data_dir=args.nih14_dir,
            nih_csv_path=args.nih14_csv,
            nih_images_dir=args.nih14_images_dir,
            nih14_train_val_list=args.nih14_train_val_list,
            nih14_test_list=args.nih14_test_list,
            image_transform=preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
            attn_tag=f"attention/{model_name}/nih14",
            **attn_kw,
        )
    else:
        result["nih14"] = None

    if args.run_deeplesion_probe and args.deeplesion_dir and Path(args.deeplesion_dir).exists():
        print("  Running DeepLesion probe ...", flush=True)
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
            attn_tag=f"attention/{model_name}/deeplesion",
            **attn_kw,
        )
    else:
        result["deeplesion"] = None

    if args.run_chestmnist_probe:
        print("  Running ChestMNIST probe ...", flush=True)
        result["chestmnist"] = run_linear_probe(
            model=model,
            dataset_name="chestmnist",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            attn_tag=f"attention/{model_name}/chestmnist",
            **attn_kw,
        )
    else:
        result["chestmnist"] = None

    if args.run_lc25000_probe and args.lc25000_dir and Path(args.lc25000_dir).exists():
        for tissue_tag, ds_name in [("lung", "lc25000_lung"), ("colon", "lc25000_colon")]:
            print(f"  Running LC25000 ({tissue_tag}) probe ...", flush=True)
            result[ds_name] = run_linear_probe(
                model=model,
                dataset_name=ds_name,
                image_size=224,
                device=device,
                batch_size=args.batch_size,
                lc25000_dir=args.lc25000_dir,
                image_transform=preprocess,
                seed=args.seed,
                attn_tag=f"attention/{model_name}/{ds_name}",
                **attn_kw,
            )
    else:
        result["lc25000_lung"]  = None
        result["lc25000_colon"] = None

    if args.run_pcam_probe and args.pcam_dir and Path(args.pcam_dir).exists():
        print("  Running PCam probe ...", flush=True)
        result["pcam"] = run_linear_probe(
            model=model,
            dataset_name="pcam",
            image_size=224,
            device=device,
            batch_size=args.batch_size,
            pcam_dir=args.pcam_dir,
            image_transform=preprocess,
            seed=args.seed,
            attn_tag=f"attention/{model_name}/pcam",
            **attn_kw,
        )
    else:
        result["pcam"] = None

    _EXTRA_MEDMNIST = [
        ("pathmnist",      "run_pathmnist_probe"),
        ("dermamnist",     "run_dermamnist_probe"),
        ("octmnist",       "run_octmnist_probe"),
        ("pneumoniamnist", "run_pneumoniamnist_probe"),
        ("organamnist",    "run_organamnist_probe"),
    ]
    for ds_name, flag in _EXTRA_MEDMNIST:
        if getattr(args, flag, False):
            print(f"  Running {ds_name} probe ...", flush=True)
            result[ds_name] = run_linear_probe(
                model=model,
                dataset_name=ds_name,
                image_size=224,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                attn_tag=f"attention/{model_name}/{ds_name}",
                **attn_kw,
            )
        else:
            result[ds_name] = None

    return result


# ── Experiments.md updater ────────────────────────────────────────────────────

def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _ds_cols(d: dict) -> str:
    return (
        f"| {_fmt(d.get('macro_auroc'))} | {_fmt(d.get('acc'))} "
        f"| {_fmt(d.get('macro_f1'))} | {_fmt(d.get('macro_recall'))} | {_fmt(d.get('macro_specificity'))} "
    )


_DS_NAMES = [
    ("Chex",   "chexpert"),
    ("NIH14",  "nih14"),
    ("DL",     "deeplesion"),
    ("CM",     "chestmnist"),
    ("Path",   "pathmnist"),
    ("Derma",  "dermamnist"),
    ("OCT",    "octmnist"),
    ("Pneu",   "pneumoniamnist"),
    ("Organ",  "organamnist"),
    ("LC-Lung",  "lc25000_lung"),
    ("LC-Colon", "lc25000_colon"),
    ("PCam",   "pcam"),
]

_METRIC_SUFFIX = "AUROC | {p} ACC | {p} F1 | {p} Rec | {p} Spec "


def append_baseline_to_experiments_md(rows: list[dict], run_dir: Path) -> None:
    exp_md = _REPO_ROOT / ".claude" / "base_experiments.md"
    if not exp_md.exists():
        return
    header_cols = "".join(
        f"| {p} AUROC | {p} ACC | {p} F1 | {p} Rec | {p} Spec "
        for p, _ in _DS_NAMES
    )
    sep_cols = "|".join(["---"] * (5 * len(_DS_NAMES)))
    lines = [
        f"\n### Baseline Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Run dir: `{run_dir.name}`\n",
        f"| Model | Type {header_cols}|",
        f"|-------|------{sep_cols}|",
    ]
    for r in rows:
        row = f"| {r['name']} | {r['type']} "
        for _, key in _DS_NAMES:
            row += _ds_cols(r.get(key) or {})
        row += "|"
        lines.append(row)
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

    tb_dir = Path(args.tb_dir) if args.tb_dir else run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    only_filter = [s.strip().lower() for s in args.only.split(",") if s.strip()]
    models_to_run = [STUDENT] + TEACHERS
    if only_filter:
        models_to_run = [m for m in models_to_run if any(f in m["name"].lower() for f in only_filter)]
        if not models_to_run:
            raise ValueError(
                f"--only filter '{args.only}' matched no models. "
                f"Available: {[m['name'] for m in [STUDENT] + TEACHERS]}"
            )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(tb_dir)) if args.attn_samples > 0 else None

    all_results = []
    loaded_encoders: list[tuple[str, object]] = []  # (name, encoder) for attention grid
    _PRINT_DATASETS = [
        ("CheXpert",   "chexpert"),
        ("NIH14",      "nih14"),
        ("DeepLesion", "deeplesion"),
        ("ChestMNIST", "chestmnist"),
        ("PathMNIST",  "pathmnist"),
        ("DermaMNIST", "dermamnist"),
        ("OCT-MNIST",  "octmnist"),
        ("PneuMNIST",  "pneumoniamnist"),
        ("OrganMNIST", "organamnist"),
        ("LC25k-Lung", "lc25000_lung"),
        ("LC25k-Colon","lc25000_colon"),
        ("PCam",       "pcam"),
    ]
    _DS_HDR = f"  {'Dataset':<14} {'AUROC':>7} {'ACC':>7} {'F1':>7} {'Rec':>7} {'Spec':>7}"
    _DS_SEP = "  " + "-" * 51

    for spec in models_to_run:
        name = spec["name"]
        print(f"\n  Loading {name} ...", flush=True)
        model_result = {"name": name, "type": spec["type"], "model_id": spec["model_id"]}
        try:
            enc = load_encoder(spec, device=device)
            preprocess = enc.preprocess
            probes = probe_model(enc, preprocess, device, args, writer=writer, model_name=name)
            model_result.update(probes)
            model_result["error"] = None
            loaded_encoders.append((name, enc))
        except Exception as e:
            print(f"  FAILED: {e}")
            model_result["error"] = str(e)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        all_results.append(model_result)

        print(f"  {'─'*51}")
        print(_DS_HDR)
        print(_DS_SEP)
        for label, key in _PRINT_DATASETS:
            d = model_result.get(key) or {}
            if not d:
                continue
            print(
                f"  {label:<14} "
                f"{_fmt(d.get('macro_auroc')):>7} {_fmt(d.get('acc')):>7} "
                f"{_fmt(d.get('macro_f1')):>7} {_fmt(d.get('macro_recall')):>7} "
                f"{_fmt(d.get('macro_specificity')):>7}"
            )
        (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))

    (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {run_dir / 'metrics.json'}")

    # Attention comparison grid — all loaded encoders, student last
    if writer is not None and loaded_encoders:
        raw_imgs = sample_probe_images(
            chexpert_images_dir=args.chexpert_images_dir,
            chexpert_csv=args.chexpert_train_csv,
            nih14_images_dir=args.nih14_images_dir,
            nih14_csv=args.nih14_csv,
            n=args.attn_samples,
        )
        if raw_imgs:
            # Move student (non-teacher) to end; teachers first
            teachers_enc = [(n, e) for n, e in loaded_encoders if n != STUDENT["name"]]
            student_enc  = [(n, e) for n, e in loaded_encoders if n == STUDENT["name"]]
            ordered = teachers_enc + student_enc
            log_attention_grid(
                writer=writer,
                tag="attention/baseline_comparison",
                named_encoders=ordered,
                raw_pil_images=raw_imgs,
                device=device,
                global_step=0,
                n_samples=args.attn_samples,
            )
            print(f"Attention grid written to TensorBoard: {tb_dir}")
        writer.close()
    elif writer is not None:
        writer.close()

    append_baseline_to_experiments_md(all_results, run_dir)
    print(f"Appended to: .claude/base_experiments.md")


if __name__ == "__main__":
    main()
