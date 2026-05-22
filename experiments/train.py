from __future__ import annotations

import faulthandler
faulthandler.enable()

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")

import argparse
import functools
import importlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from experiments.data_loaders import (
    PMCOADataset, PMCQAChoicesDataset,
    collate_oa, collate_qa, collate_combined,
    HomogeneousBatchSampler,
)
from experiments.models import FrozenTeacher
from config.experiment_cfg import EXPERIMENTS, add_shared_args
from config.baseline_cfg import TEACHERS
from tools.reporting import write_results_md, print_table_header, print_run_row, COL_W
from tools.utils import run_linear_probe


def find_best_gpu() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() == 1:
        return "cuda:0"
    best_idx, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_idx = free, i
    print(f"[find_best_gpu] selected cuda:{best_idx} ({best_free / 1024**3:.1f} GB free)")
    return f"cuda:{best_idx}"


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_pmcqa_mc_accuracy(
    student: nn.Module,
    teacher: FrozenTeacher,
    image_dir: str,
    csv_path: str,
    max_samples: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    import torch
    ds = PMCQAChoicesDataset(image_dir=image_dir, csv_path=csv_path, max_samples=max_samples, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False,
        collate_fn=lambda ex: collate_qa(ex, teacher.preprocess)
    )
    total, correct = 0, 0
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        s_img = student(images)
        option_texts = []
        for q, ch in zip(batch.questions, batch.choices):
            option_texts.extend([f"{q} {c}" for c in ch])
        t_opt = teacher.encode_text(option_texts).view(images.shape[0], 4, -1)
        logits = torch.einsum("bd,bkd->bk", s_img, t_opt)
        labels = batch.labels.to(device)
        mask = labels >= 0
        if mask.any():
            pred = logits.argmax(dim=-1)
            correct += int((pred[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    acc = 0.0 if total == 0 else float(correct) / float(total) * 100.0
    return {"pmcqa_mc_acc": acc, "n": total}


@torch.no_grad()
def eval_pmcoa_retrieval_r1(
    student: nn.Module,
    teacher: FrozenTeacher,
    image_dir: str,
    jsonl_path: str,
    max_samples: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    import torch
    ds = PMCOADataset(image_dir=image_dir, jsonl_path=jsonl_path, max_samples=max_samples, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False,
        collate_fn=lambda ex: collate_oa(ex, teacher.preprocess),
    )
    img_embs, txt_embs = [], []
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        img_embs.append(student(images).cpu())
        txt_embs.append(teacher.encode_text(batch.captions).cpu())
    I = torch.cat(img_embs, dim=0)
    T_emb = torch.cat(txt_embs, dim=0)
    sims = I @ T_emb.t()
    ranks = sims.argsort(dim=-1, descending=True)
    gt = torch.arange(sims.shape[0]).unsqueeze(1)
    r1 = float((ranks[:, :1] == gt).float().mean().item() * 100.0)
    return {"pmcoa_i2t_r1": r1, "n": int(sims.shape[0])}


def run_eval_probes(
    student: nn.Module,
    teacher: FrozenTeacher,
    args: argparse.Namespace,
    writer: SummaryWriter,
    epoch: int,
    device: str,
) -> dict:
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    def _log_probe(name: str, probe: dict) -> None:
        prefix = f"eval/{name}"
        writer.add_scalar(f"{prefix}_macro_auroc", float(probe["macro_auroc"]), epoch)
        writer.add_scalar(f"{prefix}_macro_f1",    float(probe["macro_f1"]),    epoch)
        writer.add_scalar(f"{prefix}_macro_recall", float(probe["macro_recall"]), epoch)
        print(
            f"  [epoch {epoch}] {name:12s}  AUROC={probe['macro_auroc']:.2f}"
            f"  F1={probe['macro_f1']:.2f}  Recall={probe['macro_recall']:.2f}",
            flush=True,
        )

    nih14_probe = None
    if args.run_nih14_probe:
        print(f"\n  [epoch {epoch}] running NIH14 linear probe ...", flush=True)
        nih14_probe = run_linear_probe(
            model=student,
            dataset_name="nih14_auroc",
            image_size=224,
            device=device,
            batch_size=128,
            nih_data_dir=args.nih14_dir,
            nih_csv_path=args.nih14_csv,
            nih_images_dir=args.nih14_images_dir,
            image_transform=teacher.preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
            writer=writer,
            attn_encoder_specs=TEACHERS,
            attn_tag="attention/nih14",
            attn_step=epoch,
        )
        _log_probe("nih14", nih14_probe)

    chexpert_probe = None
    if (
        args.chexpert_images_dir
        and args.chexpert_csv
        and Path(args.chexpert_images_dir).exists()
        and Path(args.chexpert_csv).exists()
    ):
        print(f"  [epoch {epoch}] running CheXpert linear probe ...", flush=True)
        chexpert_probe = run_linear_probe(
            model=student,
            dataset_name="chexpert_auroc",
            image_size=224,
            device=device,
            batch_size=128,
            chexpert_images_dir=args.chexpert_images_dir,
            chexpert_csv_path=args.chexpert_csv,
            chexpert_uncertain_policy=args.chexpert_uncertain_policy,
            image_transform=teacher.preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
            writer=writer,
            attn_encoder_specs=TEACHERS,
            attn_tag="attention/chexpert",
            attn_step=epoch,
        )
        _log_probe("chexpert", chexpert_probe)

    deeplesion_probe = None
    if (
        args.run_deeplesion_probe
        and args.deeplesion_dir
        and Path(args.deeplesion_dir).exists()
    ):
        print(f"  [epoch {epoch}] running DeepLesion linear probe ...", flush=True)
        deeplesion_probe = run_linear_probe(
            model=student,
            dataset_name="deeplesion_auroc",
            image_size=224,
            device=device,
            batch_size=128,
            deeplesion_data_dir=args.deeplesion_dir,
            image_transform=teacher.preprocess,
            max_samples=args.probe_max_samples,
            seed=args.seed,
            writer=writer,
            attn_encoder_specs=TEACHERS,
            attn_tag="attention/deeplesion",
            attn_step=epoch,
        )
        _log_probe("deeplesion", deeplesion_probe)

    chestmnist_probe = None
    if args.run_chestmnist_probe:
        print(f"  [epoch {epoch}] running ChestMNIST linear probe ...", flush=True)
        chestmnist_probe = run_linear_probe(
            model=student,
            dataset_name="chestmnist",
            image_size=224,
            device=device,
            batch_size=128,
            seed=args.seed,
            writer=writer,
            attn_encoder_specs=TEACHERS,
            attn_tag="attention/chestmnist",
            attn_step=epoch,
        )
        _log_probe("chestmnist", chestmnist_probe)

    return {
        "nih14_probe": nih14_probe,
        "chexpert_probe": chexpert_probe,
        "deeplesion_probe": deeplesion_probe,
        "chestmnist_probe": chestmnist_probe,
    }



# ── training ───────────────────────────────────────────────────────────────────

def make_run_dir(root: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loader(args: argparse.Namespace, teacher: FrozenTeacher) -> DataLoader:
    from torch.utils.data import ConcatDataset
    oa_ds = PMCOADataset(image_dir=args.pmc_oa_image_dir, jsonl_path=args.pmc_oa_jsonl, seed=args.seed)
    qa_ds = PMCQAChoicesDataset(image_dir=args.pmc_qa_image_dir, csv_path=args.pmc_qa_train_csv, seed=args.seed)
    sampler = HomogeneousBatchSampler(len(oa_ds), len(qa_ds), args.batch_size, seed=args.seed)
    preprocess = teacher.preprocess
    return DataLoader(
        ConcatDataset([oa_ds, qa_ds]),
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=functools.partial(collate_combined, preprocess=preprocess),
        persistent_workers=False,
    )


def train_loop(
    student: nn.Module,
    teacher: FrozenTeacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    writer: SummaryWriter,
    args: argparse.Namespace,
    exp_args: argparse.Namespace,
    compute_loss: Callable,
    run_dir: Path,
    device: str,
    probe_model: nn.Module,
) -> tuple[dict, dict]:
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if warmup_steps <= 0:
            return args.lr
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * t))

    best_metric = -1.0
    best_ckpt = None
    last_ckpt = run_dir / "last.pt"
    metrics_path = run_dir / "metrics.json"
    global_step = 0
    probe_result: dict = {}

    epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="ep", leave=True)
    for epoch in epoch_bar:
        student.train()
        batch_bar = tqdm(loader, desc=f"ep{epoch}", unit="batch", leave=False)
        for batch in batch_bar:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                loss_dict = compute_loss(student, teacher, batch, args, exp_args, device)
            for k, v in loss_dict.items():
                writer.add_scalar(k, float(v.detach().cpu() if isinstance(v, torch.Tensor) else v), global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            scaler.scale(loss_dict["loss/total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            loss_val = float(loss_dict["loss/total"].detach().cpu())
            batch_bar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        batch_bar.close()

        student.eval()
        
        probe_result = run_eval_probes(probe_model, teacher, args, writer, epoch, device)

        pmcqa_eval = eval_pmcqa_mc_accuracy(
            student=probe_model,
            teacher=teacher,
            image_dir=args.pmc_qa_image_dir,
            csv_path=args.pmc_qa_test_csv,
            max_samples=args.pmcqa_eval_max_samples,
            batch_size=min(64, args.batch_size),
            device=device,
            seed=args.seed,
        )
        writer.add_scalar("eval/pmcqa_mc_acc", float(pmcqa_eval["pmcqa_mc_acc"]), epoch)

        vlm_eval = None
        if args.run_vlm_eval:
            vlm_eval = {
                "pmcqa_test": pmcqa_eval,
                "pmcoa_retrieval": eval_pmcoa_retrieval_r1(
                    student=probe_model,
                    teacher=teacher,
                    image_dir=args.pmc_oa_image_dir,
                    jsonl_path=args.pmc_oa_jsonl,
                    max_samples=args.pmcoa_eval_samples,
                    batch_size=min(64, args.batch_size),
                    device=device,
                    seed=args.seed,
                ),
            }
            writer.add_scalar("eval/pmcoa_i2t_r1", float(vlm_eval["pmcoa_retrieval"]["pmcoa_i2t_r1"]), epoch)

        tracked = None
        if probe_result["chexpert_probe"] is not None:
            tracked = float(probe_result["chexpert_probe"]["macro_auroc"])
        elif probe_result["nih14_probe"] is not None:
            tracked = float(probe_result["nih14_probe"]["macro_auroc"])

        if tracked is not None and tracked > best_metric:
            best_metric = tracked
            best_ckpt = run_dir / "best.pt"
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "args": vars(args),
                    "exp_args": vars(exp_args),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metric": best_metric,
                },
                best_ckpt,
            )

        torch.save(
            {"student_state_dict": student.state_dict(), "args": vars(args), "exp_args": vars(exp_args), "epoch": epoch, "global_step": global_step},
            last_ckpt,
        )

        epoch_bar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)

        existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append({"epoch": epoch, **probe_result, "pmcqa_eval": pmcqa_eval, "vlm_eval": vlm_eval})
        metrics_path.write_text(json.dumps(existing, indent=2))

    writer.close()
    summary = {
        "best_metric": best_metric,
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "run_dir": str(run_dir),
        "last_ckpt": str(last_ckpt),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, probe_result


# ── entry point ────────────────────────────────────────────────────────────────

def _build_exp_args(config: dict, remaining: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for spec in config.get("args", []):
        p.add_argument(f"--{spec['name']}", type=spec.get("type"), default=spec["default"])
    args, _ = p.parse_known_args(remaining)
    return args


def _run_one(name: str, config: dict, shared_args: argparse.Namespace, remaining: list[str], device: str) -> None:
    exp_args = _build_exp_args(config, remaining)
    mod = importlib.import_module(config["module"])
    student, teacher, compute_loss = mod.init(shared_args, exp_args, device)
    probe_model = mod.get_probe_model(student) if hasattr(mod, "get_probe_model") else student
    loader = build_loader(shared_args, teacher)
    optimizer = torch.optim.AdamW(student.parameters(), lr=shared_args.lr, weight_decay=shared_args.weight_decay)
    scaler = GradScaler(device.split(":")[0], enabled=device.startswith("cuda"))
    run_dir = make_run_dir(shared_args.log_dir, name)
    (run_dir / "config.json").write_text(
        json.dumps({"experiment": name, **vars(shared_args), "exp_args": vars(exp_args)}, indent=2)
    )
    writer = SummaryWriter(str(run_dir / "tb"))
    try:
        summary, last_probe = train_loop(
            student, teacher, loader,
            optimizer, scaler, writer, shared_args, exp_args,
            compute_loss, run_dir, device, probe_model,
        )
        print_run_row(run_dir.name, last_probe)
        write_results_md(shared_args.log_dir, shared_args.results_md)
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"  {'FAILED ' + name:<{COL_W - 2}} {'ERROR':>14} {str(e)[:40]}")
    finally:
        loader = optimizer = scaler = student = teacher = probe_model = None
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], required=True)
    add_shared_args(p)
    shared_args, remaining = p.parse_known_args()

    torch.manual_seed(shared_args.seed)
    device = find_best_gpu()


    to_run = EXPERIMENTS if shared_args.experiment == "all" else {shared_args.experiment: EXPERIMENTS[shared_args.experiment]}

    print_table_header()
    for name, config in to_run.items():
        print(f"  Training {name} ...", flush=True)
        try:
            _run_one(name, config, shared_args, remaining, device)
        except Exception as e:
            print(f"  {'FAILED ' + name:<{COL_W - 2}} {'ERROR':>14} {str(e)[:40]}")


if __name__ == "__main__":


    main()
