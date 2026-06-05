from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
def _macro_f1_recall_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    tn = ((y_true == 0) & (y_pred == 0)).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(prec + recall > 0, 2 * prec * recall / (prec + recall), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
    return float(np.mean(f1)), float(np.mean(recall)), float(np.mean(specificity))
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sdpa_context():
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor >= 120:  # Blackwell consumer (SM120+): no Flash/MemEfficient kernel
        return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    return torch.nn.attention.sdpa_kernel([
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ])
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.data_loaders import (
    MedMNISTDataset, NIHChestXray14Dataset, CheXpertDataset, DeepLesionDataset,
    LC25000Dataset, PCamDataset,
)
from experiments.models import (
    OpenCLIPEncoder, PthOpenCLIPEncoder, PthHFCLIPEncoder,
    HFCLIPEncoder, HFAutoVisionEncoder, TimmEncoder,
)


# ── Student loader ─────────────────────────────────────────────────────────────

def load_student(
    run_dir: str,
    ckpt: str = "best.pt",
    device: str = "cpu",
) -> tuple[nn.Module, argparse.Namespace, argparse.Namespace]:
    """Load a distilled student from a results run directory.

    Returns (student, args, exp_args). The student is in eval mode on `device`.
    Falls back to last.pt if best.pt does not exist.

    Usage:
        student, args, exp_args = load_student("results/multiteacher_20260517_115837")
        feats = student(images)  # normalized embeddings
    """
    from experiments.models import StudentDualHead, StudentVisualEncoder

    run_path = Path(run_dir)
    ckpt_path = run_path / ckpt
    if not ckpt_path.exists():
        fallback = run_path / ("last.pt" if ckpt == "best.pt" else "best.pt")
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**state["args"])
    exp_args = argparse.Namespace(**state.get("exp_args", {}))

    sd = state["student_state_dict"]
    timm_name = args.timm_student

    if "proj_primary.weight" in sd:
        out_dim_primary = sd["proj_primary.weight"].shape[0]
        out_dim_secondary = sd["proj_secondary.weight"].shape[0]
        student = StudentDualHead(timm_name, out_dim_primary, out_dim_secondary)
    else:
        out_dim = sd["proj.weight"].shape[0]
        student = StudentVisualEncoder(timm_name, out_dim)

    student.load_state_dict(sd)
    return student.to(device).eval(), args, exp_args


# ── Encoder loader ─────────────────────────────────────────────────────────────

def load_encoder(spec: dict, device: str = "cpu") -> nn.Module:
    loader = spec["loader"]
    model_id = spec["model_id"]
    if loader == "open_clip":
        enc = OpenCLIPEncoder(model_id)
    elif loader == "hf_clip":
        enc = HFCLIPEncoder(model_id)
    elif loader == "hf_auto":
        enc = HFAutoVisionEncoder(model_id)
    elif loader == "pth_open_clip":
        enc = PthOpenCLIPEncoder(model_id, spec["pth_filename"], spec["arch"])
    elif loader == "pth_hf_clip":
        enc = PthHFCLIPEncoder(model_id, spec["pth_filename"], spec["base_model_id"])
    elif loader == "timm":
        enc = TimmEncoder(model_id)
    else:
        raise ValueError(f"Unknown loader: {loader}")
    return enc.to(device)


# ── Linear probe ───────────────────────────────────────────────────────────────

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-8)


def _auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    tp = np.cumsum(y) / npos
    fp = np.cumsum(1 - y) / nneg
    return float(np.trapezoid(tp, fp))


def _train_linear_probe(
    X_tr: torch.Tensor,
    y_tr: np.ndarray,
    multilabel: bool,
    n_epochs: int = 30,
    lr: float = 1e-2,
    batch_size: int = 256,
) -> torch.nn.Linear:
    device = X_tr.device
    n, d = X_tr.shape
    n_classes = y_tr.shape[1] if multilabel else int(y_tr.max()) + 1
    linear = torch.nn.Linear(d, n_classes, bias=True).to(device)
    opt = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=1e-4)
    y = (torch.from_numpy(y_tr).float().to(device) if multilabel
         else torch.from_numpy(y_tr).long().to(device))
    bar = tqdm(range(n_epochs), desc="probe", leave=False)
    for _ in bar:
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            logits = linear(X_tr[idx])
            loss = (F.binary_cross_entropy_with_logits(logits, y[idx]) if multilabel
                    else F.cross_entropy(logits, y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        bar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")
    return linear


def _fit_linear_probe(
    X_tr: torch.Tensor,
    y_tr: np.ndarray,
    X_te: torch.Tensor,
    multilabel: bool,
    n_epochs: int = 30,
    lr: float = 1e-2,
    batch_size: int = 256,
) -> np.ndarray:
    linear = _train_linear_probe(X_tr, y_tr, multilabel, n_epochs, lr, batch_size)
    with torch.no_grad():
        return linear(X_te).cpu().numpy()


def _extract_features(
    model: nn.Module,
    ds: torch.utils.data.Dataset,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3)
    feats, labels = [], []
    with torch.no_grad(), _sdpa_context():
        for imgs, lbls in tqdm(loader, desc="extract", leave=False):
            f = model(imgs.to(device))
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
            elif hasattr(f, "last_hidden_state"):
                f = f.last_hidden_state[:, 0]
            torch.cuda.synchronize()
            feats.append(f)
            labels.append(lbls.cpu().numpy() if torch.is_tensor(lbls) else np.asarray(lbls))
    return F.normalize(torch.cat(feats, 0), dim=1), np.concatenate(labels, 0)


def _multilabel_auroc(
    full_ds: torch.utils.data.Dataset,
    model: nn.Module,
    device: str,
    batch_size: int,
    seed: int,
    dataset_name: str,
    n_classes: Optional[int] = None,
) -> tuple[dict, list]:
    """Returns (metrics_dict, te_idx) so callers can access the test split."""
    n = len(full_ds)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(n * 0.2))
    te_idx, tr_idx = perm[:n_test].tolist(), perm[n_test:].tolist()
    X_tr, y_tr = _extract_features(model, torch.utils.data.Subset(full_ds, tr_idx), device, batch_size)
    X_te, y_te = _extract_features(model, torch.utils.data.Subset(full_ds, te_idx), device, batch_size)
    if n_classes is not None:
        y_tr = np.eye(n_classes)[y_tr.astype(int)]
        y_te = np.eye(n_classes)[y_te.astype(int)]
    scores = _fit_linear_probe(X_tr, y_tr, X_te, multilabel=True)
    per_class = [_auroc_binary(y_te[:, k], scores[:, k]) for k in range(y_te.shape[1])]
    y_pred = (1.0 / (1.0 + np.exp(-scores)) >= 0.5).astype(int)
    y_te_int = y_te.astype(int)
    _f1, _recall, _spec = _macro_f1_recall_specificity(y_te_int, y_pred)
    metrics = {
        "macro_auroc": float(np.nanmean(per_class)) * 100.0,
        "per_class_auroc": per_class,
        "acc": float((y_pred == y_te_int).mean()) * 100.0,
        "macro_f1": _f1 * 100.0,
        "macro_recall": _recall * 100.0,
        "macro_specificity": _spec * 100.0,
        "dataset": dataset_name,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
    }
    return metrics, te_idx


def _multilabel_auroc_presplit(
    ds_train: torch.utils.data.Dataset,
    ds_test: torch.utils.data.Dataset,
    model: nn.Module,
    device: str,
    batch_size: int,
    dataset_name: str,
    n_classes: Optional[int] = None,
    ds_val: Optional[torch.utils.data.Dataset] = None,
) -> dict:
    X_tr, y_tr = _extract_features(model, ds_train, device, batch_size)
    X_te, y_te = _extract_features(model, ds_test, device, batch_size)
    if n_classes is not None:
        y_tr = np.eye(n_classes)[y_tr.astype(int)]
        y_te = np.eye(n_classes)[y_te.astype(int)]

    linear = _train_linear_probe(X_tr, y_tr, multilabel=True)

    def _eval(X: torch.Tensor, y_true: np.ndarray) -> dict:
        with torch.no_grad():
            scores = linear(X).cpu().numpy()
        per_class = [_auroc_binary(y_true[:, k], scores[:, k]) for k in range(y_true.shape[1])]
        y_pred = (1.0 / (1.0 + np.exp(-scores)) >= 0.5).astype(int)
        y_int = y_true.astype(int)
        _f1, _recall, _spec = _macro_f1_recall_specificity(y_int, y_pred)
        return {
            "macro_auroc": float(np.nanmean(per_class)) * 100.0,
            "per_class_auroc": per_class,
            "acc": float((y_pred == y_int).mean()) * 100.0,
            "macro_f1": _f1 * 100.0,
            "macro_recall": _recall * 100.0,
            "macro_specificity": _spec * 100.0,
        }

    test_m = _eval(X_te, y_te)
    result: dict = {
        "dataset": dataset_name,
        "n_train": len(ds_train),
        "n_test": len(ds_test),
        "test": test_m,
        **test_m,  # backward compat: top-level keys mirror test
    }

    if ds_val is not None:
        X_val, y_val = _extract_features(model, ds_val, device, batch_size)
        if n_classes is not None:
            y_val = np.eye(n_classes)[y_val.astype(int)]
        val_m = _eval(X_val, y_val)
        result["n_val"] = len(ds_val)
        result["val"] = val_m
        result["val_macro_auroc"] = val_m["macro_auroc"]

    return result


def _load_raw_pil(
    ds: torch.utils.data.Dataset,
    indices: list,
    n: int,
) -> list:
    """Load up to n raw PIL images from a dataset by index, using .items or .samples path lists."""
    imgs = []
    for i in indices:
        if len(imgs) >= n:
            break
        # MedMNIST: raw pixels stored in self.ds.imgs as numpy array [N, H, W] or [N, H, W, C]
        inner = getattr(ds, "ds", None)
        imgs_arr = getattr(inner, "imgs", None) if inner is not None else None
        if imgs_arr is not None:
            try:
                arr = imgs_arr[i]
                imgs.append(Image.fromarray(arr).convert("RGB"))
            except Exception:
                pass
            continue
        for attr in ("items", "samples"):
            pairs = getattr(ds, attr, None)
            if pairs is not None:
                try:
                    imgs.append(Image.open(pairs[i][0]).convert("RGB"))
                except Exception:
                    pass
                break
    return imgs


def _get_label_strings(ds: torch.utils.data.Dataset, indices: list) -> list[str]:
    """Return short readable label strings for the given dataset indices."""
    results = []
    label_names = getattr(ds, "LABELS_14", None) or getattr(ds, "LESION_TYPES", None) or getattr(ds, "label_names", None)
    is_multilabel = getattr(ds, "LABELS_14", None) is not None or getattr(ds, "multilabel", False)
    for i in indices:
        # MedMNIST: labels stored in inner ds.labels numpy array
        inner = getattr(ds, "ds", None)
        inner_labels = getattr(inner, "labels", None) if inner is not None else None
        if inner_labels is not None and i < len(inner_labels):
            lbl = inner_labels[i].squeeze()
            if is_multilabel and label_names:
                pos = [label_names[k] for k, v in enumerate(lbl) if v > 0.5]
                results.append("\n".join(pos[:3]) if pos else "No Finding")
            elif label_names:
                results.append(label_names[int(lbl)] if int(lbl) < len(label_names) else str(int(lbl)))
            else:
                results.append("")
            continue
        label = None
        for attr in ("items", "samples"):
            pairs = getattr(ds, attr, None)
            if pairs is not None and i < len(pairs):
                label = pairs[i][1]
                break
        if label is None or label_names is None:
            results.append("")
            continue
        if is_multilabel:
            pos = [label_names[k] for k, v in enumerate(label) if v > 0.5]
            results.append("\n".join(pos[:3]) if pos else "No Finding")
        else:
            idx = int(label)
            results.append(label_names[idx] if idx < len(label_names) else str(idx))
    return results


def _load_deeplesion_bbox_lookup(dl_info_csv: Path) -> dict[str, list[tuple]]:
    """Returns {filename: [(x1,y1,x2,y2)]} from DL_info.csv Bounding_boxes column."""
    lookup: dict[str, list] = {}
    if not dl_info_csv.exists():
        return lookup
    with dl_info_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("File_name") or "").strip()
            bbox_str = (row.get("Bounding_boxes") or "").strip()
            if not fname or not bbox_str:
                continue
            try:
                coords = [float(v) for v in bbox_str.split(",")]
            except ValueError:
                continue
            if len(coords) >= 4:
                lookup.setdefault(fname, []).append((coords[0], coords[1], coords[2], coords[3]))
    return lookup


def _load_nih14_bbox_lookup(bbox_csv: Path) -> dict[str, list[tuple]]:
    """Returns {filename: [(x1,y1,x2,y2), ...]} from BBox_List_2017.csv (x,y,w,h format)."""
    lookup: dict[str, list] = {}
    if not bbox_csv.exists():
        return lookup
    with bbox_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            fname = row[0].strip()
            try:
                x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            except (ValueError, IndexError):
                continue
            lookup.setdefault(fname, []).append((x, y, x + w, y + h))
    return lookup


def _bboxes_for_paths(paths: list, lookup: dict) -> list[list[tuple]]:
    """Map a list of image Paths to their bounding boxes from a lookup dict keyed by filename."""
    return [lookup.get(Path(p).name, []) for p in paths]


def _log_attn_comparison_from_probe(
    student: nn.Module,
    encoder_specs: list,
    raw_pil_imgs: list,
    device: str,
    writer,
    tag: str,
    step: int,
    n_samples: int,
    label_strings: Optional[list[str]] = None,
    bbox_list: Optional[list] = None,
) -> None:
    """
    Log attention comparison grid for the same images used in the linear probe test split.
    SOTA encoders are loaded on CPU one at a time (no GPU OOM risk regardless of what
    else is on the training device).  Student runs on its native device.
    """
    if not raw_pil_imgs:
        return
    named_heatmaps: list[tuple[str, list]] = []
    for spec in encoder_specs:
        try:
            enc = load_encoder(spec, device="cpu")
            hmaps = extract_attn_heatmaps(enc, raw_pil_imgs, "cpu", n=n_samples)
            named_heatmaps.append((spec["name"], hmaps))
            del enc
        except Exception as e:
            print(f"  [attn viz] skipping {spec['name']}: {e}", flush=True)

    # Student runs on its own device (GPU if training, unchanged)
    student_hmaps = extract_attn_heatmaps(student, raw_pil_imgs, device, n=n_samples)
    named_heatmaps.append(("student (ours)", student_hmaps))

    grid = compose_attention_grid(named_heatmaps, raw_pil_imgs, label_strings=label_strings, bbox_list=bbox_list)
    writer.add_image(tag, grid, global_step=step, dataformats="HWC")


def run_linear_probe(
    model: nn.Module,
    dataset_name: str,
    image_size: int,
    device: str,
    batch_size: int = 128,
    nih_data_dir: Optional[str] = None,
    nih_csv_path: Optional[str] = None,
    nih_images_dir: Optional[str] = None,
    nih14_train_val_list: Optional[str] = None,
    nih14_test_list: Optional[str] = None,
    chexpert_train_csv_path: Optional[str] = None,
    chexpert_test_csv_path: Optional[str] = None,
    chexpert_images_dir: Optional[str] = None,
    chexpert_uncertain_policy: str = "zeros",
    deeplesion_data_dir: Optional[str] = None,
    deeplesion_csv_path: Optional[str] = None,
    lc25000_dir: Optional[str] = None,
    pcam_dir: Optional[str] = None,
    image_transform=None,
    max_samples: Optional[int] = None,
    seed: int = 1337,
    # attention visualization (optional)
    writer=None,
    attn_encoder_specs: Optional[list] = None,
    attn_n_samples: int = 4,
    attn_tag: Optional[str] = None,
    attn_step: int = 0,
) -> dict:
    _MEDMNIST_DATASETS = {
        "chestmnist", "pathmnist", "dermamnist", "octmnist", "pneumoniamnist", "organamnist",
    }
    if dataset_name in _MEDMNIST_DATASETS:
        train_ds = MedMNISTDataset(dataset_name, split="train", image_size=image_size)
        val_ds   = MedMNISTDataset(dataset_name, split="val",   image_size=image_size)
        test_ds  = MedMNISTDataset(dataset_name, split="test",  image_size=image_size)
        n_cls = None if train_ds.multilabel else len(train_ds.label_names)
        result = _multilabel_auroc_presplit(
            train_ds, test_ds, model, device, batch_size, dataset_name, ds_val=val_ds,
            n_classes=n_cls,
        )
        if writer is not None and attn_encoder_specs:
            te_idx = list(range(min(attn_n_samples, len(test_ds))))
            _log_attn_comparison_from_probe(
                model, attn_encoder_specs, _load_raw_pil(test_ds, te_idx, attn_n_samples),
                device, writer, attn_tag or f"attention/{dataset_name}", attn_step, attn_n_samples,
                label_strings=_get_label_strings(test_ds, te_idx),
            )
        return result

    if dataset_name in ("nih14_auroc", "nih_cxr14_auroc"):
        if nih_data_dir is None and (nih_csv_path is None or nih_images_dir is None):
            raise ValueError("Provide nih_data_dir, or both nih_csv_path and nih_images_dir for NIH14.")
        if nih14_train_val_list is None or nih14_test_list is None:
            raise ValueError("Provide nih14_train_val_list and nih14_test_list for NIH14 probe.")
        all_tv_files = [l.strip() for l in Path(nih14_train_val_list).read_text().splitlines() if l.strip()]
        test_files   = [l.strip() for l in Path(nih14_test_list).read_text().splitlines() if l.strip()]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_tv_files))
        n_val = max(1, int(len(all_tv_files) * 0.2))
        val_files   = [all_tv_files[i] for i in perm[:n_val]]
        train_files = [all_tv_files[i] for i in perm[n_val:]]
        ds_train = NIHChestXray14Dataset(
            data_dir=nih_data_dir, csv_path=nih_csv_path, images_dir=nih_images_dir,
            image_size=image_size, max_samples=max_samples, seed=seed,
            transform=image_transform, split_list=train_files,
        )
        ds_val = NIHChestXray14Dataset(
            data_dir=nih_data_dir, csv_path=nih_csv_path, images_dir=nih_images_dir,
            image_size=image_size, seed=seed, transform=image_transform, split_list=val_files,
        )
        ds_test = NIHChestXray14Dataset(
            data_dir=nih_data_dir, csv_path=nih_csv_path, images_dir=nih_images_dir,
            image_size=image_size, seed=seed, transform=image_transform, split_list=test_files,
        )
        result = _multilabel_auroc_presplit(ds_train, ds_test, model, device, batch_size, dataset_name, ds_val=ds_val)
        if writer is not None and attn_encoder_specs:
            bbox_lookup: dict = {}
            if ds_test.samples:
                bbox_csv = Path(ds_test.samples[0][0]).parent.parent / "BBox_List_2017.csv"
                bbox_lookup = _load_nih14_bbox_lookup(bbox_csv)
            all_te = list(range(len(ds_test.samples)))
            if bbox_lookup:
                bbox_te = [i for i in all_te if ds_test.samples[i][0].name in bbox_lookup]
                non_bbox_te = [i for i in all_te if i not in set(bbox_te)]
                vis_idx = (bbox_te + non_bbox_te)[:attn_n_samples]
            else:
                vis_idx = all_te[:attn_n_samples]
            vis_paths = [ds_test.samples[i][0] for i in vis_idx]
            _log_attn_comparison_from_probe(
                model, attn_encoder_specs, _load_raw_pil(ds_test, vis_idx, attn_n_samples),
                device, writer, attn_tag or f"attention/{dataset_name}", attn_step, attn_n_samples,
                label_strings=_get_label_strings(ds_test, vis_idx),
                bbox_list=_bboxes_for_paths(vis_paths, bbox_lookup),
            )
        return result

    if dataset_name == "chexpert_auroc":
        if chexpert_train_csv_path is None or chexpert_test_csv_path is None or chexpert_images_dir is None:
            raise ValueError("Provide chexpert_train_csv_path, chexpert_test_csv_path, and chexpert_images_dir.")
        ds_train_full = CheXpertDataset(
            images_dir=chexpert_images_dir, csv_path=chexpert_train_csv_path,
            image_size=image_size, max_samples=max_samples, seed=seed,
            uncertain_policy=chexpert_uncertain_policy, transform=image_transform,
        )
        n = len(ds_train_full)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        n_val = max(1, int(n * 0.2))
        ds_val   = torch.utils.data.Subset(ds_train_full, perm[:n_val])
        ds_train = torch.utils.data.Subset(ds_train_full, perm[n_val:])
        ds_test = CheXpertDataset(
            images_dir=chexpert_images_dir, csv_path=chexpert_test_csv_path,
            image_size=image_size, seed=seed,
            uncertain_policy=chexpert_uncertain_policy, transform=image_transform,
        )
        result = _multilabel_auroc_presplit(ds_train, ds_test, model, device, batch_size, dataset_name, ds_val=ds_val)
        result["uncertain_policy"] = chexpert_uncertain_policy
        if writer is not None and attn_encoder_specs:
            vis_idx = list(range(min(attn_n_samples, len(ds_test.items))))
            _log_attn_comparison_from_probe(
                model, attn_encoder_specs, _load_raw_pil(ds_test, vis_idx, attn_n_samples),
                device, writer, attn_tag or f"attention/{dataset_name}", attn_step, attn_n_samples,
                label_strings=_get_label_strings(ds_test, vis_idx),
            )
        return result

    if dataset_name == "deeplesion_auroc":
        if deeplesion_data_dir is None:
            raise ValueError("Provide deeplesion_data_dir for DeepLesion linear probe.")
        ds_split2 = DeepLesionDataset(
            data_dir=deeplesion_data_dir, csv_path=deeplesion_csv_path,
            image_size=image_size, seed=seed, transform=image_transform, allowed_splits=[2],
        )
        n_split2 = len(ds_split2)
        perm2 = np.random.default_rng(seed).permutation(n_split2).tolist()
        n_val2 = max(1, int(0.2 * n_split2))
        ds_val   = torch.utils.data.Subset(ds_split2, perm2[:n_val2])
        ds_train = torch.utils.data.Subset(ds_split2, perm2[n_val2:])
        ds_test = DeepLesionDataset(
            data_dir=deeplesion_data_dir, csv_path=deeplesion_csv_path,
            image_size=image_size, seed=seed, transform=image_transform, allowed_splits=[3],
        )
        result = _multilabel_auroc_presplit(ds_train, ds_test, model, device, batch_size, dataset_name,
                                            n_classes=len(DeepLesionDataset.LESION_TYPES), ds_val=ds_val)
        if writer is not None and attn_encoder_specs:
            vis_idx = list(range(min(attn_n_samples, len(ds_test.items))))
            te_paths = [ds_test.items[i][0] for i in vis_idx]
            dl_bbox_lookup = _load_deeplesion_bbox_lookup(ds_test.data_dir / "DL_info.csv")
            dl_bboxes = [
                dl_bbox_lookup.get(f"{Path(p).parent.name}_{Path(p).stem}.png", [])
                for p in te_paths
            ]
            _log_attn_comparison_from_probe(
                model, attn_encoder_specs, _load_raw_pil(ds_test, vis_idx, attn_n_samples),
                device, writer, attn_tag or f"attention/{dataset_name}", attn_step, attn_n_samples,
                label_strings=_get_label_strings(ds_test, vis_idx),
                bbox_list=dl_bboxes,
            )
        return result

    if dataset_name in ("lc25000_lung", "lc25000_colon"):
        if lc25000_dir is None:
            raise ValueError("Provide lc25000_dir for LC25000 linear probe.")
        tissue = "lung" if dataset_name == "lc25000_lung" else "colon"
        n_classes = 3 if tissue == "lung" else 2
        ds_train = LC25000Dataset(
            root_dir=lc25000_dir, split="train", tissue=tissue,
            image_size=image_size, transform=image_transform,
        )
        ds_test = LC25000Dataset(
            root_dir=lc25000_dir, split="test", tissue=tissue,
            image_size=image_size, transform=image_transform,
        )
        n_tr = len(ds_train)
        perm = np.random.default_rng(seed).permutation(n_tr).tolist()
        n_val = max(1, int(n_tr * 0.2))
        ds_val   = torch.utils.data.Subset(ds_train, perm[:n_val])
        ds_train = torch.utils.data.Subset(ds_train, perm[n_val:])
        result = _multilabel_auroc_presplit(
            ds_train, ds_test, model, device, batch_size, dataset_name,
            n_classes=n_classes, ds_val=ds_val,
        )
        return result

    if dataset_name == "pcam":
        if pcam_dir is None:
            raise ValueError("Provide pcam_dir for PCam linear probe.")
        ds_train = PCamDataset(pcam_dir, split="train", image_size=image_size, transform=image_transform)
        ds_val   = PCamDataset(pcam_dir, split="val",   image_size=image_size, transform=image_transform)
        ds_test  = PCamDataset(pcam_dir, split="test",  image_size=image_size, transform=image_transform)
        return _multilabel_auroc_presplit(ds_train, ds_test, model, device, batch_size, dataset_name,
                                         n_classes=2, ds_val=ds_val)

    raise ValueError(f"Unknown dataset for linear probe: {dataset_name}")


# ── Attention visualization ────────────────────────────────────────────────────

def _extract_attn_weights(encoder: nn.Module, img_t: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """
    Returns raw attention weight tensor [heads, N, N] from the last ViT layer, or None.
    img_t: [1, C, H, W] already preprocessed, on CPU.
    Handles: TimmEncoder/StudentVisualEncoder (backbone.blocks), OpenCLIPEncoder/PthOpenCLIPEncoder
    (model.visual.transformer.resblocks or model.visual.blocks for EVA-ViT),
    HFCLIPEncoder/PthHFCLIPEncoder (model.vision_model with output_attentions).
    """
    img_t = img_t.to(device)
    captured: list[torch.Tensor] = []
    hooks: list = []
    saved: list[tuple] = []  # (module, attr, old_value) — restored in finally

    def _restore():
        for m, a, v in saved:
            setattr(m, a, v)
        for h in hooks:
            h.remove()

    def _attn_drop_hook(_mod, inp, _out):
        if inp and isinstance(inp[0], torch.Tensor) and inp[0].dim() == 4:
            captured.append(inp[0][0].detach().cpu())

    def _disable_fused(mod):
        if getattr(mod, "fused_attn", False):
            saved.append((mod, "fused_attn", True))
            mod.fused_attn = False

    # Case 1: timm ViT (TimmEncoder / StudentVisualEncoder / StudentDualHead)
    backbone = getattr(encoder, "backbone", None)
    if backbone is not None and hasattr(backbone, "blocks"):
        last_attn = getattr(backbone.blocks[-1], "attn", None)
        if last_attn is not None:
            _disable_fused(last_attn)
            attn_drop = getattr(last_attn, "attn_drop", None)
            if attn_drop is not None:
                hooks.append(attn_drop.register_forward_hook(_attn_drop_hook))

    # Case 2: open_clip ViT — standard CLIP (model.visual.transformer.resblocks)
    #          or EVA-ViT (model.visual.blocks)
    if not hooks and not saved:
        inner = getattr(encoder, "model", None)
        visual = getattr(inner, "visual", None) if inner is not None else None
        if visual is not None:
            resblocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
            blocks = (resblocks
                      or getattr(visual, "blocks", None)
                      or getattr(getattr(visual, "trunk", None), "blocks", None))
            if blocks is not None:
                last_attn = getattr(blocks[-1], "attn", None)
                if last_attn is not None:
                    _disable_fused(last_attn)
                    attn_drop = getattr(last_attn, "attn_drop", None)
                    if attn_drop is not None:
                        hooks.append(attn_drop.register_forward_hook(_attn_drop_hook))
                    elif isinstance(last_attn, nn.MultiheadAttention):
                        def _mha_pre(_mod, args, kwargs):
                            return args, {**kwargs, "need_weights": True, "average_attn_weights": False}
                        def _mha_post(_mod, _inp, out):
                            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                                w = out[1]
                                if w.dim() == 3:
                                    w = w.unsqueeze(1)
                                captured.append(w[0].detach().cpu())
                        hooks.append(last_attn.register_forward_pre_hook(_mha_pre, with_kwargs=True))
                        hooks.append(last_attn.register_forward_hook(_mha_post))

    # Case 3: HF CLIP (HFCLIPEncoder / PthHFCLIPEncoder) — output_attentions
    hf_vm = None
    if not hooks and not saved:
        inner = getattr(encoder, "model", None)
        vm = getattr(inner, "vision_model", None) if inner is not None else None
        if vm is not None:
            sub_vm = getattr(vm, "vision_model", None)
            enc = getattr(vm, "encoder", None) or getattr(sub_vm, "encoder", None)
            if hasattr(enc, "layers"):
                hf_vm = vm
                # SDPA blocks output_attentions — force eager before calling
                cfg = getattr(hf_vm, "config", None)
                if cfg is not None and getattr(cfg, "_attn_implementation", "eager") != "eager":
                    saved.append((cfg, "_attn_implementation", cfg._attn_implementation))
                    cfg._attn_implementation = "eager"

    # Case 4: CNN (ResNet-style) — GradCAM on last conv layer
    use_gradcam = False
    gradcam_state: dict = {"feat": None, "grad": None}
    if not hooks and not saved and hf_vm is None:
        layer4 = getattr(getattr(encoder, "backbone", None), "layer4", None)
        if layer4 is not None:
            use_gradcam = True
            def _gc_fwd(_mod, _inp, out):
                gradcam_state["feat"] = out
            def _gc_bwd(_mod, _gin, gout):
                gradcam_state["grad"] = gout[0]
            hooks.append(layer4.register_forward_hook(_gc_fwd))
            hooks.append(layer4.register_full_backward_hook(_gc_bwd))

    try:
        if use_gradcam:
            with torch.enable_grad():
                img_t = img_t.detach().requires_grad_(True)
                output = encoder(img_t)
                if isinstance(output, tuple):
                    output = output[0]
                output.mean().backward()
            feat = gradcam_state["feat"]
            grad = gradcam_state["grad"]
            if feat is not None and grad is not None:
                weights = grad.mean(dim=[2, 3], keepdim=True)
                cam = F.relu((weights * feat).sum(dim=1)).squeeze(0).detach().cpu()
                captured.append(cam)
        else:
            with torch.no_grad():
                if hf_vm is not None:
                    out = hf_vm(pixel_values=img_t, output_attentions=True)
                    if out.attentions:
                        captured.append(out.attentions[-1][0].detach().cpu())
                else:
                    encoder(img_t)
    finally:
        _restore()

    return captured[0] if captured else None


def _attn_to_heatmap(attn_heads: torch.Tensor, out_size: int = 224) -> np.ndarray:
    # GradCAM: already a [H, W] spatial map
    if attn_heads.dim() == 2:
        hmap = attn_heads.float().numpy()
    else:
        # ViT attention: [heads, N, N] — extract CLS→patch row
        avg = attn_heads.mean(0)       # [N, N]
        cls_attn = avg[0, 1:]          # [N_patches]
        n = cls_attn.shape[0]
        h = w = int(n ** 0.5)
        if h * w != n:
            return np.zeros((out_size, out_size), dtype=np.float32)
        hmap = cls_attn.float().numpy().reshape(h, w)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    hmap_pil = Image.fromarray((hmap * 255).astype(np.uint8)).resize(
        (out_size, out_size), Image.Resampling.BILINEAR
    )
    return np.array(hmap_pil).astype(np.float32) / 255.0


def _overlay_heatmap(img_np: np.ndarray, hmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import matplotlib.cm as cm
    colored = (cm.jet(hmap)[:, :, :3] * 255).astype(np.uint8)
    return np.clip(
        (1 - alpha) * img_np.astype(np.float32) + alpha * colored.astype(np.float32), 0, 255
    ).astype(np.uint8)


def log_attention_grid(
    writer,
    tag: str,
    named_encoders: list[tuple[str, nn.Module]],
    raw_pil_images: list,
    device: str,
    global_step: int,
    n_samples: int = 4,
    cell_px: int = 224,
) -> None:
    """
    Log an attention comparison grid to TensorBoard images.

    Grid layout (left→right per row): original | SOTA_1 | ... | ours (last encoder).
    Each row is one sample image.
    named_encoders: list of (label, encoder), SOTA first, student last.
                    Each encoder must have a .preprocess attribute.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = [img for img in raw_pil_images if img is not None][:n_samples]
    if not samples:
        return

    n_rows = len(samples)
    n_cols = 1 + len(named_encoders)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)

    col_titles = ["original"] + [name for name, _ in named_encoders]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, pad=3)

    for i, pil_img in enumerate(samples):
        img_np = np.array(pil_img.resize((cell_px, cell_px)).convert("RGB"))
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis("off")
        for j, (_name, enc) in enumerate(named_encoders):
            ax = axes[i, j + 1]
            preprocess = getattr(enc, "preprocess", None)
            attn_overlay = None
            if preprocess is not None:
                img_t = preprocess(pil_img).unsqueeze(0)
                attn = _extract_attn_weights(enc, img_t, device)
                if attn is not None:
                    attn_overlay = _overlay_heatmap(img_np, _attn_to_heatmap(attn, out_size=cell_px))
            if attn_overlay is not None:
                ax.imshow(attn_overlay)
            else:
                ax.imshow(img_np)
                ax.text(cell_px // 2, cell_px // 2, "N/A",
                        ha="center", va="center", fontsize=11, color="red",
                        transform=ax.transData)
            ax.axis("off")

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h_fig = fig.canvas.get_width_height()
    grid_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_fig, w, 4)[:, :, :3]
    plt.close(fig)
    writer.add_image(tag, grid_np, global_step=global_step, dataformats="HWC")


def extract_attn_heatmaps(
    encoder: nn.Module,
    raw_pil_images: list,
    device: str,
    n: int = 4,
    cell_px: int = 224,
) -> list:
    """
    Extract attention heatmaps from encoder for the first n raw PIL images.
    Returns list of Optional[np.ndarray] (HxW float [0,1]) — None where extraction failed.
    """
    results = []
    preprocess = getattr(encoder, "preprocess", None)
    for pil_img in raw_pil_images[:n]:
        if preprocess is None:
            results.append(None)
            continue
        img_t = preprocess(pil_img).unsqueeze(0)
        attn = _extract_attn_weights(encoder, img_t, device)
        results.append(_attn_to_heatmap(attn, out_size=cell_px) if attn is not None else None)
    return results


def compose_attention_grid(
    named_heatmaps: list[tuple[str, list]],
    raw_pil_images: list,
    cell_px: int = 224,
    label_strings: Optional[list[str]] = None,
    bbox_list: Optional[list] = None,
) -> np.ndarray:
    """
    Build a [H, W, 3] uint8 grid from pre-computed heatmaps.

    named_heatmaps: [(label, [heatmap_or_None, ...])], SOTA first, student last.
    raw_pil_images: original PIL images (same order as heatmaps).
    label_strings: optional per-row ground-truth label text shown on the original column.
    Grid: rows = samples, cols = [original, model_1, ..., model_N].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(raw_pil_images)
    n_cols = 1 + len(named_heatmaps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)

    col_titles = ["original"] + [name for name, _ in named_heatmaps]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, pad=3)

    for i, pil_img in enumerate(raw_pil_images):
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img.resize((cell_px, cell_px)).convert("RGB"))
        axes[i, 0].imshow(img_np)
        if label_strings and i < len(label_strings) and label_strings[i]:
            axes[i, 0].text(
                4, 4, label_strings[i],
                ha="left", va="top", fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"),
            )
        if bbox_list and i < len(bbox_list):
            from matplotlib.patches import Rectangle
            sx = cell_px / orig_w
            sy = cell_px / orig_h
            for (x1, y1, x2, y2) in bbox_list[i]:
                axes[i, 0].add_patch(Rectangle(
                    (x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy,
                    linewidth=1.5, edgecolor="lime", facecolor="none",
                ))
        axes[i, 0].axis("off")
        for j, (_name, hmaps) in enumerate(named_heatmaps):
            ax = axes[i, j + 1]
            hmap = hmaps[i] if i < len(hmaps) else None
            if hmap is not None:
                ax.imshow(_overlay_heatmap(img_np, hmap))
            else:
                ax.imshow(img_np)
                ax.text(cell_px // 2, cell_px // 2, "N/A",
                        ha="center", va="center", fontsize=11, color="red")
            ax.axis("off")

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h_fig = fig.canvas.get_width_height()
    grid_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_fig, w, 4)[:, :, :3]
    plt.close(fig)
    return grid_np


def sample_probe_images(
    chexpert_images_dir: Optional[str] = None,
    chexpert_csv: Optional[str] = None,
    nih14_images_dir: Optional[str] = None,
    nih14_csv: Optional[str] = None,
    n: int = 4,
) -> list:
    """Load n raw PIL images from CheXpert or NIH14 for visualization."""
    result = []
    if chexpert_images_dir and chexpert_csv and Path(chexpert_csv).exists():
        base = Path(chexpert_images_dir)
        with open(chexpert_csv, newline="") as f:
            for row in csv.DictReader(f):
                if len(result) >= n:
                    break
                rel = (row.get("Path") or "").strip()
                if not rel:
                    continue
                p = base / rel
                if p.exists():
                    result.append(Image.open(p).convert("RGB"))
    if len(result) < n and nih14_images_dir and nih14_csv and Path(nih14_csv).exists():
        img_dir = Path(nih14_images_dir)
        with open(nih14_csv, newline="") as f:
            for row in csv.DictReader(f):
                if len(result) >= n:
                    break
                name = (row.get("Image Index") or "").strip()
                p = img_dir / name
                if p.exists():
                    result.append(Image.open(p).convert("RGB"))
    return result
