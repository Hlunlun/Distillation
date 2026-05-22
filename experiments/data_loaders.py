"""
Dataset loaders for distillation experiments.
Training datasets (PMC-OA, PMC-QA) and evaluation datasets (CheXpert, NIH14,
MedMNIST, DeepLesion).
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler


# ── Batch types + collation ───────────────────────────────────────────────────

@dataclass
class BatchOA:
    images: torch.Tensor
    captions: list[str]


@dataclass
class BatchQA:
    images: torch.Tensor
    questions: list[str]
    answers: list[str]
    choices: list[list[str]]
    labels: torch.Tensor


def collate_oa(examples, preprocess):
    images, captions = [], []
    for img, cap in examples:
        images.append(preprocess(img))
        captions.append(cap)
    return BatchOA(images=torch.stack(images, dim=0), captions=captions)


def collate_qa(examples, preprocess):
    images, questions, answers, choices, labels = [], [], [], [], []
    for img, q, a, ch, lab in examples:
        images.append(preprocess(img))
        questions.append(q)
        answers.append(a)
        choices.append(ch)
        labels.append(lab)
    return BatchQA(
        images=torch.stack(images, dim=0),
        questions=questions,
        answers=answers,
        choices=choices,
        labels=torch.tensor(labels, dtype=torch.long),
    )


class HomogeneousBatchSampler(Sampler):
    """Yields all-OA or all-QA batches in shuffled order, covering both datasets per epoch."""

    def __init__(self, n_oa: int, n_qa: int, batch_size: int, seed: int = 0):
        self.n_oa = n_oa
        self.n_qa = n_qa
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.n_oa // self.batch_size + self.n_qa // self.batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        self._epoch += 1
        oa_perm = torch.randperm(self.n_oa, generator=g).tolist()
        oa_batches = [oa_perm[i:i + self.batch_size]
                      for i in range(0, self.n_oa - self.batch_size + 1, self.batch_size)]
        qa_perm = (torch.randperm(self.n_qa, generator=g) + self.n_oa).tolist()
        qa_batches = [qa_perm[i:i + self.batch_size]
                      for i in range(0, self.n_qa - self.batch_size + 1, self.batch_size)]
        all_batches = oa_batches + qa_batches
        order = torch.randperm(len(all_batches), generator=g).tolist()
        for i in order:
            yield all_batches[i]


def collate_combined(examples, preprocess):
    if len(examples[0]) == 2:
        return collate_oa(examples, preprocess)
    return collate_qa(examples, preprocess)


# ── Training datasets ─────────────────────────────────────────────────────────

class PMCOADataset(Dataset):
    """Local PMC-OA jsonl loader: {"image": "...jpg", "caption": "..."}"""

    def __init__(
        self,
        image_dir: str,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        seed: int = 1337,
    ):
        self.image_dir = Path(image_dir)
        self.jsonl_path = Path(jsonl_path)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing PMC-OA images dir: {self.image_dir}")
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Missing PMC-OA jsonl: {self.jsonl_path}")
        items: list[tuple[Path, str]] = []
        with self.jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img = obj.get("image")
                cap = obj.get("caption", "")
                if not img:
                    continue
                img_path = self.image_dir / img
                if not img_path.exists():
                    continue
                items.append((img_path, cap))
        if not items:
            raise RuntimeError(f"No PMC-OA items found from: {self.jsonl_path}")
        if max_samples is not None and max_samples < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, k=max_samples)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, caption = self.items[idx]
        image = Image.open(path).convert("RGB")
        return image, caption


class PMCQAChoicesDataset(Dataset):
    """Local PMC-QA csv loader with 4-way multiple choice."""

    LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        max_samples: Optional[int] = None,
        seed: int = 1337,
    ):
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing PMC-QA images dir: {self.image_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing PMC-QA csv: {self.csv_path}")
        rows: list[dict] = []
        with self.csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fig = (row.get("Figure_path") or "").strip()
                if not fig:
                    continue
                img_path = self.image_dir / fig
                if not img_path.exists():
                    continue
                rows.append({
                    "image_path": img_path,
                    "question": (row.get("Question") or "").strip(),
                    "answer": (row.get("Answer") or "").strip(),
                    "choice_a": (row.get("Choice A") or "").strip(),
                    "choice_b": (row.get("Choice B") or "").strip(),
                    "choice_c": (row.get("Choice C") or "").strip(),
                    "choice_d": (row.get("Choice D") or "").strip(),
                    "answer_label": (row.get("Answer_label") or "").strip(),
                })
        if not rows:
            raise RuntimeError(f"No PMC-QA rows loaded from: {self.csv_path}")
        if max_samples is not None and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, k=max_samples)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.LABEL_TO_INDEX.get(row["answer_label"], -1)
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        return image, row["question"], row["answer"], choices, label


# ── Evaluation datasets (linear probe) ───────────────────────────────────────

class MedMNISTDataset(Dataset):
    def __init__(self, dataset_name: str = "chestmnist", split: str = "train", image_size: int = 224):
        import medmnist
        from medmnist import INFO
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        n_channels = info.get("n_channels", 3)
        transforms_list = [T.Resize(image_size)]
        if n_channels == 1:
            transforms_list.append(T.Grayscale(num_output_channels=3))
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ds = DataClass(split=split, transform=T.Compose(transforms_list), download=True, size=image_size)
        self.multilabel = info.get("task", "") == "multi-label, binary-class"
        label_dict = info.get("label", {})
        self.label_names = [label_dict[str(k)] for k in range(len(label_dict))]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        label = np.asarray(label).squeeze()
        if self.multilabel:
            return img, torch.from_numpy(label.astype(np.float32))
        return img, int(label.item())


class NIHChestXray14Dataset(Dataset):
    LABELS_14 = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    ]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_size: int = 224,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        csv_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        transform=None,
    ):
        if data_dir is None and (csv_path is None or images_dir is None):
            raise ValueError("Provide either data_dir, or both csv_path and images_dir for NIH14.")
        base_dir = Path(data_dir) if data_dir is not None else None
        labels_file = Path(csv_path) if csv_path is not None else None
        img_dir = Path(images_dir) if images_dir is not None else None
        if base_dir is not None:
            candidates = [
                base_dir / "Data_Entry_2017.csv",
                base_dir / "Data_Entry_2017_v2020.csv",
                base_dir / "data" / "Data_Entry_2017.csv",
                base_dir / "data" / "Data_Entry_2017_v2020.csv",
            ]
            labels_file = next((p for p in candidates if p.exists()), labels_file)
            img_candidates = [
                base_dir / "images",
                base_dir / "data" / "images",
                base_dir / "data" / "images" / "images",
            ]
            img_dir = next((p for p in img_candidates if p.exists()), img_dir)
        if labels_file is None or not labels_file.exists():
            raise FileNotFoundError(f"Missing NIH14 labels CSV: {labels_file}")
        if img_dir is None or not img_dir.exists():
            raise FileNotFoundError(f"Missing NIH14 images dir: {img_dir}")
        label_to_idx = {l: i for i, l in enumerate(self.LABELS_14)}
        samples: list[tuple[Path, np.ndarray]] = []
        with labels_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if max_samples is not None and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, k=max_samples)
        for row in rows:
            img_name = row.get("Image Index")
            if not img_name:
                continue
            img_path = img_dir / img_name
            if not img_path.exists():
                continue
            findings = (row.get("Finding Labels") or "").strip()
            y = np.zeros((len(self.LABELS_14),), dtype=np.float32)
            if findings and findings != "No Finding":
                for lab in findings.split("|"):
                    lab = lab.strip()
                    if lab in label_to_idx:
                        y[label_to_idx[lab]] = 1.0
            samples.append((img_path, y))
        if not samples:
            raise RuntimeError(f"No NIH14 samples found under: {data_dir}")
        self.samples = samples
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.from_numpy(y)


class CheXpertDataset(Dataset):
    LABELS_14 = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices",
    ]

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        image_size: int = 224,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        uncertain_policy: str = "zeros",
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.csv_path = Path(csv_path)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing CheXpert images dir: {self.images_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing CheXpert CSV: {self.csv_path}")
        if uncertain_policy not in ("zeros", "ones"):
            raise ValueError("uncertain_policy must be one of: zeros, ones")
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if max_samples is not None and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, k=max_samples)
        items: list[tuple[Path, np.ndarray]] = []
        for row in rows:
            rel = (row.get("Path") or "").strip()
            if not rel:
                continue
            img_path = self.images_dir / rel
            if not img_path.exists():
                alt = self.images_dir / Path(rel).name
                img_path = alt if alt.exists() else img_path
            if not img_path.exists():
                continue
            y = np.zeros((len(self.LABELS_14),), dtype=np.float32)
            for i, lab in enumerate(self.LABELS_14):
                v = row.get(lab)
                if v is None:
                    continue
                v = str(v).strip()
                if v == "":
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if fv == 1.0:
                    y[i] = 1.0
                elif fv == 0.0:
                    y[i] = 0.0
                elif fv == -1.0:
                    y[i] = 0.0 if uncertain_policy == "zeros" else 1.0
            items.append((img_path, y))
        if not items:
            raise RuntimeError(f"No CheXpert samples found from: {self.csv_path}")
        self.items = items
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.from_numpy(y)


class DeepLesionDataset(Dataset):
    LESION_TYPES = [
        "bone", "abdomen", "mediastinum", "liver",
        "lung", "kidney", "soft_tissue", "pelvis",
    ]

    def __init__(
        self,
        data_dir: str,
        csv_path: Optional[str] = None,
        image_size: int = 224,
        max_samples: Optional[int] = None,
        seed: int = 1337,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        resolved_csv = Path(csv_path) if csv_path else self.data_dir / "DL_info.csv"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"DeepLesion data_dir not found: {self.data_dir}")
        if not resolved_csv.exists():
            raise FileNotFoundError(f"DeepLesion CSV not found: {resolved_csv}")
        img_root = self.data_dir / "Images_png_wn"
        if not img_root.exists():
            raise FileNotFoundError(f"DeepLesion Images_png_wn not found: {img_root}")
        items: list[tuple[Path, int]] = []
        with resolved_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = (row.get("File_name") or "").strip()
                raw_type = (row.get("Coarse_lesion_type") or "").strip()
                if not file_name or not raw_type:
                    continue
                try:
                    label = int(float(raw_type)) - 1
                except (ValueError, TypeError):
                    continue
                if label < 0 or label >= 8:
                    continue
                stem = file_name[:-4] if file_name.endswith(".png") else file_name
                parts = stem.split("_")
                if len(parts) < 4:
                    continue
                folder = "_".join(parts[:3])
                slice_file = parts[3] + ".png"
                img_path = img_root / folder / slice_file
                if not img_path.exists():
                    continue
                items.append((img_path, label))
        if not items:
            raise RuntimeError(f"No DeepLesion items found. Check data_dir ({self.data_dir}) and CSV.")
        if max_samples is not None and max_samples < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, k=max_samples)
        self.items = items
        self.transform = transform or T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)

