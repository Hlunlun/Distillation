from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Optional


CHEXPERT_LABELS_14 = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency `datasets`. Install it before running this script.") from e


def _get_image(sample: dict[str, Any]) -> Any:
    for k in ("image", "jpg", "png"):
        if k in sample:
            return sample[k]
    raise KeyError("No image field found (expected one of: image/jpg/png).")


def _get_rel_path(sample: dict[str, Any], idx: int) -> str:
    for k in ("path", "image_path", "Path"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"{idx:08d}.jpg"


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _extract_label_map(sample: dict[str, Any]) -> dict[str, Optional[float]]:
    # Prefer explicit columns.
    if all(lab in sample for lab in CHEXPERT_LABELS_14):
        return {lab: _coerce_float(sample.get(lab)) for lab in CHEXPERT_LABELS_14}

    # Common packed forms.
    packed = sample.get("labels")
    if isinstance(packed, dict):
        return {lab: _coerce_float(packed.get(lab)) for lab in CHEXPERT_LABELS_14}
    if isinstance(packed, (list, tuple)) and len(packed) == len(CHEXPERT_LABELS_14):
        return {lab: _coerce_float(packed[i]) for i, lab in enumerate(CHEXPERT_LABELS_14)}

    return {lab: None for lab in CHEXPERT_LABELS_14}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="StanfordAIMI/CheXpert-v1.0-512")
    p.add_argument(
        "--split",
        type=str,
        default="all",
        help="Dataset split to export (e.g., train/validation/test) or 'all' to export every split.",
    )
    p.add_argument("--out_dir", type=str, default="/mnt/data/vlmdata/CheXpert")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--image_format", type=str, default="jpg")  # jpg|png
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    load_dataset = _require_datasets()

    try:
        if args.split == "all":
            ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
        else:
            ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    except Exception as e:
        msg = str(e)
        raise RuntimeError(
            "Failed to load CheXpert from Hugging Face.\n"
            "If this is a gated dataset, log in + accept the conditions first:\n"
            "  - `huggingface-cli login`\n"
            "  - visit the dataset page and accept access terms\n"
            f"Original error: {msg}"
        ) from e

    def export_split(split_name: str, split_ds):
        csv_path = out_dir / f"chexpert_{split_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Path", *CHEXPERT_LABELS_14])
            writer.writeheader()

            for idx, sample in enumerate(split_ds):
                if args.max_samples is not None and idx >= args.max_samples:
                    break

                rel = _get_rel_path(sample, idx)
                # Normalize any leading split dir to keep paths relative under images_dir.
                rel_path = Path(rel)
                if rel_path.is_absolute():
                    rel_path = Path(rel_path.name)

                # Ensure extension matches requested format.
                if args.image_format.lower() == "png":
                    rel_path = rel_path.with_suffix(".png")
                else:
                    rel_path = rel_path.with_suffix(".jpg")

                out_path = images_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if not out_path.exists():
                    img = _get_image(sample)
                    # HF Image feature returns a PIL Image
                    img.save(
                        out_path,
                        format=("PNG" if args.image_format.lower() == "png" else "JPEG"),
                        quality=95,
                    )

                row = {"Path": str(rel_path)}
                label_map = _extract_label_map(sample)
                for lab in CHEXPERT_LABELS_14:
                    v = label_map.get(lab)
                    row[lab] = "" if v is None else str(int(v) if float(v).is_integer() else v)
                writer.writerow(row)
        print(f"Wrote labels CSV to: {csv_path}")

    if args.split == "all":
        # DatasetDict: export every split we can see.
        for split_name, split_ds in ds.items():
            export_split(split_name, split_ds)
    else:
        export_split(args.split, ds)

    print(f"Wrote images to: {images_dir}")


if __name__ == "__main__":
    main()
