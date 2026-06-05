"""Scan PMC-OA image files and report any that PIL cannot open.

Reads the same (image_dir, jsonl) pair that PMCOADataset uses and tries
Image.open().convert("RGB") on every path that exists on disk.  Writes
a text file listing all bad paths so they can be patched out of the jsonl.

Usage:
    python tools/check_pmc_oa_images.py
    python tools/check_pmc_oa_images.py --jsonl /path/to/train.jsonl \
        --image_dir /path/to/images/ --out bad_images.txt --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing.pool import Pool
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import paths_cfg as PATHS


def _check_one(args: tuple[str, str]) -> tuple[str, str | None]:
    """Return (path_str, error_msg) where error_msg is None on success."""
    path_str, _ = args
    try:
        with Image.open(path_str) as img:
            img.convert("RGB")
        return path_str, None
    except UnidentifiedImageError:
        return path_str, "UnidentifiedImageError"
    except OSError as e:
        return path_str, f"OSError: {e}"
    except Exception as e:
        return path_str, f"{type(e).__name__}: {e}"


def load_items(image_dir: Path, jsonl_path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = obj.get("image")
            cap = obj.get("caption", "")
            if not img:
                continue
            p = image_dir / img
            if p.exists():
                items.append((str(p), cap))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PMC-OA images for PIL errors.")
    parser.add_argument("--image_dir", default=PATHS.pmc_oa_image_dir,
                        help="PMC-OA image directory")
    parser.add_argument("--jsonl", default=PATHS.pmc_oa_train_jsonl,
                        help="PMC-OA train jsonl")
    parser.add_argument("--out", default="bad_pmc_oa_images.txt",
                        help="Output file listing bad image paths (one per line)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Check only the first N items (for a quick spot-check)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    jsonl_path = Path(args.jsonl)

    if not image_dir.exists():
        sys.exit(f"ERROR: image_dir not found: {image_dir}")
    if not jsonl_path.exists():
        sys.exit(f"ERROR: jsonl not found: {jsonl_path}")

    print(f"Loading jsonl: {jsonl_path}")
    items = load_items(image_dir, jsonl_path)
    if not items:
        sys.exit("ERROR: no items found in jsonl (all missing from disk?)")

    if args.limit:
        items = items[: args.limit]

    total = len(items)
    print(f"Checking {total:,} images with {args.workers} workers...")

    bad: list[tuple[str, str]] = []
    done = 0
    report_every = max(1, total // 20)

    with Pool(processes=args.workers) as pool:
        for path_str, err in pool.imap_unordered(_check_one, items, chunksize=256):
            done += 1
            if err is not None:
                bad.append((path_str, err))
            if done % report_every == 0 or done == total:
                pct = 100 * done / total
                print(f"  {done:>8,} / {total:,}  ({pct:.0f}%)  bad so far: {len(bad)}")

    print(f"\nDone. {len(bad):,} / {total:,} images failed.")

    if bad:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            for path_str, err in bad:
                f.write(f"{path_str}\t{err}\n")
        print(f"Bad image list written to: {out_path}")
        print("\nSample failures:")
        for path_str, err in bad[:10]:
            print(f"  {err:30s}  {path_str}")
    else:
        print("All images opened successfully.")


if __name__ == "__main__":
    main()
