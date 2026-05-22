from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found.")
        return

    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.json"
        if not summary_path.exists():
            continue

        summary = _read_json(summary_path)
        best_metric = summary.get("best_metric")

        last_epoch = None
        probes: dict[str, dict] = {}
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                if isinstance(metrics, list) and metrics:
                    m = metrics[-1]
                    last_epoch = m.get("epoch")
                    for key in ("chexpert_probe", "nih14_probe", "deeplesion_probe", "chestmnist_probe"):
                        p = m.get(key)
                        probes[key] = p if isinstance(p, dict) else {}
            except Exception:
                pass

        def get(key: str, field: str):
            return (probes.get(key) or {}).get(field)

        rows.append({
            "run":             run_dir.name,
            "best_metric":     best_metric,
            "last_epoch":      last_epoch,
            # CheXpert
            "chex_auroc":      get("chexpert_probe",  "macro_auroc"),
            "chex_f1":         get("chexpert_probe",  "macro_f1"),
            "chex_recall":     get("chexpert_probe",  "macro_recall"),
            # NIH14
            "nih_auroc":       get("nih14_probe",      "macro_auroc"),
            "nih_f1":          get("nih14_probe",      "macro_f1"),
            "nih_recall":      get("nih14_probe",      "macro_recall"),
            # DeepLesion
            "dl_auroc":        get("deeplesion_probe", "macro_auroc"),
            "dl_f1":           get("deeplesion_probe", "macro_f1"),
            "dl_recall":       get("deeplesion_probe", "macro_recall"),
            # ChestMNIST
            "cm_auroc":        get("chestmnist_probe", "macro_auroc"),
            "cm_f1":           get("chestmnist_probe", "macro_f1"),
            "cm_recall":       get("chestmnist_probe", "macro_recall"),
        })

    if not rows:
        print("No runs found under results/ (expected summary.json).")
        return

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)

    cols = [
        ("run",        34),
        ("best_metric", 6),
        ("chex_auroc",  6), ("chex_f1",  6), ("chex_recall",  6),
        ("nih_auroc",   6), ("nih_f1",   6), ("nih_recall",   6),
        ("dl_auroc",    6), ("dl_f1",    6), ("dl_recall",    6),
        ("cm_auroc",    6), ("cm_f1",    6), ("cm_recall",    6),
        ("last_epoch",  5),
    ]
    header = " ".join([c[0].ljust(c[1]) for c in cols])
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " ".join([fmt(r[c[0]]).ljust(c[1]) for c in cols])
        print(line)


if __name__ == "__main__":
    main()
