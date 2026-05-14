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
        best_ckpt = summary.get("best_ckpt")

        last_epoch = None
        last_chex = None
        last_nih = None
        last_dl = None
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                if isinstance(metrics, list) and metrics:
                    m = metrics[-1]
                    last_epoch = m.get("epoch")
                    cp = m.get("chexpert_probe") or None
                    np = m.get("nih14_probe") or None
                    dp = m.get("deeplesion_probe") or None
                    last_chex = cp.get("macro_auroc") if isinstance(cp, dict) else None
                    last_nih = np.get("macro_auroc") if isinstance(np, dict) else None
                    last_dl = dp.get("macro_auroc") if isinstance(dp, dict) else None
            except Exception:
                pass

        rows.append(
            {
                "run": run_dir.name,
                "best_metric": best_metric,
                "best_ckpt": best_ckpt,
                "last_epoch": last_epoch,
                "last_chexpert_macro_auroc": last_chex,
                "last_nih14_macro_auroc": last_nih,
                "last_deeplesion_macro_auroc": last_dl,
            }
        )

    if not rows:
        print("No runs found under results/ (expected summary.json).")
        return

    # Print a compact table (no extra deps).
    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)

    cols = [
        ("run", 36),
        ("best_metric", 10),
        ("last_chexpert_macro_auroc", 10),
        ("last_nih14_macro_auroc", 10),
        ("last_deeplesion_macro_auroc", 12),
        ("last_epoch", 9),
    ]
    header = " ".join([c[0].ljust(c[1]) for c in cols])
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " ".join([fmt(r[c[0]]).ljust(c[1]) for c in cols])
        print(line)


if __name__ == "__main__":
    main()

