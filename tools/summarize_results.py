from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


PROBE_METRICS = ("macro_auroc", "macro_f1", "macro_recall", "macro_specificity", "acc")
PMC_FIELDS    = ("pmcoa_i2t_r1", "pmcoa_i2t_r5", "pmcoa_i2t_r10",
                 "pmcoa_t2i_r1", "pmcoa_t2i_r5", "pmcoa_t2i_r10")

MW = 7    # metric column width
RW = 38   # run name column width


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def _collect_best(metrics: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Returns:
      probe_best  — {ds_key: {metric: best_val_across_epochs}}
      pmc_best    — {pmc_field: best_val_across_epochs}
    """
    probe_best: dict[str, dict[str, float]] = {}
    pmc_best:   dict[str, float] = {}

    for epoch_data in metrics:
        for key, val in epoch_data.items():
            if not (key.endswith("_probe") and isinstance(val, dict) and val):
                continue
            if key not in probe_best:
                probe_best[key] = {}
            for metric in PROBE_METRICS:
                v = val.get(metric)
                if v is not None and v > probe_best[key].get(metric, -1.0):
                    probe_best[key][metric] = float(v)

        ret = ((epoch_data.get("vlm_eval") or {}).get("pmcoa_retrieval") or {})
        for field in PMC_FIELDS:
            v = ret.get(field)
            if v is not None and v > pmc_best.get(field, -1.0):
                pmc_best[field] = float(v)

    return probe_best, pmc_best


def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found.")
        return

    run_data:  list[dict] = []
    ds_order:  list[str]  = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            metrics = _read_json(metrics_path)
            if not isinstance(metrics, list) or not metrics:
                continue
        except Exception:
            continue

        probe_best, pmc_best = _collect_best(metrics)

        for k in probe_best:
            if k not in ds_order:
                ds_order.append(k)

        run_data.append({"run": run_dir.name, "probe_best": probe_best, "pmc_best": pmc_best})

    if not run_data:
        print("No runs with metrics.json found under results/.")
        return

    # ── column layout ──────────────────────────────────────────────────────────
    # per dataset block: one column per metric
    DS_BLOCK_W = MW * len(PROBE_METRICS) + len(PROBE_METRICS) - 1

    def ds_short(key: str) -> str:
        return key.replace("_probe", "").replace("_", "")

    PMC_BLOCK_W = MW * len(PMC_FIELDS) + len(PMC_FIELDS) - 1

    h1 = "run".ljust(RW)
    h2 = " " * RW
    for ds in ds_order:
        h1 += " | " + ds_short(ds).center(DS_BLOCK_W)
        h2 += " | " + " ".join(m[:6].ljust(MW) for m in PROBE_METRICS)
    h1 += " | " + "PMC-OA".center(PMC_BLOCK_W)
    h2 += " | " + " ".join(f.replace("pmcoa_", "")[:8].ljust(MW) for f in PMC_FIELDS)

    sep = "-" * max(len(h1), len(h2))
    print(h1)
    print(h2)
    print(sep)

    for rd in run_data:
        line = rd["run"].ljust(RW)
        for ds in ds_order:
            bests = rd["probe_best"].get(ds) or {}
            line += " | " + " ".join(_fmt(bests.get(m)).ljust(MW) for m in PROBE_METRICS)
        line += " | " + " ".join(_fmt(rd["pmc_best"].get(f)).ljust(MW) for f in PMC_FIELDS)
        print(line)


if __name__ == "__main__":
    main()
