from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


COL_W = 42

PROBE_DATASETS: list[tuple[str, str]] = [
    ("CheXpert",       "chexpert_probe"),
    ("NIH14",          "nih14_probe"),
    ("DeepLesion",     "deeplesion_probe"),
    ("ChestMNIST",     "chestmnist_probe"),
    ("PathMNIST",      "pathmnist_probe"),
    ("DermaMNIST",     "dermamnist_probe"),
    ("OCTMNIST",       "octmnist_probe"),
    ("PneumoniaMNIST", "pneumoniamnist_probe"),
    ("OrganMNIST",     "organamnist_probe"),
    ("PCam",           "pcam_probe"),
    ("LC25000-Lung",   "lc25000_lung_probe"),
    ("LC25000-Colon",  "lc25000_colon_probe"),
]

PMC_FIELDS = ("pmcoa_i2t_r1", "pmcoa_i2t_r5", "pmcoa_i2t_r10",
              "pmcoa_t2i_r1", "pmcoa_t2i_r5", "pmcoa_t2i_r10")


@dataclass
class _RunRow:
    time: str
    run: str
    method: str
    student: str
    teacher: str
    dataset: str
    macro_auroc: Optional[float]
    acc: Optional[float]
    macro_f1: Optional[float]
    macro_recall: Optional[float]
    macro_specificity: Optional[float]


@dataclass
class _PMCRow:
    time: str
    run: str
    i2t_r1: Optional[float]
    i2t_r5: Optional[float]
    i2t_r10: Optional[float]
    t2i_r1: Optional[float]
    t2i_r5: Optional[float]
    t2i_r10: Optional[float]


def _infer_time(run_name: str) -> str:
    parts = run_name.split("_")
    if len(parts) >= 2:
        try:
            dt = datetime.strptime(parts[-2] + parts[-1], "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return "-"


def _method_tag(a: dict) -> str:
    def on(k: str) -> bool:
        try:
            return float(a.get(k, 0.0)) > 0.0
        except Exception:
            return False
    tags = [t for t, k in [("KD-IMG", "w_img"), ("KD-OA", "w_oa"), ("KD-MC", "w_mc"), ("CE", "w_ce"), ("REL", "w_rel")] if on(k)]
    knobs = []
    if a.get("qa_ratio") is not None:
        knobs.append(f"qa={a['qa_ratio']}")
    if a.get("warmup_ratio") is not None:
        knobs.append(f"warmup={a['warmup_ratio']}")
    return "+".join(tags) + (f" ({', '.join(knobs)})" if knobs else "")


def _load_args(run_dir: Path) -> dict:
    args_json = run_dir / "args.json"
    if args_json.exists():
        return json.loads(args_json.read_text())
    for ckpt_name in ("last.pt", "best.pt"):
        ckpt_path = run_dir / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
                    return ckpt["args"]
            except Exception:
                pass
    return {}


def _rows_from_run(run_dir: Path) -> tuple[list[_RunRow], Optional[_PMCRow]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return [], None

    try:
        metrics = json.loads(metrics_path.read_text())
        if not isinstance(metrics, list) or not metrics:
            return [], None
    except Exception:
        return [], None

    # per-dataset, per-metric best across all epochs
    probe_best: dict[str, dict[str, float]] = {}
    pmc_best:   dict[str, float] = {}

    for epoch_data in metrics:
        for probe_key in (k for k in epoch_data if k.endswith("_probe")):
            val = epoch_data[probe_key]
            if not isinstance(val, dict):
                continue
            m = val.get("test") if isinstance(val.get("test"), dict) else val
            if probe_key not in probe_best:
                probe_best[probe_key] = {}
            for field in ("macro_auroc", "macro_f1", "macro_recall", "macro_specificity", "acc"):
                v = m.get(field)
                if v is not None and v > probe_best[probe_key].get(field, -1.0):
                    probe_best[probe_key][field] = float(v)

        ret = ((epoch_data.get("vlm_eval") or {}).get("pmcoa_retrieval") or {})
        for field in PMC_FIELDS:
            v = ret.get(field)
            if v is not None and v > pmc_best.get(field, -1.0):
                pmc_best[field] = float(v)

    args = _load_args(run_dir)
    time_str = _infer_time(run_dir.name)
    method  = _method_tag(args)
    student = str(args.get("timm_student", "-"))
    teacher = str(args.get("teacher_model", "-"))

    rows: list[_RunRow] = []
    for dataset, probe_key in PROBE_DATASETS:
        bests = probe_best.get(probe_key)
        if not bests or bests.get("macro_auroc") is None:
            continue
        rows.append(_RunRow(
            time=time_str, run=run_dir.name, method=method,
            student=student, teacher=teacher, dataset=dataset,
            macro_auroc=bests.get("macro_auroc"),
            acc=bests.get("acc"),
            macro_f1=bests.get("macro_f1"),
            macro_recall=bests.get("macro_recall"),
            macro_specificity=bests.get("macro_specificity"),
        ))

    pmc_row: Optional[_PMCRow] = None
    if pmc_best:
        pmc_row = _PMCRow(
            time=time_str, run=run_dir.name,
            i2t_r1=pmc_best.get("pmcoa_i2t_r1"),
            i2t_r5=pmc_best.get("pmcoa_i2t_r5"),
            i2t_r10=pmc_best.get("pmcoa_i2t_r10"),
            t2i_r1=pmc_best.get("pmcoa_t2i_r1"),
            t2i_r5=pmc_best.get("pmcoa_t2i_r5"),
            t2i_r10=pmc_best.get("pmcoa_t2i_r10"),
        )

    return rows, pmc_row


def _render_results_md(results_dir: str) -> str:
    rd = Path(results_dir)
    all_rows:    list[_RunRow] = []
    all_pmc:     list[_PMCRow] = []

    if rd.exists():
        for d in sorted(rd.iterdir()):
            if d.is_dir():
                rows, pmc_row = _rows_from_run(d)
                all_rows.extend(rows)
                if pmc_row is not None:
                    all_pmc.append(pmc_row)

    all_rows.sort(key=lambda r: (r.time, r.run, r.dataset))
    all_pmc.sort(key=lambda r: (r.time, r.run))

    def fv(x: Optional[float]) -> str:
        return "-" if x is None else f"{x:.2f}"

    # ── probe experiments table ───────────────────────────────────────────────
    probe_header = "|time|run|method|student|teacher|dataset|auroc|acc|f1|recall|specificity|\n|---|---|---|---|---|---|---:|---:|---:|---:|---:|"
    if all_rows:
        probe_body = "\n".join(
            f"|{r.time}|{r.run}|{r.method}|{r.student}|{r.teacher}|{r.dataset}"
            f"|{fv(r.macro_auroc)}|{fv(r.acc)}|{fv(r.macro_f1)}|{fv(r.macro_recall)}|{fv(r.macro_specificity)}|"
            for r in all_rows
        )
        runs_table = probe_header + "\n" + probe_body
    else:
        runs_table = "_No experiment results found yet._"

    # ── PMC-OA retrieval table ────────────────────────────────────────────────
    pmc_header = "|time|run|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|\n|---|---|---:|---:|---:|---:|---:|---:|"
    if all_pmc:
        pmc_body = "\n".join(
            f"|{r.time}|{r.run}|{fv(r.i2t_r1)}|{fv(r.i2t_r5)}|{fv(r.i2t_r10)}"
            f"|{fv(r.t2i_r1)}|{fv(r.t2i_r5)}|{fv(r.t2i_r10)}|"
            for r in all_pmc
        )
        pmc_table = pmc_header + "\n" + pmc_body
    else:
        pmc_table = "_No PMC-OA retrieval results yet._"

    # ── SOTA comparison (best auroc per dataset) ──────────────────────────────
    sota_cols = ["我用的模型", "BiomedCLIP-PubMedBERT_256-", "RadCLIP", "open-pmc-clip", "medclip-vit-base-patch16", "llava-med"]
    datasets  = [d for d, _ in PROBE_DATASETS]
    best_by_ds: dict[str, Optional[_RunRow]] = {ds: None for ds in datasets}
    for r in all_rows:
        if r.dataset in best_by_ds and r.macro_auroc is not None:
            cur = best_by_ds[r.dataset]
            if cur is None or r.macro_auroc > (cur.macro_auroc or 0.0):
                best_by_ds[r.dataset] = r

    sota_header = "|" + "|".join(["資料集", *sota_cols]) + "|\n|" + "|".join(["---"] * (len(sota_cols) + 1)) + "|"
    sota_rows = []
    for ds in datasets:
        best = best_by_ds[ds]
        our = "-" if best is None or best.macro_auroc is None else f"{best.macro_auroc:.2f} ({best.run})"
        sota_rows.append("|" + "|".join([ds, our] + ["-"] * (len(sota_cols) - 1)) + "|")
    sota_table = sota_header + "\n" + "\n".join(sota_rows)

    return "\n".join([
        "# Results", "",
        "## Experiments", "",
        runs_table, "",
        "## PMC-OA Retrieval", "",
        pmc_table, "",
        "## SOTA Comparison (Linear Probe AUROC)", "",
        sota_table, "",
        "## Notes", "",
        "- Baseline columns stay `-` until you run probes for those teacher models.",
        "- `我用的模型` is the best distilled run found under `results/` for each dataset.",
        "- All metrics are per-dataset per-metric best across all epochs (epochs may differ).",
    ]) + "\n"


def write_results_md(results_dir: str, results_md: str) -> None:
    Path(results_md).write_text(_render_results_md(results_dir))
    print(f"Updated {results_md}")


def print_table_header() -> None:
    datasets = [d for d, _ in PROBE_DATASETS]
    header = f"{'Run':<{COL_W}}" + "".join(f" {d[:10]:>11}" for d in datasets)
    print(f"\n{header}")
    print("-" * len(header))


def print_run_row(label: str, probe_result: dict) -> None:
    def auroc(key: str) -> str:
        p = probe_result.get(key) or {}
        v = p.get("test", p).get("macro_auroc") if isinstance(p.get("test"), dict) else p.get("macro_auroc")
        return "-" if v is None else f"{v:.4f}"

    values = "".join(f" {auroc(probe_key):>11}" for _, probe_key in PROBE_DATASETS)
    print(f"  {label:<{COL_W - 2}}{values}")
