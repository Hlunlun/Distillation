from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


COL_W = 42


@dataclass
class _RunRow:
    time: str
    run: str
    method: str
    student: str
    teacher: str
    dataset: str
    macro_auroc: Optional[float]


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


def _rows_from_run(run_dir: Path) -> list[_RunRow]:
    if not (run_dir / "summary.json").exists() or not (run_dir / "metrics.json").exists():
        return []
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if not isinstance(metrics, list) or not metrics:
        return []
    last = metrics[-1]

    args: Optional[dict] = None
    args_json = run_dir / "args.json"
    if args_json.exists():
        args = json.loads(args_json.read_text())
    if args is None:
        for ckpt_name in ("last.pt", "best.pt"):
            ckpt_path = run_dir / ckpt_name
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
                        args = ckpt["args"]
                        break
                except Exception:
                    pass
    if not isinstance(args, dict):
        args = {}

    time_str = _infer_time(run_dir.name)
    method = _method_tag(args)
    student = str(args.get("timm_student", "-"))
    teacher = str(args.get("teacher_model", "-"))

    rows: list[_RunRow] = []
    for dataset, key in [("CheXpert", "chexpert_probe"), ("NIH14", "nih14_probe")]:
        probe = last.get(key)
        if isinstance(probe, dict):
            rows.append(_RunRow(
                time=time_str, run=run_dir.name, method=method,
                student=student, teacher=teacher, dataset=dataset,
                macro_auroc=float(probe["macro_auroc"]) if probe.get("macro_auroc") is not None else None,
            ))
    return rows


def _render_results_md(results_dir: str) -> str:
    rd = Path(results_dir)
    all_rows: list[_RunRow] = []
    if rd.exists():
        for d in sorted(rd.iterdir()):
            if d.is_dir():
                all_rows.extend(_rows_from_run(d))
    all_rows.sort(key=lambda r: (r.time, r.run, r.dataset))

    sota_cols = ["我用的模型", "BiomedCLIP-PubMedBERT_256-", "RadCLIP", "open-pmc-clip", "medclip-vit-base-patch16", "llava-med"]
    datasets = ["NIH14", "CheXpert"]
    best_by_ds: dict[str, Optional[_RunRow]] = {ds: None for ds in datasets}
    for r in all_rows:
        if r.dataset in best_by_ds and r.macro_auroc is not None:
            cur = best_by_ds[r.dataset]
            if cur is None or r.macro_auroc > (cur.macro_auroc or 0.0):
                best_by_ds[r.dataset] = r

    def fau(x: Optional[float]) -> str:
        return "-" if x is None else f"{x:.2f}"

    runs_table = "|time|run|method|student|teacher|dataset|macro_auroc|\n|---|---|---|---|---|---:|---:|"
    if all_rows:
        runs_table += "\n" + "\n".join(
            f"|{r.time}|{r.run}|{r.method}|{r.student}|{r.teacher}|{r.dataset}|{fau(r.macro_auroc)}|"
            for r in all_rows
        )

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
        runs_table if all_rows else "_No experiment results found yet._", "",
        "## SOTA Comparison (Linear Probe AUROC)", "",
        sota_table, "",
        "## Notes", "",
        "- Baseline columns stay `-` until you run probes for those teacher models.",
        "- `我用的模型` is the best distilled run found under `results/` for each dataset.",
    ]) + "\n"


def write_results_md(results_dir: str, results_md: str) -> None:
    Path(results_md).write_text(_render_results_md(results_dir))
    print(f"Updated {results_md}")


def print_table_header() -> None:
    print(f"\n{'Run':<{COL_W}} {'CheXpert AUROC':>14} {'NIH14 AUROC':>12} {'DeepLesion AUROC':>17}")
    print("-" * (COL_W + 45))


def print_run_row(label: str, probe_result: dict) -> None:
    def fmt(x: Optional[float]) -> str:
        return "-" if x is None else f"{x:.4f}"
    chex = fmt((probe_result.get("chexpert_probe") or {}).get("macro_auroc"))
    nih  = fmt((probe_result.get("nih14_probe") or {}).get("macro_auroc"))
    dl   = fmt((probe_result.get("deeplesion_probe") or {}).get("macro_auroc"))
    print(f"  {label:<{COL_W - 2}} {chex:>14} {nih:>12} {dl:>17}")
