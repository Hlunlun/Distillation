"""ArXiv search and paper fetching tool."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

PAPER_CACHE_DIR = Path(__file__).parent.parent / "memory" / "papers"
PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Search Queries ────────────────────────────────────────────────────────────
RESEARCH_QUERIES = [
    # General distillation
    "knowledge distillation visual encoder CLIP",
    "visual encoder distillation multimodal",
    "text encoder distillation language model compression",
    "CLIP distillation lightweight vision language model",
    "vision language model compression knowledge distillation",
    "multimodal representation distillation",
    "contrastive distillation vision transformer",
    "EVA CLIP SigLIP distillation",
    "token distillation vision transformer image text",
    "feature distillation cross-modal alignment",
    # Medical VLM distillation
    "medical vision language model distillation",
    "BiomedCLIP medical image text distillation",
    "pathology vision encoder knowledge distillation",
    "radiology report generation vision encoder efficient",
    "medical image foundation model compression",
]

# ── General-domain Teachers ───────────────────────────────────────────────────
GENERAL_TEACHERS: dict[str, dict] = {
    "EVA-CLIP-18B": {
        "hf_id": None,
        "type": "teacher",
        "domain": "general",
        "params_M": 18000,
        "imagenet_zeroshot_top1": 80.4,
        "coco_i2t_r1": 86.4,
        "coco_t2i_r1": 71.6,
        "flickr30k_i2t_r1": 98.6,
        "flickr30k_t2i_r1": 90.5,
        "paper": "2402.04252",
    },
    "SigLIP-SO400M/14-384": {
        "hf_id": "google/siglip-so400m-patch14-384",
        "type": "teacher",
        "domain": "general",
        "params_M": 400,
        "imagenet_zeroshot_top1": 83.2,
        "coco_i2t_r1": None,
        "coco_t2i_r1": None,
        "flickr30k_i2t_r1": None,
        "flickr30k_t2i_r1": None,
        "paper": "2303.15343",
    },
    "OpenCLIP-ViT-G/14": {
        "hf_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "type": "teacher",
        "domain": "general",
        "params_M": 1843,
        "imagenet_zeroshot_top1": 80.1,
        "coco_i2t_r1": 82.7,
        "coco_t2i_r1": 65.5,
        "flickr30k_i2t_r1": 97.6,
        "flickr30k_t2i_r1": 87.8,
        "paper": "2212.07143",
    },
    "CLIP-ViT-L/14": {
        "hf_id": "openai/clip-vit-large-patch14",
        "type": "teacher",
        "domain": "general",
        "params_M": 307,
        "imagenet_zeroshot_top1": 75.5,
        "coco_i2t_r1": 73.7,
        "coco_t2i_r1": 57.2,
        "flickr30k_i2t_r1": 95.3,
        "flickr30k_t2i_r1": 79.8,
        "paper": "2103.00020",
    },
}

# ── Medical VLM Teachers / SOTA Comparisons ───────────────────────────────────
# Used both as teachers for medical distillation and as SOTA baselines to beat.
MEDICAL_TEACHERS: dict[str, dict] = {
    "BiomedCLIP-ViT-B/16": {
        # Trained on PMC-15M (15M biomedical image-text pairs from PubMed Central)
        "hf_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "type": "medical_teacher",
        "domain": "biomedical (general)",
        "params_M": 150,
        "visual_encoder": "ViT-B/16",
        "text_encoder": "PubMedBERT-256",
        "training_data": "PMC-15M",
        "pathvqa_acc": 76.3,       # open-ended PathVQA accuracy (%)
        "vqa_rad_acc": 78.4,       # VQA-RAD closed-ended accuracy (%)
        "roco_i2t_r1": None,
        "paper": "2303.00915",
        "notes": "Strong general biomedical baseline; weak on histopathology",
    },
    "PLIP-ViT-B/32": {
        # Pathology Language-Image Pretraining; trained on Twitter pathology images
        "hf_id": "vinid/plip",
        "type": "medical_teacher",
        "domain": "pathology",
        "params_M": 151,
        "visual_encoder": "ViT-B/32",
        "text_encoder": "GPT-2 style",
        "training_data": "PathCap-208K (Twitter)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2209.07521",
        "notes": "Pathology-specific; limited training data volume",
    },
    "QuiltNet-B-32": {
        # 768K histopathology image-text pairs from YouTube
        "hf_id": "wisdomik/QuiltNet-B-32",
        "type": "medical_teacher",
        "domain": "pathology",
        "params_M": 151,
        "visual_encoder": "ViT-B/32",
        "text_encoder": "GPT-2 style",
        "training_data": "Quilt-1M (768K filtered)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2306.11207",
        "notes": "Best open pathology model; only patch-level visual understanding",
    },
    "CONCH-ViT-B/16": {
        # CONtrastive learning from Captions for Histopathology
        "hf_id": "MahmoodLab/conch",
        "type": "medical_teacher",
        "domain": "pathology",
        "params_M": 150,
        "visual_encoder": "ViT-B/16",
        "text_encoder": "CoCa text tower",
        "training_data": "EDU-PMC (1.17M pathology pairs)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2406.19890",
        "notes": "SOTA pathology VLP; closed weights (license required)",
    },
    "UNI-ViT-L/16": {
        # Universal pathology encoder; visual-only (no text tower)
        "hf_id": "MahmoodLab/uni",
        "type": "medical_teacher",
        "domain": "pathology (vision-only)",
        "params_M": 307,
        "visual_encoder": "ViT-L/16 (DINOv2)",
        "text_encoder": None,
        "training_data": "Mass-100K (100K WSI patches)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2308.15474",
        "notes": "Best patch-level pathology features; no language alignment",
    },
    "MedCLIP-ViT-B/16": {
        # Chest X-ray focused CLIP with semantic decoupling
        "hf_id": "microsoft/medclip-vit-base-patch16",
        "type": "medical_teacher",
        "domain": "radiology (CXR)",
        "params_M": 150,
        "visual_encoder": "ViT-B/16",
        "text_encoder": "ClinicalBERT",
        "training_data": "CheXpert-320K + MIMIC-CXR",
        "pathvqa_acc": None,
        "vqa_rad_acc": 79.1,
        "roco_i2t_r1": None,
        "paper": "2210.10163",
        "notes": "Strong on CXR; poor cross-modality transfer",
    },
    "PMC-CLIP-ViT-B/32": {
        # Trained on PMC-OA (1.65M biomedical figure-caption pairs)
        "hf_id": "axiong/PMC_CLIP_ViT-B-32",
        "type": "medical_teacher",
        "domain": "biomedical (general)",
        "params_M": 151,
        "visual_encoder": "ViT-B/32",
        "text_encoder": "RoBERTa",
        "training_data": "PMC-OA-1.65M",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": 71.2,
        "paper": "2303.07240",
        "notes": "Good general biomedical; lower resolution (224px)",
    },
    "BioViL-T": {
        "hf_id": "microsoft/BiomedVLP-BioViL-T",
        "type": "medical_teacher",
        "domain": "radiology (CXR)",
        "params_M": None,
        "visual_encoder": "Hybrid (ResNet50 + ViT aggregator)",
        "text_encoder": "CXR-BERT",
        "training_data": "MIMIC-CXR (+ PubMed/MIMIC-III for text pretraining)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2301.04558",
        "notes": "Strong CXR joint space; good teacher for chest X-ray probes",
    },
    "RadCLIP": {
        "hf_id": "zluvolyote/RadCLIP",
        "type": "medical_teacher",
        "domain": "radiology",
        "params_M": None,
        "visual_encoder": "CLIP ViT-L/14 (finetuned)",
        "text_encoder": "CLIP text tower (finetuned)",
        "training_data": "radiology image-text pairs (see paper)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2403.09948",
        "notes": "Radiology-specific CLIP-style teacher; checkpoint availability varies",
    },
    "OpenPMC-CLIP": {
        "hf_id": "vector-institute/open-pmc-clip",
        "type": "medical_teacher",
        "domain": "biomedical (general)",
        "params_M": None,
        "visual_encoder": "ViT-B/16",
        "text_encoder": "PubMedBERT",
        "training_data": "Open-PMC (~2M image-text pairs)",
        "pathvqa_acc": None,
        "vqa_rad_acc": None,
        "roco_i2t_r1": None,
        "paper": "2503.14377",
        "notes": "PMC-focused CLIP-style teacher; good for figure-caption alignment",
    },
}

# ── Student Model Candidates ──────────────────────────────────────────────────
# These are the architectures we train from scratch or fine-tune as students.
STUDENT_CANDIDATES: dict[str, dict] = {
    "ViT-Ti/16": {
        "hf_id": "WinKawaks/vit-tiny-patch16-224",
        "params_M": 5.7,
        "imagenet_supervised_top1": 72.2,
        "notes": "Lightest ViT; suitable for edge deployment",
    },
    "ViT-S/16": {
        "hf_id": "WinKawaks/vit-small-patch16-224",
        "params_M": 22.1,
        "imagenet_supervised_top1": 78.1,
        "notes": "Good speed/accuracy tradeoff; popular student choice",
    },
    "ViT-S/32": {
        "hf_id": None,
        "params_M": 22.0,
        "imagenet_supervised_top1": 76.0,
        "notes": "Lower resolution patches → faster inference",
    },
    "ViT-B/32": {
        "hf_id": "openai/clip-vit-base-patch32",
        "params_M": 87.0,
        "imagenet_supervised_top1": 81.2,
        "notes": "Standard CLIP student baseline",
    },
    "ViT-B/16": {
        "hf_id": "openai/clip-vit-base-patch16",
        "params_M": 86.4,
        "imagenet_supervised_top1": 83.1,
        "notes": "Higher-res patches; good for medical (fine-grained)",
    },
    "EfficientViT-M4": {
        "hf_id": "mit-han-lab/efficientvit-m4",
        "params_M": 42.0,
        "imagenet_supervised_top1": 79.4,
        "notes": "Multiscale attention; very fast on mobile hardware",
    },
    "MobileViT-S": {
        "hf_id": "apple/mobilevit-small",
        "params_M": 5.6,
        "imagenet_supervised_top1": 78.4,
        "notes": "Hybrid CNN-ViT; good for resource-constrained medical devices",
    },
    "DeiT-S/16": {
        "hf_id": "facebook/deit-small-patch16-224",
        "params_M": 22.1,
        "imagenet_supervised_top1": 79.9,
        "notes": "Well-studied distillation target (DeiT was designed for distillation)",
    },
}

# ── Training Datasets ─────────────────────────────────────────────────────────
TRAINING_DATASETS: dict[str, dict] = {
    # General
    "LAION-400M": {
        "domain": "general",
        "size": "400M image-text pairs",
        "hf_id": "laion/laion400m",
        "notes": "Standard large-scale pretraining corpus",
    },
    "CC12M": {
        "domain": "general",
        "size": "12M image-text pairs",
        "hf_id": "clip-benchmark/wds_cc12m",
        "notes": "Conceptual Captions; good for smaller-scale runs",
    },
    # Medical — General Biomedical
    "PMC-OA": {
        "domain": "biomedical (general)",
        "size": "1.65M figure-caption pairs",
        "hf_id": "axiong/PMC-OA",
        "notes": "PubMed Central Open Access; diverse modalities (CXR, MRI, path, etc.)",
    },
    "PMC-15M": {
        "domain": "biomedical (general)",
        "size": "15M image-text pairs",
        "hf_id": None,
        "notes": "Subset used by BiomedCLIP; not publicly released in full",
    },
    "MedICaT": {
        "domain": "biomedical (general)",
        "size": "217K figures with captions",
        "hf_id": "allenai/medicat",
        "notes": "Includes sub-figure detection; useful for multi-panel medical images",
    },
    "ROCO": {
        "domain": "radiology",
        "size": "81K radiology image-caption pairs",
        "hf_id": "eltorio/ROCO-radiology",
        "notes": "Standard radiology retrieval benchmark; also used for training",
    },
    # Medical — Radiology
    "MIMIC-CXR": {
        "domain": "radiology (CXR)",
        "size": "227K CXR studies with reports",
        "hf_id": "physionet/mimic-cxr",
        "notes": "Requires PhysioNet credentialing; gold standard for CXR report generation",
    },
    "CheXpert-320K": {
        "domain": "radiology (CXR)",
        "size": "224K frontal/lateral CXR",
        "hf_id": "stanfordmlgroup/chexpert",
        "notes": "14-class labels; widely used for CXR classification + retrieval",
    },
    # Medical — Pathology
    "Quilt-1M": {
        "domain": "pathology",
        "size": "1M histopathology image-text pairs",
        "hf_id": "wisdomik/Quilt-1M",
        "notes": "YouTube narrated pathology videos; used by QuiltNet",
    },
    "OpenPath": {
        "domain": "pathology",
        "size": "208K pathology image-text pairs",
        "hf_id": "vinid/plip",
        "notes": "Social media + PubMed pathology captions; used by PLIP",
    },
    "TCGA-patches": {
        "domain": "pathology",
        "size": "~10M WSI tile patches (unlabeled)",
        "hf_id": "tcga",
        "notes": "Used for self-supervised pretraining of UNI, CONCH visual encoders",
    },
    # Medical — Evaluation / Downstream
    "PathVQA": {
        "domain": "pathology (eval)",
        "size": "32.8K QA pairs",
        "hf_id": "flaviagiammarino/path-vqa",
        "notes": "Primary pathology VQA benchmark; open and closed questions",
    },
    "VQA-RAD": {
        "domain": "radiology (eval)",
        "size": "3.5K clinical QA pairs",
        "hf_id": "flaviagiammarino/vqa-rad",
        "notes": "Radiology VQA; closed-ended subset is standard eval",
    },
    "SLAKE": {
        "domain": "radiology (eval)",
        "size": "14K bilingual medical VQA",
        "hf_id": "BoKelvin/SLAKE",
        "notes": "English + Chinese; CXR + CT + MRI",
    },
}

# ── All models merged for summary printing ────────────────────────────────────
SOTA_MODELS: dict[str, dict] = {**GENERAL_TEACHERS, **MEDICAL_TEACHERS}


def get_sota_summary() -> str:
    """Formatted multi-section model registry for prompt injection."""
    lines = ["## Model Registry\n"]

    # Section 1: General Teachers
    lines += [
        "### General-Domain Teachers",
        f"{'Model':<30} {'Params(M)':<12} {'IN0shot':<10} {'COCO i2t R@1':<15} {'Flickr i2t R@1'}",
        "-" * 85,
    ]
    for name, m in GENERAL_TEACHERS.items():
        lines.append(
            f"{name:<30} {m['params_M']:<12} "
            f"{str(m['imagenet_zeroshot_top1'] or '-'):<10} "
            f"{str(m['coco_i2t_r1'] or '-'):<15} "
            f"{str(m['flickr30k_i2t_r1'] or '-')}"
        )

    # Section 2: Medical VLM Teachers
    lines += [
        "",
        "### Medical VLM Teachers / SOTA",
        f"{'Model':<28} {'Domain':<25} {'Params(M)':<12} {'PathVQA':<10} {'VQA-RAD':<10} {'Notes'}",
        "-" * 105,
    ]
    for name, m in MEDICAL_TEACHERS.items():
        lines.append(
            f"{name:<28} {m['domain']:<25} {m['params_M']:<12} "
            f"{str(m.get('pathvqa_acc') or '-'):<10} "
            f"{str(m.get('vqa_rad_acc') or '-'):<10} "
            f"{m.get('notes','')[:60]}"
        )

    # Section 3: Student Candidates
    lines += [
        "",
        "### Student Model Candidates (to be distilled into)",
        f"{'Model':<20} {'Params(M)':<12} {'IN_sup Top1':<14} {'Notes'}",
        "-" * 75,
    ]
    for name, m in STUDENT_CANDIDATES.items():
        lines.append(
            f"{name:<20} {m['params_M']:<12} "
            f"{str(m['imagenet_supervised_top1']):<14} "
            f"{m['notes'][:55]}"
        )

    # Section 4: Medical Datasets
    lines += [
        "",
        "### Medical Training & Evaluation Datasets",
        f"{'Dataset':<18} {'Domain':<30} {'Size':<35} {'Notes'}",
        "-" * 100,
    ]
    for name, d in TRAINING_DATASETS.items():
        lines.append(
            f"{name:<18} {d['domain']:<30} {d['size']:<35} {d['notes'][:45]}"
        )

    return "\n".join(lines)


SOTA_BENCHMARKS = {
    # General
    "imagenet_zeroshot_top1": {
        "description": "ImageNet zero-shot Top-1 accuracy (%)",
        "higher_is_better": True,
    },
    "coco_i2t_r1": {
        "description": "COCO image-to-text retrieval R@1 (%)",
        "higher_is_better": True,
    },
    "coco_t2i_r1": {
        "description": "COCO text-to-image retrieval R@1 (%)",
        "higher_is_better": True,
    },
    "flickr30k_i2t_r1": {
        "description": "Flickr30K image-to-text R@1 (%)",
        "higher_is_better": True,
    },
    "flickr30k_t2i_r1": {
        "description": "Flickr30K text-to-image R@1 (%)",
        "higher_is_better": True,
    },
    "params_millions": {
        "description": "Total model parameters (millions)",
        "higher_is_better": False,
    },
    # Medical
    "pathvqa_open_acc": {
        "description": "PathVQA open-ended accuracy (%)",
        "higher_is_better": True,
    },
    "pathvqa_closed_acc": {
        "description": "PathVQA closed-ended accuracy (%)",
        "higher_is_better": True,
    },
    "vqa_rad_closed_acc": {
        "description": "VQA-RAD closed-ended accuracy (%)",
        "higher_is_better": True,
    },
    "slake_closed_acc": {
        "description": "SLAKE closed-ended accuracy (%)",
        "higher_is_better": True,
    },
    "roco_i2t_r1": {
        "description": "ROCO image-to-text retrieval R@1 (%)",
        "higher_is_better": True,
    },
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    url: str
    pdf_url: str
    categories: list[str]
    summary: Optional[str] = None
    methods: Optional[list[str]] = None
    benchmarks: Optional[dict] = None
    gaps: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(**d)

    def cache_path(self) -> Path:
        return PAPER_CACHE_DIR / f"{self.arxiv_id.replace('/', '_')}.json"

    def save(self):
        self.cache_path().write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, arxiv_id: str) -> Optional["Paper"]:
        path = PAPER_CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance") -> list[Paper]:
    """Search ArXiv and return Paper objects."""
    sort_criterion = arxiv.SortCriterion.Relevance
    if sort_by == "date":
        sort_criterion = arxiv.SortCriterion.SubmittedDate

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    papers = []
    for result in client.results(search):
        arxiv_id = result.entry_id.split("/abs/")[-1]
        cached = Paper.load(arxiv_id)
        if cached:
            papers.append(cached)
            continue

        paper = Paper(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=[a.name for a in result.authors[:6]],
            abstract=result.summary.replace("\n", " "),
            published=result.published.strftime("%Y-%m-%d"),
            url=result.entry_id,
            pdf_url=result.pdf_url,
            categories=[c for c in result.categories],
        )
        paper.save()
        papers.append(paper)
        time.sleep(0.3)

    return papers


def fetch_pdf_text(paper: Paper, max_chars: int = 12000) -> Optional[str]:
    """Download PDF and extract text (first max_chars characters)."""
    try:
        import pdfplumber
        import io

        resp = requests.get(paper.pdf_url, timeout=30)
        resp.raise_for_status()
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                if len(text) >= max_chars:
                    break
        return text[:max_chars]
    except Exception:
        return None


def load_all_cached_papers() -> list[Paper]:
    """Load all previously cached papers."""
    papers = []
    for path in sorted(PAPER_CACHE_DIR.glob("*.json")):
        try:
            papers.append(Paper.from_dict(json.loads(path.read_text())))
        except Exception:
            continue
    return papers
