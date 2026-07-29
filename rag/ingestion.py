"""
Corpus ingestion for Hybrid RAG.

Turns the ICAR knowledge base (``knowledge_base/crop_diseases.json``) plus the
general advisories into a flat list of retrievable :class:`Document` objects.
Both the BM25 and dense retrievers index the same corpus so their rankings are
comparable before fusion.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base"


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# General ICAR advisories carried over from the V1 knowledge base builder.
GENERAL_ADVISORIES: list[Document] = [
    Document(
        "advisory_ipm",
        "Integrated Pest Management (IPM) is the recommended approach by ICAR. "
        "Use biological control agents like Trichoderma, Pseudomonas, and Beauveria "
        "before chemical pesticides. Always follow pre-harvest intervals (PHI). "
        "Wear protective equipment while spraying.",
        {"crop": "general", "disease": "ipm", "source": "ICAR"},
    ),
    Document(
        "advisory_organic",
        "For organic farming in India, use Neem oil (Azadirachtin) 5ml/L as broad-spectrum "
        "pest control. Apply Trichoderma viride 5g/L for soil-borne diseases. Use Panchagavya "
        "3% for growth promotion. Apply Jeevamrutha 200L/acre for soil health.",
        {"crop": "general", "disease": "organic", "source": "ICAR-NBAIR"},
    ),
    Document(
        "advisory_seasonal",
        "Kharif season (June-October) diseases: Rice blast, Brown spot, Bacterial leaf blight. "
        "Rabi season (November-March) diseases: Wheat rust, Late blight in potato. "
        "Summer season: Tomato viral diseases, Sugarcane smut.",
        {"crop": "general", "disease": "seasonal", "source": "ICAR"},
    ),
    Document(
        "advisory_compliance",
        "Banned pesticides in India include Endosulfan (Supreme Court 2011), Monocrotophos, "
        "Methyl Parathion, Phorate, Dichlorvos, Triazophos. Use Neem oil, Trichoderma, "
        "Pseudomonas as safe organic alternatives. Contact Kisan Call Center 1800-180-1551.",
        {"crop": "general", "disease": "compliance", "source": "CIB&RC"},
    ),
]


def _disease_document(disease_id: str, info: dict) -> Document:
    crop = info.get("crop", "unknown")
    disease_name = info.get("disease", "unknown")
    pathogen = info.get("pathogen", "")
    severity = info.get("severity", "unknown")
    symptoms = ". ".join(info.get("symptoms", []))
    yield_loss = info.get("yield_loss", "varies")

    treatments = info.get("treatment", [])
    treatment_text = ". ".join(
        f"{t.get('chemical', '')} at {t.get('dosage', '')} (approved by {t.get('approved_by', 'ICAR')})"
        for t in treatments
    )
    organic = ". ".join(info.get("organic_treatment", []))
    prevention = ". ".join(info.get("prevention", []))
    resistant = ", ".join(info.get("indian_varieties_resistant", []))

    text = (
        f"Disease: {disease_name} in {crop}. Caused by {pathogen}. Severity: {severity}. "
        f"Potential yield loss: {yield_loss}. Symptoms: {symptoms}. "
        f"Chemical treatment: {treatment_text}. Organic treatment: {organic}. "
        f"Prevention: {prevention}. Resistant Indian varieties: {resistant}."
    )
    return Document(
        id=f"disease_{disease_id}",
        text=text,
        metadata={
            "crop": crop,
            "disease": disease_name,
            "disease_id": disease_id,
            "severity": severity,
            "source": "ICAR",
        },
    )


def load_corpus(kb_path: Path | None = None) -> list[Document]:
    """Load the full retrievable corpus (disease docs + general advisories)."""
    path = (kb_path or KB_ROOT / "crop_diseases.json")
    docs: list[Document] = []
    if path.exists():
        try:
            kb = json.loads(path.read_text(encoding="utf-8"))
            for disease_id, info in kb.get("diseases", {}).items():
                docs.append(_disease_document(disease_id, info))
        except Exception as e:
            logger.error("RAG ingestion failed to parse %s: %s", path, e)
    else:
        logger.warning("RAG ingestion: %s not found; only general advisories loaded", path)

    docs.extend(GENERAL_ADVISORIES)
    logger.info("RAG ingestion: %d documents in corpus", len(docs))
    return docs
