"""
Build ChromaDB Vector Knowledge Base from ICAR Advisories
Used for RAG (Retrieval-Augmented Generation) queries.
Indexes crop disease information for semantic search.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION = None
_CLIENT = None

KB_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = str(KB_ROOT / "chroma_db")


def _load_crop_diseases() -> dict:
    """Load crop diseases knowledge base."""
    path = KB_ROOT / "crop_diseases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_knowledge_db(force_rebuild: bool = False) -> None:
    """
    Build or rebuild the ChromaDB vector store from ICAR knowledge base.
    """
    global _CLIENT, _COLLECTION

    try:
        import chromadb
    except ImportError:
        logger.warning("chromadb not installed — RAG features disabled")
        return

    db_path = Path(CHROMA_PATH)
    if db_path.exists() and not force_rebuild:
        logger.info("ChromaDB already exists, skipping rebuild")
        return

    logger.info("Building ChromaDB knowledge base...")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Delete existing collection if rebuilding
        try:
            client.delete_collection("crop_advisory")
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name="crop_advisory",
            metadata={"description": "ICAR crop advisory knowledge base for Indian agriculture"},
        )

        # Load crop diseases
        kb = _load_crop_diseases()
        diseases = kb.get("diseases", {})

        documents = []
        metadatas = []
        ids = []

        for disease_id, info in diseases.items():
            crop = info.get("crop", "unknown")
            disease_name = info.get("disease", "unknown")
            pathogen = info.get("pathogen", "")
            severity = info.get("severity", "unknown")
            symptoms = ". ".join(info.get("symptoms", []))
            yield_loss = info.get("yield_loss", "varies")

            # Treatment text
            treatments = info.get("treatment", [])
            treatment_text = ". ".join([
                f"{t.get('chemical', '')} at {t.get('dosage', '')} (approved by {t.get('approved_by', 'ICAR')})"
                for t in treatments
            ])

            organic = ". ".join(info.get("organic_treatment", []))
            prevention = ". ".join(info.get("prevention", []))
            resistant = ", ".join(info.get("indian_varieties_resistant", []))

            # Build document text for embedding
            doc_text = (
                f"Disease: {disease_name} in {crop}. "
                f"Caused by {pathogen}. Severity: {severity}. "
                f"Potential yield loss: {yield_loss}. "
                f"Symptoms: {symptoms}. "
                f"Chemical treatment: {treatment_text}. "
                f"Organic treatment: {organic}. "
                f"Prevention: {prevention}. "
                f"Resistant Indian varieties: {resistant}."
            )

            documents.append(doc_text)
            metadatas.append({
                "crop": crop,
                "disease": disease_name,
                "disease_id": disease_id,
                "severity": severity,
                "source": "ICAR",
            })
            ids.append(f"disease_{disease_id}")

        # Add general ICAR advisories
        general_advisories = [
            {
                "text": "Integrated Pest Management (IPM) is the recommended approach by ICAR. "
                        "Use biological control agents like Trichoderma, Pseudomonas, and Beauveria before chemical pesticides. "
                        "Always follow pre-harvest intervals (PHI). Wear protective equipment while spraying.",
                "metadata": {"crop": "general", "disease": "ipm", "source": "ICAR"},
                "id": "advisory_ipm",
            },
            {
                "text": "For organic farming in India, use Neem oil (Azadirachtin) 5ml/L as broad-spectrum pest control. "
                        "Apply Trichoderma viride 5g/L for soil-borne diseases. "
                        "Use Panchagavya 3% for growth promotion. Apply Jeevamrutha 200L/acre for soil health.",
                "metadata": {"crop": "general", "disease": "organic", "source": "ICAR-NBAIR"},
                "id": "advisory_organic",
            },
            {
                "text": "Kharif season (June-October) diseases: Rice blast, Brown spot, Bacterial leaf blight. "
                        "Rabi season (November-March) diseases: Wheat rust, Late blight in potato. "
                        "Summer season: Tomato viral diseases, Sugarcane smut.",
                "metadata": {"crop": "general", "disease": "seasonal", "source": "ICAR"},
                "id": "advisory_seasonal",
            },
            {
                "text": "Banned pesticides in India include Endosulfan (Supreme Court 2011), Monocrotophos, "
                        "Methyl Parathion, Phorate, Dichlorvos, Triazophos. "
                        "Use Neem oil, Trichoderma, Pseudomonas as safe organic alternatives. "
                        "Contact Kisan Call Center 1800-180-1551 for guidance.",
                "metadata": {"crop": "general", "disease": "compliance", "source": "CIB&RC"},
                "id": "advisory_compliance",
            },
        ]

        for adv in general_advisories:
            documents.append(adv["text"])
            metadatas.append(adv["metadata"])
            ids.append(adv["id"])

        # Add all to ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"ChromaDB built with {len(documents)} documents")
        _CLIENT = client
        _COLLECTION = collection

    except Exception as e:
        logger.error(f"Failed to build ChromaDB: {e}")


def get_collection():
    """Get or create the ChromaDB collection."""
    global _CLIENT, _COLLECTION

    if _COLLECTION is not None:
        return _COLLECTION

    try:
        import chromadb
        _CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)

        # Check if collection exists
        try:
            _COLLECTION = _CLIENT.get_collection("crop_advisory")
            return _COLLECTION
        except Exception:
            # Build if not exists
            build_knowledge_db()
            return _COLLECTION

    except ImportError:
        logger.warning("chromadb not installed")
        return None


def rag_query(
    query: str,
    crop: str = "",
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """
    Query the knowledge base using semantic search.

    Args:
        query: Natural language query (disease name, symptoms, etc.)
        crop: Optional crop filter
        n_results: Number of results to return

    Returns:
        List of relevant documents with metadata
    """
    collection = get_collection()
    if collection is None:
        return []

    try:
        where_filter = {"crop": crop} if crop else None

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        # Format results
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": 1.0 - min(dist, 1.0),  # Convert distance to score
            })

        return output

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return []


def symptom_search(
    symptom_description: str,
    crop: str = "",
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search knowledge base by symptom description.
    Farmer says "my leaves are turning yellow with brown spots"
    → finds matching diseases.
    """
    return rag_query(
        query=f"symptoms: {symptom_description}",
        crop=crop,
        n_results=n_results,
    )


# Auto-build on first import if data exists
if (KB_ROOT / "crop_diseases.json").exists():
    try:
        build_knowledge_db()
    except Exception:
        pass

# Export
__all__ = ["build_knowledge_db", "rag_query", "symptom_search", "get_collection"]
