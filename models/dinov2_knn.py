"""
DINOv2 + FAISS KNN zero-shot novel-crop detector (Algorithm 6).

The EfficientNet-B4 classifier only knows the crops it was trained on. For crops
outside that set, we use **DINOv2** (Meta's self-supervised ViT) to embed the
image and search a small **reference gallery** of labelled embeddings with a
nearest-neighbour vote. DINOv2 embeddings are general enough to separate unseen
crops/diseases zero-shot.

This module is written to **degrade gracefully**: if torch / DINOv2 weights /
FAISS / the reference gallery are unavailable, ``predict()`` returns an
``available=False`` result instead of raising, so the ensemble simply proceeds
with the other models. That keeps the whole pipeline robust on machines without
the extra assets.

Gallery format (built offline by ``build_gallery``):
    reference_gallery/
        gallery.npz   # arrays: embeddings [N, D] float32, labels [N] str
The vote is inverse-distance weighted over the top-k neighbours.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GALLERY_DIR = "data/reference_gallery"
DEFAULT_MODEL = "facebook/dinov2-small"  # 384-dim, fits 8GB VRAM comfortably
DEFAULT_K = 5


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class DINOv2KNN:
    """Lazy DINOv2 encoder + FAISS (or numpy) KNN over a reference gallery."""

    def __init__(
        self,
        gallery_dir: str = DEFAULT_GALLERY_DIR,
        model_name: str = DEFAULT_MODEL,
        k: int = DEFAULT_K,
    ) -> None:
        self.gallery_dir = Path(gallery_dir)
        self.model_name = model_name
        self.k = k
        self._model = None
        self._processor = None
        self._device = None
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._faiss_index = None
        self._load_error: Optional[str] = None
        self._load_gallery()

    # ── gallery ──────────────────────────────────────────────────────────────
    def _load_gallery(self) -> None:
        npz = self.gallery_dir / "gallery.npz"
        if not npz.exists():
            self._load_error = f"no gallery at {npz}"
            logger.info("DINOv2KNN: %s — novel-crop detection disabled", self._load_error)
            return
        try:
            data = np.load(npz, allow_pickle=True)
            emb = _l2_normalize(data["embeddings"].astype("float32"))
            self._embeddings = emb
            self._labels = data["labels"]
            self._build_index(emb)
            logger.info(
                "DINOv2KNN: loaded gallery with %d reference embeddings (dim=%d)",
                emb.shape[0],
                emb.shape[1],
            )
        except Exception as e:  # corrupt/unreadable gallery — stay disabled
            self._load_error = f"gallery load failed: {e}"
            logger.warning("DINOv2KNN: %s", self._load_error)

    def _build_index(self, emb: np.ndarray) -> None:
        try:
            import faiss  # optional dependency

            index = faiss.IndexFlatIP(emb.shape[1])  # inner product on L2-normed = cosine
            index.add(emb)
            self._faiss_index = index
        except Exception:
            # No FAISS installed — fall back to numpy brute force (fine for a
            # small gallery). Not an error.
            self._faiss_index = None

    @property
    def available(self) -> bool:
        return self._embeddings is not None and self._labels is not None

    # ── encoder ──────────────────────────────────────────────────────────────
    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device).eval()
            logger.info("DINOv2KNN: encoder %s loaded on %s", self.model_name, self._device)
            return True
        except Exception as e:
            self._load_error = f"encoder load failed: {e}"
            logger.warning("DINOv2KNN: %s", self._load_error)
            return False

    def encode(self, image) -> Optional[np.ndarray]:
        """Return an L2-normalised DINOv2 embedding, or None if unavailable."""
        if not self._ensure_model():
            return None
        try:
            import torch

            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            # CLS token (pooler_output) if present, else mean of last hidden state.
            if getattr(out, "pooler_output", None) is not None:
                vec = out.pooler_output[0]
            else:
                vec = out.last_hidden_state[0].mean(dim=0)
            emb = vec.detach().cpu().numpy().astype("float32")
            return _l2_normalize(emb.reshape(1, -1))[0]
        except Exception as e:
            logger.warning("DINOv2KNN: encode failed: %s", e)
            return None

    # ── prediction ───────────────────────────────────────────────────────────
    def predict(self, image) -> dict[str, Any]:
        """Classify ``image`` via KNN over the gallery.

        Returns a dict with ``available`` flag; on success also ``label``,
        ``confidence`` (0..1), ``source`` and ``neighbours``.
        """
        if not self.available:
            return {"available": False, "reason": self._load_error or "gallery missing",
                    "label": "unavailable", "confidence": 0.0, "source": "dinov2_knn"}

        emb = self.encode(image)
        if emb is None:
            return {"available": False, "reason": self._load_error or "encode failed",
                    "label": "unavailable", "confidence": 0.0, "source": "dinov2_knn"}

        distances, indices = self._search(emb, self.k)
        votes: dict[str, float] = {}
        neighbours: list[dict] = []
        for sim, idx in zip(distances, indices):
            if idx < 0:
                continue
            label = str(self._labels[idx])
            # cosine similarity in [-1, 1] -> weight in (0, 1]; closer = higher.
            weight = max(0.0, (float(sim) + 1.0) / 2.0)
            votes[label] = votes.get(label, 0.0) + weight
            neighbours.append({"label": label, "similarity": round(float(sim), 4)})

        if not votes:
            return {"available": False, "reason": "no neighbours", "label": "unavailable",
                    "confidence": 0.0, "source": "dinov2_knn"}

        winner = max(votes, key=votes.get)
        confidence = votes[winner] / sum(votes.values())
        return {
            "available": True,
            "label": winner,
            "confidence": float(confidence),
            "source": "dinov2_knn",
            "neighbours": neighbours,
        }

    def _search(self, emb: np.ndarray, k: int):
        """Return (similarities, indices) for the top-k neighbours."""
        k = min(k, self._embeddings.shape[0])
        if self._faiss_index is not None:
            sims, idxs = self._faiss_index.search(emb.reshape(1, -1), k)
            return sims[0].tolist(), idxs[0].tolist()
        # numpy brute force cosine (embeddings are L2-normalised).
        sims = self._embeddings @ emb
        top = np.argsort(-sims)[:k]
        return sims[top].tolist(), top.tolist()


# ── module-level singleton ────────────────────────────────────────────────────
_INSTANCE: Optional[DINOv2KNN] = None


def get_dinov2_knn(gallery_dir: str = DEFAULT_GALLERY_DIR) -> DINOv2KNN:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DINOv2KNN(gallery_dir=gallery_dir)
    return _INSTANCE


def build_gallery(
    labelled_images: list[tuple[str, Any]],
    out_dir: str = DEFAULT_GALLERY_DIR,
    model_name: str = DEFAULT_MODEL,
) -> int:
    """Build and save a reference gallery from ``(label, PIL.Image)`` pairs.

    Returns the number of embeddings written. Used offline to populate the
    gallery of crops NOT in the EfficientNet training set.
    """
    encoder = DINOv2KNN(gallery_dir=out_dir, model_name=model_name)
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    for label, image in labelled_images:
        emb = encoder.encode(image)
        if emb is not None:
            embeddings.append(emb)
            labels.append(label)
    if not embeddings:
        raise RuntimeError("No embeddings produced — check the DINOv2 encoder install.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "gallery.npz",
        embeddings=np.stack(embeddings).astype("float32"),
        labels=np.array(labels, dtype=object),
    )
    logger.info("DINOv2KNN: wrote gallery with %d embeddings to %s", len(labels), out)
    return len(labels)
