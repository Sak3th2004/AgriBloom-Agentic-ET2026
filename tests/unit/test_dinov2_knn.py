"""Unit tests for DINOv2 KNN (models/dinov2_knn.py).

These exercise the gallery/KNN logic and graceful-degradation paths using a
tiny synthetic gallery — no torch, DINOv2 weights, or FAISS required.
"""
from __future__ import annotations

import numpy as np

from models.dinov2_knn import DINOv2KNN, _l2_normalize, build_gallery


def _write_gallery(tmp_path, vectors, labels):
    out = tmp_path / "gallery.npz"
    np.savez(
        out,
        embeddings=np.array(vectors, dtype="float32"),
        labels=np.array(labels, dtype=object),
    )
    return str(tmp_path)


def test_unavailable_when_no_gallery(tmp_path):
    knn = DINOv2KNN(gallery_dir=str(tmp_path))
    assert knn.available is False
    res = knn.predict(image=object())
    assert res["available"] is False
    assert res["label"] == "unavailable"


def test_l2_normalize_unit_norm():
    v = np.array([[3.0, 4.0]], dtype="float32")
    n = _l2_normalize(v)
    assert np.isclose(np.linalg.norm(n[0]), 1.0)


def test_knn_search_returns_nearest_label(tmp_path, monkeypatch):
    # Two well-separated clusters in 3D.
    gallery_dir = _write_gallery(
        tmp_path,
        vectors=[[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1]],
        labels=["okra_mosaic", "okra_mosaic", "millet_blast", "millet_blast"],
    )
    knn = DINOv2KNN(gallery_dir=gallery_dir, k=3)
    assert knn.available is True

    # Force encode() to return a query near the first cluster (no real model).
    query = _l2_normalize(np.array([[0.95, 0.05, 0.0]], dtype="float32"))[0]
    monkeypatch.setattr(knn, "encode", lambda image: query)

    res = knn.predict(image=object())
    assert res["available"] is True
    assert res["label"] == "okra_mosaic"
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["neighbours"]


def test_knn_confidence_reflects_vote_share(tmp_path, monkeypatch):
    gallery_dir = _write_gallery(
        tmp_path,
        vectors=[[1, 0], [1, 0], [1, 0], [0, 1]],
        labels=["a", "a", "a", "b"],
    )
    knn = DINOv2KNN(gallery_dir=gallery_dir, k=4)
    query = _l2_normalize(np.array([[1.0, 0.0]], dtype="float32"))[0]
    monkeypatch.setattr(knn, "encode", lambda image: query)
    res = knn.predict(image=object())
    assert res["label"] == "a"
    assert res["confidence"] > 0.5  # majority "a"


def test_encode_failure_degrades_gracefully(tmp_path, monkeypatch):
    gallery_dir = _write_gallery(tmp_path, [[1, 0]], ["a"])
    knn = DINOv2KNN(gallery_dir=gallery_dir)
    monkeypatch.setattr(knn, "encode", lambda image: None)  # simulate encoder failure
    res = knn.predict(image=object())
    assert res["available"] is False


def test_build_gallery_raises_without_encoder(tmp_path, monkeypatch):
    # With no DINOv2 encoder available, encode() returns None for everything.
    import models.dinov2_knn as mod

    monkeypatch.setattr(mod.DINOv2KNN, "encode", lambda self, image: None)
    try:
        build_gallery([("a", object())], out_dir=str(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
