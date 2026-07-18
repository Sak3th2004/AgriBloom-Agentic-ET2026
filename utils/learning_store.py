"""
Feedback store for AgriBloom's safe self-learning loop.

This does not auto-train the model from raw farmer messages. It records farmer
feedback and corrections as auditable JSONL so the data can be reviewed,
deduplicated, and used later for retrieval, evaluation, or fine-tuning.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_FEEDBACK_PATH = "data/feedback/farmer_feedback.jsonl"


def feedback_path() -> Path:
    return Path(os.getenv("AGRIBLOOM_FEEDBACK_PATH", DEFAULT_FEEDBACK_PATH))


def record_feedback(payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else feedback_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": payload.get("source", "web"),
        "rating": payload.get("rating"),
        "crop": payload.get("crop"),
        "problem": payload.get("problem"),
        "language": payload.get("language", "en"),
        "farmer_text": payload.get("farmer_text", ""),
        "correction": payload.get("correction", ""),
        "advice": payload.get("advice"),
        "metadata": payload.get("metadata", {}),
        "review_status": "pending",
    }

    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def load_recent_feedback(limit: int = 50, path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else feedback_path()
    if not target.exists():
        return []

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


__all__ = ["record_feedback", "load_recent_feedback", "feedback_path"]

