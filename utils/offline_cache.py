import json
import os
import time
from pathlib import Path
from typing import Any


class OfflineCache:
    def __init__(self, cache_path: str = "models/offline_cache.json") -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.cache_path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, key: str, ttl_seconds: int = 1800) -> Any | None:
        db = self._read()
        record = db.get(key)
        if not record:
            return None
        if time.time() - record.get("timestamp", 0) > ttl_seconds:
            return None
        return record.get("value")

    def set(self, key: str, value: Any) -> None:
        db = self._read()
        db[key] = {"timestamp": time.time(), "value": value}
        self._write(db)


def offline_mode_enabled() -> bool:
    return os.getenv("AGRIBLOOM_OFFLINE_DEFAULT", "false").lower() == "true"
