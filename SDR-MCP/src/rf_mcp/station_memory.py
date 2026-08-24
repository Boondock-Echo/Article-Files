from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import DATA_DIR, TUNING_RANGES_HZ, ensure_data_dirs
from .signal_analysis import DEFAULT_BANDWIDTHS_HZ, SUPPORTED_MODES, validate_bandwidth

MODES = (*SUPPORTED_MODES, "broadcast_fm")
_LOCK = threading.RLock()


def _path() -> Path:
    return DATA_DIR / "station-memories.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict]:
    path = _path()
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("Station memory file is not a JSON list")
    return value


def _write(items: list[dict]) -> None:
    ensure_data_dirs()
    path = _path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(*, name: str, frequency_hz: int, mode: str, bandwidth_hz: int | None = None,
              notes: str = "", tags: list[str] | None = None, enabled: bool = True) -> dict:
    name = str(name).strip()
    if not name or len(name) > 100:
        raise ValueError("name must contain 1-100 characters")
    frequency_hz = int(frequency_hz)
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if mode == "broadcast_fm":
        if not 88_000_000 <= frequency_hz <= 108_000_000:
            raise ValueError("broadcast_fm frequency must be from 88 through 108 MHz")
        bandwidth_hz = 200_000
    else:
        if not any(low <= frequency_hz <= high for low, high in TUNING_RANGES_HZ):
            raise ValueError("frequency is outside the Airspy HF+ tuning ranges")
        bandwidth_hz = validate_bandwidth(mode, bandwidth_hz)
    clean_tags = []
    for tag in tags or []:
        value = re.sub(r"\s+", " ", str(tag).strip().lower())
        if value and value not in clean_tags:
            clean_tags.append(value[:40])
    if len(clean_tags) > 20:
        raise ValueError("at most 20 tags are allowed")
    return {"name": name, "frequency_hz": frequency_hz, "mode": mode,
            "bandwidth_hz": bandwidth_hz, "notes": str(notes).strip()[:1000],
            "tags": clean_tags, "enabled": bool(enabled)}


def save(*, memory_id: str | None = None, replace_existing: bool = False, **values) -> dict:
    item = normalize(**values)
    with _LOCK:
        items = _load()
        existing = None
        if memory_id:
            existing = next((x for x in items if x["memory_id"] == memory_id), None)
            if existing is None:
                raise KeyError(f"Unknown station memory: {memory_id}")
        else:
            existing = next((x for x in items if x["name"].casefold() == item["name"].casefold()), None)
            if existing and not replace_existing:
                raise ValueError("A station memory with that name already exists; use replace_existing=true")
        timestamp = _now()
        item.update({"memory_id": existing["memory_id"] if existing else f"mem-{uuid4().hex[:12]}",
                     "created_at": existing["created_at"] if existing else timestamp,
                     "updated_at": timestamp})
        items = [x for x in items if x["memory_id"] != item["memory_id"]]
        items.append(item)
        _write(items)
    return item


def list_memories(*, query: str | None = None, mode: str | None = None,
                  enabled_only: bool = False) -> list[dict]:
    query = (query or "").strip().casefold()
    mode = mode.strip().lower() if mode else None
    if mode and mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    with _LOCK:
        items = _load()
    if enabled_only:
        items = [x for x in items if x["enabled"]]
    if mode:
        items = [x for x in items if x["mode"] == mode]
    if query:
        items = [x for x in items if query in " ".join(
            [x["name"], x["notes"], *x["tags"]]).casefold()]
    return sorted(items, key=lambda x: (x["frequency_hz"], x["name"].casefold()))


def get(memory_id_or_name: str) -> dict:
    value = memory_id_or_name.strip().casefold()
    for item in list_memories():
        if item["memory_id"].casefold() == value or item["name"].casefold() == value:
            return item
    raise KeyError(f"Unknown station memory: {memory_id_or_name}")


def delete(memory_id_or_name: str, *, confirm_delete: bool = False) -> dict:
    if not confirm_delete:
        raise ValueError("Deleting a station memory requires confirm_delete=true")
    target = get(memory_id_or_name)
    with _LOCK:
        _write([x for x in _load() if x["memory_id"] != target["memory_id"]])
    return {"deleted": True, "memory": target}
