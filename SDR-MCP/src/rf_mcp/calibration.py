from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, ensure_data_dirs
from .sdr_coordinator import get_receiver

_LOCK = threading.RLock()


def _path() -> Path:
    return DATA_DIR / "receiver-calibrations.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict]:
    path = _path()
    if not path.exists():
        return []
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise RuntimeError("receiver calibration registry is not a JSON list")
    return values


def _write(values: list[dict]) -> None:
    ensure_data_dirs()
    path = _path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_calibration(
    *, receiver_id: str, frequency_correction_ppm: float = 0,
    dbfs_to_dbm_offset_db: float | None = None, reference_frequency_hz: int | None = None,
    reference_source: str = "", notes: str = "", replace_existing: bool = False,
) -> dict:
    receiver = get_receiver(receiver_id)
    ppm = float(frequency_correction_ppm)
    if not math.isfinite(ppm) or not -1_000 <= ppm <= 1_000:
        raise ValueError("frequency_correction_ppm must be finite and from -1000 through 1000")
    offset = None if dbfs_to_dbm_offset_db is None else float(dbfs_to_dbm_offset_db)
    if offset is not None and (not math.isfinite(offset) or not -300 <= offset <= 300):
        raise ValueError("dbfs_to_dbm_offset_db must be finite and from -300 through 300")
    if offset is not None and not reference_source.strip():
        raise ValueError("reference_source is required for calibrated dBm conversion")
    reference_frequency = None if reference_frequency_hz is None else int(reference_frequency_hz)
    if reference_frequency is not None and reference_frequency <= 0:
        raise ValueError("reference_frequency_hz must be positive")
    with _LOCK:
        values = _load()
        existing = next((item for item in values if item["receiver_id"] == receiver_id), None)
        if existing and not replace_existing:
            raise ValueError("Calibration already exists; set replace_existing=true")
        now = _now()
        calibration = {
            "receiver_id": receiver_id, "receiver_backend": receiver["backend"],
            "frequency_correction_ppm": ppm, "dbfs_to_dbm_offset_db": offset,
            "reference_frequency_hz": reference_frequency,
            "reference_source": reference_source.strip()[:500], "notes": notes.strip()[:1000],
            "created_at": existing["created_at"] if existing else now, "updated_at": now,
        }
        values = [item for item in values if item["receiver_id"] != receiver_id]
        values.append(calibration)
        _write(values)
    return calibration


def get_calibration(receiver_id: str, *, required: bool = True) -> dict | None:
    with _LOCK:
        item = next((item for item in _load() if item["receiver_id"] == receiver_id), None)
    if item is None and required:
        raise KeyError(f"No calibration profile for receiver: {receiver_id}")
    return item


def list_calibrations() -> list[dict]:
    with _LOCK:
        return sorted(_load(), key=lambda item: item["receiver_id"])


def delete_calibration(receiver_id: str, *, confirm_delete: bool = False) -> dict:
    if not confirm_delete:
        raise ValueError("Deleting calibration requires confirm_delete=true")
    existing = get_calibration(receiver_id)
    with _LOCK:
        _write([item for item in _load() if item["receiver_id"] != receiver_id])
    return {"deleted": True, "calibration": existing}
