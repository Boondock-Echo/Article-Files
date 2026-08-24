from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import DATA_DIR, ensure_data_dirs


_LOCK = threading.RLock()
FEATURE_SCALES = {
    "carrier_prominence_db": 8.0,
    "sideband_imbalance_db": 10.0,
    "occupied_bandwidth_hz": 5_000.0,
    "envelope_coefficient_of_variation": 0.25,
    "instantaneous_frequency_std_hz": 2_500.0,
    "spectral_entropy": 0.30,
    "significant_peak_count": 6.0,
    "dominant_offset_hz": 2_000.0,
}


def _path() -> Path:
    return DATA_DIR / "signal-library.json"


def _load() -> list[dict]:
    ensure_data_dirs()
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read signal library: {exc}") from exc


def _write(items: list[dict]) -> None:
    path = _path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _name(value: str) -> str:
    value = str(value).strip()
    if not 1 <= len(value) <= 80 or re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError("name must contain 1 through 80 printable characters")
    return value


def validate_classification(observation: dict) -> dict:
    if not isinstance(observation, dict) or not isinstance(observation.get("features"), dict):
        raise ValueError("Classification result has no feature dictionary")
    missing = [key for key in FEATURE_SCALES if key not in observation["features"]]
    if missing:
        raise ValueError("Classification features are incomplete: " + ", ".join(missing))
    frequency = int(observation.get("requested_frequency_hz"))
    features = {key: float(observation["features"][key]) for key in FEATURE_SCALES}
    return {
        "frequency_hz": frequency, "features": features,
        "generic_label": observation.get("best_label"),
        "generic_confidence": observation.get("best_confidence"),
        "observed_at": observation.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "source_job_id": observation.get("job_id"),
    }


def _centroid(exemplars: list[dict]) -> dict:
    return {key: sum(item["features"][key] for item in exemplars) / len(exemplars)
            for key in FEATURE_SCALES}


def save_fingerprint(*, name: str, observation: dict, notes: str = "",
                     frequency_tolerance_hz: float = 2_500,
                     replace_existing: bool = False) -> dict:
    name = _name(name)
    notes = str(notes).strip()
    if len(notes) > 1000:
        raise ValueError("notes must contain no more than 1000 characters")
    tolerance = float(frequency_tolerance_hz)
    if not 10 <= tolerance <= 100_000:
        raise ValueError("frequency_tolerance_hz must be from 10 through 100000")
    exemplar = validate_classification(observation)
    with _LOCK:
        items = _load()
        existing = next((item for item in items if item["name"].casefold() == name.casefold()), None)
        if existing and not replace_existing:
            raise ValueError("A signal fingerprint with this name already exists")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "fingerprint_id": existing["fingerprint_id"] if existing else f"signal-{uuid4().hex[:16]}",
            "name": name, "notes": notes,
            "nominal_frequency_hz": exemplar["frequency_hz"],
            "frequency_tolerance_hz": tolerance,
            "expected_generic_label": exemplar.get("generic_label"),
            "exemplars": [exemplar], "exemplar_count": 1,
            "centroid_features": exemplar["features"],
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        items = [item for item in items if not existing or item["fingerprint_id"] != existing["fingerprint_id"]]
        items.append(record); _write(items)
        return record


def list_fingerprints() -> list[dict]:
    with _LOCK:
        return sorted(_load(), key=lambda item: (item["nominal_frequency_hz"], item["name"].casefold()))


def get_fingerprint(identifier: str) -> dict:
    text = str(identifier).strip().casefold()
    match = next((item for item in list_fingerprints()
                  if item["fingerprint_id"].casefold() == text or item["name"].casefold() == text), None)
    if not match:
        raise ValueError(f"Signal fingerprint not found: {identifier}")
    return match


def add_exemplar(identifier: str, observation: dict) -> dict:
    exemplar = validate_classification(observation)
    with _LOCK:
        target = get_fingerprint(identifier)
        if abs(exemplar["frequency_hz"] - target["nominal_frequency_hz"]) > target["frequency_tolerance_hz"]:
            raise ValueError("Classification frequency is outside this fingerprint's tolerance")
        exemplars = list(target["exemplars"])
        exemplars.append(exemplar)
        exemplars = exemplars[-20:]
        target["exemplars"] = exemplars
        target["exemplar_count"] = len(exemplars)
        target["centroid_features"] = _centroid(exemplars)
        target["updated_at"] = datetime.now(timezone.utc).isoformat()
        items = [target if item["fingerprint_id"] == target["fingerprint_id"] else item
                 for item in _load()]
        _write(items); return target


def delete_fingerprint(identifier: str) -> dict:
    with _LOCK:
        target = get_fingerprint(identifier)
        _write([item for item in _load() if item["fingerprint_id"] != target["fingerprint_id"]])
        return target


def _distance(features: dict, reference: dict) -> tuple[float, dict]:
    components = {key: abs(float(features[key]) - float(reference[key])) / scale
                  for key, scale in FEATURE_SCALES.items()}
    distance = math.sqrt(sum(value * value for value in components.values()) / len(components))
    return distance, dict(sorted(components.items(), key=lambda item: item[1], reverse=True))


def match_fingerprints(observation: dict, *, minimum_similarity: float = 0.55,
                       limit: int = 10) -> dict:
    candidate = validate_classification(observation)
    minimum_similarity = float(minimum_similarity)
    if not 0 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be from 0 through 1")
    matches = []
    for fingerprint in list_fingerprints():
        offset = candidate["frequency_hz"] - fingerprint["nominal_frequency_hz"]
        if abs(offset) > fingerprint["frequency_tolerance_hz"]:
            continue
        distances = [_distance(candidate["features"], item["features"])
                     for item in fingerprint["exemplars"]]
        distance, components = min(distances, key=lambda item: item[0])
        similarity = math.exp(-distance)
        label_agrees = (not fingerprint.get("expected_generic_label")
                        or fingerprint["expected_generic_label"] == candidate.get("generic_label"))
        if not label_agrees:
            similarity *= 0.85
        matches.append({
            "fingerprint_id": fingerprint["fingerprint_id"], "name": fingerprint["name"],
            "similarity": round(similarity, 6), "feature_distance": round(distance, 6),
            "frequency_offset_hz": offset, "generic_label_agrees": label_agrees,
            "largest_normalized_differences": [
                {"feature": key, "normalized_difference": round(value, 4)}
                for key, value in list(components.items())[:3]
            ], "exemplar_count": fingerprint["exemplar_count"],
        })
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    matches = matches[:max(1, min(int(limit), 50))]
    best = matches[0] if matches else None
    accepted = bool(best and best["similarity"] >= minimum_similarity)
    return {
        "accepted": accepted, "minimum_similarity": minimum_similarity,
        "best_match": best if accepted else None, "nearest_candidate": best,
        "match_count": len(matches), "matches": matches,
        "observation": candidate,
        "warning": "Similarity is empirical station-local evidence, not transmitter authentication.",
    }
