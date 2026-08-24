from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import SATELLITE_DIR, ensure_data_dirs
from .satellite import parse_coordinate, predict_passes
from .satellite_catalog import catalog_orbital_records, get_catalog_entry


_LOCK = threading.RLock()


def _path() -> Path:
    return SATELLITE_DIR / "observer-locations.json"


def _load() -> list[dict]:
    ensure_data_dirs()
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read observer locations: {exc}") from exc


def _write(items: list[dict]) -> None:
    path = _path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_location(*, name: str, latitude_deg: str, longitude_deg: str,
                  elevation_m: float = 0.0, make_default: bool = False,
                  replace_existing: bool = False) -> dict:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or any(ord(char) < 32 for char in name):
        raise ValueError("name must contain 1 through 64 printable characters")
    latitude = parse_coordinate(latitude_deg, axis="latitude")
    longitude = parse_coordinate(longitude_deg, axis="longitude")
    elevation = float(elevation_m)
    if not -500 <= elevation <= 10_000:
        raise ValueError("elevation_m must be from -500 through 10000")
    with _LOCK:
        items = _load()
        existing = next((item for item in items if item["name"].casefold() == name.casefold()), None)
        if existing and not replace_existing:
            raise ValueError("An observer location with this name already exists")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "location_id": existing["location_id"] if existing else f"location-{uuid4().hex[:12]}",
            "name": name, "latitude_deg": latitude, "longitude_deg": longitude,
            "elevation_m": elevation, "is_default": bool(make_default),
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        items = [item for item in items if not existing or item["location_id"] != existing["location_id"]]
        if make_default or not items:
            record["is_default"] = True
            for item in items:
                item["is_default"] = False
        items.append(record)
        _write(items)
        return record


def list_locations() -> list[dict]:
    with _LOCK:
        return sorted(_load(), key=lambda item: (not item.get("is_default", False), item["name"].casefold()))


def get_location(identifier: str | None = None) -> dict:
    items = list_locations()
    if not items:
        raise ValueError("No observer locations are saved; call save_observer_location first")
    if identifier is None or not str(identifier).strip():
        match = next((item for item in items if item.get("is_default")), None)
        if match:
            return match
        raise ValueError("No default observer location is set")
    text = str(identifier).strip().casefold()
    match = next((item for item in items
                  if item["location_id"].casefold() == text or item["name"].casefold() == text), None)
    if not match:
        raise ValueError(f"Observer location not found: {identifier}")
    return match


def delete_location(identifier: str) -> dict:
    target = get_location(identifier)
    with _LOCK:
        items = [item for item in _load() if item["location_id"] != target["location_id"]]
        if target.get("is_default") and items:
            items[0]["is_default"] = True
        _write(items)
    return {"deleted": True, "location": target,
            "new_default": next((item["name"] for item in items if item.get("is_default")), None)}


def plan_observations(*, location: dict, query: str | None = None,
                      category: str | None = None, hours: float = 24.0,
                      minimum_elevation_deg: float = 10.0,
                      candidate_limit: int = 20, result_limit: int = 8) -> dict:
    hours = float(hours)
    minimum_elevation_deg = float(minimum_elevation_deg)
    if not 0 <= minimum_elevation_deg <= 60:
        raise ValueError("minimum_elevation_deg must be from 0 through 60")
    candidate_limit = max(1, min(int(candidate_limit), 25))
    result_limit = max(1, min(int(result_limit), 12))
    records = catalog_orbital_records(query=query, category=category, limit=candidate_limit)
    geometry = []
    for record in records:
        watch = {
            **record, "satellite_name": record["name"],
            "latitude_deg": location["latitude_deg"],
            "longitude_deg": location["longitude_deg"],
            "elevation_m": location["elevation_m"], "frequency_hz": 145_000_000,
            "minimum_elevation_deg": minimum_elevation_deg,
        }
        passes = predict_passes(watch, hours=hours, limit=1)
        if passes:
            geometry.append((record, passes[0]))
    geometry.sort(key=lambda item: (-item[1]["maximum_elevation_deg"], item[1]["aos"]["at"]))
    opportunities, catalog_errors = [], []
    for record, prediction in geometry:
        if len(opportunities) >= result_limit:
            break
        try:
            entry = get_catalog_entry(record["norad_id"])
        except Exception as exc:
            catalog_errors.append({"norad_id": record["norad_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not entry["suggested_downlinks"]:
            continue
        best = entry["suggested_downlinks"][0]
        opportunities.append({
            "rank": 0, "satellite_name": record["name"], "norad_id": record["norad_id"],
            "aos_at": prediction["aos"]["at"], "tca_at": prediction["tca"]["at"],
            "los_at": prediction["los"]["at"],
            "maximum_elevation_deg": prediction["maximum_elevation_deg"],
            "duration_seconds": prediction["duration_seconds"],
            "best_downlink": best, "compatible_downlink_count": len(entry["suggested_downlinks"]),
            "review_required": True,
        })
    opportunities.sort(key=lambda item: (-item["maximum_elevation_deg"], item["aos_at"]))
    for index, item in enumerate(opportunities, 1):
        item["rank"] = index
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "location": location,
        "query": query or None, "category": category or None, "hours": hours,
        "minimum_elevation_deg": minimum_elevation_deg,
        "candidate_count": len(records), "visible_candidate_count": len(geometry),
        "opportunity_count": len(opportunities), "opportunities": opportunities,
        "catalog_errors": catalog_errors,
        "note": "Ranked by maximum elevation among tunable catalog downlinks. Review metadata before creating a profile; planning does not schedule reception.",
    }
