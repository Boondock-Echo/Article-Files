from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import SATELLITE_DIR, TUNING_RANGES_HZ, ensure_data_dirs
from .satellite import tle_checksum_valid


CATEGORY_GROUPS = {
    "active": "ACTIVE", "amateur": "AMATEUR", "weather": "WEATHER",
    "noaa": "NOAA", "goes": "GOES", "stations": "STATIONS",
    "earth_resources": "RESOURCE",
}
CACHE_TTL = timedelta(hours=2)


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()
    return SATELLITE_DIR / "catalog-cache" / f"{digest}.json"


def _download_text(url: str, *, timeout_seconds: float = 15.0,
                   cache_ttl: timedelta = CACHE_TTL) -> str:
    ensure_data_dirs()
    path = _cache_path(url)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(cached["fetched_at"])
            if datetime.now(timezone.utc) - fetched < cache_ttl:
                return str(cached["body"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    request = Request(url, headers={"User-Agent": "rf-mcp-catalog/0.50"})
    with urlopen(request, timeout=max(3.0, min(float(timeout_seconds), 30.0))) as response:  # noqa: S310
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"Satellite catalog returned HTTP {response.status}")
        body = response.read(4_000_001)
    if len(body) > 4_000_000:
        raise RuntimeError("Satellite catalog response exceeded 4 MB")
    text = body.decode("utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(),
                                     "url": url, "body": text}), encoding="utf-8")
    temporary.replace(path)
    return text


def _parse_tle_sets(text: str) -> list[dict]:
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    output, index = [], 0
    while index < len(lines):
        if lines[index].startswith("1 "):
            name, line1 = None, lines[index]
        else:
            name = lines[index].strip()
            index += 1
            if index >= len(lines) or not lines[index].startswith("1 "):
                index += 1
                continue
            line1 = lines[index]
        if index + 1 >= len(lines) or not lines[index + 1].startswith("2 "):
            index += 1
            continue
        line2 = lines[index + 1]
        index += 2
        if not tle_checksum_valid(line1) or not tle_checksum_valid(line2):
            continue
        try:
            norad = int(line1[2:7])
        except ValueError:
            continue
        output.append({"name": name or f"NORAD {norad}", "norad_id": norad,
                       "tle_line1": line1, "tle_line2": line2})
    return output


def search_catalog(*, query: str | None = None, category: str | None = None,
                   limit: int = 50) -> dict:
    query = str(query or "").strip()
    category = str(category or "").strip().lower() or None
    if not query and not category:
        raise ValueError("Provide query or category")
    if category and category not in CATEGORY_GROUPS:
        raise ValueError("category must be one of: " + ", ".join(CATEGORY_GROUPS))
    if query:
        parameters = {"NAME": query, "FORMAT": "TLE"}
        source = f"name:{query}"
    else:
        parameters = {"GROUP": CATEGORY_GROUPS[category], "FORMAT": "TLE"}
        source = f"category:{category}"
    url = "https://celestrak.org/NORAD/elements/gp.php?" + urlencode(parameters)
    records = _parse_tle_sets(_download_text(url))
    limit = max(1, min(int(limit), 200))
    return {"source": source, "category": category, "query": query or None,
            "count": min(len(records), limit), "total_matches": len(records),
            "satellites": [{"name": item["name"], "norad_id": item["norad_id"]}
                           for item in records[:limit]],
            "categories": list(CATEGORY_GROUPS),
            "cached_for_seconds": int(CACHE_TTL.total_seconds())}


def catalog_orbital_records(*, query: str | None = None,
                            category: str | None = None,
                            limit: int = 25) -> list[dict]:
    """Return validated current element sets for bounded observation planning."""
    query = str(query or "").strip()
    category = str(category or "").strip().lower() or None
    if not query and not category:
        raise ValueError("Provide query or category")
    if category and category not in CATEGORY_GROUPS:
        raise ValueError("category must be one of: " + ", ".join(CATEGORY_GROUPS))
    parameters = ({"NAME": query, "FORMAT": "TLE"} if query else
                  {"GROUP": CATEGORY_GROUPS[category], "FORMAT": "TLE"})
    url = "https://celestrak.org/NORAD/elements/gp.php?" + urlencode(parameters)
    return _parse_tle_sets(_download_text(url))[:max(1, min(int(limit), 25))]


def _tle_for(identifier: str | int) -> dict:
    text = str(identifier).strip()
    parameter = {"CATNR": int(text)} if text.isdigit() else {"NAME": text}
    url = "https://celestrak.org/NORAD/elements/gp.php?" + urlencode(
        {**parameter, "FORMAT": "TLE"}
    )
    matches = _parse_tle_sets(_download_text(url))
    if not matches:
        raise ValueError(f"No current CelesTrak satellite matched: {identifier}")
    if len(matches) > 1:
        exact = [item for item in matches if item["name"].casefold() == text.casefold()]
        if len(exact) == 1:
            return exact[0]
        choices = ", ".join(f"{item['name']} ({item['norad_id']})" for item in matches[:10])
        raise ValueError(f"Satellite name is ambiguous; use a NORAD ID. Matches: {choices}")
    return matches[0]


def _json_items(text: str) -> list[dict]:
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.get("results") or [])
    return []


def _airspy_tunable(frequency_hz: int) -> bool:
    return any(low <= frequency_hz <= high for low, high in TUNING_RANGES_HZ)


def _mode_recommendation(transmitter: dict) -> tuple[str, str, str]:
    mode = str(transmitter.get("mode") or "").strip().upper()
    description = str(transmitter.get("description") or "").upper()
    baud = transmitter.get("baud")
    if "SSTV" in mode or "SSTV" in description:
        return "sstv", "nfm", "full"
    if ("AFSK" in mode or "AX.25" in description or "APRS" in description) and (
        baud is None or int(float(baud)) <= 2400
    ):
        return "ax25_afsk1200", "nfm", "full"
    if "G3RUH" in mode or "G3RUH" in description:
        return "ax25_g3ruh9600", "nfm", "full"
    if mode in {"FM", "FMN", "NFM", "VOICE"} or "VOICE" in description:
        return "nfm_audio", "nfm", "audio"
    receiver = "usb" if mode in {"USB", "LSB", "CW", "BPSK", "PSK", "QPSK"} else "nfm"
    return "capture_only", receiver, "capture"


def get_catalog_entry(identifier: str | int) -> dict:
    tle = _tle_for(identifier)
    url = "https://db.satnogs.org/api/transmitters/?" + urlencode(
        {"satellite__norad_cat_id": tle["norad_id"], "format": "json"}
    )
    try:
        raw_transmitters = _json_items(_download_text(url, cache_ttl=timedelta(hours=1)))
        transmitter_error = None
    except Exception as exc:
        raw_transmitters, transmitter_error = [], f"{type(exc).__name__}: {exc}"
    transmitters = []
    for index, item in enumerate(raw_transmitters):
        frequency = item.get("downlink_low") or item.get("downlink_high")
        try:
            frequency = int(frequency) if frequency is not None else None
        except (TypeError, ValueError):
            frequency = None
        recommended_mode, receiver_mode, support = _mode_recommendation(item)
        tunable = bool(frequency and _airspy_tunable(frequency))
        status = str(item.get("status") or "unknown").lower()
        uuid = str(item.get("uuid") or f"transmitter-{index + 1}")
        transmitters.append({
            "transmitter_id": uuid, "description": item.get("description") or "Unnamed downlink",
            "frequency_hz": frequency, "frequency_high_hz": item.get("downlink_high"),
            "mode": item.get("mode"), "baud": item.get("baud"),
            "service": item.get("service"), "status": status,
            "alive": item.get("alive"), "unconfirmed": bool(item.get("unconfirmed", False)),
            "airspy_tunable": tunable, "recommended_rf_mcp_mode": recommended_mode,
            "recommended_receiver_mode": receiver_mode,
            "decoder_support": support if tunable else "unavailable",
            "compatibility_note": ("Within Airspy HF+ tuning range" if tunable else
                                   "Outside Airspy HF+ range or missing frequency"),
        })
    compatible = [item for item in transmitters
                  if item["airspy_tunable"] and item["status"] not in {"inactive", "invalid"}]
    compatible.sort(key=lambda item: (item["decoder_support"] != "full",
                                      item["frequency_hz"] or 0))
    downlinks = []
    seen = set()
    for item in compatible:
        key = (item["frequency_hz"], item["recommended_rf_mcp_mode"])
        if key in seen or len(downlinks) >= 16:
            continue
        seen.add(key)
        identifier_text = re.sub(r"[^a-z0-9]+", "-", str(item["description"]).lower()).strip("-")
        downlinks.append({
            "downlink_id": (identifier_text or f"downlink-{len(downlinks) + 1}")[:32].rstrip("-"),
            "label": str(item["description"])[:64], "frequency_hz": item["frequency_hz"],
            "mode": item["recommended_rf_mcp_mode"],
            "receiver_mode": item["recommended_receiver_mode"],
            "priority": len(downlinks) + 1, "enabled": True,
            "retain_audio": item["recommended_rf_mcp_mode"] in {"sstv", "nfm_audio"},
            "catalog_transmitter_id": item["transmitter_id"],
        })
    return {**tle, "transmitters": transmitters, "compatible_transmitter_count": len(compatible),
            "suggested_downlinks": downlinks, "transmitter_catalog_error": transmitter_error,
            "sources": {"orbital_elements": "CelesTrak", "transmitters": "SatNOGS DB"},
            "review_required": True,
            "warning": ("Catalog data is current public metadata, not proof that a transmitter "
                        "is active at this moment. Review modes and frequencies before saving.")}


def selected_downlinks(entry: dict, transmitter_ids: list[str] | None = None) -> list[dict]:
    downlinks = list(entry["suggested_downlinks"])
    if transmitter_ids:
        wanted = set(map(str, transmitter_ids))
        downlinks = [item for item in downlinks if item.get("catalog_transmitter_id") in wanted]
        missing = wanted - {item.get("catalog_transmitter_id") for item in downlinks}
        if missing:
            raise ValueError("Selected transmitters are unavailable or incompatible: "
                             + ", ".join(sorted(missing)))
    if not downlinks:
        raise ValueError("This satellite has no selected downlink within the Airspy HF+ range")
    return [{key: value for key, value in item.items() if key != "catalog_transmitter_id"}
            for item in downlinks]
