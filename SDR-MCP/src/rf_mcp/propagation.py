from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from .activity import RF_BANDS, build_activity_dashboard
from .config import RESULT_DIR, PLOT_DIR, ensure_data_dirs


SWPC_URLS = {
    "solar_flux": "https://services.swpc.noaa.gov/products/summary/10cm-flux.json",
    "planetary_k": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "scales": "https://services.swpc.noaa.gov/products/noaa-scales.json",
}
HF_BANDS = {name: definition for name, definition in RF_BANDS.items()
            if definition["stop_hz"] <= 31_000_000}
SPACE_WEATHER_CACHE_TTL = timedelta(minutes=15)


def _cache_path() -> Path:
    return RESULT_DIR / "space-weather-cache.json"


def _download_json(url: str, timeout_seconds: float = 12.0):
    request = Request(url, headers={"User-Agent": "rf-mcp-propagation/0.50"})
    with urlopen(request, timeout=max(3.0, min(float(timeout_seconds), 30.0))) as response:  # noqa: S310
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"NOAA SWPC returned HTTP {response.status}")
        payload = response.read(500_001)
    if len(payload) > 500_000:
        raise RuntimeError("NOAA SWPC response exceeded 500 KB")
    return json.loads(payload.decode("utf-8"))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_flux(value) -> dict:
    if not isinstance(value, dict):
        return {"value_sfu": None, "observed_at": None}
    flux = next((_float(value.get(key)) for key in ("Flux", "flux", "Value", "value")
                 if _float(value.get(key)) is not None), None)
    return {"value_sfu": flux,
            "observed_at": value.get("TimeStamp") or value.get("time_tag")}


def _parse_kp(value) -> dict:
    rows = value if isinstance(value, list) else []
    if rows and isinstance(rows[0], list):
        header = [str(item).lower() for item in rows[0]]
        data = rows[1:]
        kp_index = next((i for i, key in enumerate(header)
                         if "planetary_k" in key or key in {"kp", "k"}), 1)
        time_index = next((i for i, key in enumerate(header) if "time" in key), 0)
        for row in reversed(data):
            if isinstance(row, list) and len(row) > max(kp_index, time_index):
                kp = _float(row[kp_index])
                if kp is not None:
                    return {"value": kp, "observed_at": row[time_index]}
    if rows and isinstance(rows[-1], dict):
        row = rows[-1]
        return {"value": _float(row.get("planetary_k_index") or row.get("kp")),
                "observed_at": row.get("time_tag") or row.get("TimeStamp")}
    return {"value": None, "observed_at": None}


def _parse_scales(value) -> dict:
    current = value.get("0", {}) if isinstance(value, dict) else {}
    output = {"observed_at": None}
    if current:
        output["observed_at"] = " ".join(filter(None, [current.get("DateStamp"),
                                                        current.get("TimeStamp")])) or None
    for key in ("R", "S", "G"):
        item = current.get(key, {}) if isinstance(current, dict) else {}
        try:
            scale = int(item.get("Scale") or 0)
        except (TypeError, ValueError):
            scale = None
        output[key] = {"scale": scale, "text": item.get("Text")}
    return output


def fetch_space_weather(*, force_refresh: bool = False) -> dict:
    ensure_data_dirs()
    path = _cache_path()
    stale_cache = None
    if path.exists():
        try:
            stale_cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if not force_refresh and path.exists():
        try:
            cached = stale_cache or {}
            fetched = datetime.fromisoformat(cached["fetched_at"])
            if datetime.now(timezone.utc) - fetched < SPACE_WEATHER_CACHE_TTL:
                return {**cached, "cache_hit": True}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    raw, errors = {}, {}
    for name, url in SWPC_URLS.items():
        try:
            raw[name] = _download_json(url)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    if not raw and stale_cache:
        return {**stale_cache, "cache_hit": True, "stale": True,
                "refresh_errors": errors}
    result = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA Space Weather Prediction Center",
        "solar_flux_10_7_cm": _parse_flux(raw.get("solar_flux")),
        "planetary_k_index": _parse_kp(raw.get("planetary_k")),
        "noaa_scales": _parse_scales(raw.get("scales")),
        "errors": errors, "complete": not errors, "cache_hit": False, "stale": False,
        "cache_ttl_seconds": int(SPACE_WEATHER_CACHE_TTL.total_seconds()),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def band_for_frequency(frequency_hz: float) -> str | None:
    return next((name for name, definition in HF_BANDS.items()
                 if definition["start_hz"] <= frequency_hz <= definition["stop_hz"]), None)


def summarize_local_propagation(*, spots: list[dict], activity: dict[str, dict],
                                time_stations: list[dict], hours: float) -> dict:
    bands = []
    for name, definition in HF_BANDS.items():
        band_spots = [spot for spot in spots
                      if band_for_frequency(float(spot.get("rf_frequency_hz")
                                                  or spot.get("dial_frequency_hz", 0))) == name]
        callsigns = sorted({str(spot.get("callsign")) for spot in band_spots if spot.get("callsign")})
        snrs = [float(spot["snr_db"]) for spot in band_spots if spot.get("snr_db") is not None]
        dashboard = activity.get(name)
        latest = (dashboard or {}).get("latest_run", {})
        spot_count = len(band_spots)
        occupied = latest.get("occupied_bin_fraction")
        signal_count = int(latest.get("signal_count") or 0)
        if spot_count >= 10 or (occupied is not None and occupied >= 0.05):
            evidence = "strong_local_evidence"
        elif spot_count >= 3 or signal_count >= 3:
            evidence = "moderate_local_evidence"
        elif spot_count or signal_count:
            evidence = "limited_local_evidence"
        else:
            evidence = "no_recent_local_evidence"
        bands.append({
            "band_name": name, "label": definition["label"],
            "start_hz": definition["start_hz"], "stop_hz": definition["stop_hz"],
            "evidence_rating": evidence, "weak_signal_spot_count": spot_count,
            "unique_callsign_count": len(callsigns), "unique_callsigns": callsigns[:50],
            "median_snr_db": median(snrs) if snrs else None,
            "activity_signal_count": signal_count,
            "occupied_bin_fraction": occupied,
            "noise_floor_delta_db": (dashboard or {}).get("latest_vs_baseline", {}).get("noise_floor_delta_db"),
        })
    bands.sort(key=lambda item: ({"strong_local_evidence": 0, "moderate_local_evidence": 1,
                                  "limited_local_evidence": 2,
                                  "no_recent_local_evidence": 3}[item["evidence_rating"]],
                                 item["start_hz"]))
    return {"lookback_hours": hours, "band_count": len(bands), "bands": bands,
            "weak_signal_spot_count": len(spots), "time_station_observations": time_stations,
            "time_station_detection_count": sum(item.get("detected", False) for item in time_stations)}


def space_weather_interpretation(snapshot: dict) -> list[dict]:
    notes = []
    kp = snapshot.get("planetary_k_index", {}).get("value")
    flux = snapshot.get("solar_flux_10_7_cm", {}).get("value_sfu")
    scales = snapshot.get("noaa_scales", {})
    if kp is not None:
        notes.append({"factor": "planetary_k_index", "value": kp,
                      "interpretation": ("geomagnetically disturbed; HF paths, especially polar paths, may be degraded"
                                         if kp >= 5 else "quiet-to-unsettled geomagnetic context" if kp < 4
                                         else "active geomagnetic context")})
    if flux is not None:
        notes.append({"factor": "10_7_cm_solar_flux", "value_sfu": flux,
                      "interpretation": "solar-activity context only; local reception evidence remains decisive"})
    radio_scale = (scales.get("R") or {}).get("scale")
    if radio_scale is not None:
        notes.append({"factor": "NOAA_R_scale", "value": radio_scale,
                      "interpretation": ("NOAA reports a radio-blackout condition affecting sunlit-side HF"
                                         if radio_scale >= 1 else "no NOAA R-scale radio blackout reported")})
    return notes


def save_propagation_plot(report: dict) -> Path:
    from .plotting import pyplot

    plt = pyplot()
    ensure_data_dirs()
    bands = report["local_evidence"]["bands"]
    labels = [item["band_name"] for item in bands]
    spots = [item["weak_signal_spot_count"] for item in bands]
    occupancy = [np.nan if item["occupied_bin_fraction"] is None
                 else item["occupied_bin_fraction"] * 100 for item in bands]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(labels, spots, color="#1677b8")
    axes[0].set_ylabel("FT8/FT4/WSPR spots")
    axes[0].set_title("HF propagation evidence observed by MiniRackDisplay")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, occupancy, color="#f36d2e")
    axes[1].set_ylabel("Occupied spectral bins (%)")
    axes[1].set_xlabel("Band")
    axes[1].grid(axis="y", alpha=0.25)
    path = PLOT_DIR / f"propagation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.png"
    figure.savefig(path, dpi=150); plt.close(figure)
    return path


def save_propagation_exports(report: dict) -> tuple[Path, Path]:
    ensure_data_dirs()
    stem = f"propagation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    json_path, csv_path = RESULT_DIR / f"{stem}.json", RESULT_DIR / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["band_name", "label", "evidence_rating", "weak_signal_spot_count",
              "unique_callsign_count", "median_snr_db", "activity_signal_count",
              "occupied_bin_fraction", "noise_floor_delta_db"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(report["local_evidence"]["bands"])
    return json_path, csv_path
