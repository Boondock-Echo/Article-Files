from __future__ import annotations

import re
from math import ceil
from typing import Any

from .airspyhf import validate_duration, validate_frequency
from .config import TUNING_RANGES_HZ
from .signal_analysis import normalize_mode, validate_bandwidth


PRESET_TYPES = (
    "band_scan", "band_survey", "activity_monitor", "monitor", "watchlist", "sstv", "sstv_watch",
    "station_memory_scan",
)


def _object(config: dict | None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    return dict(config)


def _keys(config: dict, allowed: set[str], required: set[str]) -> None:
    unknown = sorted(set(config) - allowed)
    missing = sorted(required - set(config))
    if unknown:
        raise ValueError(f"Unknown preset config fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing preset config fields: {', '.join(missing)}")


def _power_of_two(value: Any, low: int, high: int, field: str) -> int:
    value = int(value)
    if value < low or value > high or value & (value - 1):
        raise ValueError(f"{field} must be a power of two from {low} through {high}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return value


def _same_range(start_hz: int, stop_hz: int) -> bool:
    return any(low <= start_hz < stop_hz <= high for low, high in TUNING_RANGES_HZ)


def _band_config(config: dict, survey: bool) -> dict:
    allowed = {
        "start_frequency_hz",
        "stop_frequency_hz",
        "capture_duration_seconds",
        "overlap_fraction",
        "fft_size",
        "threshold_above_noise_db",
        "minimum_signal_spacing_hz",
        "attenuation_steps",
        "max_signals",
    }
    if survey:
        allowed |= {
            "classify_top_signals",
            "classification_duration_seconds",
            "classification_bandwidth_hz",
        }
    _keys(config, allowed, {"start_frequency_hz", "stop_frequency_hz"})
    start = validate_frequency(config["start_frequency_hz"])
    stop = validate_frequency(config["stop_frequency_hz"])
    if not _same_range(start, stop) or stop - start < 10_000:
        raise ValueError("Band endpoints must span at least 10000 Hz in one HF+ tuning range")
    result = {
        "start_frequency_hz": start,
        "stop_frequency_hz": stop,
        "capture_duration_seconds": validate_duration(
            config.get("capture_duration_seconds", 1.0)
        ),
        "overlap_fraction": float(config.get("overlap_fraction", 0.15)),
        "fft_size": _power_of_two(config.get("fft_size", 8_192), 1_024, 65_536, "fft_size"),
        "threshold_above_noise_db": float(config.get("threshold_above_noise_db", 8.0)),
        "minimum_signal_spacing_hz": float(
            config.get("minimum_signal_spacing_hz", 1_000)
        ),
        "attenuation_steps": int(config.get("attenuation_steps", 1)),
        "max_signals": int(config.get("max_signals", 100)),
    }
    if not 0.05 <= result["overlap_fraction"] <= 0.5:
        raise ValueError("overlap_fraction must be from 0.05 through 0.5")
    if not 3 <= result["threshold_above_noise_db"] <= 60:
        raise ValueError("threshold_above_noise_db must be from 3 through 60")
    if not 100 <= result["minimum_signal_spacing_hz"] <= 100_000:
        raise ValueError("minimum_signal_spacing_hz must be from 100 through 100000")
    if not 0 <= result["attenuation_steps"] <= 8:
        raise ValueError("attenuation_steps must be from 0 through 8")
    if not 1 <= result["max_signals"] <= 500:
        raise ValueError("max_signals must be from 1 through 500")
    if survey:
        result.update(
            {
                "classify_top_signals": int(config.get("classify_top_signals", 10)),
                "classification_duration_seconds": validate_duration(
                    config.get("classification_duration_seconds", 2.0)
                ),
                "classification_bandwidth_hz": int(
                    config.get("classification_bandwidth_hz", 30_000)
                ),
            }
        )
        if not 1 <= result["classify_top_signals"] <= 20:
            raise ValueError("classify_top_signals must be from 1 through 20")
        if not 2_000 <= result["classification_bandwidth_hz"] <= 50_000:
            raise ValueError("classification_bandwidth_hz must be from 2000 through 50000")
    return result


def _monitor_config(config: dict) -> dict:
    allowed = {
        "frequency_hz",
        "mode",
        "bandwidth_hz",
        "total_duration_seconds",
        "capture_duration_seconds",
        "interval_seconds",
        "fft_size",
        "waterfall_span_hz",
        "record_audio_on_activity",
    }
    _keys(config, allowed, {"frequency_hz"})
    mode = normalize_mode(config.get("mode", "am"))
    bandwidth = validate_bandwidth(mode, config.get("bandwidth_hz"))
    capture_duration = validate_duration(config.get("capture_duration_seconds", 2.0))
    total_duration = float(config.get("total_duration_seconds", 300))
    interval = float(config.get("interval_seconds", 5))
    waterfall = int(config.get("waterfall_span_hz", 100_000))
    if not 10 <= total_duration <= 3_600:
        raise ValueError("total_duration_seconds must be from 10 through 3600")
    if not capture_duration <= interval <= 300:
        raise ValueError("interval_seconds must be at least capture duration and no more than 300")
    if ceil(total_duration / interval) > 1_000:
        raise ValueError("monitor preset would exceed the 1000-capture limit")
    if not bandwidth <= waterfall <= 600_000:
        raise ValueError("waterfall_span_hz must cover bandwidth and be no more than 600000")
    return {
        "frequency_hz": validate_frequency(config["frequency_hz"]),
        "mode": mode,
        "bandwidth_hz": bandwidth,
        "total_duration_seconds": total_duration,
        "capture_duration_seconds": capture_duration,
        "interval_seconds": interval,
        "fft_size": _power_of_two(config.get("fft_size", 8_192), 1_024, 65_536, "fft_size"),
        "waterfall_span_hz": waterfall,
        "record_audio_on_activity": _boolean(
            config.get("record_audio_on_activity", False), "record_audio_on_activity"
        ),
    }


def _watchlist_config(config: dict) -> dict:
    allowed = {"entries", "duration_seconds", "analysis_bandwidth_hz", "fft_size"}
    _keys(config, allowed, {"entries"})
    entries = config["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 10:
        raise ValueError("watchlist entries must be an array containing 1 through 10 items")
    normalized_entries = []
    seen = set()
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"watchlist entry {index} must be an object")
        _keys(item, {"frequency_hz", "label", "enabled", "notes"}, {"frequency_hz"})
        frequency = validate_frequency(item["frequency_hz"])
        if frequency in seen:
            raise ValueError(f"Duplicate watchlist frequency: {frequency}")
        seen.add(frequency)
        label = str(item.get("label", f"{frequency} Hz")).strip()
        notes = str(item.get("notes", "")).strip()
        if not label or len(label) > 80:
            raise ValueError("watchlist labels must contain 1 through 80 characters")
        if len(notes) > 500:
            raise ValueError("watchlist notes must contain no more than 500 characters")
        enabled = _boolean(item.get("enabled", True), f"watchlist entry {index} enabled")
        normalized_entries.append(
            {
                "frequency_hz": frequency,
                "label": label,
                "enabled": enabled,
                "notes": notes,
            }
        )
    if not any(item["enabled"] for item in normalized_entries):
        raise ValueError("watchlist must contain at least one enabled entry")
    bandwidth = int(config.get("analysis_bandwidth_hz", 30_000))
    if not 2_000 <= bandwidth <= 50_000:
        raise ValueError("analysis_bandwidth_hz must be from 2000 through 50000")
    return {
        "entries": normalized_entries,
        "duration_seconds": validate_duration(config.get("duration_seconds", 2.0)),
        "analysis_bandwidth_hz": bandwidth,
        "fft_size": _power_of_two(config.get("fft_size", 16_384), 4_096, 65_536, "fft_size"),
    }


def _sstv_config(config: dict) -> dict:
    allowed = {
        "frequency_hz", "duration_seconds", "receiver_mode", "retain_audio",
        "retain_iq", "deduplicate",
    }
    _keys(config, allowed, {"frequency_hz"})
    duration = float(config.get("duration_seconds", 180))
    if not 20 <= duration <= 310:
        raise ValueError("duration_seconds must be from 20 through 310")
    receiver_mode = str(config.get("receiver_mode", "usb")).strip().lower()
    if receiver_mode not in {"usb", "nfm"}:
        raise ValueError("receiver_mode must be usb or nfm")
    return {
        "frequency_hz": validate_frequency(config["frequency_hz"]),
        "duration_seconds": duration,
        "receiver_mode": receiver_mode,
        "retain_audio": _boolean(config.get("retain_audio", False), "retain_audio"),
        "retain_iq": _boolean(config.get("retain_iq", False), "retain_iq"),
        "deduplicate": _boolean(config.get("deduplicate", True), "deduplicate"),
    }


def _sstv_watch_config(config: dict) -> dict:
    allowed = {
        "frequency_hz", "receiver_mode", "watch_duration_seconds", "rearm",
        "retain_audio", "deduplicate",
    }
    _keys(config, allowed, {"frequency_hz"})
    duration = float(config.get("watch_duration_seconds", 3600))
    if not 30 <= duration <= 86_400:
        raise ValueError("watch_duration_seconds must be from 30 through 86400")
    receiver_mode = str(config.get("receiver_mode", "nfm")).strip().lower()
    if receiver_mode not in {"usb", "nfm"}:
        raise ValueError("receiver_mode must be usb or nfm")
    return {
        "frequency_hz": validate_frequency(config["frequency_hz"]),
        "receiver_mode": receiver_mode,
        "watch_duration_seconds": duration,
        "rearm": _boolean(config.get("rearm", True), "rearm"),
        "retain_audio": _boolean(config.get("retain_audio", True), "retain_audio"),
        "deduplicate": _boolean(config.get("deduplicate", True), "deduplicate"),
    }


def _station_memory_scan_config(config: dict) -> dict:
    allowed = {
        "memory_ids_or_names", "tag", "mode", "duration_seconds", "max_memories",
        "stop_on_error", "stereo", "deemphasis_us", "decode_rds_data",
        "compare_previous", "snr_change_threshold_db",
    }
    _keys(config, allowed, set())
    memories = config.get("memory_ids_or_names")
    if memories is not None:
        if not isinstance(memories, list) or not 1 <= len(memories) <= 20:
            raise ValueError("memory_ids_or_names must contain 1 through 20 strings")
        memories = [str(value).strip() for value in memories]
        if any(not value for value in memories) or len(set(value.casefold() for value in memories)) != len(memories):
            raise ValueError("memory_ids_or_names must be non-empty and unique")
    duration = float(config.get("duration_seconds", 5))
    maximum = int(config.get("max_memories", 10))
    if not 0.25 <= duration <= 10:
        raise ValueError("duration_seconds must be from 0.25 through 10")
    if not 1 <= maximum <= 20:
        raise ValueError("max_memories must be from 1 through 20")
    if duration * maximum > 120:
        raise ValueError("station-memory preset exceeds the 120-second RF-time limit")
    mode = config.get("mode")
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in {"am", "nfm", "usb", "lsb", "cw", "broadcast_fm"}:
            raise ValueError("mode must be am, nfm, usb, lsb, cw, or broadcast_fm")
    deemphasis = int(config.get("deemphasis_us", 75))
    if deemphasis not in {50, 75}:
        raise ValueError("deemphasis_us must be 50 or 75")
    snr_threshold = float(config.get("snr_change_threshold_db", 6))
    if not 0.5 <= snr_threshold <= 40:
        raise ValueError("snr_change_threshold_db must be from 0.5 through 40")
    return {
        "memory_ids_or_names": memories,
        "tag": str(config.get("tag", "")).strip().lower() or None,
        "mode": mode, "duration_seconds": duration, "max_memories": maximum,
        "stop_on_error": _boolean(config.get("stop_on_error", False), "stop_on_error"),
        "stereo": _boolean(config.get("stereo", True), "stereo"),
        "deemphasis_us": deemphasis,
        "decode_rds_data": _boolean(config.get("decode_rds_data", True), "decode_rds_data"),
        "compare_previous": _boolean(config.get("compare_previous", True), "compare_previous"),
        "snr_change_threshold_db": snr_threshold,
    }


def normalize_preset(
    *, name: str, preset_type: str, description: str, config: dict | None
) -> tuple[str, str, str, dict]:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("name must contain 1 through 64 printable characters")
    preset_type = str(preset_type).strip().lower()
    if preset_type not in PRESET_TYPES:
        raise ValueError(f"preset_type must be one of: {', '.join(PRESET_TYPES)}")
    description = str(description).strip()
    if len(description) > 500:
        raise ValueError("description must contain no more than 500 characters")
    config = _object(config)
    if preset_type == "band_scan":
        normalized = _band_config(config, False)
    elif preset_type in {"band_survey", "activity_monitor"}:
        normalized = _band_config(config, True)
    elif preset_type == "monitor":
        normalized = _monitor_config(config)
    elif preset_type == "watchlist":
        normalized = _watchlist_config(config)
    elif preset_type == "sstv":
        normalized = _sstv_config(config)
    elif preset_type == "station_memory_scan":
        normalized = _station_memory_scan_config(config)
    else:
        normalized = _sstv_watch_config(config)
    return name, preset_type, description, normalized
