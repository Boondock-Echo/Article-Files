from __future__ import annotations

import re

from .airspyhf import validate_frequency
from .sstv import SSTV_MODES


MODE_NAMES = tuple(SSTV_MODES.values())


def normalize_sstv_alert_rule(
    *, name: str, frequency_hz: int | None, sstv_mode: str | None,
    minimum_quality: float, unique_only: bool, enabled: bool,
) -> dict:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("SSTV alert rule name must contain 1 through 64 printable characters")
    if frequency_hz is not None:
        frequency_hz = validate_frequency(frequency_hz)
    if sstv_mode is not None:
        requested = str(sstv_mode).strip().casefold()
        matches = [mode for mode in MODE_NAMES if mode.casefold() == requested]
        if not matches:
            raise ValueError("sstv_mode must be one of: " + ", ".join(MODE_NAMES))
        sstv_mode = matches[0]
    minimum_quality = float(minimum_quality)
    if not 0 <= minimum_quality <= 1:
        raise ValueError("minimum_quality must be from 0 through 1")
    if not isinstance(unique_only, bool) or not isinstance(enabled, bool):
        raise ValueError("unique_only and enabled must be JSON booleans")
    return {
        "name": name, "frequency_hz": frequency_hz, "sstv_mode": sstv_mode,
        "minimum_quality": minimum_quality, "unique_only": unique_only,
        "enabled": enabled,
    }


def evaluate_sstv_rule(rule: dict, image: dict) -> tuple[bool, str]:
    if rule.get("frequency_hz") is not None:
        if int(image.get("frequency_hz") or 0) != int(rule["frequency_hz"]):
            return False, ""
    if rule.get("sstv_mode") is not None:
        if str(image.get("sstv_mode") or "").casefold() != rule["sstv_mode"].casefold():
            return False, ""
    if float(image.get("quality") or 0) < float(rule["minimum_quality"]):
        return False, ""
    if rule["unique_only"] and bool(image.get("duplicate_of")):
        return False, ""
    mode = image.get("sstv_mode") or "unknown SSTV mode"
    duplicate = " duplicate" if image.get("duplicate_of") else ""
    return True, (
        f"SSTV alert rule {rule['name']} matched{duplicate} {mode} image "
        f"on {image.get('frequency_hz')} Hz"
    )


def _event_image(image: dict) -> dict:
    fields = (
        "image_id", "job_id", "source_watch_id", "source_satellite_watch_id",
        "source_satellite_pass_id", "frequency_hz", "receiver_mode",
        "nominal_frequency_hz", "doppler_correction_mode", "doppler_plan_point_count",
        "sstv_mode", "vis_code", "vis_parity_valid", "width", "height", "quality",
        "captured_at", "duration_seconds", "duplicate_of", "duplicate_hash_distance",
        "image_artifact_id", "image_download_path",
    )
    return {field: image.get(field) for field in fields}


def evaluate_sstv_image(catalog, image: dict) -> list[dict]:
    events = []
    snapshot = _event_image(image)
    for rule in catalog.list_sstv_alert_rules(enabled=True, limit=200):
        matched, message = evaluate_sstv_rule(rule, image)
        if not matched:
            continue
        event = catalog.record_sstv_alert_event(
            rule=rule, image=snapshot, message=message
        )
        deliveries = catalog.enqueue_webhook_deliveries(event)
        event["webhook_delivery_count"] = len(deliveries)
        event["webhook_delivery_ids"] = [item["delivery_id"] for item in deliveries]
        events.append(event)
    return events
