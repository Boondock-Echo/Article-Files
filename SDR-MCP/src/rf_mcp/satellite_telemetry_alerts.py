from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone


CONDITIONS = {"above", "below", "inside", "outside", "absolute_change", "percent_change"}


def normalize_telemetry_alert_rule(
    *, catalog, name: str, schema_id_or_name: str, field_name: str,
    condition_type: str, threshold_low: float | None = None,
    threshold_high: float | None = None, change_threshold: float | None = None,
    cooldown_seconds: int = 3600, enabled: bool = True,
) -> dict:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("telemetry alert rule name must contain 1 through 64 printable characters")
    schema = catalog.get_satellite_telemetry_schema(schema_id_or_name)
    field_name = str(field_name).strip().lower()
    field = next((item for item in schema["fields"] if item["name"] == field_name), None)
    if field is None:
        raise ValueError(f"Telemetry schema has no field named {field_name}")
    if field["type"] in {"ascii", "hex"}:
        raise ValueError("telemetry alert rules require a numeric or boolean field")
    condition_type = str(condition_type).strip().lower()
    if condition_type not in CONDITIONS:
        raise ValueError("condition_type must be one of: " + ", ".join(sorted(CONDITIONS)))
    threshold_low = float(threshold_low) if threshold_low is not None else None
    threshold_high = float(threshold_high) if threshold_high is not None else None
    change_threshold = float(change_threshold) if change_threshold is not None else None
    if any(value is not None and not math.isfinite(value)
           for value in (threshold_low, threshold_high, change_threshold)):
        raise ValueError("telemetry alert thresholds must be finite numbers")
    if condition_type == "above" and threshold_high is None:
        raise ValueError("above requires threshold_high")
    if condition_type == "below" and threshold_low is None:
        raise ValueError("below requires threshold_low")
    if condition_type in {"inside", "outside"}:
        if threshold_low is None or threshold_high is None or threshold_low >= threshold_high:
            raise ValueError("inside/outside require threshold_low < threshold_high")
    if condition_type in {"absolute_change", "percent_change"}:
        if change_threshold is None or change_threshold <= 0:
            raise ValueError("change conditions require change_threshold > 0")
    cooldown_seconds = int(cooldown_seconds)
    if not 0 <= cooldown_seconds <= 2_592_000:
        raise ValueError("cooldown_seconds must be from 0 through 2592000")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    return {"name": name, "schema_id": schema["schema_id"], "field_name": field_name,
            "condition_type": condition_type, "threshold_low": threshold_low,
            "threshold_high": threshold_high, "change_threshold": change_threshold,
            "cooldown_seconds": cooldown_seconds, "enabled": enabled}


def evaluate_telemetry_alert_rule(rule: dict, value: dict,
                                  previous: dict | None = None) -> dict:
    current = value.get("numeric_value")
    if current is None:
        return {"matched": False, "reason": "field is not numeric"}
    current = float(current)
    condition = rule["condition_type"]
    matched, metric = False, current
    if condition == "above":
        matched = current > float(rule["threshold_high"])
    elif condition == "below":
        matched = current < float(rule["threshold_low"])
    elif condition == "inside":
        matched = float(rule["threshold_low"]) <= current <= float(rule["threshold_high"])
    elif condition == "outside":
        matched = current < float(rule["threshold_low"]) or current > float(rule["threshold_high"])
    else:
        prior = (previous or {}).get("numeric_value")
        if prior is None:
            return {"matched": False, "reason": "no previous observation", "metric": None}
        prior = float(prior)
        if condition == "absolute_change":
            metric = abs(current - prior)
        else:
            if prior == 0:
                return {"matched": False, "reason": "previous value is zero", "metric": None}
            metric = abs((current - prior) / prior) * 100.0
        matched = metric >= float(rule["change_threshold"])
    unit = value.get("unit") or ""
    message = (f"Satellite telemetry alert {rule['name']}: {value['field_name']}="
               f"{current:g}{(' ' + unit) if unit else ''} matched {condition}")
    if condition in {"absolute_change", "percent_change"}:
        suffix = "%" if condition == "percent_change" else (f" {unit}" if unit else "")
        message += f" ({metric:g}{suffix})"
    return {"matched": matched, "reason": "matched" if matched else "condition not met",
            "metric": metric, "message": message}


def _cooldown_active(rule: dict, now: datetime) -> bool:
    if not rule.get("last_triggered_at"):
        return False
    triggered = datetime.fromisoformat(rule["last_triggered_at"]).astimezone(timezone.utc)
    return now < triggered + timedelta(seconds=int(rule["cooldown_seconds"]))


def evaluate_telemetry_values(catalog, values: list[dict], *, emit_events: bool = True) -> list[dict]:
    outcomes = []
    for value in values:
        previous = catalog.previous_satellite_telemetry_value(value)
        rules = catalog.list_satellite_telemetry_alert_rules(
            enabled=True, schema_id=value["schema_id"], field_name=value["field_name"], limit=200
        )
        for rule in rules:
            evaluation = evaluate_telemetry_alert_rule(rule, value, previous)
            outcome = {"rule_id": rule["rule_id"], "value_id": value["value_id"], **evaluation}
            if not evaluation["matched"] or not emit_events:
                outcomes.append(outcome)
                continue
            now = datetime.now(timezone.utc)
            if _cooldown_active(rule, now):
                outcome.update({"event_emitted": False, "suppressed_by_cooldown": True})
                outcomes.append(outcome)
                continue
            event = catalog.record_satellite_telemetry_alert_event(
                rule=rule, value=value, previous=previous, message=evaluation["message"]
            )
            deliveries = catalog.enqueue_webhook_deliveries(event)
            event["webhook_delivery_count"] = len(deliveries)
            outcome.update({"event_emitted": True, "suppressed_by_cooldown": False,
                            "event": event})
            outcomes.append(outcome)
    return outcomes
