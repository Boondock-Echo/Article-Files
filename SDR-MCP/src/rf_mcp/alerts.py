from __future__ import annotations

import re

from .classification import CLASS_LABELS


CONDITION_TYPES = (
    "classification_is",
    "confidence_at_least",
    "peak_above_median_at_least",
    "observation_failed",
    "ambiguous",
)


def normalize_alert_rule(
    *,
    name: str,
    condition_type: str,
    entry_label: str | None,
    classification_label: str | None,
    min_confidence: float | None,
    threshold_db: float | None,
    enabled: bool,
) -> dict:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("alert rule name must contain 1 through 64 printable characters")
    condition_type = str(condition_type).strip().lower()
    if condition_type not in CONDITION_TYPES:
        raise ValueError(f"condition_type must be one of: {', '.join(CONDITION_TYPES)}")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    entry_label = str(entry_label).strip() if entry_label is not None else None
    if entry_label == "":
        entry_label = None
    if entry_label is not None and len(entry_label) > 64:
        raise ValueError("entry_label must be no more than 64 characters")
    classification_label = (
        str(classification_label).strip().lower()
        if classification_label is not None
        else None
    )
    if condition_type == "classification_is":
        if classification_label not in CLASS_LABELS:
            raise ValueError(
                "classification_label must be one of: " + ", ".join(CLASS_LABELS)
            )
    elif classification_label is not None:
        raise ValueError("classification_label is only valid for classification_is")
    if min_confidence is not None:
        min_confidence = float(min_confidence)
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be from 0 through 1")
    if condition_type in ("classification_is", "confidence_at_least"):
        min_confidence = 0.0 if min_confidence is None else min_confidence
    elif min_confidence is not None:
        raise ValueError(
            "min_confidence is only valid for classification_is or confidence_at_least"
        )
    if condition_type == "peak_above_median_at_least":
        if threshold_db is None:
            raise ValueError("threshold_db is required for peak_above_median_at_least")
        threshold_db = float(threshold_db)
        if not 0 <= threshold_db <= 120:
            raise ValueError("threshold_db must be from 0 through 120")
    elif threshold_db is not None:
        raise ValueError("threshold_db is only valid for peak_above_median_at_least")
    return {
        "name": name,
        "condition_type": condition_type,
        "entry_label": entry_label,
        "classification_label": classification_label,
        "min_confidence": min_confidence,
        "threshold_db": threshold_db,
        "enabled": enabled,
    }


def evaluate_rule(rule: dict, observation: dict) -> tuple[bool, str]:
    if rule.get("entry_label") and observation.get("label", "").casefold() != rule[
        "entry_label"
    ].casefold():
        return False, ""
    condition = rule["condition_type"]
    status = observation.get("status")
    if condition == "observation_failed":
        matched = status == "failed"
    elif status != "completed":
        matched = False
    elif condition == "classification_is":
        matched = (
            observation.get("best_label") == rule["classification_label"]
            and float(observation.get("best_confidence") or 0) >= rule["min_confidence"]
        )
    elif condition == "confidence_at_least":
        matched = float(observation.get("best_confidence") or 0) >= rule["min_confidence"]
    elif condition == "peak_above_median_at_least":
        matched = (
            float((observation.get("features") or {}).get("peak_above_median_db") or 0)
            >= rule["threshold_db"]
        )
    else:
        matched = bool(observation.get("ambiguous"))
    if not matched:
        return False, ""
    label = observation.get("label") or f"{observation.get('frequency_hz')} Hz"
    return True, f"Alert rule {rule['name']} matched watchlist entry {label}"


class AlertEvaluator:
    def __init__(self, catalog) -> None:
        self.catalog = catalog

    def evaluate_watchlist(self, schedule_id: str, result: dict) -> list[dict]:
        rules = self.catalog.list_alert_rules(
            schedule_id=schedule_id, enabled=True, limit=200
        )
        events = []
        for rule in rules:
            for observation in result.get("observations", []):
                matched, message = evaluate_rule(rule, observation)
                if matched:
                    event = self.catalog.record_alert_event(
                        rule=rule,
                        job_id=result.get("job_id"),
                        observation=observation,
                        message=message,
                    )
                    self.catalog.enqueue_webhook_deliveries(event)
                    events.append(event)
        return events
