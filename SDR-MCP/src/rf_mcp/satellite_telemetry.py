from __future__ import annotations

import csv
import json
import math
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import PLOT_DIR, SATELLITE_DIR, ensure_data_dirs


NUMERIC_TYPES = {
    "uint8": ("B", 1), "int8": ("b", 1), "uint16": ("H", 2),
    "int16": ("h", 2), "uint32": ("I", 4), "int32": ("i", 4),
    "uint64": ("Q", 8), "int64": ("q", 8), "float32": ("f", 4),
    "float64": ("d", 8),
}
FIELD_TYPES = set(NUMERIC_TYPES) | {"ascii", "hex", "bool"}


def _printable(value: str, label: str, maximum: int) -> str:
    value = str(value).strip()
    if not 1 <= len(value) <= maximum or re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError(f"{label} must contain 1 through {maximum} printable characters")
    return value


def normalize_telemetry_schema(
    *, name: str, description: str = "", satellite_name: str | None = None,
    match: dict | None = None, fields: list[dict], enabled: bool = True,
) -> dict:
    name = _printable(name, "schema name", 64)
    description = str(description).strip()
    if len(description) > 500:
        raise ValueError("description must contain at most 500 characters")
    satellite_name = (_printable(satellite_name, "satellite_name", 64)
                      if satellite_name else None)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    match = dict(match or {})
    allowed_match = {"source", "destination", "pid", "payload_prefix_hex",
                     "payload_prefix_offset", "require_valid_fcs"}
    unknown = set(match) - allowed_match
    if unknown:
        raise ValueError("unknown match keys: " + ", ".join(sorted(unknown)))
    normalized_match = {}
    for key in ("source", "destination"):
        if match.get(key):
            normalized_match[key] = str(match[key]).strip().upper()
    if match.get("pid") is not None:
        pid = int(match["pid"])
        if not 0 <= pid <= 255:
            raise ValueError("match pid must be from 0 through 255")
        normalized_match["pid"] = pid
    prefix = str(match.get("payload_prefix_hex", "")).replace(" ", "").lower()
    if prefix:
        if len(prefix) % 2 or not re.fullmatch(r"[0-9a-f]+", prefix):
            raise ValueError("payload_prefix_hex must contain complete hexadecimal bytes")
        normalized_match["payload_prefix_hex"] = prefix
        prefix_offset = int(match.get("payload_prefix_offset", 0))
        if not 0 <= prefix_offset <= 4096:
            raise ValueError("payload_prefix_offset must be from 0 through 4096")
        normalized_match["payload_prefix_offset"] = prefix_offset
    require_valid = match.get("require_valid_fcs", True)
    if not isinstance(require_valid, bool):
        raise ValueError("require_valid_fcs must be a JSON boolean")
    normalized_match["require_valid_fcs"] = require_valid
    if not isinstance(fields, list) or not 1 <= len(fields) <= 128:
        raise ValueError("fields must contain 1 through 128 objects")
    normalized_fields, names = [], set()
    for item in fields:
        if not isinstance(item, dict):
            raise ValueError("each telemetry field must be an object")
        field_name = str(item.get("name", "")).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", field_name):
            raise ValueError("field name must be lowercase snake_case")
        if field_name in names:
            raise ValueError(f"duplicate telemetry field name: {field_name}")
        names.add(field_name)
        field_type = str(item.get("type", "")).strip().lower()
        if field_type not in FIELD_TYPES:
            raise ValueError("field type must be one of: " + ", ".join(sorted(FIELD_TYPES)))
        offset = int(item.get("offset", 0))
        if not 0 <= offset <= 65535:
            raise ValueError("field offset must be from 0 through 65535")
        length = (NUMERIC_TYPES.get(field_type) or (None, int(item.get("length", 1))))[1]
        if field_type in {"ascii", "hex"} and not 1 <= length <= 4096:
            raise ValueError("ascii/hex field length must be from 1 through 4096")
        byte_order = str(item.get("byte_order", "big")).strip().lower()
        if byte_order not in {"big", "little"}:
            raise ValueError("byte_order must be big or little")
        scale, add = float(item.get("scale", 1.0)), float(item.get("add", 0.0))
        if not math.isfinite(scale) or not math.isfinite(add):
            raise ValueError("field scale and add must be finite numbers")
        bit_offset = int(item.get("bit_offset", 0))
        bit_length = item.get("bit_length")
        bit_length = int(bit_length) if bit_length is not None else None
        if field_type == "bool":
            length, bit_length = 1, 1
        if bit_length is not None and field_type not in {
            "uint8", "int8", "uint16", "int16", "uint32", "int32",
            "uint64", "int64", "bool",
        }:
            raise ValueError("bit ranges are supported only for integer and bool fields")
        if bit_length is not None and not (1 <= bit_length <= length * 8
                                           and 0 <= bit_offset < length * 8
                                           and bit_offset + bit_length <= length * 8):
            raise ValueError("bit range must fit inside the field bytes")
        label = _printable(item.get("label") or field_name, "field label", 80)
        unit = str(item.get("unit", "")).strip() or None
        if unit and len(unit) > 32:
            raise ValueError("field unit must contain at most 32 characters")
        normalized_fields.append({
            "name": field_name, "label": label,
            "type": field_type, "offset": offset, "length": length,
            "byte_order": byte_order, "scale": scale, "add": add,
            "unit": unit,
            "bit_offset": bit_offset, "bit_length": bit_length,
        })
    return {"name": name, "description": description, "satellite_name": satellite_name,
            "match": normalized_match, "fields": normalized_fields, "enabled": enabled}


def schema_matches(schema: dict, observation: dict, frame: dict) -> bool:
    if not schema.get("enabled", True):
        return False
    if schema.get("satellite_name") and schema["satellite_name"].casefold() != str(
        observation.get("satellite_name", "")
    ).casefold():
        return False
    match = schema.get("match", {})
    if match.get("require_valid_fcs", True) and not frame.get("fcs_valid"):
        return False
    for key in ("source", "destination"):
        if match.get(key) and str(frame.get(key, "")).upper() != match[key]:
            return False
    if match.get("pid") is not None and frame.get("pid") != match["pid"]:
        return False
    payload = bytes.fromhex(frame.get("information_hex", ""))
    if match.get("payload_prefix_hex"):
        offset = match.get("payload_prefix_offset", 0)
        prefix = bytes.fromhex(match["payload_prefix_hex"])
        if payload[offset:offset + len(prefix)] != prefix:
            return False
    return True


def decode_payload(schema: dict, payload: bytes) -> list[dict]:
    decoded = []
    for field in schema["fields"]:
        start, stop = field["offset"], field["offset"] + field["length"]
        if stop > len(payload):
            raise ValueError(
                f"payload has {len(payload)} bytes; field {field['name']} needs bytes {start}:{stop}"
            )
        raw = payload[start:stop]
        field_type = field["type"]
        numeric, text = None, None
        if field_type in NUMERIC_TYPES:
            code, _ = NUMERIC_TYPES[field_type]
            prefix = ">" if field["byte_order"] == "big" else "<"
            numeric = struct.unpack(prefix + code, raw)[0]
        elif field_type == "bool":
            numeric = int.from_bytes(raw, field["byte_order"])
        elif field_type == "ascii":
            text = raw.decode("ascii", errors="replace").rstrip("\x00 ")
        else:
            text = raw.hex()
        if numeric is not None:
            if field.get("bit_length") is not None:
                numeric = (int(numeric) >> field.get("bit_offset", 0)) & (
                    (1 << field["bit_length"]) - 1
                )
            numeric = float(numeric) * field.get("scale", 1.0) + field.get("add", 0.0)
            if field_type == "bool":
                text = "true" if numeric else "false"
        decoded.append({"field_name": field["name"], "field_label": field["label"],
                        "numeric_value": numeric, "text_value": text,
                        "raw_hex": raw.hex(), "unit": field.get("unit")})
    return decoded


def decode_observation_telemetry(catalog, observation: dict,
                                 schema_id_or_name: str | None = None) -> dict:
    schemas = ([catalog.get_satellite_telemetry_schema(schema_id_or_name)]
               if schema_id_or_name else
               catalog.list_satellite_telemetry_schemas(enabled=True, limit=500))
    frames = observation.get("details", {}).get("ax25", {}).get("frames", [])
    values, failures, matched_frames = [], [], 0
    for frame_index, frame in enumerate(frames, 1):
        for schema in schemas:
            try:
                if not schema_matches(schema, observation, frame):
                    continue
                matched_frames += 1
                decoded = decode_payload(schema, bytes.fromhex(frame.get("information_hex", "")))
                values.extend({
                    **item, "schema_id": schema["schema_id"],
                    "observation_id": observation["observation_id"],
                    "pass_id": observation.get("pass_id"), "watch_id": observation.get("watch_id"),
                    "satellite_name": observation["satellite_name"],
                    "downlink_id": observation["downlink_id"], "frame_index": frame_index,
                    "captured_at": observation["captured_at"],
                } for item in decoded)
            except Exception as exc:
                failures.append({"schema_id": schema["schema_id"], "frame_index": frame_index,
                                 "error": f"{type(exc).__name__}: {exc}"})
    stored = catalog.add_satellite_telemetry_values(values) if values else []
    from .satellite_telemetry_alerts import evaluate_telemetry_values
    alert_evaluations = evaluate_telemetry_values(catalog, stored) if stored else []
    return {"observation_id": observation["observation_id"],
            "frame_count": len(frames), "matched_frame_count": matched_frames,
            "value_count": len(stored), "values": stored, "failures": failures,
            "telemetry_alert_count": sum(
                bool(item.get("event_emitted")) for item in alert_evaluations
            ), "alert_evaluations": alert_evaluations}


def save_telemetry_plot(values: list[dict], *, title: str) -> str:
    numeric = [item for item in values if item.get("numeric_value") is not None]
    if not numeric:
        raise ValueError("No numeric telemetry values are available to plot")
    import matplotlib.pyplot as plt
    ensure_data_dirs()
    figure, axis = plt.subplots(figsize=(10, 5))
    groups = {}
    for item in reversed(numeric):
        groups.setdefault((item["field_name"], item.get("unit")), []).append(item)
    for (field, unit), items in groups.items():
        axis.plot([datetime.fromisoformat(item["captured_at"]) for item in items],
                  [item["numeric_value"] for item in items], marker="o", label=(
                      f"{field} ({unit})" if unit else field))
    axis.set_title(title); axis.set_xlabel("UTC"); axis.set_ylabel("Telemetry value")
    axis.grid(alpha=0.25); axis.legend(); figure.autofmt_xdate(); figure.tight_layout()
    path = PLOT_DIR / f"telemetry-{uuid4().hex[:12]}.png"
    figure.savefig(path, dpi=150); plt.close(figure)
    return str(path.resolve())


def export_decoded_telemetry(values: list[dict], *, output_format: str) -> str:
    output_format = str(output_format).lower()
    if output_format not in {"jsonl", "csv"}:
        raise ValueError("output_format must be jsonl or csv")
    ensure_data_dirs()
    path = SATELLITE_DIR / f"decoded-telemetry-{uuid4().hex[:12]}.{output_format}"
    if output_format == "jsonl":
        path.write_text("".join(json.dumps(item) + "\n" for item in values), encoding="utf-8")
    else:
        fields = list(values[0]) if values else ["captured_at", "satellite_name", "field_name",
                                                 "numeric_value", "text_value", "unit"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(values)
    return str(path.resolve())
