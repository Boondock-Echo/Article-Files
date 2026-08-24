from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from .config import DATA_DIR, SAMPLE_RATE, TUNING_RANGES_HZ, ensure_data_dirs

BACKENDS = {"airspyhf", "rtl_sdr", "hackrf", "pluto_soapy", "web888"}
ROLES = {"general", "primary_hf", "vhf_uhf_monitor", "wideband_survey", "satellite", "experimental"}
DEFAULT_CAPABILITIES = {
    "airspyhf": {"ranges": TUNING_RANGES_HZ, "max_bandwidth": SAMPLE_RATE, "executable": "airspyhf_info"},
    "rtl_sdr": {"ranges": ((24_000_000, 1_766_000_000),), "max_bandwidth": 2_400_000, "executable": "rtl_test"},
    "hackrf": {"ranges": ((1_000_000, 6_000_000_000),), "max_bandwidth": 20_000_000, "executable": "hackrf_info"},
    "pluto_soapy": {"ranges": ((325_000_000, 3_800_000_000),), "max_bandwidth": 20_000_000, "executable": "SoapySDRUtil"},
    "web888": {"ranges": ((0, 62_000_000),), "max_bandwidth": 1_536_000, "executable": None},
}

_LOCK = threading.RLock()
# Kept as a compatibility/testing hook; durable leases live in SQLite.
_LEASES: dict[str, dict] = {}
LEASE_SECONDS = 30 * 60


def _path() -> Path:
    return DATA_DIR / "sdr-receivers.json"


def _lease_database_path() -> Path:
    return DATA_DIR / "sdr-coordinator.sqlite3"


def _lease_connection() -> sqlite3.Connection:
    ensure_data_dirs()
    connection = sqlite3.connect(_lease_database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS receiver_leases (
            lease_id TEXT PRIMARY KEY,
            receiver_id TEXT NOT NULL UNIQUE,
            owner TEXT NOT NULL,
            purpose TEXT NOT NULL DEFAULT '',
            acquired_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )"""
    )
    return connection


def _active_leases() -> dict[str, dict]:
    now = _now()
    with _lease_connection() as connection:
        connection.execute("DELETE FROM receiver_leases WHERE expires_at<=?", (now,))
        rows = connection.execute("SELECT * FROM receiver_leases").fetchall()
    leases = {row["receiver_id"]: dict(row) for row in rows}
    _LEASES.clear()
    _LEASES.update(leases)
    return leases


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict]:
    path = _path()
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("SDR receiver registry is not a JSON list")
    return value


def _write(items: list[dict]) -> None:
    ensure_data_dirs()
    path = _path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _identifier(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ValueError("receiver_id must contain 1-64 lowercase letters, numbers, hyphens, or underscores")
    return value


def normalize_receiver(*, receiver_id: str, name: str, backend: str, role: str = "general",
                       device_selector: str = "", enabled: bool = True, verified: bool = False,
                       tuning_ranges_hz: list[list[int]] | None = None,
                       max_bandwidth_hz: int | None = None, priority: int = 50,
                       notes: str = "") -> dict:
    receiver_id = _identifier(receiver_id)
    backend = backend.strip().lower()
    role = role.strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    name = name.strip()
    if not name or len(name) > 100:
        raise ValueError("name must contain 1-100 characters")
    defaults = DEFAULT_CAPABILITIES[backend]
    ranges = tuning_ranges_hz or [list(pair) for pair in defaults["ranges"]]
    normalized_ranges = []
    for pair in ranges:
        if len(pair) != 2 or int(pair[0]) < 0 or int(pair[1]) <= int(pair[0]):
            raise ValueError("each tuning range must be [minimum_hz, maximum_hz]")
        normalized_ranges.append([int(pair[0]), int(pair[1])])
    bandwidth = int(max_bandwidth_hz or defaults["max_bandwidth"])
    if bandwidth <= 0:
        raise ValueError("max_bandwidth_hz must be positive")
    priority = int(priority)
    if not 0 <= priority <= 100:
        raise ValueError("priority must be from 0 through 100")
    return {"receiver_id": receiver_id, "name": name, "backend": backend, "role": role,
            "device_selector": device_selector.strip(), "enabled": bool(enabled),
            "verified": bool(verified), "tuning_ranges_hz": normalized_ranges,
            "max_bandwidth_hz": bandwidth, "priority": priority, "notes": notes.strip()[:1000]}


def save_receiver(**values) -> dict:
    item = normalize_receiver(**values)
    with _LOCK:
        items = _load()
        old = next((entry for entry in items if entry["receiver_id"] == item["receiver_id"]), None)
        timestamp = _now()
        item.update({"created_at": old.get("created_at", timestamp) if old else timestamp,
                     "updated_at": timestamp})
        items = [entry for entry in items if entry["receiver_id"] != item["receiver_id"]]
        items.append(item)
        _write(items)
    return item


def list_receivers() -> list[dict]:
    with _LOCK:
        items = _load()
        leases = _active_leases()
    return [dict(item, lease=leases.get(item["receiver_id"]))
            for item in sorted(items, key=lambda x: (-x["priority"], x["receiver_id"]))]


def get_receiver(receiver_id: str) -> dict:
    receiver_id = _identifier(receiver_id)
    for item in list_receivers():
        if item["receiver_id"] == receiver_id:
            return item
    raise KeyError(f"Unknown SDR receiver: {receiver_id}")


def delete_receiver(receiver_id: str, *, confirm_delete: bool = False) -> dict:
    if not confirm_delete:
        raise ValueError("Deleting a receiver requires confirm_delete=true")
    receiver_id = _identifier(receiver_id)
    with _LOCK:
        if receiver_id in _active_leases():
            raise RuntimeError("Cannot delete a receiver while it is leased")
        items = _load()
        removed = next((item for item in items if item["receiver_id"] == receiver_id), None)
        if not removed:
            raise KeyError(f"Unknown SDR receiver: {receiver_id}")
        _write([item for item in items if item["receiver_id"] != receiver_id])
    return {"deleted": True, "receiver_id": receiver_id, "receiver": removed}


def ensure_airspy_default() -> dict:
    try:
        return get_receiver("airspyhf-primary")
    except KeyError:
        return save_receiver(receiver_id="airspyhf-primary", name="Airspy HF+",
                             backend="airspyhf", role="primary_hf", enabled=True,
                             verified=True, priority=100,
                             notes="Default receiver used by the established RF MCP tools.")


def discover_backends(*, probe_hardware: bool = False) -> dict:
    results = []
    for backend, capability in DEFAULT_CAPABILITIES.items():
        executable = capability["executable"]
        resolved = shutil.which(executable) if executable else None
        entry = {"backend": backend, "executable": executable, "installed": bool(resolved),
                 "path": resolved, "probe_requested": bool(probe_hardware),
                 "probe_ok": None, "probe_output": None}
        if probe_hardware and resolved:
            command = {"airspyhf": [resolved], "rtl_sdr": [resolved, "-t"],
                       "hackrf": [resolved], "pluto_soapy": [resolved, "--find"]}[backend]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                output = "\n".join(filter(None, [completed.stdout, completed.stderr])).strip()
                entry.update(probe_ok=completed.returncode == 0, probe_output=output[-4000:])
            except (OSError, subprocess.TimeoutExpired) as exc:
                entry.update(probe_ok=False, probe_output=f"{type(exc).__name__}: {exc}")
        results.append(entry)
    return {"safe_discovery": not probe_hardware, "backends": results,
            "note": "Discovery never creates registry entries or starts an IQ capture."}


def _suggest_receiver_id(backend: str, selector: str, index: int) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "-", selector.lower()).strip("-")
    if not suffix or suffix == str(index):
        suffix = str(index + 1)
    return f"{backend.replace('_', '-')}-{suffix}"[:64].rstrip("-")


def discover_devices() -> dict:
    """Probe supported capture backends and return registration-ready devices.

    This may briefly open receivers but never starts an IQ capture or writes the registry.
    """
    devices: list[dict] = []
    diagnostics: list[dict] = []

    airspy_info = shutil.which("airspyhf_info")
    if airspy_info:
        try:
            completed = subprocess.run(
                [airspy_info], capture_output=True, text=True, timeout=10,
            )
            output = "\n".join(filter(None, [completed.stdout, completed.stderr])).strip()
            serial_match = re.search(r"S/N:\s*(\S+)", output, re.IGNORECASE)
            if completed.returncode == 0:
                selector = serial_match.group(1) if serial_match else ""
                devices.append({
                    "backend": "airspyhf", "device_selector": selector,
                    "display_name": "Airspy HF+" + (f" · {selector}" if selector else ""),
                    "suggested_receiver_id": "airspyhf-primary",
                    "suggested_name": "Airspy HF+", "suggested_role": "primary_hf",
                    "verified": True, "already_registered": False,
                })
            else:
                diagnostics.append({"backend": "airspyhf", "error": output[-1000:]})
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append({"backend": "airspyhf", "error": f"{type(exc).__name__}: {exc}"})

    rtl_test = shutil.which("rtl_test")
    if rtl_test:
        try:
            completed = subprocess.run(
                [rtl_test, "-t"], capture_output=True, text=True, timeout=15,
            )
            output = "\n".join(filter(None, [completed.stdout, completed.stderr])).strip()
            # rtl_test commonly exits nonzero after a successful probe on some releases,
            # so the enumerated "Found N device(s)" block is authoritative here.
            entries = re.findall(
                r"^\s*(\d+):\s+(.+?),\s+SN:\s*([^\s]+)\s*$", output,
                re.MULTILINE | re.IGNORECASE,
            )
            for index_text, model, serial in entries:
                index = int(index_text)
                selector = serial.strip() or index_text
                devices.append({
                    "backend": "rtl_sdr", "device_selector": selector,
                    "display_name": f"{model.strip()} · {selector}",
                    "suggested_receiver_id": _suggest_receiver_id("rtl-sdr", selector, index),
                    "suggested_name": f"RTL-SDR {serial.strip() or index + 1}",
                    "suggested_role": "vhf_uhf_monitor", "verified": True,
                    "already_registered": False,
                })
            if not entries:
                diagnostics.append({
                    "backend": "rtl_sdr",
                    "error": output[-1000:] or f"rtl_test exited with status {completed.returncode}",
                })
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append({"backend": "rtl_sdr", "error": f"{type(exc).__name__}: {exc}"})

    registered = list_receivers()
    for device in devices:
        device["already_registered"] = any(
            item["backend"] == device["backend"]
            and item.get("device_selector", "") == device["device_selector"]
            for item in registered
        )
    return {
        "device_count": len(devices), "devices": devices, "diagnostics": diagnostics,
        "writes_registry": False,
    }


def register_discovered_device(
    *, backend: str, device_selector: str, receiver_id: str, name: str,
    role: str, priority: int = 80,
) -> dict:
    """Verify a currently attached discovery result and save it to the registry."""
    backend = backend.strip().lower()
    selector = device_selector.strip()
    discovered = discover_devices()
    match = next((
        item for item in discovered["devices"]
        if item["backend"] == backend and item["device_selector"] == selector
    ), None)
    if match is None:
        raise ValueError("The selected receiver is no longer attached; scan again")
    defaults = DEFAULT_CAPABILITIES[backend]
    item = save_receiver(
        receiver_id=receiver_id, name=name, backend=backend, role=role,
        device_selector=selector, enabled=True, verified=True,
        tuning_ranges_hz=[list(pair) for pair in defaults["ranges"]],
        max_bandwidth_hz=defaults["max_bandwidth"], priority=priority,
        notes="Added through dashboard receiver setup",
    )
    return {"registered": True, "receiver": item, "discovery": match}


def plan_assignment(*, frequency_hz: int, required_bandwidth_hz: int = 0,
                    preferred_role: str | None = None, require_verified: bool = True) -> dict:
    frequency_hz, required_bandwidth_hz = int(frequency_hz), int(required_bandwidth_hz)
    if frequency_hz < 0 or required_bandwidth_hz < 0:
        raise ValueError("frequency and bandwidth must not be negative")
    if preferred_role is not None and preferred_role not in ROLES:
        raise ValueError(f"preferred_role must be one of {sorted(ROLES)}")
    candidates, rejected = [], []
    for item in list_receivers():
        reasons = []
        if not item["enabled"]: reasons.append("disabled")
        if require_verified and not item["verified"]: reasons.append("not verified")
        if not any(low <= frequency_hz <= high for low, high in item["tuning_ranges_hz"]): reasons.append("frequency outside tuning ranges")
        if required_bandwidth_hz > item["max_bandwidth_hz"]: reasons.append("required bandwidth too wide")
        if item["lease"] is not None: reasons.append("currently leased")
        if reasons:
            rejected.append({"receiver_id": item["receiver_id"], "reasons": reasons})
            continue
        role_match = preferred_role is not None and item["role"] == preferred_role
        score = item["priority"] + (30 if role_match else 0) + (10 if item["verified"] else 0)
        candidates.append({"receiver_id": item["receiver_id"], "name": item["name"],
                           "backend": item["backend"], "role": item["role"],
                           "score": score, "role_match": role_match})
    candidates.sort(key=lambda x: (-x["score"], x["receiver_id"]))
    return {"frequency_hz": frequency_hz, "required_bandwidth_hz": required_bandwidth_hz,
            "preferred_role": preferred_role, "require_verified": require_verified,
            "selected": candidates[0] if candidates else None, "candidates": candidates,
            "rejected": rejected, "dry_run": True}


def acquire_receiver(receiver_id: str, owner: str, purpose: str = "") -> dict:
    receiver_id = _identifier(receiver_id)
    get_receiver(receiver_id)
    owner = owner.strip()
    if not owner:
        raise ValueError("owner is required")
    with _LOCK:
        now = datetime.now(timezone.utc)
        lease = {
            "lease_id": f"lease-{uuid4().hex[:12]}", "receiver_id": receiver_id,
            "owner": owner[:100], "purpose": purpose.strip()[:200],
            "acquired_at": now.isoformat(), "heartbeat_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=LEASE_SECONDS)).isoformat(),
        }
        try:
            with _lease_connection() as connection:
                connection.execute(
                    "DELETE FROM receiver_leases WHERE expires_at<=?", (now.isoformat(),)
                )
                connection.execute(
                    "INSERT INTO receiver_leases VALUES (?,?,?,?,?,?,?)",
                    tuple(lease[key] for key in (
                        "lease_id", "receiver_id", "owner", "purpose", "acquired_at",
                        "heartbeat_at", "expires_at",
                    )),
                )
        except sqlite3.IntegrityError as exc:
            existing = _active_leases().get(receiver_id)
            owner_text = existing["owner"] if existing else "another process"
            raise RuntimeError(
                f"Receiver {receiver_id} is already leased by {owner_text}"
            ) from exc
        _LEASES[receiver_id] = lease
        return dict(lease)


def heartbeat_receiver(lease_id: str) -> dict:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=LEASE_SECONDS)
    with _LOCK, _lease_connection() as connection:
        cursor = connection.execute(
            "UPDATE receiver_leases SET heartbeat_at=?,expires_at=? WHERE lease_id=?",
            (now.isoformat(), expires.isoformat(), lease_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown active lease: {lease_id}")
        row = connection.execute(
            "SELECT * FROM receiver_leases WHERE lease_id=?", (lease_id,)
        ).fetchone()
    return dict(row)


def release_receiver(lease_id: str) -> dict:
    with _LOCK:
        with _lease_connection() as connection:
            row = connection.execute(
                "SELECT * FROM receiver_leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if row is not None:
                connection.execute("DELETE FROM receiver_leases WHERE lease_id=?", (lease_id,))
                _LEASES.pop(row["receiver_id"], None)
                return {"released": True, "lease": dict(row)}
    raise KeyError(f"Unknown active lease: {lease_id}")


def coordinator_status() -> dict:
    receivers = list_receivers()
    return {"receiver_count": len(receivers), "enabled_count": sum(x["enabled"] for x in receivers),
            "verified_count": sum(x["verified"] for x in receivers),
            "active_lease_count": sum(x["lease"] is not None for x in receivers),
            "active_leases": [x["lease"] for x in receivers if x["lease"] is not None],
            "receivers": receivers, "leases_are_process_local": False,
            "lease_store": str(_lease_database_path()),
            "lease_timeout_seconds": LEASE_SECONDS}


@contextmanager
def receiver_lease(receiver_id: str, owner: str, purpose: str = ""):
    lease = acquire_receiver(receiver_id, owner, purpose)
    try:
        yield lease
    finally:
        release_receiver(lease["lease_id"])
