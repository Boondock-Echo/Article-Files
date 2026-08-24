from __future__ import annotations

import json
import mimetypes
import shutil
import sqlite3
import stat as stat_module
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import DATA_DIR, ensure_data_dirs

CATALOG_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    """Persistent RF job and artifact catalog backed by SQLite."""

    def __init__(self, data_dir: Path = DATA_DIR, *, index_existing: bool = False) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.database_path = self.data_dir / "rf-mcp.sqlite3"
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._closed = False
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if index_existing:
            self.index_existing_artifacts()

    def _connect(self) -> sqlite3.Connection:
        """Return the current thread's connection.

        Connections are deliberately reused: opening SQLite and negotiating WAL for
        every catalog operation is expensive.  ``close`` may run on the application
        thread after workers have stopped, hence ``check_same_thread=False``; normal
        use remains strictly one connection per thread.
        """
        if self._closed:
            raise RuntimeError("Catalog is closed")
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self.database_path, timeout=30, check_same_thread=False
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection
            with self._connections_lock:
                if self._closed:
                    connection.close()
                    raise RuntimeError("Catalog is closed")
                self._connections.add(connection)
        return connection

    def close(self) -> None:
        """Close every connection owned by this catalog instance."""
        with self._connections_lock:
            self._closed = True
            connections = tuple(self._connections)
            self._connections.clear()
        for connection in connections:
            connection.close()
        self._local.__dict__.clear()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            # Persistent/database-wide settings belong here, not in _connect().
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT,
                    result_json_path TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS jobs_type_state_idx ON jobs(job_type, state);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER,
                    created_at TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_created_idx ON artifacts(created_at DESC);
                CREATE INDEX IF NOT EXISTS artifacts_job_idx ON artifacts(job_id);
                CREATE INDEX IF NOT EXISTS artifacts_kind_idx ON artifacts(kind);

                CREATE TABLE IF NOT EXISTS presets (
                    preset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    preset_type TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS presets_type_name_idx
                    ON presets(preset_type, name COLLATE NOCASE);

                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    preset_id TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    next_run_at TEXT NOT NULL,
                    last_attempt_at TEXT,
                    last_run_at TEXT,
                    last_job_id TEXT,
                    last_status TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(preset_id) REFERENCES presets(preset_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS schedules_due_idx
                    ON schedules(enabled, next_run_at);

                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    schedule_id TEXT NOT NULL,
                    entry_label TEXT,
                    condition_type TEXT NOT NULL,
                    classification_label TEXT,
                    min_confidence REAL,
                    threshold_db REAL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(schedule_id) REFERENCES schedules(schedule_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS alert_rules_schedule_idx
                    ON alert_rules(schedule_id, enabled);

                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    rule_id TEXT,
                    rule_name TEXT NOT NULL,
                    schedule_id TEXT,
                    job_id TEXT,
                    observation_label TEXT,
                    frequency_hz INTEGER,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    event_type TEXT NOT NULL DEFAULT 'rf_watchlist',
                    sstv_rule_id TEXT,
                    satellite_watch_id TEXT,
                    satellite_pass_id TEXT,
                    telemetry_rule_id TEXT,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    FOREIGN KEY(rule_id) REFERENCES alert_rules(rule_id) ON DELETE SET NULL,
                    FOREIGN KEY(schedule_id) REFERENCES schedules(schedule_id) ON DELETE SET NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS alert_events_created_idx
                    ON alert_events(created_at DESC);
                CREATE INDEX IF NOT EXISTS alert_events_ack_idx
                    ON alert_events(acknowledged_at, created_at DESC);

                CREATE TABLE IF NOT EXISTS sstv_alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    frequency_hz INTEGER,
                    sstv_mode TEXT,
                    minimum_quality REAL NOT NULL DEFAULT 0,
                    unique_only INTEGER NOT NULL DEFAULT 1,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS webhook_destinations (
                    destination_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    url TEXT NOT NULL,
                    signing_secret TEXT,
                    all_rules INTEGER NOT NULL DEFAULT 1,
                    rule_id TEXT,
                    sstv_rule_id TEXT,
                    satellite_watch_id TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(rule_id) REFERENCES alert_rules(rule_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS webhook_destinations_rule_idx
                    ON webhook_destinations(enabled, all_rules, rule_id);

                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    destination_id TEXT,
                    destination_name TEXT NOT NULL,
                    destination_url TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    http_status INTEGER,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    delivered_at TEXT,
                    FOREIGN KEY(event_id) REFERENCES alert_events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(destination_id) REFERENCES webhook_destinations(destination_id)
                        ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS webhook_deliveries_due_idx
                    ON webhook_deliveries(state, next_attempt_at);

                CREATE TABLE IF NOT EXISTS satellite_watches (
                    watch_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    satellite_name TEXT NOT NULL,
                    norad_id INTEGER NOT NULL,
                    tle_line1 TEXT NOT NULL,
                    tle_line2 TEXT NOT NULL,
                    latitude_deg REAL NOT NULL,
                    longitude_deg REAL NOT NULL,
                    elevation_m REAL NOT NULL DEFAULT 0,
                    frequency_hz INTEGER NOT NULL,
                    receiver_mode TEXT NOT NULL,
                    minimum_elevation_deg REAL NOT NULL DEFAULT 10,
                    lead_seconds INTEGER NOT NULL DEFAULT 60,
                    trail_seconds INTEGER NOT NULL DEFAULT 30,
                    notify_before_seconds INTEGER NOT NULL DEFAULT 600,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    tle_source TEXT NOT NULL DEFAULT 'manual',
                    auto_refresh INTEGER NOT NULL DEFAULT 0,
                    refresh_interval_seconds INTEGER NOT NULL DEFAULT 86400,
                    last_tle_refresh_at TEXT,
                    last_tle_refresh_status TEXT,
                    last_tle_refresh_error TEXT,
                    next_tle_refresh_at TEXT,
                    tle_epoch_at TEXT,
                    doppler_correction_mode TEXT NOT NULL DEFAULT 'off',
                    doppler_step_seconds INTEGER NOT NULL DEFAULT 10,
                    downlinks_json TEXT NOT NULL DEFAULT '[]',
                    downlink_selection_policy TEXT NOT NULL DEFAULT 'priority',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS satellite_watches_enabled_idx
                    ON satellite_watches(enabled, name COLLATE NOCASE);

                CREATE TABLE IF NOT EXISTS satellite_passes (
                    pass_id TEXT PRIMARY KEY,
                    watch_id TEXT,
                    satellite_name TEXT NOT NULL,
                    norad_id INTEGER NOT NULL,
                    aos_at TEXT NOT NULL,
                    tca_at TEXT NOT NULL,
                    los_at TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    stop_at TEXT NOT NULL,
                    maximum_elevation_deg REAL NOT NULL,
                    prediction_json TEXT NOT NULL,
                    notify_at TEXT,
                    prepass_event_id TEXT,
                    outcome_event_id TEXT,
                    doppler_plan_json TEXT NOT NULL DEFAULT '[]',
                    doppler_plot_path TEXT,
                    doppler_artifact_id TEXT,
                    selected_downlink_json TEXT,
                    state TEXT NOT NULL DEFAULT 'planned',
                    job_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(watch_id, aos_at),
                    FOREIGN KEY(watch_id) REFERENCES satellite_watches(watch_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS satellite_passes_due_idx
                    ON satellite_passes(state, start_at);

                CREATE TABLE IF NOT EXISTS satellite_observations (
                    observation_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    pass_id TEXT,
                    watch_id TEXT,
                    satellite_name TEXT NOT NULL,
                    downlink_id TEXT NOT NULL,
                    downlink_label TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    nominal_frequency_hz INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    packet_count INTEGER NOT NULL DEFAULT 0,
                    valid_packet_count INTEGER NOT NULL DEFAULT 0,
                    captured_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    result_json_path TEXT,
                    audio_path TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS satellite_observations_time_idx
                    ON satellite_observations(captured_at DESC);
                CREATE INDEX IF NOT EXISTS satellite_observations_watch_idx
                    ON satellite_observations(watch_id, captured_at DESC);

                CREATE TABLE IF NOT EXISTS satellite_telemetry_schemas (
                    schema_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    satellite_name TEXT,
                    match_json TEXT NOT NULL DEFAULT '{}',
                    fields_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS satellite_telemetry_schemas_enabled_idx
                    ON satellite_telemetry_schemas(enabled, name COLLATE NOCASE);

                CREATE TABLE IF NOT EXISTS satellite_telemetry_values (
                    value_id TEXT PRIMARY KEY,
                    schema_id TEXT NOT NULL,
                    observation_id TEXT NOT NULL,
                    pass_id TEXT,
                    watch_id TEXT,
                    satellite_name TEXT NOT NULL,
                    downlink_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_label TEXT NOT NULL,
                    numeric_value REAL,
                    text_value TEXT,
                    raw_hex TEXT NOT NULL,
                    unit TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(schema_id, observation_id, frame_index, field_name),
                    FOREIGN KEY(schema_id) REFERENCES satellite_telemetry_schemas(schema_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(observation_id) REFERENCES satellite_observations(observation_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS satellite_telemetry_values_time_idx
                    ON satellite_telemetry_values(captured_at DESC);
                CREATE INDEX IF NOT EXISTS satellite_telemetry_values_field_idx
                    ON satellite_telemetry_values(schema_id, field_name, captured_at DESC);

                CREATE TABLE IF NOT EXISTS satellite_telemetry_alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    schema_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    condition_type TEXT NOT NULL,
                    threshold_low REAL,
                    threshold_high REAL,
                    change_threshold REAL,
                    cooldown_seconds INTEGER NOT NULL DEFAULT 3600,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_triggered_at TEXT,
                    last_event_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(schema_id) REFERENCES satellite_telemetry_schemas(schema_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS satellite_telemetry_alert_rules_enabled_idx
                    ON satellite_telemetry_alert_rules(enabled, schema_id, field_name);
                CREATE INDEX IF NOT EXISTS webhook_deliveries_event_idx
                    ON webhook_deliveries(event_id, created_at);

                CREATE TABLE IF NOT EXISTS fm_stations (
                    frequency_hz INTEGER PRIMARY KEY,
                    pi_code TEXT,
                    ps TEXT,
                    pty INTEGER,
                    pty_name TEXT,
                    ptyn TEXT,
                    radiotext TEXT,
                    tp INTEGER,
                    ta INTEGER,
                    music_speech TEXT,
                    alternative_frequencies_json TEXT NOT NULL DEFAULT '[]',
                    stereo_detected INTEGER NOT NULL DEFAULT 0,
                    estimated_snr_db REAL,
                    pilot_to_composite_rms_db REAL,
                    rds_group_count INTEGER NOT NULL DEFAULT 0,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_survey_job_id TEXT NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS fm_stations_pi_idx ON fm_stations(pi_code);
                CREATE INDEX IF NOT EXISTS fm_stations_seen_idx ON fm_stations(last_seen_at DESC);

                CREATE TABLE IF NOT EXISTS weak_signal_spots (
                    spot_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    dial_frequency_hz INTEGER NOT NULL,
                    audio_frequency_hz REAL,
                    rf_frequency_hz REAL,
                    utc_text TEXT,
                    snr_db REAL,
                    time_offset_seconds REAL,
                    drift_hz_per_minute REAL,
                    message TEXT NOT NULL,
                    callsign TEXT,
                    grid TEXT,
                    power_dbm INTEGER,
                    is_cq INTEGER NOT NULL DEFAULT 0,
                    captured_at TEXT NOT NULL,
                    raw_line TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS weak_spots_time_idx
                    ON weak_signal_spots(captured_at DESC);
                CREATE INDEX IF NOT EXISTS weak_spots_call_idx
                    ON weak_signal_spots(callsign, captured_at DESC);
                CREATE INDEX IF NOT EXISTS weak_spots_mode_freq_idx
                    ON weak_signal_spots(mode, dial_frequency_hz, captured_at DESC);

                CREATE TABLE IF NOT EXISTS fldigi_decodes (
                    decode_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    fldigi_modem TEXT NOT NULL,
                    dial_frequency_hz INTEGER NOT NULL,
                    carrier_audio_hz INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    quality REAL,
                    callsigns_json TEXT NOT NULL DEFAULT '[]',
                    grids_json TEXT NOT NULL DEFAULT '[]',
                    captured_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS fldigi_decodes_time_idx
                    ON fldigi_decodes(captured_at DESC);
                CREATE INDEX IF NOT EXISTS fldigi_decodes_mode_idx
                    ON fldigi_decodes(mode, captured_at DESC);

                CREATE TABLE IF NOT EXISTS sstv_images (
                    image_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    frequency_hz INTEGER NOT NULL,
                    receiver_mode TEXT NOT NULL,
                    sstv_mode TEXT,
                    vis_code INTEGER,
                    vis_parity_valid INTEGER,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    quality REAL,
                    image_path TEXT NOT NULL UNIQUE,
                    audio_path TEXT,
                    captured_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    decoder_output TEXT NOT NULL DEFAULT '',
                    image_hash TEXT,
                    duplicate_of TEXT,
                    source_preset_id TEXT,
                    source_schedule_id TEXT,
                    source_watch_id TEXT,
                    source_satellite_pass_id TEXT
                );
                CREATE INDEX IF NOT EXISTS sstv_images_time_idx
                    ON sstv_images(captured_at DESC);
                CREATE INDEX IF NOT EXISTS sstv_images_freq_idx
                    ON sstv_images(frequency_hz, captured_at DESC);
                """
            )
            artifact_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(artifacts)")
            }
            if "mtime_ns" not in artifact_columns:
                connection.execute("ALTER TABLE artifacts ADD COLUMN mtime_ns INTEGER")
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(sstv_images)")
            }
            migrations = {
                "image_hash": "TEXT",
                "duplicate_of": "TEXT",
                "source_preset_id": "TEXT",
                "source_schedule_id": "TEXT",
                "source_watch_id": "TEXT",
                "source_satellite_pass_id": "TEXT",
            }
            for name, sql_type in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE sstv_images ADD COLUMN {name} {sql_type}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sstv_images_hash_idx "
                "ON sstv_images(image_hash)"
            )
            alert_event_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(alert_events)")
            }
            for name, definition in {
                "event_type": "TEXT NOT NULL DEFAULT 'rf_watchlist'",
                "sstv_rule_id": "TEXT",
                "satellite_watch_id": "TEXT",
                "satellite_pass_id": "TEXT",
                "telemetry_rule_id": "TEXT",
            }.items():
                if name not in alert_event_columns:
                    connection.execute(
                        f"ALTER TABLE alert_events ADD COLUMN {name} {definition}"
                    )
            destination_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(webhook_destinations)")
            }
            if "sstv_rule_id" not in destination_columns:
                connection.execute(
                    "ALTER TABLE webhook_destinations ADD COLUMN sstv_rule_id TEXT"
                )
            if "satellite_watch_id" not in destination_columns:
                connection.execute(
                    "ALTER TABLE webhook_destinations ADD COLUMN satellite_watch_id TEXT"
                )
            satellite_watch_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(satellite_watches)")
            }
            for name, definition in {
                "tle_source": "TEXT NOT NULL DEFAULT 'manual'",
                "notify_before_seconds": "INTEGER NOT NULL DEFAULT 600",
                "auto_refresh": "INTEGER NOT NULL DEFAULT 0",
                "refresh_interval_seconds": "INTEGER NOT NULL DEFAULT 86400",
                "last_tle_refresh_at": "TEXT",
                "last_tle_refresh_status": "TEXT",
                "last_tle_refresh_error": "TEXT",
                "next_tle_refresh_at": "TEXT",
                "tle_epoch_at": "TEXT",
                "doppler_correction_mode": "TEXT NOT NULL DEFAULT 'off'",
                "doppler_step_seconds": "INTEGER NOT NULL DEFAULT 10",
                "downlinks_json": "TEXT NOT NULL DEFAULT '[]'",
                "downlink_selection_policy": "TEXT NOT NULL DEFAULT 'priority'",
            }.items():
                if name not in satellite_watch_columns:
                    connection.execute(
                        f"ALTER TABLE satellite_watches ADD COLUMN {name} {definition}"
                    )
            satellite_pass_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(satellite_passes)")
            }
            for name, definition in {
                "notify_at": "TEXT",
                "prepass_event_id": "TEXT",
                "outcome_event_id": "TEXT",
                "doppler_plan_json": "TEXT NOT NULL DEFAULT '[]'",
                "doppler_plot_path": "TEXT",
                "doppler_artifact_id": "TEXT",
                "selected_downlink_json": "TEXT",
            }.items():
                if name not in satellite_pass_columns:
                    connection.execute(
                        f"ALTER TABLE satellite_passes ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS satellite_watches_refresh_idx "
                "ON satellite_watches(auto_refresh, next_tle_refresh_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS satellite_passes_notify_idx "
                "ON satellite_passes(prepass_event_id, notify_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sstv_alert_rules_enabled_idx "
                "ON sstv_alert_rules(enabled, frequency_hz)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS alert_events_type_idx "
                "ON alert_events(event_type, created_at DESC)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) "
                "VALUES (?,?,?)",
                (1, "baseline_v067", utc_now()),
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) "
                "VALUES (?,?,?)",
                (2, "artifact_mtime", utc_now()),
            )
            connection.execute(f"PRAGMA user_version={CATALOG_SCHEMA_VERSION}")

    def schema_status(self) -> dict:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            rows = connection.execute(
                "SELECT version,name,applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
        return {
            "current_version": version,
            "supported_version": CATALOG_SCHEMA_VERSION,
            "up_to_date": version == CATALOG_SCHEMA_VERSION,
            "migrations": [dict(row) for row in rows],
        }

    @staticmethod
    def _satellite_watch_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        result["auto_refresh"] = bool(result["auto_refresh"])
        result["downlinks"] = json.loads(result.pop("downlinks_json"))
        if not result["downlinks"]:
            result["downlinks"] = [{
                "downlink_id": "legacy-sstv", "label": f"{result['satellite_name']} SSTV",
                "frequency_hz": result["frequency_hz"], "mode": "sstv",
                "receiver_mode": result["receiver_mode"], "priority": 1,
                "enabled": True, "retain_audio": True,
            }]
        return result

    def save_satellite_watch(self, *, replace_existing: bool = False, **values) -> dict:
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM satellite_watches WHERE name=? COLLATE NOCASE",
                (values["name"],),
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"Satellite watch name already exists: {existing['name']}; "
                    "set replace_existing=true"
                )
            fields = (
                "satellite_name", "norad_id", "tle_line1", "tle_line2", "latitude_deg",
                "longitude_deg", "elevation_m", "frequency_hz", "receiver_mode",
                "minimum_elevation_deg", "lead_seconds", "trail_seconds",
                "notify_before_seconds", "enabled", "tle_source", "auto_refresh",
                "refresh_interval_seconds", "tle_epoch_at", "next_tle_refresh_at",
                "doppler_correction_mode", "doppler_step_seconds",
                "downlinks_json", "downlink_selection_policy",
            )
            prepared = dict(values)
            prepared["enabled"] = 1 if values["enabled"] else 0
            prepared["auto_refresh"] = 1 if values["auto_refresh"] else 0
            prepared["next_tle_refresh_at"] = now if values["auto_refresh"] else None
            prepared["downlinks_json"] = json.dumps(values["downlinks"])
            stored = [prepared[key] for key in fields]
            if existing is None:
                watch_id = f"satwatch-{uuid4().hex}"
                connection.execute(
                    f"INSERT INTO satellite_watches (watch_id,name,{','.join(fields)},created_at,updated_at) "
                    f"VALUES ({','.join('?' for _ in range(len(fields) + 4))})",
                    (watch_id, values["name"], *stored, now, now),
                )
            else:
                watch_id = existing["watch_id"]
                assignments = ",".join(f"{key}=?" for key in fields)
                connection.execute(
                    f"UPDATE satellite_watches SET name=?,{assignments},updated_at=? WHERE watch_id=?",
                    (values["name"], *stored, now, watch_id),
                )
                connection.execute(
                    "UPDATE satellite_passes SET state='superseded', updated_at=? "
                    "WHERE watch_id=? AND state='planned'",
                    (now, watch_id),
                )
        return self.get_satellite_watch(watch_id)

    def get_satellite_watch(self, watch_id_or_name: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM satellite_watches WHERE watch_id=? OR name=? COLLATE NOCASE",
                (watch_id_or_name, watch_id_or_name),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown satellite watch: {watch_id_or_name}")
        return self._satellite_watch_row(row)

    def list_satellite_watches(self, *, enabled: bool | None = None,
                               limit: int = 100) -> list[dict]:
        where, values = "", []
        if enabled is not None:
            where, values = " WHERE enabled=?", [1 if enabled else 0]
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_watches{where} ORDER BY name COLLATE NOCASE LIMIT ?",
                (*values, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._satellite_watch_row(row) for row in rows]

    def set_satellite_watch_enabled(self, watch_id_or_name: str, enabled: bool) -> dict:
        watch = self.get_satellite_watch(watch_id_or_name)
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE satellite_watches SET enabled=?,updated_at=? WHERE watch_id=?",
                (1 if enabled else 0, now, watch["watch_id"]),
            )
            if not enabled:
                connection.execute(
                    "UPDATE satellite_passes SET state='superseded',updated_at=? "
                    "WHERE watch_id=? AND state='planned'", (now, watch["watch_id"]),
                )
        return self.get_satellite_watch(watch["watch_id"])

    def delete_satellite_watch(self, watch_id_or_name: str) -> dict:
        watch = self.get_satellite_watch(watch_id_or_name)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE satellite_passes SET state='superseded',updated_at=? "
                "WHERE watch_id=? AND state='planned'", (utc_now(), watch["watch_id"]),
            )
            connection.execute(
                "UPDATE webhook_destinations SET satellite_watch_id=NULL "
                "WHERE satellite_watch_id=?", (watch["watch_id"],),
            )
            connection.execute("DELETE FROM satellite_watches WHERE watch_id=?", (watch["watch_id"],))
        return watch

    def due_satellite_tle_refreshes(self, now: str, *, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM satellite_watches WHERE enabled=1 AND auto_refresh=1 "
                "AND tle_source='celestrak' AND (next_tle_refresh_at IS NULL OR "
                "next_tle_refresh_at<=?) ORDER BY next_tle_refresh_at LIMIT ?",
                (now, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._satellite_watch_row(row) for row in rows]

    def record_satellite_tle_refresh(
        self, watch_id: str, *, status: str, tle_line1: str | None = None,
        tle_line2: str | None = None, tle_epoch_at: str | None = None,
        error: str | None = None, now: datetime | None = None,
    ) -> dict:
        watch = self.get_satellite_watch(watch_id)
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        retry_seconds = watch["refresh_interval_seconds"] if status == "succeeded" else 3600
        next_at = (instant + timedelta(seconds=retry_seconds)).isoformat()
        with self._lock, self._connect() as connection:
            if status == "succeeded":
                if not tle_line1 or not tle_line2 or not tle_epoch_at:
                    raise ValueError("Successful TLE refresh requires validated element lines")
                connection.execute(
                    "UPDATE satellite_watches SET tle_line1=?,tle_line2=?,tle_epoch_at=?,"
                    "last_tle_refresh_at=?,last_tle_refresh_status='succeeded',"
                    "last_tle_refresh_error=NULL,next_tle_refresh_at=?,updated_at=? "
                    "WHERE watch_id=?",
                    (tle_line1, tle_line2, tle_epoch_at, instant.isoformat(), next_at,
                     instant.isoformat(), watch_id),
                )
                connection.execute(
                    "UPDATE satellite_passes SET state='superseded',updated_at=? "
                    "WHERE watch_id=? AND state='planned'",
                    (instant.isoformat(), watch_id),
                )
            else:
                connection.execute(
                    "UPDATE satellite_watches SET last_tle_refresh_at=?,"
                    "last_tle_refresh_status='failed',last_tle_refresh_error=?,"
                    "next_tle_refresh_at=?,updated_at=? WHERE watch_id=?",
                    (instant.isoformat(), error, next_at, instant.isoformat(), watch_id),
                )
        return self.get_satellite_watch(watch_id)

    @staticmethod
    def _satellite_pass_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["prediction"] = json.loads(result.pop("prediction_json"))
        result["doppler_plan"] = json.loads(result.pop("doppler_plan_json"))
        raw_downlink = result.pop("selected_downlink_json")
        result["selected_downlink"] = json.loads(raw_downlink) if raw_downlink else None
        return result

    def save_satellite_pass(self, watch: dict, prediction: dict) -> dict:
        aos = prediction["aos"]["at"]
        lead, trail = timedelta(seconds=watch["lead_seconds"]), timedelta(seconds=watch["trail_seconds"])
        start_at = (datetime.fromisoformat(aos) - lead).isoformat()
        stop_at = (datetime.fromisoformat(prediction["los"]["at"]) + trail).isoformat()
        notify_at = (
            datetime.fromisoformat(aos) - timedelta(seconds=watch["notify_before_seconds"])
        ).isoformat()
        doppler_plan = prediction.get("doppler_track", [])
        selected_downlink = prediction.get("selected_downlink")
        stored_prediction = {key: value for key, value in prediction.items()
                             if key not in {"doppler_track", "selected_downlink"}}
        now = utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT pass_id,state,doppler_plan_json,selected_downlink_json "
                "FROM satellite_passes "
                "WHERE watch_id=? AND aos_at=?",
                (watch["watch_id"], aos),
            ).fetchone()
            if row is None:
                pass_id = f"satpass-{uuid4().hex}"
                connection.execute(
                    "INSERT INTO satellite_passes (pass_id,watch_id,satellite_name,norad_id,"
                    "aos_at,tca_at,los_at,start_at,stop_at,maximum_elevation_deg,prediction_json,"
                    "notify_at,doppler_plan_json,selected_downlink_json,state,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'planned',?,?)",
                    (pass_id, watch["watch_id"], watch["satellite_name"], watch["norad_id"],
                     aos, prediction["tca"]["at"], prediction["los"]["at"], start_at, stop_at,
                     prediction["maximum_elevation_deg"], json.dumps(stored_prediction), notify_at,
                     json.dumps(doppler_plan), json.dumps(selected_downlink),
                     now, now),
                )
            else:
                pass_id = row["pass_id"]
                if row["state"] == "superseded":
                    connection.execute(
                        "UPDATE satellite_passes SET tca_at=?,los_at=?,start_at=?,stop_at=?,"
                        "maximum_elevation_deg=?,prediction_json=?,notify_at=?,"
                        "doppler_plan_json=?,selected_downlink_json=?,state='planned',"
                        "job_id=NULL,error=NULL,prepass_event_id=NULL,outcome_event_id=NULL,"
                        "updated_at=? WHERE pass_id=?",
                        (prediction["tca"]["at"], prediction["los"]["at"], start_at, stop_at,
                         prediction["maximum_elevation_deg"], json.dumps(stored_prediction),
                         notify_at, json.dumps(doppler_plan), json.dumps(selected_downlink),
                         now, pass_id),
                    )
                elif doppler_plan and not json.loads(row["doppler_plan_json"] or "[]"):
                    connection.execute(
                        "UPDATE satellite_passes SET doppler_plan_json=?,"
                        "selected_downlink_json=COALESCE(selected_downlink_json,?),updated_at=? "
                        "WHERE pass_id=?", (json.dumps(doppler_plan),
                        json.dumps(selected_downlink), now, pass_id),
                    )
                elif selected_downlink and not row["selected_downlink_json"]:
                    connection.execute(
                        "UPDATE satellite_passes SET selected_downlink_json=?,updated_at=? "
                        "WHERE pass_id=?", (json.dumps(selected_downlink), now, pass_id),
                    )
        return self.get_satellite_pass(pass_id)

    def get_satellite_pass(self, pass_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM satellite_passes WHERE pass_id=?", (pass_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown satellite pass_id: {pass_id}")
        return self._satellite_pass_row(row)

    def set_satellite_doppler_plot(
        self, pass_id: str, *, path: str, artifact_id: str,
    ) -> dict:
        self.get_satellite_pass(pass_id)
        safe_path = str(self._safe_path(path))
        with self._connect() as connection:
            connection.execute(
                "UPDATE satellite_passes SET doppler_plot_path=?,doppler_artifact_id=?,"
                "updated_at=? WHERE pass_id=?",
                (safe_path, artifact_id, utc_now(), pass_id),
            )
        return self.get_satellite_pass(pass_id)

    def list_satellite_passes(self, *, watch_id: str | None = None,
                              state: str | None = None, limit: int = 100,
                              newest_first: bool = False) -> list[dict]:
        clauses, values = [], []
        if watch_id:
            clauses.append("watch_id=?"); values.append(watch_id)
        if state:
            clauses.append("state=?"); values.append(state)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_passes{where} ORDER BY start_at "
                f"{'DESC' if newest_first else 'ASC'} LIMIT ?",
                (*values, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._satellite_pass_row(row) for row in rows]

    def due_satellite_passes(self, now: str, *, limit: int = 10) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM satellite_passes WHERE state='planned' AND start_at<=? "
                "ORDER BY start_at LIMIT ?", (now, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._satellite_pass_row(row) for row in rows]

    def due_satellite_pass_notifications(self, now: str, *, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM satellite_passes WHERE state='planned' "
                "AND prepass_event_id IS NULL AND notify_at IS NOT NULL AND notify_at<=? "
                "ORDER BY notify_at LIMIT ?",
                (now, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._satellite_pass_row(row) for row in rows]

    def record_satellite_pass(self, pass_id: str, *, state: str,
                              job_id: str | None = None, error: str | None = None) -> dict:
        self.get_satellite_pass(pass_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE satellite_passes SET state=?,job_id=?,error=?,updated_at=? WHERE pass_id=?",
                (state, job_id, error, utc_now(), pass_id),
            )
        return self.get_satellite_pass(pass_id)

    @staticmethod
    def _satellite_observation_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json") or "{}")
        return result

    def add_satellite_observation(self, result: dict) -> dict:
        observation_id = result.get("observation_id") or f"satobs-{uuid4().hex}"
        result_path = (str(self._safe_path(result["result_json_path"], must_exist=False))
                       if result.get("result_json_path") else None)
        audio_path = (str(self._safe_path(result["audio_path"]))
                      if result.get("audio_path") else None)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO satellite_observations (observation_id,job_id,pass_id,watch_id,"
                "satellite_name,downlink_id,downlink_label,mode,nominal_frequency_hz,outcome,"
                "packet_count,valid_packet_count,captured_at,duration_seconds,result_json_path,"
                "audio_path,details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (observation_id, result["job_id"], result.get("pass_id"),
                 result.get("watch_id"), result["satellite_name"], result["downlink_id"],
                 result["downlink_label"], result["mode"],
                 int(result["nominal_frequency_hz"]), result["outcome"],
                 int(result.get("packet_count", 0)), int(result.get("valid_packet_count", 0)),
                 result["captured_at"], float(result["duration_seconds"]), result_path,
                 audio_path, json.dumps(result.get("details", {}))),
            )
        return self.get_satellite_observation(observation_id)

    def get_satellite_observation(self, observation_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM satellite_observations WHERE observation_id=?",
                (observation_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown satellite observation_id: {observation_id}")
        return self._satellite_observation_row(row)

    def list_satellite_observations(
        self, *, watch_id: str | None = None, mode: str | None = None,
        outcome: str | None = None, pass_id: str | None = None, limit: int = 100,
    ) -> list[dict]:
        clauses, values = [], []
        for column, value in (("watch_id", watch_id), ("mode", mode),
                              ("outcome", outcome), ("pass_id", pass_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_observations{where} "
                "ORDER BY captured_at DESC LIMIT ?", (*values, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._satellite_observation_row(row) for row in rows]

    def satellite_activity_summary(self, *, watch_id: str | None = None) -> dict:
        where, values = (" WHERE watch_id=?", (watch_id,)) if watch_id else ("", ())
        with self._connect() as connection:
            totals = connection.execute(
                "SELECT COUNT(*) observations,COALESCE(SUM(packet_count),0) packet_count,"
                "COALESCE(SUM(valid_packet_count),0) valid_packet_count,MIN(captured_at) first_at,"
                f"MAX(captured_at) last_at FROM satellite_observations{where}", values,
            ).fetchone()
            modes = connection.execute(
                "SELECT mode,COUNT(*) count,COALESCE(SUM(packet_count),0) packet_count "
                f"FROM satellite_observations{where} GROUP BY mode ORDER BY count DESC", values,
            ).fetchall()
            downlinks = connection.execute(
                "SELECT satellite_name,downlink_id,downlink_label,COUNT(*) count "
                f"FROM satellite_observations{where} GROUP BY satellite_name,downlink_id,"
                "downlink_label ORDER BY count DESC", values,
            ).fetchall()
        return {**dict(totals), "watch_id": watch_id,
                "by_mode": [dict(row) for row in modes],
                "by_downlink": [dict(row) for row in downlinks]}

    @staticmethod
    def _satellite_telemetry_schema_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["match"] = json.loads(result.pop("match_json") or "{}")
        result["fields"] = json.loads(result.pop("fields_json") or "[]")
        result["enabled"] = bool(result["enabled"])
        return result

    def save_satellite_telemetry_schema(self, *, replace_existing: bool = False,
                                        **values) -> dict:
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT schema_id FROM satellite_telemetry_schemas "
                "WHERE name=? COLLATE NOCASE", (values["name"],),
            ).fetchone()
            if existing and not replace_existing:
                raise ValueError(f"Telemetry schema name already exists: {values['name']}")
            if existing:
                schema_id = existing["schema_id"]
                connection.execute(
                    "DELETE FROM satellite_telemetry_values WHERE schema_id=?", (schema_id,)
                )
                connection.execute(
                    "UPDATE satellite_telemetry_alert_rules SET enabled=0,updated_at=? "
                    "WHERE schema_id=?", (now, schema_id)
                )
                connection.execute(
                    "UPDATE satellite_telemetry_schemas SET name=?,description=?,"
                    "satellite_name=?,match_json=?,fields_json=?,enabled=?,updated_at=? "
                    "WHERE schema_id=?",
                    (values["name"], values["description"], values.get("satellite_name"),
                     json.dumps(values["match"]), json.dumps(values["fields"]),
                     1 if values["enabled"] else 0, now, schema_id),
                )
            else:
                schema_id = f"telschema-{uuid4().hex}"
                connection.execute(
                    "INSERT INTO satellite_telemetry_schemas (schema_id,name,description,"
                    "satellite_name,match_json,fields_json,enabled,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (schema_id, values["name"], values["description"],
                     values.get("satellite_name"), json.dumps(values["match"]),
                     json.dumps(values["fields"]), 1 if values["enabled"] else 0, now, now),
                )
        return self.get_satellite_telemetry_schema(schema_id)

    def get_satellite_telemetry_schema(self, schema_id_or_name: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM satellite_telemetry_schemas WHERE schema_id=? "
                "OR name=? COLLATE NOCASE", (schema_id_or_name, schema_id_or_name),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown satellite telemetry schema: {schema_id_or_name}")
        return self._satellite_telemetry_schema_row(row)

    def list_satellite_telemetry_schemas(self, *, enabled: bool | None = None,
                                         limit: int = 100) -> list[dict]:
        where, values = (" WHERE enabled=?", (1 if enabled else 0,)) if enabled is not None else ("", ())
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_telemetry_schemas{where} "
                "ORDER BY name COLLATE NOCASE LIMIT ?", (*values, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._satellite_telemetry_schema_row(row) for row in rows]

    def delete_satellite_telemetry_schema(self, schema_id_or_name: str) -> dict:
        schema = self.get_satellite_telemetry_schema(schema_id_or_name)
        with self._connect() as connection:
            connection.execute("DELETE FROM satellite_telemetry_schemas WHERE schema_id=?",
                               (schema["schema_id"],))
        return schema

    def add_satellite_telemetry_values(self, values: list[dict]) -> list[dict]:
        inserted = []
        now = utc_now()
        with self._connect() as connection:
            for item in values:
                value_id = item.get("value_id") or f"telvalue-{uuid4().hex}"
                connection.execute(
                    "INSERT OR REPLACE INTO satellite_telemetry_values (value_id,schema_id,"
                    "observation_id,pass_id,watch_id,satellite_name,downlink_id,frame_index,"
                    "captured_at,field_name,field_label,numeric_value,text_value,raw_hex,unit,"
                    "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (value_id, item["schema_id"], item["observation_id"], item.get("pass_id"),
                     item.get("watch_id"), item["satellite_name"], item["downlink_id"],
                     int(item["frame_index"]), item["captured_at"], item["field_name"],
                     item["field_label"], item.get("numeric_value"), item.get("text_value"),
                     item["raw_hex"], item.get("unit"), now),
                )
                inserted.append({**item, "value_id": value_id, "created_at": now})
        return inserted

    def list_satellite_telemetry_values(
        self, *, schema_id: str | None = None, field_name: str | None = None,
        watch_id: str | None = None, pass_id: str | None = None, limit: int = 1000,
    ) -> list[dict]:
        clauses, values = [], []
        for column, value in (("schema_id", schema_id), ("field_name", field_name),
                              ("watch_id", watch_id), ("pass_id", pass_id)):
            if value is not None:
                clauses.append(f"{column}=?"); values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_telemetry_values{where} "
                "ORDER BY captured_at DESC LIMIT ?", (*values, max(1, min(int(limit), 5000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def previous_satellite_telemetry_value(self, value: dict) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM satellite_telemetry_values WHERE schema_id=? AND field_name=? "
                "AND observation_id<>? AND captured_at<=? ORDER BY captured_at DESC LIMIT 1",
                (value["schema_id"], value["field_name"], value["observation_id"],
                 value["captured_at"]),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _satellite_telemetry_alert_rule_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        return result

    def save_satellite_telemetry_alert_rule(self, *, replace_existing: bool = False,
                                            **values) -> dict:
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT rule_id FROM satellite_telemetry_alert_rules "
                "WHERE name=? COLLATE NOCASE", (values["name"],),
            ).fetchone()
            stored = (values["name"], values["schema_id"], values["field_name"],
                      values["condition_type"], values.get("threshold_low"),
                      values.get("threshold_high"), values.get("change_threshold"),
                      values["cooldown_seconds"], 1 if values["enabled"] else 0)
            if existing and not replace_existing:
                raise ValueError(f"Telemetry alert rule name already exists: {values['name']}")
            if existing:
                rule_id = existing["rule_id"]
                connection.execute(
                    "UPDATE satellite_telemetry_alert_rules SET name=?,schema_id=?,"
                    "field_name=?,condition_type=?,threshold_low=?,threshold_high=?,"
                    "change_threshold=?,cooldown_seconds=?,enabled=?,last_triggered_at=NULL,"
                    "last_event_id=NULL,updated_at=? WHERE rule_id=?",
                    (*stored, now, rule_id),
                )
            else:
                rule_id = f"telrule-{uuid4().hex}"
                connection.execute(
                    "INSERT INTO satellite_telemetry_alert_rules (rule_id,name,schema_id,"
                    "field_name,condition_type,threshold_low,threshold_high,change_threshold,"
                    "cooldown_seconds,enabled,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (rule_id, *stored, now, now),
                )
        return self.get_satellite_telemetry_alert_rule(rule_id)

    def get_satellite_telemetry_alert_rule(self, rule_id_or_name: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM satellite_telemetry_alert_rules WHERE rule_id=? "
                "OR name=? COLLATE NOCASE", (rule_id_or_name, rule_id_or_name),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown satellite telemetry alert rule: {rule_id_or_name}")
        return self._satellite_telemetry_alert_rule_row(row)

    def list_satellite_telemetry_alert_rules(
        self, *, enabled: bool | None = None, schema_id: str | None = None,
        field_name: str | None = None, limit: int = 200,
    ) -> list[dict]:
        clauses, values = [], []
        if enabled is not None:
            clauses.append("enabled=?"); values.append(1 if enabled else 0)
        for column, value in (("schema_id", schema_id), ("field_name", field_name)):
            if value is not None:
                clauses.append(f"{column}=?"); values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM satellite_telemetry_alert_rules{where} "
                "ORDER BY name COLLATE NOCASE LIMIT ?", (*values, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._satellite_telemetry_alert_rule_row(row) for row in rows]

    def delete_satellite_telemetry_alert_rule(self, rule_id_or_name: str) -> dict:
        rule = self.get_satellite_telemetry_alert_rule(rule_id_or_name)
        with self._connect() as connection:
            connection.execute("DELETE FROM satellite_telemetry_alert_rules WHERE rule_id=?",
                               (rule["rule_id"],))
        return rule

    def record_satellite_telemetry_alert_event(self, *, rule: dict, value: dict,
                                               previous: dict | None, message: str) -> dict:
        event_id, now = f"alert-{uuid4().hex}", utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO alert_events (event_id,rule_id,rule_name,schedule_id,job_id,"
                "observation_label,frequency_hz,message,details_json,event_type,"
                "telemetry_rule_id,satellite_watch_id,satellite_pass_id,created_at,"
                "acknowledged_at) VALUES (?,NULL,?,NULL,NULL,?,NULL,?,?,?, ?,?,?,?,NULL)",
                (event_id, rule["name"], value["field_name"], message,
                 json.dumps({"schema": "rf-mcp.satellite-telemetry-alert.v1",
                             "rule": rule, "value": value, "previous_value": previous}),
                 "satellite_telemetry", rule["rule_id"], value.get("watch_id"),
                 value.get("pass_id"), now),
            )
            connection.execute(
                "UPDATE satellite_telemetry_alert_rules SET last_triggered_at=?,"
                "last_event_id=?,updated_at=? WHERE rule_id=?",
                (now, event_id, now, rule["rule_id"]),
            )
        return self.get_alert_event(event_id)

    def record_satellite_pass_event(self, pass_id: str, *, event_kind: str) -> dict:
        if event_kind not in {"prepass", "outcome"}:
            raise ValueError("event_kind must be prepass or outcome")
        item = self.get_satellite_pass(pass_id)
        column = "prepass_event_id" if event_kind == "prepass" else "outcome_event_id"
        existing_id = item.get(column)
        if existing_id:
            return self.get_alert_event(existing_id)
        watch = self.get_satellite_watch(item["watch_id"]) if item.get("watch_id") else None
        event_id = f"alert-{uuid4().hex}"
        if event_kind == "prepass":
            message = (
                f"{item['satellite_name']} pass begins at {item['aos_at']} UTC; "
                f"maximum elevation {item['maximum_elevation_deg']:.1f} degrees"
            )
        else:
            message = (
                f"{item['satellite_name']} pass outcome: {item['state']}"
                + (f" ({item['error']})" if item.get("error") else "")
            )
        details = {
            "schema": "rf-mcp.satellite-pass.v1",
            "event_kind": event_kind,
            "watch": watch,
            "pass": item,
        }
        now = utc_now()
        with self._lock, self._connect() as connection:
            current = connection.execute(
                f"SELECT {column} FROM satellite_passes WHERE pass_id=?", (pass_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"Unknown satellite pass_id: {pass_id}")
            if current[column]:
                return self.get_alert_event(current[column])
            connection.execute(
                "INSERT INTO alert_events (event_id,rule_id,rule_name,schedule_id,job_id,"
                "observation_label,frequency_hz,message,details_json,event_type,sstv_rule_id,"
                "satellite_watch_id,satellite_pass_id,created_at,acknowledged_at) "
                "VALUES (?,NULL,?,NULL,?,?,?,?,?,'satellite_pass',NULL,?,?,?,NULL)",
                (event_id, f"Satellite pass: {item['satellite_name']}", item.get("job_id"),
                 event_kind, (watch or {}).get("frequency_hz"), message, json.dumps(details),
                 item.get("watch_id"), pass_id, now),
            )
            connection.execute(
                f"UPDATE satellite_passes SET {column}=?,updated_at=? WHERE pass_id=?",
                (event_id, now, pass_id),
            )
        return self.get_alert_event(event_id)

    @staticmethod
    def _sstv_image_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        if result["vis_parity_valid"] is not None:
            result["vis_parity_valid"] = bool(result["vis_parity_valid"])
        return result

    def add_sstv_image(self, result: dict) -> dict:
        image_id = result.get("image_id") or f"sstv-{uuid4().hex}"
        image_path = str(self._safe_path(result["image_path"]))
        audio_path = str(self._safe_path(result["audio_path"])) if result.get("audio_path") else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sstv_images (
                    image_id, job_id, frequency_hz, receiver_mode, sstv_mode,
                    vis_code, vis_parity_valid, width, height, quality,
                    image_path, audio_path, captured_at, duration_seconds, decoder_output,
                    image_hash, duplicate_of, source_preset_id, source_schedule_id,
                    source_watch_id, source_satellite_pass_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (image_id, result["job_id"], int(result["frequency_hz"]),
                 result["receiver_mode"], result.get("sstv_mode"), result.get("vis_code"),
                 None if result.get("vis_parity_valid") is None else
                 (1 if result["vis_parity_valid"] else 0), int(result["width"]),
                 int(result["height"]), result.get("quality"), image_path, audio_path,
                 result["captured_at"], float(result["duration_seconds"]),
                 result.get("decoder_output", ""), result.get("image_hash"),
                 result.get("duplicate_of"), result.get("source_preset_id"),
                 result.get("source_schedule_id"), result.get("source_watch_id"),
                 result.get("source_satellite_pass_id")),
            )
        return self.get_sstv_image(image_id)

    def get_sstv_image(self, image_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sstv_images WHERE image_id=?", (image_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown SSTV image_id: {image_id}")
        result = self._sstv_image_row(row)
        self._safe_path(result["image_path"])
        if result.get("audio_path"):
            self._safe_path(result["audio_path"])
        return result

    def list_sstv_images(
        self, *, frequency_hz: int | None = None, sstv_mode: str | None = None,
        include_duplicates: bool = True, source_satellite_pass_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses, values = [], []
        if frequency_hz is not None:
            clauses.append("frequency_hz=?")
            values.append(int(frequency_hz))
        if sstv_mode:
            clauses.append("sstv_mode=? COLLATE NOCASE")
            values.append(sstv_mode)
        if not include_duplicates:
            clauses.append("duplicate_of IS NULL")
        if source_satellite_pass_id is not None:
            clauses.append("source_satellite_pass_id=?")
            values.append(source_satellite_pass_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM sstv_images{where} "  # noqa: S608
                "ORDER BY captured_at DESC LIMIT ?", (*values, limit),
            ).fetchall()
        return [self._sstv_image_row(row) for row in rows]

    def find_sstv_duplicate(
        self, image_hash: str, *, frequency_hz: int, max_distance: int = 12,
        limit: int = 200,
    ) -> dict | None:
        if not image_hash:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sstv_images WHERE frequency_hz=? AND image_hash IS NOT NULL "
                "ORDER BY captured_at DESC LIMIT ?",
                (int(frequency_hz), max(1, min(int(limit), 500))),
            ).fetchall()
        target = int(image_hash, 16)
        for row in rows:
            candidate = self._sstv_image_row(row)
            distance = (target ^ int(candidate["image_hash"], 16)).bit_count()
            if distance <= max_distance:
                candidate["hash_distance"] = distance
                return candidate
        return None

    def sstv_activity_summary(self, *, since: str | None = None) -> dict:
        where = " WHERE captured_at>=?" if since else ""
        values = (since,) if since else ()
        with self._connect() as connection:
            totals = connection.execute(
                "SELECT COUNT(*) AS images, "
                "SUM(CASE WHEN duplicate_of IS NULL THEN 1 ELSE 0 END) AS unique_images, "
                "SUM(CASE WHEN duplicate_of IS NOT NULL THEN 1 ELSE 0 END) AS duplicates, "
                "MIN(captured_at) AS first_image_at, MAX(captured_at) AS last_image_at "
                f"FROM sstv_images{where}", values,
            ).fetchone()
            modes = connection.execute(
                "SELECT COALESCE(sstv_mode, 'Unknown') AS mode, COUNT(*) AS count "
                f"FROM sstv_images{where} GROUP BY COALESCE(sstv_mode, 'Unknown') "
                "ORDER BY count DESC", values,
            ).fetchall()
            frequencies = connection.execute(
                "SELECT frequency_hz, COUNT(*) AS count "
                f"FROM sstv_images{where} GROUP BY frequency_hz ORDER BY count DESC", values,
            ).fetchall()
        return {
            **dict(totals),
            "unique_images": int(totals["unique_images"] or 0),
            "duplicates": int(totals["duplicates"] or 0),
            "by_mode": [dict(row) for row in modes],
            "by_frequency": [dict(row) for row in frequencies],
            "since": since,
        }

    @staticmethod
    def _fldigi_decode_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["callsigns"] = json.loads(result.pop("callsigns_json") or "[]")
        result["grids"] = json.loads(result.pop("grids_json") or "[]")
        return result

    def add_fldigi_decode(self, result: dict) -> dict:
        decode_id = result.get("decode_id") or f"fldigi-{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fldigi_decodes (
                    decode_id, job_id, mode, fldigi_modem, dial_frequency_hz,
                    carrier_audio_hz, text, quality, callsigns_json, grids_json,
                    captured_at, duration_seconds
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (decode_id, result["job_id"], result["mode"], result["fldigi_modem"],
                 int(result["dial_frequency_hz"]), int(result["carrier_audio_hz"]),
                 result.get("text", ""), result.get("quality"),
                 json.dumps(result.get("callsigns", [])), json.dumps(result.get("grids", [])),
                 result["captured_at"], float(result["duration_seconds"])),
            )
        return {**result, "decode_id": decode_id}

    def list_fldigi_decodes(
        self, *, mode: str | None = None, dial_frequency_hz: int | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses, values = [], []
        if mode:
            clauses.append("mode=?")
            values.append(mode)
        if dial_frequency_hz is not None:
            clauses.append("dial_frequency_hz=?")
            values.append(int(dial_frequency_hz))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM fldigi_decodes{where} "  # noqa: S608
                "ORDER BY captured_at DESC LIMIT ?", (*values, limit),
            ).fetchall()
        return [self._fldigi_decode_row(row) for row in rows]

    @staticmethod
    def _weak_spot_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["is_cq"] = bool(result["is_cq"])
        return result

    def add_weak_signal_spots(self, spots: list[dict], *, job_id: str) -> list[dict]:
        inserted = []
        with self._connect() as connection:
            for spot in spots:
                spot_id = spot.get("spot_id") or f"spot-{uuid4().hex}"
                values = {**spot, "spot_id": spot_id, "job_id": job_id}
                connection.execute(
                    """
                    INSERT INTO weak_signal_spots (
                        spot_id, job_id, mode, dial_frequency_hz, audio_frequency_hz,
                        rf_frequency_hz, utc_text, snr_db, time_offset_seconds,
                        drift_hz_per_minute, message, callsign, grid, power_dbm,
                        is_cq, captured_at, raw_line
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (spot_id, job_id, values["mode"], values["dial_frequency_hz"],
                     values.get("audio_frequency_hz"), values.get("rf_frequency_hz"),
                     values.get("utc_text"), values.get("snr_db"),
                     values.get("time_offset_seconds"), values.get("drift_hz_per_minute"),
                     values.get("message", ""), values.get("callsign"), values.get("grid"),
                     values.get("power_dbm"), 1 if values.get("is_cq") else 0,
                     values["captured_at"], values.get("raw_line", "")),
                )
                inserted.append(values)
        return inserted

    def list_weak_signal_spots(
        self, *, mode: str | None = None, callsign: str | None = None,
        dial_frequency_hz: int | None = None, limit: int = 200,
    ) -> list[dict]:
        clauses, values = [], []
        if mode:
            clauses.append("mode=?")
            values.append(mode.lower())
        if callsign:
            clauses.append("callsign=? COLLATE NOCASE")
            values.append(callsign)
        if dial_frequency_hz is not None:
            clauses.append("dial_frequency_hz=?")
            values.append(int(dial_frequency_hz))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM weak_signal_spots{where} "  # noqa: S608
                "ORDER BY captured_at DESC LIMIT ?", (*values, limit),
            ).fetchall()
        return [self._weak_spot_row(row) for row in rows]

    @staticmethod
    def _fm_station_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["alternative_frequencies_hz"] = json.loads(
            result.pop("alternative_frequencies_json") or "[]"
        )
        for field in ("tp", "ta", "stereo_detected"):
            if result[field] is not None:
                result[field] = bool(result[field])
        return result

    def upsert_fm_station(self, station: dict, *, job_id: str, observed_at: str) -> dict:
        frequency_hz = int(station["frequency_hz"])
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM fm_stations WHERE frequency_hz=?", (frequency_hz,)
            ).fetchone()
            first_seen = existing["first_seen_at"] if existing else observed_at
            count = int(existing["observation_count"]) + 1 if existing else 1
            def retain(name: str):
                value = station.get(name)
                return value if value not in (None, "", []) else (existing[name] if existing else value)
            af = station.get("alternative_frequencies_hz") or (
                json.loads(existing["alternative_frequencies_json"]) if existing else []
            )
            values = {
                "pi_code": retain("pi_code"), "ps": retain("ps"),
                "pty": retain("pty"), "pty_name": retain("pty_name"),
                "ptyn": retain("ptyn"), "radiotext": retain("radiotext"),
                "tp": retain("tp"), "ta": retain("ta"),
                "music_speech": retain("music_speech"),
            }
            connection.execute(
                """
                INSERT INTO fm_stations (
                    frequency_hz, pi_code, ps, pty, pty_name, ptyn, radiotext,
                    tp, ta, music_speech, alternative_frequencies_json,
                    stereo_detected, estimated_snr_db, pilot_to_composite_rms_db,
                    rds_group_count, first_seen_at, last_seen_at,
                    last_survey_job_id, observation_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(frequency_hz) DO UPDATE SET
                    pi_code=excluded.pi_code, ps=excluded.ps, pty=excluded.pty,
                    pty_name=excluded.pty_name, ptyn=excluded.ptyn,
                    radiotext=excluded.radiotext, tp=excluded.tp, ta=excluded.ta,
                    music_speech=excluded.music_speech,
                    alternative_frequencies_json=excluded.alternative_frequencies_json,
                    stereo_detected=excluded.stereo_detected,
                    estimated_snr_db=excluded.estimated_snr_db,
                    pilot_to_composite_rms_db=excluded.pilot_to_composite_rms_db,
                    rds_group_count=excluded.rds_group_count,
                    last_seen_at=excluded.last_seen_at,
                    last_survey_job_id=excluded.last_survey_job_id,
                    observation_count=excluded.observation_count
                """,
                (frequency_hz, values["pi_code"], values["ps"], values["pty"],
                 values["pty_name"], values["ptyn"], values["radiotext"],
                 values["tp"], values["ta"], values["music_speech"], json.dumps(af),
                 1 if station.get("stereo_detected") else 0,
                 station.get("estimated_snr_db"), station.get("pilot_to_composite_rms_db"),
                 int(station.get("rds_group_count", 0)), first_seen, observed_at,
                 job_id, count),
            )
        return self.get_fm_station(frequency_hz)

    def get_fm_station(self, frequency_hz: int) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM fm_stations WHERE frequency_hz=?", (int(frequency_hz),)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown FM station frequency: {int(frequency_hz)} Hz")
        return self._fm_station_row(row)

    def list_fm_stations(self, *, rds_only: bool = False, limit: int = 200) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        where = " WHERE pi_code IS NOT NULL" if rds_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM fm_stations{where} ORDER BY frequency_hz LIMIT ?",  # noqa: S608
                (limit,),
            ).fetchall()
        return [self._fm_station_row(row) for row in rows]

    def _safe_path(self, path: Path | str, *, must_exist: bool = True) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.data_dir)
        except ValueError as exc:
            raise ValueError(f"Artifact path must be within {self.data_dir}") from exc
        if must_exist and (not resolved.exists() or not resolved.is_file()):
            raise FileNotFoundError(resolved)
        return resolved

    def upsert_job(
        self,
        job_id: str,
        job_type: str,
        state: str,
        *,
        config: dict | None = None,
        summary: dict | None = None,
        result_json_path: Path | str | None = None,
        created_at: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: str | None = None,
    ) -> None:
        safe_result = None
        if result_json_path is not None:
            safe_result = str(self._safe_path(result_json_path, must_exist=False))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, state, config_json, summary_json,
                    result_json_path, created_at, started_at, completed_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type=excluded.job_type,
                    state=excluded.state,
                    config_json=excluded.config_json,
                    summary_json=COALESCE(excluded.summary_json, jobs.summary_json),
                    result_json_path=COALESCE(excluded.result_json_path, jobs.result_json_path),
                    started_at=COALESCE(excluded.started_at, jobs.started_at),
                    completed_at=COALESCE(excluded.completed_at, jobs.completed_at),
                    error=excluded.error
                """,
                (
                    job_id,
                    job_type,
                    state,
                    json.dumps(config or {}),
                    json.dumps(summary) if summary is not None else None,
                    safe_result,
                    created_at or utc_now(),
                    started_at,
                    completed_at,
                    error,
                ),
            )

    def mark_interrupted_jobs(self) -> int:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state='interrupted', completed_at=?,
                    error=COALESCE(error, 'rf-mcp service restarted before job completion')
                WHERE state IN ('queued', 'running', 'stopping')
                """,
                (now,),
            )
            return cursor.rowcount

    @staticmethod
    def _preset_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    def save_preset(
        self,
        *,
        name: str,
        preset_type: str,
        description: str,
        config: dict,
        replace_existing: bool = False,
    ) -> dict:
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM presets WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"Preset name already exists: {existing['name']}; set replace_existing=true"
                )
            if existing is None:
                preset_id = f"preset-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO presets (
                        preset_id, name, preset_type, description, config_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        preset_id,
                        name,
                        preset_type,
                        description,
                        json.dumps(config),
                        now,
                        now,
                    ),
                )
            else:
                preset_id = existing["preset_id"]
                connection.execute(
                    """
                    UPDATE presets
                    SET name=?, preset_type=?, description=?, config_json=?, updated_at=?
                    WHERE preset_id=?
                    """,
                    (name, preset_type, description, json.dumps(config), now, preset_id),
                )
        return self.get_preset(preset_id)

    def get_preset(self, preset_id_or_name: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM presets WHERE preset_id=?",
                (preset_id_or_name,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM presets WHERE name=? COLLATE NOCASE",
                    (preset_id_or_name,),
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown RF preset: {preset_id_or_name}")
        return self._preset_row(row)

    def list_presets(self, *, preset_type: str | None = None, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        with self._connect() as connection:
            if preset_type:
                rows = connection.execute(
                    """
                    SELECT * FROM presets WHERE preset_type=?
                    ORDER BY name COLLATE NOCASE LIMIT ?
                    """,
                    (preset_type, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM presets ORDER BY name COLLATE NOCASE LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._preset_row(row) for row in rows]

    def delete_preset(self, preset_id_or_name: str) -> dict:
        preset = self.get_preset(preset_id_or_name)
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM presets WHERE preset_id=?", (preset["preset_id"],)
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Preset is used by one or more schedules; delete those first") from exc
        return preset

    @staticmethod
    def _schedule_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        return result

    def save_schedule(
        self,
        *,
        name: str,
        preset_id: str,
        interval_seconds: int,
        enabled: bool,
        next_run_at: str,
        replace_existing: bool = False,
    ) -> dict:
        self.get_preset(preset_id)
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM schedules WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"Schedule name already exists: {existing['name']}; set replace_existing=true"
                )
            if existing is None:
                schedule_id = f"schedule-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO schedules (
                        schedule_id, name, preset_id, interval_seconds, enabled,
                        next_run_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        schedule_id,
                        name,
                        preset_id,
                        interval_seconds,
                        1 if enabled else 0,
                        next_run_at,
                        now,
                        now,
                    ),
                )
            else:
                schedule_id = existing["schedule_id"]
                connection.execute(
                    """
                    UPDATE schedules
                    SET name=?, preset_id=?, interval_seconds=?, enabled=?,
                        next_run_at=?, updated_at=?
                    WHERE schedule_id=?
                    """,
                    (
                        name,
                        preset_id,
                        interval_seconds,
                        1 if enabled else 0,
                        next_run_at,
                        now,
                        schedule_id,
                    ),
                )
        return self.get_schedule(schedule_id)

    def get_schedule(self, schedule_id_or_name: str) -> dict:
        query = """
            SELECT schedules.*, presets.name AS preset_name,
                   presets.preset_type AS preset_type
            FROM schedules JOIN presets USING(preset_id)
        """
        with self._connect() as connection:
            row = connection.execute(
                query + " WHERE schedule_id=?", (schedule_id_or_name,)
            ).fetchone()
            if row is None:
                row = connection.execute(
                    query + " WHERE schedules.name=? COLLATE NOCASE", (schedule_id_or_name,)
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown RF schedule: {schedule_id_or_name}")
        return self._schedule_row(row)

    def list_schedules(self, *, enabled: bool | None = None, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        query = """
            SELECT schedules.*, presets.name AS preset_name,
                   presets.preset_type AS preset_type
            FROM schedules JOIN presets USING(preset_id)
        """
        values: list[Any] = []
        if enabled is not None:
            query += " WHERE schedules.enabled=?"
            values.append(1 if enabled else 0)
        query += " ORDER BY schedules.next_run_at, schedules.name COLLATE NOCASE LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._schedule_row(row) for row in rows]

    def due_schedules(self, now: str, *, limit: int = 20) -> list[dict]:
        query = """
            SELECT schedules.*, presets.name AS preset_name,
                   presets.preset_type AS preset_type
            FROM schedules JOIN presets USING(preset_id)
            WHERE schedules.enabled=1 AND schedules.next_run_at <= ?
            ORDER BY schedules.next_run_at LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, (now, max(1, min(limit, 100)))).fetchall()
        return [self._schedule_row(row) for row in rows]

    def advance_schedule(self, schedule_id: str, *, attempted_at: str, next_run_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE schedules
                SET last_attempt_at=?, next_run_at=?, updated_at=?
                WHERE schedule_id=?
                """,
                (attempted_at, next_run_at, attempted_at, schedule_id),
            )

    def record_schedule_result(
        self,
        schedule_id: str,
        *,
        status: str,
        attempted_at: str,
        job_id: str | None = None,
        error: str | None = None,
    ) -> dict:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE schedules
                SET last_run_at=CASE WHEN ? IN ('launched', 'completed') THEN ? ELSE last_run_at END,
                    last_job_id=COALESCE(?, last_job_id), last_status=?, last_error=?, updated_at=?
                WHERE schedule_id=?
                """,
                (status, attempted_at, job_id, status, error, attempted_at, schedule_id),
            )
        return self.get_schedule(schedule_id)

    def set_schedule_enabled(
        self, schedule_id_or_name: str, *, enabled: bool, next_run_at: str
    ) -> dict:
        schedule = self.get_schedule(schedule_id_or_name)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE schedules SET enabled=?, next_run_at=?, updated_at=?
                WHERE schedule_id=?
                """,
                (1 if enabled else 0, next_run_at, now, schedule["schedule_id"]),
            )
        return self.get_schedule(schedule["schedule_id"])

    def delete_schedule(self, schedule_id_or_name: str) -> dict:
        schedule = self.get_schedule(schedule_id_or_name)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM schedules WHERE schedule_id=?", (schedule["schedule_id"],)
            )
        return schedule

    @staticmethod
    def _alert_rule_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        return result

    def save_alert_rule(
        self,
        *,
        name: str,
        schedule_id: str,
        entry_label: str | None,
        condition_type: str,
        classification_label: str | None,
        min_confidence: float | None,
        threshold_db: float | None,
        enabled: bool,
        replace_existing: bool = False,
    ) -> dict:
        schedule = self.get_schedule(schedule_id)
        if schedule["preset_type"] != "watchlist":
            raise ValueError("v0.14 alert rules require a watchlist schedule")
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM alert_rules WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"Alert rule name already exists: {existing['name']}; "
                    "set replace_existing=true"
                )
            if existing is None:
                rule_id = f"rule-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO alert_rules (
                        rule_id, name, schedule_id, entry_label, condition_type,
                        classification_label, min_confidence, threshold_db, enabled,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id,
                        name,
                        schedule_id,
                        entry_label,
                        condition_type,
                        classification_label,
                        min_confidence,
                        threshold_db,
                        1 if enabled else 0,
                        now,
                        now,
                    ),
                )
            else:
                rule_id = existing["rule_id"]
                connection.execute(
                    """
                    UPDATE alert_rules SET
                        name=?, schedule_id=?, entry_label=?, condition_type=?,
                        classification_label=?, min_confidence=?, threshold_db=?,
                        enabled=?, updated_at=? WHERE rule_id=?
                    """,
                    (
                        name,
                        schedule_id,
                        entry_label,
                        condition_type,
                        classification_label,
                        min_confidence,
                        threshold_db,
                        1 if enabled else 0,
                        now,
                        rule_id,
                    ),
                )
        return self.get_alert_rule(rule_id)

    def get_alert_rule(self, rule_id_or_name: str) -> dict:
        query = """
            SELECT alert_rules.*, schedules.name AS schedule_name
            FROM alert_rules JOIN schedules USING(schedule_id)
        """
        with self._connect() as connection:
            row = connection.execute(
                query + " WHERE rule_id=?", (rule_id_or_name,)
            ).fetchone()
            if row is None:
                row = connection.execute(
                    query + " WHERE alert_rules.name=? COLLATE NOCASE", (rule_id_or_name,)
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown RF alert rule: {rule_id_or_name}")
        return self._alert_rule_row(row)

    def list_alert_rules(
        self,
        *,
        schedule_id: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        clauses = []
        values: list[Any] = []
        if schedule_id is not None:
            clauses.append("alert_rules.schedule_id=?")
            values.append(schedule_id)
        if enabled is not None:
            clauses.append("alert_rules.enabled=?")
            values.append(1 if enabled else 0)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = """
            SELECT alert_rules.*, schedules.name AS schedule_name
            FROM alert_rules JOIN schedules USING(schedule_id)
        """ + where + " ORDER BY alert_rules.name COLLATE NOCASE LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._alert_rule_row(row) for row in rows]

    def set_alert_rule_enabled(self, rule_id_or_name: str, enabled: bool) -> dict:
        rule = self.get_alert_rule(rule_id_or_name)
        with self._connect() as connection:
            connection.execute(
                "UPDATE alert_rules SET enabled=?, updated_at=? WHERE rule_id=?",
                (1 if enabled else 0, utc_now(), rule["rule_id"]),
            )
        return self.get_alert_rule(rule["rule_id"])

    def delete_alert_rule(self, rule_id_or_name: str) -> dict:
        rule = self.get_alert_rule(rule_id_or_name)
        with self._connect() as connection:
            connection.execute("DELETE FROM alert_rules WHERE rule_id=?", (rule["rule_id"],))
        return rule

    def record_alert_event(
        self, *, rule: dict, job_id: str | None, observation: dict, message: str
    ) -> dict:
        event_id = f"alert-{uuid4().hex}"
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alert_events (
                    event_id, rule_id, rule_name, schedule_id, job_id,
                    observation_label, frequency_hz, message, details_json,
                    created_at, acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    event_id,
                    rule["rule_id"],
                    rule["name"],
                    rule["schedule_id"],
                    job_id,
                    observation.get("label"),
                    observation.get("frequency_hz"),
                    message,
                    json.dumps({"rule": rule, "observation": observation}),
                    now,
                ),
            )
        return self.get_alert_event(event_id)

    @staticmethod
    def _alert_event_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json"))
        result["acknowledged"] = result["acknowledged_at"] is not None
        return result

    def get_alert_event(self, event_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alert_events WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown RF alert event: {event_id}")
        return self._alert_event_row(row)

    def list_alert_events(
        self,
        *,
        acknowledged: bool | None = None,
        schedule_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        clauses = []
        values: list[Any] = []
        if acknowledged is not None:
            clauses.append(
                "acknowledged_at IS NOT NULL" if acknowledged else "acknowledged_at IS NULL"
            )
        if schedule_id is not None:
            clauses.append("schedule_id=?")
            values.append(schedule_id)
        if event_type is not None:
            clauses.append("event_type=?")
            values.append(event_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM alert_events{where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608
                (*values, limit),
            ).fetchall()
        return [self._alert_event_row(row) for row in rows]

    def acknowledge_alert_event(self, event_id: str) -> dict:
        self.get_alert_event(event_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE alert_events SET acknowledged_at=COALESCE(acknowledged_at, ?) "
                "WHERE event_id=?",
                (utc_now(), event_id),
            )
        return self.get_alert_event(event_id)

    @staticmethod
    def _sstv_alert_rule_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["unique_only"] = bool(result["unique_only"])
        result["enabled"] = bool(result["enabled"])
        return result

    def save_sstv_alert_rule(
        self, *, name: str, frequency_hz: int | None, sstv_mode: str | None,
        minimum_quality: float, unique_only: bool, enabled: bool,
        replace_existing: bool = False,
    ) -> dict:
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM sstv_alert_rules WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"SSTV alert rule name already exists: {existing['name']}; "
                    "set replace_existing=true"
                )
            if existing is None:
                rule_id = f"sstv-rule-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO sstv_alert_rules (
                        rule_id, name, frequency_hz, sstv_mode, minimum_quality,
                        unique_only, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (rule_id, name, frequency_hz, sstv_mode, minimum_quality,
                     1 if unique_only else 0, 1 if enabled else 0, now, now),
                )
            else:
                rule_id = existing["rule_id"]
                connection.execute(
                    """
                    UPDATE sstv_alert_rules SET name=?, frequency_hz=?, sstv_mode=?,
                        minimum_quality=?, unique_only=?, enabled=?, updated_at=?
                    WHERE rule_id=?
                    """,
                    (name, frequency_hz, sstv_mode, minimum_quality,
                     1 if unique_only else 0, 1 if enabled else 0, now, rule_id),
                )
        return self.get_sstv_alert_rule(rule_id)

    def get_sstv_alert_rule(self, rule_id_or_name: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sstv_alert_rules WHERE rule_id=?", (rule_id_or_name,)
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM sstv_alert_rules WHERE name=? COLLATE NOCASE",
                    (rule_id_or_name,),
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown SSTV alert rule: {rule_id_or_name}")
        return self._sstv_alert_rule_row(row)

    def list_sstv_alert_rules(
        self, *, enabled: bool | None = None, limit: int = 100,
    ) -> list[dict]:
        where = " WHERE enabled=?" if enabled is not None else ""
        values = [1 if enabled else 0] if enabled is not None else []
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM sstv_alert_rules{where} "  # noqa: S608
                "ORDER BY name COLLATE NOCASE LIMIT ?",
                (*values, max(1, min(int(limit), 200))),
            ).fetchall()
        return [self._sstv_alert_rule_row(row) for row in rows]

    def set_sstv_alert_rule_enabled(self, rule_id_or_name: str, enabled: bool) -> dict:
        rule = self.get_sstv_alert_rule(rule_id_or_name)
        with self._connect() as connection:
            connection.execute(
                "UPDATE sstv_alert_rules SET enabled=?, updated_at=? WHERE rule_id=?",
                (1 if enabled else 0, utc_now(), rule["rule_id"]),
            )
        return self.get_sstv_alert_rule(rule["rule_id"])

    def delete_sstv_alert_rule(self, rule_id_or_name: str) -> dict:
        rule = self.get_sstv_alert_rule(rule_id_or_name)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE webhook_destinations SET sstv_rule_id=NULL "
                "WHERE sstv_rule_id=?", (rule["rule_id"],)
            )
            connection.execute(
                "DELETE FROM sstv_alert_rules WHERE rule_id=?", (rule["rule_id"],)
            )
        return rule

    def record_sstv_alert_event(
        self, *, rule: dict, image: dict, message: str,
    ) -> dict:
        event_id = f"alert-{uuid4().hex}"
        details = {
            "schema": "rf-mcp.sstv-alert.v1",
            "rule": rule,
            "image": image,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alert_events (
                    event_id, rule_id, rule_name, schedule_id, job_id,
                    observation_label, frequency_hz, message, details_json,
                    event_type, sstv_rule_id, created_at, acknowledged_at
                ) VALUES (?, NULL, ?, NULL, ?, ?, ?, ?, ?, 'sstv_image', ?, ?, NULL)
                """,
                (event_id, rule["name"], image.get("job_id"), image.get("sstv_mode"),
                 image.get("frequency_hz"), message, json.dumps(details),
                 rule["rule_id"], utc_now()),
            )
        return self.get_alert_event(event_id)

    @staticmethod
    def _webhook_destination_row(row: sqlite3.Row, *, include_secret: bool = False) -> dict:
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        result["all_rules"] = bool(result["all_rules"])
        secret = result.pop("signing_secret")
        result["has_signing_secret"] = bool(secret)
        if include_secret:
            result["signing_secret"] = secret
        return result

    def save_webhook_destination(
        self,
        *,
        name: str,
        url: str,
        signing_secret: str | None,
        all_rules: bool,
        rule_id: str | None,
        sstv_rule_id: str | None,
        satellite_watch_id: str | None,
        enabled: bool,
        replace_existing: bool = False,
    ) -> dict:
        if rule_id is not None:
            self.get_alert_rule(rule_id)
        if sstv_rule_id is not None:
            self.get_sstv_alert_rule(sstv_rule_id)
        if satellite_watch_id is not None:
            self.get_satellite_watch(satellite_watch_id)
        if sum(value is not None for value in (rule_id, sstv_rule_id, satellite_watch_id)) > 1:
            raise ValueError("A webhook destination can select only one alert source")
        now = utc_now()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM webhook_destinations WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if existing is not None and not replace_existing:
                raise ValueError(
                    f"Webhook destination name already exists: {existing['name']}; "
                    "set replace_existing=true"
                )
            if existing is None:
                destination_id = f"webhook-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO webhook_destinations (
                        destination_id, name, url, signing_secret, all_rules, rule_id,
                        sstv_rule_id, satellite_watch_id, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        destination_id, name, url, signing_secret, 1 if all_rules else 0,
                        rule_id, sstv_rule_id, satellite_watch_id,
                        1 if enabled else 0, now, now,
                    ),
                )
            else:
                destination_id = existing["destination_id"]
                secret = existing["signing_secret"] if signing_secret is None else signing_secret
                connection.execute(
                    """
                    UPDATE webhook_destinations SET name=?, url=?, signing_secret=?,
                        all_rules=?, rule_id=?, sstv_rule_id=?, satellite_watch_id=?,
                        enabled=?, updated_at=?
                    WHERE destination_id=?
                    """,
                    (
                        name, url, secret, 1 if all_rules else 0, rule_id,
                        sstv_rule_id, satellite_watch_id, 1 if enabled else 0, now,
                        destination_id,
                    ),
                )
        return self.get_webhook_destination(destination_id)

    def get_webhook_destination(
        self, destination_id_or_name: str, *, include_secret: bool = False
    ) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM webhook_destinations WHERE destination_id=?",
                (destination_id_or_name,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM webhook_destinations WHERE name=? COLLATE NOCASE",
                    (destination_id_or_name,),
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown webhook destination: {destination_id_or_name}")
        return self._webhook_destination_row(row, include_secret=include_secret)

    def list_webhook_destinations(self, *, enabled: bool | None = None, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        where = " WHERE enabled=?" if enabled is not None else ""
        values = [1 if enabled else 0] if enabled is not None else []
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM webhook_destinations{where} ORDER BY name COLLATE NOCASE LIMIT ?",  # noqa: S608
                (*values, limit),
            ).fetchall()
        return [self._webhook_destination_row(row) for row in rows]

    def set_webhook_destination_enabled(self, destination_id_or_name: str, enabled: bool) -> dict:
        destination = self.get_webhook_destination(destination_id_or_name)
        with self._connect() as connection:
            connection.execute(
                "UPDATE webhook_destinations SET enabled=?, updated_at=? WHERE destination_id=?",
                (1 if enabled else 0, utc_now(), destination["destination_id"]),
            )
        return self.get_webhook_destination(destination["destination_id"])

    def delete_webhook_destination(self, destination_id_or_name: str) -> dict:
        destination = self.get_webhook_destination(destination_id_or_name)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE webhook_deliveries SET state='cancelled', next_attempt_at=NULL,
                    last_error='Webhook destination deleted before delivery', updated_at=?
                WHERE destination_id=? AND state IN ('pending', 'retrying')
                """,
                (utc_now(), destination["destination_id"]),
            )
            connection.execute(
                "DELETE FROM webhook_destinations WHERE destination_id=?",
                (destination["destination_id"],),
            )
        return destination

    def enqueue_webhook_deliveries(self, event: dict) -> list[dict]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM webhook_destinations
                WHERE enabled=1 AND (
                    all_rules=1 OR rule_id=? OR sstv_rule_id=?
                    OR satellite_watch_id=?
                )
                ORDER BY name COLLATE NOCASE
                """,
                (event.get("rule_id"), event.get("sstv_rule_id"),
                 event.get("satellite_watch_id")),
            ).fetchall()
            created = []
            now = utc_now()
            schemas = {
                "sstv_image": "rf-mcp.sstv-alert.v1",
                "satellite_pass": "rf-mcp.satellite-pass.v1",
                "satellite_telemetry": "rf-mcp.satellite-telemetry-alert.v1",
            }
            payload = {
                "schema": schemas.get(event.get("event_type"), "rf-mcp.alert.v1"),
                "server": "MiniRackDisplay",
                "event": event,
            }
            for row in rows:
                delivery_id = f"delivery-{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO webhook_deliveries (
                        delivery_id, event_id, destination_id, destination_name,
                        destination_url, payload_json, state, attempt_count,
                        next_attempt_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                    """,
                    (
                        delivery_id, event["event_id"], row["destination_id"], row["name"],
                        row["url"], json.dumps(payload), now, now, now,
                    ),
                )
                created.append(delivery_id)
        return [self.get_webhook_delivery(item) for item in created]

    @staticmethod
    def _webhook_delivery_row(row: sqlite3.Row, *, include_secret: bool = False) -> dict:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        secret = result.pop("signing_secret", None)
        if include_secret:
            result["signing_secret"] = secret
        return result

    def get_webhook_delivery(self, delivery_id: str, *, include_secret: bool = False) -> dict:
        query = """
            SELECT webhook_deliveries.*, webhook_destinations.signing_secret
            FROM webhook_deliveries
            LEFT JOIN webhook_destinations USING(destination_id)
            WHERE delivery_id=?
        """
        with self._connect() as connection:
            row = connection.execute(query, (delivery_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown webhook delivery: {delivery_id}")
        return self._webhook_delivery_row(row, include_secret=include_secret)

    def due_webhook_deliveries(self, now: str, *, limit: int = 20) -> list[dict]:
        query = """
            SELECT webhook_deliveries.*, webhook_destinations.signing_secret
            FROM webhook_deliveries
            LEFT JOIN webhook_destinations USING(destination_id)
            WHERE state IN ('pending', 'retrying') AND next_attempt_at <= ?
            ORDER BY next_attempt_at LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, (now, max(1, min(int(limit), 100)))).fetchall()
        return [self._webhook_delivery_row(row, include_secret=True) for row in rows]

    def record_webhook_delivery_attempt(
        self, delivery_id: str, *, state: str, attempt_count: int,
        http_status: int | None, error: str | None, next_attempt_at: str | None,
        delivered_at: str | None,
    ) -> dict:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE webhook_deliveries SET state=?, attempt_count=?, http_status=?,
                    last_error=?, next_attempt_at=?, delivered_at=?, updated_at=?
                WHERE delivery_id=?
                """,
                (state, attempt_count, http_status, error, next_attempt_at,
                 delivered_at, now, delivery_id),
            )
        return self.get_webhook_delivery(delivery_id)

    def retry_webhook_delivery(self, delivery_id: str) -> dict:
        delivery = self.get_webhook_delivery(delivery_id)
        if delivery["destination_id"] is None:
            raise ValueError("Cannot retry after the webhook destination was deleted")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE webhook_deliveries SET state='pending', attempt_count=0,
                    next_attempt_at=?, http_status=NULL, last_error=NULL,
                    delivered_at=NULL, updated_at=? WHERE delivery_id=?
                """,
                (utc_now(), utc_now(), delivery_id),
            )
        return self.get_webhook_delivery(delivery_id)

    def list_webhook_deliveries(
        self, *, state: str | None = None, event_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        clauses = []
        values: list[Any] = []
        if state:
            clauses.append("webhook_deliveries.state=?")
            values.append(state)
        if event_id:
            clauses.append("event_id=?")
            values.append(event_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = """
            SELECT webhook_deliveries.*, NULL AS signing_secret
            FROM webhook_deliveries
        """ + where + " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._webhook_delivery_row(row) for row in rows]

    def webhook_delivery_counts(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM webhook_deliveries GROUP BY state"
            ).fetchall()
        return {row["state"]: row["count"] for row in rows}

    def register_artifact(
        self,
        path: Path | str,
        kind: str,
        *,
        job_id: str | None = None,
        mime_type: str | None = None,
        created_at: str | None = None,
    ) -> dict:
        resolved = self._safe_path(path, must_exist=False)
        stat = resolved.stat()
        if not stat_module.S_ISREG(stat.st_mode):
            raise FileNotFoundError(resolved)
        mime_type = mime_type or mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        with self._lock, self._connect() as connection:
            artifact_id = f"art-{uuid4().hex}"
            row = connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, job_id, kind, path, filename, mime_type,
                    size_bytes, mtime_ns, created_at, pinned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(path) DO UPDATE SET
                    job_id=COALESCE(excluded.job_id, artifacts.job_id),
                    kind=excluded.kind, filename=excluded.filename,
                    mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns
                RETURNING artifact_id
                """,
                (artifact_id, job_id, kind, str(resolved), resolved.name, mime_type,
                 stat.st_size, stat.st_mtime_ns, created_at or datetime.fromtimestamp(
                     stat.st_mtime, tz=timezone.utc).isoformat()),
            ).fetchone()
            artifact_id = row["artifact_id"]
        return self.get_artifact(artifact_id)

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json") or "{}")
        summary_json = result.pop("summary_json")
        result["summary"] = json.loads(summary_json) if summary_json else None
        return result

    @staticmethod
    def _artifact_row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["pinned"] = bool(result["pinned"])
        return result

    def list_jobs(
        self,
        *,
        job_type: str | None = None,
        state: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        clauses: list[str] = []
        values: list[Any] = []
        if job_type:
            clauses.append("job_type=?")
            values.append(job_type)
        if state:
            clauses.append("state=?")
            values.append(state)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608
                (*values, limit),
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown persisted RF job_id: {job_id}")
            artifacts = connection.execute(
                "SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at", (job_id,)
            ).fetchall()
        result = self._job_row(row)
        result["artifacts"] = [self._artifact_row(item) for item in artifacts]
        result_path = result.get("result_json_path")
        if result_path:
            try:
                safe_path = self._safe_path(result_path)
                result["result"] = json.loads(safe_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                result["result"] = None
        return result

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        job_id: str | None = None,
        pinned: bool | None = None,
        limit: int = 50,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        clauses: list[str] = []
        values: list[Any] = []
        if kind:
            clauses.append("kind=?")
            values.append(kind)
        if job_id:
            clauses.append("job_id=?")
            values.append(job_id)
        if pinned is not None:
            clauses.append("pinned=?")
            values.append(1 if pinned else 0)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM artifacts{where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608
                (*values, limit),
            ).fetchall()
        return [self._artifact_row(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown artifact_id: {artifact_id}")
        result = self._artifact_row(row)
        path = self._safe_path(result["path"])
        result["size_bytes"] = path.stat().st_size
        return result

    def set_pinned(self, artifact_id: str, pinned: bool) -> dict:
        self.get_artifact(artifact_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE artifacts SET pinned=? WHERE artifact_id=?",
                (1 if pinned else 0, artifact_id),
            )
        return self.get_artifact(artifact_id)

    def storage_status(self) -> dict:
        usage = shutil.disk_usage(self.data_dir)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT kind, COUNT(*) AS count, COALESCE(SUM(size_bytes), 0) AS size_bytes
                FROM artifacts GROUP BY kind ORDER BY size_bytes DESC
                """
            ).fetchall()
            total = connection.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(size_bytes), 0) AS size_bytes FROM artifacts"
            ).fetchone()
        return {
            "data_directory": str(self.data_dir),
            "filesystem": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_percent": 100 * usage.free / usage.total,
            },
            "cataloged_artifacts": dict(total),
            "by_kind": [dict(row) for row in rows],
            "database_path": str(self.database_path),
            "database_size_bytes": self.database_path.stat().st_size,
        }

    def database_health(self) -> dict:
        with self._connect() as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            job_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            artifact_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            preset_count = connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0]
            schedule_count = connection.execute("SELECT COUNT(*) FROM schedules").fetchone()[0]
            alert_rule_count = connection.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0]
            alert_event_count = connection.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0]
            sstv_alert_rule_count = connection.execute(
                "SELECT COUNT(*) FROM sstv_alert_rules"
            ).fetchone()[0]
            sstv_alert_event_count = connection.execute(
                "SELECT COUNT(*) FROM alert_events WHERE event_type='sstv_image'"
            ).fetchone()[0]
            webhook_destination_count = connection.execute(
                "SELECT COUNT(*) FROM webhook_destinations"
            ).fetchone()[0]
            webhook_delivery_count = connection.execute(
                "SELECT COUNT(*) FROM webhook_deliveries"
            ).fetchone()[0]
            satellite_watch_count = connection.execute(
                "SELECT COUNT(*) FROM satellite_watches"
            ).fetchone()[0]
            satellite_pass_count = connection.execute(
                "SELECT COUNT(*) FROM satellite_passes"
            ).fetchone()[0]
            satellite_pass_alert_count = connection.execute(
                "SELECT COUNT(*) FROM alert_events WHERE event_type='satellite_pass'"
            ).fetchone()[0]
            satellite_tle_refresh_failure_count = connection.execute(
                "SELECT COUNT(*) FROM satellite_watches "
                "WHERE last_tle_refresh_status='failed'"
            ).fetchone()[0]
            satellite_doppler_plan_count = connection.execute(
                "SELECT COUNT(*) FROM satellite_passes WHERE doppler_plan_json!='[]'"
            ).fetchone()[0]
            satellite_doppler_plot_count = connection.execute(
                "SELECT COUNT(*) FROM satellite_passes WHERE doppler_artifact_id IS NOT NULL"
            ).fetchone()[0]
        return {
            "status": "healthy" if quick_check == "ok" else "degraded",
            "quick_check": quick_check,
            "job_count": job_count,
            "artifact_count": artifact_count,
            "preset_count": preset_count,
            "schedule_count": schedule_count,
            "alert_rule_count": alert_rule_count,
            "alert_event_count": alert_event_count,
            "sstv_alert_rule_count": sstv_alert_rule_count,
            "sstv_alert_event_count": sstv_alert_event_count,
            "webhook_destination_count": webhook_destination_count,
            "webhook_delivery_count": webhook_delivery_count,
            "satellite_watch_count": satellite_watch_count,
            "satellite_pass_count": satellite_pass_count,
            "satellite_pass_alert_count": satellite_pass_alert_count,
            "satellite_tle_refresh_failure_count": satellite_tle_refresh_failure_count,
            "satellite_doppler_plan_count": satellite_doppler_plan_count,
            "satellite_doppler_plot_count": satellite_doppler_plot_count,
        }

    def cleanup(
        self,
        *,
        older_than_days: float,
        kinds: Iterable[str] | None,
        max_delete: int,
        dry_run: bool,
    ) -> dict:
        older_than_days = float(older_than_days)
        if older_than_days < 1:
            raise ValueError("older_than_days must be at least 1")
        max_delete = max(1, min(int(max_delete), 1000))
        cutoff_timestamp = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        cutoff = datetime.fromtimestamp(cutoff_timestamp, tz=timezone.utc).isoformat()
        clauses = ["pinned=0", "created_at < ?"]
        values: list[Any] = [cutoff]
        kind_list = [item for item in (kinds or []) if item]
        if kind_list:
            placeholders = ",".join("?" for _ in kind_list)
            clauses.append(f"kind IN ({placeholders})")
            values.extend(kind_list)
        where = " AND ".join(clauses)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM artifacts WHERE {where} ORDER BY created_at LIMIT ?",  # noqa: S608
                (*values, max_delete),
            ).fetchall()
            candidates = [self._artifact_row(row) for row in rows]
            deleted: list[dict] = []
            failures: list[dict] = []
            if not dry_run:
                for candidate in candidates:
                    try:
                        path = self._safe_path(candidate["path"], must_exist=False)
                        if path.exists():
                            path.unlink()
                        connection.execute(
                            "DELETE FROM artifacts WHERE artifact_id=?",
                            (candidate["artifact_id"],),
                        )
                        deleted.append(candidate)
                    except OSError as exc:
                        failures.append(
                            {"artifact_id": candidate["artifact_id"], "error": str(exc)}
                        )
        return {
            "dry_run": bool(dry_run),
            "older_than_days": older_than_days,
            "candidate_count": len(candidates),
            "candidate_bytes": sum(item["size_bytes"] for item in candidates),
            "candidates": candidates,
            "deleted_count": len(deleted),
            "deleted_bytes": sum(item["size_bytes"] for item in deleted),
            "failures": failures,
        }

    def index_existing_artifacts(self) -> int:
        """Reconcile managed artifact directories in one SQLite transaction.

        This is deliberately not run by construction; callers should schedule it
        after startup or invoke the module's maintenance command.
        """
        mapping = {
            "captures": "iq_capture",
            "plots": "plot",
            "audio": "audio_wav",
            "results": "result_json",
        }
        discovered: dict[str, tuple[Path, str, Any]] = {}
        for directory_name, kind in mapping.items():
            directory = self.data_dir / directory_name
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.name.endswith(".part"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat_module.S_ISREG(stat.st_mode):
                    discovered[str(path.resolve())] = (path.resolve(), kind, stat)

        managed_roots = tuple(str((self.data_dir / name).resolve()) for name in mapping)
        writes = 0
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT path,size_bytes,mtime_ns FROM artifacts"
            ).fetchall()
            cataloged = {row["path"]: row for row in rows}
            for path_text, (path, kind, stat) in discovered.items():
                existing = cataloged.get(path_text)
                if (existing is not None and existing["size_bytes"] == stat.st_size
                        and existing["mtime_ns"] == stat.st_mtime_ns):
                    continue
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                connection.execute(
                    """INSERT INTO artifacts
                       (artifact_id,job_id,kind,path,filename,mime_type,size_bytes,mtime_ns,
                        created_at,pinned) VALUES (?,NULL,?,?,?,?,?,?,?,0)
                       ON CONFLICT(path) DO UPDATE SET kind=excluded.kind,
                         filename=excluded.filename,mime_type=excluded.mime_type,
                         size_bytes=excluded.size_bytes,mtime_ns=excluded.mtime_ns""",
                    (f"art-{uuid4().hex}", kind, path_text, path.name, mime_type,
                     stat.st_size, stat.st_mtime_ns,
                     datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()),
                )
                writes += 1
            missing = [path for path in cataloged if path not in discovered and any(
                path == root or path.startswith(root + "/") for root in managed_roots
            )]
            if missing:
                connection.executemany("DELETE FROM artifacts WHERE path=?",
                                       ((path,) for path in missing))
                writes += len(missing)
        return writes


ensure_data_dirs()
catalog = Catalog(index_existing=False)


def main() -> None:
    """Run catalog maintenance from ``python -m rf_mcp.catalog``."""
    import argparse

    parser = argparse.ArgumentParser(description="RF MCP catalog maintenance")
    parser.add_argument("command", choices=("reconcile",))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    with Catalog(args.data_dir) as maintenance_catalog:
        writes = maintenance_catalog.index_existing_artifacts()
    print(f"Artifact reconciliation complete: {writes} catalog writes")


if __name__ == "__main__":
    main()
