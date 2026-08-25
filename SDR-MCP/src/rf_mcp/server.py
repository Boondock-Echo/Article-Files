from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings
from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent

from .receiver_backend import capture_iq, device_info, offset_capture_center
from .alerts import AlertEvaluator, normalize_alert_rule
from .activity import (
    RF_BANDS,
    build_activity_dashboard,
    save_activity_exports,
    save_activity_plot,
)
from .api_contract import API_VERSION, contract_document
from .catalog import catalog
from .calibration import (
    delete_calibration, get_calibration, list_calibrations, save_calibration,
)
from .classification import (
    classify_features,
    extract_features,
    feature_dict,
    save_classification_plot,
)
from .comparison import SUPPORTED_JOB_TYPES, compare_survey_results, save_comparison_plot
from .config import AUDIO_DIR, MAX_PEAKS, PLOT_DIR, RESULT_DIR, SAMPLE_RATE, ensure_data_dirs
from .digital_decode import (
    decode_ax25_afsk1200,
    decode_bpsk31,
    decode_cw,
    decode_rtty,
    save_decode_plot,
)
from . import __version__
from .fm_survey import compare_fm_survey_results, fm_survey_manager
from .fldigi_bridge import (
    decode_live_fldigi,
    fldigi_status,
    list_fldigi_mode_capabilities,
    normalize_fldigi_mode,
)
from .monitoring import monitor_manager
from .live_iq import LiveIQManager
from .notifications import WebhookDispatcher, normalize_webhook_destination
from .operations import acquire_long_job, active_long_job, release_long_job
from .presets import PRESET_TYPES, normalize_preset
from .propagation import (
    fetch_space_weather,
    save_propagation_exports,
    save_propagation_plot,
    space_weather_interpretation,
    summarize_local_propagation,
)
from .rds import decode_rds
from .recording_workspace import (
    add_annotation as add_session_annotation,
    add_artifacts as add_session_artifacts,
    add_bookmark as add_session_bookmark,
    compare_wav,
    create_session,
    delete_session,
    export_session,
    extract_wav_clip,
    get_session,
    list_sessions,
    search_sessions,
    wav_info,
)
from .readiness import release_readiness
from .scanning import scan_manager
from .scheduling import (
    SchedulerManager,
    normalize_schedule,
    parse_utc,
    utc_now as scheduler_utc_now,
)
from .satellite import (
    SatellitePassScheduler,
    build_doppler_plan,
    normalize_satellite_watch,
    predict_passes,
    refresh_satellite_tle as refresh_satellite_tle_data,
    save_doppler_plot,
)
from .satellite_catalog import (
    CATEGORY_GROUPS,
    get_catalog_entry,
    search_catalog,
    selected_downlinks,
)
from .satellite_receiver import export_satellite_telemetry, satellite_receiver_manager
from .satellite_planner import (
    delete_location,
    get_location,
    list_locations,
    plan_observations,
    save_location,
)
from .satellite_performance import (
    build_pass_report,
    build_pass_reports,
    export_pass_performance,
    save_pass_performance_plot,
    score_satellite_pass,
    summarize_pass_performance,
)
from .satellite_telemetry import (
    decode_observation_telemetry,
    decode_payload as decode_telemetry_payload,
    export_decoded_telemetry,
    normalize_telemetry_schema,
    save_telemetry_plot,
)
from .satellite_telemetry_alerts import (
    evaluate_telemetry_alert_rule,
    normalize_telemetry_alert_rule,
)
from .signal_analysis import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_BANDWIDTHS_HZ,
    demodulate,
    demodulate_broadcast_fm,
    downconvert,
    measure_signal,
    normalize_mode,
    save_audio_spectrum,
    save_broadcast_fm_plot,
    validate_bandwidth,
    write_wav,
    _complex_lowpass,
)
from .signal_library import (
    add_exemplar,
    delete_fingerprint,
    get_fingerprint,
    list_fingerprints,
    match_fingerprints,
    save_fingerprint,
)
from .services import ReceiverService, RfApplicationServices
from .station_memory import (
    delete as delete_station_memory_record,
    get as get_station_memory_record,
    list_memories,
    save as save_station_memory_record,
)
from .sdr_coordinator import (
    coordinator_status,
    delete_receiver,
    discover_backends,
    discover_devices,
    ensure_airspy_default,
    get_receiver,
    list_receivers,
    plan_assignment,
    release_receiver,
    acquire_receiver,
    save_receiver,
    register_discovered_device,
)
from .spectrum import (
    DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
    PSD_SCALE,
    analyze_peaks,
    averaged_psd_dbfs_per_hz,
    averaged_spectrum,
    integrate_psd_dbfs,
    iq_level_metrics,
    load_complex_float32,
    peak_dicts,
    save_plot,
    valid_passband_mask,
)
from .sstv import sstv_capabilities, sstv_manager
from .sstv_alerts import normalize_sstv_alert_rule
from .sstv_watcher import sstv_watcher_manager
from .web import RfWebApp, validate_api_token
from .weak_signal import decoder_capabilities, decode_live_weak_signal, normalize_weak_mode

# mcp currently leaves the generic lifespan annotation unresolved when its
# Pydantic settings model is first instantiated. Rebuilding after the SDK
# module has finished importing resolves FastMCP and LifespanResultT and avoids
# pydantic-settings' IncompleteFieldDefinitionWarning on Python 3.13.
FastMCPSettings.model_rebuild()

mcp = FastMCP(
    "Multi-SDR RF Lab",
    instructions=(
        "This is a receive-only RF analysis server. Frequencies are supplied in Hz. "
        "Call list_devices before the first spectrum inspection. Measurements are "
        "relative and must not be described as calibrated dBm unless a documented "
        "receiver calibration is present. Stable public API contract: 1.0."
    ),
    host=os.getenv("RF_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("RF_MCP_PORT", "8765")),
)
_SERVER_STARTED_AT = datetime.now(timezone.utc).isoformat()
_SERVER_STARTED_MONOTONIC = time.monotonic()
_MAX_INLINE_ARTIFACT_BYTES = int(os.getenv("RF_MCP_MAX_INLINE_ARTIFACT_BYTES", "10485760"))
_INTERRUPTED_JOBS_ON_STARTUP = catalog.mark_interrupted_jobs()
receiver_service = ReceiverService()
live_iq_manager = LiveIQManager()


class _LazyLiveAudioManager:
    """Keep scipy out of the lightweight MCP import/readiness path."""
    _manager = None
    def _get(self):
        if self._manager is None:
            from .live_audio import LiveAudioManager
            self._manager = LiveAudioManager(iq_manager=live_iq_manager)
        return self._manager
    def capabilities(self): return self._get().capabilities()
    def status(self): return self._get().status()
    def stop(self, session_id=None): return self._get().stop(session_id)
    def subscribe(self, settings): return self._get().subscribe(settings)
    def shutdown(self):
        if self._manager is not None: self._manager.shutdown()


live_audio_manager = _LazyLiveAudioManager()

class _LazyLiveWaterfallManager:
    _manager = None
    def _get(self):
        if self._manager is None:
            from .live_waterfall import LiveWaterfallManager
            self._manager = LiveWaterfallManager(iq_manager=live_iq_manager)
        return self._manager
    def status(self): return self._get().status()
    def stop(self, session_id=None): return self._get().stop(session_id)
    def subscribe(self, settings): return self._get().subscribe(settings)
    def shutdown(self):
        if self._manager is not None: self._manager.shutdown()


live_waterfall_manager = _LazyLiveWaterfallManager()


@mcp.tool()
def get_rf_api_contract() -> dict:
    """Return the stable v1 tool, units, measurement, and compatibility contract."""
    return contract_document(__version__)


@mcp.tool()
def get_release_readiness(probe_hardware: bool = False) -> dict:
    """Run non-destructive production checks; hardware probing is opt-in."""
    return release_readiness(catalog, probe_hardware=probe_hardware)


def _artifact_metadata(artifact: dict) -> dict:
    result = dict(artifact)
    result["download_path"] = f"/artifacts/{artifact['artifact_id']}"
    return result


def _persist_one_shot(
    *,
    job_id: str,
    job_type: str,
    result: dict,
    config: dict,
    artifacts: list[tuple[Path | str | None, str]],
) -> None:
    catalog.upsert_job(
        job_id,
        job_type,
        "completed",
        config=config,
        summary={
            "frequency_hz": result.get("requested_frequency_hz", result.get("center_frequency_hz")),
            "peak_count": result.get("peak_count"),
            "estimated_snr_db": result.get("metrics", {}).get("estimated_snr_db"),
        },
        result_json_path=result["result_json_path"],
        created_at=result.get("started_at"),
        started_at=result.get("started_at"),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    registered = []
    for path, kind in artifacts:
        if path is not None and Path(path).exists():
            registered.append(catalog.register_artifact(path, kind, job_id=job_id))
    result["job_id"] = job_id
    result["artifacts"] = registered
    result_path = Path(result["result_json_path"])
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    json_artifact = catalog.register_artifact(result_path, "result_json", job_id=job_id)
    result["artifacts"].append(json_artifact)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    catalog.register_artifact(result_path, "result_json", job_id=job_id)


@mcp.tool()
def list_devices(receiver_id: str | None = None) -> dict:
    """Inspect the selected receiver, or the default Airspy HF+ when omitted."""
    return device_info(receiver_id)


@mcp.tool()
def save_station_memory(
    name: str, frequency_hz: int, mode: str, bandwidth_hz: int | None = None,
    notes: str = "", tags: list[str] | None = None, enabled: bool = True,
    memory_id: str | None = None, replace_existing: bool = False,
) -> dict:
    """Save a named receive-only station memory for MCP and dashboard recall."""
    return save_station_memory_record(
        memory_id=memory_id, replace_existing=replace_existing, name=name,
        frequency_hz=frequency_hz, mode=mode, bandwidth_hz=bandwidth_hz,
        notes=notes, tags=tags, enabled=enabled,
    )


@mcp.tool()
def list_station_memories(
    query: str | None = None, mode: str | None = None, enabled_only: bool = False,
) -> dict:
    """List or search named station memories, optionally filtering by mode."""
    memories = list_memories(query=query, mode=mode, enabled_only=enabled_only)
    return {"count": len(memories), "memories": memories}


@mcp.tool()
def get_station_memory(memory_id_or_name: str) -> dict:
    """Get one station memory by stable ID or case-insensitive name."""
    return get_station_memory_record(memory_id_or_name)


@mcp.tool()
def delete_station_memory(memory_id_or_name: str, confirm_delete: bool = False) -> dict:
    """Delete a station memory only when confirm_delete=true."""
    return delete_station_memory_record(memory_id_or_name, confirm_delete=confirm_delete)


@mcp.tool()
def receive_station_memory(
    memory_id_or_name: str, duration_seconds: float = 5.0,
    include_media: bool = True, stereo: bool = True,
    deemphasis_us: int = 75, decode_rds_data: bool = True,
) -> CallToolResult:
    """Receive one saved station using its validated mode, frequency, and bandwidth.

    This is an explicit, short receive action. IQ is never retained. Broadcast FM
    uses its dedicated WFM/RDS path; other memories use the analog signal analyzer.
    """
    memory = get_station_memory_record(memory_id_or_name)
    if not memory["enabled"]:
        raise ValueError("Station memory is disabled")
    duration_seconds = float(duration_seconds)
    if not 0.25 <= duration_seconds <= 10:
        raise ValueError("duration_seconds must be from 0.25 through 10")
    if memory["mode"] == "broadcast_fm":
        if duration_seconds not in {5.0, 10.0}:
            raise ValueError("Broadcast FM station-memory duration must be 5 or 10 seconds")
        response = receive_broadcast_fm(
            frequency_hz=memory["frequency_hz"], duration_seconds=duration_seconds,
            stereo=stereo, deemphasis_us=deemphasis_us,
            decode_rds_data=decode_rds_data, retain_iq=False,
            include_audio=include_media, include_plot=include_media,
        )
    else:
        response = analyze_signal(
            frequency_hz=memory["frequency_hz"], mode=memory["mode"],
            bandwidth_hz=memory["bandwidth_hz"], duration_seconds=duration_seconds,
            retain_iq=False, include_audio=include_media, include_plots=include_media,
        )
    result = dict(response.structuredContent or {})
    result["station_memory"] = memory
    response.structuredContent = result
    return response


@mcp.tool()
def scan_station_memories(
    memory_ids_or_names: list[str] | None = None,
    tag: str | None = None,
    mode: str | None = None,
    duration_seconds: float = 5.0,
    max_memories: int = 10,
    stop_on_error: bool = False,
    stereo: bool = True,
    deemphasis_us: int = 75,
    decode_rds_data: bool = True,
    compare_previous: bool = True,
    snr_change_threshold_db: float = 6.0,
) -> dict:
    """Run one bounded receive round across selected or filtered station memories.

    Child receptions retain their normal jobs and artifacts. This tool suppresses
    inline media and persists a summary linking each memory to its child job.
    """
    duration_seconds = float(duration_seconds)
    if not 0.25 <= duration_seconds <= 10:
        raise ValueError("duration_seconds must be from 0.25 through 10")
    max_memories = int(max_memories)
    if not 1 <= max_memories <= 20:
        raise ValueError("max_memories must be from 1 through 20")
    if duration_seconds * max_memories > 120:
        raise ValueError("Requested scan exceeds the 120-second RF-time limit")
    snr_change_threshold_db = float(snr_change_threshold_db)
    if not 0.5 <= snr_change_threshold_db <= 40:
        raise ValueError("snr_change_threshold_db must be from 0.5 through 40")
    if memory_ids_or_names:
        if len(memory_ids_or_names) > max_memories:
            raise ValueError("memory_ids_or_names contains more than max_memories")
        memories = [get_station_memory_record(value) for value in memory_ids_or_names]
        if tag:
            wanted = tag.strip().casefold()
            memories = [item for item in memories if wanted in item["tags"]]
        if mode:
            memories = [item for item in memories if item["mode"] == mode.strip().lower()]
    else:
        memories = list_memories(mode=mode, enabled_only=True)
        if tag:
            wanted = tag.strip().casefold()
            memories = [item for item in memories if wanted in item["tags"]]
        memories = memories[:max_memories]
    memories = [item for item in memories if item["enabled"]]
    if not memories:
        raise ValueError("No enabled station memories matched the scan request")
    if any(item["mode"] == "broadcast_fm" for item in memories) and duration_seconds not in {5.0, 10.0}:
        raise ValueError("A scan containing Broadcast FM memories must use 5 or 10 seconds")

    scan_id = f"memory-scan-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc).isoformat()
    observations = []
    for memory in memories:
        try:
            response = receive_station_memory(
                memory["memory_id"], duration_seconds=duration_seconds,
                include_media=False, stereo=stereo, deemphasis_us=deemphasis_us,
                decode_rds_data=decode_rds_data,
            )
            child = dict(response.structuredContent or {})
            observations.append({
                "memory_id": memory["memory_id"], "name": memory["name"],
                "frequency_hz": memory["frequency_hz"], "mode": memory["mode"],
                "state": "completed", "job_id": child.get("job_id"),
                "metrics": child.get("metrics"), "rds": child.get("rds"),
            })
        except Exception as exc:
            observations.append({
                "memory_id": memory["memory_id"], "name": memory["name"],
                "frequency_hz": memory["frequency_hz"], "mode": memory["mode"],
                "state": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
            if stop_on_error:
                break
    completed = sum(item["state"] == "completed" for item in observations)
    previous_job_id = None
    changes = []
    if compare_previous:
        prior_jobs = catalog.list_jobs(job_type="station_memory_scan", state="completed", limit=10)
        previous = None
        for prior in prior_jobs:
            full = catalog.get_job(prior["job_id"])
            candidate = full.get("result") or {}
            candidate_observations = candidate.get("observations")
            if (isinstance(candidate_observations, list) and
                {item.get("memory_id") for item in candidate_observations} ==
                {item.get("memory_id") for item in observations}):
                previous = candidate
                previous_job_id = prior["job_id"]
                break
        if previous:
            old_by_id = {item.get("memory_id"): item for item in previous["observations"]}
            for current in observations:
                old = old_by_id.get(current["memory_id"])
                if not old:
                    changes.append({"kind": "new_memory_observation", "memory_id": current["memory_id"],
                                    "name": current["name"], "current_state": current["state"]})
                    continue
                if old.get("state") != current.get("state"):
                    changes.append({"kind": "reception_state_changed", "memory_id": current["memory_id"],
                                    "name": current["name"], "previous_state": old.get("state"),
                                    "current_state": current.get("state")})
                elif current["state"] == "failed":
                    changes.append({"kind": "repeated_failure", "memory_id": current["memory_id"],
                                    "name": current["name"], "current_error": current.get("error")})
                old_snr = (old.get("metrics") or {}).get("estimated_snr_db")
                new_snr = (current.get("metrics") or {}).get("estimated_snr_db")
                if old_snr is not None and new_snr is not None:
                    delta = float(new_snr) - float(old_snr)
                    if abs(delta) >= snr_change_threshold_db:
                        changes.append({"kind": "snr_changed", "memory_id": current["memory_id"],
                                        "name": current["name"], "previous_snr_db": old_snr,
                                        "current_snr_db": new_snr, "delta_db": delta})
                old_station = (old.get("rds") or {}).get("station") or {}
                new_station = (current.get("rds") or {}).get("station") or {}
                for field in ("program_service", "radiotext"):
                    if old_station.get(field) != new_station.get(field) and (
                        old_station.get(field) or new_station.get(field)
                    ):
                        changes.append({"kind": f"rds_{field}_changed",
                                        "memory_id": current["memory_id"], "name": current["name"],
                                        "previous_value": old_station.get(field),
                                        "current_value": new_station.get(field)})
    result = {
        "job_id": scan_id, "state": "completed" if completed == len(observations) else "partial",
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(memories), "attempted_count": len(observations),
        "completed_count": completed, "failed_count": len(observations) - completed,
        "duration_seconds_per_memory": duration_seconds,
        "planned_rf_time_seconds": duration_seconds * len(memories),
        "compared_to_job_id": previous_job_id, "change_count": len(changes),
        "changes": changes,
        "observations": observations,
    }
    ensure_data_dirs()
    result_path = RESULT_DIR / f"{scan_id}.json"
    result["result_json_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _persist_one_shot(
        job_id=scan_id, job_type="station_memory_scan", result=result,
        config={"memory_ids": [item["memory_id"] for item in memories], "tag": tag,
                "mode": mode, "duration_seconds": duration_seconds,
                "max_memories": max_memories, "stop_on_error": stop_on_error,
                "compare_previous": compare_previous,
                "snr_change_threshold_db": snr_change_threshold_db},
        artifacts=[],
    )
    return result


@mcp.tool()
def save_station_memory_scan_profile(
    name: str, memory_ids_or_names: list[str] | None = None,
    tag: str | None = None, mode: str | None = None,
    duration_seconds: float = 5.0, max_memories: int = 10,
    stop_on_error: bool = False, compare_previous: bool = True,
    snr_change_threshold_db: float = 6.0, stereo: bool = True,
    deemphasis_us: int = 75, decode_rds_data: bool = True,
    description: str = "", replace_existing: bool = False,
) -> dict:
    """Save a schedulable, bounded station-memory monitoring profile."""
    normalized_name, preset_type, normalized_description, config = normalize_preset(
        name=name, preset_type="station_memory_scan", description=description,
        config={"memory_ids_or_names": memory_ids_or_names, "tag": tag, "mode": mode,
                "duration_seconds": duration_seconds, "max_memories": max_memories,
                "stop_on_error": stop_on_error, "compare_previous": compare_previous,
                "snr_change_threshold_db": snr_change_threshold_db, "stereo": stereo,
                "deemphasis_us": deemphasis_us, "decode_rds_data": decode_rds_data},
    )
    profile = catalog.save_preset(
        name=normalized_name, preset_type=preset_type,
        description=normalized_description, config=config,
        replace_existing=replace_existing,
    )
    return {**profile, "next_steps": ["run_rf_preset", "save_rf_schedule"]}


@mcp.tool()
def discover_sdr_receivers(probe_hardware: bool = False) -> dict:
    """Find installed SDR backends; optionally run bounded read-only hardware probes.

    The safe default checks executable availability only. It neither tunes a receiver
    nor creates registry entries. Set probe_hardware=true only when idle.
    """
    return discover_backends(probe_hardware=probe_hardware)


@mcp.tool()
def discover_attached_sdr_devices() -> dict:
    """Probe Airspy HF+ and RTL-SDR hardware and return registration-ready devices."""
    return receiver_service.discover()


@mcp.tool()
def add_discovered_sdr_device(
    backend: str, device_selector: str, receiver_id: str, name: str,
    role: str, priority: int = 80,
) -> dict:
    """Verify and register one receiver returned by discover_attached_sdr_devices."""
    return receiver_service.register(
        backend=backend, device_selector=device_selector, receiver_id=receiver_id,
        name=name, role=role, priority=priority,
    )


@mcp.tool()
def save_sdr_receiver(
    receiver_id: str, name: str, backend: str, role: str = "general",
    device_selector: str = "", enabled: bool = True, verified: bool = False,
    tuning_ranges_hz: list[list[int]] | None = None,
    max_bandwidth_hz: int | None = None, priority: int = 50, notes: str = "",
) -> dict:
    """Create or replace a receiver definition used by the v0.41 coordinator."""
    return save_receiver(receiver_id=receiver_id, name=name, backend=backend, role=role,
                         device_selector=device_selector, enabled=enabled, verified=verified,
                         tuning_ranges_hz=tuning_ranges_hz,
                         max_bandwidth_hz=max_bandwidth_hz, priority=priority, notes=notes)


@mcp.tool()
def list_sdr_receivers() -> dict:
    """List configured SDR receivers, capabilities, roles, and active leases."""
    ensure_airspy_default()
    receivers = list_receivers()
    return {"count": len(receivers), "receivers": receivers}


@mcp.tool()
def get_sdr_receiver(receiver_id: str) -> dict:
    """Get one configured SDR receiver by its stable ID."""
    ensure_airspy_default()
    return get_receiver(receiver_id)


@mcp.tool()
def delete_sdr_receiver(receiver_id: str, confirm_delete: bool = False) -> dict:
    """Delete a receiver definition. Requires confirm_delete=true and no active lease."""
    return delete_receiver(receiver_id, confirm_delete=confirm_delete)


@mcp.tool()
def plan_sdr_assignment(
    frequency_hz: int, required_bandwidth_hz: int = 0,
    preferred_role: str | None = None, require_verified: bool = True,
) -> dict:
    """Dry-run deterministic receiver selection without tuning or claiming hardware."""
    ensure_airspy_default()
    return plan_assignment(frequency_hz=frequency_hz,
                           required_bandwidth_hz=required_bandwidth_hz,
                           preferred_role=preferred_role, require_verified=require_verified)


@mcp.tool()
def acquire_sdr_receiver(receiver_id: str, owner: str, purpose: str = "") -> dict:
    """Claim one configured receiver for cooperative external work until released."""
    ensure_airspy_default()
    return acquire_receiver(receiver_id, owner, purpose)


@mcp.tool()
def release_sdr_receiver(lease_id: str) -> dict:
    """Release a process-local v0.41 receiver lease by lease ID."""
    return release_receiver(lease_id)


@mcp.tool()
def get_sdr_coordinator_status() -> dict:
    """Report receiver inventory and process-local per-device leases."""
    ensure_airspy_default()
    return coordinator_status()


@mcp.tool()
def get_live_audio_capabilities() -> dict:
    """Describe the authenticated HTTP live-media channel (MCP carries no audio)."""
    return live_audio_manager.capabilities()


@mcp.tool()
def list_live_audio_sessions() -> dict:
    """List sanitized live session metadata; no tokens, client identities, IQ or audio."""
    return live_audio_manager.status()


@mcp.tool()
def stop_live_audio_session(session_id: str) -> dict:
    """Stop a live session; listeners consume media through GET /api/live-audio."""
    return live_audio_manager.stop(session_id)


@mcp.tool()
def get_rf_recovery_status() -> dict:
    """Report durable schema, lease, and restart-recovery status without using RF hardware."""
    services = RfApplicationServices(
        catalog=catalog, receivers=receiver_service,
        spectrum_capture=inspect_spectrum, signal_analyzer=analyze_signal,
        broadcast_fm_receiver=receive_broadcast_fm, live_audio=live_audio_manager,
        live_waterfall=live_waterfall_manager,
    )
    return services.recovery_status(_INTERRUPTED_JOBS_ON_STARTUP)


@mcp.tool()
def save_receiver_calibration(
    receiver_id: str, frequency_correction_ppm: float = 0,
    dbfs_to_dbm_offset_db: float | None = None,
    reference_frequency_hz: int | None = None, reference_source: str = "",
    notes: str = "", replace_existing: bool = False,
) -> dict:
    """Save traceable frequency and optional input-power calibration for one receiver."""
    return save_calibration(
        receiver_id=receiver_id, frequency_correction_ppm=frequency_correction_ppm,
        dbfs_to_dbm_offset_db=dbfs_to_dbm_offset_db,
        reference_frequency_hz=reference_frequency_hz,
        reference_source=reference_source, notes=notes,
        replace_existing=replace_existing,
    )


@mcp.tool()
def get_receiver_calibration(receiver_id: str) -> dict:
    """Get the active calibration profile for one receiver."""
    return get_calibration(receiver_id)


@mcp.tool()
def list_receiver_calibrations() -> dict:
    """List persistent receiver calibration profiles and their provenance."""
    values = list_calibrations()
    return {"count": len(values), "calibrations": values}


@mcp.tool()
def delete_receiver_calibration(
    receiver_id: str, confirm_delete: bool = False,
) -> dict:
    """Delete one calibration profile only when confirm_delete=true."""
    return delete_calibration(receiver_id, confirm_delete=confirm_delete)


@mcp.tool()
def qualify_sdr_receiver(
    receiver_id: str, test_frequency_hz: int, duration_seconds: float = 0.25,
) -> dict:
    """Perform a short IQ capture and report receiver readiness and digital-level health."""
    info = device_info(receiver_id)
    capture = capture_iq(
        test_frequency_hz, duration_seconds, receiver_id=receiver_id,
        purpose="receiver qualification",
    )
    try:
        iq = load_complex_float32(capture.path)
        levels = iq_level_metrics(iq)
        passed = bool(
            capture.captured_samples >= capture.requested_samples * 0.98
            and not levels["overload_suspected"]
        )
        return {
            "qualified": passed, "receiver_id": receiver_id,
            "backend": capture.backend, "device": info,
            "test_frequency_hz": int(test_frequency_hz),
            "sample_rate_hz": capture.sample_rate_hz,
            "requested_samples": capture.requested_samples,
            "captured_samples": capture.captured_samples,
            "digital_levels": levels, "calibration": capture.calibration,
            "checks": {
                "device_probe": True,
                "capture_length": capture.captured_samples >= capture.requested_samples * 0.98,
                "not_clipping": not levels["overload_suspected"],
            },
            "notice": "Qualification verifies operation, not traceable RF calibration.",
        }
    finally:
        Path(capture.path).unlink(missing_ok=True)


@mcp.tool()
def save_rf_preset(
    name: str,
    preset_type: str,
    config: dict,
    description: str = "",
    replace_existing: bool = False,
) -> dict:
    """Save a validated scan, survey, monitor, watchlist, or SSTV preset.

    Names are unique without regard to case. Replacing an existing name requires
    replace_existing=true and preserves its stable preset_id.
    """
    name, preset_type, description, normalized = normalize_preset(
        name=name,
        preset_type=preset_type,
        description=description,
        config=config,
    )
    return catalog.save_preset(
        name=name,
        preset_type=preset_type,
        description=description,
        config=normalized,
        replace_existing=replace_existing,
    )


@mcp.tool()
def list_rf_presets(preset_type: str | None = None, limit: int = 100) -> dict:
    """List persistent named RF presets, optionally filtered by type."""
    if preset_type is not None:
        preset_type = preset_type.strip().lower()
        if preset_type not in PRESET_TYPES:
            raise ValueError(f"preset_type must be one of: {', '.join(PRESET_TYPES)}")
    presets = catalog.list_presets(preset_type=preset_type, limit=limit)
    return {"count": len(presets), "presets": presets}


@mcp.tool()
def get_rf_preset(preset_id_or_name: str) -> dict:
    """Get one persistent RF preset by stable ID or case-insensitive name."""
    return catalog.get_preset(preset_id_or_name)


@mcp.tool()
def delete_rf_preset(preset_id_or_name: str, confirm_delete: bool = False) -> dict:
    """Delete a named RF preset only when confirm_delete=true."""
    if not confirm_delete:
        raise ValueError("Preset deletion requires confirm_delete=true")
    deleted = catalog.delete_preset(preset_id_or_name)
    return {"deleted": True, "preset": deleted}


@mcp.tool()
def list_rf_activity_bands() -> dict:
    """List Inspector-friendly HF/VHF bands available for activity profiles."""
    return {"count": len(RF_BANDS), "bands": [
        {"band_name": name, **definition} for name, definition in RF_BANDS.items()
    ], "note": "Definitions are convenient survey ranges, not regulatory advice."}


@mcp.tool()
def save_rf_activity_profile(
    name: str, band_name: str = "20m", start_frequency_hz: int | None = None,
    stop_frequency_hz: int | None = None, description: str = "",
    capture_duration_seconds: float = 1.0, threshold_above_noise_db: float = 8.0,
    minimum_signal_spacing_hz: float = 1_000, attenuation_steps: int = 1,
    classify_top_signals: int = 5, replace_existing: bool = False,
) -> dict:
    """Save a schedulable activity survey using a named band or custom endpoints."""
    band_name = str(band_name).strip().lower()
    if (start_frequency_hz is None) != (stop_frequency_hz is None):
        raise ValueError("Provide both custom frequency endpoints or neither")
    if start_frequency_hz is None:
        if band_name not in RF_BANDS:
            raise ValueError("band_name must be one of: " + ", ".join(RF_BANDS))
        definition = RF_BANDS[band_name]
        start_frequency_hz, stop_frequency_hz = definition["start_hz"], definition["stop_hz"]
    config = {
        "start_frequency_hz": int(start_frequency_hz),
        "stop_frequency_hz": int(stop_frequency_hz),
        "capture_duration_seconds": capture_duration_seconds,
        "overlap_fraction": 0.15, "fft_size": 8192,
        "threshold_above_noise_db": threshold_above_noise_db,
        "minimum_signal_spacing_hz": minimum_signal_spacing_hz,
        "attenuation_steps": attenuation_steps, "max_signals": 200,
        "classify_top_signals": classify_top_signals,
        "classification_duration_seconds": 2.0,
        "classification_bandwidth_hz": 30_000,
    }
    normalized_name, preset_type, normalized_description, normalized = normalize_preset(
        name=name, preset_type="activity_monitor", description=description,
        config=config,
    )
    profile = catalog.save_preset(
        name=normalized_name, preset_type=preset_type,
        description=normalized_description, config=normalized,
        replace_existing=replace_existing,
    )
    named = RF_BANDS.get(band_name, {})
    selected_band = (band_name if start_frequency_hz == named.get("start_hz")
                     and stop_frequency_hz == named.get("stop_hz") else "custom")
    return {**profile, "band_name": selected_band,
            "next_steps": ["run_rf_preset", "save_rf_schedule", "get_rf_activity_dashboard"]}


@mcp.tool()
def get_rf_activity_dashboard(
    profile_id_or_name: str, run_limit: int = 24,
    frequency_tolerance_hz: float = 1_500, noise_anomaly_db: float = 6.0,
    occupancy_anomaly_percent: float = 5.0, include_plot: bool = True,
    create_exports: bool = True,
) -> CallToolResult:
    """Summarize longitudinal activity, baselines, anomalies, and recurring signals."""
    preset = catalog.get_preset(profile_id_or_name)
    if preset["preset_type"] != "activity_monitor":
        raise ValueError("profile_id_or_name must identify an activity_monitor preset")
    if not 0 <= float(occupancy_anomaly_percent) <= 100:
        raise ValueError("occupancy_anomaly_percent must be from 0 through 100")
    summary, runs = build_activity_dashboard(
        catalog, preset, run_limit=run_limit,
        frequency_tolerance_hz=frequency_tolerance_hz,
        noise_anomaly_db=noise_anomaly_db,
        occupancy_anomaly_fraction=float(occupancy_anomaly_percent) / 100,
    )
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(summary, indent=2))
    ]
    artifacts = []
    if include_plot:
        plot_path = save_activity_plot(summary, runs)
        artifact = catalog.register_artifact(plot_path, "rf_activity_heatmap")
        artifacts.append(artifact)
        content.append(ImageContent(type="image",
                                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                                    mimeType="image/png"))
    if create_exports:
        json_path, csv_path = save_activity_exports(summary)
        artifacts.extend([
            catalog.register_artifact(json_path, "rf_activity_json"),
            catalog.register_artifact(csv_path, "rf_activity_csv"),
        ])
    summary["artifacts"] = artifacts
    return CallToolResult(content=content, structuredContent=summary)


@mcp.tool()
def save_hf_time_station_watchlist(
    name: str = "HF Time Stations", duration_seconds: float = 2.0,
    replace_existing: bool = False,
) -> dict:
    """Save WWV and CHU frequencies as a schedulable propagation-evidence watchlist."""
    entries = [
        {"frequency_hz": 2_500_000, "label": "WWV 2.5 MHz"},
        {"frequency_hz": 5_000_000, "label": "WWV 5 MHz"},
        {"frequency_hz": 10_000_000, "label": "WWV 10 MHz"},
        {"frequency_hz": 15_000_000, "label": "WWV 15 MHz"},
        {"frequency_hz": 20_000_000, "label": "WWV 20 MHz"},
        {"frequency_hz": 3_330_000, "label": "CHU 3.33 MHz"},
        {"frequency_hz": 7_850_000, "label": "CHU 7.85 MHz"},
        {"frequency_hz": 14_670_000, "label": "CHU 14.67 MHz"},
    ]
    normalized_name, preset_type, description, config = normalize_preset(
        name=name, preset_type="watchlist",
        description="WWV/CHU checks for local HF propagation evidence",
        config={"entries": entries, "duration_seconds": duration_seconds,
                "analysis_bandwidth_hz": 10_000, "fft_size": 8192},
    )
    return catalog.save_preset(
        name=normalized_name, preset_type=preset_type, description=description,
        config=config, replace_existing=replace_existing,
    )


@mcp.tool()
def get_space_weather_snapshot(force_refresh: bool = False) -> dict:
    """Get cached current NOAA SWPC solar flux, Kp, and R/S/G scales."""
    if not isinstance(force_refresh, bool):
        raise ValueError("force_refresh must be a JSON boolean")
    snapshot = fetch_space_weather(force_refresh=force_refresh)
    snapshot["hf_interpretation"] = space_weather_interpretation(snapshot)
    return snapshot


def _propagation_local_inputs(hours: float) -> tuple[list[dict], dict[str, dict], list[dict]]:
    cutoff = datetime.now(timezone.utc).timestamp() - float(hours) * 3600
    spots = [item for item in catalog.list_weak_signal_spots(limit=1000)
             if datetime.fromisoformat(item["captured_at"]).timestamp() >= cutoff]
    activity = {}
    for preset in catalog.list_presets(preset_type="activity_monitor", limit=100):
        config = preset["config"]
        band_name = next((name for name, definition in RF_BANDS.items()
                          if config["start_frequency_hz"] == definition["start_hz"]
                          and config["stop_frequency_hz"] == definition["stop_hz"]), None)
        if band_name is None or RF_BANDS[band_name]["stop_hz"] > 31_000_000:
            continue
        try:
            dashboard, _ = build_activity_dashboard(catalog, preset, run_limit=24)
        except ValueError:
            continue
        if datetime.fromisoformat(dashboard["latest_run"]["created_at"]).timestamp() >= cutoff:
            activity[band_name] = dashboard
    stations = []
    for job in catalog.list_jobs(job_type="watchlist_run", state="completed", limit=100):
        if datetime.fromisoformat(job["created_at"]).timestamp() < cutoff:
            continue
        full = catalog.get_job(job["job_id"])
        result = full.get("result") or {}
        for observation in result.get("observations") or []:
            label = str(observation.get("label") or "")
            if not label.upper().startswith(("WWV", "CHU")):
                continue
            confidence = observation.get("best_confidence")
            stations.append({
                "job_id": job["job_id"], "observed_at": job["created_at"],
                "label": label, "frequency_hz": observation.get("frequency_hz"),
                "classification": observation.get("best_label"),
                "confidence": confidence,
                "detected": observation.get("status") == "completed"
                            and confidence is not None and float(confidence) >= 0.30,
                "ambiguous": observation.get("ambiguous"),
            })
    return spots, activity, stations


@mcp.tool()
def get_hf_propagation_report(
    lookback_hours: float = 24.0, include_space_weather: bool = True,
    include_plot: bool = True, create_exports: bool = True,
) -> CallToolResult:
    """Combine locally observed HF evidence with optional NOAA space-weather context."""
    lookback_hours = float(lookback_hours)
    if not 0.25 <= lookback_hours <= 720:
        raise ValueError("lookback_hours must be from 0.25 through 720")
    spots, activity, stations = _propagation_local_inputs(lookback_hours)
    local = summarize_local_propagation(
        spots=spots, activity=activity, time_stations=stations, hours=lookback_hours,
    )
    weather = fetch_space_weather() if include_space_weather else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "server_name": "MiniRackDisplay", "local_evidence": local,
        "space_weather": weather,
        "space_weather_interpretation": space_weather_interpretation(weather) if weather else [],
        "summary": {
            "best_observed_bands": [item["band_name"] for item in local["bands"]
                                    if item["evidence_rating"] in {
                                        "strong_local_evidence", "moderate_local_evidence"}],
            "local_data_available": bool(spots or activity or stations),
            "space_weather_available": bool(
                weather and len(weather.get("errors", {})) < 3
            ),
        },
        "limitations": [
            "This is a station-observation report, not a path-specific ionospheric prediction.",
            "No recent evidence does not prove a band is closed.",
            "Digital levels are relative and not calibrated antenna-input power.",
            "NOAA context does not override what MiniRackDisplay actually received.",
        ],
    }
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(report, indent=2))
    ]
    artifacts = []
    if include_plot:
        path = save_propagation_plot(report)
        artifacts.append(catalog.register_artifact(path, "hf_propagation_plot"))
        content.append(ImageContent(type="image",
                                    data=base64.b64encode(path.read_bytes()).decode("ascii"),
                                    mimeType="image/png"))
    if create_exports:
        json_path, csv_path = save_propagation_exports(report)
        artifacts.extend([catalog.register_artifact(json_path, "hf_propagation_json"),
                          catalog.register_artifact(csv_path, "hf_propagation_csv")])
    report["artifacts"] = artifacts
    return CallToolResult(content=content, structuredContent=report)


def _run_watchlist_preset(
    preset: dict, source_schedule_id: str | None = None
) -> CallToolResult:
    config = preset["config"]
    enabled = [item for item in config["entries"] if item["enabled"]]
    ensure_data_dirs()
    started_at = datetime.now(timezone.utc).isoformat()
    job_id = f"watch-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    acquire_long_job(job_id)
    observations = []
    try:
        for entry in enabled:
            try:
                response = classify_signal(
                    frequency_hz=entry["frequency_hz"],
                    duration_seconds=config["duration_seconds"],
                    analysis_bandwidth_hz=config["analysis_bandwidth_hz"],
                    fft_size=config["fft_size"],
                    include_plot=False,
                    include_preview_audio=False,
                    retain_iq=False,
                )
                classification = dict(response.structuredContent or {})
                observations.append(
                    {
                        "label": entry["label"],
                        "notes": entry["notes"],
                        "frequency_hz": entry["frequency_hz"],
                        "status": "completed",
                        "classification_job_id": classification.get("job_id"),
                        "best_label": classification.get("best_label"),
                        "best_confidence": classification.get("best_confidence"),
                        "confidence_margin": classification.get("confidence_margin"),
                        "ambiguous": classification.get("ambiguous"),
                        "ranking": classification.get("ranking"),
                        "features": classification.get("features"),
                        "classification_plot_path": classification.get(
                            "classification_plot_path"
                        ),
                    }
                )
            except Exception as exc:
                observations.append(
                    {
                        "label": entry["label"],
                        "notes": entry["notes"],
                        "frequency_hz": entry["frequency_hz"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        release_long_job(job_id)

    completed = sum(item["status"] == "completed" for item in observations)
    result = {
        "job_id": job_id,
        "preset_id": preset["preset_id"],
        "preset_name": preset["name"],
        "preset_type": "watchlist",
        "source_schedule_id": source_schedule_id,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(enabled),
        "completed_count": completed,
        "failed_count": len(observations) - completed,
        "observations": observations,
    }
    result_path = RESULT_DIR / f"{job_id}.json"
    result["result_json_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _persist_one_shot(
        job_id=job_id,
        job_type="watchlist_run",
        result=result,
        config={
            "source_preset_id": preset["preset_id"],
            "source_preset_name": preset["name"],
            "source_schedule_id": source_schedule_id,
            **config,
        },
        artifacts=[],
    )
    summary = (
        f"Watchlist preset {preset['name']} inspected {len(enabled)} enabled frequencies: "
        f"{completed} completed and {len(observations) - completed} failed. "
        f"Classifications are heuristic, not authoritative.\n\n"
        + json.dumps(result, indent=2)
    )
    return CallToolResult(
        content=[TextContent(type="text", text=summary)],
        structuredContent=result,
    )


def _execute_rf_preset(
    preset: dict, source_schedule_id: str | None = None
) -> CallToolResult:
    config = dict(preset["config"])
    preset_type = preset["preset_type"]
    if preset_type == "band_scan":
        launched = scan_manager.start(
            **config,
            source_preset_id=preset["preset_id"],
            source_schedule_id=source_schedule_id,
        )
    elif preset_type in {"band_survey", "activity_monitor"}:
        launched = scan_manager.start(
            **config,
            source_preset_id=preset["preset_id"],
            source_schedule_id=source_schedule_id,
        )
    elif preset_type == "monitor":
        launched = monitor_manager.start(
            **config,
            source_preset_id=preset["preset_id"],
            source_schedule_id=source_schedule_id,
        )
    elif preset_type == "sstv":
        launched = sstv_manager.start(
            **config,
            source_preset_id=preset["preset_id"],
            source_schedule_id=source_schedule_id,
        )
    elif preset_type == "sstv_watch":
        launched = sstv_watcher_manager.start(
            **config,
            source_preset_id=preset["preset_id"],
            source_schedule_id=source_schedule_id,
        )
    elif preset_type == "station_memory_scan":
        completed = scan_station_memories(**config)
        completed.update({"source_preset_id": preset["preset_id"],
                          "source_preset_name": preset["name"],
                          "source_preset_type": preset_type,
                          "source_schedule_id": source_schedule_id})
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(completed, indent=2))],
            structuredContent=completed,
        )
    else:
        return _run_watchlist_preset(preset, source_schedule_id)
    result = {
        **launched,
        "source_preset_id": preset["preset_id"],
        "source_preset_name": preset["name"],
        "source_preset_type": preset_type,
        "source_schedule_id": source_schedule_id,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2))],
        structuredContent=result,
    )


@mcp.tool()
def run_rf_preset(preset_id_or_name: str) -> CallToolResult:
    """Run a saved RF preset by stable ID or case-insensitive name."""
    return _execute_rf_preset(catalog.get_preset(preset_id_or_name))


alert_evaluator = AlertEvaluator(catalog)


def _launch_scheduled_preset(preset_id: str, schedule_id: str | None = None) -> dict:
    response = _execute_rf_preset(catalog.get_preset(preset_id), schedule_id)
    result = dict(response.structuredContent or {})
    if schedule_id and result.get("preset_type") == "watchlist":
        events = alert_evaluator.evaluate_watchlist(schedule_id, result)
        result["alert_event_count"] = len(events)
        result["alert_event_ids"] = [event["event_id"] for event in events]
    return result


scheduler_manager = SchedulerManager(
    catalog,
    _launch_scheduled_preset,
    lambda: active_long_job() is not None,
)
webhook_dispatcher = WebhookDispatcher(catalog)


def _launch_satellite_pass(*, watch: dict, pass_record: dict,
                           duration_seconds: float) -> dict:
    downlink = pass_record.get("selected_downlink") or watch["downlinks"][0]
    if downlink["mode"] == "sstv":
        return sstv_watcher_manager.start(
            frequency_hz=downlink["frequency_hz"],
            receiver_mode=downlink["receiver_mode"],
            watch_duration_seconds=duration_seconds, rearm=True,
            retain_audio=downlink.get("retain_audio", True), deduplicate=True,
            source_satellite_watch_id=watch["watch_id"],
            source_satellite_pass_id=pass_record["pass_id"],
            doppler_correction_mode=watch.get("doppler_correction_mode", "off"),
            doppler_plan=pass_record.get("doppler_plan", []),
        )
    return satellite_receiver_manager.start(
        watch=watch, pass_record=pass_record, downlink=downlink,
        duration_seconds=duration_seconds,
    )


satellite_scheduler = SatellitePassScheduler(
    catalog,
    _launch_satellite_pass,
    lambda: active_long_job() is not None,
)


@mcp.tool()
def save_satellite_sstv_watch(
    name: str,
    satellite_name: str,
    norad_id: int,
    tle_line1: str,
    tle_line2: str,
    latitude_deg: str,
    longitude_deg: str,
    elevation_m: float = 0.0,
    frequency_hz: int = 145_800_000,
    receiver_mode: str = "nfm",
    minimum_elevation_deg: float = 10.0,
    lead_seconds: int = 60,
    trail_seconds: int = 30,
    notify_before_seconds: int = 600,
    tle_source: str = "manual",
    auto_refresh: bool = False,
    refresh_interval_seconds: int = 86400,
    doppler_correction_mode: str = "off",
    doppler_step_seconds: int = 10,
    enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Save a TLE-backed satellite pass watch for automatic SSTV reception."""
    normalized = normalize_satellite_watch(
        name=name, satellite_name=satellite_name, norad_id=norad_id,
        tle_line1=tle_line1, tle_line2=tle_line2, latitude_deg=latitude_deg,
        longitude_deg=longitude_deg, elevation_m=elevation_m,
        frequency_hz=frequency_hz, receiver_mode=receiver_mode,
        minimum_elevation_deg=minimum_elevation_deg, lead_seconds=lead_seconds,
        trail_seconds=trail_seconds, notify_before_seconds=notify_before_seconds,
        tle_source=tle_source, auto_refresh=auto_refresh,
        refresh_interval_seconds=refresh_interval_seconds,
        doppler_correction_mode=doppler_correction_mode,
        doppler_step_seconds=doppler_step_seconds, enabled=enabled,
    )
    watch = catalog.save_satellite_watch(
        replace_existing=replace_existing, **normalized
    )
    if watch["enabled"]:
        satellite_scheduler.refresh()
    return watch


@mcp.tool()
def save_satellite_receive_profile(
    name: str, satellite_name: str, norad_id: int, tle_line1: str, tle_line2: str,
    latitude_deg: str, longitude_deg: str, downlinks: list[dict],
    elevation_m: float = 0.0, minimum_elevation_deg: float = 10.0,
    lead_seconds: int = 60, trail_seconds: int = 30, notify_before_seconds: int = 600,
    tle_source: str = "manual", auto_refresh: bool = False,
    refresh_interval_seconds: int = 86400, doppler_correction_mode: str = "digital",
    doppler_step_seconds: int = 10, downlink_selection_policy: str = "priority",
    enabled: bool = True, replace_existing: bool = False,
) -> dict:
    """Save a multi-downlink satellite receive profile.

    One Airspy is allocated to one downlink per pass. Supported downlink modes are
    sstv, nfm_audio, ax25_afsk1200, ax25_g3ruh9600, and capture_only.
    """
    if not downlinks:
        raise ValueError("downlinks must contain at least one downlink")
    first = downlinks[0]
    normalized = normalize_satellite_watch(
        name=name, satellite_name=satellite_name, norad_id=norad_id,
        tle_line1=tle_line1, tle_line2=tle_line2, latitude_deg=latitude_deg,
        longitude_deg=longitude_deg, elevation_m=elevation_m,
        frequency_hz=first["frequency_hz"], receiver_mode=first.get("receiver_mode", "nfm"),
        minimum_elevation_deg=minimum_elevation_deg, lead_seconds=lead_seconds,
        trail_seconds=trail_seconds, notify_before_seconds=notify_before_seconds,
        tle_source=tle_source, auto_refresh=auto_refresh,
        refresh_interval_seconds=refresh_interval_seconds,
        doppler_correction_mode=doppler_correction_mode,
        doppler_step_seconds=doppler_step_seconds, downlinks=downlinks,
        downlink_selection_policy=downlink_selection_policy, enabled=enabled,
    )
    profile = catalog.save_satellite_watch(replace_existing=replace_existing, **normalized)
    if profile["enabled"]:
        satellite_scheduler.refresh()
    return profile


@mcp.tool()
def list_satellite_catalog_categories() -> dict:
    """List supported CelesTrak categories for assisted satellite discovery."""
    return {"categories": [{"category": key, "celestrak_group": value}
                           for key, value in CATEGORY_GROUPS.items()]}


@mcp.tool()
def search_satellite_catalog(
    query: str | None = None, category: str | None = None, limit: int = 50,
) -> dict:
    """Search current CelesTrak satellites by partial name or browse a category."""
    return search_catalog(query=query, category=category, limit=limit)


@mcp.tool()
def get_satellite_catalog_entry(satellite_name_or_norad_id: str) -> dict:
    """Preview current TLE, public transmitters, compatibility, and suggested downlinks."""
    return get_catalog_entry(satellite_name_or_norad_id)


@mcp.tool()
def create_satellite_receive_profile_from_catalog(
    satellite_name_or_norad_id: str, latitude_deg: str, longitude_deg: str,
    name: str | None = None, elevation_m: float = 0.0,
    transmitter_ids: list[str] | None = None,
    minimum_elevation_deg: float = 10.0, lead_seconds: int = 60,
    trail_seconds: int = 30, notify_before_seconds: int = 600,
    doppler_correction_mode: str = "digital", doppler_step_seconds: int = 10,
    downlink_selection_policy: str = "priority", enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Create a reviewed receive profile using current CelesTrak and SatNOGS metadata.

    Latitude and longitude are text fields so MCP Inspector accepts decimal degrees
    and DMS such as "33.96" or "33 57 36 N". Omit transmitter_ids to use all
    compatible suggested downlinks, or pass IDs returned by get_satellite_catalog_entry.
    """
    entry = get_catalog_entry(satellite_name_or_norad_id)
    downlinks = selected_downlinks(entry, transmitter_ids)
    normalized = normalize_satellite_watch(
        name=name or f"{entry['name']} receiver", satellite_name=entry["name"],
        norad_id=entry["norad_id"], tle_line1=entry["tle_line1"],
        tle_line2=entry["tle_line2"], latitude_deg=latitude_deg,
        longitude_deg=longitude_deg, elevation_m=elevation_m,
        frequency_hz=downlinks[0]["frequency_hz"],
        receiver_mode=downlinks[0]["receiver_mode"],
        minimum_elevation_deg=minimum_elevation_deg, lead_seconds=lead_seconds,
        trail_seconds=trail_seconds, notify_before_seconds=notify_before_seconds,
        tle_source="celestrak", auto_refresh=True, refresh_interval_seconds=86400,
        doppler_correction_mode=doppler_correction_mode,
        doppler_step_seconds=doppler_step_seconds, downlinks=downlinks,
        downlink_selection_policy=downlink_selection_policy, enabled=enabled,
    )
    profile = catalog.save_satellite_watch(replace_existing=replace_existing, **normalized)
    if profile["enabled"]:
        satellite_scheduler.refresh()
    return {**profile, "catalog_sources": entry["sources"],
            "catalog_review_warning": entry["warning"]}


@mcp.tool()
def save_observer_location(
    name: str, latitude_deg: str, longitude_deg: str, elevation_m: float = 0.0,
    make_default: bool = True, replace_existing: bool = False,
) -> dict:
    """Save a reusable station location; decimal degrees and DMS text are accepted."""
    return save_location(name=name, latitude_deg=latitude_deg, longitude_deg=longitude_deg,
                         elevation_m=elevation_m, make_default=make_default,
                         replace_existing=replace_existing)


@mcp.tool()
def list_observer_locations() -> dict:
    """List reusable observer locations, with the default first."""
    locations = list_locations()
    return {"count": len(locations), "locations": locations}


@mcp.tool()
def delete_observer_location(location_id_or_name: str, confirm_delete: bool = False) -> dict:
    """Delete a saved observer location; confirm_delete=true is required."""
    if not confirm_delete:
        raise ValueError("Deleting an observer location requires confirm_delete=true")
    return delete_location(location_id_or_name)


@mcp.tool()
def plan_satellite_observations(
    location_id_or_name: str | None = None, query: str | None = None,
    category: str | None = "amateur", hours: float = 24.0,
    minimum_elevation_deg: float = 10.0, candidate_limit: int = 20,
    result_limit: int = 8,
) -> dict:
    """Rank upcoming visible satellites that have Airspy-compatible downlinks.

    Omit location_id_or_name to use the default saved location. Use either a
    partial query or a CelesTrak category. This previews opportunities only.
    """
    if query and category == "amateur":
        category = None
    return plan_observations(
        location=get_location(location_id_or_name), query=query, category=category,
        hours=hours, minimum_elevation_deg=minimum_elevation_deg,
        candidate_limit=candidate_limit, result_limit=result_limit,
    )


@mcp.tool()
def create_satellite_receive_profile_at_saved_location(
    satellite_name_or_norad_id: str, location_id_or_name: str | None = None,
    name: str | None = None, transmitter_ids: list[str] | None = None,
    minimum_elevation_deg: float = 10.0, enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Create a catalog profile using a saved station location."""
    location = get_location(location_id_or_name)
    return create_satellite_receive_profile_from_catalog(
        satellite_name_or_norad_id=satellite_name_or_norad_id,
        latitude_deg=str(location["latitude_deg"]),
        longitude_deg=str(location["longitude_deg"]), elevation_m=location["elevation_m"],
        name=name, transmitter_ids=transmitter_ids,
        minimum_elevation_deg=minimum_elevation_deg, enabled=enabled,
        replace_existing=replace_existing,
    )


@mcp.tool()
def list_satellite_receive_profiles(enabled: bool | None = None,
                                    limit: int = 100) -> dict:
    """List persistent multi-downlink satellite receive profiles."""
    profiles = catalog.list_satellite_watches(enabled=enabled, limit=limit)
    return {"count": len(profiles), "profiles": profiles}


@mcp.tool()
def get_satellite_receive_profile(watch_id_or_name: str) -> dict:
    """Get one satellite receive profile by ID or case-insensitive name."""
    return catalog.get_satellite_watch(watch_id_or_name)


@mcp.tool()
def list_satellite_observations(
    watch_id_or_name: str | None = None, mode: str | None = None,
    outcome: str | None = None, limit: int = 100,
) -> dict:
    """List pass-grouped satellite audio, AX.25, and capture observations."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    observations = catalog.list_satellite_observations(
        watch_id=watch_id, mode=mode, outcome=outcome, limit=limit
    )
    return {"count": len(observations), "observations": observations}


@mcp.tool()
def get_satellite_observation(observation_id: str) -> dict:
    """Get one persisted satellite observation."""
    return catalog.get_satellite_observation(observation_id)


@mcp.tool()
def get_satellite_activity(watch_id_or_name: str | None = None) -> dict:
    """Summarize satellite observations by mode and downlink."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    return catalog.satellite_activity_summary(watch_id=watch_id)


@mcp.tool()
def get_satellite_pass_performance(pass_id: str) -> dict:
    """Score one attempted pass using reception, packet, FCS, and telemetry evidence."""
    pass_record = catalog.get_satellite_pass(pass_id)
    return build_pass_report(catalog, pass_record)


@mcp.tool()
def summarize_satellite_pass_performance(
    watch_id_or_name: str | None = None, limit: int = 200,
) -> dict:
    """Compare attempted passes and rank downlinks by historical performance."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    reports = build_pass_reports(catalog, watch_id=watch_id, limit=limit)
    return {**summarize_pass_performance(reports), "watch_id": watch_id,
            "reports": reports}


@mcp.tool()
def compare_satellite_passes(pass_ids: list[str]) -> dict:
    """Compare two through 20 selected satellite pass-performance reports."""
    if not isinstance(pass_ids, list) or not 2 <= len(pass_ids) <= 20:
        raise ValueError("pass_ids must contain 2 through 20 pass IDs")
    reports = []
    for pass_id in pass_ids:
        pass_record = catalog.get_satellite_pass(pass_id)
        reports.append(build_pass_report(catalog, pass_record))
    return {**summarize_pass_performance(reports), "reports": reports}


@mcp.tool()
def plot_satellite_pass_performance(
    watch_id_or_name: str | None = None, limit: int = 200,
) -> CallToolResult:
    """Plot pass score over time and against maximum elevation."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    reports = build_pass_reports(catalog, watch_id=watch_id, limit=limit)
    title = ((catalog.get_satellite_watch(watch_id)["name"] + " pass performance")
             if watch_id else "Satellite pass performance")
    path = save_pass_performance_plot(reports, title=title)
    artifact = catalog.register_artifact(path, "satellite_performance_plot")
    result = {"watch_id": watch_id, "pass_count": len(reports), "path": path,
              "artifact_id": artifact["artifact_id"],
              "download_path": f"/artifacts/{artifact['artifact_id']}"}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2)),
                 ImageContent(type="image", data=base64.b64encode(Path(path).read_bytes()).decode("ascii"),
                              mimeType="image/png")], structuredContent=result,
    )


@mcp.tool()
def export_satellite_pass_performance(
    watch_id_or_name: str | None = None, output_format: str = "json",
    limit: int = 500,
) -> dict:
    """Export satellite pass-performance reports as JSON or CSV."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    reports = build_pass_reports(catalog, watch_id=watch_id, limit=limit)
    path = export_pass_performance(list(reversed(reports)), output_format=output_format)
    artifact = catalog.register_artifact(path, "satellite_performance_export")
    return {"watch_id": watch_id, "pass_count": len(reports),
            "output_format": output_format, "path": path,
            "artifact_id": artifact["artifact_id"],
            "download_path": f"/artifacts/{artifact['artifact_id']}"}


@mcp.tool()
def export_satellite_packet_telemetry(
    watch_id_or_name: str | None = None, output_format: str = "jsonl",
    limit: int = 500,
) -> dict:
    """Export decoded satellite AX.25 frames as JSON Lines or CSV."""
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    observations = []
    for mode in ("ax25_afsk1200", "ax25_g3ruh9600"):
        observations.extend(catalog.list_satellite_observations(
            watch_id=watch_id, mode=mode, limit=limit
        ))
    observations.sort(key=lambda item: item["captured_at"])
    path = export_satellite_telemetry(observations[-max(1, min(int(limit), 500)):],
                                      output_format=output_format)
    artifact = catalog.register_artifact(path, "satellite_telemetry_export")
    frame_count = sum(
        len(item.get("details", {}).get("ax25", {}).get("frames", []))
        for item in observations
    )
    return {"output_format": output_format, "observation_count": len(observations),
            "frame_count": frame_count, "path": path,
            "artifact_id": artifact["artifact_id"],
            "download_path": f"/artifacts/{artifact['artifact_id']}"}


@mcp.tool()
def list_satellite_packet_frames(
    watch_id_or_name: str | None = None, callsign: str | None = None,
    valid_fcs_only: bool = False, limit: int = 200,
) -> dict:
    """List decoded 1200/9600-baud AX.25 frames across satellite passes."""
    if not isinstance(valid_fcs_only, bool):
        raise ValueError("valid_fcs_only must be a JSON boolean")
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    needle = str(callsign).strip().upper() if callsign else None
    frames = []
    for mode in ("ax25_afsk1200", "ax25_g3ruh9600"):
        for observation in catalog.list_satellite_observations(
            watch_id=watch_id, mode=mode, limit=min(max(int(limit), 1), 500)
        ):
            for index, frame in enumerate(
                observation.get("details", {}).get("ax25", {}).get("frames", []), 1
            ):
                if valid_fcs_only and not frame.get("fcs_valid"):
                    continue
                addresses = [frame.get("source"), frame.get("destination"),
                             *frame.get("digipeaters", [])]
                if needle and not any(needle in str(value).upper() for value in addresses if value):
                    continue
                frames.append({
                    "observation_id": observation["observation_id"],
                    "pass_id": observation.get("pass_id"),
                    "satellite_name": observation["satellite_name"],
                    "downlink_id": observation["downlink_id"], "mode": mode,
                    "captured_at": observation["captured_at"], "frame_index": index,
                    **frame,
                })
    frames.sort(key=lambda item: item["captured_at"], reverse=True)
    frames = frames[:min(max(int(limit), 1), 1000)]
    return {"count": len(frames), "frames": frames, "callsign": needle,
            "valid_fcs_only": valid_fcs_only}


@mcp.tool()
def save_satellite_telemetry_schema(
    name: str, fields: list[dict], description: str = "",
    satellite_name: str | None = None, match: dict | None = None,
    enabled: bool = True, replace_existing: bool = False,
) -> dict:
    """Save a safe declarative binary telemetry schema for AX.25 information bytes."""
    normalized = normalize_telemetry_schema(
        name=name, fields=fields, description=description,
        satellite_name=satellite_name, match=match, enabled=enabled,
    )
    return catalog.save_satellite_telemetry_schema(
        replace_existing=replace_existing, **normalized
    )


@mcp.tool()
def validate_satellite_telemetry_schema(
    name: str, fields: list[dict], sample_payload_hex: str,
    description: str = "", satellite_name: str | None = None,
    match: dict | None = None,
) -> dict:
    """Validate a telemetry schema and decode sample payload bytes without saving it."""
    normalized = normalize_telemetry_schema(
        name=name, fields=fields, description=description,
        satellite_name=satellite_name, match=match, enabled=True,
    )
    try:
        payload = bytes.fromhex(str(sample_payload_hex).replace(" ", ""))
    except ValueError as exc:
        raise ValueError("sample_payload_hex must contain complete hexadecimal bytes") from exc
    return {"valid": True, "schema": normalized,
            "sample_payload_bytes": len(payload),
            "decoded_fields": decode_telemetry_payload(normalized, payload)}


@mcp.tool()
def list_satellite_telemetry_schemas(enabled: bool | None = None,
                                     limit: int = 100) -> dict:
    """List persistent satellite telemetry schemas."""
    schemas = catalog.list_satellite_telemetry_schemas(enabled=enabled, limit=limit)
    return {"count": len(schemas), "schemas": schemas}


@mcp.tool()
def get_satellite_telemetry_schema(schema_id_or_name: str) -> dict:
    """Get one telemetry schema by stable ID or case-insensitive name."""
    return catalog.get_satellite_telemetry_schema(schema_id_or_name)


@mcp.tool()
def decode_satellite_packet(schema_id_or_name: str, payload_hex: str) -> dict:
    """Decode AX.25 information-field hex with one saved schema without persisting values."""
    schema = catalog.get_satellite_telemetry_schema(schema_id_or_name)
    try:
        payload = bytes.fromhex(str(payload_hex).replace(" ", ""))
    except ValueError as exc:
        raise ValueError("payload_hex must contain complete hexadecimal bytes") from exc
    return {"schema_id": schema["schema_id"], "schema_name": schema["name"],
            "payload_bytes": len(payload),
            "decoded_fields": decode_telemetry_payload(schema, payload)}


@mcp.tool()
def delete_satellite_telemetry_schema(
    schema_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete a telemetry schema and its derived values, retaining source frames."""
    if not confirm_delete:
        raise ValueError("Telemetry schema deletion requires confirm_delete=true")
    return {"deleted": True,
            "schema": catalog.delete_satellite_telemetry_schema(schema_id_or_name)}


@mcp.tool()
def decode_satellite_observation_telemetry(
    observation_id: str, schema_id_or_name: str | None = None,
) -> dict:
    """Apply matching schemas to one stored packet observation and persist values."""
    observation = catalog.get_satellite_observation(observation_id)
    return decode_observation_telemetry(catalog, observation, schema_id_or_name)


@mcp.tool()
def list_satellite_telemetry_values(
    schema_id_or_name: str | None = None, field_name: str | None = None,
    watch_id_or_name: str | None = None, limit: int = 1000,
) -> dict:
    """List decoded telemetry fields across observations and passes."""
    schema_id = (catalog.get_satellite_telemetry_schema(schema_id_or_name)["schema_id"]
                 if schema_id_or_name else None)
    watch_id = (catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
                if watch_id_or_name else None)
    values = catalog.list_satellite_telemetry_values(
        schema_id=schema_id, field_name=field_name, watch_id=watch_id, limit=limit
    )
    return {"count": len(values), "values": values}


@mcp.tool()
def plot_satellite_telemetry(
    schema_id_or_name: str, field_name: str | None = None, limit: int = 1000,
) -> CallToolResult:
    """Plot numeric telemetry values over time and return the PNG."""
    schema = catalog.get_satellite_telemetry_schema(schema_id_or_name)
    values = catalog.list_satellite_telemetry_values(
        schema_id=schema["schema_id"], field_name=field_name, limit=limit
    )
    path = save_telemetry_plot(values, title=f"{schema['name']} telemetry")
    artifact = catalog.register_artifact(path, "satellite_telemetry_plot")
    result = {"schema_id": schema["schema_id"], "field_name": field_name,
              "value_count": len(values), "path": path,
              "artifact_id": artifact["artifact_id"],
              "download_path": f"/artifacts/{artifact['artifact_id']}"}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2)),
                 ImageContent(type="image", data=base64.b64encode(Path(path).read_bytes()).decode("ascii"),
                              mimeType="image/png")], structuredContent=result,
    )


@mcp.tool()
def export_decoded_satellite_telemetry(
    schema_id_or_name: str | None = None, field_name: str | None = None,
    output_format: str = "jsonl", limit: int = 5000,
) -> dict:
    """Export persisted decoded telemetry values as JSON Lines or CSV."""
    schema_id = (catalog.get_satellite_telemetry_schema(schema_id_or_name)["schema_id"]
                 if schema_id_or_name else None)
    values = catalog.list_satellite_telemetry_values(
        schema_id=schema_id, field_name=field_name, limit=limit
    )
    path = export_decoded_telemetry(list(reversed(values)), output_format=output_format)
    artifact = catalog.register_artifact(path, "satellite_decoded_telemetry_export")
    return {"output_format": output_format, "value_count": len(values), "path": path,
            "artifact_id": artifact["artifact_id"],
            "download_path": f"/artifacts/{artifact['artifact_id']}"}


@mcp.tool()
def save_satellite_telemetry_alert_rule(
    name: str, schema_id_or_name: str, field_name: str, condition_type: str,
    threshold_low: float | None = None, threshold_high: float | None = None,
    change_threshold: float | None = None, cooldown_seconds: int = 3600,
    enabled: bool = True, replace_existing: bool = False,
) -> dict:
    """Save a threshold, range, or change alert for one decoded telemetry field."""
    normalized = normalize_telemetry_alert_rule(
        catalog=catalog, name=name, schema_id_or_name=schema_id_or_name,
        field_name=field_name, condition_type=condition_type,
        threshold_low=threshold_low, threshold_high=threshold_high,
        change_threshold=change_threshold, cooldown_seconds=cooldown_seconds,
        enabled=enabled,
    )
    return catalog.save_satellite_telemetry_alert_rule(
        replace_existing=replace_existing, **normalized
    )


@mcp.tool()
def list_satellite_telemetry_alert_rules(
    enabled: bool | None = None, schema_id_or_name: str | None = None,
    field_name: str | None = None, limit: int = 200,
) -> dict:
    """List persistent satellite telemetry alert rules and cooldown state."""
    schema_id = (catalog.get_satellite_telemetry_schema(schema_id_or_name)["schema_id"]
                 if schema_id_or_name else None)
    rules = catalog.list_satellite_telemetry_alert_rules(
        enabled=enabled, schema_id=schema_id, field_name=field_name, limit=limit
    )
    return {"count": len(rules), "rules": rules}


@mcp.tool()
def get_satellite_telemetry_alert_rule(rule_id_or_name: str) -> dict:
    """Get one satellite telemetry alert rule."""
    return catalog.get_satellite_telemetry_alert_rule(rule_id_or_name)


@mcp.tool()
def delete_satellite_telemetry_alert_rule(
    rule_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete a telemetry alert rule while retaining historical alert events."""
    if not confirm_delete:
        raise ValueError("Telemetry alert rule deletion requires confirm_delete=true")
    return {"deleted": True,
            "rule": catalog.delete_satellite_telemetry_alert_rule(rule_id_or_name)}


@mcp.tool()
def test_satellite_telemetry_alert_rule(
    schema_id_or_name: str, field_name: str, condition_type: str,
    current_value: float, previous_value: float | None = None,
    threshold_low: float | None = None, threshold_high: float | None = None,
    change_threshold: float | None = None,
) -> dict:
    """Validate and test a telemetry alert condition without saving or emitting it."""
    normalized = normalize_telemetry_alert_rule(
        catalog=catalog, name="Rule test", schema_id_or_name=schema_id_or_name,
        field_name=field_name, condition_type=condition_type,
        threshold_low=threshold_low, threshold_high=threshold_high,
        change_threshold=change_threshold, cooldown_seconds=0, enabled=True,
    )
    value = {"field_name": field_name, "numeric_value": float(current_value), "unit": None}
    previous = ({"numeric_value": float(previous_value)}
                if previous_value is not None else None)
    return {"rule": normalized,
            "evaluation": evaluate_telemetry_alert_rule(normalized, value, previous)}


@mcp.tool()
def list_satellite_telemetry_alerts(
    acknowledged: bool | None = None, limit: int = 100,
) -> dict:
    """List telemetry threshold/range/change alert events."""
    events = catalog.list_alert_events(
        acknowledged=acknowledged, event_type="satellite_telemetry", limit=limit
    )
    return {"count": len(events), "events": events}


@mcp.tool()
def acknowledge_satellite_telemetry_alert(event_id: str) -> dict:
    """Acknowledge one persisted satellite telemetry alert event."""
    event = catalog.get_alert_event(event_id)
    if event.get("event_type") != "satellite_telemetry":
        raise ValueError("event_id is not a satellite telemetry alert")
    return catalog.acknowledge_alert_event(event_id)


@mcp.tool()
def get_satellite_telemetry_health(stale_after_hours: float = 24.0) -> dict:
    """Summarize telemetry freshness, enabled rules, and recent alerts."""
    stale_after_hours = float(stale_after_hours)
    if not 0.1 <= stale_after_hours <= 8760:
        raise ValueError("stale_after_hours must be from 0.1 through 8760")
    values = catalog.list_satellite_telemetry_values(limit=5000)
    latest = {}
    for item in values:
        latest.setdefault((item["schema_id"], item["field_name"]), item)
    cutoff = datetime.now(timezone.utc).timestamp() - stale_after_hours * 3600
    fields = []
    for item in latest.values():
        item = dict(item)
        item["stale"] = datetime.fromisoformat(item["captured_at"]).timestamp() < cutoff
        fields.append(item)
    rules = catalog.list_satellite_telemetry_alert_rules(enabled=True, limit=500)
    alerts = catalog.list_alert_events(event_type="satellite_telemetry", limit=200)
    return {"latest_field_count": len(fields),
            "stale_field_count": sum(item["stale"] for item in fields),
            "enabled_rule_count": len(rules), "recent_alert_count": len(alerts),
            "latest_fields": fields, "stale_after_hours": stale_after_hours}


@mcp.tool()
def get_satellite_receive_status(job_id: str) -> dict:
    """Get status for an NFM, AX.25, or capture-only satellite job."""
    return satellite_receiver_manager.status(job_id)


@mcp.tool()
def get_satellite_receive_results(job_id: str) -> dict:
    """Get configuration and results for a satellite receiver job."""
    return satellite_receiver_manager.results(job_id)


@mcp.tool()
def stop_satellite_receive(job_id: str) -> dict:
    """Request an active satellite receiver job to stop."""
    return satellite_receiver_manager.stop(job_id)


@mcp.tool()
def list_satellite_sstv_watches(enabled: bool | None = None, limit: int = 100) -> dict:
    """List persistent pass-aware SSTV watches."""
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean or null")
    watches = catalog.list_satellite_watches(enabled=enabled, limit=limit)
    return {"count": len(watches), "watches": watches}


@mcp.tool()
def get_satellite_sstv_watch(watch_id_or_name: str) -> dict:
    """Get one pass-aware SSTV watch."""
    return catalog.get_satellite_watch(watch_id_or_name)


@mcp.tool()
def refresh_satellite_tle(watch_id_or_name: str) -> dict:
    """Fetch and atomically install a current CelesTrak TLE for one watch."""
    watch = catalog.get_satellite_watch(watch_id_or_name)
    refreshed = refresh_satellite_tle_data(catalog, watch)
    satellite_scheduler.refresh()
    return refreshed


@mcp.tool()
def predict_satellite_passes(
    watch_id_or_name: str, hours: float = 24.0, limit: int = 20,
) -> dict:
    """Predict upcoming AOS, culmination, LOS, azimuth, and elevation."""
    watch = catalog.get_satellite_watch(watch_id_or_name)
    passes = predict_passes(watch, hours=hours, limit=limit)
    return {"watch_id": watch["watch_id"], "count": len(passes), "passes": passes}


@mcp.tool()
def list_satellite_pass_windows(
    watch_id_or_name: str | None = None, state: str | None = None, limit: int = 100,
) -> dict:
    """List persisted planned and attempted satellite pass windows."""
    watch_id = None
    if watch_id_or_name is not None:
        watch_id = catalog.get_satellite_watch(watch_id_or_name)["watch_id"]
    states = {"planned", "launched", "completed", "stopped", "interrupted",
              "skipped_busy", "missed", "failed", "superseded"}
    if state is not None and state not in states:
        raise ValueError("state must be one of: " + ", ".join(sorted(states)))
    passes = catalog.list_satellite_passes(watch_id=watch_id, state=state, limit=limit)
    return {"count": len(passes), "passes": passes}


@mcp.tool()
def get_satellite_doppler_plan(
    pass_id: str, include_plot: bool = True,
) -> CallToolResult:
    """Return the persisted Doppler track for a pass and optionally its plot."""
    if not isinstance(include_plot, bool):
        raise ValueError("include_plot must be a JSON boolean")
    record = catalog.get_satellite_pass(pass_id)
    if not record.get("doppler_plan"):
        if not record.get("watch_id"):
            raise ValueError("Historical pass has no Doppler plan or surviving watch")
        watch = catalog.get_satellite_watch(record["watch_id"])
        prediction = record["prediction"]
        prediction["doppler_track"] = build_doppler_plan(watch, prediction)
        record = catalog.save_satellite_pass(watch, prediction)
    if include_plot and (
        not record.get("doppler_plot_path")
        or not Path(record["doppler_plot_path"]).exists()
    ):
        path = save_doppler_plot(record)
        artifact = catalog.register_artifact(path, "satellite_doppler_plot")
        record = catalog.set_satellite_doppler_plot(
            pass_id, path=path, artifact_id=artifact["artifact_id"]
        )
    result = dict(record)
    if result.get("doppler_artifact_id"):
        result["doppler_plot_download_path"] = (
            f"/artifacts/{result['doppler_artifact_id']}"
        )
    content = [TextContent(type="text", text=json.dumps(result, indent=2))]
    if include_plot and result.get("doppler_plot_path"):
        content.append(ImageContent(
            type="image",
            data=base64.b64encode(Path(result["doppler_plot_path"]).read_bytes()).decode("ascii"),
            mimeType="image/png",
        ))
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def set_satellite_sstv_watch_enabled(watch_id_or_name: str, enabled: bool) -> dict:
    """Enable or disable future pass scheduling for a satellite watch."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    watch = catalog.set_satellite_watch_enabled(watch_id_or_name, enabled)
    if enabled:
        satellite_scheduler.refresh()
    return watch


@mcp.tool()
def delete_satellite_sstv_watch(
    watch_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete a satellite watch while retaining historical pass records."""
    if not confirm_delete:
        raise ValueError("Satellite watch deletion requires confirm_delete=true")
    return {"deleted": True, "watch": catalog.delete_satellite_watch(watch_id_or_name)}


@mcp.tool()
def get_satellite_scheduler_status() -> dict:
    """Report pass predictor health and the next automatic watcher start."""
    return satellite_scheduler.status()


@mcp.tool()
def list_satellite_pass_alerts(
    acknowledged: bool | None = None, limit: int = 100,
) -> dict:
    """List persisted pre-pass and pass-outcome alert events."""
    if acknowledged is not None and not isinstance(acknowledged, bool):
        raise ValueError("acknowledged must be a JSON boolean or null")
    events = catalog.list_alert_events(
        acknowledged=acknowledged, event_type="satellite_pass", limit=limit
    )
    return {"count": len(events), "events": events}


@mcp.tool()
def acknowledge_satellite_pass_alert(event_id: str) -> dict:
    """Acknowledge one satellite pass alert; repeated calls are idempotent."""
    event = catalog.get_alert_event(event_id)
    if event.get("event_type") != "satellite_pass":
        raise ValueError(f"Alert event is not a satellite pass event: {event_id}")
    return catalog.acknowledge_alert_event(event_id)


@mcp.tool()
def save_rf_schedule(
    name: str,
    preset_id_or_name: str,
    interval_seconds: int,
    start_at: str | None = None,
    enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Persist a fixed-interval schedule for a saved RF preset.

    start_at is optional; when supplied it must be an ISO 8601 timestamp with a
    timezone. Intervals range from 60 seconds through seven days.
    """
    preset = catalog.get_preset(preset_id_or_name)
    name, interval_seconds, enabled, next_run_at = normalize_schedule(
        name=name,
        interval_seconds=interval_seconds,
        start_at=start_at,
        enabled=enabled,
    )
    return catalog.save_schedule(
        name=name,
        preset_id=preset["preset_id"],
        interval_seconds=interval_seconds,
        enabled=enabled,
        next_run_at=next_run_at,
        replace_existing=replace_existing,
    )


@mcp.tool()
def list_rf_schedules(enabled: bool | None = None, limit: int = 100) -> dict:
    """List persistent RF schedules, optionally filtered by enabled state."""
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean or null")
    schedules = catalog.list_schedules(enabled=enabled, limit=limit)
    return {"count": len(schedules), "schedules": schedules}


@mcp.tool()
def get_rf_schedule(schedule_id_or_name: str) -> dict:
    """Get one persistent RF schedule by stable ID or case-insensitive name."""
    return catalog.get_schedule(schedule_id_or_name)


@mcp.tool()
def set_rf_schedule_enabled(
    schedule_id_or_name: str,
    enabled: bool,
    start_at: str | None = None,
) -> dict:
    """Enable or disable a schedule; enabling resets its next execution time."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    schedule = catalog.get_schedule(schedule_id_or_name)
    if enabled:
        _, _, _, next_run_at = normalize_schedule(
            name=schedule["name"],
            interval_seconds=schedule["interval_seconds"],
            start_at=start_at,
            enabled=True,
        )
    else:
        next_run_at = schedule["next_run_at"]
    return catalog.set_schedule_enabled(
        schedule["schedule_id"], enabled=enabled, next_run_at=next_run_at
    )


@mcp.tool()
def run_rf_schedule_now(schedule_id_or_name: str) -> dict:
    """Run a schedule immediately, including when it is disabled."""
    return scheduler_manager.run_now(schedule_id_or_name)


@mcp.tool()
def delete_rf_schedule(
    schedule_id_or_name: str, confirm_delete: bool = False
) -> dict:
    """Delete a persistent RF schedule only when confirm_delete=true."""
    if not confirm_delete:
        raise ValueError("Schedule deletion requires confirm_delete=true")
    deleted = catalog.delete_schedule(schedule_id_or_name)
    return {"deleted": True, "schedule": deleted}


@mcp.tool()
def get_rf_scheduler_status() -> dict:
    """Report scheduler thread health, enabled count, and next execution time."""
    return scheduler_manager.status()


@mcp.tool()
def save_rf_alert_rule(
    name: str,
    schedule_id_or_name: str,
    condition_type: str,
    entry_label: str | None = None,
    classification_label: str | None = None,
    min_confidence: float | None = None,
    threshold_db: float | None = None,
    enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Save a persistent result-based alert rule for a watchlist schedule."""
    schedule = catalog.get_schedule(schedule_id_or_name)
    if schedule["preset_type"] != "watchlist":
        raise ValueError("v0.14 alert rules require a watchlist schedule")
    normalized = normalize_alert_rule(
        name=name,
        condition_type=condition_type,
        entry_label=entry_label,
        classification_label=classification_label,
        min_confidence=min_confidence,
        threshold_db=threshold_db,
        enabled=enabled,
    )
    if normalized["entry_label"]:
        preset = catalog.get_preset(schedule["preset_id"])
        labels = [item["label"] for item in preset["config"]["entries"]]
        if normalized["entry_label"].casefold() not in {
            label.casefold() for label in labels
        }:
            raise ValueError(
                "entry_label must match a label in the scheduled watchlist: "
                + ", ".join(labels)
            )
    return catalog.save_alert_rule(
        schedule_id=schedule["schedule_id"],
        replace_existing=replace_existing,
        **normalized,
    )


@mcp.tool()
def list_rf_alert_rules(
    schedule_id_or_name: str | None = None,
    enabled: bool | None = None,
    limit: int = 100,
) -> dict:
    """List persistent watchlist alert rules."""
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean or null")
    schedule_id = None
    if schedule_id_or_name is not None:
        schedule_id = catalog.get_schedule(schedule_id_or_name)["schedule_id"]
    rules = catalog.list_alert_rules(
        schedule_id=schedule_id, enabled=enabled, limit=limit
    )
    return {"count": len(rules), "rules": rules}


@mcp.tool()
def get_rf_alert_rule(rule_id_or_name: str) -> dict:
    """Get one alert rule by stable ID or case-insensitive name."""
    return catalog.get_alert_rule(rule_id_or_name)


@mcp.tool()
def set_rf_alert_rule_enabled(rule_id_or_name: str, enabled: bool) -> dict:
    """Enable or disable an RF alert rule."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    return catalog.set_alert_rule_enabled(rule_id_or_name, enabled)


@mcp.tool()
def delete_rf_alert_rule(
    rule_id_or_name: str, confirm_delete: bool = False
) -> dict:
    """Delete an alert rule only when confirm_delete=true; events are retained."""
    if not confirm_delete:
        raise ValueError("Alert rule deletion requires confirm_delete=true")
    deleted = catalog.delete_alert_rule(rule_id_or_name)
    return {"deleted": True, "rule": deleted}


@mcp.tool()
def list_rf_alert_events(
    acknowledged: bool | None = None,
    schedule_id_or_name: str | None = None,
    limit: int = 100,
) -> dict:
    """List stored alert events, including unacknowledged events after restarts."""
    if acknowledged is not None and not isinstance(acknowledged, bool):
        raise ValueError("acknowledged must be a JSON boolean or null")
    schedule_id = None
    if schedule_id_or_name is not None:
        schedule_id = catalog.get_schedule(schedule_id_or_name)["schedule_id"]
    events = catalog.list_alert_events(
        acknowledged=acknowledged,
        schedule_id=schedule_id,
        event_type="rf_watchlist",
        limit=limit,
    )
    return {"count": len(events), "events": events}


@mcp.tool()
def get_rf_alert_event(event_id: str) -> dict:
    """Get one persisted alert event and its observation details."""
    return catalog.get_alert_event(event_id)


@mcp.tool()
def acknowledge_rf_alert_event(event_id: str) -> dict:
    """Acknowledge one alert event; repeated calls are idempotent."""
    return catalog.acknowledge_alert_event(event_id)


@mcp.tool()
def save_rf_webhook_destination(
    name: str,
    url: str,
    rule_id_or_name: str | None = None,
    sstv_rule_id_or_name: str | None = None,
    satellite_watch_id_or_name: str | None = None,
    signing_secret: str | None = None,
    enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Save a webhook destination for all events or one RF/SSTV alert rule."""
    normalized = normalize_webhook_destination(
        name=name, url=url, signing_secret=signing_secret, enabled=enabled
    )
    if sum(value is not None for value in (
        rule_id_or_name, sstv_rule_id_or_name, satellite_watch_id_or_name
    )) > 1:
        raise ValueError("Select only one RF rule, SSTV rule, or satellite watch")
    rule_id = None
    sstv_rule_id = None
    satellite_watch_id = None
    if rule_id_or_name is not None:
        rule_id = catalog.get_alert_rule(rule_id_or_name)["rule_id"]
    if sstv_rule_id_or_name is not None:
        sstv_rule_id = catalog.get_sstv_alert_rule(sstv_rule_id_or_name)["rule_id"]
    if satellite_watch_id_or_name is not None:
        satellite_watch_id = catalog.get_satellite_watch(
            satellite_watch_id_or_name
        )["watch_id"]
    return catalog.save_webhook_destination(
        all_rules=rule_id is None and sstv_rule_id is None and satellite_watch_id is None,
        rule_id=rule_id,
        sstv_rule_id=sstv_rule_id,
        satellite_watch_id=satellite_watch_id,
        replace_existing=replace_existing,
        **normalized,
    )


@mcp.tool()
def list_rf_webhook_destinations(
    enabled: bool | None = None, limit: int = 100
) -> dict:
    """List webhook destinations without exposing signing secrets."""
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean or null")
    destinations = catalog.list_webhook_destinations(enabled=enabled, limit=limit)
    return {"count": len(destinations), "destinations": destinations}


@mcp.tool()
def get_rf_webhook_destination(destination_id_or_name: str) -> dict:
    """Get one webhook destination without exposing its signing secret."""
    return catalog.get_webhook_destination(destination_id_or_name)


@mcp.tool()
def set_rf_webhook_destination_enabled(
    destination_id_or_name: str, enabled: bool
) -> dict:
    """Enable or disable creation of future deliveries for a destination."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    return catalog.set_webhook_destination_enabled(destination_id_or_name, enabled)


@mcp.tool()
def delete_rf_webhook_destination(
    destination_id_or_name: str, confirm_delete: bool = False
) -> dict:
    """Delete a destination and cancel its queued retries when confirmed."""
    if not confirm_delete:
        raise ValueError("Webhook destination deletion requires confirm_delete=true")
    deleted = catalog.delete_webhook_destination(destination_id_or_name)
    return {"deleted": True, "destination": deleted}


@mcp.tool()
def list_rf_webhook_deliveries(
    state: str | None = None,
    event_id: str | None = None,
    limit: int = 100,
) -> dict:
    """List webhook delivery attempts and retry state."""
    states = {"pending", "retrying", "delivered", "failed", "cancelled"}
    if state is not None:
        state = state.strip().lower()
        if state not in states:
            raise ValueError("state must be one of: " + ", ".join(sorted(states)))
    deliveries = catalog.list_webhook_deliveries(
        state=state, event_id=event_id, limit=limit
    )
    return {"count": len(deliveries), "deliveries": deliveries}


@mcp.tool()
def retry_rf_webhook_delivery(delivery_id: str) -> dict:
    """Reset one delivery for immediate retry, including a terminal failure."""
    return catalog.retry_webhook_delivery(delivery_id)


@mcp.tool()
def get_rf_webhook_status() -> dict:
    """Report webhook dispatcher health and delivery counts by state."""
    return webhook_dispatcher.status()


@mcp.tool()
def save_sstv_alert_rule(
    name: str,
    frequency_hz: int | None = None,
    sstv_mode: str | None = None,
    minimum_quality: float = 0.0,
    unique_only: bool = True,
    enabled: bool = True,
    replace_existing: bool = False,
) -> dict:
    """Save a rule that matches newly decoded SSTV gallery images."""
    normalized = normalize_sstv_alert_rule(
        name=name, frequency_hz=frequency_hz, sstv_mode=sstv_mode,
        minimum_quality=minimum_quality, unique_only=unique_only, enabled=enabled,
    )
    return catalog.save_sstv_alert_rule(
        replace_existing=replace_existing, **normalized
    )


@mcp.tool()
def list_sstv_alert_rules(enabled: bool | None = None, limit: int = 100) -> dict:
    """List persistent SSTV image alert rules."""
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean or null")
    rules = catalog.list_sstv_alert_rules(enabled=enabled, limit=limit)
    return {"count": len(rules), "rules": rules}


@mcp.tool()
def get_sstv_alert_rule(rule_id_or_name: str) -> dict:
    """Get one SSTV alert rule by stable ID or case-insensitive name."""
    return catalog.get_sstv_alert_rule(rule_id_or_name)


@mcp.tool()
def set_sstv_alert_rule_enabled(rule_id_or_name: str, enabled: bool) -> dict:
    """Enable or disable an SSTV alert rule."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    return catalog.set_sstv_alert_rule_enabled(rule_id_or_name, enabled)


@mcp.tool()
def delete_sstv_alert_rule(
    rule_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete an SSTV rule only when confirmed; matching events are retained."""
    if not confirm_delete:
        raise ValueError("SSTV alert rule deletion requires confirm_delete=true")
    deleted = catalog.delete_sstv_alert_rule(rule_id_or_name)
    return {"deleted": True, "rule": deleted}


@mcp.tool()
def list_sstv_alert_events(
    acknowledged: bool | None = None, limit: int = 100,
) -> dict:
    """List persisted SSTV image alerts and acknowledgement state."""
    if acknowledged is not None and not isinstance(acknowledged, bool):
        raise ValueError("acknowledged must be a JSON boolean or null")
    events = catalog.list_alert_events(
        acknowledged=acknowledged, event_type="sstv_image", limit=limit
    )
    return {"count": len(events), "events": events}


@mcp.tool()
def get_sstv_alert_event(event_id: str) -> dict:
    """Get one SSTV image alert event."""
    event = catalog.get_alert_event(event_id)
    if event.get("event_type") != "sstv_image":
        raise ValueError(f"Alert event is not an SSTV image event: {event_id}")
    return event


@mcp.tool()
def acknowledge_sstv_alert_event(event_id: str) -> dict:
    """Acknowledge one SSTV image alert; repeated calls are idempotent."""
    get_sstv_alert_event(event_id)
    return catalog.acknowledge_alert_event(event_id)


@mcp.tool()
def decode_digital_signal(
    frequency_hz: int,
    mode: str,
    duration_seconds: float = 10.0,
    cw_wpm: float | None = None,
    rtty_baud: float = 45.45,
    rtty_shift_hz: float = 170.0,
    rtty_polarity: str = "auto",
    retain_iq: bool = False,
    include_plot: bool = True,
) -> CallToolResult:
    """Decode CW, Baudot RTTY, BPSK31, or AX.25 AFSK1200 from fresh IQ.

    Results are heuristic and include confidence, timing/framing diagnostics,
    raw Morse tokens or Baudot codes, and optional diagnostic plot evidence.
    """
    mode = mode.strip().lower()
    if mode not in {"cw", "rtty", "bpsk31", "ax25_afsk1200"}:
        raise ValueError("mode must be cw, rtty, bpsk31, or ax25_afsk1200")
    ensure_data_dirs()
    receiver_center_hz = offset_capture_center(frequency_hz)
    capture = capture_iq(receiver_center_hz, duration_seconds)
    plot_path = None
    try:
        iq = load_complex_float32(capture.path)
        offset_hz = frequency_hz - capture.center_frequency_hz
        baseband = downconvert(iq, capture.sample_rate_hz, offset_hz)
        cutoff_hz = {
            "cw": 600,
            "rtty": max(600, rtty_shift_hz * 2),
            "bpsk31": 150,
            "ax25_afsk1200": 8_000,
        }[mode]
        filtered = _complex_lowpass(baseband, capture.sample_rate_hz, cutoff_hz)
        filtered = filtered[512:]
        if mode == "cw":
            decoded = decode_cw(filtered, capture.sample_rate_hz, wpm=cw_wpm)
            decoder_config = {"cw_wpm": cw_wpm}
        elif mode == "rtty":
            decoded = decode_rtty(
                filtered,
                capture.sample_rate_hz,
                baud=rtty_baud,
                shift_hz=rtty_shift_hz,
                polarity=rtty_polarity,
            )
            decoder_config = {
                "rtty_baud": rtty_baud,
                "rtty_shift_hz": rtty_shift_hz,
                "rtty_polarity": rtty_polarity,
            }
        elif mode == "bpsk31":
            decoded = decode_bpsk31(filtered, capture.sample_rate_hz)
            decoder_config = {"symbol_rate": 31.25, "modulation": "BPSK"}
        else:
            decoded = decode_ax25_afsk1200(filtered, capture.sample_rate_hz)
            decoder_config = {
                "baud": 1_200,
                "mark_hz": 1_200,
                "space_hz": 2_200,
                "framing": "AX.25/HDLC",
            }
        stem = f"decode-{mode}-{capture.path.stem}"
        plot_path = PLOT_DIR / f"{stem}.png"
        save_decode_plot(plot_path, mode, decoded)
        decoded.pop("diagnostic", None)
        result = {
            "decoder_status": "provisional" if decoded["confidence"] < 0.8 else "decoded",
            "warning": (
                "Digital decoding is heuristic; verify important text against the "
                "diagnostic evidence and another decoder."
            ),
            "mode": mode,
            "requested_frequency_hz": int(frequency_hz),
            "receiver_center_frequency_hz": capture.center_frequency_hz,
            "sample_rate_hz": capture.sample_rate_hz,
            "duration_seconds": duration_seconds,
            "decoder": decoded,
            "decoder_config": decoder_config,
            "decode_plot_path": str(plot_path.resolve()),
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at,
        }
        result_path = RESULT_DIR / f"{stem}.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _persist_one_shot(
            job_id=stem,
            job_type="digital_decode",
            result=result,
            config={
                "frequency_hz": frequency_hz,
                "mode": mode,
                "duration_seconds": duration_seconds,
                "retain_iq": retain_iq,
                **decoder_config,
            },
            artifacts=[
                (plot_path, "digital_decode_plot"),
                (capture.path if retain_iq else None, "iq_capture"),
            ],
        )
        decoded_content = (
            repr(decoded.get("text", ""))
            if mode != "ax25_afsk1200"
            else f"{decoded['frame_count']} AX.25 frame(s), {decoded['valid_fcs_count']} valid"
        )
        summary = (
            f"{mode.upper()} decode at {frequency_hz:,} Hz produced "
            f"{decoded_content} with {decoded['confidence']:.0%} heuristic confidence. "
            "Treat uncertain text as provisional.\n\n" + json.dumps(result, indent=2)
        )
        content: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
        if include_plot:
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                    mimeType="image/png",
                )
            )
        return CallToolResult(content=content, structuredContent=result)
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)


@mcp.tool()
def list_digital_decoder_capabilities() -> dict:
    """Report native, WSJT-X, and Fldigi decoder availability."""
    result = decoder_capabilities()
    result["fldigi"] = list_fldigi_mode_capabilities()
    return result


@mcp.tool()
def decode_weak_signal(
    frequency_hz: int,
    mode: str,
    capture_cycles: int = 1,
    align_to_utc: bool = True,
    retain_iq: bool = False,
    retain_audio: bool = True,
) -> dict:
    """Capture and decode one or more UTC-aligned FT8, FT4, or WSPR periods.

    frequency_hz is the receiver's suppressed-carrier/dial frequency; decoded
    audio offsets are added to it when reporting RF frequencies. WSPR can use
    one two-minute cycle; FT8 and FT4 permit bounded multi-cycle observations.
    """
    return decode_live_weak_signal(
        frequency_hz=frequency_hz, mode=mode, capture_cycles=capture_cycles,
        align_to_utc=align_to_utc, retain_iq=retain_iq, retain_audio=retain_audio,
    )


@mcp.tool()
def monitor_weak_signal_frequency(
    frequency_hz: int,
    mode: str = "ft8",
    capture_cycles: int = 4,
    align_to_utc: bool = True,
    retain_audio: bool = False,
) -> dict:
    """Observe a weak-signal dial frequency for several consecutive decode periods."""
    return decode_live_weak_signal(
        frequency_hz=frequency_hz, mode=mode, capture_cycles=capture_cycles,
        align_to_utc=align_to_utc, retain_iq=False, retain_audio=retain_audio,
    )


@mcp.tool()
def list_weak_signal_spots(
    mode: str | None = None,
    callsign: str | None = None,
    dial_frequency_hz: int | None = None,
    limit: int = 200,
) -> dict:
    """List persisted FT8, FT4, or WSPR decodes, optionally filtered."""
    if mode is not None:
        mode = normalize_weak_mode(mode)
    spots = catalog.list_weak_signal_spots(
        mode=mode, callsign=callsign, dial_frequency_hz=dial_frequency_hz, limit=limit
    )
    return {"count": len(spots), "spots": spots}


@mcp.tool()
def get_weak_signal_activity(
    mode: str | None = None,
    dial_frequency_hz: int | None = None,
    limit: int = 1000,
) -> dict:
    """Summarize persisted weak-signal activity by callsign and grid."""
    if mode is not None:
        mode = normalize_weak_mode(mode)
    spots = catalog.list_weak_signal_spots(
        mode=mode, dial_frequency_hz=dial_frequency_hz, limit=limit
    )
    callsigns: dict[str, dict] = {}
    for spot in spots:
        callsign = spot.get("callsign")
        if not callsign:
            continue
        entry = callsigns.setdefault(callsign, {
            "callsign": callsign, "decode_count": 0, "grids": set(),
            "modes": set(), "best_snr_db": None, "latest_at": spot["captured_at"],
        })
        entry["decode_count"] += 1
        if spot.get("grid"):
            entry["grids"].add(spot["grid"])
        entry["modes"].add(spot["mode"])
        snr = spot.get("snr_db")
        if snr is not None and (entry["best_snr_db"] is None or snr > entry["best_snr_db"]):
            entry["best_snr_db"] = snr
    activity = []
    for entry in callsigns.values():
        entry["grids"] = sorted(entry["grids"])
        entry["modes"] = sorted(entry["modes"])
        activity.append(entry)
    activity.sort(key=lambda item: (-item["decode_count"], item["callsign"]))
    return {"spot_count": len(spots), "callsign_count": len(activity),
            "activity": activity}


@mcp.tool()
def get_fldigi_status() -> dict:
    """Check Fldigi installation, XML-RPC connectivity, modem, and audio playback."""
    return fldigi_status()


@mcp.tool()
def list_fldigi_modes() -> dict:
    """List configured Fldigi text modes and whether the running instance offers them."""
    return list_fldigi_mode_capabilities()


@mcp.tool()
def decode_fldigi_mode(
    frequency_hz: int,
    mode: str,
    duration_seconds: float = 30,
    carrier_audio_hz: int = 1500,
    retain_iq: bool = False,
    retain_audio: bool = True,
) -> dict:
    """Decode an Olivia/Contestia/MFSK/PSK/DominoEX/THOR/MT63/Hell session.

    frequency_hz is the USB dial frequency. The requested Fldigi modem listens
    at carrier_audio_hz within that receiver audio passband.
    """
    return decode_live_fldigi(
        frequency_hz=frequency_hz, mode=mode, duration_seconds=duration_seconds,
        carrier_audio_hz=carrier_audio_hz, retain_iq=retain_iq,
        retain_audio=retain_audio,
    )


@mcp.tool()
def list_fldigi_decodes(
    mode: str | None = None,
    dial_frequency_hz: int | None = None,
    limit: int = 200,
) -> dict:
    """List persisted Fldigi text-mode receive sessions."""
    if mode is not None:
        mode, _ = normalize_fldigi_mode(mode)
    decodes = catalog.list_fldigi_decodes(
        mode=mode, dial_frequency_hz=dial_frequency_hz, limit=limit
    )
    return {"count": len(decodes), "decodes": decodes}


@mcp.tool()
def list_sstv_decoder_capabilities() -> dict:
    """Report SSTV decoder availability, supported VIS modes, and receiver modes."""
    return sstv_capabilities()


@mcp.tool()
def decode_sstv(
    frequency_hz: int,
    duration_seconds: float = 130,
    receiver_mode: str = "usb",
    retain_audio: bool = True,
    retain_iq: bool = False,
    deduplicate: bool = True,
) -> dict:
    """Start an asynchronous SSTV capture and image decode.

    Use USB for HF dial frequencies and NFM for direct FM channels such as an
    SSTV satellite downlink. Poll get_sstv_status with the returned job_id.
    """
    return sstv_manager.start(
        frequency_hz=frequency_hz, duration_seconds=duration_seconds,
        receiver_mode=receiver_mode, retain_audio=retain_audio, retain_iq=retain_iq,
        deduplicate=deduplicate,
    )


@mcp.tool()
def monitor_sstv_frequency(
    frequency_hz: int,
    duration_seconds: float = 180,
    receiver_mode: str = "usb",
    retain_audio: bool = True,
    deduplicate: bool = True,
) -> dict:
    """Start a longer asynchronous SSTV receive window on a known frequency."""
    return sstv_manager.start(
        frequency_hz=frequency_hz, duration_seconds=duration_seconds,
        receiver_mode=receiver_mode, retain_audio=retain_audio, retain_iq=False,
        deduplicate=deduplicate,
    )


@mcp.tool()
def get_sstv_status(job_id: str) -> dict:
    """Get capture/demodulation/decoding phase and completion state."""
    return sstv_manager.status(job_id)


@mcp.tool()
def get_sstv_results(job_id: str) -> dict:
    """Get final or current SSTV job metadata and gallery image ID."""
    return sstv_manager.results(job_id)


@mcp.tool()
def stop_sstv(job_id: str) -> dict:
    """Request that an SSTV job stop after its current receiver capture."""
    return sstv_manager.stop(job_id)


@mcp.tool()
def list_sstv_images(
    frequency_hz: int | None = None,
    sstv_mode: str | None = None,
    include_duplicates: bool = True,
    limit: int = 100,
) -> dict:
    """List the persistent SSTV image gallery, optionally filtered."""
    images = catalog.list_sstv_images(
        frequency_hz=frequency_hz, sstv_mode=sstv_mode,
        include_duplicates=include_duplicates, limit=limit
    )
    return {"count": len(images), "images": images}


@mcp.tool()
def get_sstv_activity(since: str | None = None) -> dict:
    """Summarize decoded SSTV images by mode and frequency, with duplicate counts."""
    if since is not None:
        since = parse_utc(since).isoformat()
    return catalog.sstv_activity_summary(since=since)


@mcp.tool()
def get_sstv_image(image_id: str, include_image: bool = True) -> CallToolResult:
    """Retrieve one gallery entry and optionally return its decoded PNG natively."""
    metadata = catalog.get_sstv_image(image_id)
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(metadata, indent=2))
    ]
    if include_image:
        content.append(ImageContent(
            type="image",
            data=base64.b64encode(Path(metadata["image_path"]).read_bytes()).decode("ascii"),
            mimeType="image/png",
        ))
    return CallToolResult(content=content, structuredContent=metadata)


@mcp.tool()
def start_sstv_watcher(
    frequency_hz: int,
    receiver_mode: str = "nfm",
    watch_duration_seconds: float = 3600,
    rearm: bool = True,
    retain_audio: bool = True,
    deduplicate: bool = True,
) -> dict:
    """Stream IQ and trigger SSTV recording only after a valid VIS header."""
    return sstv_watcher_manager.start(
        frequency_hz=frequency_hz, receiver_mode=receiver_mode,
        watch_duration_seconds=watch_duration_seconds, rearm=rearm,
        retain_audio=retain_audio, deduplicate=deduplicate,
    )


@mcp.tool()
def get_sstv_watcher_status(job_id: str) -> dict:
    """Report live SSTV watcher phase, triggers, images, and streamed duration."""
    return sstv_watcher_manager.status(job_id)


@mcp.tool()
def get_sstv_watcher_results(job_id: str) -> dict:
    """Return decoded image records and failures from an SSTV watcher session."""
    return sstv_watcher_manager.results(job_id)


@mcp.tool()
def stop_sstv_watcher(job_id: str) -> dict:
    """Stop a streaming SSTV watcher and finish any queued image decode."""
    return sstv_watcher_manager.stop(job_id)


@mcp.tool()
def list_sstv_watch_sessions(state: str | None = None, limit: int = 50) -> dict:
    """List persisted streaming SSTV watcher sessions."""
    sessions = sstv_watcher_manager.list_sessions(state=state, limit=limit)
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def receive_broadcast_fm(
    frequency_hz: int,
    duration_seconds: float = 10.0,
    stereo: bool = True,
    deemphasis_us: int = 75,
    decode_rds_data: bool = True,
    retain_iq: bool = False,
    include_audio: bool = True,
    include_plot: bool = True,
    receiver_id: str | None = None,
) -> CallToolResult:
    """Receive broadcast wideband FM and return mono/stereo audio and MPX metrics."""
    frequency_hz = int(frequency_hz)
    if not 60_000_000 <= frequency_hz <= 110_000_000:
        raise ValueError("broadcast FM frequency must be from 60 through 110 MHz")
    if not isinstance(stereo, bool):
        raise ValueError("stereo must be a JSON boolean")
    if deemphasis_us not in {50, 75}:
        raise ValueError("deemphasis_us must be 50 or 75")
    if not isinstance(decode_rds_data, bool):
        raise ValueError("decode_rds_data must be a JSON boolean")
    ensure_data_dirs()
    receiver_center_hz = offset_capture_center(
        frequency_hz, offset_hz=150_000, receiver_id=receiver_id,
    )
    capture_options = {"receiver_id": receiver_id} if receiver_id is not None else {}
    capture = capture_iq(receiver_center_hz, duration_seconds, **capture_options)
    plot_path = None
    wav_path = None
    try:
        iq = load_complex_float32(capture.path)
        offset_hz = frequency_hz - capture.center_frequency_hz
        baseband = downconvert(iq, capture.sample_rate_hz, offset_hz)
        audio, metrics, diagnostic = demodulate_broadcast_fm(
            baseband,
            capture.sample_rate_hz,
            deemphasis_us=deemphasis_us,
            stereo=stereo,
        )
        rds_result = (
            decode_rds(diagnostic["composite"], diagnostic["composite_sample_rate_hz"])
            if decode_rds_data
            else None
        )
        stem = f"wfm-{capture.path.stem}"
        wav_path = AUDIO_DIR / f"{stem}.wav"
        plot_path = PLOT_DIR / f"{stem}.png"
        write_wav(wav_path, audio)
        save_broadcast_fm_plot(plot_path, frequency_hz, diagnostic)
        result = {
            "job_id": stem,
            "mode": "broadcast_fm_wfm",
            "requested_frequency_hz": frequency_hz,
            "receiver_center_frequency_hz": capture.center_frequency_hz,
            "sample_rate_hz": capture.sample_rate_hz,
            "receiver_id": getattr(capture, "receiver_id", receiver_id),
            "receiver_backend": getattr(capture, "backend", None),
            "duration_seconds": duration_seconds,
            "metrics": metrics,
            "rds": rds_result,
            "audio_wav_path": str(wav_path.resolve()),
            "multiplex_plot_path": str(plot_path.resolve()),
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at,
            "warning": (
                "RDS metadata is accepted only from complete checksum-valid groups; "
                "short captures may not contain every repeated field."
            ),
        }
        result_path = RESULT_DIR / f"{stem}.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _persist_one_shot(
            job_id=stem,
            job_type="broadcast_fm",
            result=result,
            config={
                "frequency_hz": frequency_hz,
                "duration_seconds": duration_seconds,
                "stereo": stereo,
                "deemphasis_us": deemphasis_us,
                "decode_rds_data": decode_rds_data,
                "retain_iq": retain_iq,
                "receiver_id": receiver_id,
            },
            artifacts=[
                (wav_path, "broadcast_fm_audio"),
                (plot_path, "broadcast_fm_multiplex_plot"),
                (capture.path if retain_iq else None, "iq_capture"),
            ],
        )
        summary = (
            f"Broadcast FM at {frequency_hz / 1e6:.3f} MHz produced "
            f"{metrics['audio_channels']}-channel audio; stereo pilot "
            f"{'detected' if metrics['stereo_detected'] else 'not detected'}; "
            f"{rds_result['group_count'] if rds_result else 0} valid RDS group(s).\n\n"
            + json.dumps(result, indent=2)
        )
        content: list[TextContent | ImageContent | AudioContent] = [
            TextContent(type="text", text=summary)
        ]
        if include_plot:
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                    mimeType="image/png",
                )
            )
        if include_audio:
            content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                    mimeType="audio/wav",
                )
            )
        return CallToolResult(content=content, structuredContent=result)
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)


@mcp.tool()
def inspect_spectrum(
    center_frequency_hz: int,
    duration_seconds: float = 2.0,
    fft_size: int = 16_384,
    threshold_above_noise_db: float = 8.0,
    max_peaks: int = 20,
    retain_iq: bool = False,
    include_plot: bool = True,
    receiver_id: str | None = None,
) -> CallToolResult:
    """Capture and inspect spectrum near a frequency, returning relative levels and peaks.

    Use for signal discovery, band inspection, relative noise measurements, and finding
    candidate carriers. Results are relative dB, not calibrated received power. The
    receiver covers 9 kHz-31 MHz and 60-260 MHz.
    """
    if not 3 <= threshold_above_noise_db <= 60:
        raise ValueError("threshold_above_noise_db must be from 3 through 60")
    if not 1 <= max_peaks <= MAX_PEAKS:
        raise ValueError(f"max_peaks must be from 1 through {MAX_PEAKS}")

    ensure_data_dirs()
    capture_options = {"receiver_id": receiver_id} if receiver_id is not None else {}
    capture = capture_iq(center_frequency_hz, duration_seconds, **capture_options)
    try:
        iq = load_complex_float32(capture.path)
        frequencies, psd_dbfs_hz = averaged_psd_dbfs_per_hz(
            iq,
            capture.center_frequency_hz,
            capture.sample_rate_hz,
            fft_size,
        )
        power_db = psd_dbfs_hz - np.max(psd_dbfs_hz)
        digital_levels = iq_level_metrics(iq)
        mask = valid_passband_mask(
            frequencies,
            capture.center_frequency_hz,
            capture.sample_rate_hz,
        )
        noise_floor_db, peaks = analyze_peaks(
            frequencies,
            power_db,
            mask,
            threshold_above_noise_db=threshold_above_noise_db,
            max_peaks=max_peaks,
        )
        plot_path = PLOT_DIR / f"{capture.path.stem}-spectrum.png"
        save_plot(
            plot_path,
            frequencies,
            power_db,
            mask,
            capture.center_frequency_hz,
            noise_floor_db,
            peaks,
        )
        valid_frequencies = frequencies[mask]
        peaks_result = peak_dicts(peaks)
        calibration = capture.calibration
        calibration_offset = (
            calibration.get("dbfs_to_dbm_offset_db") if calibration else None
        )
        for peak, peak_result in zip(peaks, peaks_result, strict=True):
            peak_index = int(np.argmin(np.abs(frequencies - peak.frequency_hz)))
            peak_result["digital_peak_psd_dbfs_hz"] = float(psd_dbfs_hz[peak_index])
            peak_result["digital_power_dbfs_10khz"] = integrate_psd_dbfs(
                frequencies,
                psd_dbfs_hz,
                peak.frequency_hz,
                DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
            )
            if calibration_offset is not None:
                peak_result["calibrated_peak_psd_dbm_hz"] = (
                    peak_result["digital_peak_psd_dbfs_hz"] + calibration_offset
                )
                peak_result["calibrated_power_dbm_10khz"] = (
                    peak_result["digital_power_dbfs_10khz"] + calibration_offset
                )
        result = {
            "measurement_scale": (
                "calibrated_dbm" if calibration_offset is not None else "relative_db"
            ),
            "digital_power_scale": {
                "scale": PSD_SCALE,
                "calibrated_rf_input_power": calibration_offset is not None,
                "psd_units": "dBFS/Hz",
                "integrated_power_units": "dBFS",
                "integration_bandwidth_hz": DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
            },
            "receiver_profile": {
                "sample_rate_hz": capture.sample_rate_hz,
                "agc": True,
                "comparable_fixed_gain": False,
                "fft_size": fft_size,
                "window": "blackman",
                "calibration": calibration,
            },
            "digital_levels": digital_levels,
            "receiver_id": getattr(capture, "receiver_id", receiver_id),
            "receiver_backend": getattr(capture, "backend", None),
            "center_frequency_hz": capture.center_frequency_hz,
            "sample_rate_hz": capture.sample_rate_hz,
            "captured_samples": capture.captured_samples,
            "duration_seconds": capture.captured_samples / capture.sample_rate_hz,
            "fft_size": fft_size,
            "bin_width_hz": capture.sample_rate_hz / fft_size,
            "analyzed_range_hz": [float(valid_frequencies.min()), float(valid_frequencies.max())],
            "relative_noise_floor_db": noise_floor_db,
            "digital_noise_floor_dbfs_hz": float(np.median(psd_dbfs_hz[mask])),
            "digital_peak_psd_dbfs_hz": float(np.max(psd_dbfs_hz[mask])),
            "peak_count": len(peaks),
            "peaks": peaks_result,
            "spectrum_plot_path": str(plot_path.resolve()),
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at,
        }
        if calibration_offset is not None:
            result["calibrated_noise_floor_dbm_hz"] = (
                result["digital_noise_floor_dbfs_hz"] + calibration_offset
            )
            result["calibrated_peak_psd_dbm_hz"] = (
                result["digital_peak_psd_dbfs_hz"] + calibration_offset
            )
        result_path = RESULT_DIR / f"{capture.path.stem}-spectrum.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _persist_one_shot(
            job_id=f"inspect-{capture.path.stem}",
            job_type="spectrum_inspection",
            result=result,
            config={
                "center_frequency_hz": center_frequency_hz,
                "duration_seconds": duration_seconds,
                "fft_size": fft_size,
                "threshold_above_noise_db": threshold_above_noise_db,
                "max_peaks": max_peaks,
                "receiver_id": receiver_id,
                "retain_iq": retain_iq,
            },
            artifacts=[
                (plot_path, "spectrum_plot"),
                (capture.path if retain_iq else None, "iq_capture"),
            ],
        )

        summary = (
            f"Inspected {capture.center_frequency_hz:,} Hz for "
            f"{result['duration_seconds']:.3f} seconds. Found {len(peaks)} peaks "
            f"at least {threshold_above_noise_db:g} dB above the median relative "
            f"noise floor. Levels are relative, not calibrated dBm.\n\n"
            + json.dumps(result, indent=2)
        )
        content: list[TextContent | ImageContent] = [
            TextContent(type="text", text=summary)
        ]
        if include_plot:
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                    mimeType="image/png",
                )
            )
        return CallToolResult(content=content, structuredContent=result)
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)


@mcp.tool()
def analyze_signal(
    frequency_hz: int,
    mode: str = "am",
    bandwidth_hz: int | None = None,
    duration_seconds: float = 5.0,
    fft_size: int = 16_384,
    cw_tone_hz: int = 700,
    retain_iq: bool = False,
    include_audio: bool = True,
    include_plots: bool = True,
    receiver_id: str | None = None,
) -> CallToolResult:
    """Measure and demodulate one RF signal as AM, USB, LSB, CW, or NFM.

    Returns relative RF measurements, native MCP WAV audio, an RF spectrum plot,
    and an audio spectrum plot. Frequency is in Hz. Results are not calibrated dBm.
    """
    mode = normalize_mode(mode)
    bandwidth_hz = validate_bandwidth(mode, bandwidth_hz)
    if not 300 <= cw_tone_hz <= 1_200:
        raise ValueError("cw_tone_hz must be from 300 through 1200 Hz")

    ensure_data_dirs()
    receiver_center_hz = offset_capture_center(int(frequency_hz), receiver_id=receiver_id)
    capture_options = {"receiver_id": receiver_id} if receiver_id is not None else {}
    capture = capture_iq(receiver_center_hz, duration_seconds, **capture_options)
    try:
        iq = load_complex_float32(capture.path)
        target_offset_hz = int(frequency_hz) - capture.center_frequency_hz
        baseband = downconvert(iq, capture.sample_rate_hz, target_offset_hz)
        audio = demodulate(baseband, capture.sample_rate_hz, mode, bandwidth_hz, cw_tone_hz)

        frequencies, power_db = averaged_spectrum(
            iq,
            capture.center_frequency_hz,
            capture.sample_rate_hz,
            fft_size,
        )
        metrics = measure_signal(
            frequencies,
            power_db,
            int(frequency_hz),
            bandwidth_hz,
            baseband,
            SAMPLE_RATE,
        )
        mask = valid_passband_mask(frequencies, capture.center_frequency_hz, capture.sample_rate_hz)
        noise_floor_db, peaks = analyze_peaks(
            frequencies,
            power_db,
            mask,
            threshold_above_noise_db=6.0,
            max_peaks=20,
        )

        stem = f"{capture.path.stem}-{int(frequency_hz)}-{mode}"
        rf_plot_path = PLOT_DIR / f"{stem}-rf.png"
        audio_plot_path = PLOT_DIR / f"{stem}-audio.png"
        wav_path = AUDIO_DIR / f"{stem}.wav"
        save_plot(
            rf_plot_path,
            frequencies,
            power_db,
            mask,
            capture.center_frequency_hz,
            noise_floor_db,
            peaks,
        )
        save_audio_spectrum(audio_plot_path, audio, mode, int(frequency_hz))
        write_wav(wav_path, audio)

        result = {
            "measurement_scale": "relative_db",
            "receiver_id": getattr(capture, "receiver_id", receiver_id),
            "receiver_backend": getattr(capture, "backend", None),
            "requested_frequency_hz": int(frequency_hz),
            "receiver_center_frequency_hz": capture.center_frequency_hz,
            "receiver_offset_hz": target_offset_hz,
            "mode": mode,
            "bandwidth_hz": bandwidth_hz,
            "duration_seconds": capture.captured_samples / capture.sample_rate_hz,
            "sample_rate_hz": capture.sample_rate_hz,
            "audio_sample_rate_hz": AUDIO_SAMPLE_RATE,
            "metrics": asdict(metrics),
            "rf_spectrum_plot_path": str(rf_plot_path.resolve()),
            "audio_spectrum_plot_path": str(audio_plot_path.resolve()),
            "audio_wav_path": str(wav_path.resolve()),
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at,
        }
        result_path = RESULT_DIR / f"{stem}.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _persist_one_shot(
            job_id=f"analyze-{capture.path.stem}",
            job_type="signal_analysis",
            result=result,
            config={
                "frequency_hz": frequency_hz,
                "mode": mode,
                "bandwidth_hz": bandwidth_hz,
                "duration_seconds": duration_seconds,
                "fft_size": fft_size,
                "cw_tone_hz": cw_tone_hz,
                "retain_iq": retain_iq,
            },
            artifacts=[
                (rf_plot_path, "rf_spectrum_plot"),
                (audio_plot_path, "audio_spectrum_plot"),
                (wav_path, "audio_wav"),
                (capture.path if retain_iq else None, "iq_capture"),
            ],
        )

        summary = (
            f"Analyzed {frequency_hz:,} Hz as {mode.upper()} with a "
            f"{bandwidth_hz:,} Hz bandwidth for {result['duration_seconds']:.3f} seconds. "
            f"Estimated SNR is {metrics.estimated_snr_db:.1f} dB; signal present is "
            f"{metrics.signal_present}. Levels are relative, not calibrated dBm.\n\n"
            + json.dumps(result, indent=2)
        )
        content: list[TextContent | ImageContent | AudioContent] = [
            TextContent(type="text", text=summary)
        ]
        if include_plots:
            for plot_path in (rf_plot_path, audio_plot_path):
                content.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                        mimeType="image/png",
                    )
                )
        if include_audio:
            content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                    mimeType="audio/wav",
                )
            )
        return CallToolResult(content=content, structuredContent=result)
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)


@mcp.tool()
def start_monitor(
    frequency_hz: int,
    mode: str = "am",
    bandwidth_hz: int | None = None,
    total_duration_seconds: float = 300,
    capture_duration_seconds: float = 2,
    interval_seconds: float = 5,
    fft_size: int = 8_192,
    waterfall_span_hz: int = 100_000,
    record_audio_on_activity: bool = False,
) -> dict:
    """Start an asynchronous single-frequency RF activity monitor.

    The monitor continues after this tool returns. Use get_monitor_status and
    get_monitor_results with the returned job_id. Only one monitor may actively
    use the receiver at a time.
    """
    return monitor_manager.start(
        frequency_hz=frequency_hz,
        mode=mode,
        bandwidth_hz=bandwidth_hz,
        total_duration_seconds=total_duration_seconds,
        capture_duration_seconds=capture_duration_seconds,
        interval_seconds=interval_seconds,
        fft_size=fft_size,
        waterfall_span_hz=waterfall_span_hz,
        record_audio_on_activity=record_audio_on_activity,
    )


@mcp.tool()
def get_monitor_status(job_id: str) -> dict:
    """Get progress and state for an RF monitoring job."""
    return monitor_manager.status(job_id)


@mcp.tool()
def get_monitor_results(job_id: str, include_waterfall: bool = True) -> CallToolResult:
    """Get current or final measurements, events, and waterfall for a monitor job."""
    result, plot_path = monitor_manager.results(job_id)
    summary = (
        f"Monitor {job_id} is {result['state']}. It has collected "
        f"{result['capture_count']} captures and detected {result['event_count']} "
        f"activity events. Levels are relative, not calibrated dBm.\n\n"
        + json.dumps(result, indent=2)
    )
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
    if include_waterfall and plot_path is not None:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def stop_monitor(job_id: str) -> dict:
    """Request that a running RF monitoring job stop after its current capture."""
    return monitor_manager.stop(job_id)


@mcp.tool()
def start_band_scan(
    start_frequency_hz: int,
    stop_frequency_hz: int,
    capture_duration_seconds: float = 1.0,
    overlap_fraction: float = 0.15,
    fft_size: int = 8_192,
    threshold_above_noise_db: float = 8.0,
    minimum_signal_spacing_hz: float = 1_000,
    attenuation_steps: int = 1,
    max_signals: int = 100,
) -> dict:
    """Start an asynchronous stitched spectrum scan across an HF+ frequency range.

    Uses fixed receiver gain for comparable segments. Attenuation is in 6 dB steps.
    Start and stop must be within the same HF or VHF tuning range.
    """
    return scan_manager.start(
        start_frequency_hz=start_frequency_hz,
        stop_frequency_hz=stop_frequency_hz,
        capture_duration_seconds=capture_duration_seconds,
        overlap_fraction=overlap_fraction,
        fft_size=fft_size,
        threshold_above_noise_db=threshold_above_noise_db,
        minimum_signal_spacing_hz=minimum_signal_spacing_hz,
        attenuation_steps=attenuation_steps,
        max_signals=max_signals,
    )


@mcp.tool()
def get_band_scan_status(job_id: str) -> dict:
    """Get progress and state for an asynchronous band scan."""
    return scan_manager.status(job_id)


@mcp.tool()
def get_band_scan_results(job_id: str, include_spectrum: bool = True) -> CallToolResult:
    """Get current or final stitched spectrum and ranked signals for a band scan."""
    result, plot_path = scan_manager.results(job_id)
    summary = (
        f"Band scan {job_id} is {result['state']}. Completed "
        f"{result['completed_steps']} of {result['config']['planned_steps']} retunes and "
        f"found {result.get('signal_count', 0)} candidate signals. Levels are relative "
        f"with fixed receiver gain, not calibrated dBm.\n\n"
        + json.dumps(result, indent=2)
    )
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
    if include_spectrum and plot_path is not None:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def stop_band_scan(job_id: str) -> dict:
    """Request that a running band scan stop after its current capture."""
    return scan_manager.stop(job_id)


@mcp.tool()
def start_band_survey(
    start_frequency_hz: int,
    stop_frequency_hz: int,
    capture_duration_seconds: float = 1.0,
    overlap_fraction: float = 0.15,
    fft_size: int = 8_192,
    threshold_above_noise_db: float = 8.0,
    minimum_signal_spacing_hz: float = 1_000,
    attenuation_steps: int = 1,
    max_signals: int = 100,
    classify_top_signals: int = 10,
    classification_duration_seconds: float = 2.0,
    classification_bandwidth_hz: int = 30_000,
) -> dict:
    """Scan a band, then classify its strongest detected signals asynchronously.

    The survey first builds a fixed-gain stitched spectrum, then retunes to as many
    as 20 ranked carriers for heuristic AM/USB/LSB/CW/NFM/digital classification.
    Use the survey status and results tools with the returned job_id.
    """
    if classify_top_signals < 1:
        raise ValueError("classify_top_signals must be at least 1 for a band survey")
    return scan_manager.start(
        start_frequency_hz=start_frequency_hz,
        stop_frequency_hz=stop_frequency_hz,
        capture_duration_seconds=capture_duration_seconds,
        overlap_fraction=overlap_fraction,
        fft_size=fft_size,
        threshold_above_noise_db=threshold_above_noise_db,
        minimum_signal_spacing_hz=minimum_signal_spacing_hz,
        attenuation_steps=attenuation_steps,
        max_signals=max_signals,
        classify_top_signals=classify_top_signals,
        classification_duration_seconds=classification_duration_seconds,
        classification_bandwidth_hz=classification_bandwidth_hz,
    )


@mcp.tool()
def get_band_survey_status(job_id: str) -> dict:
    """Get scanning/classification phase and progress for a band survey."""
    return scan_manager.status(job_id)


@mcp.tool()
def get_band_survey_results(job_id: str, include_spectrum: bool = True) -> CallToolResult:
    """Get current or final band-survey detections, classifications, and report plot."""
    result, plot_path = scan_manager.results(job_id)
    completed = sum(
        item.get("status") == "completed" for item in result.get("classifications", [])
    )
    ambiguous = sum(
        bool(item.get("ambiguous")) for item in result.get("classifications", [])
    )
    summary = (
        f"Band survey {job_id} is {result['state']} in phase {result['phase']}. "
        f"It detected {result.get('signal_count', 0)} carriers and completed "
        f"{completed} classifications; {ambiguous} are marked ambiguous. "
        f"Classifications are heuristic and levels are relative, not calibrated dBm.\n\n"
        + json.dumps(result, indent=2)
    )
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
    if include_spectrum and plot_path is not None:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def stop_band_survey(job_id: str) -> dict:
    """Stop a band survey after its current scan or classification capture."""
    return scan_manager.stop(job_id)


@mcp.tool()
def survey_broadcast_fm(
    start_frequency_hz: int = 87_900_000,
    stop_frequency_hz: int = 107_900_000,
    channel_spacing_hz: int = 200_000,
    discovery_duration_seconds: float = 0.25,
    discovery_threshold_db: float = 8.0,
    rds_duration_seconds: float = 10.0,
    deemphasis_us: int = 75,
    save_audio: bool = False,
    save_plots: bool = True,
    resume_job_id: str | None = None,
) -> dict:
    """Start or resume a persistent asynchronous broadcast-FM/RDS survey.

    A quick channel-grid pass finds occupied candidates. Longer WFM/RDS captures
    then update the station directory. A stopped, interrupted, or failed job can
    be continued by passing its ID as resume_job_id.
    """
    return fm_survey_manager.start(
        start_frequency_hz=start_frequency_hz,
        stop_frequency_hz=stop_frequency_hz,
        channel_spacing_hz=channel_spacing_hz,
        discovery_duration_seconds=discovery_duration_seconds,
        discovery_threshold_db=discovery_threshold_db,
        rds_duration_seconds=rds_duration_seconds,
        deemphasis_us=deemphasis_us,
        save_audio=save_audio,
        save_plots=save_plots,
        resume_job_id=resume_job_id,
    )


@mcp.tool()
def get_fm_survey_status(job_id: str) -> dict:
    """Get phase, progress, and counts for a broadcast-FM survey."""
    return fm_survey_manager.status(job_id)


@mcp.tool()
def get_fm_survey_results(job_id: str) -> dict:
    """Get candidates, decoded stations, artifacts, and checkpoint state."""
    return fm_survey_manager.results(job_id)


@mcp.tool()
def stop_fm_survey(job_id: str) -> dict:
    """Stop an FM survey after the current capture so it can later be resumed."""
    return fm_survey_manager.stop(job_id)


@mcp.tool()
def list_fm_stations(rds_only: bool = False, limit: int = 200) -> dict:
    """List the persistent broadcast-FM station directory in frequency order."""
    stations = catalog.list_fm_stations(rds_only=rds_only, limit=limit)
    return {"count": len(stations), "stations": stations}


@mcp.tool()
def get_fm_station(frequency_hz: int) -> dict:
    """Get the latest accumulated station and RDS metadata for one FM channel."""
    return catalog.get_fm_station(frequency_hz)


@mcp.tool()
def compare_fm_surveys(baseline_job_id: str, comparison_job_id: str) -> dict:
    """Compare two FM surveys for new, missing, and metadata-changed stations."""
    if baseline_job_id == comparison_job_id:
        raise ValueError("baseline_job_id and comparison_job_id must be different")
    jobs = [catalog.get_job(baseline_job_id), catalog.get_job(comparison_job_id)]
    for job in jobs:
        if job["job_type"] != "fm_broadcast_survey" or not isinstance(job.get("result"), dict):
            raise ValueError("Both job IDs must identify readable FM broadcast surveys")
    return compare_fm_survey_results(jobs[0]["result"], jobs[1]["result"])


@mcp.tool()
def list_comparable_surveys(
    start_frequency_hz: int,
    stop_frequency_hz: int,
    endpoint_tolerance_hz: int = 2_000,
    limit: int = 20,
) -> dict:
    """List completed scans/surveys covering approximately the requested band."""
    start_frequency_hz = int(start_frequency_hz)
    stop_frequency_hz = int(stop_frequency_hz)
    endpoint_tolerance_hz = int(endpoint_tolerance_hz)
    if stop_frequency_hz <= start_frequency_hz:
        raise ValueError("stop_frequency_hz must be greater than start_frequency_hz")
    if not 0 <= endpoint_tolerance_hz <= 100_000:
        raise ValueError("endpoint_tolerance_hz must be from 0 through 100000")
    limit = max(1, min(int(limit), 100))
    matches = []
    for job in catalog.list_jobs(state="completed", limit=200):
        if job["job_type"] not in SUPPORTED_JOB_TYPES:
            continue
        config = job.get("config", {})
        try:
            job_start = int(config["start_frequency_hz"])
            job_stop = int(config["stop_frequency_hz"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            abs(job_start - start_frequency_hz) <= endpoint_tolerance_hz
            and abs(job_stop - stop_frequency_hz) <= endpoint_tolerance_hz
        ):
            matches.append(
                {
                    "job_id": job["job_id"],
                    "job_type": job["job_type"],
                    "created_at": job["created_at"],
                    "completed_at": job["completed_at"],
                    "start_frequency_hz": job_start,
                    "stop_frequency_hz": job_stop,
                    "summary": job.get("summary"),
                }
            )
            if len(matches) >= limit:
                break
    return {"count": len(matches), "surveys": matches}


@mcp.tool()
def compare_band_surveys(
    baseline_job_id: str,
    comparison_job_id: str,
    frequency_tolerance_hz: float = 1_500,
    power_change_threshold_db: float = 6.0,
    frequency_shift_threshold_hz: float = 250,
    include_plot: bool = True,
) -> CallToolResult:
    """Compare two persisted band scans/surveys and report RF-environment changes.

    This receiver-free operation matches nearby carriers and identifies new,
    disappeared, frequency-shifted, relatively stronger/weaker, and differently
    classified signals. Relative power changes are not calibrated dBm.
    """
    if baseline_job_id == comparison_job_id:
        raise ValueError("baseline_job_id and comparison_job_id must be different")
    baseline_job = catalog.get_job(baseline_job_id)
    comparison_job = catalog.get_job(comparison_job_id)
    for role, job in (("baseline", baseline_job), ("comparison", comparison_job)):
        if job["job_type"] not in SUPPORTED_JOB_TYPES:
            raise ValueError(f"{role} job must be a band_scan or band_survey")
        if not isinstance(job.get("result"), dict):
            raise ValueError(f"{role} job has no readable persisted result")

    ensure_data_dirs()
    started_at = datetime.now(timezone.utc).isoformat()
    result = compare_survey_results(
        baseline_job["result"],
        comparison_job["result"],
        frequency_tolerance_hz=frequency_tolerance_hz,
        power_change_threshold_db=power_change_threshold_db,
        frequency_shift_threshold_hz=frequency_shift_threshold_hz,
    )
    comparison_id = f"compare-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    result.update(
        {
            "job_id": comparison_id,
            "baseline_job_id": baseline_job_id,
            "comparison_job_id": comparison_job_id,
            "baseline_created_at": baseline_job["created_at"],
            "comparison_created_at": comparison_job["created_at"],
            "started_at": started_at,
        }
    )
    plot_path = PLOT_DIR / f"{comparison_id}-band-change.png"
    save_comparison_plot(plot_path, result)
    result["comparison_plot_path"] = str(plot_path.resolve())
    result_path = RESULT_DIR / f"{comparison_id}.json"
    result["result_json_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _persist_one_shot(
        job_id=comparison_id,
        job_type="survey_comparison",
        result=result,
        config={
            "baseline_job_id": baseline_job_id,
            "comparison_job_id": comparison_job_id,
            "frequency_tolerance_hz": frequency_tolerance_hz,
            "power_change_threshold_db": power_change_threshold_db,
            "frequency_shift_threshold_hz": frequency_shift_threshold_hz,
        },
        artifacts=[(plot_path, "survey_comparison_plot")],
    )
    summary = (
        f"Compared {baseline_job_id} with {comparison_job_id}: "
        f"{result['new_count']} new, {result['disappeared_count']} disappeared, "
        f"{result['changed_count']} changed, and {result['stable_count']} stable matched signals. "
        f"Power scale: {result['power_comparison_scale']}. Measurements are not calibrated dBm.\n\n"
        + json.dumps(result, indent=2)
    )
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=summary)]
    if include_plot:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def get_server_health() -> dict:
    """Report MiniRackDisplay RF MCP, receiver, database, job, and storage health."""
    active_job = active_long_job()
    if active_job:
        receiver = {"connected": True, "state": "busy", "busy_by_job_id": active_job}
        receiver_error = None
    else:
        try:
            receiver = device_info()
            receiver_error = None
        except Exception as exc:
            receiver = None
            receiver_error = f"{type(exc).__name__}: {exc}"
    storage = catalog.storage_status()
    database = catalog.database_health()
    return {
        "status": (
            "healthy"
            if receiver is not None and database["status"] == "healthy"
            else "degraded"
        ),
        "server_name": "MiniRackDisplay",
        "rf_mcp_version": __version__,
        "api_version": API_VERSION,
        "authentication": {
            "required": bool(os.getenv("RF_MCP_API_TOKEN", "").strip()),
            "scheme": "bearer" if os.getenv("RF_MCP_API_TOKEN", "").strip() else None,
        },
        "started_at": _SERVER_STARTED_AT,
        "uptime_seconds": time.monotonic() - _SERVER_STARTED_MONOTONIC,
        "active_long_job": active_job,
        "jobs_marked_interrupted_on_startup": _INTERRUPTED_JOBS_ON_STARTUP,
        "receiver": receiver,
        "receiver_error": receiver_error,
        "database": database,
        "storage": storage,
    }


@mcp.tool()
def list_rf_jobs(job_type: str | None = None, state: str | None = None, limit: int = 50) -> dict:
    """List persisted RF inspections, analyses, monitors, and band scans."""
    jobs = catalog.list_jobs(job_type=job_type, state=state, limit=limit)
    return {"count": len(jobs), "jobs": jobs}


@mcp.tool()
def get_rf_job(job_id: str) -> dict:
    """Get a persisted RF job, its result JSON, and associated artifact IDs."""
    return catalog.get_job(job_id)


@mcp.tool()
def list_rf_artifacts(
    kind: str | None = None,
    job_id: str | None = None,
    pinned: bool | None = None,
    limit: int = 50,
) -> dict:
    """List persistent RF plots, audio, JSON, and retained IQ artifacts."""
    artifacts = [
        _artifact_metadata(item)
        for item in catalog.list_artifacts(kind=kind, job_id=job_id, pinned=pinned, limit=limit)
    ]
    return {"count": len(artifacts), "artifacts": artifacts}


@mcp.tool()
def get_rf_artifact(artifact_id: str, include_content: bool = True) -> CallToolResult:
    """Retrieve a persistent artifact by stable ID as native MCP image, audio, or text."""
    artifact = _artifact_metadata(catalog.get_artifact(artifact_id))
    path = Path(artifact["path"])
    metadata_text = json.dumps(artifact, indent=2)
    content: list[TextContent | ImageContent | AudioContent] = [
        TextContent(type="text", text=metadata_text)
    ]
    if include_content:
        if artifact["size_bytes"] > _MAX_INLINE_ARTIFACT_BYTES:
            content.append(
                TextContent(
                    type="text",
                    text=(
                        f"Artifact content was not inlined because it is {artifact['size_bytes']:,} "
                        f"bytes; the server limit is {_MAX_INLINE_ARTIFACT_BYTES:,} bytes."
                    ),
                )
            )
        elif artifact["mime_type"].startswith("image/"):
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(path.read_bytes()).decode("ascii"),
                    mimeType=artifact["mime_type"],
                )
            )
        elif artifact["mime_type"].startswith("audio/"):
            content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(path.read_bytes()).decode("ascii"),
                    mimeType=artifact["mime_type"],
                )
            )
        elif artifact["mime_type"] in {"application/json", "text/plain"}:
            content.append(TextContent(type="text", text=path.read_text(encoding="utf-8")))
        else:
            content.append(
                TextContent(type="text", text="This artifact type is cataloged but not inlineable.")
            )
    return CallToolResult(content=content, structuredContent=artifact)


@mcp.tool()
def set_rf_artifact_pinned(artifact_id: str, pinned: bool = True) -> dict:
    """Pin or unpin an RF artifact. Cleanup never removes pinned artifacts."""
    return catalog.set_pinned(artifact_id, pinned)


@mcp.tool()
def get_storage_status() -> dict:
    """Report RF artifact usage by type and MiniRackDisplay filesystem free space."""
    return catalog.storage_status()


@mcp.tool()
def clean_old_artifacts(
    older_than_days: float = 30,
    kinds: list[str] | None = None,
    max_delete: int = 100,
    dry_run: bool = True,
    confirm_delete: bool = False,
) -> dict:
    """Preview or delete unpinned RF artifacts older than a specified age.

    The safe default is a dry run. Actual deletion requires dry_run=false and
    confirm_delete=true. Only files inside the RF MCP data directory are eligible.
    """
    if not dry_run and not confirm_delete:
        raise ValueError("Actual deletion requires confirm_delete=true")
    return catalog.cleanup(
        older_than_days=older_than_days,
        kinds=kinds,
        max_delete=max_delete,
        dry_run=dry_run,
    )


def _resolve_session_artifacts(
    artifact_ids: list[str] | None = None, job_ids: list[str] | None = None,
) -> list[dict]:
    artifact_ids, job_ids = artifact_ids or [], job_ids or []
    if len(artifact_ids) > 100 or len(job_ids) > 50:
        raise ValueError("A session accepts at most 100 artifact IDs and 50 job IDs per request")
    resolved = [catalog.get_artifact(identifier) for identifier in artifact_ids]
    for job_id in job_ids:
        resolved.extend(catalog.get_job(job_id)["artifacts"])
    unique = {}
    for artifact in resolved:
        unique[artifact["artifact_id"]] = artifact
    return list(unique.values())


@mcp.tool()
def create_recording_session(
    name: str, description: str = "", tags: list[str] | None = None,
    artifact_ids: list[str] | None = None, job_ids: list[str] | None = None,
) -> dict:
    """Create a review session containing selected catalog artifacts or whole jobs."""
    return create_session(
        name=name, description=description, tags=tags,
        artifacts=_resolve_session_artifacts(artifact_ids, job_ids),
    )


@mcp.tool()
def add_recording_session_items(
    session_id_or_name: str, artifact_ids: list[str] | None = None,
    job_ids: list[str] | None = None,
) -> dict:
    """Attach additional artifacts or all artifacts from selected jobs to a session."""
    artifacts = _resolve_session_artifacts(artifact_ids, job_ids)
    if not artifacts:
        raise ValueError("Provide at least one artifact_id or job_id")
    return add_session_artifacts(session_id_or_name, artifacts)


@mcp.tool()
def list_recording_sessions(limit: int = 100) -> dict:
    """List persistent recording/review sessions, newest updates first."""
    sessions = list_sessions()[:max(1, min(int(limit), 200))]
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def get_recording_session(session_id_or_name: str) -> dict:
    """Get a session with its items, annotations, and bookmarks."""
    return get_session(session_id_or_name)


@mcp.tool()
def search_recording_sessions(query: str, limit: int = 50) -> dict:
    """Search session names, descriptions, tags, filenames, notes, and bookmarks."""
    sessions = search_sessions(query)[:max(1, min(int(limit), 200))]
    return {"query": query, "count": len(sessions), "sessions": sessions}


@mcp.tool()
def add_recording_annotation(
    session_id_or_name: str, text: str, artifact_id: str | None = None,
    start_seconds: float | None = None, end_seconds: float | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Add a session-wide or time-ranged artifact annotation."""
    return add_session_annotation(
        session_id_or_name, text=text, artifact_id=artifact_id,
        start_seconds=start_seconds, end_seconds=end_seconds, tags=tags,
    )


@mcp.tool()
def add_recording_bookmark(
    session_id_or_name: str, artifact_id: str, position_seconds: float,
    label: str, notes: str = "",
) -> dict:
    """Bookmark one point in a cataloged WAV, attaching it to the session if needed."""
    return add_session_bookmark(
        session_id_or_name, artifact=catalog.get_artifact(artifact_id),
        position_seconds=position_seconds, label=label, notes=notes,
    )


@mcp.tool()
def extract_recording_clip(
    artifact_id: str, start_seconds: float, duration_seconds: float,
    label: str = "clip", session_id_or_name: str | None = None,
) -> CallToolResult:
    """Extract a bounded WAV clip without changing the source recording."""
    source = catalog.get_artifact(artifact_id)
    if source["mime_type"] not in {"audio/wav", "audio/x-wav"} and not source["filename"].lower().endswith(".wav"):
        raise ValueError("Clip extraction currently supports WAV artifacts only")
    path, details = extract_wav_clip(
        Path(source["path"]), start_seconds=start_seconds,
        duration_seconds=duration_seconds, label=label,
    )
    artifact = catalog.register_artifact(path, "recording_clip", job_id=source.get("job_id"))
    session = None
    if session_id_or_name:
        session = add_session_artifacts(session_id_or_name, [artifact])
    result = {"source_artifact_id": artifact_id, "clip_artifact": _artifact_metadata(artifact),
              "clip_info": details, "session_id": session["session_id"] if session else None}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2)),
                 AudioContent(type="audio", data=base64.b64encode(path.read_bytes()).decode("ascii"),
                              mimeType="audio/wav")], structuredContent=result,
    )


@mcp.tool()
def compare_recording_audio(
    first_artifact_id: str, second_artifact_id: str, include_plot: bool = True,
) -> CallToolResult:
    """Compare up to the first 120 seconds of two WAV artifacts."""
    artifacts = [catalog.get_artifact(first_artifact_id), catalog.get_artifact(second_artifact_id)]
    for artifact in artifacts:
        if artifact["mime_type"] not in {"audio/wav", "audio/x-wav"} and not artifact["filename"].lower().endswith(".wav"):
            raise ValueError("Audio comparison currently supports WAV artifacts only")
    metrics, plot_path = compare_wav(Path(artifacts[0]["path"]), Path(artifacts[1]["path"]))
    job_id = f"audio-compare-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    result = {"job_id": job_id, "first_artifact_id": first_artifact_id,
              "second_artifact_id": second_artifact_id, "metrics": metrics,
              "comparison_plot_path": str(plot_path.resolve()),
              "started_at": datetime.now(timezone.utc).isoformat()}
    result_path = RESULT_DIR / f"{job_id}.json"; result["result_json_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _persist_one_shot(job_id=job_id, job_type="audio_comparison", result=result,
                      config={"first_artifact_id": first_artifact_id,
                              "second_artifact_id": second_artifact_id},
                      artifacts=[(plot_path, "audio_comparison_plot")])
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result, indent=2))]
    if include_plot:
        content.append(ImageContent(type="image",
                                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                                    mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result)


@mcp.tool()
def export_recording_session(session_id_or_name: str) -> dict:
    """Export a session manifest as JSON and its annotations as CSV."""
    session = get_session(session_id_or_name)
    json_path, csv_path = export_session(session)
    artifacts = [catalog.register_artifact(json_path, "recording_session_json"),
                 catalog.register_artifact(csv_path, "recording_annotations_csv")]
    return {"session_id": session["session_id"],
            "artifacts": [_artifact_metadata(item) for item in artifacts]}


@mcp.tool()
def delete_recording_session(
    session_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete session metadata only; attached artifacts are retained."""
    if not confirm_delete:
        raise ValueError("Session deletion requires confirm_delete=true")
    return {"deleted": True, "artifacts_retained": True,
            "session": delete_session(session_id_or_name)}


@mcp.tool()
def classify_signal(
    frequency_hz: int,
    duration_seconds: float = 5.0,
    analysis_bandwidth_hz: int = 30_000,
    fft_size: int = 16_384,
    include_plot: bool = True,
    include_preview_audio: bool = True,
    retain_iq: bool = False,
    receiver_id: str | None = None,
) -> CallToolResult:
    """Heuristically classify a live signal as AM, USB, LSB, CW, NFM, or digital/unknown.

    Classification is feature-based and probabilistic, not authoritative. The tool
    returns ranked candidates, diagnostic features, a plot, and optional audio
    demodulated using the highest-ranked supported mode.
    """
    analysis_bandwidth_hz = int(analysis_bandwidth_hz)
    receiver_center_hz = offset_capture_center(int(frequency_hz), receiver_id=receiver_id)
    capture_options = {"receiver_id": receiver_id} if receiver_id is not None else {}
    capture = capture_iq(receiver_center_hz, duration_seconds, **capture_options)
    try:
        iq = load_complex_float32(capture.path)
        target_offset_hz = int(frequency_hz) - capture.center_frequency_hz
        baseband = downconvert(iq, capture.sample_rate_hz, target_offset_hz)
        features, frequencies, power_db, time_axis, instantaneous_frequency = extract_features(
            baseband,
            capture.sample_rate_hz,
            analysis_bandwidth_hz,
            fft_size,
        )
        ranking = classify_features(features)
        best = ranking[0]
        margin = best["confidence"] - ranking[1]["confidence"]
        ambiguous = bool(best["confidence"] < 0.35 or margin < 0.08)
        recommended_mode = (
            best["label"] if best["label"] in {"am", "usb", "lsb", "cw", "nfm"} else None
        )

        stem = f"{capture.path.stem}-{int(frequency_hz)}-classification"
        plot_path = PLOT_DIR / f"{stem}.png"
        save_classification_plot(
            plot_path,
            int(frequency_hz),
            frequencies,
            power_db,
            time_axis,
            instantaneous_frequency,
            ranking,
        )

        wav_path: Path | None = None
        if recommended_mode is not None:
            preview_bandwidth = {
                "am": min(10_000, analysis_bandwidth_hz),
                "usb": min(3_000, analysis_bandwidth_hz),
                "lsb": min(3_000, analysis_bandwidth_hz),
                "cw": 500,
                "nfm": max(5_000, min(12_500, analysis_bandwidth_hz)),
            }[recommended_mode]
            audio = demodulate(
                baseband,
                capture.sample_rate_hz,
                recommended_mode,
                preview_bandwidth,
            )
            wav_path = AUDIO_DIR / f"{stem}-{recommended_mode}-preview.wav"
            write_wav(wav_path, audio)

        result = {
            "classification_method": "deterministic_heuristic_features_v1",
            "classification_is_authoritative": False,
            "receiver_id": getattr(capture, "receiver_id", receiver_id),
            "receiver_backend": getattr(capture, "backend", None),
            "requested_frequency_hz": int(frequency_hz),
            "receiver_center_frequency_hz": capture.center_frequency_hz,
            "receiver_offset_hz": target_offset_hz,
            "duration_seconds": capture.captured_samples / capture.sample_rate_hz,
            "best_label": best["label"],
            "best_confidence": best["confidence"],
            "confidence_margin": margin,
            "ambiguous": ambiguous,
            "recommended_demodulation_mode": recommended_mode,
            "ranking": ranking,
            "features": feature_dict(features),
            "classification_plot_path": str(plot_path.resolve()),
            "preview_audio_path": str(wav_path.resolve()) if wav_path else None,
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at,
        }
        result_path = RESULT_DIR / f"{stem}.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _persist_one_shot(
            job_id=f"classify-{capture.path.stem}",
            job_type="signal_classification",
            result=result,
            config={
                "frequency_hz": frequency_hz,
                "duration_seconds": duration_seconds,
                "analysis_bandwidth_hz": analysis_bandwidth_hz,
                "fft_size": fft_size,
                "retain_iq": retain_iq,
            },
            artifacts=[
                (plot_path, "classification_plot"),
                (wav_path, "classification_audio_preview"),
                (capture.path if retain_iq else None, "iq_capture"),
            ],
        )

        qualifier = "ambiguous" if ambiguous else "provisional"
        summary = (
            f"Heuristic classification at {frequency_hz:,} Hz is {best['label'].upper()} "
            f"with {best['confidence']:.1%} normalized confidence ({qualifier}). "
            f"This is feature-based guidance, not an authoritative decoder result.\n\n"
            + json.dumps(result, indent=2)
        )
        content: list[TextContent | ImageContent | AudioContent] = [
            TextContent(type="text", text=summary)
        ]
        if include_plot:
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(plot_path.read_bytes()).decode("ascii"),
                    mimeType="image/png",
                )
            )
        if include_preview_audio and wav_path is not None:
            content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                    mimeType="audio/wav",
                )
            )
        return CallToolResult(content=content, structuredContent=result)
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)


def _classification_result(job_id: str) -> dict:
    job = catalog.get_job(job_id)
    if job["job_type"] != "signal_classification" or not isinstance(job.get("result"), dict):
        raise ValueError("job_id must identify a readable signal_classification job")
    return job["result"]


@mcp.tool()
def save_signal_fingerprint(
    name: str, classification_job_id: str, notes: str = "",
    frequency_tolerance_hz: float = 2_500, replace_existing: bool = False,
) -> dict:
    """Create a named station-local fingerprint from a persisted classification job."""
    return save_fingerprint(
        name=name, observation=_classification_result(classification_job_id),
        notes=notes, frequency_tolerance_hz=frequency_tolerance_hz,
        replace_existing=replace_existing,
    )


@mcp.tool()
def add_signal_fingerprint_exemplar(
    fingerprint_id_or_name: str, classification_job_id: str,
) -> dict:
    """Add another observation to a fingerprint, retaining at most 20 exemplars."""
    return add_exemplar(
        fingerprint_id_or_name, _classification_result(classification_job_id)
    )


@mcp.tool()
def list_signal_fingerprints() -> dict:
    """List known local signals ordered by nominal frequency."""
    fingerprints = list_fingerprints()
    return {"count": len(fingerprints), "fingerprints": fingerprints}


@mcp.tool()
def get_signal_fingerprint(fingerprint_id_or_name: str) -> dict:
    """Get one fingerprint, including its exemplars and centroid features."""
    return get_fingerprint(fingerprint_id_or_name)


@mcp.tool()
def delete_signal_fingerprint(
    fingerprint_id_or_name: str, confirm_delete: bool = False,
) -> dict:
    """Delete a fingerprint only when confirm_delete=true."""
    if not confirm_delete:
        raise ValueError("Fingerprint deletion requires confirm_delete=true")
    return {"deleted": True, "fingerprint": delete_fingerprint(fingerprint_id_or_name)}


@mcp.tool()
def match_signal_classification_job(
    classification_job_id: str, minimum_similarity: float = 0.55, limit: int = 10,
) -> dict:
    """Compare a prior classification against the saved signal library without RF capture."""
    result = match_fingerprints(
        _classification_result(classification_job_id),
        minimum_similarity=minimum_similarity, limit=limit,
    )
    result["classification_job_id"] = classification_job_id
    return result


@mcp.tool()
def identify_live_signal(
    frequency_hz: int, duration_seconds: float = 5.0,
    analysis_bandwidth_hz: int = 30_000, fft_size: int = 16_384,
    minimum_similarity: float = 0.55, include_plot: bool = True,
    include_preview_audio: bool = True,
) -> CallToolResult:
    """Classify a live signal and compare it with station-local fingerprints."""
    classified = classify_signal(
        frequency_hz=frequency_hz, duration_seconds=duration_seconds,
        analysis_bandwidth_hz=analysis_bandwidth_hz, fft_size=fft_size,
        include_plot=include_plot, include_preview_audio=include_preview_audio,
        retain_iq=False,
    )
    observation = dict(classified.structuredContent or {})
    matching = match_fingerprints(observation, minimum_similarity=minimum_similarity)
    identification_id = f"identify-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    result = {
        "job_id": identification_id, "requested_frequency_hz": int(frequency_hz),
        "classification_job_id": observation.get("job_id"),
        "generic_classification": {
            "best_label": observation.get("best_label"),
            "best_confidence": observation.get("best_confidence"),
            "ambiguous": observation.get("ambiguous"),
        },
        "identification": matching, "started_at": observation.get("started_at"),
    }
    result_path = RESULT_DIR / f"{identification_id}.json"
    result["result_json_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _persist_one_shot(
        job_id=identification_id, job_type="signal_identification", result=result,
        config={"frequency_hz": frequency_hz, "duration_seconds": duration_seconds,
                "analysis_bandwidth_hz": analysis_bandwidth_hz, "fft_size": fft_size,
                "minimum_similarity": minimum_similarity}, artifacts=[],
    )
    best = matching.get("best_match")
    statement = (f"Matched saved signal {best['name']} at {best['similarity']:.1%} similarity."
                 if best else "No saved signal met the requested similarity threshold.")
    content = [TextContent(type="text", text=statement + "\n\n" + json.dumps(result, indent=2))]
    content.extend(classified.content[1:])
    return CallToolResult(content=content, structuredContent=result)


def main() -> None:
    transport = os.getenv("RF_MCP_TRANSPORT", "streamable-http")
    token = validate_api_token(os.getenv("RF_MCP_API_TOKEN"))
    if transport != "streamable-http" and token is not None:
        raise ValueError("RF_MCP_API_TOKEN is only supported with streamable-http")
    webhook_dispatcher.start()
    scheduler_manager.start()
    satellite_scheduler.start()
    try:
        if transport != "streamable-http":
            mcp.run(transport=transport)
            return

        import uvicorn

        def run_station_profile_for_web(preset_id_or_name: str) -> dict:
            preset = catalog.get_preset(preset_id_or_name)
            if preset.get("preset_type") != "station_memory_scan":
                raise ValueError("preset_id_or_name must identify a station_memory_scan profile")
            response = _execute_rf_preset(preset)
            return dict(response.structuredContent or {})

        def delete_station_profile_for_web(
            preset_id_or_name: str, confirm_delete: bool = False,
        ) -> dict:
            preset = catalog.get_preset(preset_id_or_name)
            if preset.get("preset_type") != "station_memory_scan":
                raise ValueError("Only station_memory_scan profiles can be deleted here")
            return delete_rf_preset(preset_id_or_name, confirm_delete=confirm_delete)

        def run_preset_for_web(preset_id_or_name: str) -> dict:
            response = _execute_rf_preset(catalog.get_preset(preset_id_or_name))
            return dict(response.structuredContent or {})

        def decode_native_for_web(**values) -> dict:
            response = decode_digital_signal(**values)
            return dict(response.structuredContent or {})

        application_services = RfApplicationServices(
                       catalog=catalog, receivers=receiver_service,
                       spectrum_capture=inspect_spectrum,
                       signal_analyzer=analyze_signal,
                       broadcast_fm_receiver=receive_broadcast_fm,
                       live_audio=live_audio_manager,
                       live_waterfall=live_waterfall_manager)
        app = RfWebApp(mcp.streamable_http_app(), catalog, token, __version__,
                       inspect_spectrum, analyze_signal, receive_broadcast_fm,
                       save_rf_schedule, run_rf_schedule_now, set_rf_schedule_enabled,
                       save_station_memory_scan_profile, run_station_profile_for_web,
                       save_station_memory, delete_station_memory,
                       delete_station_profile_for_web, delete_rf_schedule,
                       start_band_scan, start_band_survey, get_band_scan_status,
                       stop_band_scan, run_preset_for_web, survey_broadcast_fm,
                       get_fm_survey_status, stop_fm_survey, decode_native_for_web,
                       decode_weak_signal, decode_fldigi_mode,
                       list_digital_decoder_capabilities,
                       sstv_decode=decode_sstv,
                       sstv_watch_start=start_sstv_watcher,
                       sstv_status=get_sstv_status,
                       sstv_watch_status=get_sstv_watcher_status,
                       sstv_stop=stop_sstv,
                       sstv_watch_stop=stop_sstv_watcher,
                       sstv_capabilities=list_sstv_decoder_capabilities,
                       services=application_services)
        uvicorn.run(
            app,
            host=os.getenv("RF_MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("RF_MCP_PORT", "8765")),
        )
    finally:
        live_audio_manager.shutdown()
        live_waterfall_manager.shutdown()
        satellite_scheduler.stop()
        scheduler_manager.stop()
        webhook_dispatcher.stop()
        catalog.close()


if __name__ == "__main__":
    main()
