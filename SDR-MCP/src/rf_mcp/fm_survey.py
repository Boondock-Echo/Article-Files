from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .receiver_backend import capture_iq, offset_capture_center
from .catalog import catalog
from .config import AUDIO_DIR, FM_SURVEY_DIR, PLOT_DIR, SAMPLE_RATE, ensure_data_dirs
from .operations import acquire_long_job, release_long_job
from .rds import decode_rds
from .signal_analysis import (
    demodulate_broadcast_fm,
    downconvert,
    save_broadcast_fm_plot,
    write_wav,
)
from .spectrum import averaged_psd_dbfs_per_hz, load_complex_float32


TERMINAL_STATES = {"completed", "stopped", "failed", "interrupted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fm_channel_plan(start_hz: int, stop_hz: int, spacing_hz: int) -> list[int]:
    start_hz, stop_hz, spacing_hz = int(start_hz), int(stop_hz), int(spacing_hz)
    if not 76_000_000 <= start_hz <= 110_000_000:
        raise ValueError("start_frequency_hz must be from 76 through 110 MHz")
    if not start_hz <= stop_hz <= 110_000_000:
        raise ValueError("stop_frequency_hz must be ordered and no greater than 110 MHz")
    if spacing_hz not in {50_000, 100_000, 200_000}:
        raise ValueError("channel_spacing_hz must be 50000, 100000, or 200000")
    count = ((stop_hz - start_hz) // spacing_hz) + 1
    if count > 700:
        raise ValueError("FM survey channel plan is too large")
    return [start_hz + index * spacing_hz for index in range(count)]


def fm_candidate_score(
    iq: np.ndarray, center_frequency_hz: int, sample_rate_hz: int, target_frequency_hz: int
) -> dict:
    frequencies, psd = averaged_psd_dbfs_per_hz(
        iq, center_frequency_hz, sample_rate_hz, 4096
    )
    offset = np.abs(frequencies - target_frequency_hz)
    signal = offset <= 100_000
    noise = (offset >= 180_000) & (offset <= 300_000)
    if not np.any(noise):
        noise = offset >= 130_000
    # A broadcast carrier is narrow even though its modulation occupies the
    # surrounding channel, so preserve the strongest in-channel spectral bin.
    signal_level = float(np.max(psd[signal]))
    noise_level = float(np.median(psd[noise]))
    return {
        "discovery_signal_dbfs_hz": signal_level,
        "discovery_noise_dbfs_hz": noise_level,
        "discovery_score_db": signal_level - noise_level,
    }


def station_record(frequency_hz: int, metrics: dict, rds: dict | None) -> dict:
    metadata = (rds or {}).get("station", {})
    music_speech = metadata.get("music_speech")
    return {
        "frequency_hz": int(frequency_hz),
        "pi_code": metadata.get("pi_code"),
        "ps": metadata.get("program_service"),
        "pty": metadata.get("program_type"),
        "pty_name": metadata.get("pty_name"),
        "ptyn": metadata.get("program_type_name"),
        "radiotext": metadata.get("radiotext"),
        "tp": metadata.get("traffic_program"),
        "ta": metadata.get("traffic_announcement"),
        "music_speech": (
            "music" if music_speech is True else "speech" if music_speech is False else None
        ),
        "alternative_frequencies_hz": [
            int(round(value * 1_000_000))
            for value in metadata.get("alternative_frequencies_mhz", [])
        ],
        "stereo_detected": bool(metrics.get("stereo_detected")),
        "estimated_snr_db": metrics.get("estimated_snr_db"),
        "pilot_to_composite_rms_db": metrics.get("pilot_to_composite_rms_db"),
        "rds_group_count": int((rds or {}).get("group_count", 0)),
    }


def compare_fm_survey_results(baseline: dict, comparison: dict) -> dict:
    before = {int(item["frequency_hz"]): item for item in baseline.get("stations", [])}
    after = {int(item["frequency_hz"]): item for item in comparison.get("stations", [])}
    shared = sorted(before.keys() & after.keys())
    fields = ("pi_code", "ps", "pty", "ptyn", "radiotext", "tp", "ta",
              "music_speech", "stereo_detected")
    changed, stable = [], []
    for frequency_hz in shared:
        changes = {
            field: {"before": before[frequency_hz].get(field),
                    "after": after[frequency_hz].get(field)}
            for field in fields
            if before[frequency_hz].get(field) != after[frequency_hz].get(field)
        }
        entry = {"frequency_hz": frequency_hz, "changes": changes}
        (changed if changes else stable).append(entry)
    new_keys = sorted(after.keys() - before.keys())
    gone_keys = sorted(before.keys() - after.keys())
    return {
        "baseline_job_id": baseline.get("job_id"),
        "comparison_job_id": comparison.get("job_id"),
        "new_stations": [after[key] for key in new_keys],
        "disappeared_stations": [before[key] for key in gone_keys],
        "changed_stations": changed, "stable_stations": stable,
        "new_count": len(new_keys), "disappeared_count": len(gone_keys),
        "changed_count": len(changed), "stable_count": len(stable),
    }


@dataclass
class FMSurveyJob:
    job_id: str
    config: dict[str, Any]
    channels_hz: list[int]
    state: str = "queued"
    phase: str = "discovery"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    discovery_index: int = 0
    station_index: int = 0
    candidates: list[dict] = field(default_factory=list)
    stations: list[dict] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class FMSurveyManager:
    def __init__(self) -> None:
        self._jobs: dict[str, FMSurveyJob] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
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
        if not 0.25 <= float(discovery_duration_seconds) <= 2:
            raise ValueError("discovery_duration_seconds must be from 0.25 through 2")
        if not 3 <= float(discovery_threshold_db) <= 40:
            raise ValueError("discovery_threshold_db must be from 3 through 40")
        if not 1 <= float(rds_duration_seconds) <= 10:
            raise ValueError("rds_duration_seconds must be from 1 through 10")
        if deemphasis_us not in {50, 75}:
            raise ValueError("deemphasis_us must be 50 or 75")
        if not isinstance(save_audio, bool) or not isinstance(save_plots, bool):
            raise ValueError("save_audio and save_plots must be JSON booleans")

        if resume_job_id:
            job = self._restore(resume_job_id)
            if job.state == "completed":
                raise ValueError("A completed FM survey cannot be resumed")
            job.stop_event.clear()
            job.state = "queued"
            job.error = None
        else:
            channels = fm_channel_plan(start_frequency_hz, stop_frequency_hz, channel_spacing_hz)
            job_id = f"fm-survey-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
            config = {
                "start_frequency_hz": channels[0], "stop_frequency_hz": channels[-1],
                "channel_spacing_hz": int(channel_spacing_hz),
                "discovery_duration_seconds": float(discovery_duration_seconds),
                "discovery_threshold_db": float(discovery_threshold_db),
                "rds_duration_seconds": float(rds_duration_seconds),
                "deemphasis_us": int(deemphasis_us), "save_audio": save_audio,
                "save_plots": save_plots, "channel_count": len(channels),
            }
            job = FMSurveyJob(job_id, config, channels)

        with self._lock:
            acquire_long_job(job.job_id)
            try:
                self._jobs[job.job_id] = job
                self._checkpoint(job)
                job.thread = threading.Thread(
                    target=self._run, args=(job,), name=f"SDR-MCP-{job.job_id}", daemon=True
                )
                job.thread.start()
            except Exception:
                self._jobs.pop(job.job_id, None)
                release_long_job(job.job_id)
                raise
        return self.status(job.job_id)

    def _restore(self, job_id: str) -> FMSurveyJob:
        with self._lock:
            current = self._jobs.get(job_id)
        if current and current.state not in {"running", "queued"}:
            return current
        persisted = catalog.get_job(job_id)
        if persisted["job_type"] != "fm_broadcast_survey" or not persisted.get("result"):
            raise ValueError("resume_job_id must identify a persisted FM broadcast survey")
        result = persisted["result"]
        return FMSurveyJob(
            job_id=job_id, config=persisted["config"], channels_hz=result["channels_hz"],
            state=persisted["state"], phase=result["phase"],
            created_at=persisted["created_at"], started_at=persisted["started_at"],
            completed_at=persisted["completed_at"], error=persisted["error"],
            discovery_index=result.get("discovery_index", 0),
            station_index=result.get("station_index", 0),
            candidates=result.get("candidates", []), stations=result.get("stations", []),
        )

    def _run(self, job: FMSurveyJob) -> None:
        ensure_data_dirs()
        job.state = "running"
        job.started_at = job.started_at or utc_now()
        try:
            if job.phase == "discovery":
                for index in range(job.discovery_index, len(job.channels_hz)):
                    if job.stop_event.is_set():
                        break
                    frequency_hz = job.channels_hz[index]
                    capture = capture_iq(
                        offset_capture_center(frequency_hz, offset_hz=150_000),
                        job.config["discovery_duration_seconds"],
                    )
                    try:
                        score = fm_candidate_score(
                            load_complex_float32(capture.path), capture.center_frequency_hz,
                            capture.sample_rate_hz, frequency_hz,
                        )
                    finally:
                        Path(capture.path).unlink(missing_ok=True)
                    if score["discovery_score_db"] >= job.config["discovery_threshold_db"]:
                        job.candidates.append({"frequency_hz": frequency_hz, **score})
                    job.discovery_index = index + 1
                    self._checkpoint(job)
                if not job.stop_event.is_set():
                    job.phase = "rds_collection"
                    self._checkpoint(job)

            if job.phase == "rds_collection" and not job.stop_event.is_set():
                for index in range(job.station_index, len(job.candidates)):
                    if job.stop_event.is_set():
                        break
                    self._collect_station(job, job.candidates[index])
                    job.station_index = index + 1
                    self._checkpoint(job)

            if job.stop_event.is_set():
                job.state = "stopped"
            else:
                job.phase = "finished"
                job.state = "completed"
                job.completed_at = utc_now()
                self._write_csv(job)
            self._checkpoint(job)
            if job.state == "completed":
                catalog.register_artifact(
                    FM_SURVEY_DIR / f"{job.job_id}.json",
                    "fm_survey_json", job_id=job.job_id,
                )
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.completed_at = utc_now()
            self._checkpoint(job)
        finally:
            release_long_job(job.job_id)

    def _collect_station(self, job: FMSurveyJob, candidate: dict) -> None:
        frequency_hz = candidate["frequency_hz"]
        capture = capture_iq(
            offset_capture_center(frequency_hz, offset_hz=150_000),
            job.config["rds_duration_seconds"],
        )
        try:
            iq = load_complex_float32(capture.path)
            baseband = downconvert(
                iq, capture.sample_rate_hz, frequency_hz - capture.center_frequency_hz
            )
            audio, metrics, diagnostic = demodulate_broadcast_fm(
                baseband, capture.sample_rate_hz,
                deemphasis_us=job.config["deemphasis_us"], stereo=True,
            )
            rds = decode_rds(diagnostic["composite"], diagnostic["composite_sample_rate_hz"])
            record = station_record(frequency_hz, metrics, rds)
            record.update(candidate)
            record["estimated_snr_db"] = candidate["discovery_score_db"]
            record["observed_at"] = utc_now()
            record["audio_wav_path"] = None
            record["multiplex_plot_path"] = None
            stem = f"{job.job_id}-{frequency_hz}"
            if job.config["save_audio"]:
                path = AUDIO_DIR / f"{stem}.wav"
                write_wav(path, audio)
                record["audio_wav_path"] = str(path.resolve())
                catalog.register_artifact(path, "fm_survey_audio", job_id=job.job_id)
            if job.config["save_plots"]:
                path = PLOT_DIR / f"{stem}.png"
                save_broadcast_fm_plot(path, frequency_hz, diagnostic)
                record["multiplex_plot_path"] = str(path.resolve())
                catalog.register_artifact(path, "fm_survey_plot", job_id=job.job_id)
            catalog.upsert_fm_station(record, job_id=job.job_id, observed_at=record["observed_at"])
            job.stations.append(record)
        finally:
            Path(capture.path).unlink(missing_ok=True)

    def _result(self, job: FMSurveyJob) -> dict:
        total = len(job.channels_hz) + len(job.candidates)
        done = job.discovery_index + job.station_index
        return {
            "job_id": job.job_id, "state": job.state, "phase": job.phase,
            "created_at": job.created_at, "started_at": job.started_at,
            "completed_at": job.completed_at, "error": job.error,
            "channels_hz": job.channels_hz, "discovery_index": job.discovery_index,
            "station_index": job.station_index, "channel_count": len(job.channels_hz),
            "candidate_count": len(job.candidates), "station_count": len(job.stations),
            "progress_percent": round(100 * done / max(1, total), 1),
            "candidates": job.candidates, "stations": job.stations,
            "csv_path": str((FM_SURVEY_DIR / f"{job.job_id}.csv").resolve())
                if job.state == "completed" else None,
        }

    def _checkpoint(self, job: FMSurveyJob) -> None:
        result = self._result(job)
        path = FM_SURVEY_DIR / f"{job.job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        result["result_json_path"] = str(path.resolve())
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        catalog.upsert_job(
            job.job_id, "fm_broadcast_survey", job.state, config=job.config,
            summary={"phase": job.phase, "candidate_count": len(job.candidates),
                     "station_count": len(job.stations),
                     "discovery_index": job.discovery_index,
                     "station_index": job.station_index,
                     "channel_count": len(job.channels_hz),
                     "progress_percent": result["progress_percent"],
                     "candidate_frequencies_hz": [
                         item["frequency_hz"] for item in job.candidates
                     ],
                     "decoded_frequencies_hz": [
                         item["frequency_hz"] for item in job.stations
                     ]},
            result_json_path=path, created_at=job.created_at, started_at=job.started_at,
            completed_at=job.completed_at, error=job.error,
        )

    def _write_csv(self, job: FMSurveyJob) -> None:
        path = FM_SURVEY_DIR / f"{job.job_id}.csv"
        fields = ["frequency_hz", "pi_code", "ps", "pty", "pty_name", "ptyn",
                  "radiotext", "tp", "ta", "music_speech", "stereo_detected",
                  "estimated_snr_db", "rds_group_count", "observed_at"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(job.stations)
        catalog.register_artifact(path, "fm_survey_csv", job_id=job.job_id)

    def _get(self, job_id: str) -> FMSurveyJob:
        with self._lock:
            job = self._jobs.get(job_id)
        return job or self._restore(job_id)

    def status(self, job_id: str) -> dict:
        result = self._result(self._get(job_id))
        result.pop("candidates", None)
        result.pop("stations", None)
        return result

    def results(self, job_id: str) -> dict:
        return self._result(self._get(job_id))

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.state in TERMINAL_STATES:
            return self.status(job_id)
        job.stop_event.set()
        return self.status(job_id)


fm_survey_manager = FMSurveyManager()
