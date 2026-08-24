from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .receiver_backend import capture_iq, offset_capture_center, validate_duration, validate_frequency
from .catalog import catalog
from .config import AUDIO_DIR, PLOT_DIR, RESULT_DIR, SAMPLE_RATE, ensure_data_dirs
from .signal_analysis import (
    AUDIO_SAMPLE_RATE,
    demodulate,
    downconvert,
    measure_signal,
    normalize_mode,
    validate_bandwidth,
    write_wav,
)
from .operations import acquire_long_job, release_long_job
from .spectrum import averaged_spectrum, load_complex_float32


TERMINAL_STATES = {"completed", "stopped", "failed"}
MAX_MONITOR_SECONDS = 3_600
MAX_MONITOR_CAPTURES = 1_000


@dataclass
class MonitorSample:
    captured_at: str
    elapsed_seconds: float
    relative_peak_db: float
    relative_noise_floor_db: float
    estimated_snr_db: float
    dominant_frequency_hz: float
    dominant_offset_hz: float
    occupied_bandwidth_hz: float
    signal_present: bool
    signal_confidence: float
    duty_cycle_percent: float
    audio_clip_path: str | None = None


@dataclass
class MonitorJob:
    job_id: str
    config: dict[str, Any]
    state: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    samples: list[MonitorSample] = field(default_factory=list)
    waterfall_rows: list[np.ndarray] = field(default_factory=list)
    waterfall_frequencies_hz: np.ndarray | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    render_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class MonitorManager:
    def __init__(self) -> None:
        self._jobs: dict[str, MonitorJob] = {}
        self._lock = threading.RLock()

    def _get(self, job_id: str) -> MonitorJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown monitor job_id: {job_id}")
        return job

    def start(
        self,
        *,
        frequency_hz: int,
        mode: str,
        bandwidth_hz: int | None,
        total_duration_seconds: float,
        capture_duration_seconds: float,
        interval_seconds: float,
        fft_size: int,
        waterfall_span_hz: int,
        record_audio_on_activity: bool,
        source_preset_id: str | None = None,
        source_schedule_id: str | None = None,
    ) -> dict:
        frequency_hz = validate_frequency(frequency_hz)
        mode = normalize_mode(mode)
        bandwidth_hz = validate_bandwidth(mode, bandwidth_hz)
        capture_duration_seconds = validate_duration(capture_duration_seconds)
        total_duration_seconds = float(total_duration_seconds)
        interval_seconds = float(interval_seconds)
        if not 10 <= total_duration_seconds <= MAX_MONITOR_SECONDS:
            raise ValueError(f"total_duration_seconds must be from 10 through {MAX_MONITOR_SECONDS}")
        if interval_seconds < capture_duration_seconds or interval_seconds > 300:
            raise ValueError("interval_seconds must be at least capture duration and no more than 300")
        expected_captures = int(np.ceil(total_duration_seconds / interval_seconds))
        if expected_captures > MAX_MONITOR_CAPTURES:
            raise ValueError(f"Monitor would exceed the {MAX_MONITOR_CAPTURES}-capture limit")
        if fft_size < 1024 or fft_size > 65_536 or fft_size & (fft_size - 1):
            raise ValueError("fft_size must be a power of two from 1024 through 65536")
        waterfall_span_hz = int(waterfall_span_hz)
        if not bandwidth_hz <= waterfall_span_hz <= 600_000:
            raise ValueError("waterfall_span_hz must cover the signal bandwidth and be <= 600000")

        with self._lock:
            active = [job.job_id for job in self._jobs.values() if job.state not in TERMINAL_STATES]
            if active:
                raise RuntimeError(f"Monitor {active[0]} is already active")
            job_id = f"mon-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
            config = {
                "frequency_hz": frequency_hz,
                "mode": mode,
                "bandwidth_hz": bandwidth_hz,
                "total_duration_seconds": total_duration_seconds,
                "capture_duration_seconds": capture_duration_seconds,
                "interval_seconds": interval_seconds,
                "fft_size": fft_size,
                "waterfall_span_hz": waterfall_span_hz,
                "record_audio_on_activity": bool(record_audio_on_activity),
                "expected_captures": expected_captures,
            }
            if source_preset_id:
                config["source_preset_id"] = source_preset_id
            if source_schedule_id:
                config["source_schedule_id"] = source_schedule_id
            job = MonitorJob(job_id=job_id, config=config)
            acquire_long_job(job_id)
            try:
                self._jobs[job_id] = job
                catalog.upsert_job(
                    job_id,
                    "monitor",
                    "queued",
                    config=config,
                    created_at=job.created_at,
                )
                self._prune_locked()
                job.thread = threading.Thread(
                    target=self._run,
                    args=(job,),
                    name=f"rf-mcp-{job_id}",
                    daemon=True,
                )
                job.thread.start()
            except Exception:
                self._jobs.pop(job_id, None)
                release_long_job(job_id)
                raise
        return self.status(job_id)

    def _run(self, job: MonitorJob) -> None:
        ensure_data_dirs()
        config = job.config
        target_hz = int(config["frequency_hz"])
        receiver_center_hz = offset_capture_center(target_hz)
        with self._lock:
            job.state = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
        monotonic_start = time.monotonic()
        next_capture_at = monotonic_start
        try:
            catalog.upsert_job(
                job.job_id,
                "monitor",
                "running",
                config=job.config,
                created_at=job.created_at,
                started_at=job.started_at,
            )
            while not job.stop_event.is_set():
                elapsed = time.monotonic() - monotonic_start
                if elapsed >= config["total_duration_seconds"]:
                    break
                wait_seconds = next_capture_at - time.monotonic()
                if wait_seconds > 0 and job.stop_event.wait(wait_seconds):
                    break

                capture = capture_iq(receiver_center_hz, config["capture_duration_seconds"])
                try:
                    iq = load_complex_float32(capture.path)
                    target_offset_hz = target_hz - capture.center_frequency_hz
                    baseband = downconvert(iq, capture.sample_rate_hz, target_offset_hz)
                    frequencies, power_db = averaged_spectrum(
                        iq,
                        capture.center_frequency_hz,
                        capture.sample_rate_hz,
                        config["fft_size"],
                    )
                    metrics = measure_signal(
                        frequencies,
                        power_db,
                        target_hz,
                        config["bandwidth_hz"],
                        baseband,
                        SAMPLE_RATE,
                    )
                    span_mask = np.abs(frequencies - target_hz) <= config["waterfall_span_hz"] / 2
                    row = power_db[span_mask].astype(np.float32)
                    row_frequencies = frequencies[span_mask].astype(np.float64)
                    clip_path: str | None = None
                    if config["record_audio_on_activity"] and metrics.signal_present:
                        audio = demodulate(
                            baseband,
                            capture.sample_rate_hz,
                            config["mode"],
                            config["bandwidth_hz"],
                        )
                        audio_path = AUDIO_DIR / f"{job.job_id}-{len(job.samples):04d}.wav"
                        write_wav(audio_path, audio)
                        clip_path = str(audio_path.resolve())
                    sample = MonitorSample(
                        captured_at=capture.started_at,
                        elapsed_seconds=time.monotonic() - monotonic_start,
                        audio_clip_path=clip_path,
                        **asdict(metrics),
                    )
                    with self._lock:
                        job.samples.append(sample)
                        job.waterfall_rows.append(row)
                        if job.waterfall_frequencies_hz is None:
                            job.waterfall_frequencies_hz = row_frequencies
                finally:
                    Path(capture.path).unlink(missing_ok=True)
                next_capture_at += config["interval_seconds"]

            with self._lock:
                job.state = "stopped" if job.stop_event.is_set() else "completed"
        except Exception as exc:
            with self._lock:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.completed_at = datetime.now(timezone.utc).isoformat()
            try:
                self._write_results(job)
            finally:
                release_long_job(job.job_id)

    def status(self, job_id: str) -> dict:
        job = self._get(job_id)
        with self._lock:
            capture_count = len(job.samples)
            expected = int(job.config["expected_captures"])
            return {
                "job_id": job.job_id,
                "state": job.state,
                "frequency_hz": job.config["frequency_hz"],
                "capture_count": capture_count,
                "expected_captures": expected,
                "progress_percent": min(100.0, 100 * capture_count / max(1, expected)),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
            }

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        with self._lock:
            requested = job.state not in TERMINAL_STATES
            if requested:
                job.stop_event.set()
        result = self.status(job_id)
        result["stop_requested"] = requested
        return result

    def _events(self, samples: list[MonitorSample], interval_seconds: float) -> list[dict]:
        events: list[dict] = []
        active: list[MonitorSample] = []
        for sample in samples + [None]:  # type: ignore[list-item]
            if sample is not None and sample.signal_present:
                active.append(sample)
                continue
            if active:
                events.append(
                    {
                        "start_elapsed_seconds": active[0].elapsed_seconds,
                        "end_elapsed_seconds": active[-1].elapsed_seconds + interval_seconds,
                        "sample_count": len(active),
                        "peak_snr_db": max(item.estimated_snr_db for item in active),
                        "maximum_confidence": max(item.signal_confidence for item in active),
                        "audio_clip_paths": [
                            item.audio_clip_path for item in active if item.audio_clip_path is not None
                        ],
                    }
                )
                active = []
        return events

    def _result_dict(self, job: MonitorJob) -> dict:
        with self._lock:
            samples = [asdict(item) for item in job.samples]
            sample_objects = list(job.samples)
            state = job.state
            error = job.error
        events = self._events(sample_objects, float(job.config["interval_seconds"]))
        snrs = [item.estimated_snr_db for item in sample_objects]
        return {
            "job_id": job.job_id,
            "state": state,
            "error": error,
            "config": job.config,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "capture_count": len(samples),
            "event_count": len(events),
            "events": events,
            "summary": {
                "maximum_snr_db": max(snrs) if snrs else None,
                "median_snr_db": float(np.median(snrs)) if snrs else None,
                "activity_capture_percent": (
                    100 * sum(item.signal_present for item in sample_objects) / len(sample_objects)
                    if sample_objects else 0.0
                ),
            },
            "samples": samples,
        }

    def _save_plot(self, job: MonitorJob, result: dict) -> Path | None:
        with self._lock:
            rows = list(job.waterfall_rows)
            frequencies = None if job.waterfall_frequencies_hz is None else job.waterfall_frequencies_hz.copy()
            samples = list(job.samples)
        if not rows or frequencies is None or not samples:
            return None
        waterfall = np.vstack(rows)
        times = np.array([item.elapsed_seconds for item in samples])
        vmin, vmax = np.percentile(waterfall, [5, 99.5])
        if vmax <= vmin:
            vmax = vmin + 1

        fig, (waterfall_ax, timeline_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
        extent = [
            frequencies[0] / 1e6,
            frequencies[-1] / 1e6,
            times[-1],
            times[0],
        ]
        image = waterfall_ax.imshow(
            waterfall,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        waterfall_ax.axvline(job.config["frequency_hz"] / 1e6, color="white", linewidth=0.8)
        waterfall_ax.set_title(f"RF Monitor — {job.config['frequency_hz'] / 1e6:g} MHz")
        waterfall_ax.set_xlabel("Frequency (MHz)")
        waterfall_ax.set_ylabel("Elapsed time (s)")
        fig.colorbar(image, ax=waterfall_ax, label="Relative power (dB)")

        snr = np.array([item.estimated_snr_db for item in samples])
        present = np.array([item.signal_present for item in samples])
        timeline_ax.plot(times, snr, marker="o", markersize=3, color="#1677b8", label="Estimated SNR")
        timeline_ax.fill_between(times, 0, snr, where=present, color="#f36d2e", alpha=0.3, label="Activity")
        timeline_ax.axhline(6, color="#555", linestyle="--", linewidth=1, label="Detection threshold")
        timeline_ax.set_xlabel("Elapsed time (s)")
        timeline_ax.set_ylabel("SNR (dB)")
        timeline_ax.grid(alpha=0.25)
        timeline_ax.legend(loc="upper right", ncol=3)

        plot_path = PLOT_DIR / f"{job.job_id}-monitor.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        result["waterfall_plot_path"] = str(plot_path.resolve())
        return plot_path

    def _write_results(self, job: MonitorJob) -> tuple[dict, Path | None]:
        with job.render_lock:
            ensure_data_dirs()
            result = self._result_dict(job)
            plot_path = self._save_plot(job, result)
            result_path = RESULT_DIR / f"{job.job_id}.json"
            result["result_json_path"] = str(result_path.resolve())
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            artifacts = []
            if plot_path is not None:
                artifacts.append(
                    catalog.register_artifact(plot_path, "waterfall_plot", job_id=job.job_id)
                )
            for sample in result["samples"]:
                if sample.get("audio_clip_path"):
                    artifacts.append(
                        catalog.register_artifact(
                            sample["audio_clip_path"], "activity_audio", job_id=job.job_id
                        )
                    )
            artifacts.append(
                catalog.register_artifact(result_path, "result_json", job_id=job.job_id)
            )
            result["artifacts"] = artifacts
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            catalog.register_artifact(result_path, "result_json", job_id=job.job_id)
            catalog.upsert_job(
                job.job_id,
                "monitor",
                result["state"],
                config=job.config,
                summary=result["summary"],
                result_json_path=result_path,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error,
            )
            return result, plot_path

    def results(self, job_id: str) -> tuple[dict, Path | None]:
        job = self._get(job_id)
        return self._write_results(job)

    def _prune_locked(self) -> None:
        completed = [job for job in self._jobs.values() if job.state in TERMINAL_STATES]
        completed.sort(key=lambda item: item.completed_at or item.created_at)
        for job in completed[:-20]:
            self._jobs.pop(job.job_id, None)


monitor_manager = MonitorManager()
