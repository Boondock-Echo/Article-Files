from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .receiver_backend import capture_iq, offset_capture_center, validate_duration, validate_frequency
from .catalog import catalog
from .classification import classify_features, extract_features, feature_dict, save_classification_plot
from .config import PLOT_DIR, RESULT_DIR, SAMPLE_RATE, TUNING_RANGES_HZ, ensure_data_dirs
from .operations import acquire_long_job, release_long_job
from .signal_analysis import downconvert
from .spectrum import (
    DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
    PSD_SCALE,
    averaged_psd_dbfs_per_hz,
    integrate_psd_dbfs,
    iq_level_metrics,
    load_complex_float32,
    valid_passband_mask,
)


TERMINAL_STATES = {"completed", "stopped", "failed"}
MAX_SCAN_STEPS = 500


@dataclass
class ScanSegment:
    center_frequency_hz: int
    started_at: str
    relative_noise_floor_db: float
    relative_peak_db: float
    digital_noise_floor_dbfs_hz: float | None = None
    digital_peak_psd_dbfs_hz: float | None = None
    overload_suspected: bool = False
    clipped_component_fraction: float = 0.0
    max_component_abs: float = 0.0


@dataclass
class ScanJob:
    job_id: str
    config: dict[str, Any]
    centers_hz: list[int]
    state: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    phase: str = "scanning"
    segments: list[ScanSegment] = field(default_factory=list)
    classifications: list[dict[str, Any]] = field(default_factory=list)
    classification_total: int = 0
    frequency_parts: list[np.ndarray] = field(default_factory=list, repr=False)
    power_parts_db: list[np.ndarray] = field(default_factory=list, repr=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    render_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    last_persisted_monotonic: float | None = field(default=None, repr=False)
    last_persisted_progress_state: tuple[Any, ...] | None = field(default=None, repr=False)
    checkpoint_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _same_tuning_range(start_hz: int, stop_hz: int) -> bool:
    return any(low <= start_hz < stop_hz <= high for low, high in TUNING_RANGES_HZ)


def plan_centers(start_hz: int, stop_hz: int, overlap_fraction: float) -> list[int]:
    usable_half_width = (SAMPLE_RATE / 2) * (1 - 0.12)
    step_hz = 2 * usable_half_width * (1 - overlap_fraction)
    first = start_hz + usable_half_width
    last = stop_hz - usable_half_width
    if last <= first:
        return [round((start_hz + stop_hz) / 2)]
    centers = list(np.arange(first, last + step_hz, step_hz))
    centers[-1] = last
    return [round(value) for value in centers]


class ScanManager:
    def __init__(
        self,
        *,
        checkpoint_min_interval_seconds: float = 1.0,
        checkpoint_min_completed_steps: int = 5,
        monotonic: Any = time.monotonic,
    ) -> None:
        if checkpoint_min_interval_seconds < 0:
            raise ValueError("checkpoint_min_interval_seconds must not be negative")
        if checkpoint_min_completed_steps < 1:
            raise ValueError("checkpoint_min_completed_steps must be at least 1")
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.RLock()
        self._checkpoint_min_interval_seconds = float(checkpoint_min_interval_seconds)
        self._checkpoint_min_completed_steps = int(checkpoint_min_completed_steps)
        self._monotonic = monotonic

    def _get(self, job_id: str) -> ScanJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown band scan job_id: {job_id}")
        return job

    def start(
        self,
        *,
        start_frequency_hz: int,
        stop_frequency_hz: int,
        capture_duration_seconds: float,
        overlap_fraction: float,
        fft_size: int,
        threshold_above_noise_db: float,
        minimum_signal_spacing_hz: float,
        attenuation_steps: int,
        max_signals: int,
        classify_top_signals: int = 0,
        classification_duration_seconds: float = 2.0,
        classification_bandwidth_hz: int = 30_000,
        source_preset_id: str | None = None,
        source_schedule_id: str | None = None,
    ) -> dict:
        start_frequency_hz = validate_frequency(start_frequency_hz)
        stop_frequency_hz = validate_frequency(stop_frequency_hz)
        if not _same_tuning_range(start_frequency_hz, stop_frequency_hz):
            raise ValueError("Scan start and stop must be ordered within the same HF+ tuning range")
        if stop_frequency_hz - start_frequency_hz < 10_000:
            raise ValueError("Band scan span must be at least 10000 Hz")
        capture_duration_seconds = validate_duration(capture_duration_seconds)
        overlap_fraction = float(overlap_fraction)
        if not 0.05 <= overlap_fraction <= 0.5:
            raise ValueError("overlap_fraction must be from 0.05 through 0.5")
        if fft_size < 1024 or fft_size > 65_536 or fft_size & (fft_size - 1):
            raise ValueError("fft_size must be a power of two from 1024 through 65536")
        if not 3 <= threshold_above_noise_db <= 60:
            raise ValueError("threshold_above_noise_db must be from 3 through 60")
        minimum_signal_spacing_hz = float(minimum_signal_spacing_hz)
        if not 100 <= minimum_signal_spacing_hz <= 100_000:
            raise ValueError("minimum_signal_spacing_hz must be from 100 through 100000")
        attenuation_steps = int(attenuation_steps)
        if not 0 <= attenuation_steps <= 8:
            raise ValueError("attenuation_steps must be from 0 through 8")
        if not 1 <= max_signals <= 500:
            raise ValueError("max_signals must be from 1 through 500")
        classify_top_signals = int(classify_top_signals)
        if not 0 <= classify_top_signals <= 20:
            raise ValueError("classify_top_signals must be from 0 through 20")
        classification_duration_seconds = float(classification_duration_seconds)
        classification_bandwidth_hz = int(classification_bandwidth_hz)
        if classify_top_signals:
            classification_duration_seconds = validate_duration(classification_duration_seconds)
            if not 2_000 <= classification_bandwidth_hz <= 50_000:
                raise ValueError("classification_bandwidth_hz must be from 2000 through 50000")

        centers = plan_centers(start_frequency_hz, stop_frequency_hz, overlap_fraction)
        if len(centers) > MAX_SCAN_STEPS:
            raise ValueError(f"Scan requires {len(centers)} retunes; maximum is {MAX_SCAN_STEPS}")
        prefix = "survey" if classify_top_signals else "scan"
        job_id = f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        config = {
            "start_frequency_hz": start_frequency_hz,
            "stop_frequency_hz": stop_frequency_hz,
            "capture_duration_seconds": capture_duration_seconds,
            "overlap_fraction": overlap_fraction,
            "fft_size": fft_size,
            "threshold_above_noise_db": float(threshold_above_noise_db),
            "minimum_signal_spacing_hz": minimum_signal_spacing_hz,
            "attenuation_steps": attenuation_steps,
            "attenuation_db": attenuation_steps * 6,
            "max_signals": int(max_signals),
            "classify_top_signals": classify_top_signals,
            "classification_duration_seconds": classification_duration_seconds,
            "classification_bandwidth_hz": classification_bandwidth_hz,
            "planned_steps": len(centers),
        }
        if source_preset_id:
            config["source_preset_id"] = source_preset_id
        if source_schedule_id:
            config["source_schedule_id"] = source_schedule_id
        with self._lock:
            job = ScanJob(job_id=job_id, config=config, centers_hz=centers)
            acquire_long_job(job_id)
            try:
                self._jobs[job_id] = job
                catalog.upsert_job(
                    job_id,
                    self._job_type(job),
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

    @staticmethod
    def _job_type(job: ScanJob) -> str:
        return "band_survey" if job.config.get("classify_top_signals", 0) else "band_scan"

    def _checkpoint(
        self,
        job: ScanJob,
        *,
        current_frequency_hz: int | None = None,
        force: bool = False,
    ) -> bool:
        """Persist lightweight progress so remote dashboards survive refreshes/restarts."""
        with job.checkpoint_lock:
            now = self._monotonic()
            with self._lock:
                completed = len(job.segments)
                planned = len(job.centers_hz)
                classification_completed = len(job.classifications)
                classification_planned = job.classification_total
                overall_planned = planned + max(
                    classification_planned,
                    int(job.config.get("classify_top_signals", 0)),
                )
                overall_completed = completed + classification_completed
                stop_requested = job.stop_event.is_set()
                progress_state = (
                    job.state,
                    job.phase,
                    job.error,
                    job.completed_at,
                    stop_requested,
                    completed,
                    classification_completed,
                    classification_planned,
                )
                previous = job.last_persisted_progress_state
                transition = previous is None or progress_state[:5] != previous[:5]
                elapsed = (
                    float("inf")
                    if job.last_persisted_monotonic is None
                    else now - job.last_persisted_monotonic
                )
                completed_delta = (
                    overall_completed
                    if previous is None
                    else overall_completed - previous[5] - previous[6]
                )
                should_persist = (
                    force
                    or transition
                    or elapsed >= self._checkpoint_min_interval_seconds
                    or completed_delta >= self._checkpoint_min_completed_steps
                )
                if not should_persist:
                    return False
                summary = {
                    "phase": job.phase,
                    "completed_steps": completed,
                    "planned_steps": planned,
                    "classification_completed": classification_completed,
                    "classification_planned": classification_planned,
                    "stop_requested": stop_requested,
                    "progress_percent": (
                        100.0
                        if job.state in TERMINAL_STATES
                        else min(99.9, 100 * overall_completed / max(1, overall_planned))
                    ),
                }
                if current_frequency_hz is not None:
                    summary["current_frequency_hz"] = int(current_frequency_hz)
                state = job.state
                created_at = job.created_at
                started_at = job.started_at
                completed_at = job.completed_at
                error = job.error
            catalog.upsert_job(
                job.job_id,
                self._job_type(job),
                state,
                config=job.config,
                summary=summary,
                created_at=created_at,
                started_at=started_at,
                completed_at=completed_at,
                error=error,
            )
            with self._lock:
                job.last_persisted_monotonic = now
                job.last_persisted_progress_state = progress_state
            return True

    def _run(self, job: ScanJob) -> None:
        with self._lock:
            job.state = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
        try:
            ensure_data_dirs()
            self._checkpoint(job, force=True)
            for center_hz in job.centers_hz:
                if job.stop_event.is_set():
                    break
                capture = capture_iq(
                    center_hz,
                    job.config["capture_duration_seconds"],
                    agc=False,
                    attenuation_steps=job.config["attenuation_steps"],
                    lna=False,
                )
                try:
                    iq = load_complex_float32(capture.path)
                    levels = iq_level_metrics(iq)
                    frequencies, psd_dbfs_hz = averaged_psd_dbfs_per_hz(
                        iq,
                        capture.center_frequency_hz,
                        capture.sample_rate_hz,
                        job.config["fft_size"],
                    )
                    mask = valid_passband_mask(
                        frequencies,
                        capture.center_frequency_hz,
                        capture.sample_rate_hz,
                    )
                    frequencies = frequencies[mask]
                    psd_dbfs_hz = psd_dbfs_hz[mask]
                    band_mask = (
                        (frequencies >= job.config["start_frequency_hz"])
                        & (frequencies <= job.config["stop_frequency_hz"])
                    )
                    frequencies = frequencies[band_mask]
                    psd_dbfs_hz = psd_dbfs_hz[band_mask]
                    if frequencies.size:
                        relative_power_db = psd_dbfs_hz - np.max(psd_dbfs_hz)
                        segment = ScanSegment(
                            center_frequency_hz=center_hz,
                            started_at=capture.started_at,
                            relative_noise_floor_db=float(np.median(relative_power_db)),
                            relative_peak_db=float(np.max(relative_power_db)),
                            digital_noise_floor_dbfs_hz=float(np.median(psd_dbfs_hz)),
                            digital_peak_psd_dbfs_hz=float(np.max(psd_dbfs_hz)),
                            overload_suspected=levels["overload_suspected"],
                            clipped_component_fraction=levels["clipped_component_fraction"],
                            max_component_abs=levels["max_component_abs"],
                        )
                        with self._lock:
                            job.segments.append(segment)
                            job.frequency_parts.append(frequencies.astype(np.float64))
                            job.power_parts_db.append(psd_dbfs_hz.astype(np.float32))
                        self._checkpoint(job, current_frequency_hz=center_hz)
                finally:
                    Path(capture.path).unlink(missing_ok=True)
            if not job.stop_event.is_set() and job.config["classify_top_signals"]:
                self._classify_candidates(job)
            with self._lock:
                job.state = "stopped" if job.stop_event.is_set() else "completed"
                job.phase = "finished"
        except Exception as exc:
            with self._lock:
                job.state = "failed"
                job.phase = "finished"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.completed_at = datetime.now(timezone.utc).isoformat()
            try:
                self._checkpoint(job, force=True)
                self._write_results(job)
            finally:
                release_long_job(job.job_id)

    def status(self, job_id: str) -> dict:
        job = self._get(job_id)
        with self._lock:
            completed = len(job.segments)
            planned = len(job.centers_hz)
            classification_completed = len(job.classifications)
            classification_planned = job.classification_total
            overall_planned = planned + max(
                classification_planned,
                job.config.get("classify_top_signals", 0),
            )
            overall_completed = completed + classification_completed
            return {
                "job_id": job.job_id,
                "state": job.state,
                "phase": job.phase,
                "completed_steps": completed,
                "planned_steps": planned,
                "classification_completed": classification_completed,
                "classification_planned": classification_planned,
                "progress_percent": (
                    100.0
                    if job.state in TERMINAL_STATES
                    else min(99.9, 100 * overall_completed / max(1, overall_planned))
                ),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
            }

    def _classify_candidates(self, job: ScanJob) -> None:
        stitched = self._stitched(job)
        if stitched is None:
            return
        frequencies, power_db, psd_dbfs_hz = stitched
        _, signals = self._detect_signals(frequencies, power_db, job, psd_dbfs_hz)
        candidates = signals[: job.config["classify_top_signals"]]
        with self._lock:
            job.phase = "classifying"
            job.classification_total = len(candidates)
        self._checkpoint(job, force=True)

        for index, signal in enumerate(candidates, start=1):
            if job.stop_event.is_set():
                break
            frequency_hz = int(round(signal["frequency_hz"]))
            capture = None
            try:
                receiver_center_hz = offset_capture_center(frequency_hz)
                capture = capture_iq(
                    receiver_center_hz,
                    job.config["classification_duration_seconds"],
                    agc=False,
                    attenuation_steps=job.config["attenuation_steps"],
                    lna=False,
                )
                iq = load_complex_float32(capture.path)
                offset_hz = frequency_hz - capture.center_frequency_hz
                baseband = downconvert(iq, capture.sample_rate_hz, offset_hz)
                features, spectrum_hz, spectrum_db, time_axis, instant_frequency = extract_features(
                    baseband,
                    capture.sample_rate_hz,
                    job.config["classification_bandwidth_hz"],
                    16_384,
                )
                ranking = classify_features(features)
                best = ranking[0]
                margin = best["confidence"] - ranking[1]["confidence"]
                ambiguous = best["confidence"] < 0.35 or margin < 0.08
                plot_path = PLOT_DIR / f"{job.job_id}-signal-{index:02d}-classification.png"
                save_classification_plot(
                    plot_path,
                    frequency_hz,
                    spectrum_hz,
                    spectrum_db,
                    time_axis,
                    instant_frequency,
                    ranking,
                )
                classification = {
                    "rank": index,
                    "frequency_hz": frequency_hz,
                    "scan_relative_power_db": signal["relative_power_db"],
                    "scan_above_noise_db": signal["above_noise_db"],
                    "scan_digital_power_dbfs_10khz": signal.get(
                        "digital_power_dbfs_10khz"
                    ),
                    "status": "completed",
                    "best_label": best["label"],
                    "best_confidence": best["confidence"],
                    "confidence_margin": margin,
                    "ambiguous": ambiguous,
                    "ranking": ranking,
                    "features": feature_dict(features),
                    "classification_plot_path": str(plot_path.resolve()),
                }
            except Exception as exc:
                classification = {
                    "rank": index,
                    "frequency_hz": frequency_hz,
                    "scan_relative_power_db": signal["relative_power_db"],
                    "scan_above_noise_db": signal["above_noise_db"],
                    "scan_digital_power_dbfs_10khz": signal.get(
                        "digital_power_dbfs_10khz"
                    ),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if capture is not None:
                    Path(capture.path).unlink(missing_ok=True)
            with self._lock:
                job.classifications.append(classification)
            self._checkpoint(
                job,
                current_frequency_hz=frequency_hz,
                force=classification["status"] == "failed",
            )

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        with self._lock:
            requested = job.state not in TERMINAL_STATES
            if requested:
                job.stop_event.set()
        if requested:
            self._checkpoint(job, force=True)
        result = self.status(job_id)
        result["stop_requested"] = requested
        return result

    def _stitched(self, job: ScanJob) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        with self._lock:
            frequency_parts = [item.copy() for item in job.frequency_parts]
            power_parts = [item.copy() for item in job.power_parts_db]
        if not frequency_parts:
            return None
        bin_width_hz = SAMPLE_RATE / job.config["fft_size"]
        grid = np.arange(
            job.config["start_frequency_hz"],
            job.config["stop_frequency_hz"] + bin_width_hz,
            bin_width_hz,
        )
        summed = np.zeros(grid.size, dtype=np.float64)
        counts = np.zeros(grid.size, dtype=np.uint16)
        for frequencies, power_db in zip(frequency_parts, power_parts, strict=True):
            first = int(np.searchsorted(grid, frequencies[0], side="left"))
            last = int(np.searchsorted(grid, frequencies[-1], side="right"))
            if last <= first:
                continue
            values_linear = np.interp(
                grid[first:last], frequencies, 10 ** (power_db / 10)
            )
            summed[first:last] += values_linear
            counts[first:last] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        average = np.full(grid.size, np.nan)
        average[valid] = summed[valid] / counts[valid]
        psd_dbfs_hz = 10 * np.log10(np.maximum(average[valid], 1e-30))
        relative_power_db = psd_dbfs_hz - np.max(psd_dbfs_hz)
        return grid[valid], relative_power_db, psd_dbfs_hz

    def _detect_signals(
        self,
        frequencies: np.ndarray,
        power_db: np.ndarray,
        job: ScanJob,
        psd_dbfs_hz: np.ndarray | None = None,
    ) -> tuple[float, list[dict]]:
        noise_floor_db = float(np.median(power_db))
        bin_width_hz = float(np.median(np.diff(frequencies)))
        distance = max(1, round(job.config["minimum_signal_spacing_hz"] / bin_width_hz))
        from .lazy_imports import find_peaks

        indices, properties = find_peaks(
            power_db,
            height=noise_floor_db + job.config["threshold_above_noise_db"],
            prominence=max(3.0, job.config["threshold_above_noise_db"] / 2),
            distance=distance,
        )
        prominences = properties.get("prominences", np.zeros(len(indices)))
        signals = []
        for index, prominence in zip(indices, prominences, strict=True):
            signal = {
                "frequency_hz": float(frequencies[index]),
                "relative_power_db": float(power_db[index]),
                "above_noise_db": float(power_db[index] - noise_floor_db),
                "prominence_db": float(prominence),
            }
            if psd_dbfs_hz is not None:
                integration_mask = (
                    np.abs(frequencies - frequencies[index])
                    <= DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ / 2
                )
                actual_bandwidth_hz = float(
                    np.count_nonzero(integration_mask) * abs(bin_width_hz)
                )
                signal.update(
                    {
                        "digital_peak_psd_dbfs_hz": float(psd_dbfs_hz[index]),
                        "digital_power_dbfs_10khz": integrate_psd_dbfs(
                            frequencies,
                            psd_dbfs_hz,
                            frequencies[index],
                            DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
                        ),
                        "digital_power_integration_bandwidth_hz": actual_bandwidth_hz,
                        "digital_power_integration_truncated": bool(
                            actual_bandwidth_hz
                            < DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ * 0.9
                        ),
                    }
                )
            signals.append(signal)
        signals.sort(key=lambda item: item["relative_power_db"], reverse=True)
        return noise_floor_db, signals[: job.config["max_signals"]]

    def _write_results(self, job: ScanJob) -> tuple[dict, Path | None]:
        with job.render_lock:
            stitched = self._stitched(job)
            with self._lock:
                segments = [segment.__dict__.copy() for segment in job.segments]
                classifications = [item.copy() for item in job.classifications]
                state = job.state
                error = job.error
                phase = job.phase
            result: dict[str, Any] = {
                "job_id": job.job_id,
                "state": state,
                "phase": phase,
                "error": error,
                "config": job.config,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "completed_steps": len(segments),
                "segments": segments,
                "classification_count": len(classifications),
                "classifications": classifications,
                "measurement_scale": "relative_db_fixed_receiver_gain",
                "digital_power_scale": {
                    "scale": PSD_SCALE,
                    "calibrated_rf_input_power": False,
                    "psd_units": "dBFS/Hz",
                    "integrated_power_units": "dBFS",
                    "integration_bandwidth_hz": DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ,
                    "reference": "complex_float_full_scale_component_1.0",
                },
                "receiver_profile": {
                    "sample_rate_hz": SAMPLE_RATE,
                    "agc": False,
                    "attenuation_steps": job.config["attenuation_steps"],
                    "attenuation_db": job.config["attenuation_db"],
                    "lna": False,
                    "fft_size": job.config["fft_size"],
                    "window": "blackman",
                },
                "overload": {
                    "suspected": any(item["overload_suspected"] for item in segments),
                    "affected_segment_count": sum(
                        item["overload_suspected"] for item in segments
                    ),
                    "max_clipped_component_fraction": max(
                        (item["clipped_component_fraction"] for item in segments),
                        default=0.0,
                    ),
                    "max_component_abs": max(
                        (item["max_component_abs"] for item in segments),
                        default=0.0,
                    ),
                },
            }
            plot_path: Path | None = None
            if stitched is not None:
                frequencies, power_db, psd_dbfs_hz = stitched
                noise_floor_db, signals = self._detect_signals(
                    frequencies, power_db, job, psd_dbfs_hz
                )
                result.update(
                    {
                        "scanned_range_hz": [float(frequencies[0]), float(frequencies[-1])],
                        "bin_width_hz": float(np.median(np.diff(frequencies))),
                        "relative_noise_floor_db": noise_floor_db,
                        "digital_noise_floor_dbfs_hz": float(np.median(psd_dbfs_hz)),
                        "digital_peak_psd_dbfs_hz": float(np.max(psd_dbfs_hz)),
                        "signal_count": len(signals),
                        "signals": signals,
                        "occupied_bin_fraction": float(np.mean(
                            power_db >= noise_floor_db + job.config["threshold_above_noise_db"]
                        )),
                    }
                )
                summary_count = min(1200, frequencies.size)
                summary_indices = np.linspace(
                    0, frequencies.size - 1, summary_count, dtype=int
                )
                result["spectrum_summary"] = {
                    "frequency_hz": frequencies[summary_indices].astype(float).tolist(),
                    "relative_power_db": power_db[summary_indices].astype(float).tolist(),
                    "digital_psd_dbfs_hz": psd_dbfs_hz[summary_indices].astype(float).tolist(),
                    "point_count": int(summary_count),
                    "downsampled": bool(summary_count < frequencies.size),
                }
                from .plotting import pyplot

                plt = pyplot()
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(frequencies / 1e6, power_db, color="#1677b8", linewidth=0.7)
                ax.axhline(noise_floor_db, color="#555", linestyle="--", linewidth=1, label="Median noise")
                if signals:
                    ax.scatter(
                        [item["frequency_hz"] / 1e6 for item in signals],
                        [item["relative_power_db"] for item in signals],
                        color="#f36d2e",
                        s=15,
                        label="Detected signals",
                        zorder=3,
                    )
                completed_classifications = [
                    item for item in classifications if item.get("status") == "completed"
                ]
                for item in completed_classifications:
                    ax.annotate(
                        item["best_label"].upper(),
                        (item["frequency_hz"] / 1e6, item["scan_relative_power_db"]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color="#8a3f18",
                    )
                report_kind = "Band Survey" if job.config.get("classify_top_signals") else "Band Scan"
                ax.set_title(
                    f"Airspy HF+ {report_kind} — {job.config['start_frequency_hz'] / 1e6:g} to "
                    f"{job.config['stop_frequency_hz'] / 1e6:g} MHz"
                )
                ax.set_xlabel("Frequency (MHz)")
                ax.set_ylabel("Relative power (dB)")
                ax.grid(alpha=0.25)
                ax.legend(loc="lower center", ncol=2)
                fig.tight_layout()
                plot_path = PLOT_DIR / f"{job.job_id}-band-scan.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                result["spectrum_plot_path"] = str(plot_path.resolve())
            else:
                result.update({"signal_count": 0, "signals": []})

            result_path = RESULT_DIR / f"{job.job_id}.json"
            result["result_json_path"] = str(result_path.resolve())
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            artifacts = []
            if plot_path is not None:
                artifacts.append(
                    catalog.register_artifact(plot_path, "band_scan_plot", job_id=job.job_id)
                )
            for item in classifications:
                classification_plot = item.get("classification_plot_path")
                if classification_plot and Path(classification_plot).exists():
                    artifacts.append(
                        catalog.register_artifact(
                            classification_plot,
                            "survey_classification_plot",
                            job_id=job.job_id,
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
                self._job_type(job),
                result["state"],
                config=job.config,
                summary={
                    "signal_count": result.get("signal_count", 0),
                    "phase": result["phase"],
                    "completed_steps": result["completed_steps"],
                    "planned_steps": len(job.centers_hz),
                    "classification_count": result["classification_count"],
                    "classification_completed": result["classification_count"],
                    "classification_planned": job.classification_total,
                    "progress_percent": 100.0,
                },
                result_json_path=result_path,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error,
            )
            return result, plot_path

    def results(self, job_id: str) -> tuple[dict, Path | None]:
        return self._write_results(self._get(job_id))

    def _prune_locked(self) -> None:
        completed = [job for job in self._jobs.values() if job.state in TERMINAL_STATES]
        completed.sort(key=lambda item: item.completed_at or item.created_at)
        for job in completed[:-20]:
            self._jobs.pop(job.job_id, None)


scan_manager = ScanManager()
