from __future__ import annotations

import json
import csv
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.signal import resample_poly

from .receiver_backend import offset_capture_center, stream_iq_chunks, validate_frequency
from .catalog import catalog
from .config import SATELLITE_DIR, ensure_data_dirs
from .digital_decode import decode_ax25_afsk1200, decode_ax25_g3ruh9600, save_decode_plot
from .operations import acquire_long_job, release_long_job
from .sstv_watcher import StreamingDemodulator, doppler_frequency_at
from .weak_signal import _write_decoder_wav


TERMINAL_STATES = {"completed", "stopped", "failed", "interrupted"}


@dataclass
class SatelliteReceiveJob:
    job_id: str
    config: dict
    state: str = "queued"
    phase: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class SatelliteReceiverManager:
    def __init__(self) -> None:
        self._jobs: dict[str, SatelliteReceiveJob] = {}
        self._lock = threading.RLock()

    def start(self, *, watch: dict, pass_record: dict, downlink: dict,
              duration_seconds: float) -> dict:
        mode = str(downlink["mode"])
        if mode not in {"nfm_audio", "ax25_afsk1200", "ax25_g3ruh9600", "capture_only"}:
            raise ValueError(
                "general receiver mode must be nfm_audio, ax25_afsk1200, "
                "ax25_g3ruh9600, or capture_only"
            )
        duration_seconds = float(duration_seconds)
        if not 30 <= duration_seconds <= 86_400:
            raise ValueError("duration_seconds must be from 30 through 86400")
        frequency_hz = validate_frequency(int(downlink["frequency_hz"]))
        job_id = f"sat-rx-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        config = {
            "watch_id": watch["watch_id"], "pass_id": pass_record["pass_id"],
            "satellite_name": watch["satellite_name"], "downlink": downlink,
            "frequency_hz": frequency_hz, "duration_seconds": duration_seconds,
            "doppler_correction_mode": watch.get("doppler_correction_mode", "off"),
            "doppler_plan": pass_record.get("doppler_plan", []),
        }
        job = SatelliteReceiveJob(job_id, config)
        with self._lock:
            acquire_long_job(job_id)
            try:
                self._jobs[job_id] = job
                self._persist(job)
                job.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
                job.thread.start()
            except Exception:
                self._jobs.pop(job_id, None)
                release_long_job(job_id)
                raise
        return self.status(job_id)

    def _run(self, job: SatelliteReceiveJob) -> None:
        ensure_data_dirs()
        work_dir = SATELLITE_DIR / job.job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        job.state, job.phase = "running", "receiving"
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._persist(job)
        chunks, total_samples, sum_power, peak = [], 0, 0.0, 0.0
        try:
            config, downlink = job.config, job.config["downlink"]
            center = offset_capture_center(config["frequency_hz"], offset_hz=10_000)
            output_rate = 48_000 if downlink["mode"] == "ax25_g3ruh9600" else 12_000
            demodulator = StreamingDemodulator(
                mode=downlink.get("receiver_mode", "nfm"),
                offset_hz=config["frequency_hz"] - center,
                output_rate=output_rate,
            )
            for interleaved in stream_iq_chunks(
                center, duration_seconds=config["duration_seconds"],
                stop_event=job.stop_event, chunk_seconds=0.5,
            ):
                receive_frequency = None
                if config["doppler_correction_mode"] == "digital":
                    receive_frequency = doppler_frequency_at(
                        config["doppler_plan"], datetime.now(timezone.utc)
                    )
                offset = ((receive_frequency or config["frequency_hz"]) - center)
                baseband = demodulator.downconvert(interleaved, offset_hz=offset)
                total_samples += len(baseband)
                if len(baseband):
                    power = np.abs(baseband) ** 2
                    sum_power += float(np.sum(power))
                    peak = max(peak, float(np.max(np.abs(baseband))))
                if downlink["mode"] != "capture_only":
                    chunks.append(np.asarray(baseband, dtype=np.complex64))
                if job.stop_event.is_set():
                    break
            baseband = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.complex64)
            job.phase = "processing"
            captured_at = job.started_at
            duration = total_samples / output_rate
            details = {
                "complex_sample_count": total_samples,
                "rms": round(float(np.sqrt(sum_power / max(total_samples, 1))), 7),
                "peak": round(peak, 7),
                "doppler_correction_mode": config["doppler_correction_mode"],
                "doppler_plan_point_count": len(config["doppler_plan"]),
            }
            audio_path, packet_count, valid_count = None, 0, 0
            if downlink["mode"] in {"nfm_audio", "ax25_afsk1200", "ax25_g3ruh9600"}:
                audio = self._fm_audio(baseband)
                if downlink.get("retain_audio", True):
                    path = work_dir / "audio.wav"
                    wav_audio = (resample_poly(audio, 1, output_rate // 12_000)
                                 if output_rate != 12_000 else audio)
                    _write_decoder_wav(path, wav_audio)
                    audio_path = str(path.resolve())
                    catalog.register_artifact(path, "satellite_audio", job_id=job.job_id)
            if downlink["mode"] in {"ax25_afsk1200", "ax25_g3ruh9600"}:
                decoder = (decode_ax25_afsk1200 if downlink["mode"] == "ax25_afsk1200"
                           else decode_ax25_g3ruh9600)
                decoded = decoder(baseband, output_rate)
                plot_path = work_dir / "ax25-diagnostic.png"
                save_decode_plot(plot_path, downlink["mode"], decoded)
                catalog.register_artifact(plot_path, "satellite_ax25_plot", job_id=job.job_id)
                diagnostic = decoded.pop("diagnostic", None)
                packet_count = int(decoded["frame_count"])
                valid_count = int(decoded["valid_fcs_count"])
                details["ax25"] = decoded
                details["diagnostic_plot_path"] = str(plot_path.resolve())
                del diagnostic
            outcome = "stopped" if job.stop_event.is_set() else "completed"
            result_path = work_dir / "result.json"
            observation = catalog.add_satellite_observation({
                "job_id": job.job_id, "pass_id": config["pass_id"],
                "watch_id": config["watch_id"], "satellite_name": config["satellite_name"],
                "downlink_id": downlink["downlink_id"], "downlink_label": downlink["label"],
                "mode": downlink["mode"], "nominal_frequency_hz": config["frequency_hz"],
                "outcome": outcome, "packet_count": packet_count,
                "valid_packet_count": valid_count, "captured_at": captured_at,
                "duration_seconds": duration, "result_json_path": result_path,
                "audio_path": audio_path, "details": details,
            })
            from .satellite_telemetry import decode_observation_telemetry
            telemetry = decode_observation_telemetry(catalog, observation)
            observation["telemetry_value_count"] = telemetry["value_count"]
            observation["telemetry_decode_failures"] = telemetry["failures"]
            job.result = {**observation, "schema": "rf-mcp.satellite-observation.v1"}
            result_path.write_text(json.dumps(job.result, indent=2) + "\n", encoding="utf-8")
            catalog.register_artifact(result_path, "satellite_result_json", job_id=job.job_id)
            job.state = outcome
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.phase = "finished"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._persist(job)
            release_long_job(job.job_id)

    @staticmethod
    def _fm_audio(baseband: np.ndarray) -> np.ndarray:
        if len(baseband) < 2:
            return np.zeros(max(1, len(baseband)), dtype=np.float64)
        audio = np.concatenate(([0.0], np.angle(baseband[1:] * np.conj(baseband[:-1]))))
        audio -= np.mean(audio)
        peak = float(np.max(np.abs(audio)))
        return audio / peak * 0.85 if peak else audio

    def _persist(self, job: SatelliteReceiveJob) -> None:
        catalog.upsert_job(
            job.job_id, "satellite_receive", job.state, config=job.config,
            summary={"phase": job.phase,
                     "observation_id": (job.result or {}).get("observation_id")},
            result_json_path=(job.result or {}).get("result_json_path"),
            created_at=job.created_at, started_at=job.started_at,
            completed_at=job.completed_at, error=job.error,
        )

    def _get(self, job_id: str) -> SatelliteReceiveJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job:
            return job
        persisted = catalog.get_job(job_id)
        if persisted["job_type"] != "satellite_receive":
            raise ValueError(f"Unknown satellite receiver job_id: {job_id}")
        return SatelliteReceiveJob(
            job_id, persisted["config"], state=persisted["state"],
            phase=(persisted.get("summary") or {}).get("phase", "finished"),
            created_at=persisted["created_at"], started_at=persisted.get("started_at"),
            completed_at=persisted.get("completed_at"), error=persisted.get("error"),
            result=persisted.get("result"),
        )

    def status(self, job_id: str) -> dict:
        job = self._get(job_id)
        return {"job_id": job.job_id, "state": job.state, "phase": job.phase,
                "created_at": job.created_at, "started_at": job.started_at,
                "completed_at": job.completed_at, "error": job.error,
                "observation_id": (job.result or {}).get("observation_id")}

    def results(self, job_id: str) -> dict:
        job = self._get(job_id)
        return {**self.status(job_id), "config": job.config, "result": job.result}

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.state not in TERMINAL_STATES:
            job.stop_event.set()
        return self.status(job_id)


satellite_receiver_manager = SatelliteReceiverManager()


def export_satellite_telemetry(observations: list[dict], *, output_format: str = "jsonl") -> str:
    output_format = str(output_format).strip().lower()
    if output_format not in {"jsonl", "csv"}:
        raise ValueError("output_format must be jsonl or csv")
    ensure_data_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SATELLITE_DIR / f"telemetry-{stamp}-{uuid4().hex[:8]}.{output_format}"
    rows = []
    for observation in observations:
        frames = observation.get("details", {}).get("ax25", {}).get("frames", [])
        for index, frame in enumerate(frames, 1):
            rows.append({
                "observation_id": observation["observation_id"],
                "job_id": observation["job_id"], "pass_id": observation.get("pass_id"),
                "watch_id": observation.get("watch_id"),
                "satellite_name": observation["satellite_name"],
                "downlink_id": observation["downlink_id"], "mode": observation["mode"],
                "nominal_frequency_hz": observation["nominal_frequency_hz"],
                "captured_at": observation["captured_at"], "frame_index": index,
                "source": frame.get("source"), "destination": frame.get("destination"),
                "digipeaters": frame.get("digipeaters", []),
                "control": frame.get("control"), "pid": frame.get("pid"),
                "fcs_valid": bool(frame.get("fcs_valid")),
                "information_text": frame.get("information_text", ""),
                "information_hex": frame.get("information_hex", ""),
                "frame_hex": frame.get("frame_hex", ""),
            })
    if output_format == "jsonl":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        fields = list(rows[0]) if rows else [
            "observation_id", "job_id", "pass_id", "watch_id", "satellite_name",
            "downlink_id", "mode", "nominal_frequency_hz", "captured_at", "frame_index",
            "source", "destination", "digipeaters", "control", "pid", "fcs_valid",
            "information_text", "information_hex", "frame_hex",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "digipeaters": ",".join(row["digipeaters"])})
    return str(path.resolve())
