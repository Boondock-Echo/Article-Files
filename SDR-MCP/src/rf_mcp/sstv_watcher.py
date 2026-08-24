from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from .lazy_imports import resample_poly

from .receiver_backend import SAMPLE_RATE, offset_capture_center, stream_iq_chunks, validate_frequency
from .catalog import catalog
from .config import SSTV_DIR, ensure_data_dirs
from .operations import acquire_long_job, release_long_job
from .sstv import (
    SSTV_MODES,
    detect_vis,
    image_fingerprint,
    run_sstv_decoder,
    sstv_decoder_path,
)
from .weak_signal import _write_decoder_wav


AUDIO_RATE = 12_000
MODE_CAPTURE_SECONDS = {
    44: 120,  # Martin M1
    40: 65,   # Martin M2
    60: 120,  # Scottie S1
    56: 80,   # Scottie S2
    76: 280,  # Scottie DX
    8: 45,    # Robot 36
    12: 80,   # Robot 72
}
TERMINAL_STATES = {"completed", "stopped", "failed", "interrupted"}


class StreamingDemodulator:
    def __init__(self, *, mode: str, offset_hz: float, output_rate: int = AUDIO_RATE) -> None:
        self.mode = mode
        self.offset_hz = float(offset_hz)
        self.sample_index = 0
        self.mixer_phase = 0.0
        self.previous_baseband: complex | None = None
        self.output_rate = int(output_rate)
        if not 8_000 <= self.output_rate <= 192_000:
            raise ValueError("output_rate must be from 8000 through 192000")

    def downconvert(self, interleaved: np.ndarray, *, offset_hz: float | None = None) -> np.ndarray:
        values = np.asarray(interleaved, dtype="<f4")
        usable = len(values) - len(values) % 2
        iq = values[:usable:2].astype(np.complex64)
        iq.imag = values[1:usable:2]
        if offset_hz is not None:
            self.offset_hz = float(offset_hz)
        indices = np.arange(len(iq), dtype=np.float64)
        self.sample_index += len(iq)
        phases = self.mixer_phase + 2 * np.pi * self.offset_hz * indices / SAMPLE_RATE
        mixed = iq * np.exp(-1j * phases)
        self.mixer_phase = float(
            (self.mixer_phase + 2 * np.pi * self.offset_hz * len(iq) / SAMPLE_RATE)
            % (2 * np.pi)
        )
        divisor = math.gcd(SAMPLE_RATE, self.output_rate)
        return resample_poly(mixed, self.output_rate // divisor, SAMPLE_RATE // divisor)

    def process(self, interleaved: np.ndarray, *, offset_hz: float | None = None) -> np.ndarray:
        baseband = self.downconvert(interleaved, offset_hz=offset_hz)
        if not len(baseband):
            return np.empty(0, dtype=np.float64)
        if self.mode == "usb":
            spectrum = np.fft.fft(baseband)
            frequencies = np.fft.fftfreq(len(baseband), d=1 / self.output_rate)
            spectrum[(frequencies < 50) | (frequencies > 5_000)] = 0
            return 2 * np.real(np.fft.ifft(spectrum))
        if self.previous_baseband is None:
            previous = np.concatenate(([baseband[0]], baseband[:-1]))
        else:
            previous = np.concatenate(([self.previous_baseband], baseband[:-1]))
        self.previous_baseband = complex(baseband[-1])
        audio = np.angle(baseband * np.conj(previous))
        audio -= np.mean(audio) if audio.size else 0
        return audio


class SSTVStreamDetector:
    def __init__(
        self, *, pre_trigger_seconds: float = 3.0,
        mode_capture_seconds: dict[int, float] | None = None,
        rearm_seconds: float = 5.0,
    ) -> None:
        self.pre_trigger_samples = round(float(pre_trigger_seconds) * AUDIO_RATE)
        self.mode_capture_seconds = dict(mode_capture_seconds or MODE_CAPTURE_SECONDS)
        self.rearm_samples = round(float(rearm_seconds) * AUDIO_RATE)
        self.search_audio = np.empty(0, dtype=np.float64)
        self.active_audio: list[np.ndarray] | None = None
        self.active_samples = 0
        self.target_samples = 0
        self.trigger: dict | None = None
        self.cooldown_remaining = 0
        self.false_triggers = 0

    @property
    def receiving(self) -> bool:
        return self.active_audio is not None

    def feed(self, audio: np.ndarray) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float64)
        events: list[dict] = []
        if self.cooldown_remaining:
            consumed = min(self.cooldown_remaining, len(audio))
            self.cooldown_remaining -= consumed
            audio = audio[consumed:]
            if not len(audio):
                return events
        if self.active_audio is not None:
            self.active_audio.append(audio.copy())
            self.active_samples += len(audio)
            if self.active_samples >= self.target_samples:
                clip = np.concatenate(self.active_audio)[:self.target_samples]
                events.append({"event": "complete", "vis": self.trigger, "audio": clip})
                self.active_audio = None
                self.active_samples = self.target_samples = 0
                self.trigger = None
                self.cooldown_remaining = self.rearm_samples
            return events
        self.search_audio = np.concatenate((self.search_audio, audio))
        if len(self.search_audio) > self.pre_trigger_samples:
            self.search_audio = self.search_audio[-self.pre_trigger_samples:]
        if len(self.search_audio) < AUDIO_RATE:
            return events
        vis = detect_vis(self.search_audio, AUDIO_RATE)
        if not vis.get("detected"):
            return events
        if not vis.get("parity_valid") or vis.get("vis_code") not in self.mode_capture_seconds:
            self.false_triggers += 1
            self.search_audio = np.empty(0, dtype=np.float64)
            self.cooldown_remaining = round(0.5 * AUDIO_RATE)
            events.append({"event": "rejected", "vis": vis})
            return events
        header_sample = max(0, round(vis["header_offset_seconds"] * AUDIO_RATE))
        initial = self.search_audio[header_sample:].copy()
        self.active_audio = [initial]
        self.active_samples = len(initial)
        self.target_samples = round(self.mode_capture_seconds[vis["vis_code"]] * AUDIO_RATE)
        self.trigger = vis
        self.search_audio = np.empty(0, dtype=np.float64)
        events.append({"event": "triggered", "vis": vis,
                       "target_seconds": self.target_samples / AUDIO_RATE})
        return events


def doppler_frequency_at(plan: list[dict], at: datetime) -> float | None:
    if not plan:
        return None
    target = at.astimezone(timezone.utc).timestamp()
    times = np.asarray([
        datetime.fromisoformat(item["at"]).astimezone(timezone.utc).timestamp()
        for item in plan
    ], dtype=np.float64)
    frequencies = np.asarray(
        [item["corrected_receive_frequency_hz"] for item in plan], dtype=np.float64
    )
    return float(np.interp(target, times, frequencies))


@dataclass
class WatchJob:
    job_id: str
    config: dict
    state: str = "queued"
    phase: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    trigger_count: int = 0
    rejected_trigger_count: int = 0
    decode_failure_count: int = 0
    images: list[dict] = field(default_factory=list)
    current_vis: dict | None = None
    streamed_iq_samples: int = 0
    result_json_path: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class SSTVWatcherManager:
    def __init__(self) -> None:
        self._jobs: dict[str, WatchJob] = {}
        self._lock = threading.RLock()

    def start(
        self, *, frequency_hz: int, receiver_mode: str = "nfm",
        watch_duration_seconds: float = 3600, rearm: bool = True,
        retain_audio: bool = True, deduplicate: bool = True,
        source_preset_id: str | None = None,
        source_schedule_id: str | None = None,
        source_satellite_watch_id: str | None = None,
        source_satellite_pass_id: str | None = None,
        doppler_correction_mode: str = "off",
        doppler_plan: list[dict] | None = None,
    ) -> dict:
        frequency_hz = validate_frequency(frequency_hz)
        receiver_mode = str(receiver_mode).strip().lower()
        if receiver_mode not in {"usb", "nfm"}:
            raise ValueError("receiver_mode must be usb or nfm")
        watch_duration_seconds = float(watch_duration_seconds)
        if not 30 <= watch_duration_seconds <= 86_400:
            raise ValueError("watch_duration_seconds must be from 30 through 86400")
        if not all(isinstance(value, bool) for value in (rearm, retain_audio, deduplicate)):
            raise ValueError("rearm, retain_audio, and deduplicate must be JSON booleans")
        doppler_correction_mode = str(doppler_correction_mode).strip().lower()
        if doppler_correction_mode not in {"off", "digital"}:
            raise ValueError("doppler_correction_mode must be off or digital")
        doppler_plan = list(doppler_plan or [])
        if doppler_correction_mode == "digital" and not doppler_plan:
            raise ValueError("digital Doppler correction requires a Doppler plan")
        if not sstv_decoder_path():
            raise RuntimeError("SSTV decoder is unavailable; run install-sstv-decoder.sh")
        job_id = f"sstv-watch-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        config = {
            "frequency_hz": frequency_hz, "receiver_mode": receiver_mode,
            "watch_duration_seconds": watch_duration_seconds, "rearm": rearm,
            "retain_audio": retain_audio, "deduplicate": deduplicate,
            "pre_trigger_seconds": 3.0,
            "doppler_correction_mode": doppler_correction_mode,
            "doppler_plan": doppler_plan,
        }
        if source_preset_id:
            config["source_preset_id"] = source_preset_id
        if source_schedule_id:
            config["source_schedule_id"] = source_schedule_id
        if source_satellite_watch_id:
            config["source_satellite_watch_id"] = source_satellite_watch_id
        if source_satellite_pass_id:
            config["source_satellite_pass_id"] = source_satellite_pass_id
        job = WatchJob(job_id, config)
        with self._lock:
            acquire_long_job(job_id)
            try:
                self._jobs[job_id] = job
                catalog.upsert_job(job_id, "sstv_watch", "queued", config=config,
                                   created_at=job.created_at)
                job.thread = threading.Thread(
                    target=self._run, args=(job,), name=f"rf-mcp-{job_id}", daemon=True
                )
                job.thread.start()
            except Exception:
                self._jobs.pop(job_id, None)
                release_long_job(job_id)
                raise
        return self.status(job_id)

    def _run(self, job: WatchJob) -> None:
        ensure_data_dirs()
        job.state, job.phase = "running", "watching"
        job.started_at = datetime.now(timezone.utc).isoformat()
        work_dir = SSTV_DIR / job.job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        decode_queue: queue.Queue = queue.Queue(maxsize=4)
        decoder = threading.Thread(
            target=self._decoder_worker, args=(job, decode_queue, work_dir), daemon=True
        )
        detector = SSTVStreamDetector()
        decoder.start()
        decode_sequence = 0
        completed_single = False
        try:
            self._persist(job)
            center = offset_capture_center(job.config["frequency_hz"], offset_hz=10_000)
            demodulator = StreamingDemodulator(
                mode=job.config["receiver_mode"],
                offset_hz=job.config["frequency_hz"] - center,
            )
            for iq_values in stream_iq_chunks(
                center, duration_seconds=job.config["watch_duration_seconds"],
                stop_event=job.stop_event, chunk_seconds=0.5,
            ):
                job.streamed_iq_samples += len(iq_values) // 2
                receive_frequency = None
                if job.config.get("doppler_correction_mode") == "digital":
                    receive_frequency = doppler_frequency_at(
                        job.config.get("doppler_plan", []), datetime.now(timezone.utc)
                    )
                offset = (
                    receive_frequency - center if receive_frequency is not None
                    else job.config["frequency_hz"] - center
                )
                for event in detector.feed(demodulator.process(iq_values, offset_hz=offset)):
                    if event["event"] == "triggered":
                        with job.lock:
                            job.trigger_count += 1
                            job.current_vis = event["vis"]
                            job.phase = "receiving_image"
                    elif event["event"] == "rejected":
                        with job.lock:
                            job.rejected_trigger_count += 1
                    elif event["event"] == "complete":
                        decode_sequence += 1
                        event["sequence"] = decode_sequence
                        decode_queue.put(event)
                        with job.lock:
                            job.current_vis = None
                            job.phase = "watching" if job.config["rearm"] else "decoding"
                        if not job.config["rearm"]:
                            completed_single = True
                            job.stop_event.set()
                if job.stop_event.is_set():
                    break
            job.state = (
                "completed" if completed_single or not job.stop_event.is_set() else "stopped"
            )
            job.phase = "finishing_decodes"
        except Exception as exc:
            job.state, job.phase = "failed", "finished"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            decode_queue.put(None)
            decoder.join(timeout=240)
            if decoder.is_alive():
                job.state = "failed"
                job.error = "SSTV decoder worker did not finish within 240 seconds"
            job.phase = "finished"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._write_result(job, work_dir)
            self._persist(job)
            release_long_job(job.job_id)

    def _decoder_worker(self, job: WatchJob, decode_queue: queue.Queue, work_dir: Path) -> None:
        while True:
            event = decode_queue.get()
            if event is None:
                return
            try:
                image = self._decode_clip(job, event, work_dir)
                with job.lock:
                    job.images.append(image)
            except Exception as exc:
                sequence = int(event.get("sequence", 0))
                child_id = f"{job.job_id}-image-{sequence:03d}"
                catalog.upsert_job(
                    child_id, "sstv_decode", "failed",
                    config={"source_watch_id": job.job_id, "vis": event.get("vis")},
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                with job.lock:
                    job.decode_failure_count += 1
                    job.images.append({
                        "outcome": "decode_failed", "vis": event.get("vis"),
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    def _decode_clip(self, job: WatchJob, event: dict, work_dir: Path) -> dict:
        sequence = int(event["sequence"])
        child_id = f"{job.job_id}-image-{sequence:03d}"
        child_dir = work_dir / f"image-{sequence:03d}"
        child_dir.mkdir(parents=True, exist_ok=False)
        wav_path = child_dir / "sstv-audio.wav"
        png_path = child_dir / "sstv-image.png"
        result_path = child_dir / "result.json"
        captured_at = datetime.now(timezone.utc).isoformat()
        config = {
            "source_watch_id": job.job_id, "frequency_hz": job.config["frequency_hz"],
            "receiver_mode": job.config["receiver_mode"], "vis": event["vis"],
            "doppler_correction_mode": job.config.get("doppler_correction_mode", "off"),
            "source_satellite_pass_id": job.config.get("source_satellite_pass_id"),
        }
        catalog.upsert_job(child_id, "sstv_decode", "running", config=config,
                           created_at=captured_at, started_at=captured_at)
        _write_decoder_wav(wav_path, event["audio"])
        completed = run_sstv_decoder(wav_path, png_path)
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        if completed.returncode or not png_path.exists():
            raise RuntimeError(
                f"SSTV decoder failed with exit status {completed.returncode}: "
                f"{output or 'no diagnostic text'}"
            )
        from PIL import Image

        with Image.open(png_path) as decoded:
            width, height = decoded.size
            pixels = np.asarray(decoded.convert("RGB"), dtype=np.float32)
            fingerprint = image_fingerprint(decoded)
        duplicate = None
        if job.config["deduplicate"]:
            duplicate = catalog.find_sstv_duplicate(
                fingerprint, frequency_hz=job.config["frequency_hz"]
            )
        contrast = float(np.mean(np.std(pixels, axis=(0, 1))) / 64.0)
        quality = float(np.clip(0.25 + 0.5 * contrast + 0.25, 0, 1))
        result = {
            "job_id": child_id, "source_watch_id": job.job_id,
            "source_preset_id": job.config.get("source_preset_id"),
            "source_schedule_id": job.config.get("source_schedule_id"),
            "source_satellite_watch_id": job.config.get("source_satellite_watch_id"),
            "source_satellite_pass_id": job.config.get("source_satellite_pass_id"),
            "outcome": "duplicate" if duplicate else "decoded",
            "frequency_hz": job.config["frequency_hz"],
            "nominal_frequency_hz": job.config["frequency_hz"],
            "receiver_mode": job.config["receiver_mode"],
            "doppler_correction_mode": job.config.get("doppler_correction_mode", "off"),
            "doppler_plan_point_count": len(job.config.get("doppler_plan", [])),
            "sstv_mode": event["vis"].get("mode"),
            "vis_code": event["vis"].get("vis_code"),
            "vis_parity_valid": event["vis"].get("parity_valid"),
            "vis": event["vis"], "width": width, "height": height,
            "quality": quality, "image_path": str(png_path.resolve()),
            "audio_path": str(wav_path.resolve()) if job.config["retain_audio"] else None,
            "captured_at": captured_at,
            "duration_seconds": len(event["audio"]) / AUDIO_RATE,
            "decoder_output": output, "image_hash": fingerprint,
            "duplicate_of": ((duplicate or {}).get("duplicate_of")
                             or (duplicate or {}).get("image_id")),
            "duplicate_hash_distance": (duplicate or {}).get("hash_distance"),
        }
        stored = catalog.add_sstv_image(result)
        result["image_id"] = stored["image_id"]
        result["result_json_path"] = str(result_path.resolve())
        image_artifact = catalog.register_artifact(
            png_path, "sstv_image", job_id=child_id
        )
        result["image_artifact_id"] = image_artifact["artifact_id"]
        result["image_download_path"] = f"/artifacts/{image_artifact['artifact_id']}"
        if job.config["retain_audio"]:
            catalog.register_artifact(wav_path, "sstv_audio", job_id=child_id)
        else:
            wav_path.unlink(missing_ok=True)
        completed_at = datetime.now(timezone.utc).isoformat()
        catalog.upsert_job(
            child_id, "sstv_decode", "completed", config=config,
            summary={"phase": "finished", "image_id": stored["image_id"],
                     "outcome": result["outcome"], "source_watch_id": job.job_id},
            result_json_path=result_path, created_at=captured_at, started_at=captured_at,
            completed_at=completed_at,
        )
        from .sstv_alerts import evaluate_sstv_image
        events = evaluate_sstv_image(catalog, result)
        result["alert_event_count"] = len(events)
        result["alert_event_ids"] = [item["event_id"] for item in events]
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        catalog.register_artifact(result_path, "sstv_json", job_id=child_id)
        return result

    def _write_result(self, job: WatchJob, work_dir: Path) -> None:
        result_path = work_dir / "result.json"
        job.result_json_path = str(result_path.resolve())
        result_path.write_text(json.dumps(self.results(job.job_id), indent=2) + "\n",
                               encoding="utf-8")
        catalog.register_artifact(result_path, "sstv_watch_json", job_id=job.job_id)

    def _persist(self, job: WatchJob) -> None:
        catalog.upsert_job(
            job.job_id, "sstv_watch", job.state, config=job.config,
            summary={
                "phase": job.phase, "trigger_count": job.trigger_count,
                "image_count": len([item for item in job.images if item.get("image_id")]),
                "decode_failure_count": job.decode_failure_count,
            },
            result_json_path=job.result_json_path, created_at=job.created_at,
            started_at=job.started_at, completed_at=job.completed_at, error=job.error,
        )

    def _get(self, job_id: str) -> WatchJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        persisted = catalog.get_job(job_id)
        if persisted["job_type"] != "sstv_watch":
            raise ValueError(f"Unknown SSTV watcher job_id: {job_id}")
        result = persisted.get("result") or {}
        summary = persisted.get("summary") or {}
        return WatchJob(
            job_id, persisted["config"], state=persisted["state"],
            phase=summary.get("phase", "finished"), created_at=persisted["created_at"],
            started_at=persisted["started_at"], completed_at=persisted["completed_at"],
            error=persisted["error"], trigger_count=result.get("trigger_count", 0),
            rejected_trigger_count=result.get("rejected_trigger_count", 0),
            decode_failure_count=result.get("decode_failure_count", 0),
            images=result.get("images", []),
            streamed_iq_samples=result.get("streamed_iq_samples", 0),
            result_json_path=persisted.get("result_json_path"),
        )

    def status(self, job_id: str) -> dict:
        job = self._get(job_id)
        with job.lock:
            return {
                "job_id": job.job_id, "state": job.state, "phase": job.phase,
                "created_at": job.created_at, "started_at": job.started_at,
                "completed_at": job.completed_at, "error": job.error,
                "frequency_hz": job.config["frequency_hz"],
                "receiver_mode": job.config["receiver_mode"],
                "trigger_count": job.trigger_count,
                "rejected_trigger_count": job.rejected_trigger_count,
                "image_count": len([item for item in job.images if item.get("image_id")]),
                "decode_failure_count": job.decode_failure_count,
                "current_vis": job.current_vis,
                "streamed_seconds": job.streamed_iq_samples / SAMPLE_RATE,
                "doppler_correction_mode": job.config.get("doppler_correction_mode", "off"),
                "current_corrected_frequency_hz": doppler_frequency_at(
                    job.config.get("doppler_plan", []), datetime.now(timezone.utc)
                ) if job.config.get("doppler_correction_mode") == "digital" else None,
            }

    def results(self, job_id: str) -> dict:
        job = self._get(job_id)
        with job.lock:
            return {
                **self.status(job_id), "config": job.config,
                "images": list(job.images), "result_json_path": job.result_json_path,
            }

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.state not in TERMINAL_STATES:
            job.stop_event.set()
        return self.status(job_id)

    def list_sessions(self, *, state: str | None = None, limit: int = 50) -> list[dict]:
        return catalog.list_jobs(job_type="sstv_watch", state=state, limit=limit)


sstv_watcher_manager = SSTVWatcherManager()
