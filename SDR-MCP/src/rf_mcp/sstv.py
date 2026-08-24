from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from scipy.signal import hilbert, resample_poly

from .receiver_backend import capture_iq, offset_capture_center, validate_frequency
from .catalog import catalog
from .config import SSTV_DIR, ensure_data_dirs
from .operations import acquire_long_job, release_long_job
from .weak_signal import _write_decoder_wav, iq_cycle_to_audio


SSTV_MODES = {
    44: "Martin M1", 40: "Martin M2", 60: "Scottie S1",
    56: "Scottie S2", 76: "Scottie DX", 8: "Robot 36", 12: "Robot 72",
}
TERMINAL_STATES = {"completed", "stopped", "failed", "interrupted"}
HEADLESS_DECODER_BOOTSTRAP = """
import os
import sstv.common
sstv.common.get_terminal_size = lambda *args: os.terminal_size((120, 24))
from sstv.__main__ import main
main()
"""


def sstv_decoder_path() -> str | None:
    configured = os.getenv("RF_MCP_SSTV_DECODER")
    if configured:
        return shutil.which(configured)
    venv_decoder = Path(sys.executable).with_name("sstv")
    if venv_decoder.is_file() and os.access(venv_decoder, os.X_OK):
        return str(venv_decoder)
    return shutil.which("sstv")


def sstv_capabilities() -> dict:
    executable = sstv_decoder_path()
    return {
        "available": executable is not None,
        "executable": executable,
        "implementation": "colaclanth/sstv command-line WAV decoder",
        "modes": [
            {"vis_code": code, "name": name} for code, name in SSTV_MODES.items()
        ],
        "receiver_modes": ["usb", "nfm"],
        "auto_vis_detection": True,
        "max_capture_seconds": 310,
        "headless_terminal_compatibility": True,
        "streaming_watcher": {
            "available": True,
            "maximum_watch_seconds": 86_400,
            "pre_trigger_seconds": 3.0,
            "rearm_supported": True,
            "retains_iq": False,
        },
    }


def run_sstv_decoder(wav_path: Path, png_path: Path) -> subprocess.CompletedProcess:
    """Run the upstream decoder with a safe virtual terminal size for systemd."""
    if not sstv_decoder_path():
        raise RuntimeError("SSTV decoder is unavailable; run install-sstv-decoder.sh")
    return subprocess.run(
        [sys.executable, "-c", HEADLESS_DECODER_BOOTSTRAP,
         "-d", str(wav_path), "-o", str(png_path)],
        capture_output=True, text=True, timeout=360, check=False,
    )


def image_fingerprint(image: Image.Image) -> str:
    """Return a 256-bit difference hash suitable for near-duplicate grouping."""
    grayscale = np.asarray(
        image.convert("L").resize((17, 16), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    bits = (grayscale[:, 1:] >= grayscale[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:064x}"


def decoder_indicates_no_signal(decoder_output: str, vis: dict) -> bool:
    return (
        not vis.get("detected")
        and "couldn't find sstv header" in decoder_output.lower()
    )


def detect_vis(audio: np.ndarray, sample_rate_hz: int = 12_000) -> dict:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size < sample_rate_hz:
        return {"detected": False, "vis_code": None, "mode": None,
                "parity_valid": None, "header_offset_seconds": None}
    phase = np.unwrap(np.angle(hilbert(audio)))
    instantaneous = np.diff(phase) * sample_rate_hz / (2 * np.pi)
    frame = max(1, round(sample_rate_hz * 0.005))
    count = len(instantaneous) // frame
    frequency = np.median(instantaneous[:count * frame].reshape(count, frame), axis=1)
    frame_seconds = frame / sample_rate_hz

    def tone(start: int, seconds: float) -> float:
        length = max(1, round(seconds / frame_seconds))
        return float(np.median(frequency[start:start + length]))

    header_frames = round(0.610 / frame_seconds)
    vis_start_frames = round(0.030 / frame_seconds)
    for start in range(0, max(0, len(frequency) - header_frames - 60), 2):
        if abs(tone(start, 0.25) - 1900) > 120:
            continue
        if abs(tone(start + round(0.300 / frame_seconds), 0.010) - 1200) > 180:
            continue
        if abs(tone(start + round(0.310 / frame_seconds), 0.25) - 1900) > 120:
            continue
        bit_start = start + header_frames
        if abs(tone(bit_start, 0.030) - 1200) > 180:
            continue
        bits = []
        for index in range(8):
            measured = tone(bit_start + vis_start_frames * (index + 1), 0.030)
            bits.append(1 if abs(measured - 1100) < abs(measured - 1300) else 0)
        stop = tone(bit_start + vis_start_frames * 9, 0.030)
        if abs(stop - 1200) > 180:
            continue
        code = sum(bits[index] << index for index in range(7))
        parity_valid = sum(bits) % 2 == 0
        return {
            "detected": True, "vis_code": code, "mode": SSTV_MODES.get(code),
            "parity_valid": parity_valid,
            "header_offset_seconds": start * frame_seconds,
            "raw_bits_lsb_first": bits,
        }
    return {"detected": False, "vis_code": None, "mode": None,
            "parity_valid": None, "header_offset_seconds": None}


def iq_to_nfm_audio(
    path: Path, *, sample_count: int, sample_rate_hz: int, offset_hz: float,
) -> np.ndarray:
    raw = np.memmap(path, dtype="<f4", mode="r")
    output = []
    chunk_samples = sample_rate_hz * 5
    for start in range(0, sample_count, chunk_samples):
        stop = min(start + chunk_samples, sample_count)
        values = np.asarray(raw[start * 2:stop * 2])
        iq = values[0::2].astype(np.complex64)
        iq.imag = values[1::2]
        indices = np.arange(start, stop, dtype=np.float64)
        mixed = iq * np.exp(-2j * np.pi * offset_hz * indices / sample_rate_hz)
        baseband = resample_poly(mixed, 1, sample_rate_hz // 12_000)
        discriminator = np.empty(len(baseband), dtype=np.float64)
        discriminator[0] = 0
        discriminator[1:] = np.angle(baseband[1:] * np.conj(baseband[:-1]))
        output.append(discriminator)
    audio = np.concatenate(output) if output else np.empty(0, dtype=np.float64)
    audio -= np.mean(audio) if audio.size else 0
    peak = float(np.percentile(np.abs(audio), 99.5)) if audio.size else 0
    return 0.8 * audio / peak if peak else audio


@dataclass
class SSTVJob:
    job_id: str
    config: dict
    state: str = "queued"
    phase: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    outcome: str | None = None
    image: dict | None = None
    result_json_path: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class SSTVManager:
    def __init__(self) -> None:
        self._jobs: dict[str, SSTVJob] = {}
        self._lock = threading.RLock()

    def start(
        self, *, frequency_hz: int, duration_seconds: float = 130,
        receiver_mode: str = "usb", retain_audio: bool = True,
        retain_iq: bool = False, deduplicate: bool = True,
        source_preset_id: str | None = None,
        source_schedule_id: str | None = None,
    ) -> dict:
        frequency_hz = validate_frequency(frequency_hz)
        duration_seconds = float(duration_seconds)
        if not 20 <= duration_seconds <= 310:
            raise ValueError("duration_seconds must be from 20 through 310")
        receiver_mode = receiver_mode.strip().lower()
        if receiver_mode not in {"usb", "nfm"}:
            raise ValueError("receiver_mode must be usb or nfm")
        if (not isinstance(retain_audio, bool) or not isinstance(retain_iq, bool)
                or not isinstance(deduplicate, bool)):
            raise ValueError("retain_audio, retain_iq, and deduplicate must be JSON booleans")
        if not sstv_decoder_path():
            raise RuntimeError("SSTV decoder is unavailable; run install-sstv-decoder.sh")
        job_id = f"sstv-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        config = {
            "frequency_hz": frequency_hz, "duration_seconds": duration_seconds,
            "receiver_mode": receiver_mode, "retain_audio": retain_audio,
            "retain_iq": retain_iq, "deduplicate": deduplicate,
        }
        if source_preset_id:
            config["source_preset_id"] = source_preset_id
        if source_schedule_id:
            config["source_schedule_id"] = source_schedule_id
        job = SSTVJob(job_id, config)
        with self._lock:
            acquire_long_job(job_id)
            try:
                self._jobs[job_id] = job
                catalog.upsert_job(job_id, "sstv_decode", "queued", config=config,
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

    def _run(self, job: SSTVJob) -> None:
        ensure_data_dirs()
        job.state, job.phase, job.started_at = "running", "capturing", datetime.now(timezone.utc).isoformat()
        capture = None
        try:
            self._persist(job)
            capture = capture_iq(
                offset_capture_center(job.config["frequency_hz"], offset_hz=10_000),
                job.config["duration_seconds"], extended_duration=True,
            )
            if job.stop_event.is_set():
                job.state, job.phase = "stopped", "finished"
                return
            job.phase = "demodulating"
            offset = job.config["frequency_hz"] - capture.center_frequency_hz
            if job.config["receiver_mode"] == "usb":
                audio = iq_cycle_to_audio(
                    capture.path, first_sample=0, sample_count=capture.captured_samples,
                    sample_rate_hz=capture.sample_rate_hz, offset_hz=offset,
                )
            else:
                audio = iq_to_nfm_audio(
                    capture.path, sample_count=capture.captured_samples,
                    sample_rate_hz=capture.sample_rate_hz, offset_hz=offset,
                )
            job_dir = SSTV_DIR / job.job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            wav_path = job_dir / "sstv-audio.wav"
            png_path = job_dir / "sstv-image.png"
            _write_decoder_wav(wav_path, audio)
            vis = detect_vis(audio)
            job.phase = "decoding"
            completed = run_sstv_decoder(wav_path, png_path)
            decoder_output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            ).strip()
            if completed.returncode or not png_path.exists():
                no_signal = decoder_indicates_no_signal(decoder_output, vis)
                if no_signal:
                    job.outcome = "no_signal"
                    result_path = job_dir / "result.json"
                    job.result_json_path = str(result_path.resolve())
                    result = {
                        "job_id": job.job_id, "outcome": job.outcome,
                        "frequency_hz": job.config["frequency_hz"],
                        "receiver_mode": job.config["receiver_mode"], "vis": vis,
                        "captured_at": capture.started_at,
                        "duration_seconds": job.config["duration_seconds"],
                        "decoder_output": decoder_output,
                        "audio_path": (str(wav_path.resolve())
                                       if job.config["retain_audio"] else None),
                        "image_id": None, "image_path": None,
                        "result_json_path": job.result_json_path,
                    }
                    result_path.write_text(
                        json.dumps(result, indent=2) + "\n", encoding="utf-8"
                    )
                    catalog.register_artifact(result_path, "sstv_json", job_id=job.job_id)
                    if job.config["retain_audio"]:
                        catalog.register_artifact(wav_path, "sstv_audio", job_id=job.job_id)
                    else:
                        wav_path.unlink(missing_ok=True)
                    job.state, job.phase = "completed", "finished"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    return
                raise RuntimeError(
                    f"SSTV decoder failed with exit status {completed.returncode}: "
                    f"{decoder_output or 'no diagnostic text'}"
                )
            with Image.open(png_path) as image:
                width, height = image.size
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
                image_hash = image_fingerprint(image)
            duplicate = None
            if job.config["deduplicate"]:
                duplicate = catalog.find_sstv_duplicate(
                    image_hash, frequency_hz=job.config["frequency_hz"]
                )
            contrast = float(np.mean(np.std(pixels, axis=(0, 1))) / 64.0)
            quality = float(np.clip(0.25 + 0.5 * contrast +
                                    (0.25 if vis.get("parity_valid") else 0), 0, 1))
            result = {
                "job_id": job.job_id, "frequency_hz": job.config["frequency_hz"],
                "receiver_mode": job.config["receiver_mode"],
                "sstv_mode": vis.get("mode"), "vis_code": vis.get("vis_code"),
                "vis_parity_valid": vis.get("parity_valid"), "vis": vis,
                "width": width, "height": height, "quality": quality,
                "image_path": str(png_path.resolve()),
                "audio_path": str(wav_path.resolve()) if job.config["retain_audio"] else None,
                "captured_at": capture.started_at,
                "duration_seconds": job.config["duration_seconds"],
                "decoder_output": decoder_output,
                "iq_capture_path": str(capture.path.resolve()) if job.config["retain_iq"] else None,
                "image_hash": image_hash,
                "duplicate_of": ((duplicate or {}).get("duplicate_of")
                                 or (duplicate or {}).get("image_id")),
                "duplicate_hash_distance": (duplicate or {}).get("hash_distance"),
                "source_preset_id": job.config.get("source_preset_id"),
                "source_schedule_id": job.config.get("source_schedule_id"),
            }
            job.image = catalog.add_sstv_image(result)
            job.outcome = "duplicate" if duplicate else "decoded"
            result_path = job_dir / "result.json"
            job.result_json_path = str(result_path.resolve())
            result["image_id"] = job.image["image_id"]
            result["outcome"] = job.outcome
            result["result_json_path"] = job.result_json_path
            image_artifact = catalog.register_artifact(
                png_path, "sstv_image", job_id=job.job_id
            )
            result["image_artifact_id"] = image_artifact["artifact_id"]
            result["image_download_path"] = f"/artifacts/{image_artifact['artifact_id']}"
            if job.config["retain_audio"]:
                catalog.register_artifact(wav_path, "sstv_audio", job_id=job.job_id)
            else:
                wav_path.unlink(missing_ok=True)
            if job.config["retain_iq"]:
                catalog.register_artifact(capture.path, "iq_capture", job_id=job.job_id)
            from .sstv_alerts import evaluate_sstv_image
            events = evaluate_sstv_image(catalog, result)
            result["alert_event_count"] = len(events)
            result["alert_event_ids"] = [item["event_id"] for item in events]
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            catalog.register_artifact(result_path, "sstv_json", job_id=job.job_id)
            job.state, job.phase = "completed", "finished"
            job.completed_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            job.state, job.phase = "failed", "finished"
            job.error = f"{type(exc).__name__}: {exc}"
            job.completed_at = datetime.now(timezone.utc).isoformat()
        finally:
            if capture is not None and not job.config["retain_iq"]:
                Path(capture.path).unlink(missing_ok=True)
            self._persist(job)
            release_long_job(job.job_id)

    def _persist(self, job: SSTVJob) -> None:
        catalog.upsert_job(
            job.job_id, "sstv_decode", job.state, config=job.config,
            summary={
                "phase": job.phase,
                "image_id": (job.image or {}).get("image_id"),
                "outcome": job.outcome,
            },
            result_json_path=job.result_json_path, created_at=job.created_at,
            started_at=job.started_at, completed_at=job.completed_at, error=job.error,
        )

    def _get(self, job_id: str) -> SSTVJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            persisted = catalog.get_job(job_id)
            if persisted["job_type"] != "sstv_decode":
                raise ValueError(f"Unknown SSTV job_id: {job_id}")
            result = persisted.get("result") or {}
            return SSTVJob(
                job_id, persisted["config"], state=persisted["state"],
                phase=(persisted.get("summary") or {}).get("phase", "finished"),
                created_at=persisted["created_at"], started_at=persisted["started_at"],
                completed_at=persisted["completed_at"], error=persisted["error"],
                outcome=(persisted.get("summary") or {}).get("outcome"),
                image=(catalog.get_sstv_image(result["image_id"])
                       if result.get("image_id") else None),
                result_json_path=persisted.get("result_json_path"),
            )
        return job

    def status(self, job_id: str) -> dict:
        job = self._get(job_id)
        return {
            "job_id": job.job_id, "state": job.state, "phase": job.phase,
            "created_at": job.created_at, "started_at": job.started_at,
            "completed_at": job.completed_at, "error": job.error,
            "image_id": (job.image or {}).get("image_id"), "outcome": job.outcome,
        }

    def results(self, job_id: str) -> dict:
        job = self._get(job_id)
        return {**self.status(job_id), "config": job.config, "image": job.image,
                "result_json_path": job.result_json_path}

    def stop(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.state not in TERMINAL_STATES:
            job.stop_event.set()
        return self.status(job_id)


sstv_manager = SSTVManager()
