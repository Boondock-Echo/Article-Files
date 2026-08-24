from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .airspyhf import Capture
from .config import CAPTURE_DIR, MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, ensure_data_dirs

RTL_SDR = "rtl_sdr"
RTL_TEST = "rtl_test"
DEFAULT_SAMPLE_RATE = 768_000
MIN_SAMPLE_RATE = 225_001
MAX_SAMPLE_RATE = 3_200_000
TUNING_RANGES_HZ = ((24_000_000, 1_766_000_000),)

_DEVICE_LOCK = threading.Lock()


class RtlSdrError(RuntimeError):
    """An RTL-SDR validation, discovery, capture, or streaming error."""


def validate_frequency(frequency_hz: int) -> int:
    frequency_hz = int(frequency_hz)
    if not any(low <= frequency_hz <= high for low, high in TUNING_RANGES_HZ):
        raise ValueError("RTL-SDR frequency must be from 24 MHz through 1.766 GHz")
    return frequency_hz


def validate_sample_rate(sample_rate_hz: int) -> int:
    sample_rate_hz = int(sample_rate_hz)
    if not MIN_SAMPLE_RATE <= sample_rate_hz <= MAX_SAMPLE_RATE:
        raise ValueError(
            f"RTL-SDR sample_rate_hz must be from {MIN_SAMPLE_RATE:,} through "
            f"{MAX_SAMPLE_RATE:,}"
        )
    return sample_rate_hz


def validate_device_selector(device_selector: str) -> str:
    selector = str(device_selector or "0").strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", selector):
        raise ValueError("RTL-SDR device_selector must be an index or simple serial value")
    return selector


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RtlSdrError(f"Required executable not found: {name}")
    return executable


def _diagnostics(completed: subprocess.CompletedProcess) -> str:
    stdout = completed.stdout.decode(errors="replace") if isinstance(completed.stdout, bytes) else completed.stdout
    stderr = completed.stderr.decode(errors="replace") if isinstance(completed.stderr, bytes) else completed.stderr
    return "\n".join(part for part in (stdout, stderr) if part).strip()


def device_info(device_selector: str = "") -> dict:
    selector = validate_device_selector(device_selector)
    command = [_executable(RTL_TEST), "-t", "-d", selector]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise RtlSdrError("rtl_test timed out after 15 seconds") from exc
    details = _diagnostics(completed)
    if completed.returncode != 0:
        raise RtlSdrError(details or f"rtl_test exited with status {completed.returncode}")
    found = re.search(r"Found\s+(.+?)(?:\r?\n|$)", details, re.IGNORECASE)
    using = re.search(r"Using device\s+(.+?)(?:\r?\n|$)", details, re.IGNORECASE)
    return {
        "connected": True,
        "model": (using or found).group(1).strip() if (using or found) else "RTL-SDR",
        "device_selector": selector,
        "capture_sample_rate_hz": DEFAULT_SAMPLE_RATE,
        "sample_rate_range_hz": [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE],
        "tuning_ranges_hz": [list(item) for item in TUNING_RANGES_HZ],
        "sample_format": "unsigned_8_bit_interleaved_iq",
        "raw_output": details,
    }


def _gain_arguments(*, agc: bool, gain_db: float | None) -> list[str]:
    if agc:
        if gain_db is not None:
            raise ValueError("gain_db cannot be supplied when agc=true")
        return []
    gain_db = 0.0 if gain_db is None else float(gain_db)
    if not 0 <= gain_db <= 60:
        raise ValueError("RTL-SDR gain_db must be from 0 through 60")
    return ["-g", f"{gain_db:g}"]


def _command(
    *, center_frequency_hz: int, sample_rate_hz: int, device_selector: str,
    sample_count: int | None, output: str, agc: bool, gain_db: float | None,
    frequency_correction_ppm: int,
) -> list[str]:
    frequency_correction_ppm = int(frequency_correction_ppm)
    if not -1_000 <= frequency_correction_ppm <= 1_000:
        raise ValueError("frequency_correction_ppm must be from -1000 through 1000")
    command = [
        _executable(RTL_SDR), "-d", validate_device_selector(device_selector),
        "-f", str(validate_frequency(center_frequency_hz)),
        "-s", str(validate_sample_rate(sample_rate_hz)),
        "-p", str(frequency_correction_ppm),
    ]
    command.extend(_gain_arguments(agc=bool(agc), gain_db=gain_db))
    if sample_count is not None:
        command.extend(["-n", str(int(sample_count))])
    command.append(output)
    return command


def _convert_cu8(values: bytes) -> np.ndarray:
    unsigned = np.frombuffer(values, dtype=np.uint8).astype(np.float32)
    return (unsigned - 127.5) / 127.5


def capture_iq(
    center_frequency_hz: int,
    duration_seconds: float,
    *,
    device_selector: str = "",
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE,
    agc: bool = True,
    gain_db: float | None = None,
    frequency_correction_ppm: int = 0,
    extended_duration: bool = False,
    **_ignored_options,
) -> Capture:
    duration_seconds = float(duration_seconds)
    maximum = 310 if extended_duration else MAX_DURATION_SECONDS
    if not MIN_DURATION_SECONDS <= duration_seconds <= maximum:
        raise ValueError(
            f"RTL-SDR capture duration must be from {MIN_DURATION_SECONDS:g} through {maximum:g} seconds"
        )
    sample_rate_hz = validate_sample_rate(sample_rate_hz)
    requested_samples = round(sample_rate_hz * duration_seconds)
    timestamp = datetime.now(timezone.utc)
    capture_id = f"{timestamp:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    ensure_data_dirs()
    final_path = CAPTURE_DIR / f"{capture_id}.iq"
    partial_path = final_path.with_suffix(".iq.part")
    raw_path = CAPTURE_DIR / f"{capture_id}.cu8.part"
    command = _command(
        center_frequency_hz=center_frequency_hz, sample_rate_hz=sample_rate_hz,
        device_selector=device_selector, sample_count=requested_samples, output=str(raw_path),
        agc=agc, gain_db=gain_db, frequency_correction_ppm=frequency_correction_ppm,
    )
    try:
        with _DEVICE_LOCK:
            completed = subprocess.run(command, capture_output=True, timeout=duration_seconds + 15)
        if completed.returncode != 0:
            raise RtlSdrError(
                _diagnostics(completed) or f"rtl_sdr exited with status {completed.returncode}"
            )
        if not raw_path.exists() or raw_path.stat().st_size % 2:
            size = raw_path.stat().st_size if raw_path.exists() else 0
            raise RtlSdrError(f"Unexpected RTL-SDR IQ file size: {size} bytes")
        captured_samples = raw_path.stat().st_size // 2
        if captured_samples < requested_samples * 0.98:
            raise RtlSdrError(
                f"Short capture: received {captured_samples:,} of {requested_samples:,} requested samples"
            )
        with raw_path.open("rb") as source, partial_path.open("wb") as destination:
            while block := source.read(1_048_576):
                destination.write(_convert_cu8(block).astype("<f4", copy=False).tobytes())
        partial_path.replace(final_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    finally:
        raw_path.unlink(missing_ok=True)
    return Capture(
        path=final_path, center_frequency_hz=int(center_frequency_hz),
        sample_rate_hz=sample_rate_hz, requested_samples=requested_samples,
        captured_samples=captured_samples, started_at=timestamp.isoformat(),
        backend="rtl_sdr", device_selector=validate_device_selector(device_selector),
    )


def stream_iq_chunks(
    center_frequency_hz: int,
    *,
    duration_seconds: float,
    stop_event: threading.Event,
    device_selector: str = "",
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE,
    chunk_seconds: float = 0.5,
    agc: bool = True,
    gain_db: float | None = None,
    frequency_correction_ppm: int = 0,
    **_ignored_options,
):
    duration_seconds = float(duration_seconds)
    chunk_seconds = float(chunk_seconds)
    if not 10 <= duration_seconds <= 86_400:
        raise ValueError("Streaming duration must be from 10 through 86400 seconds")
    if not 0.1 <= chunk_seconds <= 2.0:
        raise ValueError("chunk_seconds must be from 0.1 through 2.0")
    sample_rate_hz = validate_sample_rate(sample_rate_hz)
    command = _command(
        center_frequency_hz=center_frequency_hz, sample_rate_hz=sample_rate_hz,
        device_selector=device_selector, sample_count=None, output="-", agc=agc,
        gain_db=gain_db, frequency_correction_ppm=frequency_correction_ppm,
    )
    chunk_bytes = round(sample_rate_hz * chunk_seconds) * 2
    process = None
    diagnostics: list[bytes] = []
    started = time.monotonic()
    with _DEVICE_LOCK:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

            def drain_stderr() -> None:
                assert process is not None and process.stderr is not None
                while line := process.stderr.readline():
                    diagnostics.append(line)
                    if len(diagnostics) > 200:
                        diagnostics.pop(0)

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()
            assert process.stdout is not None
            pending = bytearray()
            while not stop_event.is_set() and time.monotonic() - started < duration_seconds:
                block = process.stdout.read(chunk_bytes - len(pending))
                if not block:
                    break
                pending.extend(block)
                if len(pending) == chunk_bytes:
                    yield _convert_cu8(bytes(pending))
                    pending.clear()
            if pending and len(pending) % 2 == 0:
                yield _convert_cu8(bytes(pending))
        except FileNotFoundError as exc:
            raise RtlSdrError(f"Required executable not found: {command[0]}") from exc
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            if process is not None and process.stderr is not None:
                process.stderr.close()
    if (process is not None and process.returncode not in (0, -15)
            and not stop_event.is_set() and time.monotonic() - started < duration_seconds):
        details = b"".join(diagnostics).decode(errors="replace").strip()
        raise RtlSdrError(details or f"rtl_sdr exited with status {process.returncode}")
