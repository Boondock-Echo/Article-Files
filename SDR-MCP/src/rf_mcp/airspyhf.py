from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import (
    AIRSPYHF_INFO,
    AIRSPYHF_RX,
    CAPTURE_DIR,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    SAMPLE_RATE,
    TUNING_RANGES_HZ,
    ensure_data_dirs,
)
from .subprocess_stream import read_chunks

_DEVICE_LOCK = threading.Lock()


class AirspyError(RuntimeError):
    """A receiver validation, discovery, or capture error."""


@dataclass(frozen=True)
class Capture:
    path: Path
    center_frequency_hz: int
    sample_rate_hz: int
    requested_samples: int
    captured_samples: int
    started_at: str
    receiver_id: str | None = None
    backend: str = "airspyhf"
    device_selector: str = ""
    calibration: dict | None = None


def validate_frequency(frequency_hz: int) -> int:
    frequency_hz = int(frequency_hz)
    if not any(low <= frequency_hz <= high for low, high in TUNING_RANGES_HZ):
        ranges = " or ".join(f"{low:,}-{high:,} Hz" for low, high in TUNING_RANGES_HZ)
        raise ValueError(f"Frequency must be within {ranges}.")
    return frequency_hz


def validate_duration(duration_seconds: float) -> float:
    duration_seconds = float(duration_seconds)
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Duration must be between {MIN_DURATION_SECONDS:g} and "
            f"{MAX_DURATION_SECONDS:g} seconds."
        )
    return duration_seconds


def offset_capture_center(target_frequency_hz: int, offset_hz: int = 50_000) -> int:
    """Choose a valid receiver center that keeps a target away from DC."""
    target_frequency_hz = validate_frequency(target_frequency_hz)
    for low, high in TUNING_RANGES_HZ:
        if low <= target_frequency_hz <= high:
            if target_frequency_hz + offset_hz <= high:
                return target_frequency_hz + offset_hz
            if target_frequency_hz - offset_hz >= low:
                return target_frequency_hz - offset_hz
    raise ValueError("Unable to choose an offset receiver center")


def _run_checked(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise AirspyError(f"Required executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AirspyError(f"Receiver command timed out after {timeout:g} seconds") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "unknown receiver error").strip()
        raise AirspyError(details) from exc


def device_info() -> dict:
    executable = shutil.which(AIRSPYHF_INFO)
    if not executable:
        raise AirspyError(f"{AIRSPYHF_INFO} is not installed or not on PATH")
    result = _run_checked([executable], timeout=10)
    text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()

    def match(pattern: str) -> str | None:
        found = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        return found.group(1).strip() if found else None

    rates_text = match(r"Available sample rates:\s*(.+)$") or ""
    rates = [int(value) * 1000 for value in re.findall(r"(\d+)\s*kS/s", rates_text)]
    return {
        "connected": True,
        "model": "Airspy HF+",
        "serial_number": match(r"S/N:\s*(\S+)"),
        "part_id": match(r"Part ID:\s*(\S+)"),
        "firmware_version": match(r"Firmware Version:\s*(.+)$"),
        "library_version": match(r"library version:\s*(\S+)"),
        "available_sample_rates_hz": rates,
        "capture_sample_rate_hz": SAMPLE_RATE,
        "tuning_ranges_hz": [list(item) for item in TUNING_RANGES_HZ],
        "raw_output": text,
    }


def capture_iq(
    center_frequency_hz: int,
    duration_seconds: float,
    *,
    agc: bool = True,
    agc_threshold: str = "low",
    attenuation_steps: int = 0,
    lna: bool = False,
    extended_duration: bool = False,
) -> Capture:
    center_frequency_hz = validate_frequency(center_frequency_hz)
    duration_seconds = float(duration_seconds)
    if extended_duration:
        if not MIN_DURATION_SECONDS <= duration_seconds <= 310:
            raise ValueError("Extended decoder capture must be from 0.25 through 310 seconds")
    else:
        duration_seconds = validate_duration(duration_seconds)
    if agc_threshold not in {"low", "high"}:
        raise ValueError("agc_threshold must be 'low' or 'high'")
    attenuation_steps = int(attenuation_steps)
    if not 0 <= attenuation_steps <= 8:
        raise ValueError("attenuation_steps must be from 0 through 8")
    ensure_data_dirs()

    executable = shutil.which(AIRSPYHF_RX)
    if not executable:
        raise AirspyError(f"{AIRSPYHF_RX} is not installed or not on PATH")

    requested_samples = round(SAMPLE_RATE * duration_seconds)
    timestamp = datetime.now(timezone.utc)
    capture_id = f"{timestamp:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    final_path = CAPTURE_DIR / f"{capture_id}.iq"
    partial_path = final_path.with_suffix(".iq.part")
    frequency_mhz = f"{center_frequency_hz / 1_000_000:.6f}"

    command = [
        executable,
        "-r", str(partial_path),
        "-f", frequency_mhz,
        "-a", str(SAMPLE_RATE),
        "-n", str(requested_samples),
        "-g", "on" if agc else "off",
        "-m", "on" if lna else "off",
    ]
    if agc:
        command.extend(["-l", agc_threshold])
    else:
        command.extend(["-t", str(attenuation_steps)])

    try:
        with _DEVICE_LOCK:
            _run_checked(command, timeout=duration_seconds + 15)
        if not partial_path.exists():
            raise AirspyError("Receiver completed without creating an IQ file")
        size = partial_path.stat().st_size
        if size == 0 or size % 8:
            raise AirspyError(f"Unexpected IQ file size: {size} bytes")
        captured_samples = size // 8
        if captured_samples < requested_samples * 0.98:
            raise AirspyError(
                f"Short capture: received {captured_samples:,} of "
                f"{requested_samples:,} requested samples"
            )
        partial_path.replace(final_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return Capture(
        path=final_path,
        center_frequency_hz=center_frequency_hz,
        sample_rate_hz=SAMPLE_RATE,
        requested_samples=requested_samples,
        captured_samples=captured_samples,
        started_at=timestamp.isoformat(),
    )


def stream_iq_chunks(
    center_frequency_hz: int,
    *,
    duration_seconds: float,
    stop_event: threading.Event,
    chunk_seconds: float = 0.5,
    agc: bool = True,
    agc_threshold: str = "low",
    attenuation_steps: int = 0,
    lna: bool = False,
):
    """Yield interleaved float32 IQ chunks directly from airspyhf_rx stdout."""
    center_frequency_hz = validate_frequency(center_frequency_hz)
    duration_seconds = float(duration_seconds)
    if not 10 <= duration_seconds <= 86_400:
        raise ValueError("Streaming duration must be from 10 through 86400 seconds")
    chunk_seconds = float(chunk_seconds)
    if not 0.1 <= chunk_seconds <= 2.0:
        raise ValueError("chunk_seconds must be from 0.1 through 2.0")
    if agc_threshold not in {"low", "high"}:
        raise ValueError("agc_threshold must be 'low' or 'high'")
    attenuation_steps = int(attenuation_steps)
    if not 0 <= attenuation_steps <= 8:
        raise ValueError("attenuation_steps must be from 0 through 8")
    executable = shutil.which(AIRSPYHF_RX)
    if not executable:
        raise AirspyError(f"{AIRSPYHF_RX} is not installed or not on PATH")
    command = [
        executable, "-r", "stdout",
        "-f", f"{center_frequency_hz / 1_000_000:.6f}",
        "-a", str(SAMPLE_RATE),
        "-g", "on" if agc else "off",
        "-m", "on" if lna else "off",
    ]
    if agc:
        command.extend(["-l", agc_threshold])
    else:
        command.extend(["-t", str(attenuation_steps)])
    chunk_bytes = round(SAMPLE_RATE * chunk_seconds) * 8
    diagnostics: list[bytes] = []
    process = None
    started = time.monotonic()
    with _DEVICE_LOCK:
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            )

            def drain_stderr() -> None:
                assert process is not None and process.stderr is not None
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    diagnostics.append(line)
                    if len(diagnostics) > 200:
                        diagnostics.pop(0)

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()
            assert process.stdout is not None
            for block in read_chunks(
                process.stdout, chunk_bytes, stop_event, started + duration_seconds,
            ):
                if len(block) % 8 == 0:
                    yield np.frombuffer(block, dtype="<f4")
        except FileNotFoundError as exc:
            raise AirspyError(f"Required executable not found: {command[0]}") from exc
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
        details = b"".join(diagnostics).decode("utf-8", errors="replace").strip()
        raise AirspyError(details or f"airspyhf_rx exited with status {process.returncode}")
