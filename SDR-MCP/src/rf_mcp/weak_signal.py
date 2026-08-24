from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram

from .receiver_backend import capture_iq, offset_capture_center
from .catalog import catalog
from .config import WEAK_SIGNAL_DIR, ensure_data_dirs


MODE_PERIOD_SECONDS = {"ft8": 15.0, "ft4": 7.5, "wspr": 120.0}
MODE_EXECUTABLE = {"ft8": "jt9", "ft4": "jt9", "wspr": "wsprd"}
CAPTURE_LEAD_SECONDS = 1.0
DECODER_PREROLL_SECONDS = 0.25
CALLSIGN_RE = re.compile(r"(?<![A-Z0-9/])(?=[A-Z0-9/]{3,12}(?![A-Z0-9/]))(?=[A-Z0-9/]*[A-Z])(?=[A-Z0-9/]*\d)[A-Z0-9]+(?:/[A-Z0-9]+)?")
GRID_RE = re.compile(r"\b[A-R]{2}\d{2}(?:[A-X]{2})?\b", re.IGNORECASE)


def normalize_weak_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in MODE_PERIOD_SECONDS:
        raise ValueError("mode must be ft8, ft4, or wspr")
    return mode


def _executable(mode: str) -> str | None:
    name = os.getenv(f"RF_MCP_{MODE_EXECUTABLE[mode].upper()}", MODE_EXECUTABLE[mode])
    return shutil.which(name)


def decoder_capabilities() -> dict:
    modes = {}
    for mode in MODE_PERIOD_SECONDS:
        executable = _executable(mode)
        modes[mode] = {
            "available": executable is not None,
            "executable": executable,
            "decoder": MODE_EXECUTABLE[mode],
            "period_seconds": MODE_PERIOD_SECONDS[mode],
            "implementation": "WSJT-X command-line decoder",
        }
    return {
        "native_modes": ["cw", "rtty", "bpsk31", "ax25_afsk1200"],
        "weak_signal_modes": modes,
        "receive_only": True,
    }


def decoder_command(mode: str, wav_path: Path, work_dir: Path) -> list[str]:
    mode = normalize_weak_mode(mode)
    executable = _executable(mode)
    if not executable:
        package = "wsjtx"
        raise RuntimeError(
            f"{MODE_EXECUTABLE[mode]} was not found; install the Debian {package} package "
            "or configure its RF_MCP_* executable environment variable"
        )
    if mode == "ft8":
        return [executable, "-8", "-a", str(work_dir), str(wav_path)]
    if mode == "ft4":
        return [executable, "-5", "-a", str(work_dir), str(wav_path)]
    return [executable, "-a", str(work_dir), str(wav_path)]


def _message_fields(message: str) -> tuple[str | None, str | None, bool]:
    upper = message.upper().strip()
    grid = GRID_RE.search(upper)
    grid_text = grid.group(0).upper() if grid else None
    calls = [value for value in CALLSIGN_RE.findall(upper) if value != grid_text]
    return (calls[-1] if calls else None, grid_text,
            upper.startswith("CQ ") or upper == "CQ")


def parse_jt9_output(text: str, *, mode: str, dial_frequency_hz: int,
                     captured_at: str) -> list[dict]:
    spots = []
    pattern = re.compile(
        r"^\s*(\d{4,6})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(\d+(?:\.\d+)?)\s+\S\s+(.*?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        utc_text, snr, dt, audio_hz, message = match.groups()
        callsign, grid, is_cq = _message_fields(message)
        audio_frequency_hz = float(audio_hz)
        spots.append({
            "mode": mode, "dial_frequency_hz": int(dial_frequency_hz),
            "audio_frequency_hz": audio_frequency_hz,
            "rf_frequency_hz": int(dial_frequency_hz) + audio_frequency_hz,
            "utc_text": utc_text, "snr_db": float(snr),
            "time_offset_seconds": float(dt), "drift_hz_per_minute": None,
            "message": message.strip(), "callsign": callsign, "grid": grid,
            "power_dbm": None, "is_cq": is_cq, "captured_at": captured_at,
            "raw_line": line.rstrip(),
        })
    return spots


def parse_wsprd_output(text: str, *, dial_frequency_hz: int,
                       captured_at: str) -> list[dict]:
    spots = []
    pattern = re.compile(
        r"^\s*(\d{4})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
        r"(\d+(?:\.\d+)?)\s+(-?\d+)\s+([A-Z0-9/<>]+)"
        r"(?:\s+([A-R]{2}\d{2}(?:[A-X]{2})?))?(?:\s+(-?\d+))?",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        utc_text, snr, dt, frequency_mhz, drift, callsign, grid, power = match.groups()
        rf_hz = float(frequency_mhz) * 1_000_000
        spots.append({
            "mode": "wspr", "dial_frequency_hz": int(dial_frequency_hz),
            "audio_frequency_hz": rf_hz - int(dial_frequency_hz),
            "rf_frequency_hz": rf_hz, "utc_text": utc_text,
            "snr_db": float(snr), "time_offset_seconds": float(dt),
            "drift_hz_per_minute": float(drift), "message": " ".join(
                value for value in (callsign, grid, power) if value is not None
            ), "callsign": callsign.upper(), "grid": grid.upper() if grid else None,
            "power_dbm": int(power) if power is not None else None,
            "is_cq": False, "captured_at": captured_at, "raw_line": line.rstrip(),
        })
    return spots


def _write_decoder_wav(path: Path, audio_12k: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio_12k))) if audio_12k.size else 0.0
    if peak:
        audio_12k = 0.8 * audio_12k / peak
    wavfile.write(path, 12_000, np.int16(np.clip(audio_12k, -1, 1) * 32767))


def analyze_weak_audio(audio_12k: np.ndarray, sample_rate_hz: int = 12_000) -> dict:
    """Return simple level and spectral diagnostics for a decoder WAV."""
    audio = np.asarray(audio_12k, dtype=np.float64)
    if not audio.size:
        return {"classification": "empty", "sample_count": 0}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    rms_dbfs = 20 * np.log10(max(rms, 1e-12))
    peak_dbfs = 20 * np.log10(max(peak, 1e-12))
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.995))
    frequencies, _, power = spectrogram(
        audio, fs=sample_rate_hz, window="hann", nperseg=2048,
        noverlap=1536, scaling="spectrum", mode="psd",
    )
    band = (frequencies >= 100) & (frequencies <= 5_000)
    band_power = power[band]
    spectral_contrast_db = 0.0
    occupied_fraction = 0.0
    if band_power.size:
        db = 10 * np.log10(np.maximum(band_power, 1e-20))
        spectral_contrast_db = float(np.percentile(db, 99.9) - np.median(db))
        occupied_fraction = float(np.mean(db > np.median(db) + 8))
    if rms_dbfs < -55:
        classification = "very_low_level"
    elif clipping_fraction > 0.001:
        classification = "clipping"
    elif spectral_contrast_db >= 12 and occupied_fraction > 0:
        classification = "concentrated_tone_energy"
    else:
        classification = "noise_or_weak_signals"
    return {
        "classification": classification, "sample_count": int(audio.size),
        "duration_seconds": audio.size / sample_rate_hz,
        "rms_dbfs": round(float(rms_dbfs), 2),
        "peak_dbfs": round(float(peak_dbfs), 2),
        "clipping_fraction": round(clipping_fraction, 7),
        "spectral_contrast_db": round(spectral_contrast_db, 2),
        "occupied_time_frequency_fraction": round(occupied_fraction, 5),
    }


def plot_weak_audio(audio_12k: np.ndarray, path: Path, *, mode: str,
                    dial_frequency_hz: int, cycle: int) -> None:
    """Write a waveform and audio-frequency waterfall for visual diagnosis."""
    audio = np.asarray(audio_12k, dtype=np.float64)
    time_axis = np.arange(audio.size) / 12_000
    frequencies, times, power = spectrogram(
        audio, fs=12_000, window="hann", nperseg=2048, noverlap=1792,
        scaling="spectrum", mode="psd",
    )
    mask = (frequencies >= 0) & (frequencies <= 5_000)
    power_db = 10 * np.log10(np.maximum(power[mask], 1e-20))
    vmin, vmax = np.percentile(power_db, [10, 99.5]) if power_db.size else (-100, 0)
    fig, (wave_ax, waterfall_ax) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 3]},
        constrained_layout=True,
    )
    wave_ax.plot(time_axis, audio, linewidth=0.5, color="#f36d2e")
    wave_ax.set(xlabel="Time (seconds)", ylabel="Amplitude", xlim=(0, time_axis[-1]))
    wave_ax.grid(alpha=0.2)
    image = waterfall_ax.pcolormesh(
        times, frequencies[mask], power_db, shading="auto", cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    waterfall_ax.set(
        title=f"{mode.upper()} audio waterfall — {dial_frequency_hz / 1e6:.6f} MHz — cycle {cycle}",
        xlabel="Time (seconds)", ylabel="Audio frequency (Hz)", ylim=(0, 5_000),
    )
    fig.colorbar(image, ax=waterfall_ax, label="Relative power (dB)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def iq_cycle_to_audio(
    path: Path, *, first_sample: int, sample_count: int,
    sample_rate_hz: int, offset_hz: float,
) -> np.ndarray:
    """Convert one USB interval to 12 kHz audio without loading a long IQ file."""
    raw = np.memmap(path, dtype="<f4", mode="r")
    output = []
    chunk_samples = sample_rate_hz * 5
    stop_sample = first_sample + sample_count
    for start in range(first_sample, stop_sample, chunk_samples):
        stop = min(start + chunk_samples, stop_sample)
        values = np.asarray(raw[start * 2:stop * 2])
        iq = values[0::2].astype(np.complex64)
        iq.imag = values[1::2]
        indices = np.arange(start, stop, dtype=np.float64)
        oscillator = np.exp(-2j * np.pi * offset_hz * indices / sample_rate_hz)
        complex_audio = resample_poly(iq * oscillator, 1, sample_rate_hz // 12_000)
        spectrum = np.fft.fft(complex_audio)
        frequencies = np.fft.fftfreq(len(complex_audio), d=1 / 12_000)
        spectrum[(frequencies < 50) | (frequencies > 5_000)] = 0
        output.append(2 * np.real(np.fft.ifft(spectrum)))
    return np.concatenate(output) if output else np.empty(0, dtype=np.float64)


def seconds_to_next_period(period_seconds: float, now: float | None = None) -> float:
    now = time.time() if now is None else float(now)
    remainder = now % period_seconds
    return 0.0 if remainder < 0.02 else period_seconds - remainder


def decode_live_weak_signal(
    *, frequency_hz: int, mode: str, capture_cycles: int = 1,
    align_to_utc: bool = True, retain_iq: bool = False,
    retain_audio: bool = True,
) -> dict:
    mode = normalize_weak_mode(mode)
    capture_cycles = int(capture_cycles)
    maximum = {"ft8": 8, "ft4": 12, "wspr": 1}[mode]
    if not 1 <= capture_cycles <= maximum:
        raise ValueError(f"capture_cycles for {mode} must be from 1 through {maximum}")
    if not isinstance(align_to_utc, bool) or not isinstance(retain_iq, bool) \
            or not isinstance(retain_audio, bool):
        raise ValueError("align_to_utc, retain_iq, and retain_audio must be JSON booleans")
    if not _executable(mode):
        decoder_command(mode, Path("missing.wav"), WEAK_SIGNAL_DIR)

    ensure_data_dirs()
    period = MODE_PERIOD_SECONDS[mode]
    wait_seconds = seconds_to_next_period(period) if align_to_utc else 0.0
    capture_lead = min(CAPTURE_LEAD_SECONDS, wait_seconds) if align_to_utc else 0.0
    sleep_seconds = max(0.0, wait_seconds - capture_lead)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    duration = period * capture_cycles
    capture = capture_iq(
        offset_capture_center(int(frequency_hz), offset_hz=10_000),
        duration + capture_lead,
        extended_duration=True,
    )
    job_id = f"weak-{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    work_dir = WEAK_SIGNAL_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=False)
    wav_paths, plot_paths, diagnostics, raw_outputs, spots = [], [], [], [], []
    try:
        samples_per_cycle = round(capture.sample_rate_hz * period)
        capture_time = datetime.fromisoformat(capture.started_at)
        first_cycle_time = capture_time + timedelta(seconds=capture_lead)
        preroll = DECODER_PREROLL_SECONDS if capture_lead else 0.0
        first_sample = round(capture.sample_rate_hz * max(0.0, capture_lead - preroll))
        for cycle in range(capture_cycles):
            chunk = iq_cycle_to_audio(
                capture.path, first_sample=first_sample + cycle * samples_per_cycle,
                sample_count=samples_per_cycle, sample_rate_hz=capture.sample_rate_hz,
                offset_hz=int(frequency_hz) - capture.center_frequency_hz,
            )
            cycle_time = first_cycle_time + timedelta(seconds=cycle * period)
            wav_path = work_dir / f"{cycle_time:%y%m%d_%H%M%S}.wav"
            _write_decoder_wav(wav_path, chunk)
            wav_paths.append(str(wav_path.resolve()))
            diagnostic = analyze_weak_audio(chunk)
            diagnostic["cycle"] = cycle + 1
            diagnostics.append(diagnostic)
            plot_path = work_dir / f"{cycle_time:%y%m%d_%H%M%S}-waterfall.png"
            plot_weak_audio(
                chunk, plot_path, mode=mode, dial_frequency_hz=int(frequency_hz),
                cycle=cycle + 1,
            )
            plot_paths.append(str(plot_path.resolve()))
            command = decoder_command(mode, wav_path, work_dir)
            completed = subprocess.run(
                command, cwd=work_dir, capture_output=True, text=True,
                timeout=max(60, period * 2), check=False,
            )
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            raw_outputs.append({"cycle": cycle + 1, "return_code": completed.returncode,
                                "command": command, "output": output})
            if completed.returncode != 0:
                details = output.strip() or "decoder returned no diagnostic text"
                raise RuntimeError(
                    f"{MODE_EXECUTABLE[mode]} failed for cycle {cycle + 1} "
                    f"with exit status {completed.returncode}: {details}"
                )
            parser = parse_wsprd_output if mode == "wspr" else parse_jt9_output
            kwargs = {"dial_frequency_hz": int(frequency_hz),
                      "captured_at": capture.started_at}
            if mode != "wspr":
                kwargs["mode"] = mode
            spots.extend(parser(output, **kwargs))
        spots = catalog.add_weak_signal_spots(spots, job_id=job_id)
        result = {
            "job_id": job_id, "mode": mode, "dial_frequency_hz": int(frequency_hz),
            "capture_cycles": capture_cycles, "period_seconds": period,
            "duration_seconds": duration, "utc_alignment_wait_seconds": wait_seconds,
            "capture_lead_seconds": capture_lead,
            "decoder_preroll_seconds": preroll,
            "decoder": MODE_EXECUTABLE[mode], "decode_count": len(spots),
            "spots": spots, "audio_wav_paths": wav_paths if retain_audio else [],
            "audio_diagnostics": diagnostics,
            "waterfall_plot_paths": plot_paths,
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "started_at": capture.started_at, "decoder_runs": raw_outputs,
        }
        result_path = work_dir / "result.json"
        result["result_json_path"] = str(result_path.resolve())
        catalog.upsert_job(
            job_id, "weak_signal_decode", "completed",
            config={"frequency_hz": frequency_hz, "mode": mode,
                    "capture_cycles": capture_cycles, "align_to_utc": align_to_utc},
            summary={"decode_count": len(spots), "mode": mode,
                     "audio_diagnostics": diagnostics},
            result_json_path=result_path, created_at=capture.started_at,
            started_at=capture.started_at, completed_at=datetime.now(timezone.utc).isoformat(),
        )
        waterfall_artifacts = []
        for plot_path in map(Path, plot_paths):
            artifact = catalog.register_artifact(
                plot_path, "weak_signal_waterfall", job_id=job_id)
            waterfall_artifacts.append({
                **artifact, "download_path": f"/artifacts/{artifact['artifact_id']}"
            })
        result["waterfall_artifacts"] = waterfall_artifacts
        result_path.write_text(__import__("json").dumps(result, indent=2) + "\n", encoding="utf-8")
        catalog.register_artifact(result_path, "weak_signal_json", job_id=job_id)
        if retain_audio:
            for wav_path in map(Path, wav_paths):
                catalog.register_artifact(wav_path, "weak_signal_audio", job_id=job_id)
        else:
            for wav_path in map(Path, wav_paths):
                wav_path.unlink(missing_ok=True)
        if retain_iq:
            catalog.register_artifact(capture.path, "iq_capture", job_id=job_id)
        return result
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)
