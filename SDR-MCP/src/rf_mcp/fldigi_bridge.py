from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import xmlrpc.client
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .receiver_backend import capture_iq, offset_capture_center
from .catalog import catalog
from .config import FLDIGI_DIR, ensure_data_dirs
from .weak_signal import CALLSIGN_RE, GRID_RE, _write_decoder_wav, iq_cycle_to_audio


FLDIGI_MODES = {
    "bpsk63": "BPSK63", "bpsk125": "BPSK125", "qpsk31": "QPSK31",
    "qpsk63": "QPSK63", "mfsk16": "MFSK16", "mfsk32": "MFSK32",
    "olivia-8-250": "OLIVIA-8-250", "olivia-8-500": "OLIVIA-8-500",
    "olivia-16-500": "OLIVIA-16-500", "olivia-16-1000": "OLIVIA-16-1000",
    "contestia-8-250": "CONTESTIA-8-250", "contestia-8-500": "CONTESTIA-8-500",
    "contestia-16-500": "CONTESTIA-16-500", "dominoex-11": "DOMINOEX-11",
    "dominoex-16": "DOMINOEX-16", "thor-11": "THOR-11", "thor-16": "THOR-16",
    "mt63-500l": "MT63-500L", "mt63-1000l": "MT63-1000L",
    "mt63-2000l": "MT63-2000L", "feldhell": "HELL", "slowhell": "SLOWHELL",
}
ALIASES = {
    "psk63": "bpsk63", "psk125": "bpsk125", "olivia": "olivia-8-500",
    "contestia": "contestia-8-500", "dominoex": "dominoex-11",
    "thor": "thor-11", "mt63": "mt63-1000l", "hell": "feldhell",
    "hellschreiber": "feldhell",
}


def normalize_fldigi_mode(mode: str) -> tuple[str, str]:
    normalized = mode.strip().lower().replace("_", "-").replace("/", "-")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in FLDIGI_MODES:
        raise ValueError("Unsupported Fldigi mode; call list_fldigi_modes for valid names")
    return normalized, FLDIGI_MODES[normalized]


def _rpc_url() -> str:
    return os.getenv("RF_MCP_FLDIGI_XMLRPC_URL", "http://127.0.0.1:7362/RPC2")


def _rpc() -> xmlrpc.client.ServerProxy:
    return xmlrpc.client.ServerProxy(_rpc_url(), allow_none=True, use_builtin_types=True)


def _binary_text(value) -> str:
    if isinstance(value, xmlrpc.client.Binary):
        value = value.data
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def fldigi_status() -> dict:
    executable = shutil.which(os.getenv("RF_MCP_FLDIGI", "fldigi"))
    playback = playback_command(Path("audio.wav"))[0]
    status = {
        "installed": executable is not None,
        "executable": executable,
        "xmlrpc_url": _rpc_url(),
        "xmlrpc_connected": False,
        "version": None,
        "current_modem": None,
        "available_modem_count": 0,
        "audio_playback_available": shutil.which(playback) is not None,
        "audio_playback_executable": shutil.which(playback),
        "error": None,
    }
    try:
        server = _rpc()
        status["version"] = server.fldigi.version()
        status["current_modem"] = server.modem.get_name()
        status["available_modem_count"] = len(server.modem.get_names())
        status["xmlrpc_connected"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def list_fldigi_mode_capabilities() -> dict:
    status = fldigi_status()
    available = []
    if status["xmlrpc_connected"]:
        names = {str(name).upper() for name in _rpc().modem.get_names()}
        available = [key for key, value in FLDIGI_MODES.items() if value.upper() in names]
    return {
        "status": status,
        "configured_modes": [
            {"mode": key, "fldigi_modem": value, "available": key in available}
            for key, value in FLDIGI_MODES.items()
        ],
        "aliases": ALIASES,
    }


def playback_command(wav_path: Path) -> list[str]:
    template = os.getenv(
        "RF_MCP_FLDIGI_PLAYBACK", "aplay -q -D plughw:Loopback,0,0 {wav}"
    )
    command = [str(wav_path) if item == "{wav}" else item for item in shlex.split(template)]
    if "{wav}" in command or not command:
        raise ValueError("RF_MCP_FLDIGI_PLAYBACK must contain a standalone {wav} argument")
    if str(wav_path) not in command:
        raise ValueError("RF_MCP_FLDIGI_PLAYBACK must contain a standalone {wav} argument")
    return command


def extract_text_entities(text: str) -> dict:
    upper = text.upper()
    grids = sorted(set(match.group(0).upper() for match in GRID_RE.finditer(upper)))
    callsigns = sorted(set(value for value in CALLSIGN_RE.findall(upper) if value not in grids))
    printable = "".join(char for char in text if char in "\n\r\t" or ord(char) >= 32)
    printable = re.sub(r"[ \t]+", " ", printable)
    printable = re.sub(r"\n{3,}", "\n\n", printable).strip()
    return {"text": printable, "callsigns": callsigns, "grids": grids}


def _read_rx_text(server) -> str:
    try:
        length = int(server.text.get_rx_length())
        return _binary_text(server.text.get_rx(0, length)) if length else ""
    except Exception:
        return _binary_text(server.rx.get_data())


def decode_live_fldigi(
    *, frequency_hz: int, mode: str, duration_seconds: float = 30,
    carrier_audio_hz: int = 1500, retain_iq: bool = False,
    retain_audio: bool = True,
) -> dict:
    mode, fldigi_modem = normalize_fldigi_mode(mode)
    duration_seconds = float(duration_seconds)
    if not 2 <= duration_seconds <= 120:
        raise ValueError("duration_seconds must be from 2 through 120")
    carrier_audio_hz = int(carrier_audio_hz)
    if not 200 <= carrier_audio_hz <= 4_500:
        raise ValueError("carrier_audio_hz must be from 200 through 4500")
    if not isinstance(retain_iq, bool) or not isinstance(retain_audio, bool):
        raise ValueError("retain_iq and retain_audio must be JSON booleans")
    status = fldigi_status()
    if not status["installed"] or not status["xmlrpc_connected"]:
        raise RuntimeError(
            "Fldigi is not ready; call get_fldigi_status and run the v0.22 setup steps"
        )
    if not status["audio_playback_available"]:
        raise RuntimeError("Configured Fldigi audio playback executable was not found")

    ensure_data_dirs()
    job_id = f"fldigi-{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    job_dir = FLDIGI_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    capture = capture_iq(
        offset_capture_center(int(frequency_hz), offset_hz=10_000), duration_seconds,
        extended_duration=True,
    )
    wav_path = job_dir / "receiver-audio.wav"
    try:
        audio = iq_cycle_to_audio(
            capture.path, first_sample=0, sample_count=capture.captured_samples,
            sample_rate_hz=capture.sample_rate_hz,
            offset_hz=int(frequency_hz) - capture.center_frequency_hz,
        )
        _write_decoder_wav(wav_path, audio)
        server = _rpc()
        available_names = {str(name).upper() for name in server.modem.get_names()}
        if fldigi_modem.upper() not in available_names:
            raise RuntimeError(f"Installed Fldigi does not advertise modem {fldigi_modem}")
        server.main.rx_only()
        server.main.set_wf_sideband("USB")
        server.main.set_squelch(False)
        server.modem.set_by_name(fldigi_modem)
        server.modem.set_carrier(carrier_audio_hz)
        server.text.clear_rx()
        server.main.rx()
        completed = subprocess.run(
            playback_command(wav_path), capture_output=True, text=True,
            timeout=duration_seconds + 15, check=False,
        )
        if completed.returncode:
            details = (completed.stderr or completed.stdout or "no diagnostic text").strip()
            raise RuntimeError(f"Fldigi audio playback failed: {details}")
        entities = extract_text_entities(_read_rx_text(server))
        quality = float(server.modem.get_quality())
        result = {
            "job_id": job_id, "mode": mode, "fldigi_modem": fldigi_modem,
            "dial_frequency_hz": int(frequency_hz),
            "carrier_audio_hz": carrier_audio_hz,
            "estimated_signal_frequency_hz": int(frequency_hz) + carrier_audio_hz,
            "duration_seconds": duration_seconds, "quality": quality,
            **entities, "captured_at": capture.started_at,
            "audio_wav_path": str(wav_path.resolve()) if retain_audio else None,
            "iq_capture_path": str(capture.path.resolve()) if retain_iq else None,
            "fldigi_version": status["version"],
        }
        result = catalog.add_fldigi_decode(result)
        result_path = job_dir / "result.json"
        result["result_json_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        catalog.upsert_job(
            job_id, "fldigi_decode", "completed",
            config={"frequency_hz": frequency_hz, "mode": mode,
                    "duration_seconds": duration_seconds,
                    "carrier_audio_hz": carrier_audio_hz},
            summary={"character_count": len(result["text"]), "quality": quality,
                     "callsign_count": len(result["callsigns"])},
            result_json_path=result_path, created_at=capture.started_at,
            started_at=capture.started_at, completed_at=datetime.now(timezone.utc).isoformat(),
        )
        catalog.register_artifact(result_path, "fldigi_json", job_id=job_id)
        if retain_audio:
            catalog.register_artifact(wav_path, "fldigi_audio", job_id=job_id)
        else:
            wav_path.unlink(missing_ok=True)
        if retain_iq:
            catalog.register_artifact(capture.path, "iq_capture", job_id=job_id)
        return result
    finally:
        if not retain_iq:
            Path(capture.path).unlink(missing_ok=True)
