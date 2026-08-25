"""Bounded, artifact-free live receiver audio.

The live path intentionally never calls the capture/recording/catalog APIs.  A
slow listener loses its oldest encoded chunks (live-edge policy) instead of
applying back-pressure to the receiver or growing memory without bound.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import math
import queue
import shutil
import subprocess
import threading
import time
from typing import Iterator
from uuid import uuid4

import numpy as np
from scipy.signal import butter, sosfilt

from . import config, receiver_backend
from .live_iq import LiveIQManager
from .signal_analysis import normalize_mode, validate_bandwidth

LIVE_MODES = ("broadcast_fm", "am", "nfm", "usb", "lsb", "cw")


class LiveAudioState(StrEnum):
    STARTING = "starting"
    STREAMING = "streaming"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LiveAudioConfig:
    frequency_hz: int
    mode: str
    bandwidth_hz: int
    receiver_id: str | None = None
    deemphasis_us: int = 75
    output_sample_rate_hz: int = 48_000
    channel_count: int = 1
    maximum_duration_seconds: float = 300.0

    def validated(self) -> "LiveAudioConfig":
        mode = self.mode.strip().lower()
        if mode not in LIVE_MODES:
            raise ValueError(f"unsupported live mode: {mode}")
        receiver_backend.validate_frequency(self.frequency_hz, self.receiver_id)
        bandwidth = 200_000 if mode == "broadcast_fm" else validate_bandwidth(normalize_mode(mode), self.bandwidth_hz)
        if self.deemphasis_us not in (50, 75):
            raise ValueError("deemphasis_us must be 50 or 75")
        if self.output_sample_rate_hz != 48_000 or self.channel_count != 1:
            raise ValueError("live audio currently supports 48 kHz mono only")
        if not 0 < self.maximum_duration_seconds <= config.LIVE_AUDIO_MAX_DURATION_SECONDS:
            raise ValueError(f"maximum_duration_seconds must be at most {config.LIVE_AUDIO_MAX_DURATION_SECONDS:g}")
        return LiveAudioConfig(self.frequency_hz, mode, bandwidth, self.receiver_id,
                               self.deemphasis_us, 48_000, 1,
                               float(self.maximum_duration_seconds))


def complex_iq(samples: np.ndarray) -> np.ndarray:
    """Normalize backend complex or interleaved float32 IQ to complex64."""
    values = np.asarray(samples)
    if np.iscomplexobj(values):
        return values.astype(np.complex64, copy=False).reshape(-1)
    flat = values.astype(np.float32, copy=False).reshape(-1)
    if flat.size % 2:
        raise ValueError("interleaved IQ chunk contains an odd number of floats")
    return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)


class StreamingResampler:
    """Stateful linear resampler whose phase and boundary sample survive chunks."""
    def __init__(self, input_rate: int, output_rate: int):
        self.step = float(input_rate) / output_rate
        self.position = 0.0
        self.previous: float | None = None

    def process(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.previous is not None:
            x = np.concatenate(([self.previous], x))
        if x.size < 2:
            if x.size: self.previous = float(x[-1])
            return np.empty(0)
        positions = np.arange(self.position, x.size - 1, self.step)
        out = np.interp(positions, np.arange(x.size), x)
        next_position = self.position + len(positions) * self.step
        self.position = next_position - (x.size - 1)
        self.previous = float(x[-1])
        return out


class StreamingDemodulator:
    """Incremental mono demodulator with oscillator, filters and conditioning state."""
    def __init__(self, settings: LiveAudioConfig, input_rate_hz: int, offset_hz: float = 0):
        self.settings = settings.validated()
        self.input_rate = int(input_rate_hz)
        cutoff = 100_000 if settings.mode == "broadcast_fm" else settings.bandwidth_hz / 2
        cutoff = min(cutoff, self.input_rate * .45)
        self.sos = butter(6, cutoff, btype="lowpass", fs=self.input_rate, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.complex128)
        self.phase = 0.0
        self.phase_step = -2 * math.pi * offset_hz / self.input_rate
        self.cw_phase = 0.0
        self.previous_complex: complex | None = None
        intermediate_rate = 192_000 if settings.mode == "broadcast_fm" else 48_000
        self.resampler = StreamingResampler(self.input_rate, intermediate_rate)
        self.output_resampler = StreamingResampler(intermediate_rate, 48_000)
        self.audio_sos = butter(6, min(15_000, intermediate_rate * .4), btype="lowpass", fs=intermediate_rate, output="sos")
        self.audio_zi = np.zeros((self.audio_sos.shape[0], 2))
        a = math.exp(-1 / (intermediate_rate * settings.deemphasis_us * 1e-6))
        self.deemphasis_a, self.deemphasis_value = a, 0.0
        self.agc_gain = 1.0
        self.agc_envelope = 1e-3
        self.pending = np.empty(0, dtype=np.int16)

    def _demodulate(self, iq: np.ndarray) -> np.ndarray:
        mode = self.settings.mode
        if mode in ("nfm", "broadcast_fm"):
            if self.previous_complex is None:
                previous = iq[0] if iq.size else 0j
            else: previous = self.previous_complex
            shifted = np.concatenate(([previous], iq))
            audio = np.angle(shifted[1:] * np.conj(shifted[:-1]))
            if iq.size: self.previous_complex = complex(iq[-1])
            return audio
        if mode == "am": return np.abs(iq)
        if mode == "cw":
            # CW BFO phase shares the continuous oscillator state.
            step = 2 * np.pi * 700 / self.input_rate
            mixed = np.real(iq * np.exp(1j * (self.cw_phase + step * np.arange(iq.size))))
            self.cw_phase = float((self.cw_phase + step * iq.size) % (2 * np.pi))
            return mixed
        return np.real(iq)  # IQ has already selected the requested USB/LSB passband.

    def process(self, samples: np.ndarray) -> list[bytes]:
        iq = complex_iq(samples)
        if not iq.size: return []
        indices = np.arange(iq.size)
        iq *= np.exp(1j * (self.phase + self.phase_step * indices))
        self.phase = float((self.phase + self.phase_step * iq.size) % (2 * math.pi))
        filtered, self.zi = sosfilt(self.sos, iq, zi=self.zi)
        audio = self.resampler.process(self._demodulate(filtered))
        audio, self.audio_zi = sosfilt(self.audio_sos, audio, zi=self.audio_zi)
        if self.settings.mode == "broadcast_fm":
            out = np.empty_like(audio)
            for i, value in enumerate(audio):
                self.deemphasis_value = (1-self.deemphasis_a)*value + self.deemphasis_a*self.deemphasis_value
                out[i] = self.deemphasis_value
            audio = self.output_resampler.process(out)
        # Slow RMS AGC and a hard soft-knee limiter replace per-recording peak scaling.
        conditioned = np.empty_like(audio)
        for index, value in enumerate(audio):
            # Sample-wise attack/release makes gain independent of backend chunking.
            coefficient = .999 if abs(value) > self.agc_envelope else .99995
            self.agc_envelope = coefficient*self.agc_envelope + (1-coefficient)*abs(value)
            target = min(50.0, .20 / max(self.agc_envelope, 1e-6))
            self.agc_gain = .9999*self.agc_gain + .0001*target
            conditioned[index] = value*self.agc_gain
        pcm = np.round(np.tanh(conditioned * 1.5) * 32767).astype("<i2")
        self.pending = np.concatenate((self.pending, pcm))
        frame_samples, frames = 960, []  # 20 ms Opus frames at 48 kHz
        while self.pending.size >= frame_samples:
            frames.append(self.pending[:frame_samples].tobytes())
            self.pending = self.pending[frame_samples:]
        return frames


class OpusEncoder:
    """FFmpeg PCM-to-Ogg/Opus adapter; stderr is continuously drained."""
    @staticmethod
    def available() -> bool: return shutil.which(config.LIVE_AUDIO_FFMPEG) is not None

    def __init__(self):
        self.process = subprocess.Popen([config.LIVE_AUDIO_FFMPEG, "-hide_banner", "-loglevel", "warning",
            "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
            "-application", "lowdelay", "-f", "ogg", "pipe:1"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.stderr = deque(maxlen=16)
        threading.Thread(target=self._drain_error, daemon=True).start()

    def _drain_error(self):
        assert self.process.stderr
        for line in iter(self.process.stderr.readline, b""): self.stderr.append(line.decode("utf-8", "replace").strip())

    def write(self, pcm: bytes):
        assert self.process.stdin
        self.process.stdin.write(pcm)

    def chunks(self) -> Iterator[bytes]:
        assert self.process.stdout
        while chunk := self.process.stdout.read(4096): yield chunk

    def close(self):
        if self.process.stdin and not self.process.stdin.closed: self.process.stdin.close()
        try: self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired: self.process.kill()


@dataclass
class _Session:
    session_id: str
    config: LiveAudioConfig
    receiver_id: str
    state: LiveAudioState
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    client_count: int = 0
    termination_reason: str | None = None
    error: str | None = None
    def public(self):
        result = asdict(self); result["state"] = self.state.value; result["config"] = asdict(self.config)
        return result


class LiveSubscription:
    def __init__(self, manager, session_id, frames): self.manager, self.session_id, self.frames = manager, session_id, frames
    def close(self): self.manager.unsubscribe(self.session_id, self.frames)


class LiveAudioManager:
    def __init__(self, encoder_factory=OpusEncoder, iq_manager: LiveIQManager | None = None):
        self.encoder_factory, self._lock = encoder_factory, threading.RLock()
        self.iq_manager = iq_manager or LiveIQManager()
        self._sessions, self._listeners = {}, {}
        self._stops, self._history = {}, deque(maxlen=config.LIVE_AUDIO_HISTORY_SIZE)

    def capabilities(self):
        return {"available": self.encoder_factory.available(), "iq_streaming": True,
                "modes": list(LIVE_MODES), "codecs": ["ogg/opus"], "sample_rate_hz": 48000,
                "channels": 1, "broadcast_fm": "mono; stereo requires a continuous pilot PLL",
                "maximum_duration_seconds": config.LIVE_AUDIO_MAX_DURATION_SECONDS,
                "artifacts_created": False}

    def subscribe(self, requested: LiveAudioConfig) -> LiveSubscription:
        if not self.encoder_factory.available(): raise RuntimeError("live audio encoder unavailable")
        cfg = requested.validated(); receiver, _ = receiver_backend.resolve_receiver(cfg.receiver_id)
        rid = receiver["receiver_id"]
        with self._lock:
            active = next((s for s in self._sessions.values() if s.receiver_id == rid and
                           s.config == cfg and s.state in
                           (LiveAudioState.STARTING, LiveAudioState.STREAMING)), None)
            session = active
            if session is None:
                iq_subscription = self.iq_manager.subscribe(cfg.frequency_hz, rid)
                session = _Session(uuid4().hex, cfg, rid, LiveAudioState.STARTING, _now())
                self._sessions[session.session_id] = session; self._listeners[session.session_id] = []
                stop = threading.Event(); self._stops[session.session_id] = stop
                threading.Thread(target=self._produce, args=(session, stop, iq_subscription), daemon=True,
                                 name=f"live-audio-{rid}").start()
            if session.client_count >= config.LIVE_AUDIO_MAX_CLIENTS: raise RuntimeError("live audio client limit reached")
            frames = queue.Queue(maxsize=config.LIVE_AUDIO_OUTPUT_QUEUE_CHUNKS)
            self._listeners[session.session_id].append(frames); session.client_count += 1
            return LiveSubscription(self, session.session_id, frames)

    def _broadcast(self, session, chunk):
        with self._lock:
            for listener in self._listeners.get(session.session_id, []):
                if listener.full():
                    try: listener.get_nowait()
                    except queue.Empty: pass
                try: listener.put_nowait(chunk)
                except queue.Full: pass

    def _produce(self, session, stop, iq_subscription):
        encoder = None
        try:
            encoder = self.encoder_factory(); sample_rate = receiver_backend.SAMPLE_RATE
            dsp = StreamingDemodulator(session.config, sample_rate)
            threading.Thread(target=lambda: [self._broadcast(session, c) for c in encoder.chunks()], daemon=True).start()
            session.state, session.started_at = LiveAudioState.STREAMING, _now()
            deadline = time.monotonic() + session.config.maximum_duration_seconds
            while not stop.is_set() and time.monotonic() < deadline:
                try: iq = iq_subscription.chunks.get(timeout=min(.1, max(0, deadline-time.monotonic())))
                except queue.Empty: continue
                if iq is None:
                    if iq_subscription.error: raise RuntimeError(iq_subscription.error)
                    break
                for pcm in dsp.process(iq): encoder.write(pcm)
            session.termination_reason = "stopped" if stop.is_set() else "duration_limit"
            session.state = LiveAudioState.COMPLETED
        except Exception as exc:
            session.state, session.termination_reason = LiveAudioState.FAILED, "error"
            session.error = f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            stop.set()
            if iq_subscription is not None: iq_subscription.close()
            if encoder is not None: encoder.close()
            session.ended_at = _now(); self._broadcast(session, None)
            with self._lock: self._history.append(session.public())

    def unsubscribe(self, session_id, frames):
        with self._lock:
            listeners = self._listeners.get(session_id, [])
            if frames in listeners: listeners.remove(frames)
            session = self._sessions.get(session_id)
            if session: session.client_count = len(listeners)
            if session and not listeners and session.state in (LiveAudioState.STARTING, LiveAudioState.STREAMING):
                session.state = LiveAudioState.STOPPING; session.termination_reason = "stopped"
                self._stops[session_id].set()

    def status(self):
        with self._lock: return {"sessions": [s.public() for s in self._sessions.values()], "history": list(self._history)}

    def stop(self, session_id: str | None = None):
        with self._lock:
            targets = [session_id] if session_id else list(self._sessions)
            found = False
            for sid in targets:
                if sid in self._stops:
                    found = True; self._stops[sid].set()
                    s = self._sessions[sid]
                    if s.state in (LiveAudioState.STARTING, LiveAudioState.STREAMING): s.state = LiveAudioState.STOPPING
            return {"stopped": found, "session_id": session_id}

    def shutdown(self):
        self.stop()
        self.iq_manager.shutdown()


def _now(): return datetime.now(timezone.utc).isoformat()
