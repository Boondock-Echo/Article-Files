"""Bounded, artifact-free, live IQ waterfall production."""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import queue
import threading
import time
from uuid import uuid4

import numpy as np

from . import config, receiver_backend


class LiveWaterfallState(StrEnum):
    STARTING = "starting"
    STREAMING = "streaming"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


def complex_iq(samples: np.ndarray) -> np.ndarray:
    """Normalize backend complex or interleaved float32 IQ to complex64."""
    values = np.asarray(samples)
    if np.iscomplexobj(values):
        return values.astype(np.complex64, copy=False).reshape(-1)
    flat = values.astype(np.float32, copy=False).reshape(-1)
    if flat.size % 2:
        raise ValueError("interleaved IQ chunk contains an odd number of floats")
    return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)


@dataclass(frozen=True)
class LiveWaterfallConfig:
    center_frequency_hz: int
    receiver_id: str | None = None
    fft_size: int = 4096
    update_rate_hz: float = 10.0
    span_hz: int = 500_000
    minimum_power_db: float = -110.0
    maximum_power_db: float = -20.0
    display_bins: int = 512
    quantization_bits: int = 8
    maximum_duration_seconds: float = 300.0

    def validated(self) -> "LiveWaterfallConfig":
        receiver_backend.validate_frequency(self.center_frequency_hz, self.receiver_id)
        if self.fft_size < 256 or self.fft_size > 65536 or self.fft_size & (self.fft_size - 1):
            raise ValueError("fft_size must be a power of two from 256 through 65536")
        if not .5 <= self.update_rate_hz <= 30:
            raise ValueError("update_rate_hz must be from 0.5 through 30")
        sample_rate = int(receiver_backend.SAMPLE_RATE)
        if not 1_000 <= self.span_hz <= sample_rate:
            raise ValueError(f"span_hz must be from 1000 through {sample_rate}")
        if not 16 <= self.display_bins <= 2048:
            raise ValueError("display_bins must be from 16 through 2048")
        if self.quantization_bits not in (8, 16):
            raise ValueError("quantization_bits must be 8 or 16")
        if not np.isfinite(self.minimum_power_db) or not np.isfinite(self.maximum_power_db) or self.minimum_power_db >= self.maximum_power_db:
            raise ValueError("minimum_power_db must be less than maximum_power_db")
        if self.maximum_power_db - self.minimum_power_db > 200:
            raise ValueError("display dynamic range must not exceed 200 dB")
        if not 0 < self.maximum_duration_seconds <= config.LIVE_WATERFALL_MAX_DURATION_SECONDS:
            raise ValueError(f"maximum_duration_seconds must be at most {config.LIVE_WATERFALL_MAX_DURATION_SECONDS:g}")
        return LiveWaterfallConfig(self.center_frequency_hz, self.receiver_id, self.fft_size,
            float(self.update_rate_hz), self.span_hz, self.minimum_power_db,
            self.maximum_power_db, self.display_bins, self.quantization_bits,
            float(self.maximum_duration_seconds))


def make_spectral_row(samples: np.ndarray, settings: LiveWaterfallConfig,
                      window: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Return a quantized row and the exact first/last display-bin frequencies."""
    cfg = settings.validated()
    iq = complex_iq(samples)
    if iq.size != cfg.fft_size:
        raise ValueError("one FFT row requires exactly fft_size IQ samples")
    win = np.asarray(window) if window is not None else np.hanning(cfg.fft_size).astype(np.float32)
    if win.size != cfg.fft_size:
        raise ValueError("window length must equal fft_size")
    power = 20 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(iq * win))) /
                                      max(float(win.sum()), 1.0), 1e-12))
    sample_rate = float(receiver_backend.SAMPLE_RATE)
    frequencies = cfg.center_frequency_hz + np.fft.fftshift(np.fft.fftfreq(cfg.fft_size, 1 / sample_rate))
    mask = np.abs(frequencies - cfg.center_frequency_hz) <= cfg.span_hz / 2
    power, frequencies = power[mask], frequencies[mask]
    targets = np.linspace(0, max(power.size - 1, 0), min(cfg.display_bins, power.size))
    power = np.interp(targets, np.arange(power.size), power)
    frequencies = np.interp(targets, np.arange(frequencies.size), frequencies)
    maximum = (1 << cfg.quantization_bits) - 1
    dtype = np.uint8 if cfg.quantization_bits == 8 else np.dtype("<u2")
    row = np.rint(np.clip((power - cfg.minimum_power_db) /
                          (cfg.maximum_power_db - cfg.minimum_power_db), 0, 1) * maximum).astype(dtype)
    return row, float(frequencies[0]), float(frequencies[-1])


@dataclass
class _Session:
    session_id: str
    config: LiveWaterfallConfig
    receiver_id: str
    state: LiveWaterfallState
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    client_count: int = 0
    rows_produced: int = 0
    termination_reason: str | None = None
    error: str | None = None

    def public(self):
        value = asdict(self); value["state"] = self.state.value; value["config"] = asdict(self.config)
        return value


class WaterfallSubscription:
    def __init__(self, manager, session_id, rows):
        self.manager, self.session_id, self.rows = manager, session_id, rows
    def close(self): self.manager.unsubscribe(self.session_id, self.rows)


class LiveWaterfallManager:
    def __init__(self):
        self._lock = threading.RLock(); self._sessions = {}; self._listeners = {}; self._stops = {}
        self._rows = {}; self._history = deque(maxlen=config.LIVE_AUDIO_HISTORY_SIZE)

    def subscribe(self, requested: LiveWaterfallConfig) -> WaterfallSubscription:
        cfg = requested.validated(); receiver, _ = receiver_backend.resolve_receiver(cfg.receiver_id); rid = receiver["receiver_id"]
        with self._lock:
            active = next((s for s in self._sessions.values() if s.receiver_id == rid and s.state in
                           (LiveWaterfallState.STARTING, LiveWaterfallState.STREAMING)), None)
            if active and active.config != cfg: raise RuntimeError("receiver busy with an incompatible live waterfall")
            session = active
            if session is None:
                session = _Session(uuid4().hex, cfg, rid, LiveWaterfallState.STARTING, _now())
                self._sessions[session.session_id] = session; self._listeners[session.session_id] = []
                self._rows[session.session_id] = deque(maxlen=config.LIVE_WATERFALL_HISTORY_ROWS)
                stop = threading.Event(); self._stops[session.session_id] = stop
                threading.Thread(target=self._produce, args=(session, stop), daemon=True,
                                 name=f"live-waterfall-{rid}").start()
            if session.client_count >= config.LIVE_WATERFALL_MAX_CLIENTS: raise RuntimeError("live waterfall client limit reached")
            rows = queue.Queue(maxsize=config.LIVE_WATERFALL_QUEUE_ROWS)
            for frame in self._rows[session.session_id]:
                if rows.full(): rows.get_nowait()
                rows.put_nowait(frame)
            self._listeners[session.session_id].append(rows); session.client_count += 1
            return WaterfallSubscription(self, session.session_id, rows)

    def _broadcast(self, session, frame):
        with self._lock:
            if frame is not None: self._rows[session.session_id].append(frame)
            for listener in self._listeners.get(session.session_id, []):
                if listener.full():
                    try: listener.get_nowait()
                    except queue.Empty: pass
                try: listener.put_nowait(frame)
                except queue.Full: pass

    def _produce(self, session, stop):
        generator = None
        try:
            cfg = session.config; window = np.hanning(cfg.fft_size).astype(np.float32)
            pending = np.empty(0, np.complex64); next_row = time.monotonic()
            session.state = LiveWaterfallState.STREAMING; session.started_at = _now()
            generator = receiver_backend.stream_iq_chunks(cfg.center_frequency_hz,
                duration_seconds=cfg.maximum_duration_seconds, stop_event=stop,
                receiver_id=session.receiver_id, lease_owner=f"waterfall-{session.session_id}", purpose="live waterfall")
            for chunk in generator:
                if stop.is_set(): break
                pending = np.concatenate((pending, complex_iq(chunk)))
                while pending.size >= cfg.fft_size:
                    block, pending = pending[:cfg.fft_size], pending[cfg.fft_size:]
                    now = time.monotonic()
                    if now < next_row: continue
                    row, low, high = make_spectral_row(block, cfg, window)
                    frame = {"session_id": session.session_id, "sequence": session.rows_produced,
                             "timestamp": _now(), "frequency_start_hz": low, "frequency_end_hz": high,
                             "bin_count": int(row.size), "bits": cfg.quantization_bits,
                             "encoding": "base64", "row": base64.b64encode(row.tobytes()).decode("ascii")}
                    session.rows_produced += 1; self._broadcast(session, frame)
                    next_row = now + 1 / cfg.update_rate_hz
            session.termination_reason = "stopped" if stop.is_set() else "duration_limit"
            session.state = LiveWaterfallState.COMPLETED
        except Exception as exc:
            session.state = LiveWaterfallState.FAILED; session.termination_reason = "error"
            session.error = f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            stop.set()
            if generator is not None:
                try: generator.close()
                except Exception: pass
            session.ended_at = _now(); self._broadcast(session, None)
            with self._lock: self._history.append(session.public())

    def unsubscribe(self, session_id, rows):
        with self._lock:
            listeners = self._listeners.get(session_id, [])
            if rows in listeners: listeners.remove(rows)
            session = self._sessions.get(session_id)
            if session: session.client_count = len(listeners)
            if session and not listeners and session.state in (LiveWaterfallState.STARTING, LiveWaterfallState.STREAMING):
                session.state = LiveWaterfallState.STOPPING; session.termination_reason = "last_listener_disconnected"
                self._stops[session_id].set()

    def status(self):
        with self._lock: return {"sessions": [s.public() for s in self._sessions.values()], "history": list(self._history)}

    def stop(self, session_id=None):
        with self._lock:
            targets = [session_id] if session_id else list(self._sessions); found = False
            for sid in targets:
                if sid in self._stops:
                    found = True; self._stops[sid].set(); session = self._sessions[sid]
                    if session.state in (LiveWaterfallState.STARTING, LiveWaterfallState.STREAMING): session.state = LiveWaterfallState.STOPPING
            return {"stopped": found, "session_id": session_id}

    def shutdown(self): self.stop()


def _now(): return datetime.now(timezone.utc).isoformat()
