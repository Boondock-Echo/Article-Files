"""Deterministic streaming receiver used by integration and browser tests.

The fake is deliberately an ordinary :class:`ReceiverBackend`: exercising it
therefore covers lease ownership and cleanup as well as the DSP and ASGI paths.
It never probes or opens SDR hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock
import time
from typing import Iterator
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np

from .airspyhf import Capture


@dataclass
class FakeStreamProfile:
    sample_rate_hz: int = 768_000
    tone_hz: float = 12_000.0
    amplitude: float = 0.7
    startup_delay_seconds: float = 0.0
    jitter_seconds: tuple[float, ...] = ()
    short_read_fractions: tuple[float, ...] = (1.0,)
    stall_after_chunks: int | None = None
    stall_seconds: float = 0.0
    disconnect_after_chunks: int | None = None


@dataclass
class FakeStreamingReceiverBackend:
    """Scriptable IQ source with repeatable phase and interruptible waits."""

    profile: FakeStreamProfile = field(default_factory=FakeStreamProfile)
    name: str = "fake"
    streams_opened: int = 0
    streams_closed: int = 0
    chunks_emitted: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def device_info(self, receiver: dict) -> dict:
        return {"receiver_id": receiver["receiver_id"], "backend": self.name,
                "model": "deterministic-test-tone", "sample_rate_hz": self.profile.sample_rate_hz,
                "hardware": False}

    def capture_iq(self, receiver: dict, center_frequency_hz: int,
                   duration_seconds: float, **options) -> Capture:
        stop = Event()
        chunks = list(self.stream_iq_chunks(receiver, center_frequency_hz,
                      duration_seconds=duration_seconds, stop_event=stop, **options))
        samples = np.concatenate(chunks) if chunks else np.empty(0, np.complex64)
        if options.get("output_path"):
            path = Path(options["output_path"])
        else:
            temporary = tempfile.NamedTemporaryFile(suffix=".cf32", delete=False)
            path = Path(temporary.name)
            temporary.close()
        samples.tofile(path)
        return Capture(path=path, sample_rate_hz=self.profile.sample_rate_hz,
                       center_frequency_hz=center_frequency_hz,
                       requested_samples=round(duration_seconds*self.profile.sample_rate_hz),
                       captured_samples=int(samples.size),
                       started_at=datetime.now(timezone.utc).isoformat(),
                       receiver_id=receiver.get("receiver_id"), backend=self.name)

    @staticmethod
    def _wait(stop_event: Event, seconds: float) -> bool:
        return seconds > 0 and stop_event.wait(seconds)

    def stream_iq_chunks(self, receiver: dict, center_frequency_hz: int, *,
                         duration_seconds: float, stop_event: Event,
                         chunk_seconds: float = 0.1, **_options) -> Iterator[np.ndarray]:
        profile = self.profile
        with self._lock:
            self.streams_opened += 1
        emitted = 0
        sample_index = 0
        deadline = time.monotonic() + duration_seconds
        try:
            if self._wait(stop_event, profile.startup_delay_seconds):
                return
            while not stop_event.is_set() and time.monotonic() < deadline:
                if profile.disconnect_after_chunks is not None and emitted >= profile.disconnect_after_chunks:
                    raise ConnectionError("scripted fake receiver disconnect")
                if profile.stall_after_chunks is not None and emitted == profile.stall_after_chunks:
                    if self._wait(stop_event, profile.stall_seconds):
                        return
                fraction = profile.short_read_fractions[emitted % len(profile.short_read_fractions)]
                count = max(1, round(profile.sample_rate_hz * chunk_seconds * fraction))
                indexes = np.arange(sample_index, sample_index + count, dtype=np.float64)
                phase = 2 * np.pi * profile.tone_hz * indexes / profile.sample_rate_hz
                yield (profile.amplitude * np.exp(1j * phase)).astype(np.complex64)
                sample_index += count
                emitted += 1
                with self._lock:
                    self.chunks_emitted += 1
                jitter = profile.jitter_seconds[(emitted - 1) % len(profile.jitter_seconds)] if profile.jitter_seconds else 0
                if self._wait(stop_event, max(0.0, chunk_seconds + jitter)):
                    return
        finally:
            with self._lock:
                self.streams_closed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"model": "deterministic-test-tone", "streams_opened": self.streams_opened,
                    "streams_closed": self.streams_closed, "chunks_emitted": self.chunks_emitted}
