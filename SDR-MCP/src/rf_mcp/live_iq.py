"""Shared, bounded in-memory IQ streams for live consumers.

One hardware producer is maintained for each receiver and tuning.  Subscribers
have independent live-edge queues, so a slow DSP consumer can never apply
back-pressure to the receiver.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import queue
import threading
import time
from typing import Hashable
from uuid import uuid4

from . import config, receiver_backend


@dataclass
class _IQSession:
    session_id: str
    receiver_id: str
    tuning: tuple[Hashable, ...]
    frequency_hz: int
    stop: threading.Event
    listeners: list[queue.Queue]
    created_at: str
    subscribed_monotonic: float
    producer_started_monotonic: float | None = None
    receiver_stream_created_monotonic: float | None = None
    first_chunk_monotonic: float | None = None
    latest_chunk_monotonic: float | None = None
    dropped_chunks: int = 0
    shutdown_completed_monotonic: float | None = None
    state: str = "starting"
    error: str | None = None


class IQSubscription:
    def __init__(self, manager: "LiveIQManager", session_id: str, chunks: queue.Queue):
        self.manager, self.session_id, self.chunks = manager, session_id, chunks
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.manager.unsubscribe(self.session_id, self.chunks)

    @property
    def error(self) -> str | None:
        return self.manager.error(self.session_id)


class LiveIQManager:
    """Own receiver leases and fan IQ chunks out to compatible consumers."""

    def __init__(self, queue_chunks: int | None = None):
        self.queue_chunks = queue_chunks or config.LIVE_AUDIO_OUTPUT_QUEUE_CHUNKS
        self._lock = threading.RLock()
        self._sessions: dict[str, _IQSession] = {}

    def subscribe(self, frequency_hz: int, receiver_id: str | None = None,
                  *, tuning: tuple[Hashable, ...] = ()) -> IQSubscription:
        receiver_backend.validate_frequency(frequency_hz, receiver_id)
        receiver, _ = receiver_backend.resolve_receiver(receiver_id)
        rid = receiver["receiver_id"]
        hardware_tuning = (int(frequency_hz), int(receiver_backend.SAMPLE_RATE), *tuning)
        with self._lock:
            active = next((item for item in self._sessions.values()
                           if item.receiver_id == rid and item.state in
                           ("starting", "streaming", "stopping")), None)
            if active is not None and active.state == "stopping":
                raise RuntimeError("receiver busy stopping the previous live session")
            if active is not None and active.tuning != hardware_tuning:
                raise RuntimeError("receiver busy with an incompatible live session")
            session = active
            chunks: queue.Queue = queue.Queue(maxsize=self.queue_chunks)
            if session is None:
                session = _IQSession(uuid4().hex, rid, hardware_tuning, int(frequency_hz),
                                     threading.Event(), [chunks], _now(), time.monotonic())
                self._sessions[session.session_id] = session
                threading.Thread(target=self._produce, args=(session,), daemon=True,
                                 name=f"live-iq-{rid}").start()
            else:
                session.listeners.append(chunks)
            return IQSubscription(self, session.session_id, chunks)

    def _publish(self, session: _IQSession, chunk) -> None:
        with self._lock:
            for listener in tuple(session.listeners):
                if listener.full():
                    try:
                        listener.get_nowait()
                        if chunk is not None: session.dropped_chunks += 1
                    except queue.Empty: pass
                try: listener.put_nowait(chunk)
                except queue.Full:
                    if chunk is not None: session.dropped_chunks += 1

    def _produce(self, session: _IQSession) -> None:
        generator = None
        final_state = "completed"
        try:
            session.producer_started_monotonic = time.monotonic()
            session.state = "streaming"
            generator = receiver_backend.stream_iq_chunks(
                session.frequency_hz,
                duration_seconds=max(config.LIVE_AUDIO_MAX_DURATION_SECONDS,
                                     config.LIVE_WATERFALL_MAX_DURATION_SECONDS),
                chunk_seconds=config.LIVE_IQ_CHUNK_SECONDS,
                stop_event=session.stop, receiver_id=session.receiver_id,
                lease_owner=f"live-iq-{session.session_id}", purpose="shared live IQ")
            session.receiver_stream_created_monotonic = time.monotonic()
            for chunk in generator:
                if session.stop.is_set(): break
                now = time.monotonic()
                if session.first_chunk_monotonic is None: session.first_chunk_monotonic = now
                session.latest_chunk_monotonic = now
                self._publish(session, chunk)
        except Exception as exc:
            final_state = "failed"
            session.error = f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            session.stop.set()
            if generator is not None:
                try: generator.close()
                except Exception: pass
            session.state = final_state
            session.shutdown_completed_monotonic = time.monotonic()
            self._publish(session, None)

    def unsubscribe(self, session_id: str, chunks: queue.Queue) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None: return
            if chunks in session.listeners: session.listeners.remove(chunks)
            if not session.listeners and session.state in ("starting", "streaming"):
                session.state = "stopping"
                session.stop.set()

    def shutdown(self) -> None:
        with self._lock:
            for session in self._sessions.values(): session.stop.set()

    def error(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.error if session is not None else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
