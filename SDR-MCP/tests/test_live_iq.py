from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rf_mcp.live_iq import LiveIQManager


def _backend(monkeypatch):
    started = threading.Event()
    released = threading.Event()
    calls = []

    class Stream:
        def __init__(self, stop): self.stop = stop
        def __iter__(self):
            started.set()
            while not self.stop.wait(.005):
                yield np.array([len(calls)], dtype=np.complex64)
        def close(self): released.set()

    monkeypatch.setattr("rf_mcp.receiver_backend.validate_frequency", lambda *_: None)
    monkeypatch.setattr("rf_mcp.receiver_backend.resolve_receiver",
                        lambda receiver_id: ({"receiver_id": receiver_id or "r1"}, None))

    def stream(frequency, **kwargs):
        calls.append((frequency, kwargs))
        return Stream(kwargs["stop_event"])

    monkeypatch.setattr("rf_mcp.receiver_backend.stream_iq_chunks", stream)
    return started, released, calls


def test_compatible_consumers_share_one_hardware_stream(monkeypatch):
    started, released, calls = _backend(monkeypatch)
    manager = LiveIQManager(queue_chunks=2)
    first = manager.subscribe(10_000_000, "r1")
    assert started.wait(1)
    second = manager.subscribe(10_000_000, "r1")
    assert first.session_id == second.session_id
    first.chunks.get(timeout=1)
    second.chunks.get(timeout=1)
    assert len(calls) == 1
    first.close()
    assert not released.wait(.05)
    second.close()
    assert released.wait(1)


def test_incompatible_retune_is_rejected_as_receiver_busy(monkeypatch):
    started, released, _ = _backend(monkeypatch)
    manager = LiveIQManager()
    subscription = manager.subscribe(10_000_000, "r1")
    assert started.wait(1)
    with pytest.raises(RuntimeError, match="receiver busy"):
        manager.subscribe(11_000_000, "r1")
    subscription.close()
    assert released.wait(1)


def test_slow_consumer_evicts_old_chunks_without_blocking_producer(monkeypatch):
    started, released, calls = _backend(monkeypatch)
    manager = LiveIQManager(queue_chunks=2)
    subscription = manager.subscribe(10_000_000)
    assert started.wait(1)
    time.sleep(.05)
    assert subscription.chunks.qsize() == 2
    assert len(calls) == 1
    subscription.close()
    assert released.wait(1)
