from __future__ import annotations

import threading
import time
import os

import numpy as np
import pytest

from rf_mcp.live_iq import LiveIQManager
from rf_mcp import config
from rf_mcp.subprocess_stream import read_chunks


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


def test_live_iq_chunk_config_bounds_and_legacy_environment(monkeypatch):
    monkeypatch.delenv("RF_MCP_LIVE_IQ_CHUNK_SECONDS", raising=False)
    monkeypatch.setenv("RF_MCP_LIVE_CHUNK_SECONDS", "0.25")
    assert config._bounded_float_env(
        "RF_MCP_LIVE_IQ_CHUNK_SECONDS", "RF_MCP_LIVE_CHUNK_SECONDS", "0.10", 0.1, 2.0,
    ) == 0.25
    monkeypatch.setenv("RF_MCP_LIVE_IQ_CHUNK_SECONDS", "2.1")
    with pytest.raises(config.ConfigurationError, match="RF_MCP_LIVE_IQ_CHUNK_SECONDS.*between"):
        config._bounded_float_env(
            "RF_MCP_LIVE_IQ_CHUNK_SECONDS", "RF_MCP_LIVE_CHUNK_SECONDS", "0.10", 0.1, 2.0,
        )


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
    session = manager._sessions[subscription.session_id]
    assert session.dropped_chunks > 0
    assert (session.subscribed_monotonic <= session.producer_started_monotonic <=
            session.receiver_stream_created_monotonic <= session.first_chunk_monotonic <=
            session.latest_chunk_monotonic <= session.shutdown_completed_monotonic)


def test_configured_chunk_duration_reaches_selected_backend(monkeypatch):
    started, released, calls = _backend(monkeypatch)
    monkeypatch.setattr(config, "LIVE_IQ_CHUNK_SECONDS", 0.375)
    subscription = LiveIQManager().subscribe(10_000_000, "r1")
    assert started.wait(1)
    assert calls[0][1]["chunk_seconds"] == pytest.approx(0.375)
    subscription.close()
    assert released.wait(1)


def test_stop_interrupts_partially_filled_pipe_read_promptly():
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    stop = threading.Event()
    completed = threading.Event()

    def consume():
        list(read_chunks(stream, 1024, stop, time.monotonic() + 10))
        completed.set()

    thread = threading.Thread(target=consume)
    thread.start()
    os.write(write_fd, b"partial")
    time.sleep(0.02)
    stopped_at = time.monotonic()
    stop.set()
    assert completed.wait(0.2)
    assert time.monotonic() - stopped_at < 0.15
    thread.join()
    os.close(write_fd)
    stream.close()


def test_synthetic_chunk_schedule_latency_100ms_vs_500ms(monkeypatch):
    """Smaller chunks improve first output; stop stays independent of chunk size."""
    monkeypatch.setattr("rf_mcp.receiver_backend.validate_frequency", lambda *_: None)
    monkeypatch.setattr(
        "rf_mcp.receiver_backend.resolve_receiver",
        lambda receiver_id: ({"receiver_id": receiver_id or "r1"}, None),
    )

    def measure(chunk_seconds):
        monkeypatch.setattr(config, "LIVE_IQ_CHUNK_SECONDS", chunk_seconds)

        def stream(_frequency, **kwargs):
            stop = kwargs["stop_event"]
            while not stop.wait(kwargs["chunk_seconds"]):
                yield np.zeros(2, dtype=np.float32)

        monkeypatch.setattr("rf_mcp.receiver_backend.stream_iq_chunks", stream)
        manager = LiveIQManager()
        subscribed_at = time.monotonic()
        subscription = manager.subscribe(10_000_000, f"r{chunk_seconds}")
        subscription.chunks.get(timeout=1)
        first_latency = time.monotonic() - subscribed_at
        stopped_at = time.monotonic()
        subscription.close()
        session = manager._sessions[subscription.session_id]
        deadline = time.monotonic() + 0.3
        while session.shutdown_completed_monotonic is None and time.monotonic() < deadline:
            time.sleep(0.005)
        return first_latency, session.shutdown_completed_monotonic - stopped_at

    fast_first, fast_stop = measure(0.1)
    slow_first, slow_stop = measure(0.5)
    assert fast_first < 0.25
    assert slow_first > 0.4
    assert slow_first - fast_first > 0.25
    assert fast_stop < 0.15
    assert slow_stop < 0.15
