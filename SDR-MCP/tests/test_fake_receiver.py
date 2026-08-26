from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rf_mcp.fake_receiver import FakeStreamProfile, FakeStreamingReceiverBackend


RECEIVER = {"receiver_id": "fake-ci"}


def test_fake_receiver_tone_short_reads_and_repeatable_phase():
    backend = FakeStreamingReceiverBackend(FakeStreamProfile(
        sample_rate_hz=1000, tone_hz=100, short_read_fractions=(1, .5),
        jitter_seconds=(-.1,), disconnect_after_chunks=2))
    stream = backend.stream_iq_chunks(RECEIVER, 100_000_000, duration_seconds=2,
                                      chunk_seconds=.1, stop_event=threading.Event())
    first, second = next(stream), next(stream)
    assert (first.size, second.size) == (100, 50)
    combined = np.concatenate((first, second))
    assert np.allclose(np.angle(combined[1:] * np.conj(combined[:-1])), 2*np.pi*.1,
                       atol=1e-5)
    with pytest.raises(ConnectionError, match="scripted"):
        next(stream)
    assert backend.snapshot() == {"model": "deterministic-test-tone", "streams_opened": 1,
                                  "streams_closed": 1, "chunks_emitted": 2}


def test_fake_receiver_startup_stall_is_interruptible_and_cleans_up():
    backend = FakeStreamingReceiverBackend(FakeStreamProfile(startup_delay_seconds=5))
    stop = threading.Event()
    stream = backend.stream_iq_chunks(RECEIVER, 1, duration_seconds=10,
                                      stop_event=stop)
    result = []
    thread = threading.Thread(target=lambda: result.extend(stream))
    started = time.monotonic(); thread.start(); stop.set(); thread.join(.5)
    assert not thread.is_alive()
    assert time.monotonic() - started < .5
    assert result == []
    assert backend.streams_opened == backend.streams_closed == 1
