from __future__ import annotations

import numpy as np
import io
import queue
import threading
import time

from rf_mcp.live_audio import (LiveAudioConfig, LiveAudioManager, LiveAudioState,
                               OpusEncoder, StreamingDemodulator, _Session, complex_iq)


def _settings(mode="nfm"):
    # Avoid hardware-backed validation in focused DSP tests.
    return LiveAudioConfig(100_000_000, mode, 12_500)


def test_interleaved_float_iq_conversion():
    converted = complex_iq(np.array([1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(converted, np.array([1 + 2j, 3 + 4j]))


def test_fm_discriminator_and_resampler_are_continuous(monkeypatch):
    monkeypatch.setattr(LiveAudioConfig, "validated", lambda self: self)
    rate = 192_000
    phase = 2 * np.pi * 1_000 * np.arange(rate // 5) / rate
    iq = np.exp(1j * phase).astype(np.complex64)
    continuous = StreamingDemodulator(_settings(), rate)
    split = StreamingDemodulator(_settings(), rate)
    reference = b"".join(continuous.process(iq))
    pieces = []
    start = 0
    for boundary in (137, 1403, 9127, len(iq)):
        pieces.extend(split.process(iq[start:boundary]))
        start = boundary
    assert b"".join(pieces) == reference


def test_broadcast_fm_deemphasis_and_agc_are_continuous(monkeypatch):
    monkeypatch.setattr(LiveAudioConfig, "validated", lambda self: self)
    rate = 192_000
    phase = 2*np.pi*5_000*np.arange(40_000)/rate
    iq = np.exp(1j*(phase + .2*np.sin(2*np.pi*700*np.arange(40_000)/rate))).astype(np.complex64)
    whole = StreamingDemodulator(_settings("broadcast_fm"), rate)
    split = StreamingDemodulator(_settings("broadcast_fm"), rate)
    expected = b"".join(whole.process(iq))
    actual = b"".join(sum((split.process(part) for part in np.array_split(iq, 17)), []))
    assert actual == expected


def test_opus_encoder_yields_complete_pages_without_read_ahead():
    page = b"OggS" + bytes(22) + b"\x02" + bytes((3, 2)) + b"abcde"
    encoder = OpusEncoder.__new__(OpusEncoder)
    encoder.process = type("Process", (), {"stdout": io.BytesIO(page + page)})()
    assert list(encoder.chunks()) == [page, page]


def test_pcm_frames_are_fixed_48khz_mono_int16(monkeypatch):
    monkeypatch.setattr(LiveAudioConfig, "validated", lambda self: self)
    dsp = StreamingDemodulator(_settings("am"), 192_000)
    frames = dsp.process(np.ones(20_000, dtype=np.complex64))
    assert frames
    assert all(len(frame) == 960 * 2 for frame in frames)
    values = np.frombuffer(b"".join(frames), dtype="<i2")
    assert values.min() >= -32768 and values.max() <= 32767


def test_different_demodulators_can_share_compatible_iq(monkeypatch):
    monkeypatch.setattr(LiveAudioConfig, "validated", lambda self: self)
    monkeypatch.setattr("rf_mcp.receiver_backend.resolve_receiver",
                        lambda _id: ({"receiver_id": "r1"}, None))

    class IQSubscription:
        def __init__(self): self.chunks, self.closed = queue.Queue(), threading.Event()
        @property
        def error(self): return None
        def close(self): self.closed.set()

    class IQManager:
        def __init__(self): self.subscriptions = []
        def subscribe(self, *args):
            subscription = IQSubscription(); self.subscriptions.append((args, subscription))
            return subscription
        def shutdown(self): pass

    class Encoder:
        @staticmethod
        def available(): return True
        def chunks(self): return iter(())
        def write(self, _pcm): pass
        def close(self): pass

    iq = IQManager(); manager = LiveAudioManager(Encoder, iq)
    am = manager.subscribe(_settings("am"))
    fm = manager.subscribe(_settings("nfm"))
    assert am.session_id != fm.session_id
    assert [call[0] for call, _ in iq.subscriptions] == [100_000_000, 100_000_000]
    am.close(); fm.close()
    assert all(subscription.closed.wait(1) for _, subscription in iq.subscriptions)


def test_public_startup_and_stop_metrics_are_ordered():
    session = _Session("s", _settings(), "r1", LiveAudioState.COMPLETED, "now",
                       first_iq_monotonic=10, first_pcm_monotonic=10.1,
                       first_encoded_chunk_monotonic=10.2,
                       stop_requested_monotonic=11, stopped_monotonic=11.025,
                       queue_drops=3)
    public = session.public()
    assert public["first_iq_monotonic"] < public["first_pcm_monotonic"] < public["first_encoded_chunk_monotonic"]
    assert public["queue_drops"] == 3
    assert public["time_to_stop_ms"] == 25


def test_slow_encoder_is_bounded_emits_first_output_and_cleans_up(monkeypatch):
    monkeypatch.setattr(LiveAudioConfig, "validated", lambda self: self)
    monkeypatch.setattr("rf_mcp.receiver_backend.resolve_receiver",
                        lambda _id: ({"receiver_id": "slow"}, None))
    monkeypatch.setattr("rf_mcp.receiver_backend.SAMPLE_RATE", 192_000)
    monkeypatch.setattr("rf_mcp.config.LIVE_AUDIO_PCM_QUEUE_FRAMES", 2)

    class IQSubscription:
        def __init__(self): self.chunks, self.closed = queue.Queue(), threading.Event()
        error = None
        def close(self): self.closed.set()
    class IQManager:
        def __init__(self): self.subscription = IQSubscription()
        def subscribe(self, *_args): return self.subscription
        def shutdown(self): pass
    class SlowEncoder:
        instances = []
        @staticmethod
        def available(): return True
        def __init__(self):
            self.output = queue.Queue(); self.closed = False; self.__class__.instances.append(self)
        def write(self, _pcm):
            time.sleep(.03)
            if self.output.empty(): self.output.put(b"OggS-first")
        def chunks(self):
            while (chunk := self.output.get()) is not None: yield chunk
        def close(self): self.closed = True; self.output.put(None)

    iq = IQManager(); manager = LiveAudioManager(SlowEncoder, iq)
    subscription = manager.subscribe(_settings("am"))
    samples = np.ones(20_000, dtype=np.complex64)
    for _ in range(12): iq.subscription.chunks.put(samples)
    assert subscription.frames.get(timeout=2) == b"OggS-first"
    iq.subscription.chunks.put(None)
    assert iq.subscription.closed.wait(2)
    deadline = time.monotonic() + 2
    while manager.status()["sessions"][0]["state"] not in ("completed", "failed") and time.monotonic() < deadline:
        time.sleep(.01)
    status = manager.status()["sessions"][0]
    assert status["discarded_pcm_frames"] > 0
    assert SlowEncoder.instances[0].closed
    assert not any(t.name.startswith("live-audio-encoder-") and t.is_alive()
                   for t in threading.enumerate())
    subscription.close()
