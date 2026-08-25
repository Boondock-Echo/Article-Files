from __future__ import annotations

import numpy as np
import queue
import threading

from rf_mcp.live_audio import LiveAudioConfig, LiveAudioManager, StreamingDemodulator, complex_iq


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
