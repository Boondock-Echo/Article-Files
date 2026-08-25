from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from rf_mcp import config
from rf_mcp.live_waterfall import (LiveWaterfallConfig, LiveWaterfallManager,
                                   LiveWaterfallState, _Session, make_spectral_row)


def settings(**changes):
    values = dict(center_frequency_hz=10_000_000, fft_size=1024, update_rate_hz=10,
                  span_hz=500_000, display_bins=128, maximum_duration_seconds=1)
    values.update(changes)
    return LiveWaterfallConfig(**values)


def test_fft_row_is_quantized_and_bin_frequencies_are_cropped(monkeypatch):
    monkeypatch.setattr('rf_mcp.receiver_backend.validate_frequency', lambda *_: None)
    monkeypatch.setattr('rf_mcp.receiver_backend.SAMPLE_RATE', 2_000_000)
    n = np.arange(1024)
    iq = np.exp(2j * np.pi * 100_000 * n / 2_000_000).astype(np.complex64)
    row, low, high = make_spectral_row(iq, settings())
    assert row.dtype == np.uint8
    assert row.shape == (128,)
    assert low == pytest.approx(9_750_000, abs=2_000)
    assert high == pytest.approx(10_250_000, abs=2_000)
    assert row.max() > row.min()


def test_config_bounds_and_frequency_validation(monkeypatch):
    calls = []
    monkeypatch.setattr('rf_mcp.receiver_backend.validate_frequency', lambda *args: calls.append(args))
    settings().validated()
    assert calls == [(10_000_000, None)]
    with pytest.raises(ValueError): settings(fft_size=1000).validated()
    with pytest.raises(ValueError): settings(update_rate_hz=31).validated()
    with pytest.raises(ValueError): settings(maximum_duration_seconds=config.LIVE_WATERFALL_MAX_DURATION_SECONDS + 1).validated()


def test_auto_receiver_policy_is_resolved_before_iq_subscription(monkeypatch):
    monkeypatch.setattr('rf_mcp.receiver_backend.validate_frequency', lambda *_: pytest.fail('auto is not a receiver ID'))
    monkeypatch.setattr('rf_mcp.live_waterfall.plan_assignment', lambda **_values: {
        'selected': {'receiver_id': 'idle-waterfall'},
    })
    monkeypatch.setattr('rf_mcp.receiver_backend.resolve_receiver',
                        lambda receiver_id: ({'receiver_id': receiver_id}, None))

    subscribed = {}
    class IQSubscription:
        chunks = queue.Queue()
        error = None
        def close(self): pass
    class IQManager:
        def subscribe(self, frequency_hz, receiver_id):
            subscribed.update(frequency_hz=frequency_hz, receiver_id=receiver_id)
            result = IQSubscription(); result.chunks.put(None); return result
        def shutdown(self): pass

    manager = LiveWaterfallManager(iq_manager=IQManager())
    subscription = manager.subscribe(settings(receiver_id='auto'))
    assert subscribed == {'frequency_hz': 10_000_000, 'receiver_id': 'idle-waterfall'}
    assert manager.status()['sessions'][0]['receiver_id'] == 'idle-waterfall'
    assert subscription.rows.get(timeout=2) is None


def test_history_and_listener_queues_are_bounded(monkeypatch):
    manager = LiveWaterfallManager()
    monkeypatch.setattr(config, 'LIVE_WATERFALL_HISTORY_ROWS', 3)
    # Exercise the same bounded structures without starting receiver hardware.
    session = type('Session', (), {'session_id': 's'})()
    manager._rows['s'] = __import__('collections').deque(maxlen=3)
    listener = queue.Queue(maxsize=2); manager._listeners['s'] = [listener]
    for index in range(5): manager._broadcast(session, {'sequence': index})
    assert [x['sequence'] for x in manager._rows['s']] == [2, 3, 4]
    assert [listener.get_nowait()['sequence'] for _ in range(2)] == [3, 4]


def test_producer_closes_iq_generator_and_uses_no_artifact_api(monkeypatch):
    closed = threading.Event()
    class IQ:
        def __iter__(self):
            yield np.ones(1024, dtype=np.complex64)
        def close(self): closed.set()
    monkeypatch.setattr('rf_mcp.receiver_backend.validate_frequency', lambda *_: None)
    monkeypatch.setattr('rf_mcp.receiver_backend.resolve_receiver', lambda _id: ({'receiver_id': 'r'}, None))
    monkeypatch.setattr('rf_mcp.receiver_backend.stream_iq_chunks', lambda *a, **k: IQ())
    # The live path must never acquire these artifact-producing APIs.
    for name in ('capture_iq', 'inspect_spectrum'):
        if hasattr(__import__('rf_mcp.receiver_backend', fromlist=[name]), name):
            monkeypatch.setattr('rf_mcp.receiver_backend.' + name, lambda *a, **k: pytest.fail(name))
    manager = LiveWaterfallManager(); subscription = manager.subscribe(settings())
    frame = subscription.rows.get(timeout=2)
    assert frame['encoding'] == 'base64' and frame['bin_count'] == 128
    assert subscription.rows.get(timeout=2) is None
    assert closed.wait(1)
    assert manager.status()['history'][0]['termination_reason'] == 'duration_limit'


def test_last_disconnect_stops_session(monkeypatch):
    monkeypatch.setattr('rf_mcp.receiver_backend.validate_frequency', lambda *_: None)
    monkeypatch.setattr('rf_mcp.receiver_backend.resolve_receiver', lambda _id: ({'receiver_id': 'r'}, None))
    release = threading.Event(); closed = threading.Event()
    class IQ:
        def __iter__(self):
            while not release.wait(.01): yield np.empty(0, np.complex64)
        def close(self): closed.set()
    monkeypatch.setattr('rf_mcp.receiver_backend.stream_iq_chunks', lambda *a, **k: IQ())
    manager = LiveWaterfallManager(); subscription = manager.subscribe(settings())
    subscription.close(); release.set()
    assert closed.wait(2)
    assert manager.status()['sessions'][0]['termination_reason'] == 'stopped'


def test_public_row_rate_drop_and_stop_metrics():
    session = _Session('s', settings(), 'r', LiveWaterfallState.COMPLETED, 'now',
                       rows_produced=6, first_iq_monotonic=1,
                       first_row_monotonic=1.1, latest_row_monotonic=2.1,
                       stop_requested_monotonic=2.2, stopped_monotonic=2.25,
                       rows_dropped=2)
    public = session.public()
    assert public['first_iq_monotonic'] < public['first_row_monotonic']
    assert public['effective_row_rate_hz'] == 5
    assert public['rows_dropped'] == 2
    assert public['stop_latency_ms'] == 50
