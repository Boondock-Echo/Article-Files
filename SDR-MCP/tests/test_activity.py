from __future__ import annotations

from time import perf_counter

from rf_mcp.activity import _cluster_signals, summarize_activity_runs


def _run(job_id: str, created_at: str, frequencies: list[float]) -> dict:
    return {
        "job_id": job_id,
        "created_at": created_at,
        "signals": [
            {"frequency_hz": frequency, "above_noise_db": index + 1}
            for index, frequency in enumerate(frequencies)
        ],
        "signal_count": len(frequencies),
    }


def test_clusters_include_boundary_and_count_runs_not_repeated_signals() -> None:
    runs = [
        _run("one", "2026-01-01T00:00:00Z", [1_000, 1_100]),
        _run("two", "2026-01-02T00:00:00Z", [1_050, 1_150]),
    ]

    clusters = _cluster_signals(runs, 100)

    assert clusters == [{
        "mean_frequency_hz": 1_075.0,
        "minimum_frequency_hz": 1_000.0,
        "maximum_frequency_hz": 1_150.0,
        "observation_count": 4,
        "run_count": 2,
        "detection_rate": 1.0,
        "maximum_above_noise_db": 2.0,
        "first_seen_at": "2026-01-01T00:00:00Z",
        "last_seen_at": "2026-01-02T00:00:00Z",
    }]


def test_cluster_results_are_deterministic_for_out_of_order_input() -> None:
    chronological = [
        _run("early", "2026-01-01T00:00:00Z", [4_000, 1_000]),
        _run("late", "2026-01-03T00:00:00Z", [1_050, 4_050]),
    ]
    reversed_runs = list(reversed(chronological))

    assert _cluster_signals(reversed_runs, 100) == _cluster_signals(chronological, 100)
    assert _cluster_signals(reversed_runs, 100)[0]["first_seen_at"] == "2026-01-01T00:00:00Z"


def test_new_signals_use_inclusive_tolerance_and_empty_history() -> None:
    runs = [
        _run("empty", "2026-01-01T00:00:00Z", []),
        _run("history", "2026-01-02T00:00:00Z", [2_000, 1_000]),
        _run("latest", "2026-01-03T00:00:00Z", [2_100, 899, 5_000]),
    ]

    summary = summarize_activity_runs(runs, frequency_tolerance_hz=100)
    assert summary["new_signal_frequencies_hz"] == [899.0, 5_000.0]

    empty_history = summarize_activity_runs(
        [_run("empty", "2026-01-01T00:00:00Z", []),
         _run("latest", "2026-01-02T00:00:00Z", [123])]
    )
    assert empty_history["new_signal_frequencies_hz"] == [123.0]

    no_prior_run = summarize_activity_runs([_run("only", "2026-01-01T00:00:00Z", [123])])
    assert no_prior_run["new_signal_frequencies_hz"] == []


def test_thousands_of_signals_complete_without_quadratic_slowdown() -> None:
    # Widely spaced signals are the old implementation's worst case: every new
    # observation scanned every existing cluster and every latest signal scanned
    # all historical observations.
    frequencies = [float(index * 10) for index in range(8_000)]
    runs = [
        _run("history", "2026-01-01T00:00:00Z", frequencies),
        _run("latest", "2026-01-02T00:00:00Z", [value + 2 for value in frequencies]),
    ]

    started = perf_counter()
    summary = summarize_activity_runs(runs, frequency_tolerance_hz=1)
    elapsed = perf_counter() - started

    assert summary["cluster_count"] == 16_000
    assert summary["new_signal_count"] == 8_000
    assert elapsed < 2.5
