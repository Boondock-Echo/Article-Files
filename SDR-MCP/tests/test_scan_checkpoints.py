from __future__ import annotations

import pytest

from rf_mcp.scanning import ScanJob, ScanManager, ScanSegment


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def segment(frequency_hz: int) -> ScanSegment:
    return ScanSegment(frequency_hz, "2026-08-24T00:00:00+00:00", -40.0, 0.0)


def test_step_checkpoints_are_coalesced_by_time_or_progress(monkeypatch):
    clock = FakeClock()
    manager = ScanManager(
        checkpoint_min_interval_seconds=10,
        checkpoint_min_completed_steps=3,
        monotonic=clock,
    )
    job = ScanJob("scan-coalesce", {"classify_top_signals": 0}, [1, 2, 3, 4, 5])
    job.state = "running"
    writes = []
    monkeypatch.setattr(
        "rf_mcp.scanning.catalog.upsert_job",
        lambda *args, **kwargs: writes.append(kwargs["summary"]),
    )

    assert manager._checkpoint(job)
    job.segments.append(segment(1))
    assert not manager._checkpoint(job, current_frequency_hz=1)
    job.segments.append(segment(2))
    assert not manager._checkpoint(job, current_frequency_hz=2)
    job.segments.append(segment(3))
    assert manager._checkpoint(job, current_frequency_hz=3)
    job.segments.append(segment(4))
    clock.advance(10)
    assert manager._checkpoint(job, current_frequency_hz=4)

    assert [write["completed_steps"] for write in writes] == [0, 3, 4]
    assert len(job.segments) == 4
    assert job.last_persisted_monotonic == clock.value


def test_transitions_errors_and_stop_requests_persist_immediately(monkeypatch):
    clock = FakeClock()
    manager = ScanManager(
        checkpoint_min_interval_seconds=60,
        checkpoint_min_completed_steps=100,
        monotonic=clock,
    )
    job = ScanJob("scan-transitions", {"classify_top_signals": 0}, [1])
    manager._jobs[job.job_id] = job
    writes = []
    monkeypatch.setattr(
        "rf_mcp.scanning.catalog.upsert_job",
        lambda *args, **kwargs: writes.append((args[2], kwargs)),
    )

    manager._checkpoint(job)
    job.state = "running"
    manager._checkpoint(job)
    job.error = "receiver disconnected"
    manager._checkpoint(job)
    result = manager.stop(job.job_id)

    assert result["stop_requested"] is True
    assert [state for state, _ in writes] == ["queued", "running", "running", "running"]
    assert writes[-2][1]["error"] == "receiver disconnected"
    assert writes[-1][1]["summary"]["stop_requested"] is True


@pytest.mark.parametrize("outcome", ["completed", "stopped", "failed"])
def test_run_forces_durable_final_state(monkeypatch, outcome):
    clock = FakeClock()
    manager = ScanManager(
        checkpoint_min_interval_seconds=60,
        checkpoint_min_completed_steps=100,
        monotonic=clock,
    )
    centers = [1] if outcome == "failed" else []
    job = ScanJob(
        "scan-final",
        {
            "classify_top_signals": 0,
            "capture_duration_seconds": 0.1,
            "attenuation_steps": 0,
        },
        centers,
    )
    if outcome == "stopped":
        job.stop_event.set()
    writes = []
    monkeypatch.setattr("rf_mcp.scanning.ensure_data_dirs", lambda: None)
    monkeypatch.setattr("rf_mcp.scanning.release_long_job", lambda job_id: None)
    monkeypatch.setattr(manager, "_write_results", lambda current: ({}, None))
    monkeypatch.setattr(
        "rf_mcp.scanning.catalog.upsert_job",
        lambda *args, **kwargs: writes.append((args[2], kwargs)),
    )
    if outcome == "failed":
        monkeypatch.setattr(
            "rf_mcp.scanning.capture_iq",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
        )

    manager._run(job)

    assert job.state == outcome
    assert writes[-1][0] == outcome
    assert writes[-1][1]["completed_at"] == job.completed_at
    assert writes[-1][1]["summary"]["progress_percent"] == 100.0
    if outcome == "failed":
        assert writes[-1][1]["error"] == "RuntimeError: capture failed"
