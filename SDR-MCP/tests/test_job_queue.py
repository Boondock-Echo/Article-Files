from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from rf_mcp.job_queue import JobQueue, VALID_TRANSITIONS
from rf_mcp.cooperative_cancellation import ActiveOperationRegistry


def receiver(receiver_id, low, high, *, backend="rtl_sdr"):
    return {"receiver_id": receiver_id, "enabled": True, "verified": True,
            "backend": backend, "role": "general", "priority": 50,
            "tuning_ranges_hz": [[low, high]], "max_bandwidth_hz": 2_400_000}


def enqueue(queue, frequency, **kwargs):
    return queue.enqueue(operation_type="capture", request_config={"frequency_hz": frequency},
                         required_tuning_ranges=[[frequency, frequency]], **kwargs)


def test_concurrent_receivers_and_incompatible_head_of_line(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite")
    impossible = enqueue(queue, 9_000_000_000, priority=999)
    first = enqueue(queue, 100_000_000)
    second = enqueue(queue, 7_100_000)
    assigned = queue.dispatch_once([receiver("vhf", 80_000_000, 200_000_000),
                                    receiver("hf", 1_000_000, 30_000_000)])
    assert {job["queue_id"] for job in assigned} == {first["queue_id"], second["queue_id"]}
    assert queue.get(impossible["queue_id"])["blocking_reasons"] == ["no compatible receiver"]


def test_priority_ordering_and_aging(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite", aging_seconds=1)
    old = enqueue(queue, 100_000_000, priority=100, queue_id="old")
    with queue._connect() as db:
        created = (datetime.now(timezone.utc) - timedelta(seconds=250)).isoformat()
        db.execute("UPDATE admission_jobs SET created_at=? WHERE queue_id=?", (created, old["queue_id"]))
    enqueue(queue, 100_000_000, priority=300, queue_id="new")
    assigned = queue.dispatch_once([receiver("one", 80_000_000, 200_000_000)])
    assert assigned[0]["queue_id"] == "old"


def test_deadline_expiration_and_queued_cancel(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite")
    expired = enqueue(queue, 100, deadline=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
    cancelled = enqueue(queue, 200)
    assert queue.cancel(cancelled["queue_id"])["state"] == "cancelled"
    assert queue.dispatch_once([receiver("all", 0, 1000)]) == []
    assert queue.get(expired["queue_id"])["state"] == "expired"


def test_restart_recovery_requeues_and_releases(tmp_path):
    path = tmp_path / "queue.sqlite"
    queue = JobQueue(path)
    job = enqueue(queue, 100)
    queue.dispatch_once([receiver("all", 0, 1000)])
    queue.transition(job["queue_id"], "running")
    recovered = JobQueue(path)
    result = recovered.get(job["queue_id"])
    assert result["state"] == "queued"
    assert result["assigned_receiver_id"] is None
    with recovered._connect() as db:
        assert db.execute("SELECT count(*) FROM receiver_leases").fetchone()[0] == 0


def test_lease_release_allows_waiter_on_next_dispatch(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite")
    first, second = enqueue(queue, 100), enqueue(queue, 100)
    assigned = queue.dispatch_once([receiver("all", 0, 1000)])[0]
    queue.transition(assigned["queue_id"], "running")
    queue.transition(assigned["queue_id"], "completed")
    next_job = queue.dispatch_once([receiver("all", 0, 1000)])[0]
    assert {first["queue_id"], second["queue_id"]} == {assigned["queue_id"], next_job["queue_id"]}


def test_atomic_assignment_race(tmp_path):
    path = tmp_path / "queue.sqlite"
    queue = JobQueue(path)
    enqueue(queue, 100); enqueue(queue, 100)
    barrier = threading.Barrier(2); results = []
    def dispatch():
        local = JobQueue(path); barrier.wait()
        results.extend(local.dispatch_once([receiver("only", 0, 1000)]))
    threads = [threading.Thread(target=dispatch) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert len(results) == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM receiver_leases").fetchone()[0] == 1


def test_transitions_are_central_and_enforced(tmp_path):
    assert "running" in VALID_TRANSITIONS["starting"]
    queue = JobQueue(tmp_path / "queue.sqlite")
    job = enqueue(queue, 100)
    with pytest.raises(ValueError, match="invalid job transition"):
        queue.transition(job["queue_id"], "completed")


def takeover_pair(tmp_path, *, preemptible=True):
    queue = JobQueue(tmp_path / "takeover.sqlite")
    old = enqueue(queue, 100, queue_id="old", preemptible=preemptible)
    queue.dispatch_once([receiver("only", 0, 1000)])
    queue.transition(old["queue_id"], "running")
    new = enqueue(queue, 100, queue_id="new")
    return queue, old, new


def test_successful_cooperative_takeover_and_audit_never_overlaps(tmp_path):
    queue, old, new = takeover_pair(tmp_path)
    operations = ActiveOperationRegistry()
    observed = []
    def stop(operation_id):
        with queue._connect() as db:
            observed.append(db.execute("SELECT owner FROM receiver_leases").fetchall())
        queue.transition(operation_id, "preempted")
        with queue._connect() as db:
            observed.append(db.execute("SELECT owner FROM receiver_leases").fetchall())
    operations.register("old", stop)
    result = queue.request_receiver_takeover(new_queue_id="new", blocking_queue_id="old",
        confirm_takeover=True, requested_by="operator@example", reason="urgent signal",
        operations=operations, receivers=[receiver("only", 0, 1000)])
    assert result["status"] == "assigned"
    assert [[row[0] for row in snapshot] for snapshot in observed] == [["queue:old"], []]
    with queue._connect() as db:
        assert [row[0] for row in db.execute("SELECT owner FROM receiver_leases")] == ["queue:new"]
    event = queue.events("new")[0]
    assert event["actor"] == "operator@example"
    assert event["detail"]["displaced_job_id"] == "old"


def test_takeover_refuses_non_preemptible_and_requires_authorized_identity(tmp_path):
    queue, old, new = takeover_pair(tmp_path, preemptible=False)
    with pytest.raises(PermissionError, match="non_preemptible"):
        queue.request_receiver_takeover(new_queue_id="new", blocking_queue_id="old",
            confirm_takeover=True, requested_by="operator", reason="test",
            receivers=[receiver("only", 0, 1000)])
    queue.set_preemptible("old")
    with pytest.raises(PermissionError, match="authorized"):
        queue.request_receiver_takeover(new_queue_id="new", blocking_queue_id="old",
            confirm_takeover=True, requested_by="", reason="test",
            receivers=[receiver("only", 0, 1000)])


def test_takeover_timeout_keeps_replacement_queued(tmp_path):
    queue, old, new = takeover_pair(tmp_path)
    operations = ActiveOperationRegistry(); operations.register("old", lambda _job_id: None)
    result = queue.request_receiver_takeover(new_queue_id="new", blocking_queue_id="old",
        confirm_takeover=True, requested_by="operator", reason="test", timeout=.01,
        operations=operations, receivers=[receiver("only", 0, 1000)])
    assert result["error"] == "takeover_timeout"
    assert queue.get("new")["state"] == "queued"


def test_replacement_cancellation_wins_race_while_stopping(tmp_path):
    queue, old, new = takeover_pair(tmp_path)
    operations = ActiveOperationRegistry()
    def stop(operation_id):
        queue.cancel("new")
        queue.transition(operation_id, "completed")
    operations.register("old", stop)
    result = queue.request_receiver_takeover(new_queue_id="new", blocking_queue_id="old",
        confirm_takeover=True, requested_by="operator", reason="test",
        operations=operations, receivers=[receiver("only", 0, 1000)])
    assert result["status"] == "replacement_cancelled"
    assert queue.get("new")["state"] == "cancelled"
