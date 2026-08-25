"""Durable, receiver-aware admission queue.

The queue and receiver leases intentionally share a SQLite database.  This makes
the transition from queued work to an assigned receiver a single transaction,
including when several server processes dispatch at the same time.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from . import sdr_coordinator

STATES = frozenset({"queued", "waiting_for_receiver", "starting", "running", "stopping",
                    "completed", "failed", "cancelled", "expired", "preempted"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "expired", "preempted"})
VALID_TRANSITIONS = {
    "queued": {"waiting_for_receiver", "starting", "cancelled", "expired", "preempted"},
    "waiting_for_receiver": {"queued", "starting", "cancelled", "expired", "preempted"},
    "starting": {"running", "stopping", "failed", "cancelled", "preempted"},
    "running": {"stopping", "completed", "failed", "preempted"},
    "stopping": {"completed", "failed", "cancelled", "preempted"},
    "completed": set(), "failed": set(), "cancelled": set(), "expired": set(),
    "preempted": set(),
}

DEFAULT_PRIORITIES = {
    "time_critical": 400, "scheduled_deadline": 400,
    "interactive": 300, "scheduled": 200,
    "background": 100, "survey": 100, "monitor": 100,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobQueue:
    """SQLite job repository and atomic compatible-receiver dispatcher."""

    def __init__(self, database_path: str | Path | None = None, *,
                 aging_seconds: float = 300, lease_seconds: int | None = None):
        self.database_path = Path(database_path or sdr_coordinator._lease_database_path())
        self.aging_seconds = max(float(aging_seconds), .001)
        self.lease_seconds = int(lease_seconds or sdr_coordinator.LEASE_SECONDS)
        self._condition = threading.Condition()
        self._revision = 0
        self._initialize()
        self.recover_interrupted()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS receiver_leases (
                lease_id TEXT PRIMARY KEY, receiver_id TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT '', acquired_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL, expires_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS admission_jobs (
                queue_id TEXT PRIMARY KEY, catalog_job_id TEXT, operation_type TEXT NOT NULL,
                request_config TEXT NOT NULL, receiver_policy TEXT NOT NULL,
                required_tuning_ranges TEXT NOT NULL, required_bandwidth_hz INTEGER NOT NULL,
                required_backends TEXT NOT NULL, preferred_role TEXT, priority INTEGER NOT NULL,
                priority_class TEXT NOT NULL, created_at TEXT NOT NULL, deadline TEXT,
                estimated_rf_duration_seconds REAL, state TEXT NOT NULL,
                cancellation_requested INTEGER NOT NULL DEFAULT 0, assigned_receiver_id TEXT,
                lease_id TEXT, started_at TEXT, ended_at TEXT, failure_reason TEXT,
                blocking_reasons TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS admission_jobs_state_priority ON admission_jobs(state, priority, created_at)")

    def _notify(self) -> None:
        with self._condition:
            self._revision += 1
            self._condition.notify_all()

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        item = dict(row)
        for key in ("request_config", "receiver_policy", "required_tuning_ranges",
                    "required_backends", "blocking_reasons"):
            item[key] = json.loads(item[key])
        item["cancellation_requested"] = bool(item["cancellation_requested"])
        return item

    def enqueue(self, *, operation_type: str, request_config: dict,
                catalog_job_id: str | None = None, receiver_id: str | None = "auto",
                receiver_policy: dict | str | None = None,
                required_tuning_ranges: Iterable[Iterable[int]] | None = None,
                required_bandwidth_hz: int = 0, required_backends: Iterable[str] | None = None,
                preferred_role: str | None = None, priority: int | None = None,
                priority_class: str = "interactive", deadline: str | None = None,
                estimated_rf_duration_seconds: float | None = None,
                queue_id: str | None = None) -> dict:
        if not operation_type.strip():
            raise ValueError("operation_type is required")
        if not isinstance(request_config, dict):
            raise ValueError("request_config must be an object")
        if priority_class not in DEFAULT_PRIORITIES:
            raise ValueError(f"unknown priority_class: {priority_class}")
        ranges = [[int(a), int(b)] for a, b in (required_tuning_ranges or [])]
        if any(a < 0 or b < a for a, b in ranges):
            raise ValueError("required tuning ranges must be ordered and non-negative")
        bandwidth = int(required_bandwidth_hz)
        if bandwidth < 0:
            raise ValueError("required_bandwidth_hz must not be negative")
        if deadline is not None:
            datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        now = _utcnow().isoformat()
        queue_id = queue_id or f"queue-{uuid4().hex}"
        policy = receiver_policy if isinstance(receiver_policy, dict) else {
            "mode": receiver_policy or ("automatic" if receiver_id in (None, "", "auto") else "requested"),
            "receiver_id": "auto" if receiver_id in (None, "", "auto") else receiver_id,
        }
        values = (queue_id, catalog_job_id, operation_type.strip(), self._json(request_config),
                  self._json(policy), self._json(ranges), bandwidth,
                  self._json(sorted(set(required_backends or []))), preferred_role,
                  int(DEFAULT_PRIORITIES[priority_class] if priority is None else priority),
                  priority_class, now, deadline, estimated_rf_duration_seconds, "queued", now)
        with self._connect() as db:
            db.execute("""INSERT INTO admission_jobs
                (queue_id,catalog_job_id,operation_type,request_config,receiver_policy,
                 required_tuning_ranges,required_bandwidth_hz,required_backends,preferred_role,
                 priority,priority_class,created_at,deadline,estimated_rf_duration_seconds,state,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
        self._notify()
        return self.get(queue_id)

    def get(self, queue_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM admission_jobs WHERE queue_id=?", (queue_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown queued job: {queue_id}")
        return self._decorate(self._row(row))

    def list(self, *, states: Iterable[str] | None = None, limit: int = 200) -> list[dict]:
        wanted = list(states or [])
        if any(state not in STATES for state in wanted):
            raise ValueError("unknown job state")
        sql, args = "SELECT * FROM admission_jobs", []
        if wanted:
            sql += f" WHERE state IN ({','.join('?' for _ in wanted)})"
            args.extend(wanted)
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(max(1, min(int(limit), 1000)))
        with self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        return [self._decorate(self._row(row)) for row in rows]

    def _effective_priority(self, row: dict, now: datetime) -> float:
        age = max(0.0, (now - datetime.fromisoformat(row["created_at"])).total_seconds())
        return row["priority"] + age / self.aging_seconds

    def _decorate(self, item: dict) -> dict:
        if item["state"] not in {"queued", "waiting_for_receiver"}:
            item.update(queue_position=None, effective_priority=item["priority"])
            return item
        now = _utcnow()
        pending = self.list_raw_pending()
        ranked = sorted(pending, key=lambda x: (-self._effective_priority(x, now), x["created_at"], x["queue_id"]))
        item["queue_position"] = next((i + 1 for i, x in enumerate(ranked) if x["queue_id"] == item["queue_id"]), None)
        item["effective_priority"] = self._effective_priority(item, now)
        return item

    def list_raw_pending(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM admission_jobs WHERE state IN ('queued','waiting_for_receiver')").fetchall()
        return [self._row(row) for row in rows]

    def transition(self, queue_id: str, state: str, *, reason: str | None = None) -> dict:
        if state not in STATES:
            raise ValueError(f"unknown state: {state}")
        now = _utcnow().isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM admission_jobs WHERE queue_id=?", (queue_id,)).fetchone()
            if row is None: raise KeyError(f"Unknown queued job: {queue_id}")
            if state not in VALID_TRANSITIONS[row["state"]]:
                raise ValueError(f"invalid job transition: {row['state']} -> {state}")
            started = now if state == "running" and not row["started_at"] else row["started_at"]
            ended = now if state in TERMINAL_STATES else row["ended_at"]
            db.execute("UPDATE admission_jobs SET state=?,started_at=?,ended_at=?,failure_reason=?,updated_at=? WHERE queue_id=?",
                       (state, started, ended, reason, now, queue_id))
            if state in TERMINAL_STATES and row["lease_id"]:
                db.execute("DELETE FROM receiver_leases WHERE lease_id=?", (row["lease_id"],))
        self._notify()
        return self.get(queue_id)

    def cancel(self, queue_id: str) -> dict:
        item = self.get(queue_id)
        if item["state"] in TERMINAL_STATES: return item
        if item["state"] in {"queued", "waiting_for_receiver"}:
            return self.transition(queue_id, "cancelled")
        with self._connect() as db:
            db.execute("UPDATE admission_jobs SET cancellation_requested=1,updated_at=? WHERE queue_id=?",
                       (_utcnow().isoformat(), queue_id))
        self._notify(); return self.get(queue_id)

    def recover_interrupted(self) -> int:
        now = _utcnow().isoformat()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT lease_id FROM admission_jobs WHERE state IN ('starting','running','stopping')").fetchall()
            for row in rows:
                if row[0]: db.execute("DELETE FROM receiver_leases WHERE lease_id=?", (row[0],))
            result = db.execute("""UPDATE admission_jobs SET state='queued',assigned_receiver_id=NULL,
                lease_id=NULL,started_at=NULL,blocking_reasons='["service restarted"]',updated_at=?
                WHERE state IN ('starting','running','stopping')""", (now,))
        if result.rowcount: self._notify()
        return result.rowcount

    def dispatch_once(self, receivers: list[dict] | None = None) -> list[dict]:
        """Assign as many jobs as possible; receiver compatibility avoids global HOL blocking."""
        now = _utcnow(); assigned = []
        receiver_items = receivers if receivers is not None else sdr_coordinator._load()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM receiver_leases WHERE expires_at<=?", (now.isoformat(),))
            busy = {r[0] for r in db.execute("SELECT receiver_id FROM receiver_leases")}
            rows = [self._row(r) for r in db.execute("SELECT * FROM admission_jobs WHERE state IN ('queued','waiting_for_receiver')")]
            for job in rows:
                if job["deadline"] and datetime.fromisoformat(job["deadline"].replace("Z", "+00:00")) <= now:
                    db.execute("UPDATE admission_jobs SET state='expired',ended_at=?,updated_at=? WHERE queue_id=?", (now.isoformat(), now.isoformat(), job["queue_id"]))
            rows = [j for j in rows if not j["deadline"] or datetime.fromisoformat(j["deadline"].replace("Z", "+00:00")) > now]
            # Consider the best compatible job independently for every available receiver.
            for receiver in sorted(receiver_items, key=lambda r: (-r.get("priority", 0), r["receiver_id"])):
                rid = receiver["receiver_id"]
                if rid in busy or not receiver.get("enabled", False): continue
                candidates = [(self._effective_priority(j, now), j) for j in rows if self._compatible(j, receiver)]
                if not candidates: continue
                _, job = max(candidates, key=lambda pair: (pair[0], -datetime.fromisoformat(pair[1]["created_at"]).timestamp()))
                lease_id = f"lease-{uuid4().hex[:12]}"; expires = now + timedelta(seconds=self.lease_seconds)
                try:
                    db.execute("INSERT INTO receiver_leases VALUES (?,?,?,?,?,?,?)",
                               (lease_id, rid, f"queue:{job['queue_id']}", job["operation_type"], now.isoformat(), now.isoformat(), expires.isoformat()))
                except sqlite3.IntegrityError: continue
                db.execute("""UPDATE admission_jobs SET state='starting',assigned_receiver_id=?,lease_id=?,
                    blocking_reasons='[]',updated_at=? WHERE queue_id=? AND state IN ('queued','waiting_for_receiver')""",
                           (rid, lease_id, now.isoformat(), job["queue_id"]))
                busy.add(rid); rows.remove(job); assigned.append(job["queue_id"])
            for job in rows:
                reasons = self._blocking_reasons(job, receiver_items, busy)
                db.execute("UPDATE admission_jobs SET state='waiting_for_receiver',blocking_reasons=?,updated_at=? WHERE queue_id=?",
                           (self._json(reasons), now.isoformat(), job["queue_id"]))
        if assigned: self._notify()
        return [self.get(queue_id) for queue_id in assigned]

    @staticmethod
    def _compatible(job: dict, receiver: dict) -> bool:
        policy_id = job["receiver_policy"].get("receiver_id", "auto")
        if policy_id not in (None, "", "auto") and receiver["receiver_id"] != policy_id: return False
        if job["required_backends"] and receiver.get("backend") not in job["required_backends"]: return False
        if job["preferred_role"] and receiver.get("role") != job["preferred_role"]: return False
        if job["required_bandwidth_hz"] > receiver.get("max_bandwidth_hz", 0): return False
        available = receiver.get("tuning_ranges_hz", [])
        return all(any(low <= need_low and need_high <= high for low, high in available)
                   for need_low, need_high in job["required_tuning_ranges"])

    def _blocking_reasons(self, job: dict, receivers: list[dict], busy: set[str]) -> list[str]:
        compatible = [r for r in receivers if self._compatible(job, r) and r.get("enabled", False)]
        if not compatible: return ["no compatible receiver"]
        if all(r["receiver_id"] in busy for r in compatible): return ["compatible receivers are leased"]
        return ["higher-priority compatible work"]

    def wait_for_change(self, *, after_revision: int = 0, timeout: float = 30) -> dict:
        with self._condition:
            if self._revision <= after_revision: self._condition.wait(max(0, min(timeout, 60)))
            return {"revision": self._revision, "changed": self._revision > after_revision}


class Dispatcher:
    """Wakeable dispatch loop; ``starter`` receives each atomically assigned job."""
    def __init__(self, queue: JobQueue, starter: Callable[[dict], None] | None = None):
        self.queue, self.starter = queue, starter
        self._stop = threading.Event(); self._wake = threading.Event(); self._thread = None

    def wake(self) -> None: self._wake.set()
    def run_once(self) -> list[dict]:
        jobs = self.queue.dispatch_once()
        if self.starter:
            for job in jobs: self.starter(job)
        return jobs
    def start(self) -> None:
        if self._thread and self._thread.is_alive(): return
        self._thread = threading.Thread(target=self._run, name="rf-job-dispatcher", daemon=True); self._thread.start()
    def _run(self) -> None:
        revision = -1
        while not self._stop.is_set():
            self.run_once()
            change = self.queue.wait_for_change(after_revision=revision, timeout=1)
            revision = change["revision"]
            if not change["changed"]:
                self._wake.wait(29)
            self._wake.clear()
    def stop(self) -> None:
        self._stop.set(); self.wake()
        if self._thread: self._thread.join(timeout=5)
