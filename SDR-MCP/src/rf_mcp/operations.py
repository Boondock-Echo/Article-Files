from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()
_ACTIVE_LONG_JOBS: dict[str, str] = {}
_CPU_LIMIT = max(1, int(os.getenv("RF_MCP_DECODER_CPU_LIMIT", "64")))


def acquire_long_job(job_id: str, receiver_id: str = "default", *, cpu_units: int = 1) -> None:
    with _LOCK:
        if receiver_id in _ACTIVE_LONG_JOBS:
            raise RuntimeError(
                f"Long-running RF job {_ACTIVE_LONG_JOBS[receiver_id]} is already active "
                f"on receiver {receiver_id}"
            )
        if len(_ACTIVE_LONG_JOBS) + max(1, cpu_units) - 1 >= _CPU_LIMIT:
            raise RuntimeError("System-wide decoder CPU limit reached")
        _ACTIVE_LONG_JOBS[receiver_id] = job_id


def release_long_job(job_id: str) -> None:
    with _LOCK:
        for receiver_id, active_id in list(_ACTIVE_LONG_JOBS.items()):
            if active_id == job_id:
                del _ACTIVE_LONG_JOBS[receiver_id]


def active_long_job() -> str | None:
    with _LOCK:
        return next(iter(_ACTIVE_LONG_JOBS.values()), None)


def active_long_jobs() -> dict[str, str]:
    with _LOCK:
        return dict(_ACTIVE_LONG_JOBS)
