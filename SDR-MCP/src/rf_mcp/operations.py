from __future__ import annotations

import threading

_LOCK = threading.Lock()
_ACTIVE_LONG_JOB: str | None = None


def acquire_long_job(job_id: str) -> None:
    global _ACTIVE_LONG_JOB
    with _LOCK:
        if _ACTIVE_LONG_JOB is not None:
            raise RuntimeError(f"Long-running RF job {_ACTIVE_LONG_JOB} is already active")
        _ACTIVE_LONG_JOB = job_id


def release_long_job(job_id: str) -> None:
    global _ACTIVE_LONG_JOB
    with _LOCK:
        if _ACTIVE_LONG_JOB == job_id:
            _ACTIVE_LONG_JOB = None


def active_long_job() -> str | None:
    with _LOCK:
        return _ACTIVE_LONG_JOB
