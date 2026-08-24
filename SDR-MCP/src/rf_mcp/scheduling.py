from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable


MIN_INTERVAL_SECONDS = 60
MAX_INTERVAL_SECONDS = 7 * 24 * 60 * 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("start_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("start_at must include a timezone offset or Z")
    return parsed.astimezone(timezone.utc)


def normalize_schedule(
    *,
    name: str,
    interval_seconds: int,
    start_at: str | None,
    enabled: bool,
    now: datetime | None = None,
) -> tuple[str, int, bool, str]:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or re.search(r"[\x00-\x1f\x7f]", name):
        raise ValueError("schedule name must contain 1 through 64 printable characters")
    interval_seconds = int(interval_seconds)
    if not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS:
        raise ValueError(
            f"interval_seconds must be from {MIN_INTERVAL_SECONDS} through "
            f"{MAX_INTERVAL_SECONDS}"
        )
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    now = (now or utc_now()).astimezone(timezone.utc)
    first_run = parse_utc(start_at) if start_at else now + timedelta(seconds=interval_seconds)
    if first_run < now:
        first_run = now
    return name, interval_seconds, enabled, first_run.isoformat()


class SchedulerManager:
    """Persistent fixed-interval scheduler with at-most-one catch-up execution."""

    def __init__(
        self,
        catalog,
        launch_preset: Callable[[str, str], dict],
        receiver_busy: Callable[[], bool],
        *,
        poll_seconds: float = 2.0,
    ) -> None:
        self.catalog = catalog
        self.launch_preset = launch_preset
        self.receiver_busy = receiver_busy
        self.poll_seconds = max(0.1, float(poll_seconds))
        self._stop_event = threading.Event()
        self._execution_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_loop_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="rf-mcp-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def status(self) -> dict:
        enabled = self.catalog.list_schedules(enabled=True, limit=200)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "poll_seconds": self.poll_seconds,
            "enabled_schedule_count": len(enabled),
            "next_run_at": enabled[0]["next_run_at"] if enabled else None,
            "last_loop_error": self._last_loop_error,
        }

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
                self._last_loop_error = None
            except Exception as exc:
                self._last_loop_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.wait(self.poll_seconds)

    def tick(self, now: datetime | None = None) -> list[dict]:
        now = (now or utc_now()).astimezone(timezone.utc)
        due = self.catalog.due_schedules(now.isoformat(), limit=20)
        return [self._execute(schedule, now, require_due=True) for schedule in due]

    def run_now(self, schedule_id_or_name: str, now: datetime | None = None) -> dict:
        now = (now or utc_now()).astimezone(timezone.utc)
        schedule = self.catalog.get_schedule(schedule_id_or_name)
        return self._execute(schedule, now, require_due=False)

    def _execute(self, schedule: dict, now: datetime, *, require_due: bool) -> dict:
        with self._execution_lock:
            schedule = self.catalog.get_schedule(schedule["schedule_id"])
            if require_due and (
                not schedule["enabled"] or parse_utc(schedule["next_run_at"]) > now
            ):
                return {**schedule, "execution_status": "not_due"}
            next_run = now + timedelta(seconds=schedule["interval_seconds"])
            attempted_at = now.isoformat()
            self.catalog.advance_schedule(
                schedule["schedule_id"],
                attempted_at=attempted_at,
                next_run_at=next_run.isoformat(),
            )
            if self.receiver_busy():
                return self.catalog.record_schedule_result(
                    schedule["schedule_id"],
                    status="skipped_busy",
                    attempted_at=attempted_at,
                    error="Airspy receiver is already occupied by another long-running job",
                )
            try:
                launched = self.launch_preset(
                    schedule["preset_id"], schedule["schedule_id"]
                )
                job_id = launched.get("job_id")
                status = ("completed" if schedule["preset_type"] in
                          {"watchlist", "station_memory_scan"} else "launched")
                return self.catalog.record_schedule_result(
                    schedule["schedule_id"],
                    status=status,
                    attempted_at=attempted_at,
                    job_id=job_id,
                )
            except Exception as exc:
                return self.catalog.record_schedule_result(
                    schedule["schedule_id"],
                    status="failed",
                    attempted_at=attempted_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
