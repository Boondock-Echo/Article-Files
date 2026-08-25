"""Common cooperative cancellation contract for receiver-owning operations.

Managers for streams, scans, surveys, monitors, satellite/SSTV reception and
scheduled station-memory work register the same small adapter.  Importantly,
the registry is keyed by operation/job identity -- it is never a lease deletion
API.  A lease can only disappear when its owner reports completion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class CooperativeCancellation(Protocol):
    """The stop interface implemented by every receiver-owning manager."""

    def request_stop(self, operation_id: str, *, reason: str) -> None:
        """Request a graceful stop and return without forcibly releasing hardware."""


@dataclass(frozen=True)
class StopAdapter:
    callback: Callable[..., object]

    def request_stop(self, operation_id: str, *, reason: str) -> None:
        self.callback(operation_id)


class ActiveOperationRegistry:
    """Process-local link from durable operation IDs to their owning managers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._operations: dict[str, CooperativeCancellation] = {}

    def register(self, operation_id: str, manager: CooperativeCancellation | Callable) -> None:
        adapter = manager if isinstance(manager, CooperativeCancellation) else StopAdapter(manager)
        with self._lock:
            self._operations[operation_id] = adapter

    def unregister(self, operation_id: str) -> None:
        with self._lock:
            self._operations.pop(operation_id, None)

    def can_stop(self, operation_id: str) -> bool:
        with self._lock:
            return operation_id in self._operations

    def request_stop(self, operation_id: str, *, reason: str) -> None:
        with self._lock:
            manager = self._operations.get(operation_id)
        if manager is None:
            raise RuntimeError("cooperative_stop_unavailable")
        manager.request_stop(operation_id, reason=reason)


ACTIVE_OPERATIONS = ActiveOperationRegistry()
