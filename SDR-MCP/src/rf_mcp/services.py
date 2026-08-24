from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .catalog import Catalog
from .sdr_coordinator import (
    coordinator_status,
    discover_devices,
    register_discovered_device,
)


class ReceiverService:
    """Application boundary for receiver inventory and guided onboarding."""

    def discover(self) -> dict:
        return discover_devices()

    def register(self, **values) -> dict:
        return register_discovered_device(**values)

    def status(self) -> dict:
        return coordinator_status()


@dataclass(frozen=True)
class RfApplicationServices:
    """Shared use-case container consumed by MCP and browser adapters.

    Keeping framework adapters dependent on this object prevents dashboard routes
    from growing separate RF behavior and provides a stable seam for unit tests.
    """

    catalog: Catalog
    receivers: ReceiverService
    spectrum_capture: Callable
    signal_analyzer: Callable
    broadcast_fm_receiver: Callable

    def recovery_status(self, interrupted_jobs_on_startup: int) -> dict:
        return {
            "catalog_schema": self.catalog.schema_status(),
            "interrupted_jobs_recovered_on_startup": interrupted_jobs_on_startup,
            "receiver_coordination": self.receivers.status(),
            "recovery_policy": {
                "unfinished_jobs": "marked interrupted at service startup",
                "receiver_leases": "SQLite-backed; expired leases are reclaimed automatically",
                "artifacts": "retained and remain catalog-addressable",
            },
        }
