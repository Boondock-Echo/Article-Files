from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event
import time
from typing import Iterator, Protocol
from uuid import uuid4

import numpy as np

from . import airspyhf
from . import rtl_sdr
from .airspyhf import AirspyError, Capture
from .calibration import get_calibration
from .sdr_coordinator import (
    acquire_receiver,
    assign_and_acquire_receiver,
    ensure_airspy_default,
    get_receiver,
    heartbeat_receiver,
    release_receiver,
)


@dataclass(frozen=True)
class ReceiverCapabilities:
    backend: str
    tuning_ranges_hz: tuple[tuple[int, int], ...]
    sample_rates_hz: tuple[int, ...]
    maximum_bandwidth_hz: int
    supports_streaming: bool


class ReceiverBackend(Protocol):
    """Hardware adapter used by RF workflows instead of a receiver CLI directly."""

    name: str

    def device_info(self, receiver: dict) -> dict: ...

    def capture_iq(
        self, receiver: dict, center_frequency_hz: int, duration_seconds: float, **options,
    ) -> Capture: ...

    def stream_iq_chunks(
        self, receiver: dict, center_frequency_hz: int, *, duration_seconds: float,
        stop_event: Event, **options,
    ) -> Iterator[np.ndarray]: ...


class AirspyHfBackend:
    name = "airspyhf"

    def device_info(self, receiver: dict) -> dict:
        result = airspyhf.device_info()
        result.update(
            receiver_id=receiver["receiver_id"],
            backend=self.name,
            device_selector=receiver.get("device_selector", ""),
        )
        return result

    def capture_iq(
        self, receiver: dict, center_frequency_hz: int, duration_seconds: float, **options,
    ) -> Capture:
        capture = airspyhf.capture_iq(center_frequency_hz, duration_seconds, **options)
        return replace(
            capture,
            receiver_id=receiver["receiver_id"],
            backend=self.name,
            device_selector=receiver.get("device_selector", ""),
        )

    def stream_iq_chunks(
        self, receiver: dict, center_frequency_hz: int, *, duration_seconds: float,
        stop_event: Event, **options,
    ) -> Iterator[np.ndarray]:
        yield from airspyhf.stream_iq_chunks(
            center_frequency_hz,
            duration_seconds=duration_seconds,
            stop_event=stop_event,
            **options,
        )


class RtlSdrBackend:
    name = "rtl_sdr"

    def device_info(self, receiver: dict) -> dict:
        result = rtl_sdr.device_info(receiver.get("device_selector", ""))
        result.update(receiver_id=receiver["receiver_id"], backend=self.name)
        return result

    def capture_iq(
        self, receiver: dict, center_frequency_hz: int, duration_seconds: float, **options,
    ) -> Capture:
        capture = rtl_sdr.capture_iq(
            center_frequency_hz, duration_seconds,
            device_selector=receiver.get("device_selector", ""), **options,
        )
        return replace(capture, receiver_id=receiver["receiver_id"])

    def stream_iq_chunks(
        self, receiver: dict, center_frequency_hz: int, *, duration_seconds: float,
        stop_event: Event, **options,
    ) -> Iterator[np.ndarray]:
        yield from rtl_sdr.stream_iq_chunks(
            center_frequency_hz, duration_seconds=duration_seconds, stop_event=stop_event,
            device_selector=receiver.get("device_selector", ""), **options,
        )


_BACKENDS: dict[str, ReceiverBackend] = {
    "airspyhf": AirspyHfBackend(),
    "rtl_sdr": RtlSdrBackend(),
}


def register_backend(backend: ReceiverBackend) -> None:
    """Register a backend adapter. Primarily useful for optional packages and tests."""
    name = str(backend.name).strip().lower()
    if not name:
        raise ValueError("Receiver backend name is required")
    _BACKENDS[name] = backend


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def resolve_receiver(receiver_id: str | None = None) -> tuple[dict, ReceiverBackend]:
    receiver = (
        ensure_airspy_default() if receiver_id is None else get_receiver(receiver_id)
    )
    if not receiver.get("enabled", False):
        raise RuntimeError(f"Receiver {receiver['receiver_id']} is disabled")
    backend_name = receiver["backend"]
    backend = _BACKENDS.get(backend_name)
    if backend is None:
        raise NotImplementedError(
            f"Receiver backend {backend_name!r} is registered for planning but does not "
            "yet have a capture adapter"
        )
    return receiver, backend


def _lease(receiver: dict, owner: str | None, purpose: str) -> dict:
    return acquire_receiver(
        receiver["receiver_id"],
        owner or f"capture-{uuid4().hex[:12]}",
        purpose,
    )


def device_info(receiver_id: str | None = None) -> dict:
    receiver, backend = resolve_receiver(receiver_id)
    result = backend.device_info(receiver)
    result["calibration"] = get_calibration(receiver["receiver_id"], required=False)
    return result


def capture_iq(
    center_frequency_hz: int,
    duration_seconds: float,
    *,
    receiver_id: str | None = None,
    lease_owner: str | None = None,
    purpose: str = "IQ capture",
    required_bandwidth_hz: int = 0,
    preferred_role: str | None = None,
    **options,
) -> Capture:
    # Omitted IDs retain the established MCP default. Dashboard/API callers send
    # the versioned policy value "auto" to opt into automatic admission.
    if receiver_id is not None:
        assigned = assign_and_acquire_receiver(
            frequency_hz=center_frequency_hz, required_bandwidth_hz=required_bandwidth_hz,
            receiver_id=receiver_id, preferred_role=preferred_role,
            owner=lease_owner or f"capture-{uuid4().hex[:12]}", purpose=purpose,
            implemented_backends=set(_BACKENDS),
        )
        receiver = get_receiver(assigned["receiver_id"])
        backend = _BACKENDS[assigned["backend"]]
        lease = assigned["lease"]
    else:
        receiver, backend = resolve_receiver(receiver_id)
        lease = _lease(receiver, lease_owner, purpose)
    calibration = get_calibration(receiver["receiver_id"], required=False)
    if backend.name == "rtl_sdr" and calibration is not None:
        options.setdefault(
            "frequency_correction_ppm", round(calibration["frequency_correction_ppm"])
        )
    try:
        captured = backend.capture_iq(
            receiver, center_frequency_hz, duration_seconds, **options,
        )
        return replace(captured, calibration=calibration)
    finally:
        release_receiver(lease["lease_id"])


def stream_iq_chunks(
    center_frequency_hz: int,
    *,
    duration_seconds: float,
    stop_event: Event,
    receiver_id: str | None = None,
    lease_owner: str | None = None,
    purpose: str = "streaming IQ capture",
    required_bandwidth_hz: int = 0,
    preferred_role: str | None = None,
    **options,
) -> Iterator[np.ndarray]:
    if receiver_id is not None:
        assigned = assign_and_acquire_receiver(
            frequency_hz=center_frequency_hz, required_bandwidth_hz=required_bandwidth_hz,
            receiver_id=receiver_id, preferred_role=preferred_role,
            owner=lease_owner or f"stream-{uuid4().hex[:12]}", purpose=purpose,
            implemented_backends=set(_BACKENDS),
        )
        receiver = get_receiver(assigned["receiver_id"])
        backend = _BACKENDS[assigned["backend"]]
        lease = assigned["lease"]
    else:
        receiver, backend = resolve_receiver(receiver_id)
        lease = _lease(receiver, lease_owner, purpose)
    calibration = get_calibration(receiver["receiver_id"], required=False)
    if backend.name == "rtl_sdr" and calibration is not None:
        options.setdefault(
            "frequency_correction_ppm", round(calibration["frequency_correction_ppm"])
        )
    try:
        last_heartbeat = time.monotonic()
        chunks = backend.stream_iq_chunks(
            receiver, center_frequency_hz, duration_seconds=duration_seconds,
            stop_event=stop_event, **options,
        )
        for chunk in chunks:
            if time.monotonic() - last_heartbeat >= 30:
                heartbeat_receiver(lease["lease_id"])
                last_heartbeat = time.monotonic()
            yield chunk
    finally:
        release_receiver(lease["lease_id"])


def validate_frequency(frequency_hz: int, receiver_id: str | None = None) -> int:
    receiver, _backend = resolve_receiver(receiver_id)
    frequency_hz = int(frequency_hz)
    if not any(low <= frequency_hz <= high for low, high in receiver["tuning_ranges_hz"]):
        ranges = " or ".join(
            f"{low:,}-{high:,} Hz" for low, high in receiver["tuning_ranges_hz"]
        )
        raise ValueError(
            f"Frequency is outside receiver {receiver['receiver_id']} tuning ranges: {ranges}"
        )
    return frequency_hz


def validate_duration(duration_seconds: float) -> float:
    return airspyhf.validate_duration(duration_seconds)


def offset_capture_center(
    target_frequency_hz: int, offset_hz: int = 50_000,
    receiver_id: str | None = None,
) -> int:
    if receiver_id == "auto":
        # Final compatibility and range validation happens atomically when the
        # capture lease is acquired.
        return int(target_frequency_hz) + int(offset_hz)
    receiver, _backend = resolve_receiver(receiver_id)
    target_frequency_hz = validate_frequency(target_frequency_hz, receiver["receiver_id"])
    for low, high in receiver["tuning_ranges_hz"]:
        if low <= target_frequency_hz <= high:
            if target_frequency_hz + offset_hz <= high:
                return target_frequency_hz + offset_hz
            if target_frequency_hz - offset_hz >= low:
                return target_frequency_hz - offset_hz
    raise ValueError("Unable to choose a valid offset receiver center")


SAMPLE_RATE = airspyhf.SAMPLE_RATE
