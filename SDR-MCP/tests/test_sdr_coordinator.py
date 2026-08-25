from __future__ import annotations

import threading

import pytest

from rf_mcp import sdr_coordinator as coordinator
from rf_mcp import receiver_backend
from rf_mcp.airspyhf import Capture


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "DATA_DIR", tmp_path)
    coordinator._LEASES.clear()
    original_backends = dict(receiver_backend._BACKENDS)
    yield
    coordinator._LEASES.clear()
    receiver_backend._BACKENDS.clear()
    receiver_backend._BACKENDS.update(original_backends)


def receiver(receiver_id="air", **overrides):
    values = dict(receiver_id=receiver_id, name=receiver_id, backend="airspyhf",
                  role="primary_hf", verified=True, priority=50)
    values.update(overrides)
    return coordinator.save_receiver(**values)


def test_registry_round_trip_and_replace_preserves_created_at():
    first = receiver(notes="first")
    second = receiver(notes="second", priority=90)
    assert second["created_at"] == first["created_at"]
    assert coordinator.get_receiver("air")["notes"] == "second"
    assert coordinator.list_receivers()[0]["priority"] == 90


def test_validation_rejects_bad_ranges_and_names():
    with pytest.raises(ValueError, match="receiver_id"):
        receiver("NOT VALID")
    with pytest.raises(ValueError, match="tuning range"):
        receiver(tuning_ranges_hz=[[100, 100]])


def test_plan_prefers_matching_role_then_priority():
    receiver("primary", priority=90)
    receiver("sat", backend="rtl_sdr", role="satellite", priority=70,
             tuning_ranges_hz=[[24_000_000, 1_700_000_000]], max_bandwidth_hz=2_400_000)
    result = coordinator.plan_assignment(frequency_hz=145_800_000,
                                         required_bandwidth_hz=20_000,
                                         preferred_role="satellite")
    assert result["selected"]["receiver_id"] == "sat"
    assert result["dry_run"] is True


def test_plan_explains_rejections_and_verification_override():
    receiver("disabled", enabled=False)
    receiver("wide", backend="hackrf", verified=False, max_bandwidth_hz=20_000_000,
             tuning_ranges_hz=[[1_000_000, 6_000_000_000]])
    strict = coordinator.plan_assignment(frequency_hz=500_000_000,
                                         required_bandwidth_hz=5_000_000)
    assert strict["selected"] is None
    assert any("not verified" in x["reasons"] for x in strict["rejected"])
    relaxed = coordinator.plan_assignment(frequency_hz=500_000_000,
                                          required_bandwidth_hz=5_000_000,
                                          require_verified=False)
    assert relaxed["selected"]["receiver_id"] == "wide"


def test_same_receiver_cannot_be_leased_twice():
    receiver()
    lease = coordinator.acquire_receiver("air", "job-one")
    with pytest.raises(RuntimeError, match="job-one"):
        coordinator.acquire_receiver("air", "job-two")
    assert coordinator.release_receiver(lease["lease_id"])["released"] is True


def test_lease_survives_process_memory_reset_and_blocks_second_claim():
    receiver()
    lease = coordinator.acquire_receiver("air", "first-process")
    coordinator._LEASES.clear()  # Simulate a fresh interpreter with the same data directory.
    assert coordinator.get_receiver("air")["lease"]["lease_id"] == lease["lease_id"]
    with pytest.raises(RuntimeError, match="first-process"):
        coordinator.acquire_receiver("air", "second-process")
    coordinator.release_receiver(lease["lease_id"])


def test_expired_durable_lease_is_reclaimed():
    receiver()
    first = coordinator.acquire_receiver("air", "crashed-process")
    with coordinator._lease_connection() as connection:
        connection.execute(
            "UPDATE receiver_leases SET expires_at='2000-01-01T00:00:00+00:00' "
            "WHERE lease_id=?", (first["lease_id"],),
        )
    second = coordinator.acquire_receiver("air", "replacement-process")
    assert second["lease_id"] != first["lease_id"]
    coordinator.release_receiver(second["lease_id"])


def test_lease_heartbeat_extends_expiration():
    receiver()
    lease = coordinator.acquire_receiver("air", "stream")
    renewed = coordinator.heartbeat_receiver(lease["lease_id"])
    assert renewed["heartbeat_at"] >= lease["heartbeat_at"]
    assert renewed["expires_at"] >= lease["expires_at"]
    coordinator.release_receiver(lease["lease_id"])


def test_different_receivers_can_be_leased_concurrently():
    receiver("one")
    receiver("two")
    barrier = threading.Barrier(3)
    leases = []

    def claim(identifier):
        leases.append(coordinator.acquire_receiver(identifier, identifier))
        barrier.wait()
        barrier.wait()

    threads = [threading.Thread(target=claim, args=(identifier,)) for identifier in ("one", "two")]
    for thread in threads: thread.start()
    barrier.wait()
    assert coordinator.coordinator_status()["active_lease_count"] == 2
    barrier.wait()
    for thread in threads: thread.join()
    for lease in leases: coordinator.release_receiver(lease["lease_id"])


def test_atomic_auto_assigns_concurrent_jobs_to_different_receivers():
    receiver("one")
    receiver("two")
    barrier = threading.Barrier(3)
    assignments = []

    def admit(owner):
        barrier.wait()
        assignments.append(coordinator.assign_and_acquire_receiver(
            frequency_hz=10_000_000, required_bandwidth_hz=100_000,
            receiver_id="auto", owner=owner, implemented_backends={"airspyhf"},
        ))

    threads = [threading.Thread(target=admit, args=(owner,)) for owner in ("job-one", "job-two")]
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join()
    assert {item["receiver_id"] for item in assignments} == {"one", "two"}
    for item in assignments: coordinator.release_receiver(item["lease"]["lease_id"])


def test_atomic_assignment_rejects_range_bandwidth_and_explicit_fallback():
    receiver("narrow", tuning_ranges_hz=[[1_000_000, 2_000_000]], max_bandwidth_hz=10_000)
    receiver("compatible", tuning_ranges_hz=[[9_000_000, 11_000_000]], max_bandwidth_hz=500_000)
    with pytest.raises(RuntimeError, match="frequency outside tuning ranges.*required bandwidth too wide"):
        coordinator.assign_and_acquire_receiver(
            frequency_hz=10_000_000, required_bandwidth_hz=100_000,
            receiver_id="narrow", owner="pinned",
        )
    # Pinning never silently selects the other compatible receiver.
    assert coordinator.coordinator_status()["active_lease_count"] == 0


def test_atomic_auto_ignores_leased_higher_ranked_receiver():
    receiver("preferred", priority=100)
    receiver("idle", priority=10)
    held = coordinator.acquire_receiver("preferred", "existing")
    assigned = coordinator.assign_and_acquire_receiver(
        frequency_hz=10_000_000, receiver_id="auto", owner="new-job",
    )
    assert assigned["receiver_id"] == "idle"
    coordinator.release_receiver(assigned["lease"]["lease_id"])
    coordinator.release_receiver(held["lease_id"])


def test_delete_requires_confirmation_and_rejects_active_lease():
    receiver()
    with pytest.raises(ValueError, match="confirm_delete"):
        coordinator.delete_receiver("air")
    lease = coordinator.acquire_receiver("air", "test")
    with pytest.raises(RuntimeError, match="leased"):
        coordinator.delete_receiver("air", confirm_delete=True)
    coordinator.release_receiver(lease["lease_id"])
    assert coordinator.delete_receiver("air", confirm_delete=True)["deleted"] is True


def test_discovery_default_does_not_run_commands(monkeypatch):
    monkeypatch.setattr(coordinator.shutil, "which", lambda name: f"/usr/bin/{name}" if name else None)
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *args, **kwargs: pytest.fail("ran probe"))
    result = coordinator.discover_backends()
    assert result["safe_discovery"] is True
    assert next(x for x in result["backends"] if x["backend"] == "airspyhf")["installed"] is True


def test_default_airspy_is_idempotent():
    coordinator.ensure_airspy_default()
    coordinator.ensure_airspy_default()
    assert len(coordinator.list_receivers()) == 1


def test_guided_discovery_returns_registration_ready_rtl_devices(monkeypatch):
    monkeypatch.setattr(coordinator.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **options):
        if command[0].endswith("airspyhf_info"):
            return type("Completed", (), {
                "returncode": 0, "stdout": "AirSpy HF+\nS/N: ABC123\n", "stderr": "",
            })()
        return type("Completed", (), {
            "returncode": 1, "stdout": "",
            "stderr": ("Found 2 device(s):\n"
                       "  0:  Realtek, RTL2838UHIDIR, SN: 00000042\n"
                       "  1:  Generic, RTL2832U OEM, SN: 00000043\n"),
        })()

    monkeypatch.setattr(coordinator.subprocess, "run", fake_run)
    result = coordinator.discover_devices()
    assert result["device_count"] == 3
    rtl = [item for item in result["devices"] if item["backend"] == "rtl_sdr"]
    assert [item["device_selector"] for item in rtl] == ["00000042", "00000043"]
    assert rtl[0]["suggested_receiver_id"] == "rtl-sdr-00000042"
    assert rtl[0]["suggested_role"] == "vhf_uhf_monitor"
    assert result["writes_registry"] is False


def test_guided_registration_requires_current_discovery(monkeypatch):
    discovered = {"devices": [{
        "backend": "rtl_sdr", "device_selector": "00000042",
        "display_name": "RTL-SDR", "suggested_receiver_id": "rtl-sdr-00000042",
        "suggested_name": "RTL-SDR 00000042", "suggested_role": "vhf_uhf_monitor",
        "verified": True, "already_registered": False,
    }], "device_count": 1, "diagnostics": [], "writes_registry": False}
    monkeypatch.setattr(coordinator, "discover_devices", lambda: discovered)
    result = coordinator.register_discovered_device(
        backend="rtl_sdr", device_selector="00000042", receiver_id="rtl-vhf",
        name="VHF receiver", role="vhf_uhf_monitor", priority=85,
    )
    assert result["registered"] is True
    assert result["receiver"]["verified"] is True
    assert result["receiver"]["device_selector"] == "00000042"
    with pytest.raises(ValueError, match="no longer attached"):
        coordinator.register_discovered_device(
            backend="rtl_sdr", device_selector="missing", receiver_id="missing",
            name="Missing", role="general",
        )


def test_backend_capture_holds_lease_and_records_receiver(tmp_path):
    observed = {}

    class FakeBackend:
        name = "airspyhf"

        def device_info(self, selected):
            return {"receiver_id": selected["receiver_id"]}

        def capture_iq(self, selected, center_frequency_hz, duration_seconds, **options):
            observed["lease"] = coordinator.get_receiver(selected["receiver_id"])["lease"]
            return Capture(
                path=tmp_path / "capture.iq", center_frequency_hz=center_frequency_hz,
                sample_rate_hz=768_000, requested_samples=768_000,
                captured_samples=768_000, started_at="2026-08-12T00:00:00+00:00",
                receiver_id=selected["receiver_id"], backend=self.name,
            )

        def stream_iq_chunks(self, selected, center_frequency_hz, **options):
            yield b"chunk"

    receiver("air")
    receiver_backend.register_backend(FakeBackend())
    capture = receiver_backend.capture_iq(
        10_000_000, 1, receiver_id="air", lease_owner="test-job",
    )
    assert observed["lease"]["owner"] == "test-job"
    assert capture.receiver_id == "air"
    assert capture.backend == "airspyhf"
    assert coordinator.coordinator_status()["active_lease_count"] == 0


def test_backend_stream_releases_lease_when_generator_closes():
    class FakeBackend:
        name = "airspyhf"

        def device_info(self, selected):
            return {}

        def capture_iq(self, selected, center_frequency_hz, duration_seconds, **options):
            raise AssertionError("not used")

        def stream_iq_chunks(self, selected, center_frequency_hz, **options):
            assert coordinator.get_receiver(selected["receiver_id"])["lease"] is not None
            yield b"first"
            yield b"second"

    receiver("air")
    receiver_backend.register_backend(FakeBackend())
    stream = receiver_backend.stream_iq_chunks(
        10_000_000, duration_seconds=10, stop_event=threading.Event(), receiver_id="air",
    )
    assert next(stream) == b"first"
    assert coordinator.coordinator_status()["active_lease_count"] == 1
    stream.close()
    assert coordinator.coordinator_status()["active_lease_count"] == 0


def test_planned_backend_without_adapter_fails_explicitly():
    receiver(
        "hackrf", backend="hackrf", tuning_ranges_hz=[[1_000_000, 6_000_000_000]],
        max_bandwidth_hz=20_000_000,
    )
    with pytest.raises(NotImplementedError, match="does not yet have a capture adapter"):
        receiver_backend.resolve_receiver("hackrf")
