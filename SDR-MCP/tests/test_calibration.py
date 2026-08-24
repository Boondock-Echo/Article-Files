from __future__ import annotations

import pytest

from rf_mcp import calibration, receiver_backend, sdr_coordinator
from rf_mcp.airspyhf import Capture


@pytest.fixture(autouse=True)
def isolated_calibrations(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    monkeypatch.setattr(calibration, "ensure_data_dirs", lambda: tmp_path.mkdir(exist_ok=True))
    sdr_coordinator._LEASES.clear()
    yield
    sdr_coordinator._LEASES.clear()


def rtl_receiver():
    return sdr_coordinator.save_receiver(
        receiver_id="rtl-cal", name="Calibrated RTL", backend="rtl_sdr",
        role="general", device_selector="CAL001", verified=True,
    )


def test_calibration_round_trip_requires_power_provenance():
    rtl_receiver()
    with pytest.raises(ValueError, match="reference_source"):
        calibration.save_calibration(
            receiver_id="rtl-cal", dbfs_to_dbm_offset_db=-42,
        )
    saved = calibration.save_calibration(
        receiver_id="rtl-cal", frequency_correction_ppm=1.75,
        dbfs_to_dbm_offset_db=-42.5, reference_frequency_hz=100_000_000,
        reference_source="-60 dBm laboratory signal generator",
    )
    assert saved["receiver_backend"] == "rtl_sdr"
    assert calibration.get_calibration("rtl-cal")["frequency_correction_ppm"] == 1.75
    assert calibration.list_calibrations() == [saved]
    with pytest.raises(ValueError, match="confirm_delete"):
        calibration.delete_calibration("rtl-cal")
    assert calibration.delete_calibration("rtl-cal", confirm_delete=True)["deleted"] is True


def test_rtl_capture_automatically_receives_saved_ppm(monkeypatch, tmp_path):
    rtl_receiver()
    saved = calibration.save_calibration(
        receiver_id="rtl-cal", frequency_correction_ppm=3.2,
        reference_source="frequency counter",
    )
    observed = {}

    def fake_capture(receiver, center_frequency_hz, duration_seconds, **options):
        observed.update(options)
        return Capture(
            path=tmp_path / "capture.iq", center_frequency_hz=center_frequency_hz,
            sample_rate_hz=768_000, requested_samples=10, captured_samples=10,
            started_at="2026-08-13T00:00:00+00:00", receiver_id=receiver["receiver_id"],
            backend="rtl_sdr",
        )

    monkeypatch.setattr(receiver_backend._BACKENDS["rtl_sdr"], "capture_iq", fake_capture)
    capture = receiver_backend.capture_iq(100_000_000, 1, receiver_id="rtl-cal")
    assert observed["frequency_correction_ppm"] == 3
    assert capture.calibration == saved


def test_frequency_only_profile_does_not_claim_dbm_calibration():
    rtl_receiver()
    saved = calibration.save_calibration(
        receiver_id="rtl-cal", frequency_correction_ppm=-2,
        reference_source="WWV carrier comparison",
    )
    assert saved["dbfs_to_dbm_offset_db"] is None
