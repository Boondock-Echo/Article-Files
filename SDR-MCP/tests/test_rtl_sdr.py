from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest

from rf_mcp import receiver_backend, rtl_sdr, sdr_coordinator


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rtl_sdr, "CAPTURE_DIR", tmp_path / "captures")
    monkeypatch.setattr(rtl_sdr, "ensure_data_dirs", lambda: (tmp_path / "captures").mkdir(exist_ok=True))
    sdr_coordinator._LEASES.clear()
    yield
    sdr_coordinator._LEASES.clear()


def save_rtl(device_selector="00000042"):
    return sdr_coordinator.save_receiver(
        receiver_id="rtl-one", name="RTL-SDR One", backend="rtl_sdr",
        role="vhf_uhf_monitor", device_selector=device_selector, verified=True,
        tuning_ranges_hz=[[24_000_000, 1_766_000_000]],
        max_bandwidth_hz=2_400_000,
    )


def test_cu8_conversion_maps_center_and_extremes():
    converted = rtl_sdr._convert_cu8(bytes([0, 127, 128, 255]))
    assert converted.dtype == np.float32
    assert converted[0] == pytest.approx(-1.0)
    assert converted[-1] == pytest.approx(1.0)
    assert converted[1] == pytest.approx(-1 / 255)
    assert converted[2] == pytest.approx(1 / 255)


def test_command_uses_serial_frequency_rate_gain_ppm_and_sample_count(monkeypatch):
    monkeypatch.setattr(rtl_sdr.shutil, "which", lambda name: f"/usr/bin/{name}")
    command = rtl_sdr._command(
        center_frequency_hz=145_800_000, sample_rate_hz=768_000,
        device_selector="00000042", sample_count=1234, output="capture.cu8",
        agc=False, gain_db=28.0, frequency_correction_ppm=-3,
    )
    assert command == [
        "/usr/bin/rtl_sdr", "-d", "00000042", "-f", "145800000",
        "-s", "768000", "-p", "-3", "-g", "28", "-n", "1234", "capture.cu8",
    ]


def test_device_selector_rejects_command_like_values():
    with pytest.raises(ValueError, match="device_selector"):
        rtl_sdr.validate_device_selector("0; touch /tmp/no")


def test_device_info_parses_rtl_test_output(monkeypatch):
    monkeypatch.setattr(rtl_sdr.shutil, "which", lambda name: f"/usr/bin/{name}")
    completed = type("Completed", (), {
        "returncode": 0, "stdout": b"",
        "stderr": b"Found 1 device(s):\nUsing device 0: Generic RTL2832U OEM\n",
    })()
    monkeypatch.setattr(rtl_sdr.subprocess, "run", lambda *args, **kwargs: completed)
    result = rtl_sdr.device_info("0")
    assert result["connected"] is True
    assert result["model"] == "0: Generic RTL2832U OEM"
    assert result["sample_format"] == "unsigned_8_bit_interleaved_iq"


def test_capture_converts_cu8_file_to_float32_iq(tmp_path, monkeypatch):
    monkeypatch.setattr(rtl_sdr.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **options):
        raw_path = Path(command[-1])
        requested = int(command[command.index("-n") + 1])
        raw_path.write_bytes(np.tile(np.array([0, 255], dtype=np.uint8), requested).tobytes())
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr(rtl_sdr.subprocess, "run", fake_run)
    capture = rtl_sdr.capture_iq(100_000_000, 0.25, sample_rate_hz=225_001)
    values = np.fromfile(capture.path, dtype="<f4")
    assert capture.backend == "rtl_sdr"
    assert capture.captured_samples == round(225_001 * 0.25)
    assert len(values) == capture.captured_samples * 2
    assert values[:2].tolist() == pytest.approx([-1.0, 1.0])


def test_stream_uses_stdout_and_yields_normalized_float32(monkeypatch):
    monkeypatch.setattr(rtl_sdr.shutil, "which", lambda name: f"/usr/bin/{name}")
    sample_count = round(225_001 * 0.1)
    payload = np.tile(np.array([0, 255], dtype=np.uint8), sample_count).tobytes()
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdout, self.stderr, self.returncode = io.BytesIO(payload), io.BytesIO(b""), 0

        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def wait(self, timeout=None): return self.returncode
        def kill(self): self.returncode = -9

    def fake_popen(command, **options):
        captured["command"], captured["options"] = command, options
        return FakeProcess()

    monkeypatch.setattr(rtl_sdr.subprocess, "Popen", fake_popen)
    chunks = list(rtl_sdr.stream_iq_chunks(
        145_800_000, duration_seconds=10, stop_event=threading.Event(),
        chunk_seconds=0.1, sample_rate_hz=225_001,
    ))
    assert captured["command"][-1] == "-"
    assert captured["options"]["bufsize"] == 0
    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert chunks[0][:2].tolist() == pytest.approx([-1.0, 1.0])


def test_backend_selects_rtl_and_releases_coordinator_lease(monkeypatch, tmp_path):
    save_rtl()
    observed = {}

    def fake_capture(center_frequency_hz, duration_seconds, **options):
        observed["lease"] = sdr_coordinator.get_receiver("rtl-one")["lease"]
        return rtl_sdr.Capture(
            path=tmp_path / "capture.iq", center_frequency_hz=center_frequency_hz,
            sample_rate_hz=768_000, requested_samples=100, captured_samples=100,
            started_at="2026-08-12T00:00:00+00:00", backend="rtl_sdr",
            device_selector=options["device_selector"],
        )

    monkeypatch.setattr(rtl_sdr, "capture_iq", fake_capture)
    capture = receiver_backend.capture_iq(
        500_000_000, 1, receiver_id="rtl-one", lease_owner="rtl-test",
    )
    assert observed["lease"]["owner"] == "rtl-test"
    assert capture.receiver_id == "rtl-one"
    assert capture.device_selector == "00000042"
    assert sdr_coordinator.coordinator_status()["active_lease_count"] == 0


def test_offset_center_uses_selected_receiver_range():
    save_rtl()
    assert receiver_backend.offset_capture_center(
        500_000_000, receiver_id="rtl-one",
    ) == 500_050_000
