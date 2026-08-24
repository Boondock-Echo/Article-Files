from __future__ import annotations

import pytest
from types import SimpleNamespace

from rf_mcp import station_memory


@pytest.fixture(autouse=True)
def isolated_memories(tmp_path, monkeypatch):
    monkeypatch.setattr(station_memory, "DATA_DIR", tmp_path)


def test_save_list_get_replace_and_delete():
    first = station_memory.save(name="WWV 10", frequency_hz=10_000_000,
                                mode="am", tags=["Time", " time "], notes="reference")
    assert first["bandwidth_hz"] == 10_000
    assert first["tags"] == ["time"]
    assert station_memory.get("wwv 10")["memory_id"] == first["memory_id"]
    with pytest.raises(ValueError, match="already exists"):
        station_memory.save(name="WWV 10", frequency_hz=10_000_000, mode="am")
    replaced = station_memory.save(name="WWV 10", frequency_hz=15_000_000,
                                   mode="am", replace_existing=True)
    assert replaced["memory_id"] == first["memory_id"]
    assert replaced["created_at"] == first["created_at"]
    with pytest.raises(ValueError, match="confirm_delete"):
        station_memory.delete(first["memory_id"])
    assert station_memory.delete(first["memory_id"], confirm_delete=True)["deleted"]


def test_broadcast_fm_normalization_and_search():
    station_memory.save(name="Local FM", frequency_hz=100_100_000,
                        mode="broadcast_fm", bandwidth_hz=1, tags=["music"])
    station_memory.save(name="Disabled", frequency_hz=7_100_000,
                        mode="lsb", enabled=False)
    item = station_memory.list_memories(query="music")[0]
    assert item["bandwidth_hz"] == 200_000
    assert station_memory.list_memories(mode="broadcast_fm")[0]["name"] == "Local FM"
    assert len(station_memory.list_memories(enabled_only=True)) == 1


def test_validation_rejects_bad_frequency_mode_and_bandwidth():
    with pytest.raises(ValueError, match="tuning ranges"):
        station_memory.save(name="Gap", frequency_hz=40_000_000, mode="am")
    with pytest.raises(ValueError, match="mode must"):
        station_memory.save(name="Bad", frequency_hz=10_000_000, mode="wfm")
    with pytest.raises(ValueError, match="bandwidth"):
        station_memory.save(name="Wide CW", frequency_hz=7_000_000,
                            mode="cw", bandwidth_hz=10_000)


def test_receive_station_memory_routes_generic_mode_safely(monkeypatch):
    from rf_mcp import server
    memory = {"memory_id": "mem-1", "name": "Forty", "frequency_hz": 7_100_000,
              "mode": "lsb", "bandwidth_hz": 3_000, "enabled": True}
    calls = []
    monkeypatch.setattr(server, "get_station_memory_record", lambda value: memory)
    monkeypatch.setattr(server, "analyze_signal",
                        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
                            structuredContent={"job_id": "analyze-1"}, content=[]))
    result = server.receive_station_memory("Forty", duration_seconds=5, include_media=False)
    assert calls[0]["mode"] == "lsb"
    assert calls[0]["bandwidth_hz"] == 3_000
    assert calls[0]["retain_iq"] is False
    assert calls[0]["include_audio"] is False
    assert result.structuredContent["station_memory"]["memory_id"] == "mem-1"


def test_receive_station_memory_routes_broadcast_fm_and_rejects_disabled(monkeypatch):
    from rf_mcp import server
    memory = {"memory_id": "mem-fm", "name": "FM", "frequency_hz": 100_100_000,
              "mode": "broadcast_fm", "bandwidth_hz": 200_000, "enabled": True}
    calls = []
    monkeypatch.setattr(server, "get_station_memory_record", lambda value: memory)
    monkeypatch.setattr(server, "receive_broadcast_fm",
                        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
                            structuredContent={"job_id": "wfm-1"}, content=[]))
    server.receive_station_memory("FM", duration_seconds=10)
    assert calls[0]["decode_rds_data"] is True
    assert calls[0]["retain_iq"] is False
    with pytest.raises(ValueError, match="5 or 10"):
        server.receive_station_memory("FM", duration_seconds=6)
    memory["enabled"] = False
    with pytest.raises(ValueError, match="disabled"):
        server.receive_station_memory("FM")


def test_scan_station_memories_filters_tag_and_links_child_jobs(tmp_path, monkeypatch):
    from rf_mcp import server
    memories = [
        {"memory_id": "mem-a", "name": "A", "frequency_hz": 10_000_000,
         "mode": "am", "bandwidth_hz": 10_000, "enabled": True, "tags": ["time"]},
        {"memory_id": "mem-b", "name": "B", "frequency_hz": 7_100_000,
         "mode": "lsb", "bandwidth_hz": 3_000, "enabled": True, "tags": ["ham"]},
    ]
    calls = []
    monkeypatch.setattr(server, "list_memories", lambda **kwargs: memories)
    monkeypatch.setattr(server, "receive_station_memory",
                        lambda value, **kwargs: calls.append((value, kwargs)) or SimpleNamespace(
                            structuredContent={"job_id": f"child-{value}", "metrics": {}}))
    monkeypatch.setattr(server, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(server, "_persist_one_shot", lambda **kwargs: None)
    result = server.scan_station_memories(tag="ham", duration_seconds=5)
    assert [item[0] for item in calls] == ["mem-b"]
    assert result["completed_count"] == 1
    assert result["observations"][0]["job_id"] == "child-mem-b"
    assert calls[0][1]["include_media"] is False


def test_scan_station_memories_continues_or_stops_on_error(tmp_path, monkeypatch):
    from rf_mcp import server
    memories = [
        {"memory_id": value, "name": value, "frequency_hz": 10_000_000,
         "mode": "am", "bandwidth_hz": 10_000, "enabled": True, "tags": []}
        for value in ("one", "two", "three")
    ]
    monkeypatch.setattr(server, "list_memories", lambda **kwargs: memories)
    def receive(value, **kwargs):
        if value == "two":
            raise RuntimeError("receiver problem")
        return SimpleNamespace(structuredContent={"job_id": f"child-{value}"})
    monkeypatch.setattr(server, "receive_station_memory", receive)
    monkeypatch.setattr(server, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(server, "_persist_one_shot", lambda **kwargs: None)
    continued = server.scan_station_memories(duration_seconds=5, stop_on_error=False)
    stopped = server.scan_station_memories(duration_seconds=5, stop_on_error=True)
    assert continued["attempted_count"] == 3 and continued["failed_count"] == 1
    assert stopped["attempted_count"] == 2 and stopped["failed_count"] == 1
    assert stopped["state"] == "partial"


def test_scan_station_memories_enforces_time_and_broadcast_duration(monkeypatch):
    from rf_mcp import server
    with pytest.raises(ValueError, match="120-second"):
        server.scan_station_memories(duration_seconds=10, max_memories=20)
    fm = {"memory_id": "fm", "name": "FM", "frequency_hz": 100_100_000,
          "mode": "broadcast_fm", "bandwidth_hz": 200_000,
          "enabled": True, "tags": []}
    monkeypatch.setattr(server, "list_memories", lambda **kwargs: [fm])
    with pytest.raises(ValueError, match="5 or 10"):
        server.scan_station_memories(duration_seconds=2)


def test_station_memory_scan_preset_validation():
    from rf_mcp.presets import normalize_preset
    _, preset_type, _, config = normalize_preset(
        name="Hourly ham", preset_type="station_memory_scan", description="",
        config={"tag": "Ham", "duration_seconds": 5, "max_memories": 12,
                "compare_previous": True, "snr_change_threshold_db": 4},
    )
    assert preset_type == "station_memory_scan"
    assert config["tag"] == "ham"
    assert config["snr_change_threshold_db"] == 4
    with pytest.raises(ValueError, match="120-second"):
        normalize_preset(name="Too long", preset_type="station_memory_scan",
                         description="", config={"duration_seconds": 10, "max_memories": 20})


def test_station_memory_scan_detects_state_snr_and_rds_changes(tmp_path, monkeypatch):
    from rf_mcp import server
    memories = [
        {"memory_id": "a", "name": "A", "frequency_hz": 10_000_000,
         "mode": "am", "bandwidth_hz": 10_000, "enabled": True, "tags": []},
        {"memory_id": "fm", "name": "FM", "frequency_hz": 100_100_000,
         "mode": "broadcast_fm", "bandwidth_hz": 200_000, "enabled": True, "tags": []},
    ]
    prior = {"observations": [
        {"memory_id": "a", "state": "completed", "metrics": {"estimated_snr_db": 2}},
        {"memory_id": "fm", "state": "completed", "metrics": {},
         "rds": {"station": {"program_service": "OLD", "radiotext": "Before"}}},
    ]}
    class FakeCatalog:
        def list_jobs(self, **kwargs): return [{"job_id": "previous"}]
        def get_job(self, job_id): return {"result": prior}
    monkeypatch.setattr(server, "catalog", FakeCatalog())
    monkeypatch.setattr(server, "list_memories", lambda **kwargs: memories)
    def receive(value, **kwargs):
        if value == "a":
            return SimpleNamespace(structuredContent={"job_id": "new-a",
                                                       "metrics": {"estimated_snr_db": 10}})
        return SimpleNamespace(structuredContent={"job_id": "new-fm", "metrics": {},
            "rds": {"station": {"program_service": "NEW", "radiotext": "After"}}})
    monkeypatch.setattr(server, "receive_station_memory", receive)
    monkeypatch.setattr(server, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(server, "_persist_one_shot", lambda **kwargs: None)
    result = server.scan_station_memories(duration_seconds=5, snr_change_threshold_db=6)
    assert result["compared_to_job_id"] == "previous"
    assert {item["kind"] for item in result["changes"]} == {
        "snr_changed", "rds_program_service_changed", "rds_radiotext_changed"}
