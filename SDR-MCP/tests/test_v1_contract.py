from __future__ import annotations

from pathlib import Path
import sqlite3

from rf_mcp.api_contract import API_VERSION, STABLE_CORE_TOOLS, contract_document
from rf_mcp.catalog import Catalog
from rf_mcp import readiness


def test_v1_contract_is_machine_readable_and_additive_client_safe():
    contract = contract_document("1.0.0")
    assert API_VERSION == contract["api_version"] == "1.0"
    assert contract["schema"] == "SDR-MCP.api-contract.v1"
    assert contract["stability"] == "stable"
    assert contract["frequency_unit"] == "Hz"
    assert contract["compatibility_rules"]["client_rule"].startswith("Ignore unknown")
    assert contract["deprecation_policy"]["removal_in_minor_release"] is False
    for required in ("list_devices", "inspect_spectrum", "analyze_signal",
                     "get_rf_job", "get_rf_artifact"):
        assert required in STABLE_CORE_TOOLS


def test_release_readiness_passes_core_checks_without_hardware(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path, index_existing=False)
    monkeypatch.setenv("RF_MCP_HOST", "127.0.0.1")
    monkeypatch.delenv("RF_MCP_API_TOKEN", raising=False)
    monkeypatch.setattr(readiness, "registered_backends", lambda: ("airspyhf", "rtl_sdr"))
    monkeypatch.setattr(readiness, "discover_backends", lambda probe_hardware=False: {
        "backends": [{"backend": "airspyhf", "installed": False},
                     {"backend": "rtl_sdr", "installed": False}],
    })
    result = readiness.release_readiness(catalog)
    assert result["ready"] is True
    assert result["status"] == "ready_with_warnings"
    assert result["required_failure_count"] == 0
    assert result["probe_hardware"] is False


def test_release_readiness_rejects_unauthenticated_lan_bind(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path, index_existing=False)
    monkeypatch.setenv("RF_MCP_HOST", "0.0.0.0")
    monkeypatch.delenv("RF_MCP_API_TOKEN", raising=False)
    monkeypatch.setattr(readiness, "registered_backends", lambda: ("airspyhf",))
    monkeypatch.setattr(readiness, "discover_backends", lambda probe_hardware=False: {
        "backends": [{"backend": "airspyhf", "installed": True}],
    })
    result = readiness.release_readiness(catalog)
    assert result["ready"] is False
    assert result["status"] == "not_ready"
    failed = [item for item in result["checks"] if not item["passed"]]
    assert any(item["name"] == "lan_authentication" and item["required"] for item in failed)


def test_v069_catalog_upgrades_in_place_without_losing_jobs(tmp_path):
    database = tmp_path / "SDR-MCP.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, job_type TEXT NOT NULL, state TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}', summary_json TEXT,
                result_json_path TEXT, created_at TEXT NOT NULL, started_at TEXT,
                completed_at TEXT, error TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO jobs(job_id,job_type,state,created_at) VALUES (?,?,?,?)",
            ("v069-job", "spectrum_inspection", "completed",
             "2026-08-12T00:00:00+00:00"),
        )
        connection.execute("PRAGMA user_version=0")
    upgraded = Catalog(tmp_path, index_existing=False)
    assert upgraded.get_job("v069-job")["state"] == "completed"
    assert upgraded.schema_status()["current_version"] == 2
