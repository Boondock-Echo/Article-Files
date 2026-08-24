from __future__ import annotations

import os
import shutil
from pathlib import Path

from .api_contract import API_VERSION
from .catalog import Catalog
from .config import DATA_DIR
from .receiver_backend import registered_backends
from .sdr_coordinator import discover_backends


def release_readiness(catalog: Catalog, *, probe_hardware: bool = False) -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "required": required,
                       "detail": detail})

    schema = catalog.schema_status()
    add("catalog_schema", schema["up_to_date"],
        f"schema {schema['current_version']} of {schema['supported_version']}")
    database = catalog.database_health()
    add("catalog_database", database["status"] == "healthy", database["status"])
    data_dir = Path(catalog.data_dir)
    add("data_directory", data_dir.exists() and os.access(data_dir, os.W_OK), str(data_dir))
    add("api_contract", API_VERSION == "1.0", f"API {API_VERSION}")
    add("capture_backend", bool(registered_backends()),
        ", ".join(registered_backends()))
    auth_enabled = bool(os.getenv("RF_MCP_API_TOKEN", "").strip())
    lan_bind = os.getenv("RF_MCP_HOST", "127.0.0.1") not in {"127.0.0.1", "localhost", "::1"}
    add("lan_authentication", not lan_bind or auth_enabled,
        "enabled" if auth_enabled else ("not required for loopback" if not lan_bind else "LAN bind without token"))
    tools = discover_backends(probe_hardware=probe_hardware)
    installed_capture = [item["backend"] for item in tools["backends"]
                         if item["backend"] in registered_backends() and item["installed"]]
    add("receiver_cli", bool(installed_capture),
        ", ".join(installed_capture) if installed_capture else "no supported receiver CLI detected",
        required=False)
    required_failures = [item for item in checks if item["required"] and not item["passed"]]
    warnings = [item for item in checks if not item["required"] and not item["passed"]]
    return {
        "schema": "rf-mcp.release-readiness.v1", "ready": not required_failures,
        "status": "ready" if not required_failures and not warnings else
                  ("ready_with_warnings" if not required_failures else "not_ready"),
        "checks": checks, "required_failure_count": len(required_failures),
        "warning_count": len(warnings), "probe_hardware": bool(probe_hardware),
    }
