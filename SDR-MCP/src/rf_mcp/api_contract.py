from __future__ import annotations

API_VERSION = "1.0"
API_STABILITY = "stable"

# These names form the compatibility-guaranteed v1 core. Specialized tools remain
# supported but may gain optional fields and capabilities in minor releases.
STABLE_CORE_TOOLS = (
    "get_rf_api_contract",
    "get_server_health",
    "get_release_readiness",
    "list_devices",
    "discover_attached_sdr_devices",
    "add_discovered_sdr_device",
    "list_sdr_receivers",
    "get_sdr_receiver",
    "inspect_spectrum",
    "analyze_signal",
    "receive_broadcast_fm",
    "qualify_sdr_receiver",
    "save_receiver_calibration",
    "get_receiver_calibration",
    "list_receiver_calibrations",
    "list_rf_jobs",
    "get_rf_job",
    "list_rf_artifacts",
    "get_rf_artifact",
    "get_storage_status",
    "get_rf_recovery_status",
)

COMPATIBILITY_RULES = {
    "major": "May remove or rename stable tools, required parameters, or documented fields.",
    "minor": "May add tools, optional parameters, enum values, and response fields.",
    "patch": "Bug and security fixes without intentional public-contract changes.",
    "client_rule": "Ignore unknown response fields and handle documented error types.",
}


def contract_document(server_version: str) -> dict:
    return {
        "schema": "rf-mcp.api-contract.v1",
        "api_version": API_VERSION,
        "server_version": server_version,
        "stability": API_STABILITY,
        "frequency_unit": "Hz",
        "time_standard": "UTC ISO 8601",
        "measurement_policy": {
            "default": "relative digital-domain levels",
            "dbm_claim_requires": "receiver calibration with documented reference_source",
        },
        "stable_core_tools": list(STABLE_CORE_TOOLS),
        "compatibility_rules": COMPATIBILITY_RULES,
        "deprecation_policy": {
            "minimum_notice": "one minor release before removal in the next major release",
            "removal_in_minor_release": False,
        },
    }
