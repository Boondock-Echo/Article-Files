from __future__ import annotations

import os
from pathlib import Path

SAMPLE_RATE = 768_000
MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = float(os.getenv("RF_MCP_MAX_DURATION", "10"))
MAX_PEAKS = 50

DATA_DIR = Path(os.getenv("RF_MCP_DATA_DIR", Path.home() / "rf-mcp-data"))
CAPTURE_DIR = DATA_DIR / "captures"
PLOT_DIR = DATA_DIR / "plots"
RESULT_DIR = DATA_DIR / "results"
AUDIO_DIR = DATA_DIR / "audio"
FM_SURVEY_DIR = DATA_DIR / "fm-surveys"
WEAK_SIGNAL_DIR = DATA_DIR / "weak-signal"
FLDIGI_DIR = DATA_DIR / "fldigi"
SSTV_DIR = DATA_DIR / "sstv"
SATELLITE_DIR = DATA_DIR / "satellite"

AIRSPYHF_RX = os.getenv("RF_MCP_AIRSPYHF_RX", "airspyhf_rx")
AIRSPYHF_INFO = os.getenv("RF_MCP_AIRSPYHF_INFO", "airspyhf_info")

# The Airspy HF+ has a gap between its HF and VHF ranges.
TUNING_RANGES_HZ = ((9_000, 31_000_000), (60_000_000, 260_000_000))


def ensure_data_dirs() -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FM_SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    WEAK_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    FLDIGI_DIR.mkdir(parents=True, exist_ok=True)
    SSTV_DIR.mkdir(parents=True, exist_ok=True)
    SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
