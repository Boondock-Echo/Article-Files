from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_server_readiness_does_not_load_optional_heavy_libraries() -> None:
    result = _isolated_python(
        """
import sys
import rf_mcp.server
for prefix in ('matplotlib', 'PIL', 'scipy', 'skyfield'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    )
    assert result.returncode == 0, result.stderr


def test_plotting_and_scipy_load_only_when_representative_features_run() -> None:
    result = _isolated_python(
        """
import sys
from rf_mcp.activity import save_activity_plot
from rf_mcp.lazy_imports import find_peaks
assert 'matplotlib.pyplot' not in sys.modules
assert 'scipy.signal' not in sys.modules
find_peaks([0.0, 1.0, 0.0])
assert 'scipy.signal' in sys.modules
try:
    save_activity_plot({'hourly': [], 'persistent_signals': []}, [])
except (KeyError, ValueError):
    pass
assert 'matplotlib.pyplot' in sys.modules
import matplotlib
assert matplotlib.get_backend().lower() == 'agg'
"""
    )
    assert result.returncode == 0, result.stderr
