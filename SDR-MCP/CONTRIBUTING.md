# Contributing

RF MCP targets Python 3.11 or newer on Debian ARM64. Hardware-independent tests
must pass without an SDR attached.

Before submitting a change:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . pytest
python -m compileall -q src tests
pytest -q
```

Changes to a stable core tool must preserve existing required parameters and
documented response fields throughout the 1.x line. Additive optional fields are
allowed. Any proposed breaking change belongs in the next major version and
requires migration documentation.

DSP changes should include deterministic synthetic-signal tests. Receiver
backend changes should include command construction, sample conversion, cleanup,
and lease tests. Clearly separate mocked tests from live hardware results.
