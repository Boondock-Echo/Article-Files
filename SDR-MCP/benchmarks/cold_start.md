# Server cold-start benchmark

Measured on 2026-08-24 in the development container with CPython, using a fresh
process for each result. The command imports `rf_mcp.server`, measures elapsed
wall-clock time with `time.perf_counter()`, and records peak resident memory with
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.

| Revision | Import time | Peak RSS | Plotting/Pillow/SciPy/Skyfield loaded |
| --- | ---: | ---: | --- |
| Before (`e19b9b3`) | 5.624 s | 187,204 KiB | Yes (except Skyfield) |
| After | 2.356 s | 86,276 KiB | No |

These single-run measurements are intended as a reproducible cold-start
reference, not a microbenchmark. The import-smoke test is the regression gate:
core MCP readiness must not load Matplotlib, Pillow, SciPy, or Skyfield.

```bash
PYTHONPATH=src python - <<'PY'
import resource
import time

started = time.perf_counter()
import rf_mcp.server  # noqa: E402
print(f"seconds={time.perf_counter() - started:.3f} "
      f"rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
PY
```
