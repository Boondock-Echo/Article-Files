# Live startup service targets

All latency measurements use the process-local monotonic clock exposed by the
live status endpoints. Frequencies and authentication data are intentionally
excluded from these results.

## Initial targets

* First waterfall row: no more than **500 ms after the first receiver IQ output**.
* First encoded audio chunk: no more than **1,000 ms after first IQ**. Browser
  playback time is also reported by the dashboard and is the audible-playback
  acceptance measurement.
* Live queue drops: **0** during an unloaded startup run.

## Backend results

Record at least ten cold starts and ten warm starts for each backend. Report
median, p95, maximum, and drop totals rather than combining unlike hardware.

| Backend | Run set | First row after IQ (median/p95/max ms) | Audible after IQ (median/p95/max ms) | Drops |
| --- | --- | --- | --- | --- |
| Airspy HF+ | Cold / warm | Pending hardware run | Pending hardware run | Pending |
| RTL-SDR | Cold / warm | Pending hardware run | Pending hardware run | Pending |

Hardware results are deliberately marked pending in source control: simulated
test timings are deterministic correctness checks, not representative device
benchmarks. Copy status snapshots immediately after each run and calculate each
duration only from timestamps produced by the same server process.
