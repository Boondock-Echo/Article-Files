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

## Acceptance thresholds

Mocked CI uses deterministic tone input and enforces: headers **≤100 ms**, first
waterfall data **≤500 ms**, first audio data **≤1,500 ms**, inter-chunk gap
**≤500 ms**, and stop/lease release **≤500 ms**. Hardware smoke tests are
opt-in (`RF_MCP_HARDWARE_SMOKE=1`) because scheduler and USB startup variance is
real: headers **≤250 ms**, first waterfall **≤2,000 ms**, first audio **≤4,000
ms**, inter-chunk gap **≤1,500 ms**, and stop/lease release **≤2,000 ms**.

Every saved result must include backend manufacturer/model, configured and
observed sample rate, host load averages, direct/proxy/remote stage, and proxy
topology. Never compare a mocked result with a hardware result.

Run `scripts/live-diagnostic-matrix.py` with the direct application URL, the
production proxy URL, and a browser-reachable URL before and after a change.
Rank the deltas for header, receiver startup, first IQ, first encoded output or
row, proxy transit, and browser play/render; optimize the largest measured stage.

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
