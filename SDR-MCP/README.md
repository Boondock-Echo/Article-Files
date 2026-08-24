# RF Lab MCP for Airspy HF+ and RTL-SDR

A receive-only Model Context Protocol server for spectrum inspection with
Airspy HF+ and RTL-SDR receivers on Debian Linux. Version 1.0.1 corrects the
dashboard content-security policy so its authenticated high-contrast stylesheet
loads reliably. Version 1.0.0 declared the
stable API 1.0 core contract, semantic compatibility and deprecation policies,
production-readiness checks, CI across Python 3.11–3.13, and security and
contributor guidance. It preserves the established v0.69 receiver, calibration,
job, artifact, dashboard, decoder, and satellite behavior. Version 0.69.0 added persistent
per-receiver calibration profiles, automatic RTL-SDR PPM correction, optional
traceable dBFS-to-dBm conversion, calibration provenance in device and spectrum
results, and a short hardware qualification capture with length and overload
checks. Results never claim dBm calibration unless a documented conversion
offset is present. Version 0.68.0 introduced a
shared application-service boundary used by both MCP and browser adapters for
receiver onboarding, recovery reporting, spectrum capture, signal analysis, and
broadcast-FM reception. The dashboard now delivers its stylesheet and JavaScript
as authenticated assets instead of one inline script, allowing a stricter
`script-src 'self'` content-security policy and independent frontend evolution.
Version 0.67.0 made receiver
coordination restart-safe and multi-process-safe with SQLite-backed leases,
automatic expiration and streaming heartbeats. It records an explicit catalog
schema version and migration history, includes stopping jobs in startup recovery,
and adds `get_rf_recovery_status` for operational verification. Version 0.66.0 added guided
receiver onboarding in the dashboard: scan attached hardware, review a
registration-ready card, choose a name and role, and add the verified receiver
without copying serial numbers or writing MCP JSON. The same workflow is
available through `discover_attached_sdr_devices` and
`add_discovered_sdr_device`. Version 0.65.0 added a
functional RTL-SDR backend with index or serial selection, safe command
construction, configurable sample rate, automatic or manual gain, PPM
correction, CU8-to-normalized-float IQ conversion, capture and streaming
support, and receiver-aware offset tuning beyond the Airspy frequency range.
Version 0.64.0 introduced a receiver-backend
contract and routes established capture and streaming workflows through the
registered Airspy HF+ adapter. Receiver leases now cover real hardware access,
capture metadata identifies the selected receiver and backend, and unsupported
planned backends fail explicitly instead of silently using the Airspy. Version 0.63.0 adds persistent per-step scan
checkpoints, shared live job cards with phase, elapsed time, and ETA, a visual
band-scan position display, and direct plot, waterfall, image, and audio result
previews. Version 0.62.0 added immediate dashboard action feedback with busy
labels, spinners, and status toasts; active RF jobs refresh more frequently; and
FM surveys render a live channel map for scanned, candidate, and decoded
frequencies. Version 0.61.0 improved UTC weak-signal capture with
an early-start guard and decoder preroll, and produces per-cycle FT8/FT4/WSPR
audio diagnostics plus waveform/waterfall plots in the dashboard and artifact
catalog. Version 0.60.1 corrected the packaged upgrade
instructions and restores the authentication configuration script to the release
archive. Version 0.60.0 added mode-aware amateur-band activity
frequency presets throughout the Digital Modes dashboard, retains Custom tuning,
and shows the renamed Fldigi decoder-audio-center control only when it applies.
Version 0.59.0 made decoded digital results actionable:
FT8/FT4/WSPR and Fldigi rows can repopulate the decoder or prefill a station
memory, while guided empty states offer a known FT8 starting point instead of a
dead-end table. Version 0.58.0 added a Quick Start home view with
favorite station cards, recent activity shortcuts, and direct task launchers.
Favorite memories persist through the existing `favorite` station-memory tag,
and primary receiver forms now accept explicit Hz, kHz, or MHz input. Version 0.57.0 added a sticky, dashboard-wide receiver
status bar backed by the persisted active-job record, elapsed/progress details,
contextual Open and safe Stop controls, plus a recent activity drawer that links
jobs back to their relevant workspaces. Version 0.56.0 made dashboard listening explicit:
FM directory rows show in-place capture progress, populate a nearby persistent
audio player with RDS details and a WAV download, attempt playback automatically,
and report browser autoplay blocking. Station-memory Listen now opens the relevant
receiver view before capture, and all manual audio receivers share the same
autoplay-or-press-Play feedback. NumPy is temporarily constrained below 2.5 to
avoid a warning-strict incompatibility in Skyfield 1.55. Version 0.55.0 added browser-managed SSTV capture and
VIS-triggered watching, live job controls, decoder status, and an authenticated,
filterable image gallery with duplicate handling. Version 0.54.0 added a Digital Modes dashboard for
native CW/RTTY/BPSK31/AX.25, WSJT-X FT8/FT4/WSPR, Fldigi text modes, decoder
capability status, and persisted spot/text browsing. Version 0.53.0 added browser-managed Broadcast FM
surveys, resumable survey jobs, and a searchable RDS station directory with
one-click listening and station-memory creation. Version 0.52.0 added a browser Scan & Analyze
workspace for common-band/custom scans, heuristic surveys, live job status and
stop controls, plus one-click execution of saved RF presets. Version 0.51.0
reorganized the dashboard into
task-oriented views and adds guided setup, MHz-friendly memory entry, memory
editing/deletion, scan-profile deletion, schedule deletion, and clearer progress
and empty states. Version 0.50.3 made RF Operations self-starting in
the browser: create station memories and scan profiles, run the first scan, then
optionally schedule it. Version 0.50.2 resolved the FastMCP settings model's
generic lifespan annotation before startup, eliminating its Pydantic incomplete-field
warning. Version 0.50.1 hardened browser JSON responses against
non-finite RF measurements and configures a writable Matplotlib cache for the
systemd service. Version 0.50.0 added an RF Operations dashboard with
station status, scan trends, alert review, recent audio, and browser-managed scan
schedules. Version 0.49.0 added schedulable station-memory scan
profiles with restart-safe fixed intervals and change detection for reception
state, repeated failures, SNR, RDS station identity, and RadioText.

## Requirements

- Debian 13 on ARM64 (tested target: Raspberry Pi 5)
- Airspy HF+ with `airspyhf_rx` and `airspyhf_info`
- Optional RTL-SDR with `rtl_sdr` and `rtl_test`
- Python 3.11 or newer
- NumPy, SciPy, and Matplotlib

## Receiver backend architecture

The established tools continue to select `airspyhf-primary` automatically, so
existing MCP clients do not need to change. RF workflows now call the shared
receiver-backend layer rather than invoking the Airspy command adapter directly.
Each capture or stream holds a coordinator lease for its selected receiver.

The receiver registry can describe RTL-SDR, HackRF, Pluto/SoapySDR, and Web-888
devices for discovery and assignment planning. In v0.65.0, `airspyhf` and
`rtl_sdr` have capture and streaming adapters. Selecting HackRF, Pluto/SoapySDR,
or Web-888 for capture raises a clear `NotImplementedError`; it never falls back
to different hardware.

Install the optional RTL-SDR command-line tools with:

```bash
sudo apt install rtl-sdr
```

The easiest setup is in the dashboard. Open **System**, find **Add a receiver**,
and select **Scan for receivers**. The dashboard detects supported attached
hardware, uses the RTL-SDR serial when available, and prefills the receiver name,
role, stable ID, and priority. Review those values and select **Add receiver**.

For headless setup, call `discover_attached_sdr_devices`, then pass the selected
card values to `add_discovered_sdr_device`. Manual registration remains available
through `save_sdr_receiver` when defining hardware that is not currently attached:

```json
{
  "receiver_id": "rtl-vhf",
  "name": "RTL-SDR VHF receiver",
  "backend": "rtl_sdr",
  "role": "vhf_uhf_monitor",
  "device_selector": "00000042",
  "verified": true,
  "priority": 80
}
```

Pass `receiver_id="rtl-vhf"` to `list_devices`, `inspect_spectrum`,
`analyze_signal`, `receive_broadcast_fm`, or `classify_signal`. Established calls
that omit `receiver_id` continue to use `airspyhf-primary`.

## Getting started on a Raspberry Pi

This walkthrough assumes a Raspberry Pi 5 running 64-bit Raspberry Pi OS Lite
or Debian 13, connected to the same trusted LAN as the computer that will run
the MCP client. Commands run on the Pi unless a step explicitly says otherwise.
The examples use the login name `pi`; keep using your own login name if it is
different.

### 1. Prepare the Pi

1. Install a 64-bit Debian 13-based image, enable SSH in Raspberry Pi Imager,
   and boot the Pi. A Pi 5 with at least 4 GB RAM is recommended. A Pi 4 may
   work, but is not the tested target and long FFTs or decoders will be slower.
2. From another computer, connect over SSH (substitute the Pi's hostname or IP):

   ```bash
   ssh pi@raspberrypi.local
   ```

3. Confirm that the OS and Python meet the requirements, then update the Pi:

   ```bash
   uname -m
   python3 --version
   sudo apt update
   sudo apt full-upgrade -y
   sudo reboot
   ```

   `uname -m` should print `aarch64`, and Python must be 3.11 or newer. After
   the reboot, reconnect with SSH. If `.local` names do not work on your
   network, find the address with `hostname -I` on the Pi and use that address.

### 2. Connect and verify the receiver

Connect the SDR directly to a USB port for the first test. If it behaves
intermittently, use a short, shielded USB cable and a powered hub; inadequate
power and USB noise are common Raspberry Pi RF problems. Do not connect a
transmitter directly to the SDR input. This server controls receivers only, but
an excessive input signal can still damage receiver hardware.

Install the base receiver and Python packages:

```bash
sudo apt install -y airspyhf libairspyhf1 libairspyhf-dev rtl-sdr \
  python3-venv python3-pip python3-numpy python3-scipy \
  python3-matplotlib python3-skyfield openssl unzip
```

Only one receiver family is required; installing both command-line packages
makes later discovery easier. Test the attached device **as your normal user**:

```bash
# Airspy HF+
airspyhf_info

# RTL-SDR (use this instead when an RTL-SDR is attached)
timeout 15 rtl_test -t
```

A successful command identifies the radio without `sudo`. If the device is not
found, unplug and reconnect it, inspect `dmesg --ctime | tail -n 30`, and reboot
once so the package's udev rules take effect. If an RTL-SDR is claimed by the
Linux DVB driver, create a blacklist and reboot:

```bash
printf 'blacklist dvb_usb_rtl28xxu\n' | sudo tee /etc/modprobe.d/rtl-sdr-blacklist.conf
sudo reboot
```

Do not continue until the appropriate hardware test works for the same user
that will run `rf-mcp`.

### 3. Put the release in the expected directory

The service installer intentionally requires the project to be exactly
`~/rf-mcp`. Use **one** of these methods.

For the release ZIP, copy it to the Pi (for example, with `scp` from your other
computer), then extract it:

```bash
mkdir -p ~/rf-mcp
unzip rf-mcp-multi-sdr-v1.0.1.zip -d ~/rf-mcp
cd ~/rf-mcp
```

When working from this repository instead, copy the contents of `SDR-MCP` so
that `pyproject.toml` is directly inside `~/rf-mcp`—not inside a second
`SDR-MCP` directory:

```bash
mkdir -p ~/rf-mcp
cp -a /path/to/Article-Files/SDR-MCP/. ~/rf-mcp/
cd ~/rf-mcp
```

Verify the layout before installing:

```bash
test -f pyproject.toml && test -f scripts/install-service.sh && echo 'Layout OK'
```

### 4. Create the Python environment and test the software

Use Debian's scientific Python packages through `--system-site-packages`; this
avoids rebuilding NumPy and SciPy on the Pi:

```bash
cd ~/rf-mcp
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
pytest -q
```

All tests should pass. The editable install is intentional: the service runs
the code in `~/rf-mcp`, and a future release can be installed over that same
directory. Leave the environment with `deactivate` when desired.

### 5. Perform a local, interactive smoke test

Start the server bound only to the Pi itself:

```bash
cd ~/rf-mcp
source .venv/bin/activate
rf-mcp
```

Keep that terminal open. In a second SSH session, check its public health
endpoint:

```bash
curl --fail http://127.0.0.1:8765/healthz
```

A JSON response with a healthy service confirms that Python startup works.
Stop the foreground server with **Ctrl+C**. If startup fails, read the complete
terminal error before proceeding; hardware is not needed merely to answer the
health check.

### 6. Enable authentication and install the background service

The systemd unit listens on all LAN interfaces, so configure authentication
**before** enabling it:

```bash
cd ~/rf-mcp
chmod +x scripts/configure-auth.sh scripts/install-service.sh
./scripts/configure-auth.sh
./scripts/install-service.sh
```

`configure-auth.sh` prints a bearer token and stores it in the root-readable
file `/etc/rf-mcp.env`. Copy the token once into a password manager; do not put
it in shell history, screenshots, source control, or chat. Running the script
again rotates the token and restarts an existing service.

The installer creates `~/rf-mcp-data`, installs the unit, and starts it as the
normal login user. Confirm startup:

```bash
systemctl --no-pager --full status rf-mcp
curl --fail http://127.0.0.1:8765/healthz
journalctl -u rf-mcp -n 50 --no-pager
```

The health response should say authentication is required. The systemd unit
starts automatically after later reboots. Useful administration commands are:

```bash
sudo systemctl restart rf-mcp
sudo systemctl stop rf-mcp
journalctl -u rf-mcp -f
```

### 7. Find the Pi and open the dashboard

Get the Pi's current addresses:

```bash
hostname
hostname -I
```

From a browser on the same trusted LAN, open:

```text
http://PI_ADDRESS:8765/dashboard
```

Enter the bearer token at the login page. The dashboard session is separate
from MCP client authentication. In **System → Add a receiver**, select **Scan
for receivers**, review the detected receiver, give it a useful name and role,
and select **Add receiver**. Airspy-only installations also retain the default
`airspyhf-primary` receiver.

Port 8765 normally needs no router change on a home LAN. If a host firewall is
enabled on the Pi, allow it only from the local subnet (replace the example
subnet with yours):

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp
```

Never add a router port-forward for 8765. The service uses plain HTTP, so its
bearer token is not encrypted in transit. Use a VPN or a TLS reverse proxy for
access outside the trusted LAN.

### 8. Connect an MCP client and make the first capture

On a desktop computer with Node.js installed, start MCP Inspector:

```bash
npx @modelcontextprotocol/inspector
```

Choose **Streamable HTTP** and use:

```text
http://PI_ADDRESS:8765/mcp
```

Add this request header, substituting the saved token:

```text
Authorization: Bearer YOUR_TOKEN
```

Connect, call `list_devices`, and confirm that the expected backend and receiver
appear. Then call `inspect_spectrum` with a frequency supported by that radio:

```json
{
  "center_frequency_hz": 10000000,
  "duration_seconds": 2,
  "fft_size": 16384,
  "threshold_above_noise_db": 8,
  "max_peaks": 20,
  "retain_iq": false,
  "include_plot": true
}
```

The result should contain structured measurements followed by a spectrum image.
Set `include_plot` to `false` for automated clients that only need measurements.
For an RTL-SDR whose tuning range does not include 10 MHz, use a local signal in
the receiver's supported range, such as a broadcast-FM frequency, and pass its
registered `receiver_id`.

MCP client configuration formats differ, but every remote client needs the same
three values: transport **Streamable HTTP**, URL
`http://PI_ADDRESS:8765/mcp`, and header `Authorization: Bearer YOUR_TOKEN`.
Use the client's remote-HTTP configuration rather than a local `command`/stdio
configuration—the server process is running on the Pi, not on the desktop.

### 9. Verify operation after a reboot

Reboot once before considering setup complete:

```bash
sudo reboot
```

After reconnecting, run:

```bash
systemctl is-active rf-mcp
curl --fail http://127.0.0.1:8765/healthz
journalctl -u rf-mcp -b --no-pager | tail -n 50
```

Reconnect MCP Inspector and repeat `list_devices`. If captures fail while the
service itself is healthy, rerun `airspyhf_info` or `timeout 15 rtl_test -t` as
the normal user after stopping the service. A `resource busy` result usually
means the service already owns the receiver; stop `rf-mcp` before a direct
hardware test and start it again afterward.

### Optional decoders

The basic spectrum, AM/SSB/CW/NFM, and broadcast-FM workflows do not require the
optional decoder services. Add them only after the first capture succeeds:

```bash
cd ~/rf-mcp
./scripts/install-wsjt-decoders.sh   # FT8, FT4, and WSPR
./scripts/install-fldigi-decoders.sh # Fldigi text modes
./scripts/install-sstv-decoder.sh    # SSTV images
sudo systemctl restart rf-mcp
```

Each installer may add sizable packages. Run only the scripts for features you
intend to use, then check the corresponding capability/status tool in Inspector.

### Analyze and demodulate a signal

Call `analyze_signal` with, for example:

```json
{
  "frequency_hz": 10000000,
  "mode": "am",
  "bandwidth_hz": 10000,
  "duration_seconds": 5,
  "fft_size": 16384,
  "retain_iq": false,
  "include_audio": true,
  "include_plots": true
}
```

Supported modes are `am`, `usb`, `lsb`, `cw`, and `nfm`. The tool deliberately
places the requested signal 50 kHz away from receiver center, avoiding the DC
region, then downconverts it locally. It returns an RF plot, audio-spectrum
plot, WAV audio, structured metrics, and saved artifact paths.

### Decode CW and RTTY

Version 0.16.0 adds `decode_digital_signal`. The decoder captures fresh IQ and
runs locally on MiniRackDisplay; it does not require fldigi, WSJT-X, an Internet
service, or an audio loopback device. Decode CW near 7.030 MHz:

```json
{
  "frequency_hz": 7030000,
  "mode": "cw",
  "duration_seconds": 15,
  "cw_wpm": null,
  "retain_iq": false,
  "include_plot": true
}
```

When `cw_wpm` is null, timing is estimated from detected key-down intervals.
Set it from 5 through 60 when the sender's speed is known. The result includes
decoded text, the original dot/dash token sequence, estimated WPM, unknown
character count, and heuristic confidence.

Decode conventional amateur RTTY:

```json
{
  "frequency_hz": 14085000,
  "mode": "rtty",
  "duration_seconds": 20,
  "rtty_baud": 45.45,
  "rtty_shift_hz": 170,
  "rtty_polarity": "auto",
  "retain_iq": false,
  "include_plot": true
}
```

For RTTY, `frequency_hz` is the midpoint between the mark and space tones.
Supported baud rates are 40–300 and shifts are 80–1000 Hz. Polarity may be
`normal`, `reverse`, or `auto`; automatic mode tries both and selects the result
with better framing. Results include decoded text, the raw five-bit Baudot
codes, selected polarity, framing-error count, estimated center offset, and
confidence.

Every decode is stored as a `digital_decode` job with a JSON result and PNG
diagnostic plot. Set `retain_iq=true` only when the raw complex-float capture is
needed for later investigation. These decoders are intentionally conservative:
fading, interference, mistuning, nonstandard shifts, hand-sent Morse, and weak
signals can produce incomplete or incorrect text. Verify operationally
important messages against the plot, retained IQ, and an independent decoder.

### Decode BPSK31 and AX.25 packet

Version 0.17.0 adds two more native paths to `decode_digital_signal`. For a
BPSK31 signal whose actual RF center is 14.070150 MHz:

```json
{
  "frequency_hz": 14070150,
  "mode": "bpsk31",
  "duration_seconds": 20,
  "retain_iq": false,
  "include_plot": true
}
```

BPSK31 decoding uses the specified 31.25-symbol/s differential phase behavior,
removes residual carrier offset, searches symbol timing, and separates Varicode
characters at runs of two or more zero bits. Results contain decoded text, raw
Varicode strings, unknown count, timing offset, estimated frequency offset, and
confidence. BPSK31 has no protocol-level error detection, so readable text is
not proof that every character is correct.

For a 1200-baud packet/APRS channel centered at 144.390 MHz:

```json
{
  "frequency_hz": 144390000,
  "mode": "ax25_afsk1200",
  "duration_seconds": 15,
  "retain_iq": false,
  "include_plot": true
}
```

The AX.25 path FM-demodulates the IQ, detects 1200/2200 Hz Amateur Bell 202
tones, recovers NRZI symbols, locates HDLC flags, removes stuffed bits, verifies
CRC-16-CCITT, and parses destination, source, digipeater path, control, PID, and
information fields. `fcs_valid=true` is the strongest evidence that a packet
was recovered intact; frames with invalid checksums remain visible as
diagnostic evidence and must not be treated as valid traffic.

The requested AX.25 frequency is the RF channel center. The requested BPSK31
frequency is the narrow signal's actual RF center, which may be the transceiver
dial frequency plus an audio offset. A preliminary spectrum inspection is the
easiest way to identify the exact carrier.

QPSK31, PSK63, Olivia, Contestia, MFSK, DominoEX, THOR, MT63, and
Hellschreiber are available through the Fldigi adapter described below. JS8Call,
POCSAG, DMR, D-STAR, and other digital-voice codecs remain unsupported.

### Decode FT8, FT4, and WSPR

Version 0.21 uses the reference WSJT-X command-line engines rather than
reimplementing their synchronization, LDPC/FEC, and protocol decoders. Install
them once on MiniRackDisplay:

```bash
cd ~/rf-mcp
chmod +x scripts/install-wsjt-decoders.sh
./scripts/install-wsjt-decoders.sh
sudo systemctl restart rf-mcp
```

Call `list_digital_decoder_capabilities` to confirm that `jt9` and `wsprd` are
available. The existing CW, RTTY, BPSK31, and AX.25 decoders remain available
when WSJT-X is absent.

Decode four FT8 periods at the 20-meter dial frequency:

```json
{
  "frequency_hz": 14074000,
  "mode": "ft8",
  "capture_cycles": 4,
  "align_to_utc": true,
  "retain_iq": false,
  "retain_audio": true
}
```

`frequency_hz` is the suppressed-carrier/dial frequency, not an individual
signal in the waterfall. The result adds each decoded audio offset to the dial
frequency and returns UTC text, SNR, time offset, decoded message, callsign,
Maidenhead grid, CQ status, and raw decoder evidence. FT8 permits 1–8 periods;
FT4 permits 1–12 periods.

WSPR uses one complete two-minute period:

```json
{
  "frequency_hz": 14095600,
  "mode": "wspr",
  "capture_cycles": 1,
  "align_to_utc": true,
  "retain_iq": false,
  "retain_audio": true
}
```

WSPR results also include frequency drift and advertised transmitter power.
With `align_to_utc=true`, the call waits for the next protocol boundary before
capturing. A WSPR request can therefore take almost four minutes in the worst
case: up to two minutes waiting plus the two-minute receive period.

Use `monitor_weak_signal_frequency` for a bounded consecutive-period
observation, `list_weak_signal_spots` to filter stored results by mode,
callsign, or dial frequency, and `get_weak_signal_activity` to aggregate
decode count, grids, modes, best SNR, and most recent reception by callsign.

Long weak-signal captures use a special 125-second ceiling; this does not raise
the ordinary ten-second capture limit. USB conversion runs in bounded chunks
and temporary IQ is deleted unless `retain_iq=true`. A single WSPR IQ capture
is approximately 738 MB at 768 kS/s, so retain it only when needed.

### Decode Fldigi text modes

Version 0.22 integrates a dedicated, receive-only Fldigi instance. It uses the
documented XML-RPC interface to select the modem and carrier, clear and read
the RX pane, obtain modem quality, and force receive-only operation. Generated
USB audio is played through the Linux `snd-aloop` device rather than a physical
speaker or microphone.

Install the local decoder service once:

```bash
cd ~/rf-mcp
chmod +x scripts/install-fldigi-decoders.sh
./scripts/install-fldigi-decoders.sh
sudo systemctl restart rf-mcp
```

The installer adds Fldigi, Xvfb, ALSA utilities, and the persistent loopback
sound module. It starts `rf-mcp-fldigi.service` under the same unprivileged user
as rf-mcp. XML-RPC listens only on `127.0.0.1:7362` and exposes a restricted
receive/control method allowlist; transmit methods are not exposed.

Verify the pipeline with `get_fldigi_status`, followed by
`list_fldigi_modes`. Available configured modes include:

- BPSK63/125 and QPSK31/63;
- MFSK16/32;
- common Olivia and Contestia tone/bandwidth combinations;
- DominoEX 11/16 and THOR 11/16;
- MT63 500L/1000L/2000L;
- Feld Hell and Slow Hell.

Decode an Olivia 8/500 signal at a known USB dial frequency:

```json
{
  "frequency_hz": 14106500,
  "mode": "olivia-8-500",
  "duration_seconds": 60,
  "carrier_audio_hz": 1500,
  "retain_iq": false,
  "retain_audio": true
}
```

The adapter also accepts convenient defaults such as `olivia`, `contestia`,
`dominoex`, `thor`, `mt63`, `hell`, and `hellschreiber`. Call
`list_fldigi_modes` for the exact normalized names and the modem names actually
advertised by the installed Fldigi version.

`frequency_hz` is the USB dial frequency. `carrier_audio_hz` identifies the
signal within the audio passband; it defaults to 1500 Hz and may be set from
200 through 4500 Hz. This is a selected-mode decoder, not an automatic mode
classifier. A wrong mode, tone count, bandwidth, or carrier setting generally
produces empty or garbled text.

Results contain recovered text, unique callsigns and Maidenhead grids,
Fldigi's quality value, selected modem, estimated RF signal frequency, and WAV,
IQ, and JSON paths as requested. `list_fldigi_decodes` retrieves persistent
sessions by normalized mode or dial frequency. Fldigi quality is diagnostic;
it is not protocol-level error detection or proof that every character is
correct.

### Decode SSTV images and browse the gallery

Version 0.23 adds an asynchronous SSTV receiver around the mature
`colaclanth/sstv` image decoder. rf-mcp performs the Airspy capture, USB or NFM
demodulation, independent VIS-code/parity detection, job management, artifact
storage, and gallery indexing. Install the decoder into the project virtual
environment once:

```bash
cd ~/rf-mcp
chmod +x scripts/install-sstv-decoder.sh
./scripts/install-sstv-decoder.sh
sudo systemctl restart rf-mcp
```

Version 0.23.1 also supplies a virtual terminal size to the upstream decoder.
This is required when rf-mcp runs under systemd, where no interactive TTY is
attached; without it, the decoder's progress logger can raise `OSError 25`.

Version 0.24 makes an ordinary quiet receive window a completed job with
`outcome: "no_signal"`; decoder crashes and malformed output still produce a
failed job. Successful jobs report `decoded` or `duplicate`. A 256-bit visual
difference hash groups near-identical images received on the same frequency.
The original PNG is retained—duplicate detection labels gallery entries rather
than deleting evidence. Pass `include_duplicates: false` to `list_sstv_images`
for a de-duplicated view, and call `get_sstv_activity` for counts by mode and
frequency.

To monitor automatically, first save an SSTV preset:

```json
{
  "name": "ISS SSTV watcher",
  "preset_type": "sstv",
  "description": "Periodic NFM receive window on 145.800 MHz",
  "config": {
    "frequency_hz": 145800000,
    "duration_seconds": 180,
    "receiver_mode": "nfm",
    "retain_audio": false,
    "retain_iq": false,
    "deduplicate": true
  }
}
```

Then schedule the saved preset with `save_rf_schedule`:

```json
{
  "name": "ISS SSTV every 30 minutes",
  "preset_id_or_name": "ISS SSTV watcher",
  "interval_seconds": 1800,
  "enabled": true
}
```

The receiver-wide long-job lock prevents a scheduled SSTV capture from
colliding with scans, monitors, surveys, weak-signal decodes, or an interactive
SSTV request. A busy scheduled attempt is recorded as `skipped_busy` and the
following interval remains scheduled normally.

#### Signal-triggered streaming SSTV watcher

Version 0.25 can hold the Airspy on one frequency and process its binary IQ
stream directly in memory. Unlike the fixed-window monitor, it does not create
a large temporary IQ file during silence. A rolling three-second 12 kHz audio
buffer is sufficient to preserve the calibration leader and VIS header when a
trigger is recognized.

Start a one-hour watcher on the common ISS SSTV downlink:

```json
{
  "frequency_hz": 145800000,
  "receiver_mode": "nfm",
  "watch_duration_seconds": 3600,
  "rearm": true,
  "retain_audio": true,
  "deduplicate": true
}
```

Use `start_sstv_watcher`, then poll `get_sstv_watcher_status`. The live status
reports streamed seconds, accepted and rejected triggers, the current VIS
mode, decoded image count, and decoder failures. Retrieve the accumulated
records with `get_sstv_watcher_results`; stop early with `stop_sstv_watcher`.
`list_sstv_watch_sessions` retrieves persisted sessions after completion or a
service restart.

After a valid even-parity VIS header, the watcher records a bounded amount of
audio appropriate for Martin M1/M2, Scottie S1/S2/DX, or Robot 36/72. Decoding
runs on a worker thread while the Airspy stream continues to be drained, which
prevents an image conversion from stalling the receiver pipe. With `rearm=true`
the watcher returns to header search after a five-second cooldown and can
collect multiple images in one session. With `rearm=false`, it finishes after
the first complete triggered clip.

Streaming sessions may run from 30 seconds through 24 hours. They exclusively
occupy the Airspy for that time, so scheduled scans and other long-running
decoders will report `skipped_busy`. Raw IQ retention is intentionally not
offered for streaming watchers: at 768 kS/s, a one-hour float32 complex stream
would exceed 22 GB. Only triggered WAV, PNG, and JSON artifacts are retained.

#### SSTV image alerts and signed webhooks

Version 0.26 evaluates persistent rules whenever either the streaming watcher
or a fixed-window SSTV job decodes an image. Rules can select a dial frequency,
decoded mode, minimum heuristic quality, and whether near-duplicate gallery
images should be ignored. Omit `frequency_hz` or `sstv_mode` to match any
value. For example, pass this to `save_sstv_alert_rule`:

```json
{
  "name": "New ISS SSTV image",
  "frequency_hz": 145800000,
  "minimum_quality": 0.4,
  "unique_only": true,
  "enabled": true,
  "replace_existing": false
}
```

Manage rules with `list_sstv_alert_rules`, `get_sstv_alert_rule`,
`set_sstv_alert_rule_enabled`, and `delete_sstv_alert_rule`. Supported mode
filters are Martin M1/M2, Scottie S1/S2/DX, and Robot 36/72, using the exact
mode names returned by the gallery.

The existing webhook outbox and dispatcher are shared with RF watchlist
alerts. Subscribe a destination to one SSTV rule with
`save_rf_webhook_destination` and `sstv_rule_id_or_name`:

```json
{
  "name": "SSTV Receiver",
  "url": "https://example.net/hooks/sstv",
  "sstv_rule_id_or_name": "New ISS SSTV image",
  "signing_secret": "replace-with-at-least-16-random-characters",
  "enabled": true,
  "replace_existing": false
}
```

A destination may select one RF rule, one SSTV rule, or—if neither selector is
given—all alert events. SSTV POST bodies use the `rf-mcp.sstv-alert.v1` schema.
They include compact image metadata and an `image_download_path` such as
`/artifacts/artifact-...`; image bytes and decoder logs are not embedded. The
artifact endpoint requires the same bearer token as the RF web service when
authentication is enabled.

Review events with `list_sstv_alert_events` and `get_sstv_alert_event`, then
mark handled events with `acknowledge_sstv_alert_event`. Delivery inspection,
retry, signing, private-network safeguards, and destination management use the
existing RF webhook tools described below. Deleting an SSTV rule retains
historical events; a destination that selected only that rule becomes dormant
until it is updated or deleted.

#### Satellite-pass-aware SSTV reception

Version 0.27 uses Skyfield and the SGP4 orbital model to predict when a selected
satellite will cross above a configured elevation mask at your receiving
location. It persists each AOS, culmination, and LOS window, starts the existing
streaming SSTV watcher before AOS, and stops it after LOS. Install the Debian 13
package `python3-skyfield` as shown in the main installation command above.

Create an ISS watch with `save_satellite_sstv_watch`. Replace the example
coordinates and TLE lines with your receiver location and a current element set:

```json
{
  "name": "ISS SSTV passes",
  "satellite_name": "ISS (ZARYA)",
  "norad_id": 25544,
  "tle_line1": "1 25544U ...current line 1...",
  "tle_line2": "2 25544 ...current line 2...",
  "latitude_deg": 33.0,
  "longitude_deg": -117.9,
  "elevation_m": 100,
  "frequency_hz": 145800000,
  "receiver_mode": "nfm",
  "minimum_elevation_deg": 10,
  "lead_seconds": 60,
  "trail_seconds": 30,
  "notify_before_seconds": 600,
  "tle_source": "celestrak",
  "auto_refresh": true,
  "refresh_interval_seconds": 86400,
  "doppler_correction_mode": "digital",
  "doppler_step_seconds": 10,
  "enabled": true,
  "replace_existing": false
}
```

The NORAD ID must match the TLE. Use `predict_satellite_passes` to preview up
to seven days without occupying the Airspy. Each prediction includes UTC times,
AOS/TCA/LOS azimuth and elevation, maximum elevation, duration, TLE epoch and
age. `tle_stale=true` is reported once the element epoch is more than 14 days
from the prediction start.

Version 0.28 optionally manages the element lifecycle. Set
`tle_source="celestrak"` and `auto_refresh=true` to retrieve only that watch's
NORAD catalog number from CelesTrak over HTTPS. The default refresh interval is
24 hours; accepted values are six hours through seven days. Responses are
limited to 8 KiB and must contain exactly one matching two-line element set
with valid line checksums. A valid update is committed atomically, supersedes
only future planned windows, and rebuilds the 48-hour horizon. A download,
format, catalog-number, checksum, or propagation failure retains the
last-known-good TLE, records the diagnostic, and schedules a one-hour retry.
Because the legacy TLE format has a five-digit catalog field, this managed
source currently supports NORAD IDs through 99999; OMM support is deferred.

Use `refresh_satellite_tle` for an immediate managed refresh. Watch records
expose `last_tle_refresh_at`, `last_tle_refresh_status`,
`last_tle_refresh_error`, `next_tle_refresh_at`, and `tle_epoch_at`.
`get_satellite_scheduler_status` summarizes refresh failures. Manual mode
remains supported with `tle_source="manual"` and `auto_refresh=false`; in that
mode, replace the same named watch with current lines and
`replace_existing=true`.

The pass scheduler refreshes a 48-hour horizon every 30 minutes and persists
planned windows across service restarts. It records `launched`, `completed`,
`stopped`, `skipped_busy`, `missed`, `failed`, `interrupted`, or `superseded`
outcomes. If another long RF job owns the receiver at the start of the window,
that pass is recorded as `skipped_busy`; it will not interrupt the active job.
Prediction schedules reception opportunity, not actual satellite activity—the
ISS and other satellites do not transmit SSTV on every visible pass.

Use `list_satellite_sstv_watches`, `get_satellite_sstv_watch`,
`set_satellite_sstv_watch_enabled`, and `delete_satellite_sstv_watch` to manage
watches. Inspect scheduled and historical windows with
`list_satellite_pass_windows`, and use `get_satellite_scheduler_status` for
thread health and the next start time. Decoded images and SSTV webhook payloads
include `source_satellite_watch_id` and `source_satellite_pass_id`, allowing a
gallery image or alert to be traced back to its predicted pass.

Version 0.28 also creates a persistent pre-pass alert at
`notify_before_seconds` before AOS and one outcome alert after a pass becomes
`completed`, `stopped`, `interrupted`, `skipped_busy`, `missed`, or `failed`.
Use `list_satellite_pass_alerts` and `acknowledge_satellite_pass_alert` to work
with them. Event IDs are stored on the pass record, preventing duplicate
notifications after a service restart.

Existing all-alert webhook destinations receive satellite alerts automatically.
To subscribe a destination only to one watch, call
`save_rf_webhook_destination` with `satellite_watch_id_or_name`:

```json
{
  "name": "ISS pass receiver",
  "url": "https://example.net/hooks/iss",
  "satellite_watch_id_or_name": "ISS SSTV passes",
  "signing_secret": "replace-with-at-least-16-random-characters",
  "enabled": true,
  "replace_existing": false
}
```

These POST bodies use `rf-mcp.satellite-pass.v1` and share the existing HMAC
signature, durable outbox, bounded retries, delivery inspection, and private
network safeguards. Deleting a watch retains its historical pass and alert
records and makes a watch-only destination dormant.

#### Doppler-aware satellite reception

Version 0.29 derives topocentric range and range rate from the same Skyfield
SGP4 solution used for pass prediction. AOS, TCA, and LOS now report
`range_km`, `range_rate_km_s`, `doppler_shift_hz`, and
`corrected_receive_frequency_hz`. Positive shift means the approaching
satellite is received above the nominal downlink; the sign reverses as it
recedes.

Every newly scheduled pass persists a full correction plan from the watcher's
lead time through its trail time. `doppler_step_seconds` may be 1–60 seconds;
10 seconds is the default. Retrieve that plan with
`get_satellite_doppler_plan`. With `include_plot=true`, the tool creates and
catalogs a PNG plotting Doppler shift and elevation against UTC, returns the
image through MCP, and supplies an authenticated artifact download path.

Set `doppler_correction_mode="digital"` on a satellite watch to apply the plan.
The Airspy remains tuned to the nominal downlink with its normal wide 768 kS/s
stream. rf-mcp interpolates the correction for every streamed chunk and moves
the digital downconverter to the predicted received frequency. This avoids
restarting `airspyhf_rx`, preserves mixer phase between correction steps, and
does not introduce retune gaps into SSTV scan lines. Set the mode to `off` to
record Doppler metadata without applying correction.

Digital correction is predictive, not a measurement of the received carrier.
Its accuracy depends on current orbital elements, the configured observer
coordinates, and the Pi's UTC clock. It does not compensate receiver oscillator
error, transmitter offset, or an incorrect nominal downlink frequency. For the
145.800 MHz ISS NFM signal correction is optional; it is more consequential for
narrower USB satellite signals and future telemetry decoders.

#### Multi-downlink satellite profiles

Version 0.30 generalizes the pass receiver beyond SSTV. Use
`save_satellite_receive_profile` with one through 16 downlinks. Each downlink
has a stable `downlink_id`, label, frequency, mode, receiver mode, priority,
enabled flag, and audio-retention flag. Supported modes are `sstv`,
`nfm_audio`, `ax25_afsk1200`, `ax25_g3ruh9600`, and `capture_only`.

An Airspy HF+ is one receiver, so rf-mcp selects one enabled downlink for each
pass. The `priority` policy always selects the lowest priority number; the
`round_robin` policy rotates across upcoming passes. It does not claim
simultaneous reception of separated frequencies. Each persisted pass records
its selected downlink and a frequency-specific Doppler plan.

For example, a profile can prioritize a packet downlink while retaining an NFM
audio downlink for round-robin passes:

```json
{
  "name": "Amateur satellite",
  "satellite_name": "EXAMPLE-SAT",
  "norad_id": 12345,
  "tle_line1": "replace with current line 1",
  "tle_line2": "replace with current line 2",
  "latitude_deg": 47.0,
  "longitude_deg": -122.0,
  "downlink_selection_policy": "round_robin",
  "doppler_correction_mode": "digital",
  "downlinks": [
    {"downlink_id": "packet", "label": "Packet", "frequency_hz": 145825000,
     "mode": "ax25_afsk1200", "receiver_mode": "nfm", "priority": 1,
     "enabled": true, "retain_audio": true},
    {"downlink_id": "voice", "label": "Voice", "frequency_hz": 145800000,
     "mode": "nfm_audio", "receiver_mode": "nfm", "priority": 2,
     "enabled": true, "retain_audio": true}
  ]
}
```

Use `list_satellite_observations`, `get_satellite_observation`, and
`get_satellite_activity` to inspect pass-grouped results. AX.25 observations
include parsed frames, valid-FCS counts, and a diagnostic plot. NFM observations
can retain a WAV file. Capture-only observations retain level statistics rather
than raw IQ, limiting storage growth. Active general receiver jobs are exposed
through `get_satellite_receive_status`, `get_satellite_receive_results`, and
`stop_satellite_receive`. Rotor control is intentionally outside this release.

#### 9600-baud satellite packet telemetry

Version 0.31 adds `ax25_g3ruh9600` as a satellite downlink mode. The decoder
uses FM discrimination, searches symbol timing and polarity, applies the common
self-synchronizing x^17+x^12+1 descrambler, converts NRZI to HDLC bits, removes
bit stuffing, and validates each AX.25 frame with its received FCS. Parsed
results retain source, destination, digipeaters, control, PID, information text,
information hex, and the complete raw frame hex.

Use `list_satellite_packet_frames` to search both 1200-baud AFSK and 9600-baud
G3RUH observations. It can filter by callsign and optionally return only frames
with a valid FCS. `export_satellite_packet_telemetry` creates either JSON Lines
or CSV and registers the export as a downloadable artifact. JSONL preserves
the most natural typed representation; CSV is convenient for spreadsheets and
external analysis.

Example downlink entry:

```json
{
  "downlink_id": "packet-9600",
  "label": "9600 packet telemetry",
  "frequency_hz": 145900000,
  "mode": "ax25_g3ruh9600",
  "receiver_mode": "nfm",
  "priority": 1,
  "enabled": true,
  "retain_audio": false
}
```

The decoder is protocol-generic: it exposes AX.25 payload bytes but does not
pretend that every spacecraft uses the same mission-specific telemetry field
layout. Invalid-FCS frames remain visible for RF troubleshooting and are
clearly marked rather than silently presented as trustworthy telemetry.

#### Declarative telemetry schemas

Version 0.32 turns selected AX.25 information bytes into named engineering
values without executing user-supplied code. A schema can match satellite name,
source and destination callsigns, PID, FCS validity, and a hexadecimal payload
signature at a configured offset. Fields use explicit byte offsets and safe
types: signed or unsigned 8/16/32/64-bit integers, 32/64-bit floating point,
ASCII, hexadecimal bytes, booleans, and bounded bit fields. Numeric fields may
apply `value * scale + add` and retain a unit.

For example:

```json
{
  "name": "Example beacon v1",
  "satellite_name": "EXAMPLE-SAT",
  "match": {
    "source": "EXSAT-1",
    "pid": 240,
    "payload_prefix_hex": "a501",
    "require_valid_fcs": true
  },
  "fields": [
    {"name": "battery_voltage", "label": "Battery voltage", "type": "uint16",
     "offset": 2, "byte_order": "big", "scale": 0.001, "unit": "V"},
    {"name": "temperature", "label": "Board temperature", "type": "int8",
     "offset": 4, "unit": "C"}
  ]
}
```

Use `validate_satellite_telemetry_schema` with known sample payload hex before
saving. `save_satellite_telemetry_schema` persists the definition. New packet
observations are decoded automatically against enabled matching schemas; use
`decode_satellite_observation_telemetry` to process older observations after a
schema is added. A short payload produces a recorded per-frame decoding failure
instead of fabricated values.

Use `list_satellite_telemetry_values` for decoded history,
`plot_satellite_telemetry` for numeric time-series PNGs, and
`export_decoded_satellite_telemetry` for CSV or JSON Lines. Replacing a schema
invalidates and removes its previously derived values because their meaning may
have changed; source observations and immutable AX.25 frame bytes are retained.

#### Telemetry alerts and health

Version 0.33 evaluates decoded numeric telemetry as it is persisted. Rules can
match values `above` or `below` a threshold, `inside` or `outside` a range, or
an `absolute_change` or `percent_change` from the most recent prior observation.
Change rules deliberately compare different observations rather than multiple
frames in the same capture.

Create rules with `save_satellite_telemetry_alert_rule`. Use
`test_satellite_telemetry_alert_rule` to exercise a condition without saving it
or emitting an event. For example:

```json
{
  "name": "Battery voltage low",
  "schema_id_or_name": "Example beacon v1",
  "field_name": "battery_voltage",
  "condition_type": "below",
  "threshold_low": 3.3,
  "cooldown_seconds": 3600,
  "enabled": true
}
```

Matching rules create durable `satellite_telemetry` alert events and enqueue
deliveries for existing all-rule webhook destinations using payload schema
`rf-mcp.satellite-telemetry-alert.v1`. A per-rule cooldown suppresses repeated
notifications while a value remains abnormal; zero disables suppression. Use
`list_satellite_telemetry_alerts` and
`acknowledge_satellite_telemetry_alert` to manage event history.

`get_satellite_telemetry_health` reports the latest stored value for every
schema field, marks fields older than a configurable freshness interval, and
summarizes enabled rules and recent telemetry alerts. Replacing a telemetry
schema clears its derived values and disables rules tied to that schema so old
threshold assumptions cannot silently apply to a changed binary layout. The
rules may be reviewed and deliberately replaced to re-enable them.

#### Satellite pass-performance analytics

Version 0.34 derives a reproducible 0–100 performance score for each attempted
pass. Packet-mode reports combine receiver outcome, relative signal level,
capture-window coverage, packet yield, valid-FCS rate, and decoded telemetry
yield. Audio and capture-only modes use outcome, relative level, and coverage
without penalizing them for having no packets. Every report exposes the
individual normalized components rather than presenting the composite score as
a black box.

`get_satellite_pass_performance` reports one pass. Use
`compare_satellite_passes` for a selected set, or
`summarize_satellite_pass_performance` to compare the recent history of a
receive profile. The summary groups results by downlink and reports pass count,
mean and median score, packet totals, valid-FCS rate, and telemetry yield. Its
recommendation is the highest observed mean—not an automatic profile change.

`plot_satellite_pass_performance` produces a two-panel PNG showing score over
time and score versus maximum pass elevation. This helps distinguish a weak
downlink or decoding configuration from simply poor observing geometry.
`export_satellite_pass_performance` writes the reports as JSON or CSV.

Signal level remains relative Airspy amplitude, not calibrated dBm. The score
labels its confidence `low` when packet evidence is missing, `medium` for fewer
than ten packets, and `high` only with a larger sample. Planned and superseded
passes are excluded. Busy, missed, failed, stopped, and interrupted attempts
remain visible so selection bias does not make reception history look better
than it was. Rotor control remains outside this release.

#### Inspector-friendly location and satellite discovery

Version 0.35 changes latitude and longitude arguments in the manual and
catalog-assisted profile tools to strings. This prevents MCP Inspector's number
widget from forcing whole-number increments. The server still persists numeric
decimal degrees and accepts either signed decimal text (`33.96`, `-117.95`) or
DMS text (`33 57 36 N`, `117 57 0 W`). Compass suffixes take care of the sign.

The assisted workflow uses current CelesTrak orbital elements and SatNOGS DB
transmitter metadata:

1. Call `list_satellite_catalog_categories` to see category names.
2. Call `search_satellite_catalog` with a partial name or a category such as
   `amateur`, `weather`, `noaa`, `goes`, `stations`, or `active`.
3. Call `get_satellite_catalog_entry` with the selected NORAD ID. Review the
   TLE, public transmitters, frequencies, modes, baud rates, Airspy tuning
   compatibility, decoder support, and suggested downlinks.
4. Call `create_satellite_receive_profile_from_catalog` with the NORAD ID and
   location strings. Omit `transmitter_ids` to use every compatible suggestion,
   or pass only the reviewed transmitter IDs you want.

The server never treats catalog metadata as proof that a spacecraft is
transmitting now. Frequencies outside the Airspy HF+ ranges are shown but cannot
be selected. Known SSTV, 1200-baud AFSK/AX.25, explicit G3RUH, and NFM voice
descriptions map to supported receivers. Unrecognized digital modes use
`capture_only`; a generic `FSK 9600` label is not assumed to be AX.25/G3RUH.
This conservative mapping avoids plausible-looking but incorrect decoding.

Catalog downloads are cached locally. CelesTrak name/category/TLE responses are
held for two hours, and SatNOGS transmitter responses for one hour. Existing
saved profiles continue to operate from their last-known-good TLE if either
public service is temporarily unavailable. Catalog-created profiles enable the
existing daily managed CelesTrak refresh after their initial reviewed creation.

#### Saved locations and observation planning

Version 0.36 removes repeated coordinate entry from the normal Inspector
workflow. Call `save_observer_location` once with a friendly name such as
`MiniRackDisplay`; decimal-degree and DMS strings are both accepted. Set
`make_default=true` so later tools can omit the location. Use
`list_observer_locations` to review saved stations. Deletion is guarded by an
explicit `confirm_delete=true` argument.

Call `plan_satellite_observations` with a category such as `amateur`, `noaa`,
or `goes`, or supply a partial satellite name in `query`. The planner obtains a
bounded current CelesTrak set, predicts visibility from the saved location,
checks the leading candidates against SatNOGS transmitter metadata, removes
satellites with no Airspy-compatible suggested downlink, and ranks the results
by maximum elevation. Each result includes AOS, TCA, LOS, duration, the best
suggested downlink, and the number of compatible downlinks.

The default bounds are intentionally modest: 20 orbital candidates and eight
returned opportunities. A plan is a preview, not a scheduled recording, and
catalog metadata still requires review. After choosing a result, call
`create_satellite_receive_profile_at_saved_location` with its NORAD ID. This
creates the managed profile without re-entering coordinates; the existing
satellite scheduler then predicts and records its future pass windows.

The watcher is also available as an `sstv_watch` RF preset. For example:

```json
{
  "name": "Live ISS SSTV hour",
  "preset_type": "sstv_watch",
  "description": "One-hour signal-triggered watcher",
  "config": {
    "frequency_hz": 145800000,
    "receiver_mode": "nfm",
    "watch_duration_seconds": 3600,
    "rearm": true,
    "retain_audio": true,
    "deduplicate": true
  }
}
```

Call `list_sstv_decoder_capabilities` to confirm that the executable is found.
The supported image modes are Martin M1/M2, Scottie S1/S2/DX, and Robot 36/72.
PD and Pasokon modes are not claimed by this release.

For a typical amateur HF transmission, start `decode_sstv` with the USB dial
frequency:

```json
{
  "frequency_hz": 14230000,
  "duration_seconds": 130,
  "receiver_mode": "usb",
  "retain_audio": true,
  "retain_iq": false
}
```

For an FM SSTV downlink such as 145.800 MHz, use `receiver_mode: "nfm"` and a
longer observation window:

```json
{
  "frequency_hz": 145800000,
  "duration_seconds": 180,
  "receiver_mode": "nfm",
  "retain_audio": true,
  "retain_iq": false
}
```

Both `decode_sstv` and `monitor_sstv_frequency` return immediately with a
`job_id`. Poll `get_sstv_status`, then retrieve the record with
`get_sstv_results`. `stop_sstv` requests cancellation. A capture already in
progress cannot interrupt `airspyhf_rx`, so the job stops at the next phase
boundary.

Use `list_sstv_images` to filter the persistent gallery by frequency or mode.
`get_sstv_image` returns the metadata and, by default, the actual PNG as native
MCP image content. Each record includes dimensions, VIS code, parity result,
decoder diagnostics, capture time, and a 0-to-1 heuristic quality score. That
score combines image contrast and VIS validity; it is useful for sorting, not a
calibrated signal or image-quality measurement.

The 130-second default covers most common images when capture starts near the
beginning. `monitor_sstv_frequency` defaults to 180 seconds to improve the odds
of catching an unknown start time. Scottie DX may require the 310-second
maximum. At 768 kS/s, temporary float32 I/Q occupies about 799 MB for 130
seconds and 1.90 GB for 310 seconds. It is deleted after decoding unless
`retain_iq` is true.

### Receive broadcast FM (WFM)

Version 0.18.0 adds `receive_broadcast_fm` for stations from 60 through 110 MHz.
The tool captures the full FM channel, places it away from the receiver's DC
region, FM-demodulates the multiplex, and writes 48 kHz PCM audio. Receive a
North American station at 88.5 MHz:

```json
{
  "frequency_hz": 88500000,
  "duration_seconds": 20,
  "stereo": true,
  "deemphasis_us": 75,
  "decode_rds_data": true,
  "retain_iq": false,
  "include_audio": true,
  "include_plot": true
}
```

Use 75 µs de-emphasis for North American broadcasting and 50 µs where that is
the regional standard. The stereo decoder recovers the baseband
`(L+R)/2` signal, detects and measures the 19 kHz pilot, regenerates the
suppressed 38 kHz stereo subcarrier from the pilot phase, and reconstructs the
`(L-R)/2` information. If no credible pilot is present, a stereo request safely
falls back to one-channel mono audio.

The result reports:

- whether a stereo pilot was detected and stereo decoding was used;
- pilot level relative to composite RMS;
- selected 50 or 75 µs de-emphasis;
- audio channel count and sample rate;
- energy in the 54–60 kHz band around the 57 kHz RDS subcarrier;
- a provisional `rds_candidate_detected` indicator.

The saved multiplex plot shows the first 200 ms of discriminator output and the
0–80 kHz multiplex spectrum with markers at 19, 38, and 57 kHz. Every reception
is persisted as a `broadcast_fm` job with WAV, PNG, and JSON artifacts. Raw IQ
is deleted unless `retain_iq=true`.

`rds_candidate_detected` remains a fast energy measurement and can be raised by
noise, multipath, or other subcarriers. Version 0.19 adds the independently
validated decoder described below.

### Decode RDS station data

With `decode_rds_data=true`, `receive_broadcast_fm` now regenerates the 57 kHz
reference from the 19 kHz pilot, extracts the RDS subcarrier, recovers the
1,187.5 bit/s differential bi-phase stream, and searches timing and polarity.
It accepts groups only after finding the required checksum/offset sequence
`A → B → C/C′ → D`.

The v0.19 result contains:

- Programme Identification (`pi_code`);
- numeric Programme Type and optional eight-character PTY name;
- Traffic Programme and Traffic Announcement flags;
- Music/Speech indication;
- eight-character Programme Service name with completeness status;
- 2A/2B RadioText with A/B message-reset handling and received segments;
- decoded alternative FM frequencies from 0A groups;
- 4A UTC clock time and local offset;
- every accepted group with its four data words, type, version, bit offset,
  parsed fields, and correction count;
- valid block/group counts, raw recovered bit count, timing, polarity, and
  aggregate confidence.

After synchronization, the decoder can correct one bad bit in B, C/C′, or D.
Every correction increments `block_error_count` and reduces confidence. It does
not silently correct an A block because A is the synchronization anchor. Invalid
or incomplete groups cannot update station metadata.

RDS fields are transmitted on different repetition schedules. A short capture
may recover PI and PS but miss RadioText, alternative frequencies, or clock
time. Use 20–30 seconds for an initial station query and compare repeated
results under fading or multipath. An empty field means it was not recovered in
that capture, not necessarily that the station does not transmit it.

The decoder preserves raw valid groups for applications it does not yet
interpret. Open Data Applications, TMC traffic messages, EON, paging, and
country-specific character-table extensions require additional application
parsers; they are not mislabeled as ordinary RadioText.

### Survey the broadcast-FM band

Version 0.20 adds `survey_broadcast_fm`. It runs asynchronously: a short
discovery capture checks each channel, then occupied candidates receive longer
WFM/RDS captures. The North American defaults cover every odd 200 kHz channel
from 87.9 through 107.9 MHz:

```json
{
  "start_frequency_hz": 87900000,
  "stop_frequency_hz": 107900000,
  "channel_spacing_hz": 200000,
  "discovery_duration_seconds": 0.25,
  "discovery_threshold_db": 8,
  "rds_duration_seconds": 10,
  "deemphasis_us": 75,
  "save_audio": false,
  "save_plots": true
}
```

Use `get_fm_survey_status` while it runs and `get_fm_survey_results` for the
candidate/station records. `stop_fm_survey` stops safely after the current
capture. Resume the exact checkpoint by calling `survey_broadcast_fm` with
`resume_job_id` set to that job ID; completed discovery and RDS captures are not
repeated.

Every decoded observation updates the persistent station directory. Use
`list_fm_stations`, optionally with `rds_only=true`, or `get_fm_station` with an
exact channel frequency. Empty metadata from a weak later capture does not
erase previously decoded PI, PS, PTYN, RadioText, or alternative frequencies.
The directory retains first/last observation times, observation count, latest
survey, stereo state, discovery score, pilot level, and RDS group count.

Completed surveys produce checkpoint/result JSON and a spreadsheet-friendly
CSV. Per-station multiplex plots are optional and enabled by default; WAV audio
samples are optional and disabled by default. `compare_fm_surveys` reports new,
disappeared, stable, and metadata-changed channels between any two persisted FM
survey jobs.

The default full-band discovery pass takes roughly 25 seconds of receiver time,
plus ten seconds for every occupied candidate and processing/retune overhead.
Results remain relative receiver measurements, not calibrated dBm.

### Monitor a frequency

Start a five-minute monitor that samples WWV every five seconds:

```json
{
  "frequency_hz": 10000000,
  "mode": "am",
  "bandwidth_hz": 10000,
  "total_duration_seconds": 300,
  "capture_duration_seconds": 2,
  "interval_seconds": 5,
  "fft_size": 8192,
  "waterfall_span_hz": 100000,
  "record_audio_on_activity": false
}
```

`start_monitor` returns immediately with a `job_id`. Pass that ID to:

- `get_monitor_status` for progress;
- `get_monitor_results` for current or final samples, grouped events, and the
  waterfall/SNR image;
- `stop_monitor` to stop after the current capture.

Only one monitor can run at a time. Jobs continue when an MCP client disconnects
but do not yet survive a restart of the `rf-mcp` service. Completed JSON and PNG
artifacts remain on disk. Monitoring is limited to one hour and 1,000 captures
per job. Enabling `record_audio_on_activity` can consume significant storage.

### Scan a band

Start an asynchronous scan of the 20-meter amateur band:

```json
{
  "start_frequency_hz": 14000000,
  "stop_frequency_hz": 14350000,
  "capture_duration_seconds": 1,
  "overlap_fraction": 0.15,
  "fft_size": 8192,
  "threshold_above_noise_db": 8,
  "minimum_signal_spacing_hz": 1000,
  "attenuation_steps": 1,
  "max_signals": 100
}
```

`start_band_scan` returns a `job_id`. Use `get_band_scan_status`,
`get_band_scan_results`, and `stop_band_scan` with that ID. Results can be read
while the scan is still in progress.

The scanner rejects the outer 12% of each receiver passband, overlaps adjacent
captures, averages their linear power where they overlap, and then detects and
ranks peaks in the stitched spectrum. Receiver AGC is disabled during a scan;
`attenuation_steps` selects 0–48 dB in 6 dB increments. The default is one step
(6 dB). Results are comparable within a scan but are not calibrated dBm.

Band scans cannot cross the Airspy HF+ gap between 31 and 60 MHz. A monitor and
a band scan cannot run simultaneously. Scans are limited to 500 retunes.

### Survey and classify a band

Version 0.9 combines scanning and classification into one asynchronous job.
For example, survey the 20-meter amateur band and classify its five strongest
detected carriers:

```json
{
  "start_frequency_hz": 14000000,
  "stop_frequency_hz": 14350000,
  "capture_duration_seconds": 1,
  "overlap_fraction": 0.15,
  "fft_size": 8192,
  "threshold_above_noise_db": 8,
  "minimum_signal_spacing_hz": 1000,
  "attenuation_steps": 1,
  "max_signals": 100,
  "classify_top_signals": 5,
  "classification_duration_seconds": 2,
  "classification_bandwidth_hz": 30000
}
```

Call `start_band_survey`, then use `get_band_survey_status`,
`get_band_survey_results`, or `stop_band_survey`. Status reports separate
`scanning`, `classifying`, and `finished` phases. The final result contains the
stitched spectrum, all detected carriers, ranked classifications for the
strongest signals, one annotated survey plot, and individual diagnostic plots.

Surveys support at most 20 classification retunes. Classification failures are
recorded per carrier without discarding the rest of the survey. The labels are
heuristic rather than protocol-decoder results, and fading signals may change
between the scan pass and the classification retune.

### RF activity profiles and dashboard

Version 0.37 turns repeated band surveys into a longitudinal RF-environment
record. `list_rf_activity_bands` provides convenient definitions for the HF
amateur bands plus FM broadcast, civil airband, two meters, and NOAA weather
radio. They are survey conveniences rather than regulatory band-plan advice.

Create a reusable profile with `save_rf_activity_profile`:

```json
{
  "name": "20m activity",
  "band_name": "20m",
  "capture_duration_seconds": 1,
  "threshold_above_noise_db": 8,
  "attenuation_steps": 1,
  "classify_top_signals": 5
}
```

Run it immediately with `run_rf_preset`, or pass its name to the existing
`save_rf_schedule` tool for unattended observations. Fixed attenuation and the
same profile are important for meaningful comparison. Each v0.37 scan stores a
bounded 1,200-point spectrum summary and the fraction of spectral bins above
the configured detection threshold; this is a compact history, not retained IQ.

After one or more completed runs, call `get_rf_activity_dashboard`. It reports
signal count, occupied-bin fraction, digital noise floor, overload warnings,
the latest run versus the historical median, recurring frequency clusters,
new signals, and configurable anomaly flags. The dashboard also creates a
frequency-versus-survey heatmap plus JSON and CSV exports by default.

The noise measurement is digital dBFS/Hz and is not calibrated antenna-input
power. Occupancy is a sampled fraction from stepped scans, not simultaneous
whole-band occupancy. Old pre-v0.37 runs remain usable for signal and noise
history, but heatmap and occupied-bin data appear only on new runs containing
the compact spectrum summary.

### HF propagation assistant

Version 0.38 answers “what bands appear usable from this station?” using local
evidence rather than a generic propagation widget. `get_hf_propagation_report`
groups recent FT8, FT4, and WSPR spots by amateur band; includes the latest
v0.37 activity-profile occupancy, signal count, and noise-baseline delta; and
summarizes scheduled WWV/CHU watchlist results. Bands are labeled as strong,
moderate, limited, or having no recent local evidence. No recent evidence does
not prove that a band is closed.

Use `save_hf_time_station_watchlist` to create eight standard WWV and CHU
checks, then run it with `run_rf_preset` or schedule it with `save_rf_schedule`.
The watchlist records heuristic carrier classifications and confidence. It
does not attempt to decode the time code or claim transmitter identity solely
from energy at the nominal frequency.

`get_space_weather_snapshot` retrieves current 10.7-cm solar flux, planetary
Kp, and NOAA R/S/G scales from the official NOAA Space Weather Prediction
Center JSON products. Responses are capped, validated, cached for 15 minutes,
and returned with per-product errors. If a refresh completely fails, a
last-known cache is returned and explicitly marked stale. Set
`force_refresh=true` only when an immediate network refresh is warranted.

`get_hf_propagation_report` adds that external context by default and produces
a local-evidence plot plus JSON and CSV exports. NOAA R-scale conditions and
elevated Kp are explained as possible HF impairments; solar flux is retained as
context. These inputs never override the receiver’s local evidence. The report
is not a path-specific MUF/LUF prediction and does not model transmitter power,
antenna patterns, takeoff angle, or the ionosphere along a selected endpoint.

### Compare two surveys

Version 0.10.0 compares completed scans and surveys without occupying the
receiver. First find saved observations of a band with
`list_comparable_surveys`:

```json
{
  "start_frequency_hz": 14000000,
  "stop_frequency_hz": 14350000,
  "endpoint_tolerance_hz": 2000,
  "limit": 20
}
```

Then pass two returned IDs to `compare_band_surveys`:

```json
{
  "baseline_job_id": "survey-BASELINE-ID",
  "comparison_job_id": "survey-CURRENT-ID",
  "frequency_tolerance_hz": 1500,
  "power_change_threshold_db": 6,
  "frequency_shift_threshold_hz": 250,
  "include_plot": true
}
```

The comparison uniquely matches the nearest carriers within the tolerance and
reports stable, changed, new, and disappeared signals. When both surveys
classified a matched carrier, it also identifies a changed best-label result.
The comparison JSON and two-panel change plot are saved as a persistent
`survey_comparison` job and can themselves be retrieved after a restart.

For v0.10 and earlier jobs, power changes require care: each scan was normalized
to its own strongest carrier. Those legacy deltas describe within-scan relative
power, so a different strongest carrier can shift every reported level.

### Repeatable digital power measurements

Version 0.11.0 records two complementary scales in every new band scan and
survey:

- the existing normalized `relative_power_db`, retained for plots, ranking, and
  backward compatibility;
- a Blackman-window-corrected power spectral density in `dBFS/Hz`, referenced
  to complex-float full scale rather than antenna-input watts.

Each detected carrier also includes `digital_power_dbfs_10khz`, calculated by
integrating the PSD across a fixed 10 kHz measurement bandwidth. Integrated
power is substantially less dependent on FFT size than a single-bin peak and
is the preferred value for repeat observations made with the same receiver
profile.

Every result records the sample rate, AGC state, attenuator setting, LNA state,
FFT size, and window. It also reports maximum IQ component level, RMS magnitude,
crest factor, clipped-component fraction, and `overload_suspected`. An overload
warning means the associated power and classification results may be distorted.

`compare_band_surveys` automatically uses the integrated digital scale when
both jobs have compatible fixed-gain profiles. Its
`power_comparison_scale` will then be `digital_power_dbfs_10khz`. Comparisons
involving older jobs or incompatible gain profiles fall back to
`legacy_relative_db` and retain the v0.10 warning about within-scan
normalization.

These values are repeatable receiver-output measurements, not calibrated dBm,
dBm/Hz, field strength, or antenna-terminal power. Converting them into those
units would require a frequency-dependent calibration against a known RF
source, including the selected Airspy gain configuration and external signal
path.

### Persistent presets and frequency watchlists

Version 0.12.0 stores named presets in the existing SQLite catalog. Four preset
types are supported: `band_scan`, `band_survey`, `monitor`, and `watchlist`.
Use `save_rf_preset`, `list_rf_presets`, `get_rf_preset`, `run_rf_preset`, and
`delete_rf_preset`.

Save a reusable 20-meter survey:

```json
{
  "name": "20 Meter Morning Survey",
  "preset_type": "band_survey",
  "description": "Fixed-gain 20m survey for daily comparisons",
  "replace_existing": false,
  "config": {
    "start_frequency_hz": 14000000,
    "stop_frequency_hz": 14350000,
    "capture_duration_seconds": 1,
    "attenuation_steps": 1,
    "classify_top_signals": 5
  }
}
```

Omitted fields are expanded to validated defaults when the preset is saved, so
`get_rf_preset` shows the complete effective configuration. Names are unique
without regard to case. Updating an existing name requires
`replace_existing=true` and preserves its stable `preset_id`.

Save a frequency watchlist:

```json
{
  "name": "WWV Watchlist",
  "preset_type": "watchlist",
  "description": "Check three WWV frequencies",
  "config": {
    "duration_seconds": 2,
    "analysis_bandwidth_hz": 30000,
    "fft_size": 16384,
    "entries": [
      {"frequency_hz": 5000000, "label": "WWV 5 MHz"},
      {"frequency_hz": 10000000, "label": "WWV 10 MHz"},
      {"frequency_hz": 15000000, "label": "WWV 15 MHz"}
    ]
  }
}
```

Run either preset by ID or case-insensitive name:

```json
{"preset_id_or_name": "WWV Watchlist"}
```

Scan, survey, and monitor presets launch their existing asynchronous job types.
Their job configuration records `source_preset_id`. A watchlist runs
synchronously, classifies each enabled frequency, and saves a consolidated
`watchlist_run` job containing labels, notes, results, errors, and child
classification job IDs. Watchlists are limited to ten entries to bound receiver
time and MCP request duration.

Preset deletion is guarded and does not delete jobs previously launched from
that preset:

```json
{
  "preset_id_or_name": "WWV Watchlist",
  "confirm_delete": true
}
```

### Persistent preset schedules

Version 0.13.0 can run any saved preset on a fixed interval. The scheduler
starts with the MCP service and reads its state from the existing SQLite
catalog. Create an hourly schedule for the survey above:

```json
{
  "name": "20m Hourly",
  "preset_id_or_name": "20 Meter Morning Survey",
  "interval_seconds": 3600,
  "start_at": "2026-08-11T15:00:00Z",
  "enabled": true,
  "replace_existing": false
}
```

`start_at` is optional. If omitted, the first run is one interval after the
schedule is saved. If supplied, it must include `Z` or a timezone offset.
Intervals may range from 60 seconds through seven days. Schedule names are
case-insensitively unique, and replacing one preserves its stable
`schedule_id`.

The v0.13 schedule tools are:

- `save_rf_schedule`, `list_rf_schedules`, and `get_rf_schedule`;
- `set_rf_schedule_enabled` to pause or resume a schedule;
- `run_rf_schedule_now`, which also works while a schedule is disabled;
- `get_rf_scheduler_status` for thread health and the next due time;
- `delete_rf_schedule`, guarded by `confirm_delete=true`.

After a restart, an overdue schedule runs once and advances from the current
time; it does not replay every missed interval. If another long-running RF job
owns the Airspy when a schedule is due, that occurrence is recorded as
`skipped_busy` and is not queued. Asynchronous scan, survey, and monitor runs
record `last_status=launched`; follow `last_job_id` with `get_rf_job` for final
state. Watchlists complete synchronously and record `last_status=completed`.
Every scheduled job records `source_schedule_id` as well as its source preset.

Deleting a schedule does not delete its prior jobs. A preset referenced by a
schedule cannot be deleted until the schedule is deleted. This release supports
fixed intervals, not cron expressions or local-calendar schedules.

### Persistent watchlist alerts

Version 0.14.0 evaluates alert rules after a scheduled watchlist finishes. It
stores events locally; it does not send email, SMS, webhooks, or any RF
transmission. Create a rule that records an event when the WWV 10 MHz entry is
classified as AM with at least 60% heuristic confidence:

```json
{
  "name": "WWV AM Detected",
  "schedule_id_or_name": "WWV Hourly",
  "condition_type": "classification_is",
  "entry_label": "WWV 10 MHz",
  "classification_label": "am",
  "min_confidence": 0.6,
  "enabled": true,
  "replace_existing": false
}
```

`entry_label` is optional. When present, it must match a label in the scheduled
watchlist and limits the rule to that entry. Without it, the rule is evaluated
against every watchlist observation. Supported conditions are:

- `classification_is`: match one of `am`, `usb`, `lsb`, `cw`, `nfm`, or
  `digital_or_unknown`, optionally requiring `min_confidence`;
- `confidence_at_least`: match any best classification at or above
  `min_confidence`;
- `peak_above_median_at_least`: require `threshold_db` in the classification
  features;
- `observation_failed`: record capture or classification failures;
- `ambiguous`: record observations the heuristic classifier marks ambiguous.

Manage rules with `save_rf_alert_rule`, `list_rf_alert_rules`,
`get_rf_alert_rule`, `set_rf_alert_rule_enabled`, and
`delete_rf_alert_rule`. Review events with `list_rf_alert_events` or
`get_rf_alert_event`, then mark one handled with:

```json
{"event_id": "alert-..."}
```

using `acknowledge_rf_alert_event`. Acknowledgement is idempotent. Event details
contain a snapshot of both the rule and matching watchlist observation. Deleting
a rule or its parent schedule does not delete historical events; their rule and
schedule IDs become null while snapshot names and details remain available.
Rule names are case-insensitively unique and replacement preserves the stable
`rule_id`.

### Signed alert webhooks

Version 0.15.0 can deliver stored alert events to HTTPS webhook receivers. A
destination may receive every alert or only events from one named rule. This
example subscribes an HTTPS endpoint to every alert:

```json
{
  "name": "RF Alert Receiver",
  "url": "https://example.net/hooks/rf",
  "signing_secret": "replace-with-at-least-16-random-characters",
  "enabled": true,
  "replace_existing": false
}
```

Use `save_rf_webhook_destination`. Add `rule_id_or_name` to subscribe only to
one v0.14 alert rule. Signing is optional, but strongly recommended. Secrets
are never returned by list or get tools; `has_signing_secret` indicates whether
one is configured. When replacing a destination, omitting `signing_secret`
preserves its existing secret.

Each POST uses `Content-Type: application/json` and contains the complete alert
event beneath a versioned `rf-mcp.alert.v1` envelope. Signed requests include:

```text
X-RF-MCP-Event: alert-...
X-RF-MCP-Timestamp: 2026-08-10T12:00:00+00:00
X-RF-MCP-Signature-256: sha256=...
```

The signature is lowercase hexadecimal HMAC-SHA256 over the UTF-8 bytes of
`timestamp + "." + request_body`, using the configured secret. Receivers should
verify the signature with a constant-time comparison and reject stale
timestamps.

Delivery state is written before any network request. Successful 2xx responses
become `delivered`. Network errors, HTTP 408/425/429, and 5xx responses retry up
to five total attempts with bounded backoff; other responses become `failed`.
Redirects are not followed. Use `list_rf_webhook_deliveries` to inspect the
outbox, `retry_rf_webhook_delivery` to reset one delivery, and
`get_rf_webhook_status` for dispatcher health and state counts.

HTTPS public destinations are allowed by default. Private, loopback,
link-local, and reserved addresses are rejected, and targets are validated
again immediately before delivery. For a trusted HTTPS service on your LAN,
add this to `/etc/rf-mcp.env` and restart the service:

```text
RF_MCP_ALLOW_PRIVATE_WEBHOOKS=true
```

Plain HTTP requires both explicit settings and should only be used on a trusted,
isolated network:

```text
RF_MCP_ALLOW_PRIVATE_WEBHOOKS=true
RF_MCP_ALLOW_INSECURE_WEBHOOKS=true
```

Manage destinations with `save_rf_webhook_destination`,
`list_rf_webhook_destinations`, `get_rf_webhook_destination`,
`set_rf_webhook_destination_enabled`, and
`delete_rf_webhook_destination`. Deleting a destination cancels its pending
retries but retains delivery history. Disabling it prevents future events from
being queued while leaving existing outbox items unchanged.

### Persistent history and artifacts

Version 0.6 adds these operational tools:

- `get_server_health` reports MiniRackDisplay, receiver, uptime, database,
  active-job, and storage health;
- `list_rf_jobs` and `get_rf_job` retrieve persisted work after restarts;
- `list_rf_artifacts` and `get_rf_artifact` retrieve old PNG, WAV, and JSON
  results through MCP using stable `artifact_id` values;
- `set_rf_artifact_pinned` protects important files from cleanup;
- `get_storage_status` summarizes usage by artifact type;
- `clean_old_artifacts` previews or removes old, unpinned artifacts.

The catalog database is `~/rf-mcp-data/rf-mcp.sqlite3`. Jobs left queued or
running when the service restarts are marked `interrupted`; completed artifacts
remain available. Existing files from earlier versions are indexed as standalone
artifacts when v0.6 starts.

Cleanup is deliberately conservative. First preview candidates:

```json
{
  "older_than_days": 30,
  "kinds": ["iq_capture"],
  "max_delete": 100,
  "dry_run": true,
  "confirm_delete": false
}
```

Actual deletion requires both `dry_run=false` and `confirm_delete=true`. Only
unpinned files located beneath `~/rf-mcp-data` are eligible. The default inline
artifact limit is 10 MiB; larger artifacts remain cataloged but are not embedded
in MCP results. Override it with `RF_MCP_MAX_INLINE_ARTIFACT_BYTES` if needed.

Version 0.8 adds a `download_path` to artifact metadata. Download any cataloged
file without base64 expansion at:

```text
http://MiniRackDisplay:8765/artifacts/ARTIFACT_ID
```

When authentication is enabled, supply the same bearer header. Downloads are
streamed in bounded chunks and resolved only through catalog IDs, preventing
requests for arbitrary filesystem paths.

### Classify a signal

Call `classify_signal` on a known carrier:

```json
{
  "frequency_hz": 10000000,
  "duration_seconds": 5,
  "analysis_bandwidth_hz": 30000,
  "fft_size": 16384,
  "include_plot": true,
  "include_preview_audio": true,
  "retain_iq": false
}
```

The classifier measures carrier strength, positive/negative sideband balance,
occupied bandwidth, envelope variation, instantaneous-frequency motion,
spectral entropy, and significant peak count. It returns normalized candidate
scores, the best label, a confidence margin, and an `ambiguous` flag. When the
best label has a supported analog demodulator, the result includes a WAV preview.

This is a deterministic heuristic classifier, not a trained model or protocol
decoder. Fading, interference, weak signals, unusual filtering, and digital
waveforms can produce ambiguous or incorrect labels. Downstream agents should
describe results as likely or provisional and retain the ranked alternatives.

### Station-local signal library

Version 0.39 adds identity matching on top of—not in place of—the generic
classifier. First run `classify_signal` while a known signal is present. Use
the returned `job_id` with `save_signal_fingerprint`, giving it a meaningful
local name such as `My thermostat telemetry` or `WWV-like 10 MHz reference`.
The library stores morphology features and a nominal-frequency tolerance; it
does not store or replay RF content.

Add observations made under different fading and modulation conditions with
`add_signal_fingerprint_exemplar`. Each fingerprint retains at most 20
exemplars and updates a descriptive centroid. Multiple exemplars are strongly
recommended before treating a match as useful.

Use `match_signal_classification_job` to compare an already-persisted
classification without occupying the receiver. `identify_live_signal` performs
a new classification and compares it with every saved fingerprint inside its
frequency tolerance. Results contain similarity, normalized feature distance,
frequency offset, generic-label agreement, and the three features contributing
the largest mismatch. `minimum_similarity` defaults to 0.55 and should be
tuned using known positive and negative examples at this station.

Signal strength is deliberately excluded from the identity distance because it
changes substantially with fading, antenna configuration, and propagation.
Frequency is used as a gate rather than proof of identity. Similarity is
empirical station-local evidence—not transmitter authentication, protocol
decoding, or a claim that two emitters are physically the same device. Manage
the library with `list_signal_fingerprints`, `get_signal_fingerprint`, and the
confirmation-guarded `delete_signal_fingerprint`.

### Scheduled station-memory monitoring

## v0.50 RF Operations dashboard

Version 0.50 turns the browser dashboard into an RF operations console. The new
Operations section summarizes the latest state and estimated SNR for every station
seen in a memory scan, plots completed and failed observations over recent rounds,
and marks rounds containing detected changes. It also combines memory-scan changes
with persisted alert events for review and acknowledgement.

The dashboard can create fixed-interval schedules from saved
`station_memory_scan` profiles, run them immediately, and enable or disable them.
These authenticated endpoints accept only narrowly validated JSON and delegate to
the same scheduler functions used by the MCP tools. Recent WAV artifacts can be
played directly from the Operations section. Navigation links provide quick access
to Operations, memories, spectrum, audio, FM/RDS, and system status.

Create a station-memory scan profile with `save_station_memory_scan_profile`, open
`http://MiniRackDisplay:8765/dashboard`, and use **Operations → Schedules** to set
its interval. A disabled schedule can still be run manually with **Run now**.

## v0.49 scheduled memory scans

Version 0.49 adds `station_memory_scan` to the persistent RF preset system.
Create a validated profile with `save_station_memory_scan_profile`, then run it
manually with `run_rf_preset` or attach it to the existing scheduler with
`save_rf_schedule`. Profiles may select explicit memory IDs/names or enabled
memories by exact tag and mode, and retain all v0.48 count, duration, and
120-second RF-time limits.

The fixed-interval scheduler already persists its next run, last attempt,
status, child job ID, and error. A restart performs at most one catch-up run,
then advances to the next interval. If another long-running receiver job is
active, the due run is recorded as `skipped_busy` rather than competing for the
Airspy. Station-memory rounds are synchronous and therefore recorded as
`completed` when their summary has been safely persisted.

With `compare_previous=true`, each round searches backward for the most recent
completed scan containing the same set of stable memory IDs. It reports:

- reception transitions between completed and failed;
- repeated reception failures;
- estimated-SNR changes meeting `snr_change_threshold_db`;
- changes to checksum-validated RDS Program Service names; and
- changes to checksum-validated RDS RadioText.

The summary includes `compared_to_job_id`, `change_count`, and structured
`changes`. A first run—or a run whose memory set has changed—establishes a new
baseline without manufacturing comparisons against an unrelated scan. Change
records are evidence for review, not transmitter authentication or calibrated
power measurements.

Create an hourly ham monitoring profile:

```json
{
  "name": "Hourly ham memories",
  "tag": "ham",
  "duration_seconds": 5,
  "max_memories": 10,
  "stop_on_error": false,
  "compare_previous": true,
  "snr_change_threshold_db": 6
}
```

Then schedule the returned preset:

```json
{
  "name": "Hourly ham schedule",
  "preset_id_or_name": "Hourly ham memories",
  "interval_seconds": 3600,
  "enabled": true
}
```

Use `list_rf_schedules`, `get_rf_schedule`, `set_rf_schedule_enabled`, and
`run_rf_schedule_now` for operations. Use `list_rf_jobs` filtered to
`station_memory_scan` and `get_rf_job` for monitoring history. Change detection
does not send a webhook by itself; external delivery remains opt-in and should
be configured deliberately through the existing alert/notification facilities.

### Scan station memories

Version 0.48 adds `scan_station_memories` for one bounded, sequential receive
round. Supply explicit memory names/IDs or omit them to scan enabled memories in
frequency order. Optional `tag` and `mode` filters narrow the selection, while
`max_memories` limits the round to at most 20 entries.

The server enforces no more than 120 seconds of planned RF time per call.
`duration_seconds` applies to each selected memory and remains capped at ten
seconds. A round containing Broadcast FM must use five or ten seconds. Inline
media is suppressed during a scan, but each child reception retains its normal
job, WAV/plots, metrics, and RDS artifacts in the catalog.

By default, one failed station is recorded and the scan continues.
`stop_on_error=true` ends the round after the first failure. The persistent
`station_memory_scan` summary reports requested, attempted, completed, and
failed counts; planned RF time; each memory’s exact frequency/mode; child job
IDs; useful metrics/RDS data; and individual error messages.

Example ham-memory round:

```json
{
  "tag": "ham",
  "duration_seconds": 5,
  "max_memories": 10,
  "stop_on_error": false
}
```

Example selected round:

```json
{
  "memory_ids_or_names": ["WWV 10 MHz", "Local FM"],
  "duration_seconds": 5,
  "max_memories": 2
}
```

This is a finite inspection round, not continuous monitoring or a scheduler.
Existing watchlists, schedules, and monitors remain the right tools for ongoing
operation.

### Receive a station memory

Version 0.47 adds `receive_station_memory`, which accepts a saved memory ID or
case-insensitive name and invokes the correct established receiver pipeline.
AM, NFM, USB, LSB, and CW memories use `analyze_signal` with the saved frequency
and bandwidth. Broadcast FM memories use `receive_broadcast_fm` with stereo,
de-emphasis, and RDS options.

Captures are explicitly requested, limited to ten seconds, and always use
`retain_iq=false`. Disabled memories cannot be received. `include_media=false`
suppresses inline MCP audio/images while the normal catalog artifacts remain
available. Broadcast FM duration must be five or ten seconds; other modes
accept the normal 0.25-to-10-second range.

The dashboard station table now offers both **Recall** and **Listen**. Recall
only fills the controls as before. Listen first recalls the memory and then
submits the visible receive form, providing an unambiguous operator action and
the same progress, WAV player, plots, RDS display, authentication, and error
handling as a manual form submission.

Example MCP request:

```json
{
  "memory_id_or_name": "WWV 10 MHz",
  "duration_seconds": 5,
  "include_media": true
}
```

The returned structured result includes the exact station-memory record used,
making the job traceable even if that memory is edited later.

### Station memories

Version 0.46 adds a persistent station-memory bank shared by MCP and the web
dashboard. Each memory contains a stable ID, name, receive frequency, mode,
validated bandwidth, notes, tags, enabled state, and creation/update times.
Supported modes are AM, NFM, USB, LSB, CW, and dedicated `broadcast_fm`.

Create and update entries with `save_station_memory`. Names are unique without
regard to case; replacing a name requires `replace_existing=true`, while
updating by `memory_id` preserves the stable identity. Use
`list_station_memories` to search names, notes, and tags or filter by mode and
enabled state. `get_station_memory` accepts an ID or name, and
`delete_station_memory` requires `confirm_delete=true`.

The dashboard displays enabled memories in frequency order. Selecting
**Recall** fills the appropriate controls: Broadcast FM memories populate the
MHz field, while AM/NFM/USB/LSB/CW memories populate frequency, mode, and
bandwidth in **Tune and listen**. Spectrum center frequency is also updated.
Recall deliberately does not start a capture; the operator must review the
settings and select the receive action.

Memories use conservative Airspy HF+ tuning-range validation. Broadcast FM is
limited to 88–108 MHz and normalized to 200 kHz bandwidth. Other modes reuse
the same bandwidth validation as `analyze_signal`. Memory metadata is stored
atomically in `~/rf-mcp-data/station-memories.json`; it contains configuration
only, not IQ or audio content.

### Browser Broadcast FM and RDS

Version 0.45 adds a dedicated **Broadcast FM and RDS** dashboard panel. Enter
the station frequency directly in MHz (for example `100.1`), select a five- or
ten-second capture, choose stereo or mono, and select 75 µs de-emphasis for the
Americas or 50 µs where appropriate. The resulting WAV plays in the browser and
the multiplex plot shows the recovered WFM baseband components.

RDS decoding is enabled for dashboard requests. When checksum-valid groups are
present, the page displays the Program Service name, RadioText, and accepted
group count. Ten seconds is normally a better starting point than five, but a
short capture may still miss slowly repeated fields. Missing RDS text does not
mean the station lacks RDS; weak signals, multipath, tuning error, and capture
length all affect decoding.

The dashboard limits this workflow to the conventional 88–108 MHz broadcast
band, even though the underlying MCP tool supports a slightly wider engineering
range. Only five- and ten-second recordings are accepted. IQ retention is
forced off, inline media is suppressed, and the cataloged WAV and multiplex
plot are streamed through authenticated artifact URLs.

This panel calls the established `receive_broadcast_fm` implementation, so its
stereo-pilot detection, de-emphasis, RDS checksum rules, capture serialization,
job history, and artifact handling remain identical to MCP operation.

### Browser tune and listen

Version 0.44 adds a **Tune and listen** panel beneath live spectrum inspection.
Enter the signal frequency in Hz, choose AM, narrow FM, USB, LSB, or CW, select
the bandwidth and a two-, five-, or ten-second duration, then select
**Record audio**. The resulting 48 kHz WAV plays in the browser, accompanied by
the RF spectrum and demodulated-audio spectrum plots.

Changing mode automatically selects a sensible starting bandwidth: 10 kHz AM,
12.5 kHz NFM, 3 kHz USB/LSB, or 500 Hz CW. The server independently validates
the applicable range for every mode and rejects unsupported fields. CW uses the
existing 700 Hz default sidetone.

Browser demodulation calls the same `analyze_signal` pipeline used by MCP. IQ
retention is forced off, captures cannot exceed ten seconds, and WAV/plot files
are registered as normal artifacts. Inline base64 audio and plots are disabled
for the API response; the browser streams the cataloged WAV and images through
their protected artifact URLs instead.

The audio is an offline recording made after each request, not an unbounded
real-time stream. Simultaneous requests remain subject to the Airspy device
lock. Audio quality depends on correct frequency, mode, bandwidth, signal
strength, interference, and propagation; level and SNR estimates are relative,
not calibrated receiver-input measurements.

### Browser spectrum inspection

Version 0.43 adds a **Live spectrum inspection** panel to the web dashboard.
Enter the center frequency in integer Hz, select a one-, two-, five-, or
ten-second capture and FFT size, adjust the relative peak threshold if needed,
then select **Inspect**. The page displays the resulting spectrum plot, relative
noise floor, and detected peak count and refreshes the job/artifact lists.

Browser requests call the same `inspect_spectrum` implementation exposed by
MCP. They therefore inherit Airspy tuning validation, the capture lock,
duration and peak limits, passband filtering, persistent job history, and
stable artifact registration. Processing runs outside the web event loop so
health and dashboard refresh requests remain responsive during capture.

The browser endpoint accepts only five fields: center frequency, duration, FFT
size, peak threshold, and maximum peaks. IQ retention is always forced off and
plots are cataloged normally. Unsupported fields are rejected rather than
ignored. Levels remain relative and must not be interpreted as calibrated dBm.

Spectrum capture requires an authenticated dashboard session or bearer header.
Artifact images may be viewed using that dashboard session, while the MCP
endpoint itself continues to require the bearer header and does not accept a
dashboard cookie as substitute authentication.

### MiniRackDisplay web dashboard

Version 0.42 adds a responsive, read-only operations dashboard at:

```text
http://MiniRackDisplay:8765/dashboard
```

It summarizes service state, the multi-SDR inventory and active leases, recent
RF jobs, recent downloadable artifacts, and catalog storage. The page refreshes
every 15 seconds and requires no Internet connection, CDN, or browser plugin.
It does not expose controls that can start captures, alter schedules, delete
data, or transmit RF; those operations remain in the authenticated MCP tools.

When `RF_MCP_API_TOKEN` is configured, opening the dashboard displays a local
login form. Enter the same API token used by MCP Inspector. A random, HTTP-only,
same-site browser session lasts at most 12 hours and can be ended immediately
with **Sign out**. The token itself is not stored in the browser cookie. On an
unauthenticated private installation, the dashboard opens directly.

The machine-readable summary is available at `/api/dashboard`. It accepts the
normal `Authorization: Bearer ...` header or an active dashboard session.
`/healthz` remains public and intentionally reports only basic service status.
Artifact links retain the existing authentication and stable catalog-ID checks.

Like the MCP endpoint, the dashboard is designed for a trusted private LAN.
HTTP does not encrypt the login token in transit. Use a VPN or TLS reverse proxy
before accessing it across an untrusted network, and never expose port 8765
directly to the public Internet.

### Multi-SDR coordinator

Version 0.41 adds a persistent receiver inventory and dry-run assignment planner
for Airspy HF+, RTL-SDR, HackRF, Pluto through SoapySDR, and WEB-888. Existing
analysis and decoder tools continue to use the proven Airspy HF+ path; the
coordinator does not silently move them to hardware that has not been verified.

Start with `discover_sdr_receivers`. Its default `probe_hardware=false` only
checks whether each backend executable is installed. Set it to true while all
receivers are idle to run bounded information/test probes; no IQ capture is
started and discovery never creates inventory entries.

`list_sdr_receivers` creates the default verified `airspyhf-primary` entry on
first use. Add other devices with `save_sdr_receiver`, initially using
`verified=false`. Give each a role (`general`, `primary_hf`, `vhf_uhf_monitor`,
`wideband_survey`, `satellite`, or `experimental`), a 0-to-100 priority, and an
unambiguous device selector such as an index, serial, or Soapy device string.
The built-in tuning limits are conservative defaults and can be overridden for
the exact hardware variant.

Use `plan_sdr_assignment` with a frequency, required bandwidth, and optional
preferred role. It is always a dry run: the result ranks eligible devices and
explains every rejection (disabled, unverified, out of range, too narrow, or
busy). The safe default `require_verified=true` prevents experimental inventory
entries from being selected.

`acquire_sdr_receiver` and `release_sdr_receiver` provide cooperative,
per-device durable leases. Two independent receivers may be leased
simultaneously, but the same receiver cannot be claimed twice across server
processes. Expired leases are reclaimed and active streams send heartbeats.
Inspect them with `get_sdr_coordinator_status`. Receiver deletion is confirmation
guarded and is rejected while leased.

Airspy HF+ and RTL-SDR have capture adapters. HackRF, Pluto/SoapySDR, and Web-888
remain inventory/planning entries and fail explicitly if selected for capture.
The server is receive-only and never enables transmit functionality.

### Recording and review workspace

Version 0.40 groups existing jobs and artifacts into persistent review
sessions. `create_recording_session` accepts individual `artifact_ids`, whole
`job_ids`, or both. Job attachment resolves its currently cataloged artifacts;
sessions retain stable IDs and descriptive metadata rather than copying the
underlying files. Add more material later with `add_recording_session_items`.

`add_recording_annotation` creates session-wide notes or time ranges tied to an
attached artifact. Tags, session names, descriptions, filenames, annotation
text, annotation tags, bookmark labels, and bookmark notes are searched by
`search_recording_sessions`. `add_recording_bookmark` validates a point against
the actual duration of a cataloged WAV and attaches that recording to the
session when necessary.

`extract_recording_clip` copies a bounded 0.05-to-600-second region from a WAV
into a new cataloged audio artifact without modifying the source. A requested
clip that reaches past the end is safely shortened and reports its actual
duration. Supply `session_id_or_name` to attach the new clip automatically.
Compressed audio formats are not silently transcoded; clip and bookmark tools
currently require WAV.

`compare_recording_audio` compares at most the first 120 seconds of two WAV
artifacts. It reports RMS levels, RMS difference, alignment-sensitive waveform
correlation, spectral centroids, and difference RMS, and can return a waveform
and spectrum plot. These metrics assist review but do not authenticate a
speaker, transmitter, or content source.

`export_recording_session` creates a JSON manifest and annotation CSV.
`delete_recording_session` requires confirmation and deletes only the session
metadata; attached source recordings, extracted clips, jobs, and artifacts are
retained. Normal artifact retention and explicit cleanup rules still apply.

## Install as a systemd service

Only do this after interactive and Inspector tests pass:

```bash
cd ~/rf-mcp
chmod +x scripts/install-service.sh
./scripts/install-service.sh
```

Useful service commands:

```bash
systemctl status rf-mcp
journalctl -u rf-mcp -f
sudo systemctl restart rf-mcp
```

## Measurement behavior

- Captures use 768 kS/s because Debian's `airspyhf_rx` command currently
  advertises only that capture rate.
- The analyzer excludes 12% at each filter edge and 1.5 kHz around center/DC.
- Levels are relative dB, normalized to the strongest FFT bin. They are not
  calibrated dBm.
- IQ captures are deleted after analysis unless `retain_iq=true`.
- Captures are serialized so two clients cannot claim the receiver at once.
- Receiver leases are persisted in `rf-mcp-data/sdr-coordinator.sqlite3`, so a
  second server process cannot claim hardware already in use. Capture leases
  expire after 30 minutes if their owner disappears; active streams renew their
  lease every 30 seconds.
- Duration defaults to two seconds and is limited to ten seconds. Override the
  upper bound with `RF_MCP_MAX_DURATION`, with care.

### Receiver calibration and qualification

Use `qualify_sdr_receiver` after installing or moving a receiver. It probes the
device, makes a short capture at the requested frequency, verifies capture
length, checks digital overload indicators, and reports the calibration applied.
This is an operational test, not a traceable RF calibration.

`save_receiver_calibration` stores frequency correction in PPM and an optional
`dbfs_to_dbm_offset_db`. RTL-SDR captures automatically receive the stored PPM
correction. A dBm offset requires a non-empty `reference_source`, such as the
signal-generator level and setup used to derive it. Spectrum results then include
both the original dBFS values and calibrated dBm/Hz or integrated dBm values,
along with the complete calibration profile. Without that offset, measurements
remain explicitly relative/digital-domain values.

Related tools are `get_receiver_calibration`, `list_receiver_calibrations`, and
the confirmation-protected `delete_receiver_calibration`.

## Stable API 1.0 contract

Call `get_rf_api_contract` to retrieve the machine-readable stable core tool
list, units, measurement rules, and compatibility policy. Throughout the 1.x
line, stable tools will not lose required parameters or documented response
fields. Minor releases may add tools, optional parameters, enum values, and
response fields; clients should ignore fields they do not recognize. Breaking
changes require a new major version.

Call `get_release_readiness` before production deployment. Its default checks
are non-destructive and do not touch the receiver. Set `probe_hardware=true` to
also run the supported backend probes. A private-LAN bind without bearer
authentication is reported as a required readiness failure.

Release history is in `CHANGELOG.md`; deployment and vulnerability guidance is
in `SECURITY.md`; development expectations are in `CONTRIBUTING.md`.

## Security

This release is intended for a trusted private LAN. It is receive-only, but a
client can initiate captures and consume CPU/storage. Bearer authentication is
strongly recommended on a shared LAN. Do not expose the HTTP service publicly;
use a TLS reverse proxy or VPN before crossing an untrusted network.

## Restart and power-loss recovery

The artifact catalog uses an explicitly versioned SQLite schema. On startup,
jobs left in `queued`, `running`, or `stopping` state are marked `interrupted`
with a recovery explanation; their existing artifacts are retained. Expired
receiver leases are removed automatically on the next coordinator operation.
Call `get_rf_recovery_status` to inspect the schema version, startup recovery
count, active durable leases, and current recovery policy.

## Upgrade an existing installation

The service is deliberately fixed to `~/rf-mcp`; do not extract a release into
a version-suffixed directory. Assuming the downloaded ZIP is in your home
directory, extract its project files directly into the existing installation:

```bash
cd ~
sudo systemctl stop rf-mcp
unzip -o rf-mcp-multi-sdr-v1.0.1.zip -d rf-mcp
cd ~/rf-mcp
source .venv/bin/activate
python -m pip install -e .
pytest -q
chmod +x scripts/install-service.sh scripts/configure-auth.sh
./scripts/install-service.sh
```

`scripts/install-service.sh` installs or refreshes the systemd unit and starts
the service. There is no top-level `install.sh`. Decoder installation scripts
under `scripts/` are optional and do not need to be rerun during an ordinary
application upgrade.

Reconnect or refresh the server in MCP Inspector so it reloads the updated
`inspect_spectrum` input and output schemas.
