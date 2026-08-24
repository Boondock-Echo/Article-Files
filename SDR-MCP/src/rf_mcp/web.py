from __future__ import annotations

import asyncio
import hmac
import html
import json
import math
import re
import secrets
import time
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qs, quote

from .services import RfApplicationServices


ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]
_ARTIFACT_PATH = re.compile(r"^/artifacts/(?P<artifact_id>art-[0-9a-f]+)$")
_SSTV_IMAGE_PATH = re.compile(r"^/sstv-images/(?P<image_id>[A-Za-z0-9_-]+)$")
_SESSION_SECONDS = 12 * 60 * 60


def validate_api_token(token: str | None) -> str | None:
    """Validate an optional bearer token without ever logging its value."""
    if token is None or not token.strip():
        return None
    token = token.strip()
    if len(token) < 32:
        raise ValueError("RF_MCP_API_TOKEN must contain at least 32 characters")
    if re.fullmatch(r"[A-Za-z0-9._~-]+", token) is None:
        raise ValueError(
            "RF_MCP_API_TOKEN may contain only letters, numbers, dot, underscore, tilde, and hyphen"
        )
    return token


def _headers(scope: dict) -> dict[bytes, bytes]:
    return {key.lower(): value for key, value in scope.get("headers", [])}


def authorized(scope: dict, token: str | None) -> bool:
    if token is None:
        return True
    value = _headers(scope).get(b"authorization", b"")
    prefix = b"Bearer "
    if not value.startswith(prefix):
        return False
    try:
        candidate = value[len(prefix) :].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return hmac.compare_digest(candidate, token)


async def _response(
    send: Callable,
    status: int,
    body: bytes,
    content_type: bytes = b"application/json",
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    headers = [(b"content-type", content_type), (b"content-length", str(len(body)).encode())]
    headers.extend(extra_headers or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _json_safe(value):
    """Return a strict-JSON-safe copy, replacing non-finite measurements with null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(_json_safe(value), separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


class RfWebApp:
    """ASGI boundary providing bearer auth, dashboard, health, and downloads."""

    def __init__(self, mcp_app: ASGIApp, catalog, token: str | None, version: str,
                 spectrum_capture: Callable | None = None,
                 signal_analyzer: Callable | None = None,
                 broadcast_fm_receiver: Callable | None = None,
                 schedule_create: Callable | None = None,
                 schedule_run: Callable | None = None,
                 schedule_toggle: Callable | None = None,
                 scan_profile_create: Callable | None = None,
                 scan_profile_run: Callable | None = None,
                 station_memory_create: Callable | None = None,
                 station_memory_delete: Callable | None = None,
                 scan_profile_delete: Callable | None = None,
                 schedule_delete: Callable | None = None,
                 band_scan_start: Callable | None = None,
                 band_survey_start: Callable | None = None,
                 band_job_status: Callable | None = None,
                 band_job_stop: Callable | None = None,
                 preset_run: Callable | None = None,
                 fm_survey_start: Callable | None = None,
                 fm_survey_status: Callable | None = None,
                 fm_survey_stop: Callable | None = None,
                 digital_decode: Callable | None = None,
                 weak_decode: Callable | None = None,
                 fldigi_decode: Callable | None = None,
                 decoder_capabilities: Callable | None = None,
                 sstv_decode: Callable | None = None,
                 sstv_watch_start: Callable | None = None,
                 sstv_status: Callable | None = None,
                 sstv_watch_status: Callable | None = None,
                 sstv_stop: Callable | None = None,
                 sstv_watch_stop: Callable | None = None,
                 sstv_capabilities: Callable | None = None,
                 services: RfApplicationServices | None = None):
        self.mcp_app = mcp_app
        self.catalog = catalog
        self.token = validate_api_token(token)
        self.version = version
        self.spectrum_capture = spectrum_capture
        self.signal_analyzer = signal_analyzer
        self.broadcast_fm_receiver = broadcast_fm_receiver
        self.schedule_create = schedule_create
        self.schedule_run = schedule_run
        self.schedule_toggle = schedule_toggle
        self.scan_profile_create = scan_profile_create
        self.scan_profile_run = scan_profile_run
        self.station_memory_create = station_memory_create
        self.station_memory_delete = station_memory_delete
        self.scan_profile_delete = scan_profile_delete
        self.schedule_delete = schedule_delete
        self.band_scan_start = band_scan_start
        self.band_survey_start = band_survey_start
        self.band_job_status = band_job_status
        self.band_job_stop = band_job_stop
        self.preset_run = preset_run
        self.fm_survey_start = fm_survey_start
        self.fm_survey_status = fm_survey_status
        self.fm_survey_stop = fm_survey_stop
        self.digital_decode = digital_decode
        self.weak_decode = weak_decode
        self.fldigi_decode = fldigi_decode
        self.decoder_capabilities = decoder_capabilities
        self.sstv_decode = sstv_decode
        self.sstv_watch_start = sstv_watch_start
        self.sstv_status = sstv_status
        self.sstv_watch_status = sstv_watch_status
        self.sstv_stop = sstv_stop
        self.sstv_watch_stop = sstv_watch_stop
        self.sstv_capabilities = sstv_capabilities
        self.services = services
        if services is not None:
            self.spectrum_capture = services.spectrum_capture
            self.signal_analyzer = services.signal_analyzer
            self.broadcast_fm_receiver = services.broadcast_fm_receiver
        self._sessions: dict[str, float] = {}
        self._dashboard_assets: tuple[bytes, bytes, bytes] | None = None

    def _dashboard_authorized(self, scope: dict) -> bool:
        if authorized(scope, self.token):
            return True
        cookie = _headers(scope).get(b"cookie", b"").decode("latin-1", "ignore")
        values = {}
        for part in cookie.split(";"):
            if "=" in part:
                key, value = part.strip().split("=", 1)
                values[key] = value
        session = values.get("rf_mcp_session", "")
        expires = self._sessions.get(session, 0)
        if expires <= time.time():
            self._sessions.pop(session, None)
            return False
        return True

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.mcp_app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/healthz":
            await _response(
                send,
                200,
                _json_bytes(
                    {
                        "status": "ok",
                        "service": "rf-mcp",
                        "version": self.version,
                        "authentication_required": self.token is not None,
                    }
                ),
            )
            return

        if path in {"/", "/dashboard"}:
            if not self._dashboard_authorized(scope):
                await self._login_page(send)
            else:
                await _response(send, 200, self._dashboard_html(), b"text/html; charset=utf-8",
                                self._security_headers())
            return

        if path in {"/assets/rf-dashboard.css", "/assets/rf-dashboard.js"}:
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}))
            else:
                _document, stylesheet, script = self._frontend_assets()
                if path.endswith(".css"):
                    await _response(send, 200, stylesheet, b"text/css; charset=utf-8",
                                    self._security_headers())
                else:
                    await _response(send, 200, script, b"text/javascript; charset=utf-8",
                                    self._security_headers())
            return

        if path == "/dashboard/login" and scope.get("method") == "POST":
            await self._login(scope, receive, send)
            return

        if path == "/dashboard/logout" and scope.get("method") == "POST":
            cookie = _headers(scope).get(b"cookie", b"").decode("latin-1", "ignore")
            match = re.search(r"(?:^|;\s*)rf_mcp_session=([^;]*)", cookie)
            if match:
                self._sessions.pop(match.group(1), None)
            await _response(send, 303, b"", b"text/plain",
                            [(b"location", b"/dashboard"),
                             (b"set-cookie", b"rf_mcp_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")])
            return

        if path == "/api/dashboard":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._dashboard_data(send)
            return

        operations = {
            "/api/receivers/discover": self._discover_receivers,
            "/api/receivers/register": self._register_receiver,
            "/api/station-memories": self._create_station_memory,
            "/api/station-memories/delete": self._delete_station_memory,
            "/api/station-scan-profiles": self._create_station_scan_profile,
            "/api/station-scan-profiles/run": self._run_station_scan_profile,
            "/api/station-scan-profiles/delete": self._delete_station_scan_profile,
            "/api/station-schedules": self._create_station_schedule,
            "/api/station-schedules/run": self._run_station_schedule,
            "/api/station-schedules/toggle": self._toggle_station_schedule,
            "/api/station-schedules/delete": self._delete_station_schedule,
            "/api/band-scan/start": self._start_band_scan,
            "/api/band-survey/start": self._start_band_survey,
            "/api/band-jobs/status": self._band_job_status,
            "/api/band-jobs/stop": self._stop_band_job,
            "/api/presets/run": self._run_preset,
            "/api/fm-surveys/start": self._start_fm_survey,
            "/api/fm-surveys/status": self._fm_survey_status,
            "/api/fm-surveys/stop": self._stop_fm_survey,
            "/api/digital/native": self._decode_native_digital,
            "/api/digital/weak": self._decode_weak_digital,
            "/api/digital/fldigi": self._decode_fldigi,
            "/api/digital/capabilities": self._digital_capabilities,
            "/api/sstv/decode": self._decode_sstv,
            "/api/sstv/watch": self._start_sstv_watch,
            "/api/sstv/status": self._sstv_job_status,
            "/api/sstv/watch-status": self._sstv_watcher_status,
            "/api/sstv/stop": self._stop_sstv,
            "/api/sstv/watch-stop": self._stop_sstv_watch,
            "/api/sstv/capabilities": self._sstv_decoder_capabilities,
            "/api/alerts/acknowledge": self._acknowledge_alert,
        }
        if path in operations and scope.get("method") == "POST":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await operations[path](receive, send)
            return

        if path == "/api/spectrum" and scope.get("method") == "POST":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._capture_spectrum(receive, send)
            return

        if path == "/api/demodulate" and scope.get("method") == "POST":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._demodulate(receive, send)
            return

        if path == "/api/broadcast-fm" and scope.get("method") == "POST":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._broadcast_fm(receive, send)
            return

        match = _ARTIFACT_PATH.fullmatch(path)
        if match:
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._artifact(match.group("artifact_id"), send)
            return

        image_match = _SSTV_IMAGE_PATH.fullmatch(path)
        if image_match:
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._sstv_image(image_match.group("image_id"), send)
            return

        if not authorized(scope, self.token):
            await _response(
                send,
                401,
                _json_bytes({"error": "unauthorized"}),
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        await self.mcp_app(scope, receive, send)

    @staticmethod
    def _security_headers() -> list[tuple[bytes, bytes]]:
        return [(b"cache-control", b"private, no-store"),
                (b"x-content-type-options", b"nosniff"),
                (b"x-frame-options", b"DENY"),
                (b"referrer-policy", b"no-referrer"),
                (b"content-security-policy", b"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:")]

    async def _login_page(self, send: Callable, failed: bool = False) -> None:
        message = '<p class="error">That token was not accepted.</p>' if failed else ""
        disabled = "" if self.token is not None else '<p>Authentication is not enabled. Reload the dashboard.</p>'
        body = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniRackDisplay Login</title><style>
body{{margin:0;background:#08111f;color:#e5edf8;font:16px system-ui;display:grid;place-items:center;min-height:100vh}}main{{width:min(390px,calc(100% - 40px));background:#111e31;border:1px solid #29405f;border-radius:16px;padding:28px;box-shadow:0 20px 60px #0008}}h1{{margin:0 0 6px;color:#fff}}p{{color:#9fb1ca}}label{{display:block;margin:24px 0 8px}}input{{box-sizing:border-box;width:100%;padding:12px;border-radius:8px;border:1px solid #49617f;background:#08111f;color:#fff}}button{{width:100%;margin-top:16px;padding:12px;border:0;border-radius:8px;background:#f36d2e;color:#fff;font-weight:700}}.error{{color:#ff9c8b}}
</style></head><body><main><h1>MiniRackDisplay</h1><p>RF MCP dashboard · v{html.escape(self.version)}</p>{message}{disabled}<form method="post" action="/dashboard/login"><label for="token">API token</label><input id="token" name="token" type="password" autocomplete="current-password" required><button>Open dashboard</button></form></main></body></html>"""
        await _response(send, 401, body.encode(), b"text/html; charset=utf-8", self._security_headers())

    async def _login(self, scope: dict, receive: Callable, send: Callable) -> None:
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > 8192:
                await _response(send, 413, b"request too large", b"text/plain")
                return
            if not message.get("more_body", False):
                break
        supplied = parse_qs(body.decode("utf-8", "replace")).get("token", [""])[0]
        if self.token is None or not hmac.compare_digest(supplied, self.token):
            await self._login_page(send, failed=True)
            return
        session = secrets.token_urlsafe(32)
        self._sessions[session] = time.time() + _SESSION_SECONDS
        cookie = f"rf_mcp_session={session}; Path=/; HttpOnly; SameSite=Strict; Max-Age={_SESSION_SECONDS}".encode()
        await _response(send, 303, b"", b"text/plain",
                        [(b"location", b"/dashboard"), (b"set-cookie", cookie)])

    async def _dashboard_data(self, send: Callable) -> None:
        try:
            from .operations import active_long_job
            from .sdr_coordinator import coordinator_status, ensure_airspy_default
            from .station_memory import list_memories
            ensure_airspy_default()
            jobs = self.catalog.list_jobs(limit=30)
            active_job_id = active_long_job()
            active_rf_job = None
            if active_job_id:
                try:
                    active_rf_job = self.catalog.get_job(active_job_id)
                except ValueError:
                    active_rf_job = {"job_id": active_job_id, "job_type": "rf_operation",
                                     "state": "running", "config": {}, "summary": {}}
            artifacts = self.catalog.list_artifacts(limit=12)
            profiles = self.catalog.list_presets(preset_type="station_memory_scan", limit=100)
            rf_presets = self.catalog.list_presets(limit=100)
            schedules = [item for item in self.catalog.list_schedules(limit=100)
                         if item.get("preset_type") == "station_memory_scan"]
            scan_jobs = self.catalog.list_jobs(
                job_type="station_memory_scan", state="completed", limit=24)
            scan_history = []
            station_status: dict[str, dict] = {}
            for job in reversed(scan_jobs):
                result = (self.catalog.get_job(job["job_id"]).get("result") or {})
                scan_history.append({
                    "job_id": job["job_id"], "created_at": job.get("created_at"),
                    "completed_count": result.get("completed_count", 0),
                    "failed_count": result.get("failed_count", 0),
                    "change_count": result.get("change_count", 0),
                    "changes": result.get("changes", []),
                })
                for observation in result.get("observations", []):
                    metrics = observation.get("metrics") or {}
                    station_status[observation["memory_id"]] = {
                        "memory_id": observation["memory_id"],
                        "name": observation.get("name"),
                        "frequency_hz": observation.get("frequency_hz"),
                        "mode": observation.get("mode"),
                        "state": observation.get("state"),
                        "estimated_snr_db": metrics.get("estimated_snr_db"),
                        "observed_at": result.get("completed_at") or job.get("created_at"),
                        "job_id": job["job_id"],
                    }
            alerts = self.catalog.list_alert_events(limit=30)
            fm_stations = self.catalog.list_fm_stations(limit=200)
            fm_survey_jobs = self.catalog.list_jobs(job_type="fm_broadcast_survey", limit=20)
            weak_spots = self.catalog.list_weak_signal_spots(limit=100)
            fldigi_decodes = self.catalog.list_fldigi_decodes(limit=50)
            sstv_images = self.catalog.list_sstv_images(limit=100)
            sstv_jobs = (self.catalog.list_jobs(job_type="sstv_decode", limit=20) +
                         self.catalog.list_jobs(job_type="sstv_watch", limit=20))
            result = {"status": "ok", "server_name": "MiniRackDisplay", "version": self.version,
                      "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "active_long_job": active_job_id, "active_rf_job": active_rf_job,
                      "coordinator": coordinator_status(),
                      "storage": self.catalog.storage_status(), "jobs": jobs,
                      "station_memories": list_memories(enabled_only=True),
                      "station_scan_profiles": profiles,
                      "rf_presets": rf_presets,
                      "station_schedules": schedules,
                      "station_scan_history": scan_history,
                      "station_status": list(station_status.values()),
                      "recent_alerts": alerts,
                      "fm_stations": fm_stations,
                      "fm_survey_jobs": fm_survey_jobs,
                      "weak_signal_spots": weak_spots,
                      "fldigi_decodes": fldigi_decodes,
                      "sstv_images": [{**item, "image_url":
                                       f"/sstv-images/{item['image_id']}"}
                                      for item in sstv_images],
                      "sstv_jobs": sstv_jobs,
                      "artifacts": [{**item, "download_path": f"/artifacts/{item['artifact_id']}"}
                                    for item in artifacts]}
            await _response(send, 200, _json_bytes(result), extra_headers=self._security_headers())
        except Exception as exc:
            await _response(send, 500, _json_bytes({"error": "dashboard_unavailable",
                                                    "detail": f"{type(exc).__name__}: {exc}"}),
                            extra_headers=self._security_headers())

    async def _json_request(self, receive: Callable) -> dict:
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > 8192:
                raise ValueError("request body exceeds 8192 bytes")
            if not message.get("more_body", False):
                break
        values = json.loads(body or b"{}")
        if not isinstance(values, dict):
            raise ValueError("request body must be a JSON object")
        return values

    async def _operation(self, receive: Callable, send: Callable, *, allowed: set[str],
                         callback: Callable, unavailable: str,
                         validator: Callable[[dict], None] | None = None) -> None:
        if callback is None:
            await _response(send, 503, _json_bytes({"error": unavailable}),
                            extra_headers=self._security_headers())
            return
        try:
            values = await self._json_request(receive)
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"unsupported fields: {', '.join(unknown)}")
            if validator:
                validator(values)
            result = await asyncio.to_thread(callback, **values)
            await _response(send, 200, _json_bytes({"status": "ok", "result": result}),
                            extra_headers=self._security_headers())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_operation",
                                                    "detail": str(exc)}),
                            extra_headers=self._security_headers())
        except Exception as exc:
            await _response(send, 409, _json_bytes({"error": "operation_failed",
                                                    "detail": f"{type(exc).__name__}: {exc}"}),
                            extra_headers=self._security_headers())

    async def _discover_receivers(self, receive: Callable, send: Callable) -> None:
        from .sdr_coordinator import discover_devices
        callback = self.services.receivers.discover if self.services else discover_devices
        await self._operation(
            receive, send, allowed=set(), callback=callback,
            unavailable="receiver_discovery_unavailable",
        )

    async def _register_receiver(self, receive: Callable, send: Callable) -> None:
        from .sdr_coordinator import register_discovered_device
        callback = self.services.receivers.register if self.services else register_discovered_device
        await self._operation(
            receive, send,
            allowed={"backend", "device_selector", "receiver_id", "name", "role", "priority"},
            callback=callback,
            unavailable="receiver_registration_unavailable",
        )

    async def _create_station_schedule(self, receive: Callable, send: Callable) -> None:
        def validate(values: dict) -> None:
            required = {"name", "preset_id_or_name", "interval_seconds"}
            missing = sorted(required - set(values))
            if missing:
                raise ValueError(f"missing required fields: {', '.join(missing)}")
            preset = self.catalog.get_preset(str(values["preset_id_or_name"]))
            if preset.get("preset_type") != "station_memory_scan":
                raise ValueError("preset_id_or_name must identify a station_memory_scan profile")
        await self._operation(
            receive, send,
            allowed={"name", "preset_id_or_name", "interval_seconds", "start_at",
                     "enabled", "replace_existing"},
            callback=self.schedule_create, unavailable="schedule_creation_unavailable",
            validator=validate)

    async def _create_station_scan_profile(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"name", "memory_ids_or_names", "duration_seconds", "max_memories",
                     "compare_previous", "snr_change_threshold_db", "description",
                     "replace_existing"},
            callback=self.scan_profile_create, unavailable="scan_profile_creation_unavailable")

    async def _run_station_scan_profile(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"preset_id_or_name"},
                              callback=self.scan_profile_run,
                              unavailable="scan_profile_execution_unavailable")

    async def _create_station_memory(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"name", "frequency_hz", "mode", "bandwidth_hz", "notes", "tags",
                     "enabled", "memory_id", "replace_existing"},
            callback=self.station_memory_create, unavailable="station_memory_creation_unavailable")

    async def _delete_station_memory(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send,
                              allowed={"memory_id_or_name", "confirm_delete"},
                              callback=self.station_memory_delete,
                              unavailable="station_memory_deletion_unavailable")

    async def _delete_station_scan_profile(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send,
                              allowed={"preset_id_or_name", "confirm_delete"},
                              callback=self.scan_profile_delete,
                              unavailable="scan_profile_deletion_unavailable")

    async def _delete_station_schedule(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send,
                              allowed={"schedule_id_or_name", "confirm_delete"},
                              callback=self.schedule_delete,
                              unavailable="schedule_deletion_unavailable")

    async def _start_band_scan(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"start_frequency_hz", "stop_frequency_hz", "capture_duration_seconds",
                     "overlap_fraction", "fft_size", "threshold_above_noise_db",
                     "minimum_signal_spacing_hz", "attenuation_steps", "max_signals"},
            callback=self.band_scan_start, unavailable="band_scan_unavailable")

    async def _start_band_survey(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"start_frequency_hz", "stop_frequency_hz", "capture_duration_seconds",
                     "overlap_fraction", "fft_size", "threshold_above_noise_db",
                     "minimum_signal_spacing_hz", "attenuation_steps", "max_signals",
                     "classify_top_signals", "classification_duration_seconds",
                     "classification_bandwidth_hz"},
            callback=self.band_survey_start, unavailable="band_survey_unavailable")

    async def _band_job_status(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"},
                              callback=self.band_job_status,
                              unavailable="band_status_unavailable")

    async def _stop_band_job(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"},
                              callback=self.band_job_stop,
                              unavailable="band_stop_unavailable")

    async def _run_preset(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"preset_id_or_name"},
                              callback=self.preset_run,
                              unavailable="preset_execution_unavailable")

    async def _start_fm_survey(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"start_frequency_hz", "stop_frequency_hz", "channel_spacing_hz",
                     "discovery_duration_seconds", "discovery_threshold_db",
                     "rds_duration_seconds", "deemphasis_us", "save_audio", "save_plots",
                     "resume_job_id"},
            callback=self.fm_survey_start, unavailable="fm_survey_unavailable")

    async def _fm_survey_status(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"},
                              callback=self.fm_survey_status,
                              unavailable="fm_survey_status_unavailable")

    async def _stop_fm_survey(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"},
                              callback=self.fm_survey_stop,
                              unavailable="fm_survey_stop_unavailable")

    async def _decode_native_digital(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"frequency_hz", "mode", "duration_seconds", "cw_wpm", "rtty_baud",
                     "rtty_shift_hz", "rtty_polarity", "retain_iq", "include_plot"},
            callback=self.digital_decode, unavailable="native_decoder_unavailable")

    async def _decode_weak_digital(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"frequency_hz", "mode", "capture_cycles", "align_to_utc",
                     "retain_iq", "retain_audio"},
            callback=self.weak_decode, unavailable="weak_signal_decoder_unavailable")

    async def _decode_fldigi(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"frequency_hz", "mode", "duration_seconds", "carrier_audio_hz",
                     "retain_iq", "retain_audio"},
            callback=self.fldigi_decode, unavailable="fldigi_decoder_unavailable")

    async def _digital_capabilities(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed=set(),
                              callback=self.decoder_capabilities,
                              unavailable="decoder_capabilities_unavailable")

    @staticmethod
    def _validate_sstv_start(values: dict, *, watcher: bool = False) -> None:
        if "frequency_hz" not in values:
            raise ValueError("frequency_hz is required")
        frequency = int(values["frequency_hz"])
        if not 9_000 <= frequency <= 260_000_000:
            raise ValueError("frequency_hz must be from 9000 through 260000000")
        mode = values.get("receiver_mode", "nfm" if watcher else "usb")
        if mode not in {"usb", "nfm"}:
            raise ValueError("receiver_mode must be usb or nfm")
        field = "watch_duration_seconds" if watcher else "duration_seconds"
        duration = float(values.get(field, 3600 if watcher else 130))
        minimum, maximum = (30, 86_400) if watcher else (20, 310)
        if not minimum <= duration <= maximum:
            raise ValueError(f"{field} must be from {minimum} through {maximum}")
        for key in ({"rearm", "retain_audio", "deduplicate"} if watcher else
                    {"retain_audio", "retain_iq", "deduplicate"}):
            if key in values and not isinstance(values[key], bool):
                raise ValueError(f"{key} must be a JSON boolean")

    async def _decode_sstv(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"frequency_hz", "duration_seconds", "receiver_mode", "retain_audio",
                     "retain_iq", "deduplicate"},
            callback=self.sstv_decode, unavailable="sstv_decoder_unavailable",
            validator=lambda values: self._validate_sstv_start(values))

    async def _start_sstv_watch(self, receive: Callable, send: Callable) -> None:
        await self._operation(
            receive, send,
            allowed={"frequency_hz", "receiver_mode", "watch_duration_seconds", "rearm",
                     "retain_audio", "deduplicate"},
            callback=self.sstv_watch_start, unavailable="sstv_watcher_unavailable",
            validator=lambda values: self._validate_sstv_start(values, watcher=True))

    async def _sstv_job_status(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"}, callback=self.sstv_status,
                              unavailable="sstv_status_unavailable")

    async def _sstv_watcher_status(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"},
                              callback=self.sstv_watch_status,
                              unavailable="sstv_watcher_status_unavailable")

    async def _stop_sstv(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"}, callback=self.sstv_stop,
                              unavailable="sstv_stop_unavailable")

    async def _stop_sstv_watch(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"job_id"}, callback=self.sstv_watch_stop,
                              unavailable="sstv_watcher_stop_unavailable")

    async def _sstv_decoder_capabilities(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed=set(), callback=self.sstv_capabilities,
                              unavailable="sstv_capabilities_unavailable")

    async def _run_station_schedule(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"schedule_id_or_name"},
                              callback=self.schedule_run,
                              unavailable="schedule_execution_unavailable")

    async def _toggle_station_schedule(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send,
                              allowed={"schedule_id_or_name", "enabled", "start_at"},
                              callback=self.schedule_toggle,
                              unavailable="schedule_control_unavailable")

    async def _acknowledge_alert(self, receive: Callable, send: Callable) -> None:
        await self._operation(receive, send, allowed={"event_id"},
                              callback=self.catalog.acknowledge_alert_event,
                              unavailable="alert_acknowledgement_unavailable")

    async def _capture_spectrum(self, receive: Callable, send: Callable) -> None:
        if self.spectrum_capture is None:
            await _response(send, 503, _json_bytes({"error": "capture_unavailable"}),
                            extra_headers=self._security_headers())
            return
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > 8192:
                await _response(send, 413, _json_bytes({"error": "request_too_large"}))
                return
            if not message.get("more_body", False):
                break
        try:
            values = json.loads(body or b"{}")
            if not isinstance(values, dict):
                raise ValueError("request body must be a JSON object")
            allowed = {"center_frequency_hz", "duration_seconds", "fft_size",
                       "threshold_above_noise_db", "max_peaks"}
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"unsupported fields: {', '.join(unknown)}")
            if "center_frequency_hz" not in values:
                raise ValueError("center_frequency_hz is required")
            frequency = int(values["center_frequency_hz"])
            duration = float(values.get("duration_seconds", 2))
            fft_size = int(values.get("fft_size", 16_384))
            threshold = float(values.get("threshold_above_noise_db", 8))
            max_peaks = int(values.get("max_peaks", 20))
            if duration < 0.25 or duration > 10:
                raise ValueError("dashboard duration_seconds must be from 0.25 through 10")
            if fft_size not in {2048, 4096, 8192, 16384, 32768, 65536}:
                raise ValueError("fft_size must be a supported power of two from 2048 through 65536")
            response = await asyncio.to_thread(
                self.spectrum_capture, center_frequency_hz=frequency,
                duration_seconds=duration, fft_size=fft_size,
                threshold_above_noise_db=threshold, max_peaks=max_peaks,
                retain_iq=False, include_plot=False,
            )
            result = dict(response.structuredContent or {})
            artifacts = self.catalog.list_artifacts(job_id=result.get("job_id"), limit=10)
            plot = next((item for item in artifacts if item["kind"] == "spectrum_plot"), None)
            payload = {"status": "completed", "job_id": result.get("job_id"),
                       "center_frequency_hz": result.get("center_frequency_hz"),
                       "duration_seconds": result.get("duration_seconds"),
                       "relative_noise_floor_db": result.get("relative_noise_floor_db"),
                       "peak_count": result.get("peak_count"), "peaks": result.get("peaks", []),
                       "plot_artifact": ({**plot, "download_path": f"/artifacts/{plot['artifact_id']}"}
                                         if plot else None),
                       "measurement_notice": "Relative levels only; not calibrated dBm."}
            await _response(send, 200, _json_bytes(payload), extra_headers=self._security_headers())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_capture_request",
                                                    "detail": str(exc)}),
                            extra_headers=self._security_headers())
        except Exception as exc:
            await _response(send, 409, _json_bytes({"error": "capture_failed",
                                                    "detail": f"{type(exc).__name__}: {exc}"}),
                            extra_headers=self._security_headers())

    async def _demodulate(self, receive: Callable, send: Callable) -> None:
        if self.signal_analyzer is None:
            await _response(send, 503, _json_bytes({"error": "demodulation_unavailable"}),
                            extra_headers=self._security_headers())
            return
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > 8192:
                await _response(send, 413, _json_bytes({"error": "request_too_large"}))
                return
            if not message.get("more_body", False):
                break
        try:
            values = json.loads(body or b"{}")
            if not isinstance(values, dict):
                raise ValueError("request body must be a JSON object")
            allowed = {"frequency_hz", "mode", "bandwidth_hz", "duration_seconds",
                       "fft_size", "cw_tone_hz"}
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"unsupported fields: {', '.join(unknown)}")
            if "frequency_hz" not in values:
                raise ValueError("frequency_hz is required")
            frequency = int(values["frequency_hz"])
            mode = str(values.get("mode", "am")).strip().lower()
            limits = {"am": (2_000, 20_000, 10_000), "usb": (1_000, 6_000, 3_000),
                      "lsb": (1_000, 6_000, 3_000), "cw": (100, 2_000, 500),
                      "nfm": (5_000, 30_000, 12_500)}
            if mode not in limits:
                raise ValueError("mode must be am, usb, lsb, cw, or nfm")
            low, high, default = limits[mode]
            bandwidth = int(values.get("bandwidth_hz", default))
            if not low <= bandwidth <= high:
                raise ValueError(f"{mode} bandwidth_hz must be from {low} through {high}")
            duration = float(values.get("duration_seconds", 5))
            if duration < 0.25 or duration > 10:
                raise ValueError("dashboard duration_seconds must be from 0.25 through 10")
            fft_size = int(values.get("fft_size", 16_384))
            if fft_size not in {2048, 4096, 8192, 16384, 32768, 65536}:
                raise ValueError("fft_size must be a supported power of two from 2048 through 65536")
            cw_tone = int(values.get("cw_tone_hz", 700))
            if not 300 <= cw_tone <= 1200:
                raise ValueError("cw_tone_hz must be from 300 through 1200")
            response = await asyncio.to_thread(
                self.signal_analyzer, frequency_hz=frequency, mode=mode,
                bandwidth_hz=bandwidth, duration_seconds=duration, fft_size=fft_size,
                cw_tone_hz=cw_tone, retain_iq=False, include_audio=False,
                include_plots=False,
            )
            result = dict(response.structuredContent or {})
            artifacts = self.catalog.list_artifacts(job_id=result.get("job_id"), limit=10)
            enriched = [{**item, "download_path": f"/artifacts/{item['artifact_id']}"}
                        for item in artifacts]
            audio = next((item for item in enriched if item["kind"] == "audio_wav"), None)
            rf_plot = next((item for item in enriched if item["kind"] == "rf_spectrum_plot"), None)
            audio_plot = next((item for item in enriched if item["kind"] == "audio_spectrum_plot"), None)
            await _response(send, 200, _json_bytes({
                "status": "completed", "job_id": result.get("job_id"),
                "frequency_hz": result.get("requested_frequency_hz"),
                "mode": result.get("mode"), "bandwidth_hz": result.get("bandwidth_hz"),
                "duration_seconds": result.get("duration_seconds"),
                "metrics": result.get("metrics", {}), "audio_artifact": audio,
                "rf_plot_artifact": rf_plot, "audio_plot_artifact": audio_plot,
                "measurement_notice": "Receive-only; relative levels, not calibrated dBm.",
            }), extra_headers=self._security_headers())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_demodulation_request",
                                                    "detail": str(exc)}),
                            extra_headers=self._security_headers())
        except Exception as exc:
            await _response(send, 409, _json_bytes({"error": "demodulation_failed",
                                                    "detail": f"{type(exc).__name__}: {exc}"}),
                            extra_headers=self._security_headers())

    async def _broadcast_fm(self, receive: Callable, send: Callable) -> None:
        if self.broadcast_fm_receiver is None:
            await _response(send, 503, _json_bytes({"error": "broadcast_fm_unavailable"}),
                            extra_headers=self._security_headers())
            return
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > 8192:
                await _response(send, 413, _json_bytes({"error": "request_too_large"}))
                return
            if not message.get("more_body", False):
                break
        try:
            values = json.loads(body or b"{}")
            if not isinstance(values, dict):
                raise ValueError("request body must be a JSON object")
            allowed = {"frequency_hz", "duration_seconds", "stereo", "deemphasis_us",
                       "decode_rds_data"}
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"unsupported fields: {', '.join(unknown)}")
            if "frequency_hz" not in values:
                raise ValueError("frequency_hz is required")
            frequency = int(values["frequency_hz"])
            if not 88_000_000 <= frequency <= 108_000_000:
                raise ValueError("dashboard broadcast FM frequency must be from 88 through 108 MHz")
            duration = float(values.get("duration_seconds", 10))
            if duration not in {5.0, 10.0}:
                raise ValueError("dashboard broadcast FM duration must be 5 or 10 seconds")
            stereo = values.get("stereo", True)
            decode_rds = values.get("decode_rds_data", True)
            if not isinstance(stereo, bool) or not isinstance(decode_rds, bool):
                raise ValueError("stereo and decode_rds_data must be JSON booleans")
            deemphasis = int(values.get("deemphasis_us", 75))
            if deemphasis not in {50, 75}:
                raise ValueError("deemphasis_us must be 50 or 75")
            response = await asyncio.to_thread(
                self.broadcast_fm_receiver, frequency_hz=frequency,
                duration_seconds=duration, stereo=stereo, deemphasis_us=deemphasis,
                decode_rds_data=decode_rds, retain_iq=False, include_audio=False,
                include_plot=False,
            )
            result = dict(response.structuredContent or {})
            artifacts = self.catalog.list_artifacts(job_id=result.get("job_id"), limit=10)
            enriched = [{**item, "download_path": f"/artifacts/{item['artifact_id']}"}
                        for item in artifacts]
            audio = next((item for item in enriched if item["kind"] == "broadcast_fm_audio"), None)
            plot = next((item for item in enriched
                         if item["kind"] == "broadcast_fm_multiplex_plot"), None)
            await _response(send, 200, _json_bytes({
                "status": "completed", "job_id": result.get("job_id"),
                "frequency_hz": result.get("requested_frequency_hz"),
                "duration_seconds": result.get("duration_seconds"),
                "metrics": result.get("metrics", {}), "rds": result.get("rds"),
                "audio_artifact": audio, "multiplex_plot_artifact": plot,
                "measurement_notice": "Receive-only; RDS fields require checksum-valid groups.",
            }), extra_headers=self._security_headers())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_broadcast_fm_request",
                                                    "detail": str(exc)}),
                            extra_headers=self._security_headers())
        except Exception as exc:
            await _response(send, 409, _json_bytes({"error": "broadcast_fm_failed",
                                                    "detail": f"{type(exc).__name__}: {exc}"}),
                            extra_headers=self._security_headers())

    def _frontend_assets(self) -> tuple[bytes, bytes, bytes]:
        if self._dashboard_assets is None:
            source = self._dashboard_source_html().decode("utf-8")
            style_match = re.search(r"<style>(.*?)</style>", source, re.DOTALL)
            script_match = re.search(r"<script>(.*?)</script>", source, re.DOTALL)
            if style_match is None or script_match is None:
                raise RuntimeError("dashboard source is missing its stylesheet or script")
            document = source[:style_match.start()] + (
                '<link rel="stylesheet" href="/assets/rf-dashboard.css">'
            ) + source[style_match.end():script_match.start()] + (
                '<script src="/assets/rf-dashboard.js"></script>'
            ) + source[script_match.end():]
            self._dashboard_assets = (
                document.encode("utf-8"), style_match.group(1).encode("utf-8"),
                script_match.group(1).encode("utf-8"),
            )
        return self._dashboard_assets

    def _dashboard_html(self) -> bytes:
        return self._frontend_assets()[0]

    def _dashboard_source_html(self) -> bytes:
        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniRackDisplay RF Dashboard</title><style>
:root{{--bg:#07101d;--card:#101d2e;--line:#263d59;--text:#e8eef7;--muted:#91a6c0;--accent:#f36d2e;--good:#38c793;--warn:#ffbf5b}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07101d,#0b1728);color:var(--text);font:14px system-ui}}header{{display:flex;align-items:center;justify-content:space-between;padding:22px clamp(18px,4vw,54px);border-bottom:1px solid var(--line)}}h1{{font-size:24px;margin:0}}header span,.muted{{color:var(--muted)}}main{{padding:24px clamp(18px,4vw,54px);max-width:1500px;margin:auto}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px;overflow:auto}}.metric b{{font-size:28px;display:block;margin-top:8px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}}h2{{font-size:17px;margin:0 0 14px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;text-align:left;border-bottom:1px solid #21344c;white-space:nowrap}}th{{color:var(--muted);font-size:12px}}.pill{{display:inline-block;padding:3px 8px;border-radius:12px;background:#223750}}.on{{color:var(--good)}}a{{color:#ff9a6a}}button{{background:transparent;color:var(--muted);border:1px solid var(--line);border-radius:7px;padding:7px 10px}}#error{{display:none;color:#ff9c8b;margin:12px 0}}.capture{{margin-top:18px}}.controls{{display:grid;grid-template-columns:2fr 1fr 1fr 1fr auto;gap:10px;align-items:end}}label{{color:var(--muted);font-size:12px}}input,select{{display:block;width:100%;margin-top:5px;background:#08111f;border:1px solid #3b5473;border-radius:7px;color:var(--text);padding:9px}}.primary{{background:var(--accent);border-color:var(--accent);color:white;font-weight:700;padding:10px 16px}}#spectrumPlot,.analysisPlot{{display:none;width:100%;max-height:520px;object-fit:contain;background:#fff;border-radius:8px;margin-top:14px}}#captureStatus,#audioStatus{{margin-top:12px}}audio{{display:none;width:100%;margin-top:14px}}.plots{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}#receiverBar{{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;padding:10px clamp(18px,4vw,54px);background:#0a1626;border-bottom:1px solid var(--line);box-shadow:0 6px 18px #0008}}#receiverDot{{width:10px;height:10px;border-radius:50%;background:var(--good);flex:none}}#receiverBar.busy #receiverDot{{background:var(--warn);box-shadow:0 0 0 4px #ffbf5b22}}#receiverActivity{{margin-left:auto}}#activityDrawer{{position:fixed;right:18px;top:132px;z-index:30;width:min(520px,calc(100vw - 36px));max-height:70vh;overflow:auto;display:none;background:#101d2e;border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 14px 44px #000b}}#activityDrawer.open{{display:block}}.activityItem{{padding:11px 0;border-bottom:1px solid #21344c}}.activityItem:last-child{{border-bottom:0}}@media(max-width:850px){{.summary{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}.controls{{grid-template-columns:1fr 1fr}}#receiverDetail{{display:none}}}}@media(max-width:450px){{.summary{{grid-template-columns:1fr}}.controls{{grid-template-columns:1fr}}.plots{{grid-template-columns:1fr}}#receiverBar{{gap:7px;padding:9px 12px}}}}
</style></head><body><header><div><h1>MiniRackDisplay</h1><span>RF MCP dashboard · v{html.escape(self.version)}</span></div><form method="post" action="/dashboard/logout"><button>Sign out</button></form></header><main><div id="error"></div><section class="summary"><div class="card metric"><span>Service</span><b id="service">…</b></div><div class="card metric"><span>Receivers</span><b id="receivers">…</b></div><div class="card metric"><span>Active leases</span><b id="leases">…</b></div><div class="card metric"><span>Recent jobs</span><b id="jobcount">…</b></div></section><section class="card capture"><h2>Live spectrum inspection</h2><form id="captureForm" class="controls"><label>Center frequency (Hz)<input id="frequency" type="number" min="9000" max="260000000" step="1" value="10000000" required></label><label>Duration<select id="duration"><option value="1">1 second</option><option value="2" selected>2 seconds</option><option value="5">5 seconds</option><option value="10">10 seconds</option></select></label><label>FFT size<select id="fft"><option>8192</option><option selected>16384</option><option>32768</option><option>65536</option></select></label><label>Peak threshold (dB)<input id="threshold" type="number" min="3" max="60" step="1" value="8"></label><button id="captureButton" class="primary" type="submit">Inspect</button></form><div id="captureStatus" class="muted">Receive-only. Levels are relative, not calibrated dBm.</div><img id="spectrumPlot" alt="Latest spectrum inspection plot"></section><section class="card capture"><h2>Tune and listen</h2><form id="audioForm" class="controls"><label>Signal frequency (Hz)<input id="audioFrequency" type="number" min="9000" max="260000000" step="1" value="10000000" required></label><label>Mode<select id="audioMode"><option value="am">AM</option><option value="nfm">Narrow FM</option><option value="usb">USB</option><option value="lsb">LSB</option><option value="cw">CW</option></select></label><label>Bandwidth (Hz)<input id="audioBandwidth" type="number" value="10000" min="100" max="30000"></label><label>Duration<select id="audioDuration"><option value="2">2 seconds</option><option value="5" selected>5 seconds</option><option value="10">10 seconds</option></select></label><button id="audioButton" class="primary" type="submit">Record audio</button></form><div id="audioStatus" class="muted">Select the correct mode and bandwidth for the signal.</div><audio id="audioPlayer" controls></audio><div class="plots"><img id="rfAnalysisPlot" class="analysisPlot" alt="RF analysis plot"><img id="audioAnalysisPlot" class="analysisPlot" alt="Audio spectrum plot"></div></section><section class="card capture"><h2>Broadcast FM and RDS</h2><form id="fmForm" class="controls"><label>Station frequency (MHz)<input id="fmFrequency" type="number" min="88" max="108" step="0.1" value="100.1" required></label><label>Duration<select id="fmDuration"><option value="5">5 seconds</option><option value="10" selected>10 seconds</option></select></label><label>Audio<select id="fmStereo"><option value="true" selected>Stereo when detected</option><option value="false">Mono</option></select></label><label>De-emphasis<select id="fmDeemphasis"><option value="75" selected>75 µs (Americas)</option><option value="50">50 µs</option></select></label><button id="fmButton" class="primary" type="submit">Receive station</button></form><div id="fmStatus" class="muted">RDS decoding requires checksum-valid repeated groups and may need a strong signal.</div><div id="rdsStatus" class="muted"></div><audio id="fmPlayer" controls></audio><img id="fmPlot" class="analysisPlot" alt="Broadcast FM multiplex analysis"></section><section class="grid"><div class="card"><h2>Receiver inventory</h2><table><thead><tr><th>Name</th><th>Backend</th><th>Role</th><th>State</th></tr></thead><tbody id="receiverRows"></tbody></table></div><div class="card"><h2>Storage</h2><div id="storage" class="muted">Loading…</div></div><div class="card"><h2>Recent RF jobs</h2><table><thead><tr><th>Type</th><th>State</th><th>Created</th></tr></thead><tbody id="jobRows"></tbody></table></div><div class="card"><h2>Recent artifacts</h2><table><thead><tr><th>File</th><th>Kind</th><th>Size</th></tr></thead><tbody id="artifactRows"></tbody></table></div></section></main><script>
const el=id=>document.getElementById(id), txt=(tag,value)=>{{const n=document.createElement(tag);n.textContent=value??'—';return n}}, fmt=n=>n==null?'—':n<1024?n+' B':n<1048576?(n/1024).toFixed(1)+' KiB':(n/1048576).toFixed(1)+' MiB';
const uxStyle=document.createElement('style');uxStyle.textContent='button{{transition:filter .15s,opacity .15s,transform .1s}}button:active{{transform:translateY(1px)}}button[aria-busy="true"]{{cursor:progress;opacity:.78}}button[aria-busy="true"]::before{{content:"";display:inline-block;width:12px;height:12px;margin-right:7px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;vertical-align:-2px;animation:spin .7s linear infinite}}@keyframes spin{{to{{transform:rotate(360deg)}}}}#toast{{position:fixed;left:50%;bottom:24px;z-index:60;transform:translate(-50%,18px);padding:11px 16px;background:#172a41;border:1px solid #3b5473;border-radius:9px;box-shadow:0 10px 30px #000a;opacity:0;pointer-events:none;transition:.2s}}#toast.show{{opacity:1;transform:translate(-50%,0)}}#fmBandMap{{display:grid;grid-template-columns:repeat(auto-fit,minmax(7px,1fr));gap:2px;min-height:92px;padding:12px;background:#08111f;border:1px solid var(--line);border-radius:9px;align-items:end}}.fmChannel{{height:32px;background:#1c3048;border-radius:2px;position:relative}}.fmChannel.scanned{{background:#315a7b}}.fmChannel.candidate{{height:58px;background:var(--accent)}}.fmChannel.decoded{{height:76px;background:var(--good)}}.fmChannel.current{{outline:2px solid var(--warn);animation:pulse 1s ease-in-out infinite alternate}}@keyframes pulse{{to{{filter:brightness(1.6)}}}}.fmLegend{{display:flex;gap:15px;flex-wrap:wrap;margin:8px 0 16px}}.fmKey{{display:inline-flex;align-items:center;gap:5px}}.fmSwatch{{width:11px;height:11px;border-radius:2px;background:#1c3048}}';document.head.append(uxStyle);const toast=txt('div','');toast.id='toast';toast.setAttribute('role','status');toast.setAttribute('aria-live','polite');document.body.append(toast);let toastTimer;function showToast(message){{toast.textContent=message;toast.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.classList.remove('show'),2400)}}function setButtonBusy(button,busy,label='Working…'){{if(!button)return;if(busy){{button.dataset.idleLabel=button.textContent;button.textContent=label;button.disabled=true;button.setAttribute('aria-busy','true')}}else{{button.textContent=button.dataset.idleLabel||button.textContent;button.disabled=false;button.removeAttribute('aria-busy')}}}}
uxStyle.textContent+=' .jobCards,.resultCards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.jobCard,.resultCard{{padding:13px;background:#08111f;border:1px solid var(--line);border-radius:10px;overflow:hidden}}.jobCard.running{{border-color:var(--warn)}}.jobCard.failed{{border-color:#d75c55}}.jobTop,.jobMeta,.bandLabels{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.progressTrack,.bandProgressMap{{height:12px;background:#1c3048;border-radius:8px;overflow:hidden;margin:10px 0}}.progressFill{{height:100%;background:linear-gradient(90deg,var(--accent),var(--warn));border-radius:8px;transition:width .4s}}.completed .progressFill{{background:var(--good)}}.jobError{{color:#ff9c8b;white-space:normal}}.resultCard img{{display:block;width:100%;height:160px;object-fit:contain;background:#000;border-radius:6px;margin-bottom:9px}}.resultCard audio{{display:block;width:100%;margin:8px 0}}';
let pendingSubmitButton=null;document.addEventListener('submit',event=>{{const button=event.submitter;if(!button)return;pendingSubmitButton=button;setButtonBusy(button,true,'Working…');showToast((button.dataset.idleLabel||'Action')+' started')}},true);setInterval(()=>{{for(const button of document.querySelectorAll('button[aria-busy="true"]'))if(!button.disabled)setButtonBusy(button,false)}},100);
const nav=document.createElement('nav');nav.style.cssText='display:flex;gap:10px;padding:10px clamp(18px,4vw,54px);border-bottom:1px solid var(--line);overflow:auto';nav.innerHTML='<a href="#home">Home</a><a href="#operations">Operations</a><a href="#memories">Memories</a><a href="#spectrum">Spectrum</a><a href="#scan">Scan & Analyze</a><a href="#digital">Digital Modes</a><a href="#sstv">SSTV Gallery</a><a href="#listen">Listen</a><a href="#fm">FM/RDS</a><a href="#system">System</a>';document.querySelector('header').after(nav);for(const a of nav.querySelectorAll('a'))a.style.cssText='padding:7px 11px;border-radius:7px;background:#15263b;text-decoration:none';
const receiverBar=document.createElement('div');receiverBar.id='receiverBar';receiverBar.setAttribute('role','status');receiverBar.setAttribute('aria-live','polite');receiverBar.innerHTML='<span id="receiverDot"></span><strong id="receiverState">Receiver ready</strong><span id="receiverDetail" class="muted">No RF operation is active.</span><button id="receiverOpen" type="button" style="display:none">Open</button><button id="receiverStop" type="button" style="display:none">Stop</button><button id="receiverActivity" type="button">Activity <span id="activityCount" class="pill">0</span></button>';nav.after(receiverBar);const activityDrawer=document.createElement('aside');activityDrawer.id='activityDrawer';activityDrawer.innerHTML='<div style="display:flex;align-items:center;gap:10px"><h2 style="margin:0 auto 0 0">Receiver activity</h2><button id="activityClose" type="button">Close</button></div><p class="muted">Recent captures and long-running RF operations.</p><div id="activityItems"></div>';document.body.append(activityDrawer);el('receiverActivity').addEventListener('click',()=>activityDrawer.classList.toggle('open'));el('activityClose').addEventListener('click',()=>activityDrawer.classList.remove('open'));
function rows(id,items,cells){{const body=el(id);body.replaceChildren();for(const item of items){{const tr=document.createElement('tr');for(const cell of cells(item))tr.append(cell);body.append(tr)}}if(!items.length){{const tr=document.createElement('tr'),td=txt('td','No entries');td.colSpan=5;tr.append(td);body.append(tr)}}}}
function guidedEmpty(id,colspan,message,label,handler){{const body=el(id),tr=document.createElement('tr'),td=document.createElement('td'),copy=txt('span',message);copy.className='muted';td.colSpan=colspan;td.append(copy);if(label)td.append(' ',actionButton(label,handler,true));tr.append(td);body.replaceChildren(tr)}}
const firstCapture=document.querySelector('.capture');firstCapture.id='spectrum';document.querySelectorAll('.capture')[1].id='listen';document.querySelectorAll('.capture')[2].id='fm';document.querySelector('.grid').id='system';
const receiverSetup=document.createElement('div');receiverSetup.className='card';receiverSetup.innerHTML='<h2>Add a receiver</h2><p class="muted">Connect an Airspy HF+ or RTL-SDR, then scan. No JSON or serial-number copying is required.</p><button id="receiverScan" class="primary" type="button">Scan for receivers</button><div id="receiverScanStatus" class="muted" style="margin-top:10px">Hardware has not been scanned yet.</div><div id="receiverCandidates" style="display:grid;gap:10px;margin-top:12px"></div>';el('system').prepend(receiverSetup);
const memorySection=document.createElement('section');memorySection.id='memories';memorySection.className='card capture';memorySection.innerHTML='<h2>Station memories</h2><p class="muted">Save frequently used stations once, then recall, listen, edit, scan, or schedule them.</p><form id="memoryForm" class="controls"><label>Name<input id="memoryName" maxlength="64" placeholder="WWV 10 MHz" required></label><label>Frequency<input id="memoryFrequency" type="number" min="0.009" max="260" step="0.000001" placeholder="MHz" required><small>MHz</small></label><label>Mode<select id="memoryMode"><option value="am">AM</option><option value="nfm">NFM</option><option value="usb">USB</option><option value="lsb">LSB</option><option value="cw">CW</option><option value="broadcast_fm">Broadcast FM</option></select></label><label>Bandwidth (Hz)<input id="memoryBandwidth" type="number" min="100" max="250000" value="10000" required></label><label>Tags<input id="memoryTags" placeholder="time, utility"></label><button id="memorySave" class="primary">Save memory</button><button id="memoryCancel" type="button" style="display:none">Cancel edit</button></form><div id="memoryStatus" class="muted">Frequency entry uses MHz, including decimals such as 10.000000 or 100.1.</div><table><thead><tr><th>Name</th><th>Frequency</th><th>Mode</th><th>Tags</th><th>Actions</th></tr></thead><tbody id="memoryRows"></tbody></table>';firstCapture.before(memorySection);
const homeSection=document.createElement('section');homeSection.id='home';homeSection.className='card capture';homeSection.innerHTML='<h2>Quick Start</h2><p class="muted">Listen to a favorite station or jump directly to a common receiver task.</p><div class="controls"><button id="quickListen" class="primary" type="button">Tune & listen</button><button id="quickFm" type="button">Broadcast FM</button><button id="quickDigital" type="button">Digital modes</button><button id="quickSstv" type="button">SSTV</button><button id="quickScan" type="button">Scan a band</button></div><h3>Favorite stations</h3><div id="favoriteCards" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px"></div><h3>Live and recent RF jobs</h3><div id="jobProgressCards" class="jobCards"></div><h3>Visual results</h3><div id="recentResultCards" class="resultCards"></div><h3>Recent receiver activity</h3><div id="homeRecent"></div>';document.querySelector('main').prepend(homeSection);el('quickListen').onclick=()=>showView('listen');el('quickFm').onclick=()=>showView('fm');el('quickDigital').onclick=()=>showView('digital');el('quickSstv').onclick=()=>showView('sstv');el('quickScan').onclick=()=>showView('scan');
const operations=document.createElement('section');operations.id='operations';operations.className='card capture';operations.innerHTML='<h2>RF Operations</h2><p class="muted">Start here: select saved station memories, create a scan profile, then run it once. Scheduling is optional.</p><h3>1. Create a memory scan profile</h3><form id="profileForm" class="controls"><label>Profile name<input id="profileName" maxlength="64" placeholder="Daily station check" required></label><label>Station memories<select id="profileMemories" multiple size="4" required></select></label><label>Seconds per station<select id="profileDuration"><option value="2">2</option><option value="5" selected>5</option><option value="10">10</option></select></label><button class="primary">Create profile</button></form><div id="profileStatus" class="muted">Use Ctrl/Command-click to select more than one memory.</div><table><thead><tr><th>Profile</th><th>Stations</th><th>Duration</th><th>Action</th></tr></thead><tbody id="profileRows"></tbody></table><h3>2. Latest station status</h3><div id="stationStatus" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px"></div><h3>Memory scan trend</h3><canvas id="trend" width="900" height="180" style="width:100%;height:180px;background:#08111f;border-radius:8px"></canvas><h3>3. Optional schedules</h3><form id="scheduleForm" class="controls"><label>Name<input id="scheduleName" maxlength="64" required></label><label>Scan profile<select id="scheduleProfile" required></select></label><label>Interval (minutes)<input id="scheduleMinutes" type="number" min="1" max="10080" value="60" required></label><label>Enabled<select id="scheduleEnabled"><option value="true">Yes</option><option value="false">No</option></select></label><button class="primary">Create</button></form><div id="scheduleStatus" class="muted"></div><table><thead><tr><th>Name</th><th>Profile</th><th>Next run</th><th>State</th><th>Actions</th></tr></thead><tbody id="scheduleRows"></tbody></table><h3>Recent changes and alerts</h3><table><thead><tr><th>Time</th><th>Source</th><th>Event</th><th>Review</th></tr></thead><tbody id="alertRows"></tbody></table><h3>Recent audio</h3><audio id="recentAudio" controls style="width:100%"></audio><div id="recentAudioLinks"></div>';memorySection.before(operations);
const fmSurvey=document.createElement('div');fmSurvey.innerHTML='<hr style="border:0;border-top:1px solid var(--line);margin:24px 0"><h2>FM band survey & station directory</h2><p class="muted">Discover occupied FM channels, collect RDS metadata, and build a persistent local station directory.</p><form id="fmSurveyForm" class="controls"><label>Band plan<select id="fmSurveyPlan"><option value="87.9,107.9,0.2" selected>Americas · 87.9–107.9 MHz · 200 kHz</option><option value="87.5,108,0.1">Europe · 87.5–108 MHz · 100 kHz</option><option value="76,95,0.1">Japan · 76–95 MHz · 100 kHz</option></select></label><label>Discovery threshold (dB)<input id="fmSurveyThreshold" type="number" min="3" max="40" value="8"></label><label>RDS capture<select id="fmSurveyRds"><option value="5">5 seconds</option><option value="10" selected>10 seconds</option></select></label><label>Save<select id="fmSurveySave"><option value="plots">Plots</option><option value="audio">Audio + plots</option><option value="none">Directory only</option></select></label><button id="fmSurveyButton" class="primary">Start survey</button></form><div id="fmSurveyStatus" class="muted">A full-band survey can take several minutes and occupies the receiver.</div><h3>Band activity map</h3><div id="fmBandSummary" class="muted">Start a survey to visualize scanned, occupied, and decoded channels.</div><div id="fmBandMap" aria-label="FM survey channel map"></div><div class="fmLegend"><span class="fmKey"><i class="fmSwatch"></i>Unscanned</span><span class="fmKey"><i class="fmSwatch" style="background:#315a7b"></i>Scanned</span><span class="fmKey"><i class="fmSwatch" style="background:var(--accent)"></i>Signal candidate</span><span class="fmKey"><i class="fmSwatch" style="background:var(--good)"></i>RDS/audio collected</span></div><h3>Survey jobs</h3><table><thead><tr><th>State</th><th>Phase</th><th>Created</th><th>Progress</th><th>Actions</th></tr></thead><tbody id="fmSurveyRows"></tbody></table><h3>Discovered stations</h3><label style="max-width:320px">Filter directory<input id="fmStationFilter" placeholder="Frequency, station, type, or radiotext"></label><div id="fmDirectoryPlayer" style="display:none;margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:9px;background:#08111f"><strong id="fmDirectoryNow">Now listening</strong><div id="fmDirectoryStatus" class="muted" aria-live="polite"></div><audio id="fmDirectoryAudio" controls style="width:100%;margin-top:8px"></audio><div id="fmDirectoryRds" class="muted"></div><a id="fmDirectoryDownload" style="display:none">Download WAV</a></div><table><thead><tr><th>Frequency</th><th>Station</th><th>RDS</th><th>Stereo / SNR</th><th>Last seen</th><th>Actions</th></tr></thead><tbody id="fmStationRows"></tbody></table>';el('fm').append(fmSurvey);
const scanSection=document.createElement('section');scanSection.id='scan';scanSection.className='card capture';scanSection.innerHTML='<h2>Scan & Analyze</h2><p class="muted">Scan a frequency range for carriers, or survey it and heuristically classify the strongest signals. The receiver is occupied while a job runs.</p><form id="bandForm" class="controls"><label>Common band<select id="bandChoice"><option value="custom">Custom</option><option value="1.8,2.0">160 m</option><option value="3.5,4.0">80 m</option><option value="7.0,7.3">40 m</option><option value="10.1,10.15">30 m</option><option value="14.0,14.35" selected>20 m</option><option value="18.068,18.168">17 m</option><option value="21.0,21.45">15 m</option><option value="24.89,24.99">12 m</option><option value="28.0,29.7">10 m</option></select></label><label>Start (MHz)<input id="bandStart" type="number" min="0.009" max="260" step="0.000001" value="14.0" required></label><label>Stop (MHz)<input id="bandStop" type="number" min="0.009" max="260" step="0.000001" value="14.35" required></label><label>Operation<select id="bandOperation"><option value="survey" selected>Survey + classify</option><option value="scan">Spectrum scan only</option></select></label><button id="bandStartButton" class="primary">Start</button></form><div class="controls" style="margin-top:10px"><label>Seconds per segment<select id="bandDuration"><option value="0.25">0.25</option><option value="0.5">0.5</option><option value="1" selected>1</option><option value="2">2</option></select></label><label>Threshold above noise (dB)<input id="bandThreshold" type="number" min="3" max="40" value="8"></label><label>Classify strongest<input id="bandClassify" type="number" min="1" max="20" value="10"></label></div><div id="bandStatus" class="muted">Levels are relative; classifications are heuristic.</div><h3>Recent scan and survey jobs</h3><table><thead><tr><th>Type</th><th>State</th><th>Created</th><th>Progress</th><th>Actions</th></tr></thead><tbody id="bandJobRows"></tbody></table><h3>Saved RF presets</h3><p class="muted">Run presets previously created through either the dashboard or MCP tools.</p><table><thead><tr><th>Name</th><th>Type</th><th>Description</th><th>Action</th></tr></thead><tbody id="presetRows"></tbody></table>';el('system').before(scanSection);
const digitalSection=document.createElement('section');digitalSection.id='digital';digitalSection.className='card capture';digitalSection.innerHTML='<h2>Digital Modes</h2><p class="muted">Receive-only decoding. Presets are conventional activity frequencies, not assigned channels; activity varies by region and time.</p><div id="decoderStatus" class="muted">Checking decoder availability…</div><form id="digitalForm" class="controls"><label>Decoder family<select id="digitalFamily"><option value="native">Native / packet</option><option value="weak">FT8 / FT4 / WSPR</option><option value="fldigi">Fldigi text modes</option></select></label><label>Mode<select id="digitalMode"></select></label><label>Band / activity frequency<select id="digitalPreset"></select></label><label>Dial / signal frequency (MHz)<input id="digitalFrequency" type="number" min="0.009" max="260" step="0.000001" value="14.074" required></label><label id="digitalDurationLabel">Duration (seconds)<input id="digitalDuration" type="number" min="1" max="120" value="15"></label><button id="digitalButton" class="primary">Decode</button></form><div id="digitalOptions" class="controls" style="margin-top:10px"><label id="digitalCarrierLabel">Decoder audio center (Hz)<input id="digitalCarrier" type="number" min="100" max="4000" value="1500"><small>Fldigi only: center of the target signal in the USB audio passband.</small></label><label id="digitalRetainLabel">Retain demodulated audio<select id="digitalRetainAudio"><option value="true" selected>Yes</option><option value="false">No</option></select></label></div><div id="digitalResult" class="muted">Choose a mode and a common band preset, or select Custom.</div><pre id="digitalText" style="white-space:pre-wrap;background:#08111f;padding:12px;border-radius:8px;display:none"></pre><div id="digitalDiagnostics" class="muted"></div><img id="digitalWaterfall" class="analysisPlot" alt="Weak-signal audio waveform and waterfall"><h3>Recent weak-signal spots</h3><label style="max-width:320px">Filter spots<input id="spotFilter" placeholder="Callsign, grid, mode, or message"></label><table><thead><tr><th>Time</th><th>Mode</th><th>Frequency</th><th>Callsign</th><th>Grid</th><th>SNR</th><th>Message</th><th>Actions</th></tr></thead><tbody id="spotRows"></tbody></table><h3>Recent Fldigi text</h3><table><thead><tr><th>Time</th><th>Mode</th><th>Dial frequency</th><th>Decoded text</th><th>Actions</th></tr></thead><tbody id="fldigiRows"></tbody></table>';el('system').before(digitalSection);
const sstvSection=document.createElement('section');sstvSection.id='sstv';sstvSection.className='card capture';sstvSection.innerHTML='<h2>SSTV receive & gallery</h2><p class="muted">Use USB for HF SSTV (for example 14.230 MHz) and NFM for direct satellite channels such as 145.800 MHz. A watcher waits for a valid VIS header and can capture multiple images.</p><div id="sstvCapabilities" class="muted">Checking SSTV decoder…</div><form id="sstvForm" class="controls"><label>Frequency (MHz)<input id="sstvFrequency" type="number" min="0.009" max="260" step="0.000001" value="14.230" required></label><label>Receiver mode<select id="sstvMode"><option value="usb" selected>USB (HF)</option><option value="nfm">NFM (satellite/FM)</option></select></label><label>Operation<select id="sstvOperation"><option value="decode">Capture and decode</option><option value="watch">Watch for transmissions</option></select></label><label>Duration (seconds)<input id="sstvDuration" type="number" min="20" max="310" value="130" required></label><button id="sstvStart" class="primary">Start</button></form><div class="controls" style="margin-top:10px"><label>Retain demodulated audio<select id="sstvAudio"><option value="true" selected>Yes</option><option value="false">No</option></select></label><label>Duplicate handling<select id="sstvDeduplicate"><option value="true" selected>Mark duplicates</option><option value="false">Keep independently</option></select></label></div><div id="sstvStatus" class="muted">No SSTV job started from this dashboard yet.</div><h3>Recent SSTV jobs</h3><table><thead><tr><th>Type</th><th>State</th><th>Created</th><th>Actions</th></tr></thead><tbody id="sstvJobRows"></tbody></table><div style="display:flex;align-items:end;gap:12px;flex-wrap:wrap"><h3 style="margin-right:auto">Decoded image gallery</h3><label>Filter<input id="sstvFilter" placeholder="Mode, frequency, or image ID"></label><label><input id="sstvHideDuplicates" type="checkbox"> Hide duplicates</label></div><div id="sstvGallery" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px"></div>';el('system').before(sstvSection);
const setupProgress=document.createElement('div');setupProgress.id='setupProgress';setupProgress.style.cssText='padding:12px;margin:10px 0 18px;border:1px solid var(--line);border-radius:9px;background:#0a1626';operations.querySelector('h2').after(setupProgress);
const dashboardViews=['home','operations','memories','spectrum','scan','digital','sstv','listen','fm','system'];function showView(id){{if(!dashboardViews.includes(id))id='home';for(const name of dashboardViews){{const section=el(name);if(section)section.style.display=name===id?(name==='system'?'grid':'block'):'none'}}for(const link of nav.querySelectorAll('a')){{const active=link.getAttribute('href')==='#'+id;link.style.background=active?'var(--accent)':'#15263b';link.style.color=active?'white':''}}history.replaceState(null,'','#'+id)}}for(const link of nav.querySelectorAll('a'))link.addEventListener('click',event=>{{event.preventDefault();showView(link.getAttribute('href').slice(1))}});showView(location.hash.slice(1)||'home');
const bandProgressPanel=document.createElement('div');bandProgressPanel.innerHTML='<h3>Live scan position</h3><div id="bandProgressSummary" class="muted">Start a scan to see its position and ETA.</div><div id="bandProgressMap" class="bandProgressMap" aria-label="Band scan progress"></div>';scanSection.querySelector('h3').before(bandProgressPanel);
function recallMemory(m){{el('frequency').value=m.frequency_hz;if(m.mode==='broadcast_fm'){{el('fmFrequency').value=(m.frequency_hz/1e6).toFixed(1);el('fmFrequency').scrollIntoView({{behavior:'smooth',block:'center'}})}}else{{el('audioFrequency').value=m.frequency_hz;el('audioMode').value=m.mode;el('audioBandwidth').value=m.bandwidth_hz;el('audioFrequency').scrollIntoView({{behavior:'smooth',block:'center'}})}}}}
function listenMemory(m){{recallMemory(m);showView(m.mode==='broadcast_fm'?'fm':'listen');setTimeout(()=>{{if(m.mode==='broadcast_fm')el('fmForm').requestSubmit();else el('audioForm').requestSubmit()}},0)}}
async function toggleFavorite(m){{const tags=[...(m.tags||[])],index=tags.findIndex(t=>t.toLowerCase()==='favorite');if(index>=0)tags.splice(index,1);else tags.push('favorite');await postOperation('/api/station-memories',{{memory_id:m.memory_id,name:m.name,frequency_hz:m.frequency_hz,mode:m.mode,bandwidth_hz:m.bandwidth_hz,tags,enabled:m.enabled!==false,replace_existing:true}});await refreshMemories();await refreshOperations()}}
function configureDigital(family,mode,frequencyHz){{el('digitalFamily').value=family;updateDigitalModes();el('digitalMode').value=(mode||'').toLowerCase();el('digitalFrequency').value=(Number(frequencyHz||14074000)/1e6).toFixed(6);updateDigitalPresets(el('digitalFrequency').value);showView('digital');digitalSection.scrollIntoView({{behavior:'smooth',block:'start'}})}}
function prepareMemory(name,frequencyHz,mode='usb',bandwidth=3000,tags=[]){{resetMemoryForm();el('memoryName').value=name||'Received signal';el('memoryFrequency').value=(Number(frequencyHz)/1e6).toFixed(6).replace(/0+$/,'').replace(/\\.$/,'');el('memoryMode').value=mode;el('memoryBandwidth').value=bandwidth;el('memoryTags').value=tags.join(', ');el('memoryStatus').textContent='Review the prefilled station, then choose Save memory.';showView('memories')}}
let editingMemoryId=null;function resetMemoryForm(){{editingMemoryId=null;el('memoryForm').reset();el('memoryBandwidth').value=10000;el('memorySave').textContent='Save memory';el('memoryCancel').style.display='none'}}function editMemory(m){{editingMemoryId=m.memory_id;el('memoryName').value=m.name;el('memoryFrequency').value=(m.frequency_hz/1e6).toFixed(6).replace(/0+$/,'').replace(/\\.$/,'');el('memoryMode').value=m.mode;el('memoryBandwidth').value=m.bandwidth_hz;el('memoryTags').value=(m.tags||[]).join(', ');el('memorySave').textContent='Save changes';el('memoryCancel').style.display='inline-block';memorySection.scrollIntoView({{behavior:'smooth'}})}}
async function refreshMemories(){{try{{const response=await fetch('/api/dashboard',{{cache:'no-store'}}),d=await response.json();rows('memoryRows',d.station_memories||[],m=>{{const td=document.createElement('td'),recall=actionButton('Recall',()=>recallMemory(m)),listen=actionButton('Listen',()=>listenMemory(m),true),edit=actionButton('Edit',()=>editMemory(m)),remove=actionButton('Delete',async()=>{{if(!confirm(`Delete station memory “${{m.name}}”?`))return;try{{await postOperation('/api/station-memories/delete',{{memory_id_or_name:m.memory_id,confirm_delete:true}});el('memoryStatus').textContent='Memory deleted.';await refreshMemories();await refreshOperations()}}catch(e){{el('memoryStatus').textContent='Delete failed: '+e.message}}}});td.append(recall,' ',listen,' ',edit,' ',remove);return[txt('td',m.name),txt('td',(m.frequency_hz/1e6).toFixed(6).replace(/0+$/,'').replace(/\\.$/,'')+' MHz'),txt('td',m.mode.toUpperCase()),txt('td',(m.tags||[]).join(', ')),td]}})}}catch(e){{el('memoryStatus').textContent='Memory refresh failed: '+e.message}}}}
async function postOperation(path,body){{try{{const response=await fetch(path,{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(body)}}),d=await response.json();if(!response.ok)throw new Error(d.detail||d.error);return d.result}}finally{{if(pendingSubmitButton){{setButtonBusy(pendingSubmitButton,false);pendingSubmitButton=null}}}}}}
function receiverCandidateCard(device){{const card=document.createElement('div');card.style.cssText='padding:12px;background:#08111f;border:1px solid var(--line);border-radius:9px';const title=txt('strong',device.display_name),state=txt('span',device.already_registered?'Already registered':'Ready to add');state.className='pill';state.style.marginLeft='8px';card.append(title,state);if(device.already_registered)return card;const form=document.createElement('form');form.className='controls';form.style.marginTop='10px';form.innerHTML='<label>Name<input data-field="name" maxlength="100" required></label><label>Use for<select data-field="role"><option value="primary_hf">Primary HF</option><option value="vhf_uhf_monitor">VHF/UHF monitoring</option><option value="satellite">Satellite</option><option value="wideband_survey">Wideband survey</option><option value="general">General</option></select></label><label>Receiver ID<input data-field="receiver_id" pattern="[a-z0-9][a-z0-9_-]{{0,63}}" required></label><label>Priority<input data-field="priority" type="number" min="0" max="100" value="80"></label><button class="primary">Add receiver</button>';form.querySelector('[data-field="name"]').value=device.suggested_name;form.querySelector('[data-field="role"]').value=device.suggested_role;form.querySelector('[data-field="receiver_id"]').value=device.suggested_receiver_id;form.addEventListener('submit',async event=>{{event.preventDefault();const button=event.submitter;setButtonBusy(button,true,'Verifying…');try{{const result=await postOperation('/api/receivers/register',{{backend:device.backend,device_selector:device.device_selector,name:form.querySelector('[data-field="name"]').value,role:form.querySelector('[data-field="role"]').value,receiver_id:form.querySelector('[data-field="receiver_id"]').value,priority:Number(form.querySelector('[data-field="priority"]').value)}});el('receiverScanStatus').textContent=`Added ${{result.receiver.name}}. It is ready to use.`;showToast('Receiver added');await scanReceivers();await refresh()}}catch(e){{el('receiverScanStatus').textContent='Could not add receiver: '+e.message}}finally{{setButtonBusy(button,false)}}}});card.append(form);return card}}
async function scanReceivers(){{const button=el('receiverScan'),container=el('receiverCandidates'),status=el('receiverScanStatus');setButtonBusy(button,true,'Scanning…');status.textContent='Checking USB receiver tools and attached hardware…';container.replaceChildren();try{{const result=await postOperation('/api/receivers/discover',{{}});status.textContent=result.device_count?`Found ${{result.device_count}} receiver${{result.device_count===1?'':'s'}}.`:'No supported receivers found. Check USB connections and installed driver tools.';for(const device of result.devices)container.append(receiverCandidateCard(device));for(const diagnostic of result.diagnostics||[]){{const note=txt('div',`${{diagnostic.backend}}: ${{diagnostic.error}}`);note.className='muted';container.append(note)}}}}catch(e){{status.textContent='Receiver scan failed: '+e.message}}finally{{setButtonBusy(button,false)}}}}el('receiverScan').addEventListener('click',scanReceivers);
function parseFrequency(value,defaultUnit='hz'){{const match=String(value).trim().toLowerCase().replaceAll(',','').match(/^([0-9]+(?:[.][0-9]+)?)[ 	]*(hz|khz|mhz)?$/);if(!match)throw new Error('Enter a frequency such as 14074000, 145800 kHz, or 14.074 MHz');const unit=match[2]||defaultUnit,multiplier={{hz:1,khz:1e3,mhz:1e6}}[unit],frequency=Math.round(Number(match[1])*multiplier);if(!Number.isFinite(frequency)||frequency<9000||frequency>260000000)throw new Error('Frequency must be between 9 kHz and 260 MHz');return frequency}}function smartFrequencyForm(formId,inputId,defaultUnit,statusId,storeAs='hz'){{const input=el(inputId);input.type='text';input.removeAttribute('min');input.removeAttribute('max');input.removeAttribute('step');input.title='Accepts Hz, kHz, or MHz';el(formId).addEventListener('submit',event=>{{try{{const hz=parseFrequency(input.value,defaultUnit);input.value=storeAs==='mhz'?String(hz/1e6):String(hz)}}catch(e){{event.preventDefault();event.stopImmediatePropagation();el(statusId).textContent=e.message}}}},true)}}smartFrequencyForm('captureForm','frequency','hz','captureStatus');smartFrequencyForm('audioForm','audioFrequency','hz','audioStatus');smartFrequencyForm('fmForm','fmFrequency','mhz','fmStatus','mhz');smartFrequencyForm('memoryForm','memoryFrequency','mhz','memoryStatus','mhz');smartFrequencyForm('digitalForm','digitalFrequency','mhz','digitalResult','mhz');smartFrequencyForm('sstvForm','sstvFrequency','mhz','sstvStatus','mhz');
el('bandChoice').addEventListener('change',()=>{{if(el('bandChoice').value==='custom')return;const [start,stop]=el('bandChoice').value.split(',');el('bandStart').value=start;el('bandStop').value=stop}});el('bandOperation').addEventListener('change',()=>{{el('bandClassify').disabled=el('bandOperation').value==='scan'}});el('bandForm').addEventListener('submit',async event=>{{event.preventDefault();const status=el('bandStatus'),button=el('bandStartButton'),survey=el('bandOperation').value==='survey',body={{start_frequency_hz:Math.round(Number(el('bandStart').value)*1e6),stop_frequency_hz:Math.round(Number(el('bandStop').value)*1e6),capture_duration_seconds:Number(el('bandDuration').value),overlap_fraction:0.15,fft_size:8192,threshold_above_noise_db:Number(el('bandThreshold').value),minimum_signal_spacing_hz:1000,attenuation_steps:1,max_signals:100}};if(survey)Object.assign(body,{{classify_top_signals:Number(el('bandClassify').value),classification_duration_seconds:2,classification_bandwidth_hz:30000}});button.disabled=true;status.textContent=survey?'Starting survey…':'Starting scan…';try{{const result=await postOperation(survey?'/api/band-survey/start':'/api/band-scan/start',body);status.textContent=`Started ${{survey?'survey':'scan'}} ${{result.job_id||''}}. Use Status below to follow progress.`;await refreshOperations()}}catch(e){{status.textContent='Start failed: '+e.message}}finally{{button.disabled=false}}}});
el('fmSurveyForm').addEventListener('submit',async event=>{{event.preventDefault();const [start,stop,spacing]=el('fmSurveyPlan').value.split(',').map(Number),save=el('fmSurveySave').value,status=el('fmSurveyStatus'),button=el('fmSurveyButton');button.disabled=true;status.textContent='Starting FM discovery survey…';try{{const result=await postOperation('/api/fm-surveys/start',{{start_frequency_hz:Math.round(start*1e6),stop_frequency_hz:Math.round(stop*1e6),channel_spacing_hz:Math.round(spacing*1e6),discovery_duration_seconds:0.25,discovery_threshold_db:Number(el('fmSurveyThreshold').value),rds_duration_seconds:Number(el('fmSurveyRds').value),deemphasis_us:75,save_audio:save==='audio',save_plots:save!=='none'}});status.textContent=`FM survey started · ${{result.job_id||''}}`;await refreshOperations()}}catch(e){{status.textContent='FM survey failed: '+e.message}}finally{{button.disabled=false}}}});el('fmStationFilter').addEventListener('input',refreshOperations);
const digitalModes={{native:[['cw','CW Morse'],['rtty','RTTY Baudot'],['bpsk31','BPSK31'],['ax25_afsk1200','AX.25 AFSK1200']],weak:[['ft8','FT8'],['ft4','FT4'],['wspr','WSPR']],fldigi:[['olivia-8-250','Olivia 8/250'],['contestia-8-250','Contestia 8/250'],['mfsk16','MFSK16'],['psk63','PSK63'],['dominoex-11','DominoEX 11'],['thor-11','THOR 11'],['mt63-1000l','MT63 1000L'],['hell','Feld Hell']]}};const digitalFrequencies={{ft8:[['160 m',1.840],['80 m',3.573],['60 m',5.357],['40 m',7.074],['30 m',10.136],['20 m',14.074],['17 m',18.100],['15 m',21.074],['12 m',24.915],['10 m',28.074],['6 m',50.313],['2 m',144.174]],ft4:[['80 m',3.575],['40 m',7.0475],['30 m',10.140],['20 m',14.080],['17 m',18.104],['15 m',21.140],['12 m',24.919],['10 m',28.180],['6 m',50.318],['2 m',144.170]],wspr:[['160 m',1.8366],['80 m',3.5686],['60 m',5.2872],['40 m',7.0386],['30 m',10.1387],['20 m',14.0956],['17 m',18.1046],['15 m',21.0946],['12 m',24.9246],['10 m',28.1246],['6 m',50.293],['2 m',144.489]],cw:[['80 m',3.560],['40 m',7.030],['30 m',10.106],['20 m',14.060],['17 m',18.086],['15 m',21.060],['12 m',24.906],['10 m',28.060]],rtty:[['80 m',3.580],['40 m',7.080],['30 m',10.142],['20 m',14.080],['17 m',18.100],['15 m',21.080],['12 m',24.925],['10 m',28.080]],bpsk31:[['160 m',1.838],['80 m',3.580],['40 m',7.070],['30 m',10.142],['20 m',14.070],['17 m',18.097],['15 m',21.080],['12 m',24.920],['10 m',28.120]],ax25_afsk1200:[['North America APRS',144.390],['ISS packet',145.825]],'olivia-8-250':[['80 m',3.583],['40 m',7.073],['30 m',10.143],['20 m',14.073],['17 m',18.099],['15 m',21.083],['12 m',24.923],['10 m',28.123]],'contestia-8-250':[['40 m',7.073],['20 m',14.073]],mfsk16:[['80 m',3.580],['40 m',7.080],['20 m',14.080]],psk63:[['80 m',3.580],['40 m',7.070],['20 m',14.070],['15 m',21.080],['10 m',28.120]],'dominoex-11':[['40 m',7.070],['20 m',14.070]],'thor-11':[['40 m',7.080],['20 m',14.080]],'mt63-1000l':[['80 m',3.590],['40 m',7.090],['20 m',14.090]],hell:[['80 m',3.580],['40 m',7.040],['20 m',14.063]]}};function updateDigitalPresets(preferred){{const mode=el('digitalMode').value,preset=el('digitalPreset'),items=digitalFrequencies[mode]||[],target=Number(preferred||el('digitalFrequency').value);preset.replaceChildren();for(const [band,mhz] of items){{const o=txt('option',`${{band}} · ${{mhz}} MHz`);o.value=mhz;preset.append(o)}}const custom=txt('option','Custom frequency');custom.value='custom';preset.append(custom);let best=[...preset.options].find(o=>o.value!=='custom'&&Math.abs(Number(o.value)-target)<0.000001);if(!best&&items.length)best=[...preset.options].find(o=>o.textContent.startsWith('20 m'))||preset.options[0];preset.value=best?best.value:'custom';if(best)el('digitalFrequency').value=best.value}};function updateDigitalModes(){{const family=el('digitalFamily').value,mode=el('digitalMode'),old=mode.value;mode.replaceChildren();for(const [value,label] of digitalModes[family]){{const o=txt('option',label);o.value=value;mode.append(o)}}if([...mode.options].some(o=>o.value===old))mode.value=old;el('digitalCarrierLabel').style.display=family==='fldigi'?'block':'none';el('digitalRetainLabel').style.display=family==='native'?'none':'block';el('digitalDurationLabel').firstChild.textContent=family==='weak'?'Capture cycles ': 'Duration (seconds) ';el('digitalDuration').value=family==='weak'?1:family==='fldigi'?30:15;updateDigitalPresets()}}el('digitalFamily').addEventListener('change',updateDigitalModes);el('digitalMode').addEventListener('change',()=>updateDigitalPresets());el('digitalPreset').addEventListener('change',()=>{{if(el('digitalPreset').value!=='custom')el('digitalFrequency').value=el('digitalPreset').value}});el('digitalFrequency').addEventListener('input',()=>{{const match=[...el('digitalPreset').options].find(o=>o.value!=='custom'&&Math.abs(Number(o.value)-Number(el('digitalFrequency').value))<0.000001);el('digitalPreset').value=match?match.value:'custom'}});updateDigitalModes();el('digitalForm').addEventListener('submit',async event=>{{event.preventDefault();const family=el('digitalFamily').value,mode=el('digitalMode').value,status=el('digitalResult'),button=el('digitalButton'),frequency=Math.round(Number(el('digitalFrequency').value)*1e6);let path,body;if(family==='native'){{path='/api/digital/native';body={{frequency_hz:frequency,mode,duration_seconds:Number(el('digitalDuration').value),retain_iq:false,include_plot:true}}}}else if(family==='weak'){{path='/api/digital/weak';body={{frequency_hz:frequency,mode,capture_cycles:Number(el('digitalDuration').value),align_to_utc:true,retain_iq:false,retain_audio:el('digitalRetainAudio').value==='true'}}}}else{{path='/api/digital/fldigi';body={{frequency_hz:frequency,mode,duration_seconds:Number(el('digitalDuration').value),carrier_audio_hz:Number(el('digitalCarrier').value),retain_iq:false,retain_audio:el('digitalRetainAudio').value==='true'}}}}button.disabled=true;status.textContent=family==='weak'?'Waiting for UTC cycle and decoding…':'Capturing and decoding…';try{{const result=await postOperation(path,body),decoder=result.decoder||{{}},textValue=decoder.text??result.text??(result.spots||[]).map(s=>s.message).join('\\n')??'';const diagnostics=result.audio_diagnostics||[];status.textContent=`${{mode.toUpperCase()}} complete · ${{result.decode_count??(result.spots||[]).length??0}} decode(s)${{decoder.confidence!=null?' · '+Math.round(decoder.confidence*100)+'% heuristic confidence':''}}`;el('digitalDiagnostics').textContent=diagnostics.length?diagnostics.map(x=>`Cycle ${{x.cycle}}: ${{String(x.classification||'unknown').replaceAll('_',' ')}} · RMS ${{x.rms_dbfs??'—'}} dBFS · peak ${{x.peak_dbfs??'—'}} dBFS · spectral contrast ${{x.spectral_contrast_db??'—'}} dB`).join(' | '):'';const plots=result.waterfall_artifacts||[];if(plots.length){{const image=el('digitalWaterfall');image.src=plots[plots.length-1].download_path+'?t='+Date.now();image.style.display='block'}}el('digitalText').textContent=textValue||JSON.stringify(result,null,2);el('digitalText').style.display='block';await refreshOperations()}}catch(e){{status.textContent='Decode failed: '+e.message}}finally{{button.disabled=false}}}});
function updateSstvOperation(){{const watch=el('sstvOperation').value==='watch',duration=el('sstvDuration');duration.min=watch?30:20;duration.max=watch?86400:310;duration.value=watch?3600:130}}el('sstvOperation').addEventListener('change',updateSstvOperation);el('sstvMode').addEventListener('change',()=>{{if(el('sstvMode').value==='nfm'&&Number(el('sstvFrequency').value)<60)el('sstvFrequency').value='145.800';else if(el('sstvMode').value==='usb'&&Number(el('sstvFrequency').value)>60)el('sstvFrequency').value='14.230'}});el('sstvForm').addEventListener('submit',async event=>{{event.preventDefault();const watch=el('sstvOperation').value==='watch',button=el('sstvStart'),status=el('sstvStatus'),common={{frequency_hz:Math.round(Number(el('sstvFrequency').value)*1e6),receiver_mode:el('sstvMode').value,retain_audio:el('sstvAudio').value==='true',deduplicate:el('sstvDeduplicate').value==='true'}};const body=watch?{{...common,watch_duration_seconds:Number(el('sstvDuration').value),rearm:true}}:{{...common,duration_seconds:Number(el('sstvDuration').value),retain_iq:false}};button.disabled=true;status.textContent=watch?'Starting VIS-triggered watcher…':'Starting SSTV capture…';try{{const result=await postOperation(watch?'/api/sstv/watch':'/api/sstv/decode',body);status.textContent=`${{watch?'Watcher':'Decode'}} started · ${{result.job_id}}. Use Status below to follow it.`;await refreshOperations()}}catch(e){{status.textContent='SSTV start failed: '+e.message}}finally{{button.disabled=false}}}});el('sstvFilter').addEventListener('input',refreshOperations);el('sstvHideDuplicates').addEventListener('change',refreshOperations);
const memoryBandwidths={{am:10000,nfm:12500,usb:3000,lsb:3000,cw:500,broadcast_fm:200000}};el('memoryMode').addEventListener('change',()=>el('memoryBandwidth').value=memoryBandwidths[el('memoryMode').value]);el('memoryCancel').addEventListener('click',resetMemoryForm);el('memoryForm').addEventListener('submit',async event=>{{event.preventDefault();const status=el('memoryStatus'),values={{name:el('memoryName').value,frequency_hz:Math.round(Number(el('memoryFrequency').value)*1e6),mode:el('memoryMode').value,bandwidth_hz:Math.round(Number(el('memoryBandwidth').value)),tags:el('memoryTags').value.split(',').map(v=>v.trim()).filter(Boolean),enabled:true}};if(editingMemoryId){{values.memory_id=editingMemoryId;values.replace_existing=true}}try{{await postOperation('/api/station-memories',values);status.textContent=editingMemoryId?'Station memory updated.':'Station memory saved. It is now available in Operations.';resetMemoryForm();await refreshMemories();await refreshOperations()}}catch(e){{status.textContent='Memory failed: '+e.message}}}});
function drawTrend(history){{const c=el('trend'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);x.strokeStyle='#263d59';x.beginPath();x.moveTo(35,10);x.lineTo(35,h-25);x.lineTo(w-10,h-25);x.stroke();if(!history.length)return;const max=Math.max(1,...history.map(v=>Number(v.completed_count||0)+Number(v.failed_count||0)));for(let i=0;i<history.length;i++){{const v=history[i],px=45+i*(w-65)/Math.max(1,history.length-1),good=Number(v.completed_count||0),bad=Number(v.failed_count||0);x.fillStyle='#38c793';x.fillRect(px-5,h-25-good*(h-45)/max,8,good*(h-45)/max);if(bad){{x.fillStyle='#ff9c8b';x.fillRect(px-5,h-25-(good+bad)*(h-45)/max,8,bad*(h-45)/max)}}if(v.change_count){{x.fillStyle='#ffbf5b';x.beginPath();x.arc(px,12,Math.min(8,3+Number(v.change_count)),0,Math.PI*2);x.fill()}}}}}}
function actionButton(label,handler,primary=false){{const b=txt('button',label);b.type='button';if(primary)b.className='primary';b.addEventListener('click',async()=>{{setButtonBusy(b,true,label+'…');showToast(label+' started');try{{await handler()}}catch(e){{showToast(label+' failed: '+e.message)}}finally{{setButtonBusy(b,false)}}}});return b}}
function renderProfiles(d){{const memories=el('profileMemories'),selected=[...memories.selectedOptions].map(o=>o.value);memories.replaceChildren();for(const m of d.station_memories||[]){{const o=txt('option',`${{m.name}} · ${{m.mode.toUpperCase()}} · ${{m.frequency_hz}} Hz`);o.value=m.memory_id;o.selected=selected.includes(m.memory_id);memories.append(o)}}if(!memories.options.length){{const o=txt('option','No station memories exist yet — create one in the Station memories section');o.disabled=true;memories.append(o);el('profileStatus').textContent='A station memory is required before a scan profile can be created.'}}rows('profileRows',d.station_scan_profiles||[],p=>{{const config=p.config||{{}},ids=config.memory_ids_or_names||[];const action=document.createElement('td');action.append(actionButton('Run scan now',async()=>{{const status=el('profileStatus');status.textContent='Scanning selected memories…';try{{const result=await postOperation('/api/station-scan-profiles/run',{{preset_id_or_name:p.preset_id}});status.textContent=`Scan completed: ${{result.completed_count||0}} received, ${{result.failed_count||0}} failed.`;await refresh();await refreshOperations()}}catch(e){{status.textContent='Scan failed: '+e.message}}}},true));return[txt('td',p.name),txt('td',ids.length?ids.length:'Filtered/all'),txt('td',(config.duration_seconds||5)+' sec'),action]}})}}
function renderOperations(d){{const status=el('stationStatus');status.replaceChildren();for(const s of d.station_status||[]){{const card=document.createElement('div');card.style.cssText='background:#0a1626;border:1px solid var(--line);border-radius:9px;padding:12px';const snr=s.estimated_snr_db==null?'—':Number(s.estimated_snr_db).toFixed(1)+' dB';card.innerHTML=`<strong></strong><div>${{s.state}} · ${{snr}}</div><small class="muted">${{(s.observed_at||'').replace('T',' ').slice(0,19)}}</small>`;card.querySelector('strong').textContent=s.name||s.memory_id;status.append(card)}}if(!status.children.length)status.textContent='No memory scans recorded yet.';drawTrend(d.station_scan_history||[]);const select=el('scheduleProfile'),selected=select.value;select.replaceChildren();for(const p of d.station_scan_profiles||[]){{const o=txt('option',p.name);o.value=p.preset_id;select.append(o)}}if(selected)select.value=selected;rows('scheduleRows',d.station_schedules||[],s=>{{const actions=document.createElement('td');actions.append(actionButton('Run now',async()=>{{try{{await postOperation('/api/station-schedules/run',{{schedule_id_or_name:s.schedule_id}});await refreshOperations()}}catch(e){{el('scheduleStatus').textContent=e.message}}}},true),' ',actionButton(s.enabled?'Disable':'Enable',async()=>{{try{{await postOperation('/api/station-schedules/toggle',{{schedule_id_or_name:s.schedule_id,enabled:!s.enabled}});await refreshOperations()}}catch(e){{el('scheduleStatus').textContent=e.message}}}}));return[txt('td',s.name),txt('td',s.preset_name),txt('td',(s.next_run_at||'').replace('T',' ').slice(0,19)),txt('td',s.enabled?'Enabled':'Disabled'),actions]}});const events=[];for(const scan of [...(d.station_scan_history||[])].reverse())for(const change of scan.changes||[])events.push({{time:scan.created_at,source:change.name||change.memory_id,event:change.kind}});for(const alert of d.recent_alerts||[])events.push({{time:alert.created_at,source:alert.rule_name||alert.event_type,event:alert.message||alert.event_type,alert}});events.sort((a,b)=>(b.time||'').localeCompare(a.time||''));rows('alertRows',events.slice(0,30),e=>{{const review=document.createElement('td');if(e.alert&&!e.alert.acknowledged)review.append(actionButton('Acknowledge',async()=>{{await postOperation('/api/alerts/acknowledge',{{event_id:e.alert.event_id}});await refreshOperations()}}));else review.textContent=e.alert?'Acknowledged':'Scan change';return[txt('td',(e.time||'').replace('T',' ').slice(0,19)),txt('td',e.source),txt('td',e.event),review]}});const audio=(d.artifacts||[]).filter(a=>(a.mime_type||'').startsWith('audio/')||a.kind==='audio_wav'),links=el('recentAudioLinks');links.replaceChildren();for(const a of audio){{const b=actionButton('Play '+a.filename,()=>{{const p=el('recentAudio');p.src=a.download_path+'?t='+Date.now();p.style.display='block';p.load();p.play()}});links.append(b,' ')}}}}
function renderManagement(d){{const memoryCount=(d.station_memories||[]).length,profileCount=(d.station_scan_profiles||[]).length,scanCount=(d.station_scan_history||[]).length;el('setupProgress').innerHTML=`<strong>Setup progress</strong><div style="margin-top:7px">${{memoryCount?'✓':'○'}} ${{memoryCount}} station memor${{memoryCount===1?'y':'ies'}} &nbsp; ${{profileCount?'✓':'○'}} ${{profileCount}} scan profile${{profileCount===1?'':'s'}} &nbsp; ${{scanCount?'✓':'○'}} ${{scanCount}} completed scan${{scanCount===1?'':'s'}}</div>`;[...(el('profileRows').rows||[])].forEach((row,i)=>{{const p=(d.station_scan_profiles||[])[i];if(!p)return;row.lastElementChild.append(' ',actionButton('Delete',async()=>{{if(!confirm(`Delete scan profile “${{p.name}}”? Schedules using it must be deleted first.`))return;try{{await postOperation('/api/station-scan-profiles/delete',{{preset_id_or_name:p.preset_id,confirm_delete:true}});el('profileStatus').textContent='Profile deleted.';await refreshOperations()}}catch(e){{el('profileStatus').textContent='Delete failed: '+e.message}}}}))}});[...(el('scheduleRows').rows||[])].forEach((row,i)=>{{const s=(d.station_schedules||[])[i];if(!s)return;row.lastElementChild.append(' ',actionButton('Delete',async()=>{{if(!confirm(`Delete schedule “${{s.name}}”?`))return;try{{await postOperation('/api/station-schedules/delete',{{schedule_id_or_name:s.schedule_id,confirm_delete:true}});el('scheduleStatus').textContent='Schedule deleted.';await refreshOperations()}}catch(e){{el('scheduleStatus').textContent='Delete failed: '+e.message}}}}))}})}}
function renderBand(d){{const jobs=(d.jobs||[]).filter(j=>['band_scan','band_survey'].includes(j.job_type));rows('bandJobRows',jobs,j=>{{const progress=txt('td',(j.summary&&j.summary.completed_steps!=null)?`${{j.summary.completed_steps}} / ${{j.summary.planned_steps||'—'}}`:'—'),actions=document.createElement('td'),statusButton=actionButton('Status',async()=>{{try{{const s=await postOperation('/api/band-jobs/status',{{job_id:j.job_id}});el('bandStatus').textContent=`${{j.job_id}} · ${{s.state||'unknown'}} · ${{s.phase||''}} · ${{s.completed_steps||0}}/${{(s.config||{{}}).planned_steps||'—'}} steps`;await refreshOperations()}}catch(e){{el('bandStatus').textContent='Status failed: '+e.message}}}});actions.append(statusButton);if(['queued','running','stopping'].includes(j.state))actions.append(' ',actionButton('Stop',async()=>{{if(!confirm('Stop this RF job after its current capture?'))return;try{{await postOperation('/api/band-jobs/stop',{{job_id:j.job_id}});el('bandStatus').textContent='Stop requested.';await refreshOperations()}}catch(e){{el('bandStatus').textContent='Stop failed: '+e.message}}}}));return[txt('td',j.job_type.replace('_',' ')),txt('td',j.state),txt('td',(j.created_at||'').replace('T',' ').slice(0,19)),progress,actions]}});rows('presetRows',d.rf_presets||[],p=>{{const action=document.createElement('td');action.append(actionButton('Run',async()=>{{if(!confirm(`Run preset “${{p.name}}” now? It may occupy the receiver.`))return;el('bandStatus').textContent='Starting preset…';try{{const result=await postOperation('/api/presets/run',{{preset_id_or_name:p.preset_id}});el('bandStatus').textContent=`Preset started${{result.job_id?' · '+result.job_id:''}}.`;await refreshOperations()}}catch(e){{el('bandStatus').textContent='Preset failed: '+e.message}}}},true));return[txt('td',p.name),txt('td',p.preset_type),txt('td',p.description||'—'),action]}})}}
let fmDirectoryBusy=false;async function listenToDirectoryStation(s,button){{if(fmDirectoryBusy)return;fmDirectoryBusy=true;button.disabled=true;button.textContent='Receiving…';const panel=el('fmDirectoryPlayer'),status=el('fmDirectoryStatus'),player=el('fmDirectoryAudio'),download=el('fmDirectoryDownload'),station=s.ps||'Unknown station',mhz=(s.frequency_hz/1e6).toFixed(1);panel.style.display='block';el('fmDirectoryNow').textContent=`${{station}} · ${{mhz}} MHz`;status.textContent='Capturing 10 seconds of wideband FM audio…';el('fmDirectoryRds').textContent='';download.style.display='none';panel.scrollIntoView({{behavior:'smooth',block:'nearest'}});try{{const response=await fetch('/api/broadcast-fm',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{frequency_hz:s.frequency_hz,duration_seconds:10,stereo:true,deemphasis_us:75,decode_rds_data:true}})}}),d=await response.json();if(!response.ok)throw new Error(d.detail||d.error);if(!d.audio_artifact)throw new Error('The receiver completed without producing an audio file');player.src=d.audio_artifact.download_path+'?t='+Date.now();player.load();download.href=d.audio_artifact.download_path;download.download=d.audio_artifact.filename||'';download.style.display='inline-block';const groups=(d.rds||{{}}).group_count||0,decoded=(d.rds||{{}}).station||{{}};el('fmDirectoryRds').textContent=groups?`RDS: ${{decoded.program_service||station}} · ${{decoded.radiotext||''}} · ${{groups}} valid group(s)`:'No checksum-valid RDS groups decoded in this capture.';status.textContent='Capture complete. Playing the new recording…';try{{await player.play()}}catch(e){{status.textContent='Capture complete. Press Play below to hear the recording.'}}await refresh()}}catch(e){{status.textContent='Listen failed: '+e.message}}finally{{fmDirectoryBusy=false;button.disabled=false;button.textContent='Listen'}}}}
function renderFmBandMap(job){{const map=el('fmBandMap'),summary=el('fmBandSummary');map.replaceChildren();if(!job){{summary.textContent='Start a survey to visualize scanned, occupied, and decoded channels.';return}}const config=job.config||{{}},progress=job.summary||{{}},start=Number(config.start_frequency_hz||87900000),stop=Number(config.stop_frequency_hz||107900000),spacing=Number(config.channel_spacing_hz||200000),count=Number(progress.channel_count||config.channel_count||Math.floor((stop-start)/spacing)+1),scanned=Number(progress.discovery_index||0),candidates=new Set(progress.candidate_frequencies_hz||[]),decoded=new Set(progress.decoded_frequencies_hz||[]),active=['queued','running','stopping'].includes(job.state);for(let index=0;index<count;index++){{const frequency=start+index*spacing,bar=document.createElement('span');bar.className='fmChannel';if(index<scanned)bar.classList.add('scanned');if(candidates.has(frequency))bar.classList.add('candidate');if(decoded.has(frequency))bar.classList.add('decoded');if(active&&((progress.phase==='discovery'&&index===scanned)||(progress.phase==='rds_collection'&&frequency===[...candidates][Number(progress.station_index||0)])))bar.classList.add('current');bar.title=`${{(frequency/1e6).toFixed(1)}} MHz · ${{bar.classList.contains('decoded')?'RDS/audio collected':bar.classList.contains('candidate')?'signal candidate':index<scanned?'scanned':'not scanned'}}`;map.append(bar)}}summary.textContent=`${{(start/1e6).toFixed(1)}}–${{(stop/1e6).toFixed(1)}} MHz · ${{scanned}}/${{count}} channels scanned · ${{candidates.size}} candidates · ${{decoded.size}} collected · ${{job.state}}`}}
function renderFmDirectory(d){{renderFmBandMap((d.fm_survey_jobs||[])[0]);rows('fmSurveyRows',d.fm_survey_jobs||[],j=>{{const result=j.summary||{{}},progress=`${{result.discovery_index||0}} discovered · ${{result.station_index||0}} decoded`,actions=document.createElement('td');actions.append(actionButton('Status',async()=>{{try{{const s=await postOperation('/api/fm-surveys/status',{{job_id:j.job_id}});el('fmSurveyStatus').textContent=`${{s.state}} · ${{s.phase}} · discovery ${{s.discovery_index||0}}/${{(s.config||{{}}).channel_count||'—'}} · stations ${{s.station_index||0}}/${{(s.candidates||[]).length||'—'}}`;await refreshOperations()}}catch(e){{el('fmSurveyStatus').textContent='Status failed: '+e.message}}}}));if(['queued','running','stopping'].includes(j.state))actions.append(' ',actionButton('Stop',async()=>{{if(!confirm('Stop the FM survey after its current channel?'))return;try{{await postOperation('/api/fm-surveys/stop',{{job_id:j.job_id}});el('fmSurveyStatus').textContent='Stop requested; this survey can be resumed later.';await refreshOperations()}}catch(e){{el('fmSurveyStatus').textContent='Stop failed: '+e.message}}}}));if(['stopped','failed','interrupted'].includes(j.state))actions.append(' ',actionButton('Resume',async()=>{{try{{await postOperation('/api/fm-surveys/start',{{resume_job_id:j.job_id}});el('fmSurveyStatus').textContent='Survey resumed.';await refreshOperations()}}catch(e){{el('fmSurveyStatus').textContent='Resume failed: '+e.message}}}},true));return[txt('td',j.state),txt('td',result.phase||'—'),txt('td',(j.created_at||'').replace('T',' ').slice(0,19)),txt('td',progress),actions]}});const query=el('fmStationFilter').value.trim().toLowerCase(),stations=(d.fm_stations||[]).filter(s=>!query||`${{s.frequency_hz}} ${{s.ps||''}} ${{s.pty_name||''}} ${{s.radiotext||''}}`.toLowerCase().includes(query));rows('fmStationRows',stations,s=>{{const action=document.createElement('td'),listen=actionButton('Listen',()=>listenToDirectoryStation(s,listen),true),memory=actionButton('Save memory',()=>{{el('memoryName').value=s.ps||`${{(s.frequency_hz/1e6).toFixed(1)}} FM`;el('memoryFrequency').value=(s.frequency_hz/1e6).toFixed(1);el('memoryMode').value='broadcast_fm';el('memoryBandwidth').value=200000;showView('memories')}});listen.disabled=fmDirectoryBusy;action.append(listen,' ',memory);const rds=[s.pi_code,s.pty_name,s.radiotext].filter(Boolean).join(' · '),quality=`${{s.stereo_detected?'Stereo':'Mono/unknown'}} · ${{s.estimated_snr_db==null?'—':Number(s.estimated_snr_db).toFixed(1)+' dB'}}`;return[txt('td',(s.frequency_hz/1e6).toFixed(1)+' MHz'),txt('td',s.ps||'Unknown'),txt('td',rds||'No RDS'),txt('td',quality),txt('td',(s.last_seen_at||'').replace('T',' ').slice(0,19)),action]}})}}
const jobViews={{band_scan:'scan',band_survey:'scan',fm_broadcast_survey:'fm',sstv_decode:'sstv',sstv_watch:'sstv',station_memory_scan:'operations',weak_signal_decode:'digital',fldigi_decode:'digital',digital_decode:'digital',satellite_receive:'operations'}};const jobLabels={{band_scan:'Band scan',band_survey:'Band survey',fm_broadcast_survey:'FM survey',sstv_decode:'SSTV decode',sstv_watch:'SSTV watcher',station_memory_scan:'Memory scan',weak_signal_decode:'Weak-signal decode',fldigi_decode:'Fldigi decode',digital_decode:'Digital decode',satellite_receive:'Satellite receive'}};function jobLabel(j){{return jobLabels[j.job_type]||(j.job_type||'RF operation').replaceAll('_',' ')}}function jobProgress(j){{const s=j.summary||{{}},c=j.config||{{}};if(s.completed_steps!=null)return `${{s.completed_steps}}/${{s.planned_steps||c.planned_steps||'—'}} steps`;if(s.discovery_index!=null)return `${{s.discovery_index}} discovered · ${{s.station_index||0}} decoded`;if(s.image_count!=null)return `${{s.image_count}} image(s)`;if(s.phase)return String(s.phase).replaceAll('_',' ');const started=Date.parse(j.started_at||j.created_at||'');if(Number.isFinite(started)&&['queued','running','stopping'].includes(j.state))return `${{Math.max(0,Math.floor((Date.now()-started)/1000))}} sec elapsed`;return j.state||'unknown'}}function stopSpec(j){{if(['band_scan','band_survey'].includes(j.job_type))return['/api/band-jobs/stop',{{job_id:j.job_id}}];if(j.job_type==='fm_broadcast_survey')return['/api/fm-surveys/stop',{{job_id:j.job_id}}];if(j.job_type==='sstv_decode')return['/api/sstv/stop',{{job_id:j.job_id}}];if(j.job_type==='sstv_watch')return['/api/sstv/watch-stop',{{job_id:j.job_id}}];return null}}async function stopActiveJob(j){{const spec=stopSpec(j);if(!spec)return;if(!confirm(`Stop ${{jobLabel(j)}} after its current receiver operation?`))return;const button=el('receiverStop');button.disabled=true;button.textContent='Stopping…';try{{await postOperation(spec[0],spec[1]);el('receiverDetail').textContent='Stop requested. Waiting for the current receiver operation to finish.';await refreshOperations()}}catch(e){{el('receiverDetail').textContent='Stop failed: '+e.message}}finally{{button.disabled=false;button.textContent='Stop'}}}}function renderGlobalActivity(d){{const active=d.active_rf_job,bar=el('receiverBar'),open=el('receiverOpen'),stop=el('receiverStop');bar.classList.toggle('busy',Boolean(active));if(active){{el('receiverState').textContent=jobLabel(active);el('receiverDetail').textContent=`${{active.state||'running'}} · ${{jobProgress(active)}} · ${{active.job_id}}`;const view=jobViews[active.job_type];open.style.display=view?'inline-block':'none';open.onclick=()=>showView(view);const spec=stopSpec(active);stop.style.display=spec?'inline-block':'none';stop.onclick=()=>stopActiveJob(active)}}else{{el('receiverState').textContent='Receiver ready';el('receiverDetail').textContent='No RF operation is active.';open.style.display=stop.style.display='none'}}const jobs=d.jobs||[];el('activityCount').textContent=jobs.filter(j=>['queued','running','stopping'].includes(j.state)).length;const items=el('activityItems');items.replaceChildren();for(const j of jobs.slice(0,20)){{const item=document.createElement('div');item.className='activityItem';const top=document.createElement('div');top.style.cssText='display:flex;align-items:center;gap:8px';const title=txt('strong',jobLabel(j)),state=txt('span',j.state||'unknown');state.className='pill';top.append(title,state);const detail=txt('div',`${{jobProgress(j)}} · ${{(j.created_at||'').replace('T',' ').slice(0,19)}}`);detail.className='muted';item.append(top,detail);const view=jobViews[j.job_type];if(view)item.append(actionButton('Open',()=>{{showView(view);activityDrawer.classList.remove('open')}}));items.append(item)}}if(!jobs.length)items.append(txt('p','No receiver activity has been recorded yet.'))}}
function renderHome(d){{const favorites=(d.station_memories||[]).filter(m=>(m.tags||[]).some(t=>t.toLowerCase()==='favorite')),cards=el('favoriteCards');cards.replaceChildren();for(const m of favorites){{const card=document.createElement('div');card.style.cssText='padding:14px;border:1px solid var(--line);border-radius:9px;background:#08111f';card.append(txt('strong',m.name),txt('div',`${{(m.frequency_hz/1e6).toFixed(6).replace(/0+$/,'').replace(/\\.$/,'')}} MHz · ${{m.mode.toUpperCase()}}`),actionButton('Listen',()=>listenMemory(m),true),' ',actionButton('Remove favorite',async()=>{{try{{await toggleFavorite(m)}}catch(e){{el('error').textContent='Favorite failed: '+e.message;el('error').style.display='block'}}}}));cards.append(card)}}if(!favorites.length){{const empty=txt('p','No favorites yet. Add the tag “favorite” to a station memory to place it here.');empty.className='muted';cards.append(empty)}}const recent=el('homeRecent');recent.replaceChildren();for(const j of (d.jobs||[]).slice(0,6)){{const line=document.createElement('div');line.className='activityItem';line.append(txt('strong',jobLabel(j)),txt('span',` · ${{j.state}} · ${{(j.created_at||'').replace('T',' ').slice(0,19)}}`));const view=jobViews[j.job_type];if(view)line.append(' ',actionButton('Open',()=>showView(view)));recent.append(line)}}if(!(d.jobs||[]).length)recent.append(txt('p','No receiver activity yet.'))}}
function renderDigital(d){{const query=el('spotFilter').value.trim().toLowerCase(),spots=(d.weak_signal_spots||[]).filter(s=>!query||`${{s.mode}} ${{s.callsign||''}} ${{s.grid||''}} ${{s.message||''}}`.toLowerCase().includes(query));rows('spotRows',spots,s=>{{const frequency=s.dial_frequency_hz||s.rf_frequency_hz||0,actions=document.createElement('td');actions.append(actionButton('Decode again',()=>configureDigital('weak',s.mode,frequency),true),' ',actionButton('Save memory',()=>prepareMemory(`${{(s.mode||'Digital').toUpperCase()}} ${{s.callsign||'channel'}}`,frequency,'usb',3000,['digital',(s.mode||'').toLowerCase()])));return[txt('td',(s.captured_at||'').replace('T',' ').slice(0,19)),txt('td',(s.mode||'').toUpperCase()),txt('td',((s.rf_frequency_hz||frequency)/1e6).toFixed(6)+' MHz'),txt('td',s.callsign),txt('td',s.grid),txt('td',s.snr_db==null?'—':s.snr_db+' dB'),txt('td',s.message),actions]}});if(!spots.length)guidedEmpty('spotRows',8,query?'No spots match this filter.':'No weak-signal spots yet. FT8 on 14.074 MHz is a good first test.','Try FT8 on 20 m',()=>configureDigital('weak','ft8',14074000));const fldigi=d.fldigi_decodes||[];rows('fldigiRows',fldigi,x=>{{const frequency=x.dial_frequency_hz||0,actions=document.createElement('td');actions.append(actionButton('Decode again',()=>configureDigital('fldigi',x.mode,frequency),true),' ',actionButton('Save memory',()=>prepareMemory(`${{x.mode||'Fldigi'}} channel`,frequency,'usb',3000,['digital',(x.mode||'').toLowerCase()])));return[txt('td',(x.captured_at||'').replace('T',' ').slice(0,19)),txt('td',x.mode),txt('td',(frequency/1e6).toFixed(6)+' MHz'),txt('td',x.text),actions]}});if(!fldigi.length)guidedEmpty('fldigiRows',5,'No Fldigi text has been decoded yet. Start with a known active frequency and matching mode.','Configure Fldigi decode',()=>configureDigital('fldigi','psk31',14070000))}}
function renderSstv(d){{rows('sstvJobRows',d.sstv_jobs||[],j=>{{const watch=j.job_type==='sstv_watch',actions=document.createElement('td');actions.append(actionButton('Status',async()=>{{try{{const s=await postOperation(watch?'/api/sstv/watch-status':'/api/sstv/status',{{job_id:j.job_id}});el('sstvStatus').textContent=`${{j.job_id}} · ${{s.state}} · ${{s.phase||'—'}}${{s.trigger_count!=null?' · '+s.trigger_count+' trigger(s)':''}}`;await refreshOperations()}}catch(e){{el('sstvStatus').textContent='Status failed: '+e.message}}}}));if(['queued','running','stopping'].includes(j.state))actions.append(' ',actionButton('Stop',async()=>{{if(!confirm('Stop this SSTV receiver after its current capture?'))return;try{{await postOperation(watch?'/api/sstv/watch-stop':'/api/sstv/stop',{{job_id:j.job_id}});el('sstvStatus').textContent='Stop requested.';await refreshOperations()}}catch(e){{el('sstvStatus').textContent='Stop failed: '+e.message}}}}));return[txt('td',watch?'Watcher':'Decode'),txt('td',j.state),txt('td',(j.created_at||'').replace('T',' ').slice(0,19)),actions]}});const query=el('sstvFilter').value.trim().toLowerCase(),hide=el('sstvHideDuplicates').checked,images=(d.sstv_images||[]).filter(x=>(!hide||!x.duplicate_of)&&(!query||`${{x.image_id}} ${{x.sstv_mode||''}} ${{x.frequency_hz||''}}`.toLowerCase().includes(query))),gallery=el('sstvGallery');gallery.replaceChildren();if(!images.length)gallery.append(txt('p','No matching SSTV images yet.'));for(const x of images){{const card=document.createElement('article');card.style.cssText='background:#08111f;border:1px solid var(--line);border-radius:9px;overflow:hidden';const link=document.createElement('a'),image=document.createElement('img');link.href=x.image_url;link.target='_blank';image.src=x.image_url;image.alt=`${{x.sstv_mode||'SSTV'}} image at ${{x.frequency_hz}} Hz`;image.loading='lazy';image.style.cssText='width:100%;height:210px;object-fit:contain;background:#000;display:block';link.append(image);const meta=document.createElement('div');meta.style.padding='10px';meta.append(txt('b',x.sstv_mode||'Unknown SSTV mode'),txt('div',`${{(x.frequency_hz/1e6).toFixed(6)}} MHz · ${{x.width}}×${{x.height}}`),txt('div',(x.captured_at||'').replace('T',' ').slice(0,19)));if(x.duplicate_of)meta.append(txt('div','Duplicate of '+x.duplicate_of));card.append(link,meta);gallery.append(card)}}}}
function formatDuration(seconds){{seconds=Math.max(0,Math.round(seconds));if(seconds<60)return seconds+' sec';if(seconds<3600)return Math.floor(seconds/60)+'m '+seconds%60+'s';return Math.floor(seconds/3600)+'h '+Math.floor(seconds%3600/60)+'m'}}
function jobPercent(j){{const s=j.summary||{{}},c=j.config||{{}};if(Number.isFinite(Number(s.progress_percent)))return Math.max(0,Math.min(100,Number(s.progress_percent)));if(['completed','stopped'].includes(j.state))return 100;if(s.completed_steps!=null&&Number(s.planned_steps||c.planned_steps))return 100*Number(s.completed_steps)/Number(s.planned_steps||c.planned_steps);return 0}}
function jobTiming(j,pct){{const started=Date.parse(j.started_at||j.created_at||''),ended=Date.parse(j.completed_at||'');if(!Number.isFinite(started))return '';const elapsed=((Number.isFinite(ended)?ended:Date.now())-started)/1000;if(['queued','running','stopping'].includes(j.state)&&pct>1&&pct<100)return formatDuration(elapsed)+' elapsed · about '+formatDuration(elapsed*(100-pct)/pct)+' remaining';return formatDuration(elapsed)+' elapsed'}}
function renderJobProgressCards(d){{const host=el('jobProgressCards');host.replaceChildren();for(const j of (d.jobs||[]).slice(0,6)){{const pct=jobPercent(j),card=document.createElement('article');card.className='jobCard '+(j.state||'');const top=document.createElement('div');top.className='jobTop';top.append(txt('strong',jobLabel(j)));const state=txt('span',j.state||'unknown');state.className='pill';top.append(state);const track=document.createElement('div');track.className='progressTrack';const fill=document.createElement('div');fill.className='progressFill';fill.style.width=pct+'%';track.append(fill);const s=j.summary||{{}},phase=String(s.phase||j.phase||jobProgress(j)).replaceAll('_',' '),detail=txt('div',phase+' · '+Math.round(pct)+'%');detail.className='muted';const timing=txt('small',jobTiming(j,pct));timing.className='muted';card.append(top,detail,track,timing);if(s.current_frequency_hz)card.append(txt('div','Current: '+(Number(s.current_frequency_hz)/1e6).toFixed(6)+' MHz'));if(j.error){{const error=txt('div',j.error);error.className='jobError';card.append(error)}}const view=jobViews[j.job_type];if(view)card.append(actionButton('Open',()=>showView(view)));const spec=stopSpec(j);if(spec&&['queued','running','stopping'].includes(j.state))card.append(' ',actionButton('Stop',()=>stopActiveJob(j)));host.append(card)}}if(!host.children.length){{const p=txt('p','No RF jobs yet. Use a Quick Start action above.');p.className='muted';host.append(p)}}}}
function renderResultCards(d){{const host=el('recentResultCards');host.replaceChildren();const artifacts=d.artifacts||[],jobs=new Map((d.jobs||[]).map(j=>[j.job_id,j]));for(const a of artifacts.filter(a=>(a.mime_type||'').startsWith('image/')||(a.mime_type||'').startsWith('audio/')).slice(0,6)){{const card=document.createElement('article');card.className='resultCard';const job=jobs.get(a.job_id);if((a.mime_type||'').startsWith('image/')){{const link=document.createElement('a'),img=document.createElement('img');link.href=a.download_path;link.target='_blank';img.src=a.download_path;img.loading='lazy';img.alt=a.filename;link.append(img);card.append(link)}}else{{const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=a.download_path;card.append(audio)}}card.append(txt('strong',job?jobLabel(job):String(a.kind||'RF result').replaceAll('_',' ')),txt('div',a.filename));const link=txt('a','Download');link.href=a.download_path;card.append(link);host.append(card)}}if(!host.children.length){{const p=txt('p','Plots, waterfalls, decoded images, and audio will appear here.');p.className='muted';host.append(p)}}}}
function renderBandProgress(d){{const job=(d.jobs||[]).find(j=>['band_scan','band_survey'].includes(j.job_type)),host=el('bandProgressMap'),summary=el('bandProgressSummary');host.replaceChildren();if(!job){{summary.textContent='Start a scan to see its position and ETA.';return}}const pct=jobPercent(job),fill=document.createElement('div');fill.className='progressFill';fill.style.width=pct+'%';host.append(fill);const c=job.config||{{}},s=job.summary||{{}},start=Number(c.start_frequency_hz||0)/1e6,stop=Number(c.stop_frequency_hz||0)/1e6,current=Number(s.current_frequency_hz||c.start_frequency_hz||0)/1e6;summary.textContent=`${{start.toFixed(3)}} MHz → ${{stop.toFixed(3)}} MHz · now ${{current.toFixed(6)}} MHz · ${{Math.round(pct)}}% · ${{jobTiming(job,pct)}}`}}
async function refreshOperations(){{try{{const response=await fetch('/api/dashboard',{{cache:'no-store'}}),d=await response.json();if(!response.ok)throw new Error(d.error||response.status);renderGlobalActivity(d);renderHome(d);renderJobProgressCards(d);renderResultCards(d);renderProfiles(d);renderOperations(d);renderManagement(d);renderBand(d);renderBandProgress(d);renderFmDirectory(d);renderDigital(d);renderSstv(d)}}catch(e){{el('scheduleStatus').textContent='Operations refresh failed: '+e.message}}}}
async function refreshDecoderCapabilities(){{try{{const c=await postOperation('/api/digital/capabilities',{{}}),fldigi=c.fldigi||{{}};el('decoderStatus').textContent=`Native decoders ready · WSJT-X ${{c.wsjt_x?.available?'available':'not detected'}} · Fldigi ${{fldigi.available||fldigi.connected?'available':'not detected'}}`}}catch(e){{el('decoderStatus').textContent='Capability check failed: '+e.message}}}}
async function refreshSstvCapabilities(){{try{{const c=await postOperation('/api/sstv/capabilities',{{}});el('sstvCapabilities').textContent=`Decoder ${{c.available?'available':'not detected'}} · ${{(c.receiver_modes||['USB','NFM']).join(', ')}} · ${{(c.modes||[]).length||'multiple'}} VIS modes`}}catch(e){{el('sstvCapabilities').textContent='Capability check failed: '+e.message}}}}
el('profileForm').addEventListener('submit',async event=>{{event.preventDefault();const ids=[...el('profileMemories').selectedOptions].map(o=>o.value),status=el('profileStatus');if(!ids.length){{status.textContent='Select at least one station memory.';return}}try{{const duration=Number(el('profileDuration').value);await postOperation('/api/station-scan-profiles',{{name:el('profileName').value,memory_ids_or_names:ids,duration_seconds:duration,max_memories:ids.length,compare_previous:true}});status.textContent='Profile created. Use Run scan now below to collect the first observations.';event.target.reset();await refreshOperations()}}catch(e){{status.textContent='Profile failed: '+e.message}}}});
el('scheduleForm').addEventListener('submit',async event=>{{event.preventDefault();const status=el('scheduleStatus');try{{await postOperation('/api/station-schedules',{{name:el('scheduleName').value,preset_id_or_name:el('scheduleProfile').value,interval_seconds:Math.round(Number(el('scheduleMinutes').value)*60),enabled:el('scheduleEnabled').value==='true'}});status.textContent='Schedule saved.';event.target.reset();await refreshOperations()}}catch(e){{status.textContent='Schedule failed: '+e.message}}}});
async function refresh(){{try{{const response=await fetch('/api/dashboard',{{cache:'no-store'}});if(!response.ok)throw new Error('Status '+response.status);const d=await response.json();el('service').textContent=d.active_long_job?'Busy':'Online';el('service').className=d.active_long_job?'':'on';el('receivers').textContent=d.coordinator.receiver_count;el('leases').textContent=d.coordinator.active_lease_count;el('jobcount').textContent=d.jobs.length;rows('receiverRows',d.coordinator.receivers,r=>[txt('td',r.name),txt('td',r.backend),txt('td',r.role),txt('td',r.lease?'Leased':r.enabled?(r.verified?'Ready':'Unverified'):'Disabled')]);rows('jobRows',d.jobs,j=>[txt('td',j.job_type),txt('td',j.state),txt('td',(j.created_at||'').replace('T',' ').slice(0,19))]);rows('artifactRows',d.artifacts,a=>{{const link=txt('a',a.filename);link.href=a.download_path;const td=document.createElement('td');td.append(link);return[td,txt('td',a.kind),txt('td',fmt(a.size_bytes))]}});const s=d.storage;el('storage').textContent=JSON.stringify(s,null,2);el('storage').style.whiteSpace='pre-wrap';el('error').style.display='none'}}catch(e){{el('error').textContent='Dashboard refresh failed: '+e.message;el('error').style.display='block'}}}}
el('captureForm').addEventListener('submit',async event=>{{event.preventDefault();const button=el('captureButton'),status=el('captureStatus');button.disabled=true;status.textContent='Capturing and analyzing…';try{{const response=await fetch('/api/spectrum',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{center_frequency_hz:Number(el('frequency').value),duration_seconds:Number(el('duration').value),fft_size:Number(el('fft').value),threshold_above_noise_db:Number(el('threshold').value),max_peaks:20}})}});const d=await response.json();if(!response.ok)throw new Error(d.detail||d.error);status.textContent=`Completed ${{(d.center_frequency_hz/1e6).toFixed(6)}} MHz · ${{d.peak_count}} peaks · noise floor ${{Number(d.relative_noise_floor_db).toFixed(1)}} dB (relative)`;if(d.plot_artifact){{const image=el('spectrumPlot');image.src=d.plot_artifact.download_path+'?t='+Date.now();image.style.display='block'}}await refresh()}}catch(e){{status.textContent='Capture failed: '+e.message}}finally{{button.disabled=false}}}});refresh();setInterval(refresh,15000);
const modeDefaults={{am:10000,nfm:12500,usb:3000,lsb:3000,cw:500}};el('audioMode').addEventListener('change',()=>el('audioBandwidth').value=modeDefaults[el('audioMode').value]);el('audioForm').addEventListener('submit',async event=>{{event.preventDefault();const button=el('audioButton'),status=el('audioStatus');button.disabled=true;status.textContent='Capturing and demodulating…';try{{const response=await fetch('/api/demodulate',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{frequency_hz:Number(el('audioFrequency').value),mode:el('audioMode').value,bandwidth_hz:Number(el('audioBandwidth').value),duration_seconds:Number(el('audioDuration').value),fft_size:16384,cw_tone_hz:700}})}});const d=await response.json();if(!response.ok)throw new Error(d.detail||d.error);const snr=d.metrics.estimated_snr_db;status.textContent=`${{d.mode.toUpperCase()}} · ${{d.bandwidth_hz}} Hz · estimated SNR ${{Number(snr).toFixed(1)}} dB (relative)`;if(d.audio_artifact){{const player=el('audioPlayer');player.src=d.audio_artifact.download_path+'?t='+Date.now();player.style.display='block';player.load();try{{await player.play()}}catch(e){{status.textContent+=' · Press Play to hear the recording.'}}}}for(const [id,item] of [['rfAnalysisPlot',d.rf_plot_artifact],['audioAnalysisPlot',d.audio_plot_artifact]])if(item){{const image=el(id);image.src=item.download_path+'?t='+Date.now();image.style.display='block'}}await refresh()}}catch(e){{status.textContent='Demodulation failed: '+e.message}}finally{{button.disabled=false}}}});
el('fmForm').addEventListener('submit',async event=>{{event.preventDefault();const button=el('fmButton'),status=el('fmStatus');button.disabled=true;status.textContent='Receiving wideband FM and decoding RDS…';try{{const response=await fetch('/api/broadcast-fm',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify({{frequency_hz:Math.round(Number(el('fmFrequency').value)*1e6),duration_seconds:Number(el('fmDuration').value),stereo:el('fmStereo').value==='true',deemphasis_us:Number(el('fmDeemphasis').value),decode_rds_data:true}})}});const d=await response.json();if(!response.ok)throw new Error(d.detail||d.error);const m=d.metrics||{{}};status.textContent=`${{(d.frequency_hz/1e6).toFixed(1)}} MHz · ${{m.audio_channels||1}} channel(s) · stereo pilot ${{m.stereo_detected?'detected':'not detected'}}`;const station=(d.rds||{{}}).station||{{}},groups=(d.rds||{{}}).group_count||0;el('rdsStatus').textContent=groups?`RDS: ${{station.program_service||'Unknown station'}} · ${{station.radiotext||''}} · ${{groups}} valid group(s)`:'No checksum-valid RDS groups decoded in this capture.';if(d.audio_artifact){{const player=el('fmPlayer');player.src=d.audio_artifact.download_path+'?t='+Date.now();player.style.display='block';player.load();try{{await player.play()}}catch(e){{status.textContent+=' · Press Play to hear the recording.'}}}}if(d.multiplex_plot_artifact){{const image=el('fmPlot');image.src=d.multiplex_plot_artifact.download_path+'?t='+Date.now();image.style.display='block'}}await refresh()}}catch(e){{status.textContent='Broadcast FM failed: '+e.message}}finally{{button.disabled=false}}}});refresh();refreshMemories();refreshOperations();refreshDecoderCapabilities();refreshSstvCapabilities();el('spotFilter').addEventListener('input',refreshOperations);setInterval(refresh,15000);setInterval(refreshMemories,15000);setInterval(refreshOperations,15000);setInterval(()=>{{if(el('receiverBar').classList.contains('busy'))refreshOperations()}},3000);
</script></body></html>""".encode()

    async def _artifact(self, artifact_id: str, send: Callable) -> None:
        try:
            artifact = self.catalog.get_artifact(artifact_id)
            path = Path(artifact["path"])
            size = path.stat().st_size
        except (ValueError, FileNotFoundError):
            await _response(send, 404, _json_bytes({"error": "artifact_not_found"}))
            return
        except OSError:
            await _response(send, 500, _json_bytes({"error": "artifact_unavailable"}))
            return

        disposition = f"attachment; filename*=UTF-8''{quote(artifact['filename'])}".encode("ascii")
        headers = [
            (b"content-type", artifact["mime_type"].encode("ascii", "replace")),
            (b"content-length", str(size).encode()),
            (b"content-disposition", disposition),
            (b"x-content-type-options", b"nosniff"),
            (b"cache-control", b"private, no-store"),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        try:
            with path.open("rb") as source:
                while chunk := source.read(256 * 1024):
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
        except OSError:
            pass
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _sstv_image(self, image_id: str, send: Callable) -> None:
        try:
            image = self.catalog.get_sstv_image(image_id)
            path = Path(image["image_path"])
            size = path.stat().st_size
        except (ValueError, FileNotFoundError):
            await _response(send, 404, _json_bytes({"error": "sstv_image_not_found"}))
            return
        except OSError:
            await _response(send, 500, _json_bytes({"error": "sstv_image_unavailable"}))
            return
        headers = [(b"content-type", b"image/png"), (b"content-length", str(size).encode()),
                   (b"x-content-type-options", b"nosniff"),
                   (b"cache-control", b"private, no-store")]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        try:
            with path.open("rb") as source:
                while chunk := source.read(256 * 1024):
                    await send({"type": "http.response.body", "body": chunk,
                                "more_body": True})
        except OSError:
            pass
        await send({"type": "http.response.body", "body": b"", "more_body": False})
