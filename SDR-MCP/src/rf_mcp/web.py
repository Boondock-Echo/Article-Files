from __future__ import annotations

import asyncio
import hmac
import html
from importlib import resources
import json
import math
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
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
        self.live_audio = services.live_audio if services is not None else None
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
                        "service": "SDR-MCP",
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

        if path in {"/api/dashboard/jobs", "/api/dashboard/artifacts"}:
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._dashboard_history(scope, send, path.rsplit("/", 1)[-1])
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

        if path == "/api/live-audio" and scope.get("method") == "GET":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}),
                                extra_headers=[(b"www-authenticate", b"Bearer")])
            else:
                await self._live_audio_stream(scope, receive, send)
            return

        if path == "/api/live-audio/status" and scope.get("method") == "GET":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}))
            elif self.live_audio is None:
                await _response(send, 503, _json_bytes({"error": "live_audio_unavailable"}))
            else:
                await _response(send, 200, _json_bytes(self.live_audio.status()))
            return

        if path == "/api/live-audio/stop" and scope.get("method") == "POST":
            if not self._dashboard_authorized(scope):
                await _response(send, 401, _json_bytes({"error": "unauthorized"}))
            elif self.live_audio is None:
                await _response(send, 503, _json_bytes({"error": "live_audio_unavailable"}))
            else:
                body = await self._read_json(receive)
                await _response(send, 200, _json_bytes(self.live_audio.stop(body.get("session_id"))))
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

    async def _live_audio_stream(self, scope: dict, receive: Callable, send: Callable) -> None:
        from .live_audio import LiveAudioConfig
        if self.live_audio is None:
            await _response(send, 503, _json_bytes({"error": "live_audio_unavailable"}))
            return
        try:
            query = parse_qs(scope.get("query_string", b"").decode("ascii"), strict_parsing=True)
            required = lambda key: query[key][0]
            settings = LiveAudioConfig(
                frequency_hz=int(required("frequency_hz")), mode=required("mode"),
                bandwidth_hz=int(required("bandwidth_hz")),
                receiver_id=query.get("receiver_id", [None])[0] or None,
                deemphasis_us=int(query.get("deemphasis_us", ["75"])[0]),
                maximum_duration_seconds=float(query.get("maximum_duration_seconds", ["300"])[0]))
            subscription = self.live_audio.subscribe(settings)
        except (KeyError, ValueError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_live_audio_request", "detail": str(exc)}))
            return
        except RuntimeError as exc:
            message = str(exc)
            status = 409 if "busy" in message else 503
            await _response(send, status, _json_bytes({"error": "receiver_busy" if status == 409 else "live_audio_unavailable", "detail": message}))
            return
        await send({"type": "http.response.start", "status": 200, "headers": [
            (b"content-type", b"audio/ogg; codecs=opus"), (b"cache-control", b"no-store"),
            (b"x-content-type-options", b"nosniff"), (b"x-accel-buffering", b"no")]})
        try:
            while True:
                frame_task = asyncio.create_task(asyncio.to_thread(subscription.frames.get))
                disconnect_task = asyncio.create_task(receive())
                done, pending = await asyncio.wait((frame_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()
                if disconnect_task in done and disconnect_task.result().get("type") == "http.disconnect": break
                if frame_task not in done: continue
                chunk = frame_task.result()
                if chunk is None: break
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            subscription.close()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

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

    async def _dashboard_history(self, scope: dict, send: Callable, resource: str) -> None:
        """Return one validated, bounded page of jobs or artifacts."""
        try:
            raw = parse_qs(scope.get("query_string", b"").decode(), keep_blank_values=True)
            def one(name: str, default: str = "") -> str:
                return raw.get(name, [default])[-1].strip()
            limit_text, cursor_text = one("limit", "20"), one("cursor", "0")
            if not limit_text.isdigit() or not 1 <= int(limit_text) <= 100:
                raise ValueError("limit must be an integer from 1 to 100")
            if not cursor_text.isdigit():
                raise ValueError("cursor must be a non-negative integer")
            limit, offset = int(limit_text), int(cursor_text)
            if resource == "jobs":
                job_type, state = one("job_type"), one("state")
                if not all(re.fullmatch(r"[A-Za-z0-9_-]{0,64}", value)
                           for value in (job_type, state)):
                    raise ValueError("job_type and state contain invalid characters")
                time_range = one("time_range", "all")
                hours = {"all": None, "1h": 1, "24h": 24, "7d": 168, "30d": 720}
                if time_range not in hours:
                    raise ValueError("time_range must be all, 1h, 24h, 7d, or 30d")
                after = ((datetime.now(timezone.utc) - timedelta(hours=hours[time_range])).isoformat()
                         if hours[time_range] else None)
                items = self.catalog.list_jobs(job_type=job_type or None, state=state or None,
                    created_after=after, search=one("q") or None, offset=offset, limit=limit + 1)
            else:
                kind = one("kind")
                if not re.fullmatch(r"[A-Za-z0-9_-]{0,64}", kind):
                    raise ValueError("kind contains invalid characters")
                items = self.catalog.list_artifacts(kind=kind or None,
                    filename=one("filename") or None, offset=offset, limit=limit + 1)
                items = [{**item, "download_path": f"/artifacts/{item['artifact_id']}"}
                         for item in items]
            page = items[:limit]
            await _response(send, 200, _json_bytes({"items": page, "count": len(page),
                "has_more": len(items) > limit,
                "next_cursor": str(offset + limit) if len(items) > limit else None}))
        except (ValueError, UnicodeDecodeError) as exc:
            await _response(send, 400, _json_bytes({"error": "invalid_parameters",
                                                    "detail": str(exc)}))

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
        """Load the dashboard files from package resources for wheel/zip installs."""
        if self._dashboard_assets is None:
            assets = resources.files("rf_mcp").joinpath("assets")
            document = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
            document = document.replace("__RF_MCP_VERSION__", html.escape(self.version))
            self._dashboard_assets = (
                document.encode("utf-8"),
                assets.joinpath("dashboard.css").read_bytes(),
                assets.joinpath("dashboard.js").read_bytes(),
            )
        return self._dashboard_assets

    def _dashboard_html(self) -> bytes:
        return self._frontend_assets()[0]

    def _dashboard_source_html(self) -> bytes:
        """Return the packaged dashboard document (kept for caller compatibility)."""
        return self._dashboard_html()

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
