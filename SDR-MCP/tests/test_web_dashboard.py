from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from importlib import resources
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from rf_mcp.web import RfWebApp


class FakeCatalog:
    def list_jobs(self, **kwargs):
        return [{"job_id": "job-1", "job_type": "spectrum", "state": "completed",
                 "created_at": "2026-08-11T12:00:00+00:00"}]

    def list_artifacts(self, **kwargs):
        return [{"artifact_id": "art-deadbeef", "filename": "plot.png",
                 "kind": "spectrum_plot", "size_bytes": 1234, "path": "/tmp/plot.png",
                 "mime_type": "image/png"}]

    def storage_status(self):
        return {"total_size_bytes": 1234, "free_bytes": 10_000}

    def list_presets(self, **kwargs):
        return [{"preset_id": "preset-memory", "name": "Hourly stations",
                 "preset_type": "station_memory_scan"}]

    def get_preset(self, value):
        return self.list_presets()[0]

    def list_schedules(self, **kwargs):
        return [{"schedule_id": "schedule-memory", "name": "Hourly",
                 "preset_id": "preset-memory", "preset_name": "Hourly stations",
                 "preset_type": "station_memory_scan", "interval_seconds": 3600,
                 "enabled": True, "next_run_at": "2026-08-11T13:00:00+00:00"}]

    def get_job(self, job_id):
        return {"job_id": job_id, "result": {"completed_at": "2026-08-11T12:00:05+00:00",
            "completed_count": 1, "failed_count": 0, "change_count": 1,
            "changes": [{"kind": "snr_changed", "memory_id": "memory-1", "name": "WWV"}],
            "observations": [{"memory_id": "memory-1", "name": "WWV",
                "frequency_hz": 10_000_000, "mode": "am", "state": "completed",
                "metrics": {"estimated_snr_db": 12.5}}]}}

    def list_alert_events(self, **kwargs):
        return [{"event_id": "alert-1", "event_type": "watchlist",
                 "rule_name": "Signal change", "created_at": "2026-08-11T12:01:00+00:00",
                 "acknowledged": False}]

    def acknowledge_alert_event(self, event_id):
        return {"event_id": event_id, "acknowledged": True}

    def list_fm_stations(self, **kwargs):
        return [{"frequency_hz": 100_100_000, "ps": "TESTFM", "pi_code": "1234",
                 "pty_name": "Rock", "radiotext": "Test Radio", "stereo_detected": True,
                 "estimated_snr_db": 18.5, "rds_group_count": 12,
                 "last_seen_at": "2026-08-11T12:02:00+00:00"}]

    def list_weak_signal_spots(self, **kwargs):
        return [{"mode": "ft8", "dial_frequency_hz": 14_074_000,
                 "rf_frequency_hz": 14_075_500, "callsign": "K1ABC", "grid": "FN31",
                 "snr_db": -12, "message": "CQ K1ABC FN31",
                 "captured_at": "2026-08-11T12:03:00+00:00"}]

    def list_fldigi_decodes(self, **kwargs):
        return [{"mode": "olivia-8-250", "dial_frequency_hz": 14_071_000,
                 "text": "TEST DECODE", "captured_at": "2026-08-11T12:04:00+00:00"}]

    def list_sstv_images(self, **kwargs):
        return [{"image_id": "sstv-test", "job_id": "sstv-job-1",
                 "frequency_hz": 14_230_000, "receiver_mode": "usb",
                 "sstv_mode": "Martin M1", "width": 320, "height": 256,
                 "quality": 0.8, "duplicate_of": None,
                 "image_path": "/tmp/sstv-test.png",
                 "captured_at": "2026-08-11T12:05:00+00:00"}]


async def downstream(scope, receive, send):
    raise AssertionError(f"Unexpected downstream route: {scope['path']}")


async def request(app, path, *, method="GET", headers=None, body=b""):
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "path": path, "method": method,
               "headers": headers or []}, receive, send)
    return messages


def response_body(messages):
    return b"".join(item.get("body", b"") for item in messages[1:])


def response_headers(messages):
    return dict(messages[0]["headers"])


def test_dashboard_assets_are_packaged_and_have_stable_landmarks():
    assets = resources.files("rf_mcp").joinpath("assets")
    document = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")

    assert 'id="captureForm"' in document
    assert 'id="receiverRows"' in document
    assert 'href="/assets/rf-dashboard.css"' in document
    assert 'src="/assets/rf-dashboard.js"' in document
    assert ".dashboard-nav" in stylesheet
    assert "function renderForm" in script
    assert "function renderCard" in script
    assert "function renderTableRow" in script
    assert "function renderStatus" in script
    assert "function renderEmptyState" in script
    assert "uxStyle" not in script


def test_storage_status_component_handles_capacity_states():
    """Storage remains useful for complete, low, incomplete, and zero-sized data."""
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the storage status model")
    model = script[script.index("function storageStatusModel"):
                   script.index("function renderStorageStatus")]
    program = "const STORAGE_WARNING_PERCENT=85,STORAGE_CRITICAL_PERCENT=95;" + model + """
const cases = [
  storageStatusModel({filesystem:{total_bytes:1000,used_bytes:500,free_bytes:500}}),
  storageStatusModel({filesystem:{total_bytes:1000,used_bytes:900,free_bytes:100}}),
  storageStatusModel({filesystem:{total_bytes:1000,used_bytes:960,free_bytes:40}}),
  storageStatusModel({filesystem:{free_bytes:100}}),
  storageStatusModel({filesystem:{total_bytes:0,used_bytes:0,free_bytes:0}}),
];
console.log(JSON.stringify(cases));
"""
    result = subprocess.run(
        [node, "-e", program], check=True, text=True, capture_output=True
    )
    cases = json.loads(result.stdout)
    assert (cases[0]["percent"], cases[0]["tone"]) == (50, "normal")
    assert (cases[1]["percent"], cases[1]["tone"]) == (90, "warning")
    assert (cases[2]["percent"], cases[2]["tone"]) == (96, "critical")
    assert cases[3]["percent"] is None and cases[3]["tone"] == "unknown"
    assert cases[4]["percent"] is None and cases[4]["tone"] == "unknown"

    assert "function renderStorageStatus" in script
    assert "document.createElement('progress')" in script
    assert "captures may fail" in script
    assert "STORAGE_WARNING_PERCENT=85,STORAGE_CRITICAL_PERCENT=95" in script
    assert "JSON.stringify(s,null,2)" not in script


def test_catalog_storage_status_exposes_capacity_and_legacy_filesystem(tmp_path, monkeypatch):
    from rf_mcp.catalog import Catalog
    monkeypatch.setattr(
        "rf_mcp.catalog.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=400, free=600),
    )
    with Catalog(tmp_path) as catalog:
        storage = catalog.storage_status()
    assert storage["total_bytes"] == 1000
    assert storage["used_bytes"] == 400
    assert storage["free_bytes"] == 600
    assert storage["used_percent"] == 40
    assert storage["filesystem"] == {
        "total_bytes": 1000, "used_bytes": 400, "free_bytes": 600,
        "free_percent": 60,
    }


def test_dashboard_preserves_legacy_storage_fields(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    messages = asyncio.run(request(
        RfWebApp(downstream, FakeCatalog(), None, "0.58.0"), "/api/dashboard"
    ))
    storage = json.loads(response_body(messages))["storage"]
    assert storage["total_size_bytes"] == 1234
    assert storage["free_bytes"] == 10_000


def test_dashboard_uses_centralized_semantic_timestamp_formatting():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "new Intl.DateTimeFormat" in script
    assert "new Intl.RelativeTimeFormat" in script
    assert "function renderTimestamp" in script
    assert "function timestampCell" in script
    assert "const date=new Date(value)" in script
    assert "time.dateTime=exact" in script
    assert "time.title=complete" in script
    assert "time.tabIndex=0" in script
    assert "time.setAttribute('aria-label',complete)" in script
    assert "replace('T',' ').slice(0,19)" not in script

    # Every API-backed operational timestamp renderer delegates to the helper.
    for timestamp in (
        "s.observed_at", "s.next_run_at", "e.time", "j.created_at",
        "s.last_seen_at", "s.captured_at", "x.captured_at",
    ):
        assert f"renderTimestamp({timestamp}" in script or f"timestampCell({timestamp}" in script


def test_dashboard_has_one_coordinated_polling_owner():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    # One dashboard request supplies every renderer; no independent summary,
    # memory, operation, or workspace poll is allowed.
    assert script.count("fetch('/api/dashboard'") == 1
    assert "renderDashboardSummary(snapshot)" in script
    assert "renderMemories(snapshot)" in script
    assert "renderDashboardWorkspace(snapshot)" in script
    assert "refreshMemories" not in script
    assert "refreshOperations" not in script
    assert "setInterval(refresh" not in script
    assert "dashboardRefreshTimer=setTimeout(refreshDashboard,delay)" in script


def test_dashboard_coordinator_guards_in_flight_and_stale_requests():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "if(dashboardRefreshInFlight)return dashboardRefreshInFlight" in script
    assert "const requestId=++dashboardRequestId" in script
    assert "if(requestId<lastRenderedDashboardRequestId)return" in script
    assert "lastRenderedDashboardRequestId=requestId" in script
    assert "finally{dashboardRefreshInFlight=null}" in script


def test_dashboard_coordinator_uses_active_and_idle_intervals():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "DASHBOARD_IDLE_INTERVAL_MS=15000" in script
    assert "DASHBOARD_ACTIVE_INTERVAL_MS=3000" in script
    assert "dashboardHasActiveJob(snapshot)?DASHBOARD_ACTIVE_INTERVAL_MS:DASHBOARD_IDLE_INTERVAL_MS" in script
    assert "['queued','running','stopping'].includes(job.state)" in script


def test_dashboard_sync_status_is_accessible_and_visibility_aware():
    assets = resources.files("rf_mcp").joinpath("assets")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")

    assert 'id="syncStatusText" aria-hidden="true">Updated just now' in script
    assert 'id="syncAnnouncement" class="sr-only" role="status" aria-live="polite" aria-atomic="true"' in script
    assert 'id="syncRefresh" type="button">Refresh now' in script
    assert "Date.parse(snapshot.generated_at)" in script
    assert "(stale?'Stale · ':'')+dashboardRelativeAge()" in script
    assert "document.visibilityState==='hidden'" in script
    assert "document.addEventListener('visibilitychange'" in script
    assert "dashboardMissedRefreshes>1?'error':stale?'warning':'current'" in script
    assert '.sync-status[data-state="warning"]' in stylesheet
    assert '.sync-status[data-state="error"]' in stylesheet


def test_dashboard_navigation_is_grouped_and_accessible():
    assets = resources.files("rf_mcp").joinpath("assets")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")

    for category in ("Overview", "Receive", "Analyze", "Automation", "Administration"):
        assert category in script
    assert "aria-label','Dashboard" in script
    assert "aria-controls','dashboardNav" in script
    assert "aria-expanded','false" in script
    assert ".dashboard-nav-toggle{display:none" in stylesheet
    assert "@media(max-width:700px)" in stylesheet
    assert ".dashboard-nav.open{display:grid" in stylesheet


def test_dashboard_navigation_restores_history_and_active_semantics():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "setAttribute('aria-current','page')" in script
    assert "removeAttribute('aria-current')" in script
    assert "document.title=`${viewTitles[id]} · MiniRackDisplay`" in script
    assert "history.pushState({view:id,resultId}" in script
    assert "window.addEventListener('popstate',restoreHashView)" in script
    assert "window.addEventListener('hashchange',restoreHashView)" in script
    assert "showView(view||'home',{updateHistory:false,resultId:" in script
    assert "heading.focus()" in script
    for legacy_hash in ("listen", "scan", "system"):
        assert f"'{legacy_hash}'" in script


def test_dashboard_exposes_accessible_status_and_progress_semantics():
    assets = resources.files("rf_mcp").joinpath("assets")
    document = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")

    assert 'id="error" role="alert"' in document
    assert "receiverBar.setAttribute('role','status')" in script
    assert "region.setAttribute('aria-live','polite')" in script
    assert "status.setAttribute('aria-live','polite')" in script
    assert "document.createElement('progress')" in script
    assert "track.max=100;track.value=job.progress" in script
    assert "progress.max=100;progress.value=pct" in script
    assert "setAttribute('aria-label',jobLabel" in script


def test_dashboard_has_contextual_operation_workspace_landmarks():
    assets = resources.files("rf_mcp").joinpath("assets")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")

    for view in ("scan", "fm", "digital", "sstv"):
        assert f'data-operation-view="{view}"' in script
        assert f'id="{view}WorkspaceProgress"' in script
        assert f'id="{view}WorkspaceResults"' in script
    for landmark in (
        ".operation-workspace",
        ".workspace-progress",
        ".workspace-results",
        ".progress-card",
        ".empty-state",
    ):
        assert landmark in stylesheet


def test_dashboard_javascript_routes_jobs_and_artifacts_to_workspaces():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    # These declarations are the JavaScript routing contract: contextual renderers
    # filter jobs by type and resolve an artifact's originating job through job_id.
    assert "scan:['band_scan','band_survey']" in script
    assert "fm:['fm_broadcast_survey']" in script
    assert "sstv:['sstv_decode','sstv_watch']" in script
    assert "digital:['weak_signal_decode','fldigi_decode','digital_decode']" in script
    assert "jobTypes.includes(j.type)" in script
    assert "artifactsByJob.get(raw.job_id)" in script
    assert "job.artifacts.filter" in script
    assert "renderOperationWorkspace(d,view,jobTypes" in script
    # Home deliberately invokes both renderers without a type filter.
    assert "renderJobProgressCards(d);renderResultCards(d);renderContextualWorkspaces(d)" in script


def test_activity_drawer_has_dialog_and_keyboard_semantics():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert 'aria-expanded="false" aria-controls="activityDrawer"' in script
    assert "activityDrawer.setAttribute('role','dialog')" in script
    assert "activityDrawer.setAttribute('aria-labelledby','activityDrawerTitle')" in script
    assert "activityTrigger.setAttribute('aria-expanded',String(open))" in script
    assert "if(open)activityDrawer.focus();else activityTrigger.focus()" in script
    assert "if(event.key==='Escape')" in script
    assert "setActivityDrawer(false)" in script


def test_dashboard_supports_keyboard_focus_and_reduced_motion():
    stylesheet = resources.files("rf_mcp").joinpath("assets/dashboard.css").read_text(
        encoding="utf-8"
    )

    assert "a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert 'button[aria-busy="true"]::before{animation:none!important}' in stylesheet
    assert ".fmChannel.current{animation:none!important}" in stylesheet
    assert "transition:none!important" in stylesheet


def test_dashboard_without_auth_serves_html_and_json(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    app = RfWebApp(downstream, FakeCatalog(), None, "0.42.0")
    page = asyncio.run(request(app, "/dashboard"))
    data = asyncio.run(request(app, "/api/dashboard"))
    assert page[0]["status"] == 200
    assert b"MiniRackDisplay" in response_body(page)
    assert b"v0.42.0" in response_body(page)
    assert data[0]["status"] == 200
    assert b'"artifact_id":"art-deadbeef"' in response_body(data)
    assert b"content-security-policy" in response_headers(page)
    policy = response_headers(page)[b"content-security-policy"]
    assert b"style-src 'self' 'unsafe-inline'" in policy
    assert b"script-src 'self'" in policy
    assert b'<link rel="stylesheet" href="/assets/rf-dashboard.css">' in response_body(page)
    assert b'<script src="/assets/rf-dashboard.js"></script>' in response_body(page)
    assert b"<script>" not in response_body(page)
    stylesheet = asyncio.run(request(app, "/assets/rf-dashboard.css"))
    script = asyncio.run(request(app, "/assets/rf-dashboard.js"))
    assert stylesheet[0]["status"] == script[0]["status"] == 200
    assert response_headers(stylesheet)[b"content-type"] == b"text/css; charset=utf-8"
    assert b":root{" in response_body(stylesheet)
    assert b"--bg:#07101d" in response_body(stylesheet)
    assert b"color:var(--text)" in response_body(stylesheet)
    assert b"scanReceivers" in response_body(script)
    assert b"Scan for receivers" in response_body(script)
    assert b"/api/receivers/discover" in response_body(script)
    assert b"/api/receivers/register" in response_body(script)


def test_dashboard_guided_receiver_discovery_and_registration(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    discovered = {"device_count": 1, "devices": [{
        "backend": "rtl_sdr", "device_selector": "00000042",
        "display_name": "RTL-SDR · 00000042", "suggested_receiver_id": "rtl-sdr-00000042",
        "suggested_name": "RTL-SDR 00000042", "suggested_role": "vhf_uhf_monitor",
        "verified": True, "already_registered": False,
    }], "diagnostics": [], "writes_registry": False}
    monkeypatch.setattr(sdr_coordinator, "discover_devices", lambda: discovered)
    app = RfWebApp(downstream, FakeCatalog(), None, "0.66.0")
    scan = asyncio.run(request(app, "/api/receivers/discover", method="POST", body=b"{}"))
    assert scan[0]["status"] == 200
    assert b'"device_selector":"00000042"' in response_body(scan)

    monkeypatch.setattr(sdr_coordinator, "register_discovered_device", lambda **values: {
        "registered": True, "receiver": {**values, "verified": True},
    })
    payload = json.dumps({
        "backend": "rtl_sdr", "device_selector": "00000042", "receiver_id": "rtl-vhf",
        "name": "VHF receiver", "role": "vhf_uhf_monitor", "priority": 80,
    }).encode()
    added = asyncio.run(request(app, "/api/receivers/register", method="POST", body=payload))
    assert added[0]["status"] == 200
    assert b'"registered":true' in response_body(added)


def test_dashboard_token_login_cookie_and_logout(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    token = "correct-token-" + "x" * 32
    app = RfWebApp(downstream, FakeCatalog(), token, "0.42.0")
    denied = asyncio.run(request(app, "/dashboard"))
    bad = asyncio.run(request(app, "/dashboard/login", method="POST",
                              body=urlencode({"token": "wrong"}).encode()))
    login = asyncio.run(request(app, "/dashboard/login", method="POST",
                                body=urlencode({"token": token}).encode()))
    assert denied[0]["status"] == 401
    assert bad[0]["status"] == 401
    assert login[0]["status"] == 303
    cookie = response_headers(login)[b"set-cookie"].split(b";", 1)[0]
    page = asyncio.run(request(app, "/dashboard", headers=[(b"cookie", cookie)]))
    data = asyncio.run(request(app, "/api/dashboard", headers=[(b"cookie", cookie)]))
    assert page[0]["status"] == data[0]["status"] == 200
    logout = asyncio.run(request(app, "/dashboard/logout", method="POST",
                                 headers=[(b"cookie", cookie)]))
    assert logout[0]["status"] == 303
    denied_again = asyncio.run(request(app, "/api/dashboard", headers=[(b"cookie", cookie)]))
    assert denied_again[0]["status"] == 401


def test_dashboard_accepts_bearer_header_without_cookie(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    token = "bearer-" + "z" * 32
    app = RfWebApp(downstream, FakeCatalog(), token, "0.42.0")
    messages = asyncio.run(request(
        app, "/api/dashboard",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    ))
    assert messages[0]["status"] == 200


def test_dashboard_exposes_rf_operations_data(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.0")
    messages = asyncio.run(request(app, "/api/dashboard"))
    body = response_body(messages)
    assert messages[0]["status"] == 200
    assert b'"station_scan_profiles"' in body
    assert b'"station_schedules"' in body
    assert b'"station_scan_history"' in body
    assert b'"estimated_snr_db":12.5' in body
    assert b'"recent_alerts"' in body
    page = response_body(asyncio.run(request(app, "/dashboard"))) + response_body(
        asyncio.run(request(app, "/assets/rf-dashboard.js"))
    )
    assert b"RF Operations" in page
    assert b"Memory scan trend" in page
    assert b"Create a memory scan profile" in page
    assert b"Save memory" in page
    assert b"Run scan now" in page
    assert b"Setup progress" in page
    assert b"Save changes" in page
    assert b"Scan &amp; Analyze" not in page  # text is emitted directly in inline HTML
    assert b"Scan & Analyze" in page
    assert b"Saved RF presets" in page
    assert b"FM band survey &amp; station directory" not in page
    assert b"FM band survey & station directory" in page
    assert b"Discovered stations" in page
    assert b"fmDirectoryPlayer" in page
    assert b"fmDirectoryAudio" in page
    assert b"Download WAV" in page
    assert b"Receiving" in page
    assert b"Press Play below to hear the recording" in page
    assert b"receiverBar" in page
    assert b"Receiver activity" in page
    assert b"stopActiveJob" in page
    assert b"Quick Start" in page
    assert b"Favorite stations" in page
    assert b"parseFrequency" in page
    assert b"FREQUENCY_MULTIPLIERS" in page
    assert b"Hz" in page and b"kHz" in page and b"MHz" in page
    assert b"Digital Modes" in page
    assert b"Decode again" in page
    assert b"Try FT8 on 20 m" in page
    assert b"Configure Fldigi decode" in page
    assert b"Band / activity frequency" in page
    assert b"Decoder audio center" in page
    assert b"North America APRS" in page
    assert b"ISS packet" in page
    assert b"Recent weak-signal spots" in page
    assert b"digitalWaterfall" in page
    assert b"spectral contrast" in page
    assert b"Band activity map" in page
    assert b"fmBandMap" in page
    assert b"setButtonBusy" in page
    assert b"aria-busy" in page
    assert b"Live and recent RF jobs" in page
    assert b"Visual results" in page
    assert b"jobProgressCards" in page
    assert b"bandProgressMap" in page
    assert b"renderBandProgress" in page
    assert b"about " in page
    assert b"SSTV receive &amp; gallery" not in page
    assert b"SSTV receive & gallery" in page
    assert b"Decoded image gallery" in page
    assert b"recentAudio" in page


def test_dashboard_exposes_active_rf_job_for_global_status(tmp_path, monkeypatch):
    from rf_mcp import operations, sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    monkeypatch.setattr(operations, "active_long_job", lambda: "survey-active")
    catalog = FakeCatalog()
    catalog.get_job = lambda job_id: {
        "job_id": job_id, "job_type": "band_survey", "state": "running",
        "created_at": "2026-08-12T12:00:00+00:00",
        "config": {"planned_steps": 20},
        "summary": {"completed_steps": 7, "planned_steps": 20},
    }
    messages = asyncio.run(request(
        RfWebApp(downstream, catalog, None, "0.57.0"), "/api/dashboard"
    ))
    payload = json.loads(response_body(messages))
    assert messages[0]["status"] == 200
    assert payload["active_long_job"] == "survey-active"
    assert payload["active_rf_job"]["job_type"] == "band_survey"
    assert payload["active_rf_job"]["summary"]["completed_steps"] == 7


def test_dashboard_station_schedule_operations_validate_and_dispatch():
    calls = []
    create = lambda **kwargs: calls.append(("create", kwargs)) or {"schedule_id": "s-1"}
    run = lambda **kwargs: calls.append(("run", kwargs)) or {"execution_status": "completed"}
    toggle = lambda **kwargs: calls.append(("toggle", kwargs)) or {"enabled": kwargs["enabled"]}
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.0", None, None, None,
                   create, run, toggle)
    created = asyncio.run(request(app, "/api/station-schedules", method="POST",
        body=b'{"name":"Hourly","preset_id_or_name":"preset-memory","interval_seconds":3600,"enabled":true}'))
    ran = asyncio.run(request(app, "/api/station-schedules/run", method="POST",
        body=b'{"schedule_id_or_name":"s-1"}'))
    toggled = asyncio.run(request(app, "/api/station-schedules/toggle", method="POST",
        body=b'{"schedule_id_or_name":"s-1","enabled":false}'))
    bad = asyncio.run(request(app, "/api/station-schedules", method="POST",
        body=b'{"name":"Bad","preset_id_or_name":"preset-memory","interval_seconds":60,"extra":1}'))
    assert created[0]["status"] == ran[0]["status"] == toggled[0]["status"] == 200
    assert bad[0]["status"] == 400
    assert [item[0] for item in calls] == ["create", "run", "toggle"]


def test_dashboard_acknowledges_alert_and_requires_auth():
    token = "token-" + "a" * 32
    app = RfWebApp(downstream, FakeCatalog(), token, "0.50.0")
    denied = asyncio.run(request(app, "/api/alerts/acknowledge", method="POST",
                                 body=b'{"event_id":"alert-1"}'))
    accepted = asyncio.run(request(app, "/api/alerts/acknowledge", method="POST",
        headers=[(b"authorization", f"Bearer {token}".encode())],
        body=b'{"event_id":"alert-1"}'))
    assert denied[0]["status"] == 401
    assert accepted[0]["status"] == 200
    assert b'"acknowledged":true' in response_body(accepted)


def test_dashboard_creates_memory_profile_and_runs_profile():
    calls = []
    make_profile = lambda **kwargs: calls.append(("profile", kwargs)) or {"preset_id": "p-1"}
    run_profile = lambda **kwargs: calls.append(("run", kwargs)) or {"completed_count": 1}
    make_memory = lambda **kwargs: calls.append(("memory", kwargs)) or {"memory_id": "m-1"}
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.3", None, None, None,
                   None, None, None, make_profile, run_profile, make_memory)
    memory = asyncio.run(request(app, "/api/station-memories", method="POST",
        body=b'{"name":"WWV","frequency_hz":10000000,"mode":"am","bandwidth_hz":10000,"enabled":true}'))
    profile = asyncio.run(request(app, "/api/station-scan-profiles", method="POST",
        body=b'{"name":"Time stations","memory_ids_or_names":["m-1"],"duration_seconds":5,"max_memories":1}'))
    run = asyncio.run(request(app, "/api/station-scan-profiles/run", method="POST",
        body=b'{"preset_id_or_name":"p-1"}'))
    assert memory[0]["status"] == profile[0]["status"] == run[0]["status"] == 200
    assert [item[0] for item in calls] == ["memory", "profile", "run"]


def test_dashboard_management_delete_endpoints_dispatch_confirmations():
    calls = []
    def remove(kind):
        return lambda **kwargs: calls.append((kind, kwargs)) or {"deleted": True}
    app = RfWebApp(
        downstream, FakeCatalog(), None, "0.51.0",
        station_memory_delete=remove("memory"),
        scan_profile_delete=remove("profile"), schedule_delete=remove("schedule"),
    )
    cases = [
        ("/api/station-memories/delete",
         b'{"memory_id_or_name":"m-1","confirm_delete":true}'),
        ("/api/station-scan-profiles/delete",
         b'{"preset_id_or_name":"p-1","confirm_delete":true}'),
        ("/api/station-schedules/delete",
         b'{"schedule_id_or_name":"s-1","confirm_delete":true}'),
    ]
    responses = [asyncio.run(request(app, path, method="POST", body=body))
                 for path, body in cases]
    assert all(item[0]["status"] == 200 for item in responses)
    assert [item[0] for item in calls] == ["memory", "profile", "schedule"]


def test_dashboard_band_scan_survey_status_stop_and_preset_endpoints():
    calls = []
    def callback(kind, result):
        return lambda **kwargs: calls.append((kind, kwargs)) or result
    app = RfWebApp(
        downstream, FakeCatalog(), None, "0.52.0",
        band_scan_start=callback("scan", {"job_id": "scan-1"}),
        band_survey_start=callback("survey", {"job_id": "survey-1"}),
        band_job_status=callback("status", {"state": "running"}),
        band_job_stop=callback("stop", {"state": "stopping"}),
        preset_run=callback("preset", {"job_id": "preset-1"}),
    )
    cases = [
        ("/api/band-scan/start", b'{"start_frequency_hz":14000000,"stop_frequency_hz":14350000}'),
        ("/api/band-survey/start", b'{"start_frequency_hz":14000000,"stop_frequency_hz":14350000,"classify_top_signals":5}'),
        ("/api/band-jobs/status", b'{"job_id":"scan-1"}'),
        ("/api/band-jobs/stop", b'{"job_id":"scan-1"}'),
        ("/api/presets/run", b'{"preset_id_or_name":"preset-memory"}'),
    ]
    responses = [asyncio.run(request(app, path, method="POST", body=body))
                 for path, body in cases]
    assert all(item[0]["status"] == 200 for item in responses)
    assert [item[0] for item in calls] == ["scan", "survey", "status", "stop", "preset"]


def test_dashboard_fm_survey_start_status_stop_and_directory():
    calls = []
    def callback(kind, result):
        return lambda **kwargs: calls.append((kind, kwargs)) or result
    app = RfWebApp(
        downstream, FakeCatalog(), None, "0.53.0",
        fm_survey_start=callback("start", {"job_id": "fm-survey-1"}),
        fm_survey_status=callback("status", {"state": "running"}),
        fm_survey_stop=callback("stop", {"state": "stopping"}),
    )
    dashboard = asyncio.run(request(app, "/api/dashboard"))
    cases = [
        ("/api/fm-surveys/start", b'{"start_frequency_hz":87900000,"stop_frequency_hz":107900000,"channel_spacing_hz":200000}'),
        ("/api/fm-surveys/status", b'{"job_id":"fm-survey-1"}'),
        ("/api/fm-surveys/stop", b'{"job_id":"fm-survey-1"}'),
    ]
    responses = [asyncio.run(request(app, path, method="POST", body=body))
                 for path, body in cases]
    assert dashboard[0]["status"] == 200
    assert b'"ps":"TESTFM"' in response_body(dashboard)
    assert all(item[0]["status"] == 200 for item in responses)
    assert [item[0] for item in calls] == ["start", "status", "stop"]


def test_dashboard_digital_decoder_endpoints_and_persisted_results():
    calls = []
    def callback(kind, result):
        return lambda **kwargs: calls.append((kind, kwargs)) or result
    app = RfWebApp(
        downstream, FakeCatalog(), None, "0.54.0",
        digital_decode=callback("native", {"decoder": {"text": "CQ", "confidence": .8}}),
        weak_decode=callback("weak", {"spots": [{"message": "CQ TEST"}]}),
        fldigi_decode=callback("fldigi", {"text": "HELLO"}),
        decoder_capabilities=callback("caps", {"wsjt_x": {"available": True}}),
    )
    dashboard = asyncio.run(request(app, "/api/dashboard"))
    cases = [
        ("/api/digital/native", b'{"frequency_hz":14070000,"mode":"bpsk31","duration_seconds":10}'),
        ("/api/digital/weak", b'{"frequency_hz":14074000,"mode":"ft8","capture_cycles":1}'),
        ("/api/digital/fldigi", b'{"frequency_hz":14071000,"mode":"olivia-8-250","duration_seconds":30}'),
        ("/api/digital/capabilities", b'{}'),
    ]
    responses = [asyncio.run(request(app, path, method="POST", body=body))
                 for path, body in cases]
    assert b'"callsign":"K1ABC"' in response_body(dashboard)
    assert b'"text":"TEST DECODE"' in response_body(dashboard)
    assert all(item[0]["status"] == 200 for item in responses)
    assert [item[0] for item in calls] == ["native", "weak", "fldigi", "caps"]


def test_dashboard_sstv_controls_gallery_and_authenticated_image(tmp_path):
    calls = []
    image_path = tmp_path / "decoded.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nTEST")

    def callback(kind, result):
        return lambda **kwargs: calls.append((kind, kwargs)) or result

    catalog = FakeCatalog()
    catalog.list_sstv_images = lambda **kwargs: [{
        "image_id": "sstv-test", "job_id": "sstv-job-1",
        "frequency_hz": 14_230_000, "receiver_mode": "usb",
        "sstv_mode": "Martin M1", "width": 320, "height": 256,
        "quality": 0.8, "duplicate_of": None, "image_path": str(image_path),
        "captured_at": "2026-08-11T12:05:00+00:00",
    }]
    catalog.get_sstv_image = lambda image_id: catalog.list_sstv_images()[0]
    app = RfWebApp(
        downstream, catalog, None, "0.55.0",
        sstv_decode=callback("decode", {"job_id": "sstv-1"}),
        sstv_watch_start=callback("watch", {"job_id": "watch-1"}),
        sstv_status=callback("status", {"state": "running"}),
        sstv_watch_status=callback("watch-status", {"state": "running"}),
        sstv_stop=callback("stop", {"state": "stopping"}),
        sstv_watch_stop=callback("watch-stop", {"state": "stopping"}),
        sstv_capabilities=callback("caps", {"available": True}),
    )
    dashboard = asyncio.run(request(app, "/api/dashboard"))
    cases = [
        ("/api/sstv/decode", b'{"frequency_hz":14230000,"duration_seconds":130,"receiver_mode":"usb"}'),
        ("/api/sstv/watch", b'{"frequency_hz":145800000,"watch_duration_seconds":3600,"receiver_mode":"nfm"}'),
        ("/api/sstv/status", b'{"job_id":"sstv-1"}'),
        ("/api/sstv/watch-status", b'{"job_id":"watch-1"}'),
        ("/api/sstv/stop", b'{"job_id":"sstv-1"}'),
        ("/api/sstv/watch-stop", b'{"job_id":"watch-1"}'),
        ("/api/sstv/capabilities", b'{}'),
    ]
    responses = [asyncio.run(request(app, path, method="POST", body=body))
                 for path, body in cases]
    image = asyncio.run(request(app, "/sstv-images/sstv-test"))
    token = "sstv-token-" + "x" * 32
    secure_app = RfWebApp(downstream, catalog, token, "0.55.0")
    denied_image = asyncio.run(request(secure_app, "/sstv-images/sstv-test"))
    authorized_image = asyncio.run(request(
        secure_app, "/sstv-images/sstv-test",
        headers=[(b"authorization", f"Bearer {token}".encode())],
    ))
    assert b'"image_url":"/sstv-images/sstv-test"' in response_body(dashboard)
    assert all(item[0]["status"] == 200 for item in responses)
    assert [item[0] for item in calls] == [
        "decode", "watch", "status", "watch-status", "stop", "watch-stop", "caps"
    ]
    assert image[0]["status"] == 200
    assert response_headers(image)[b"content-type"] == b"image/png"
    assert response_body(image).startswith(b"\x89PNG")
    assert denied_image[0]["status"] == 401
    assert authorized_image[0]["status"] == 200


def test_login_rejects_large_request_body():
    token = "token-" + "q" * 32
    app = RfWebApp(downstream, FakeCatalog(), token, "0.42.0")
    messages = asyncio.run(request(app, "/dashboard/login", method="POST", body=b"x" * 9000))
    assert messages[0]["status"] == 413


def test_dashboard_spectrum_capture_forces_safe_options(tmp_path, monkeypatch):
    from rf_mcp import sdr_coordinator
    monkeypatch.setattr(sdr_coordinator, "DATA_DIR", tmp_path)
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structuredContent={
            "job_id": "inspect-test", "center_frequency_hz": kwargs["center_frequency_hz"],
            "duration_seconds": kwargs["duration_seconds"], "relative_noise_floor_db": -41.2,
            "peak_count": 1, "peaks": [{"frequency_hz": 10_001_000}],
        })

    catalog = FakeCatalog()
    catalog.list_artifacts = lambda **kwargs: [{
        "artifact_id": "art-deadbeef", "filename": "spectrum.png",
        "kind": "spectrum_plot", "size_bytes": 123, "mime_type": "image/png",
        "path": "/tmp/spectrum.png",
    }]
    app = RfWebApp(downstream, catalog, None, "0.43.0", capture)
    body = b'{"center_frequency_hz":10000000,"duration_seconds":2,"fft_size":16384}'
    messages = asyncio.run(request(app, "/api/spectrum", method="POST", body=body))
    assert messages[0]["status"] == 200
    assert calls[0]["retain_iq"] is False
    assert calls[0]["include_plot"] is False
    assert b'"download_path":"/artifacts/art-deadbeef"' in response_body(messages)


def test_dashboard_spectrum_replaces_nonfinite_measurements_with_json_null():
    def capture(**kwargs):
        return SimpleNamespace(structuredContent={
            "job_id": "inspect-infinite", "center_frequency_hz": 10_000_000,
            "duration_seconds": 1, "relative_noise_floor_db": -40.0,
            "peak_count": 1, "peaks": [{"frequency_hz": 10_001_000,
                                          "prominence_db": float("inf")}],
        })
    catalog = FakeCatalog()
    catalog.list_artifacts = lambda **kwargs: []
    app = RfWebApp(downstream, catalog, None, "0.50.1", capture)
    messages = asyncio.run(request(app, "/api/spectrum", method="POST",
        body=b'{"center_frequency_hz":10000000,"duration_seconds":1}'))
    payload = json.loads(response_body(messages))
    assert messages[0]["status"] == 200
    assert payload["peaks"][0]["prominence_db"] is None
    assert b"Infinity" not in response_body(messages)


def test_dashboard_spectrum_rejects_unknown_fields_and_excess_duration():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.43.0", lambda **kwargs: None)
    unknown = asyncio.run(request(app, "/api/spectrum", method="POST",
                                  body=b'{"center_frequency_hz":10000000,"retain_iq":true}'))
    long = asyncio.run(request(app, "/api/spectrum", method="POST",
                               body=b'{"center_frequency_hz":10000000,"duration_seconds":11}'))
    assert unknown[0]["status"] == 400
    assert long[0]["status"] == 400


def test_dashboard_demodulation_returns_audio_and_forces_safe_options():
    calls = []

    def analyze(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structuredContent={
            "job_id": "analyze-test", "requested_frequency_hz": kwargs["frequency_hz"],
            "mode": kwargs["mode"], "bandwidth_hz": kwargs["bandwidth_hz"],
            "duration_seconds": kwargs["duration_seconds"],
            "metrics": {"estimated_snr_db": 12.3, "signal_present": True},
        })

    catalog = FakeCatalog()
    catalog.list_artifacts = lambda **kwargs: [
        {"artifact_id": "art-audio", "filename": "audio.wav", "kind": "audio_wav",
         "size_bytes": 100, "mime_type": "audio/wav", "path": "/tmp/audio.wav"},
        {"artifact_id": "art-rf", "filename": "rf.png", "kind": "rf_spectrum_plot",
         "size_bytes": 100, "mime_type": "image/png", "path": "/tmp/rf.png"},
        {"artifact_id": "art-audio-plot", "filename": "audio.png",
         "kind": "audio_spectrum_plot", "size_bytes": 100,
         "mime_type": "image/png", "path": "/tmp/audio.png"},
    ]
    app = RfWebApp(downstream, catalog, None, "0.44.0", None, analyze)
    body = b'{"frequency_hz":7100000,"mode":"lsb","bandwidth_hz":3000,"duration_seconds":5}'
    messages = asyncio.run(request(app, "/api/demodulate", method="POST", body=body))
    assert messages[0]["status"] == 200
    assert calls[0]["retain_iq"] is False
    assert calls[0]["include_audio"] is False
    assert calls[0]["include_plots"] is False
    assert b'"download_path":"/artifacts/art-audio"' in response_body(messages)


def test_dashboard_demodulation_validates_mode_bandwidth_and_fields():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.44.0", None,
                   lambda **kwargs: None)
    bad_mode = asyncio.run(request(app, "/api/demodulate", method="POST",
                                    body=b'{"frequency_hz":10000000,"mode":"wfm"}'))
    bad_bandwidth = asyncio.run(request(app, "/api/demodulate", method="POST",
                                         body=b'{"frequency_hz":10000000,"mode":"cw","bandwidth_hz":5000}'))
    retained = asyncio.run(request(app, "/api/demodulate", method="POST",
                                    body=b'{"frequency_hz":10000000,"retain_iq":true}'))
    assert bad_mode[0]["status"] == 400
    assert bad_bandwidth[0]["status"] == 400
    assert retained[0]["status"] == 400


def test_dashboard_broadcast_fm_returns_audio_plot_and_rds():
    calls = []

    def receive_fm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structuredContent={
            "job_id": "wfm-test", "requested_frequency_hz": kwargs["frequency_hz"],
            "duration_seconds": kwargs["duration_seconds"],
            "metrics": {"audio_channels": 2, "stereo_detected": True},
            "rds": {"group_count": 4, "station": {"program_service": "TESTFM"}},
        })

    catalog = FakeCatalog()
    catalog.list_artifacts = lambda **kwargs: [
        {"artifact_id": "art-fm-audio", "filename": "fm.wav",
         "kind": "broadcast_fm_audio", "size_bytes": 100,
         "mime_type": "audio/wav", "path": "/tmp/fm.wav"},
        {"artifact_id": "art-fm-plot", "filename": "fm.png",
         "kind": "broadcast_fm_multiplex_plot", "size_bytes": 100,
         "mime_type": "image/png", "path": "/tmp/fm.png"},
    ]
    app = RfWebApp(downstream, catalog, None, "0.45.0", None, None, receive_fm)
    body = b'{"frequency_hz":100100000,"duration_seconds":10,"stereo":true,"deemphasis_us":75,"decode_rds_data":true}'
    messages = asyncio.run(request(app, "/api/broadcast-fm", method="POST", body=body))
    assert messages[0]["status"] == 200
    assert calls[0]["retain_iq"] is False
    assert calls[0]["include_audio"] is False
    assert calls[0]["include_plot"] is False
    assert b'"program_service":"TESTFM"' in response_body(messages)
    assert b'"download_path":"/artifacts/art-fm-audio"' in response_body(messages)


def test_dashboard_broadcast_fm_validates_band_duration_and_booleans():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.45.0", None, None,
                   lambda **kwargs: None)
    outside = asyncio.run(request(app, "/api/broadcast-fm", method="POST",
                                   body=b'{"frequency_hz":10700000}'))
    duration = asyncio.run(request(app, "/api/broadcast-fm", method="POST",
                                    body=b'{"frequency_hz":100100000,"duration_seconds":6}'))
    boolean = asyncio.run(request(app, "/api/broadcast-fm", method="POST",
                                   body=b'{"frequency_hz":100100000,"stereo":"yes"}'))
    assert outside[0]["status"] == 400
    assert duration[0]["status"] == 400
    assert boolean[0]["status"] == 400


def test_dashboard_responsive_markup_primitives_and_mobile_tables():
    assets = resources.files("rf_mcp").joinpath("assets")
    document = assets.joinpath("dashboard.html").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")

    assert 'class="field"' in document
    assert 'class="action-row"' in document
    assert 'class="table-container" role="region" aria-label="Recent RF jobs table"' in document
    assert 'class="responsive-table"' in document
    for component in (".field{", ".field-help{", ".action-row{", ".table-container{"):
        assert component in stylesheet
    assert "min-height:44px" in stylesheet
    assert "@media(max-width:850px)" in stylesheet
    assert "@media(max-width:450px)" in stylesheet
    assert ".responsive-table.mobile-cards td::before{content:attr(data-label)" in stylesheet
    assert "labelTableRow(renderTableRow(cells(item)),body)" in script
    assert "container.setAttribute('aria-label'" in script
    for table_body in ("memoryRows", "scheduleRows", "bandJobRows", "artifactRows", "jobRows", "fmStationRows", "sstvJobRows"):
        assert f"'{table_body}'" in script
    assert "b.classList.add('destructive')" in script


def test_dashboard_frequency_helpers_convert_validate_and_format_boundaries():
    """The browser helpers preserve integer-Hz API precision for every display unit."""
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard JavaScript helpers")

    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )
    helpers = script[script.index("const FREQUENCY_MULTIPLIERS"):
                     script.index("const frequencyFields")]
    program = helpers + """
const cases = [
  parseFrequencyInput('9000', 'hz'),
  parseFrequencyInput('9', 'khz'),
  parseFrequencyInput('0.009', 'mhz'),
  parseFrequencyInput('100.1', 'mhz'),
  parseFrequencyInput('14.074001', 'mhz'),
  parseFrequencyInput('260', 'mhz'),
  formatFrequency(100100000),
  formatFrequency(14074001),
];
const invalid = [];
for (const args of [['8.999','khz'], ['260.000001','mhz'], ['100.1234567','mhz'], ['oops','mhz']]) {
  try { parseFrequencyInput(...args); invalid.push(false) } catch (_) { invalid.push(true) }
}
console.log(JSON.stringify({cases, invalid}));
"""
    result = subprocess.run(
        [node, "-e", program], check=True, text=True, capture_output=True
    )
    assert json.loads(result.stdout) == {
        "cases": [9000, 9000, 9000, 100100000, 14074001, 260000000,
                  "100.1 MHz", "14.074001 MHz"],
        "invalid": [True, True, True, True],
    }


def test_dashboard_frequency_controls_share_memory_recall_and_integer_hz_payloads():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "function parseFrequencyInput" in script
    assert "function formatFrequency" in script
    assert "selectedFrequencyHz=m.frequency_hz" in script
    assert "['frequency','audioFrequency','fmFrequency','digitalFrequency','sstvFrequency','bandStart']" in script
    assert "frequency_hz:frequencyValue('fmFrequency')" in script
    assert "center_frequency_hz:frequencyValue('frequency')" in script
    assert "start_frequency_hz:frequencyValue('bandStart')" in script
    assert "Stop frequency must be greater than start frequency" in script
    assert "outside the selected receiver’s supported range" in script


def test_dashboard_frequency_enhancement_allows_initially_empty_forms():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(
        encoding="utf-8"
    )

    # The create-memory frequency is intentionally empty until the user enters
    # one. Enhancing that input must not abort the rest of dashboard startup.
    assert "initialValue=input.value.trim()" in script
    assert "initialHz=initialValue===''?null:parseFrequencyInput" in script
    assert "if(initialHz!==null)input.value=formatFrequency" in script


def test_dashboard_uses_accessible_confirmation_dialog_for_consequential_actions():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.0")
    script = response_body(asyncio.run(request(app, "/assets/rf-dashboard.js")))
    css = response_body(asyncio.run(request(app, "/assets/rf-dashboard.css")))

    assert b"confirm(" not in script
    assert b"requestConfirmation" in script
    assert b"confirmAndRun" in script
    assert b"aria-labelledby" in script and b"aria-describedby" in script
    assert b"Affected: " in script
    assert b"event.key!=='Tab'" in script
    assert b"addEventListener('cancel'" in script
    assert b"confirmationOpener.focus()" in script
    assert b"confirmationPending" in script
    assert b"Destructive action" in css and b"border-style:double" in css


def test_confirmation_cancel_and_confirm_paths_are_separate():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.0")
    script = response_body(asyncio.run(request(app, "/assets/rf-dashboard.js")))

    assert b"confirmationCancel.addEventListener('click',()=>closeConfirmation(false))" in script
    assert b"if(!await requestConfirmation(options))return false" in script
    assert b"confirmationConfirm.addEventListener('click'" in script
    assert b"await action();return true" in script


def test_confirmed_actions_preserve_exact_api_requests():
    app = RfWebApp(downstream, FakeCatalog(), None, "0.50.0")
    script = response_body(asyncio.run(request(app, "/assets/rf-dashboard.js")))

    expected_calls = [
        b"postOperation('/api/station-memories/delete',{memory_id_or_name:m.memory_id,confirm_delete:true})",
        b"postOperation('/api/station-scan-profiles/delete',{preset_id_or_name:p.preset_id,confirm_delete:true})",
        b"postOperation('/api/station-schedules/delete',{schedule_id_or_name:s.schedule_id,confirm_delete:true})",
        b"postOperation('/api/band-jobs/stop',{job_id:j.job_id})",
        b"postOperation('/api/fm-surveys/stop',{job_id:j.job_id})",
        b"postOperation(watch?'/api/sstv/watch-stop':'/api/sstv/stop',{job_id:j.job_id})",
    ]
    for call in expected_calls:
        assert call in script

    # Low-severity acknowledgement remains immediate and idempotent.
    assert b"postOperation('/api/alerts/acknowledge',{event_id:e.alert.event_id})" in script


def test_dashboard_uses_one_accessible_job_presentation_lifecycle():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(encoding="utf-8")

    assert "function dashboardJobs(d)" in script
    for field in ("id:raw.job_id", "label:jobLabels", "view:jobViews", "state,phase:",
                  "progress,timing:", "stoppable:", "error:raw.error", "artifacts:artifactsByJob"):
        assert field in script
    assert "activeJobStates=new Set(['queued','running','stopping'])" in script
    assert "terminalJobStates=new Set(['completed','failed','stopped','cancelled','interrupted'])" in script
    for label in ("'Status'", "'Stop'", "'Resume'", "'Open'", "'View results'"):
        assert label in script
    assert "dashboardJobs(d).filter" in script


def test_job_start_and_stop_status_are_accessible_and_persistent():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(encoding="utf-8")

    assert "jobAnnouncement.setAttribute('aria-live','polite')" in script
    assert "Job ID ${id}" in script
    assert "announceStarted('scan',result" in script
    assert "announceStarted('fm',result" in script
    assert "announceStarted('sstv',result" in script
    assert "workspace.scrollIntoView({block:'start'})" in script
    assert "pendingStops.add(job.job_id)" in script
    assert "pendingStops.has(raw.job_id)&&!terminalJobStates.has(raw.state)?'stopping'" in script
    assert "if(terminalJobStates.has(raw.state))pendingStops.delete(raw.job_id)" in script


def test_completed_jobs_navigate_to_their_originating_result_card():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(encoding="utf-8")

    assert "showView(job.view,{resultId:job.id})" in script
    assert "card.dataset.resultJob=job.id" in script
    assert "?result='+encodeURIComponent(resultId)" in script
    assert "new URLSearchParams(query).get('result')" in script
    assert "CSS.escape(resultId)" in script
    assert "result.scrollIntoView({block:'center'})" in script
    assert "result.focus()" in script
    assert "history.pushState({view:id,resultId}" in script


def test_receiver_owner_disables_duplicate_starts_with_local_context():
    script = resources.files("rf_mcp").joinpath("assets/dashboard.js").read_text(encoding="utf-8")

    assert "#bandStartButton,#fmSurveyButton,#sstvStart" in script
    assert "button.disabled=Boolean(active)" in script
    assert "Receiver occupied by ${owner}" in script
    assert "workspace.querySelector('.receiver-owner')" in script


def test_dashboard_image_viewer_is_accessible_and_used_for_every_image_path():
    assets = resources.files("rf_mcp").joinpath("assets")
    script = assets.joinpath("dashboard.js").read_text(encoding="utf-8")
    stylesheet = assets.joinpath("dashboard.css").read_text(encoding="utf-8")

    assert "imageViewer.setAttribute('aria-labelledby','imageViewerTitle')" in script
    assert "imageViewer.setAttribute('aria-describedby','imageViewerCaption')" in script
    assert 'role="toolbar" aria-label="Image zoom controls"' in script
    for control in ("imageViewerZoomIn", "imageViewerZoomOut", "imageViewerReset",
                    "imageViewerClose", "imageViewerDownload"):
        assert control in script
    assert "imageViewer.addEventListener('cancel'" in script
    assert "event.key==='Escape'" in script
    assert "imageViewerOpener?.isConnected" in script
    assert "imageViewerOpener.focus()" in script
    assert "event.key!=='Tab'" in script
    assert "event.target===imageViewer" in script
    assert "imageViewerDownload.href=downloadUrl||image.src" in script
    assert "download.href=downloadUrl||image.src||'#'" in script
    assert "View full size" in script

    for image_id in ("spectrumPlot", "rfAnalysisPlot", "audioAnalysisPlot", "fmPlot",
                     "digitalWaterfall"):
        assert f"'{image_id}'" in script
    assert "makeZoomableImage(img,{title,caption:" in script
    assert "makeZoomableImage(image,{title:imageName,caption:" in script
    assert "a.filename" in script
    assert "x.sstv_mode" in script
    assert "x.frequency_hz" in script

    for selector in (".image-viewer::backdrop", ".image-viewer-viewport",
                     ".image-viewer-image", ".image-viewer-caption",
                     ".image-viewer-toolbar", ".image-thumbnail-actions"):
        assert selector in stylesheet
    assert "overflow:auto;overscroll-behavior:contain" in stylesheet
    assert "@media(forced-colors:active)" in stylesheet
    assert "@media(prefers-reduced-motion:reduce)" in stylesheet
    assert "target='_blank'" not in script
    assert 'target="_blank"' not in script
