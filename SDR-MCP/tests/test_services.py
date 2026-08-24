from __future__ import annotations

from rf_mcp.services import ReceiverService, RfApplicationServices


class FakeCatalog:
    def schema_status(self):
        return {"current_version": 1, "supported_version": 1, "up_to_date": True}


class FakeReceivers:
    def status(self):
        return {"receiver_count": 2, "leases_are_process_local": False}


def noop(**values):
    return values


def test_application_service_recovery_status_is_framework_independent():
    services = RfApplicationServices(
        catalog=FakeCatalog(), receivers=FakeReceivers(), spectrum_capture=noop,
        signal_analyzer=noop, broadcast_fm_receiver=noop,
    )
    status = services.recovery_status(3)
    assert status["catalog_schema"]["up_to_date"] is True
    assert status["interrupted_jobs_recovered_on_startup"] == 3
    assert status["receiver_coordination"]["receiver_count"] == 2


def test_web_adapter_uses_shared_service_callbacks():
    from rf_mcp.web import RfWebApp

    services = RfApplicationServices(
        catalog=FakeCatalog(), receivers=FakeReceivers(), spectrum_capture=lambda: "spectrum",
        signal_analyzer=lambda: "analysis", broadcast_fm_receiver=lambda: "fm",
    )

    async def downstream(scope, receive, send):
        raise AssertionError("not called")

    app = RfWebApp(downstream, FakeCatalog(), None, "0.68.0", services=services)
    assert app.spectrum_capture is services.spectrum_capture
    assert app.signal_analyzer is services.signal_analyzer
    assert app.broadcast_fm_receiver is services.broadcast_fm_receiver


def test_receiver_service_delegates_to_application_boundary(monkeypatch):
    from rf_mcp import services as services_module

    monkeypatch.setattr(services_module, "discover_devices", lambda: {"device_count": 1})
    monkeypatch.setattr(
        services_module, "register_discovered_device",
        lambda **values: {"registered": True, "receiver": values},
    )
    service = ReceiverService()
    assert service.discover()["device_count"] == 1
    assert service.register(receiver_id="rtl-vhf")["receiver"]["receiver_id"] == "rtl-vhf"
