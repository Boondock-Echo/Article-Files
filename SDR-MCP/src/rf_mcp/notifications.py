from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_DELIVERY_ATTEMPTS = 5
RETRY_DELAYS_SECONDS = (5, 30, 120, 600, 1800)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def validate_webhook_url(url: str, *, resolve_host: bool = True) -> str:
    url = str(url).strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("webhook URL must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("webhook URL cannot contain credentials or a fragment")
    if parsed.scheme != "https" and not _env_true("RF_MCP_ALLOW_INSECURE_WEBHOOKS"):
        raise ValueError(
            "HTTP webhooks are disabled; use HTTPS or set "
            "RF_MCP_ALLOW_INSECURE_WEBHOOKS=true"
        )
    if resolve_host and not _env_true("RF_MCP_ALLOW_PRIVATE_WEBHOOKS"):
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
            }
        except socket.gaierror as exc:
            raise ValueError(f"webhook hostname cannot be resolved: {parsed.hostname}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError(
                    "private, loopback, link-local, and reserved webhook targets are disabled; "
                    "set RF_MCP_ALLOW_PRIVATE_WEBHOOKS=true for a trusted LAN target"
                )
    return url


def normalize_webhook_destination(
    *,
    name: str,
    url: str,
    signing_secret: str | None,
    enabled: bool,
    resolve_host: bool = True,
) -> dict:
    name = str(name).strip()
    if not 1 <= len(name) <= 64 or any(ord(char) < 32 for char in name):
        raise ValueError("destination name must contain 1 through 64 printable characters")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a JSON boolean")
    if signing_secret is not None:
        signing_secret = str(signing_secret)
        if not 16 <= len(signing_secret) <= 256:
            raise ValueError("signing_secret must contain 16 through 256 characters")
    return {
        "name": name,
        "url": validate_webhook_url(url, resolve_host=resolve_host),
        "signing_secret": signing_secret,
        "enabled": enabled,
    }


def signed_headers(body: bytes, secret: str | None, timestamp: str, event_id: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SDR-MCP-webhook/0.50",
        "X-RF-MCP-Event": event_id,
        "X-RF-MCP-Timestamp": timestamp,
    }
    if secret:
        digest = hmac.new(
            secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256
        ).hexdigest()
        headers["X-RF-MCP-Signature-256"] = f"sha256={digest}"
    return headers


class WebhookDispatcher:
    def __init__(self, catalog, *, poll_seconds: float = 2.0, timeout_seconds: float = 5.0):
        self.catalog = catalog
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 15.0))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_loop_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="SDR-MCP-webhooks", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_seconds + 1)

    def status(self) -> dict:
        counts = self.catalog.webhook_delivery_counts()
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "poll_seconds": self.poll_seconds,
            "timeout_seconds": self.timeout_seconds,
            "last_loop_error": self._last_loop_error,
            "delivery_counts": counts,
        }

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
                self._last_loop_error = None
            except Exception as exc:
                self._last_loop_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.wait(self.poll_seconds)

    def tick(self, now: datetime | None = None) -> list[dict]:
        now = (now or utc_now()).astimezone(timezone.utc)
        return [self._deliver(item, now) for item in self.catalog.due_webhook_deliveries(now.isoformat())]

    def _deliver(self, delivery: dict, now: datetime) -> dict:
        attempt = delivery["attempt_count"] + 1
        body = json.dumps(delivery["payload"], separators=(",", ":"), sort_keys=True).encode()
        timestamp = now.isoformat()
        headers = signed_headers(
            body, delivery.get("signing_secret"), timestamp, delivery["event_id"]
        )
        status = None
        error = None
        try:
            validate_webhook_url(delivery["destination_url"])
            request = Request(delivery["destination_url"], data=body, headers=headers, method="POST")
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout_seconds
            ) as response:  # noqa: S310
                status = response.status
            if 200 <= status < 300:
                return self.catalog.record_webhook_delivery_attempt(
                    delivery["delivery_id"], state="delivered", attempt_count=attempt,
                    http_status=status, error=None, next_attempt_at=None, delivered_at=timestamp
                )
            error = f"HTTP {status}"
        except HTTPError as exc:
            status = exc.code
            error = f"HTTP {exc.code}"
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        retryable = status is None or status in {408, 425, 429} or status >= 500
        if retryable and attempt < MAX_DELIVERY_ATTEMPTS:
            next_at = now + timedelta(seconds=RETRY_DELAYS_SECONDS[attempt - 1])
            state = "retrying"
        else:
            next_at = None
            state = "failed"
        return self.catalog.record_webhook_delivery_attempt(
            delivery["delivery_id"], state=state, attempt_count=attempt,
            http_status=status, error=error, next_attempt_at=next_at.isoformat() if next_at else None,
            delivered_at=None
        )
