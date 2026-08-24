from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from .config import PLOT_DIR, ensure_data_dirs


SPEED_OF_LIGHT_M_S = 299_792_458.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _skyfield():
    try:
        from skyfield.api import EarthSatellite, load, wgs84
    except ImportError as exc:
        raise RuntimeError(
            "Satellite prediction requires Skyfield; install python3-skyfield"
        ) from exc
    return EarthSatellite, load, wgs84


def _name(value: str) -> str:
    value = str(value).strip()
    if not 1 <= len(value) <= 64 or re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError("satellite watch name must contain 1 through 64 printable characters")
    return value


def parse_coordinate(value: str | float | int, *, axis: str) -> float:
    """Parse decimal degrees or a compact DMS coordinate with N/S/E/W."""
    axis = str(axis).lower()
    if axis not in {"latitude", "longitude"}:
        raise ValueError("axis must be latitude or longitude")
    text = str(value).strip().upper().replace("º", "°")
    direction = text[-1] if text and text[-1] in "NSEW" else None
    if direction:
        text = text[:-1].strip()
    cleaned = re.sub(r"[°'\"′″,:]", " ", text)
    parts = [part for part in cleaned.split() if part]
    try:
        if len(parts) == 1:
            result = float(parts[0])
        elif 2 <= len(parts) <= 3:
            degrees, minutes = float(parts[0]), float(parts[1])
            seconds = float(parts[2]) if len(parts) == 3 else 0.0
            if not 0 <= minutes < 60 or not 0 <= seconds < 60:
                raise ValueError
            sign = -1 if degrees < 0 else 1
            result = sign * (abs(degrees) + minutes / 60 + seconds / 3600)
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"{axis} must be decimal degrees or DMS, for example 33.96 or 33 57 36 N"
        ) from exc
    if direction:
        if axis == "latitude" and direction not in "NS":
            raise ValueError("latitude direction must be N or S")
        if axis == "longitude" and direction not in "EW":
            raise ValueError("longitude direction must be E or W")
        result = abs(result) * (-1 if direction in "SW" else 1)
    limit = 90 if axis == "latitude" else 180
    if not -limit <= result <= limit:
        raise ValueError(f"{axis} must be from {-limit} through {limit} degrees")
    return result


DOWNLINK_MODES = {
    "sstv", "nfm_audio", "ax25_afsk1200", "ax25_g3ruh9600", "capture_only",
}


def normalize_satellite_downlinks(downlinks: list[dict]) -> list[dict]:
    if not isinstance(downlinks, list) or not 1 <= len(downlinks) <= 16:
        raise ValueError("downlinks must contain 1 through 16 objects")
    normalized, identifiers = [], set()
    for index, item in enumerate(downlinks):
        if not isinstance(item, dict):
            raise ValueError("each downlink must be an object")
        identifier = str(item.get("downlink_id") or f"downlink-{index + 1}").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", identifier):
            raise ValueError("downlink_id must contain 1–32 lowercase letters, numbers, _ or -")
        if identifier in identifiers:
            raise ValueError(f"duplicate downlink_id: {identifier}")
        identifiers.add(identifier)
        label = str(item.get("label") or identifier).strip()
        if not 1 <= len(label) <= 64:
            raise ValueError("downlink label must contain 1 through 64 characters")
        frequency_hz = int(item["frequency_hz"])
        if not (9_000 <= frequency_hz <= 31_000_000 or 60_000_000 <= frequency_hz <= 260_000_000):
            raise ValueError("downlink frequency is outside the Airspy HF+ tuning ranges")
        mode = str(item.get("mode", "capture_only")).strip().lower()
        if mode not in DOWNLINK_MODES:
            raise ValueError("downlink mode must be one of: " + ", ".join(sorted(DOWNLINK_MODES)))
        receiver_mode = str(item.get("receiver_mode", "nfm")).strip().lower()
        if receiver_mode not in {"nfm", "usb"}:
            raise ValueError("downlink receiver_mode must be nfm or usb")
        if mode in {"nfm_audio", "ax25_afsk1200", "ax25_g3ruh9600"} and receiver_mode != "nfm":
            raise ValueError(f"{mode} requires receiver_mode='nfm'")
        priority = int(item.get("priority", index + 1))
        if not 1 <= priority <= 1000:
            raise ValueError("downlink priority must be from 1 through 1000")
        enabled = item.get("enabled", True)
        retain_audio = item.get("retain_audio", mode in {"sstv", "nfm_audio"})
        if not isinstance(enabled, bool) or not isinstance(retain_audio, bool):
            raise ValueError("downlink enabled and retain_audio must be JSON booleans")
        normalized.append({
            "downlink_id": identifier, "label": label, "frequency_hz": frequency_hz,
            "mode": mode, "receiver_mode": receiver_mode, "priority": priority,
            "enabled": enabled, "retain_audio": retain_audio,
        })
    if not any(item["enabled"] for item in normalized):
        raise ValueError("at least one downlink must be enabled")
    return sorted(normalized, key=lambda item: (item["priority"], item["downlink_id"]))


def tle_checksum_valid(line: str) -> bool:
    line = str(line).rstrip("\r\n")
    if len(line) != 69 or not line[-1].isdigit():
        return False
    checksum = sum(int(char) for char in line[:68] if char.isdigit())
    checksum += line[:68].count("-")
    return checksum % 10 == int(line[-1])


def parse_tle_response(text: str, *, norad_id: int) -> tuple[str, str, str | None]:
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)
    name = next((line.strip() for line in lines if not line.startswith(("1 ", "2 "))), None)
    if line1 is None or line2 is None or len(lines) > 3:
        raise ValueError("CelesTrak response did not contain exactly one element set")
    if not tle_checksum_valid(line1) or not tle_checksum_valid(line2):
        raise ValueError("CelesTrak TLE checksum validation failed")
    try:
        first_id, second_id = int(line1[2:7]), int(line2[2:7])
    except ValueError as exc:
        raise ValueError("CelesTrak TLE catalog number is malformed") from exc
    if first_id != int(norad_id) or second_id != int(norad_id):
        raise ValueError(
            f"CelesTrak returned catalog number {first_id}/{second_id}, expected {norad_id}"
        )
    return line1, line2, name


def fetch_celestrak_tle(norad_id: int, *, timeout_seconds: float = 10.0) -> dict:
    norad_id = int(norad_id)
    if not 1 <= norad_id <= 99999:
        raise ValueError("TLE refresh supports NORAD catalog numbers 1 through 99999")
    url = (
        "https://celestrak.org/NORAD/elements/gp.php?"
        f"CATNR={norad_id}&FORMAT=TLE"
    )
    request = Request(url, headers={"User-Agent": "SDR-MCP-tle/0.50"})
    try:
        with urlopen(request, timeout=max(2.0, min(float(timeout_seconds), 30.0))) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"CelesTrak returned HTTP {response.status}")
            payload = response.read(8193)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"CelesTrak TLE request failed: {exc}") from exc
    if len(payload) > 8192:
        raise RuntimeError("CelesTrak TLE response exceeded 8192 bytes")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("CelesTrak TLE response was not ASCII") from exc
    line1, line2, name = parse_tle_response(text, norad_id=norad_id)
    return {"tle_line1": line1, "tle_line2": line2, "satellite_name": name,
            "source": "celestrak", "source_url": url}


def normalize_satellite_watch(
    *, name: str, satellite_name: str, norad_id: int, tle_line1: str,
    tle_line2: str, latitude_deg: float, longitude_deg: float,
    elevation_m: float = 0.0, frequency_hz: int = 145_800_000,
    receiver_mode: str = "nfm", minimum_elevation_deg: float = 10.0,
    lead_seconds: int = 60, trail_seconds: int = 30,
    notify_before_seconds: int = 600, tle_source: str = "manual",
    auto_refresh: bool = False, refresh_interval_seconds: int = 86400,
    doppler_correction_mode: str = "off", doppler_step_seconds: int = 10,
    downlinks: list[dict] | None = None, downlink_selection_policy: str = "priority",
    enabled: bool = True,
) -> dict:
    EarthSatellite, load, _ = _skyfield()
    satellite_name = str(satellite_name).strip()
    if not 1 <= len(satellite_name) <= 64:
        raise ValueError("satellite_name must contain 1 through 64 characters")
    norad_id = int(norad_id)
    line1, line2 = str(tle_line1).strip(), str(tle_line2).strip()
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("tle_line1 and tle_line2 must be a valid two-line element set")
    try:
        satellite = EarthSatellite(line1, line2, satellite_name, load.timescale(builtin=True))
    except Exception as exc:
        raise ValueError(f"Invalid TLE: {exc}") from exc
    if int(satellite.model.satnum) != norad_id:
        raise ValueError(
            f"TLE NORAD catalog number {satellite.model.satnum} does not match {norad_id}"
        )
    latitude_deg = parse_coordinate(latitude_deg, axis="latitude")
    longitude_deg = parse_coordinate(longitude_deg, axis="longitude")
    elevation_m = float(elevation_m)
    if not -90 <= latitude_deg <= 90:
        raise ValueError("latitude_deg must be from -90 through 90")
    if not -180 <= longitude_deg <= 180:
        raise ValueError("longitude_deg must be from -180 through 180")
    if not -500 <= elevation_m <= 10_000:
        raise ValueError("elevation_m must be from -500 through 10000")
    frequency_hz = int(frequency_hz)
    if not (9_000 <= frequency_hz <= 31_000_000 or 60_000_000 <= frequency_hz <= 260_000_000):
        raise ValueError("frequency_hz is outside the Airspy HF+ tuning ranges")
    receiver_mode = str(receiver_mode).strip().lower()
    if receiver_mode not in {"usb", "nfm"}:
        raise ValueError("receiver_mode must be usb or nfm")
    minimum_elevation_deg = float(minimum_elevation_deg)
    if not 0 <= minimum_elevation_deg <= 60:
        raise ValueError("minimum_elevation_deg must be from 0 through 60")
    lead_seconds, trail_seconds = int(lead_seconds), int(trail_seconds)
    if not 0 <= lead_seconds <= 600 or not 0 <= trail_seconds <= 600:
        raise ValueError("lead_seconds and trail_seconds must be from 0 through 600")
    notify_before_seconds = int(notify_before_seconds)
    if not 0 <= notify_before_seconds <= 3600:
        raise ValueError("notify_before_seconds must be from 0 through 3600")
    tle_source = str(tle_source).strip().lower()
    if tle_source not in {"manual", "celestrak"}:
        raise ValueError("tle_source must be manual or celestrak")
    refresh_interval_seconds = int(refresh_interval_seconds)
    if not 21600 <= refresh_interval_seconds <= 604800:
        raise ValueError("refresh_interval_seconds must be from 21600 through 604800")
    if not isinstance(auto_refresh, bool) or not isinstance(enabled, bool):
        raise ValueError("auto_refresh and enabled must be JSON booleans")
    if auto_refresh and tle_source != "celestrak":
        raise ValueError("auto_refresh requires tle_source='celestrak'")
    doppler_correction_mode = str(doppler_correction_mode).strip().lower()
    if doppler_correction_mode not in {"off", "digital"}:
        raise ValueError("doppler_correction_mode must be off or digital")
    doppler_step_seconds = int(doppler_step_seconds)
    if not 1 <= doppler_step_seconds <= 60:
        raise ValueError("doppler_step_seconds must be from 1 through 60")
    if downlinks is None:
        downlinks = [{
            "downlink_id": "legacy-sstv", "label": f"{satellite_name} SSTV",
            "frequency_hz": frequency_hz, "mode": "sstv",
            "receiver_mode": receiver_mode, "priority": 1, "enabled": True,
            "retain_audio": True,
        }]
    downlinks = normalize_satellite_downlinks(downlinks)
    downlink_selection_policy = str(downlink_selection_policy).strip().lower()
    if downlink_selection_policy not in {"priority", "round_robin"}:
        raise ValueError("downlink_selection_policy must be priority or round_robin")
    epoch = satellite.epoch.utc_datetime().astimezone(timezone.utc).isoformat()
    return {
        "name": _name(name), "satellite_name": satellite_name, "norad_id": norad_id,
        "tle_line1": line1, "tle_line2": line2, "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg, "elevation_m": elevation_m,
        "frequency_hz": frequency_hz, "receiver_mode": receiver_mode,
        "minimum_elevation_deg": minimum_elevation_deg,
        "lead_seconds": lead_seconds, "trail_seconds": trail_seconds,
        "notify_before_seconds": notify_before_seconds,
        "tle_source": tle_source, "auto_refresh": auto_refresh,
        "refresh_interval_seconds": refresh_interval_seconds,
        "doppler_correction_mode": doppler_correction_mode,
        "doppler_step_seconds": doppler_step_seconds,
        "downlinks": downlinks, "downlink_selection_policy": downlink_selection_policy,
        "tle_epoch_at": epoch, "enabled": enabled,
    }


def predict_passes(watch: dict, *, start: datetime | None = None,
                   hours: float = 24.0, limit: int = 20) -> list[dict]:
    EarthSatellite, load, wgs84 = _skyfield()
    hours = float(hours)
    if not 0.25 <= hours <= 168:
        raise ValueError("hours must be from 0.25 through 168")
    start = (start or utc_now()).astimezone(timezone.utc)
    finish = start + timedelta(hours=hours)
    ts = load.timescale(builtin=True)
    satellite = EarthSatellite(
        watch["tle_line1"], watch["tle_line2"], watch["satellite_name"], ts
    )
    observer = wgs84.latlon(
        watch["latitude_deg"], watch["longitude_deg"], elevation_m=watch["elevation_m"]
    )
    times, events = satellite.find_events(
        observer, ts.from_datetime(start), ts.from_datetime(finish),
        altitude_degrees=watch["minimum_elevation_deg"],
    )
    passes, current = [], None
    for time_value, event in zip(times, events):
        instant = time_value.utc_datetime().astimezone(timezone.utc)
        topocentric = (satellite - observer).at(time_value)
        altitude, azimuth, _ = topocentric.altaz()
        point = {
            "at": instant.isoformat(), "azimuth_deg": round(float(azimuth.degrees), 2),
            "elevation_deg": round(float(altitude.degrees), 2),
        }
        _, _, distance, _, _, range_rate = topocentric.frame_latlon_and_rates(observer)
        rate_m_s = float(range_rate.km_per_s) * 1000.0
        shift_hz = -float(watch["frequency_hz"]) * rate_m_s / SPEED_OF_LIGHT_M_S
        point.update({
            "range_km": round(float(distance.km), 3),
            "range_rate_km_s": round(float(range_rate.km_per_s), 6),
            "doppler_shift_hz": round(shift_hz, 2),
            "corrected_receive_frequency_hz": round(float(watch["frequency_hz"]) + shift_hz, 2),
        })
        if int(event) == 0:
            current = {"aos": point}
        elif int(event) == 1 and current is not None:
            current["tca"] = point
        elif int(event) == 2 and current is not None and "tca" in current:
            current["los"] = point
            aos = datetime.fromisoformat(current["aos"]["at"])
            los = datetime.fromisoformat(current["los"]["at"])
            current.update({
                "satellite_name": watch["satellite_name"], "norad_id": watch["norad_id"],
                "frequency_hz": watch["frequency_hz"],
                "minimum_elevation_deg": watch["minimum_elevation_deg"],
                "maximum_elevation_deg": current["tca"]["elevation_deg"],
                "duration_seconds": round((los - aos).total_seconds(), 1),
            })
            passes.append(current)
            current = None
            if len(passes) >= max(1, min(int(limit), 100)):
                break
    epoch = satellite.epoch.utc_datetime().astimezone(timezone.utc)
    age_days = (start - epoch).total_seconds() / 86400
    for item in passes:
        item["tle_epoch"] = epoch.isoformat()
        item["tle_age_days"] = round(age_days, 2)
        item["tle_stale"] = abs(age_days) > 14
    return passes


def build_doppler_plan(watch: dict, prediction: dict, *, step_seconds: int | None = None) -> list[dict]:
    EarthSatellite, load, wgs84 = _skyfield()
    step_seconds = int(step_seconds or watch.get("doppler_step_seconds", 10))
    if not 1 <= step_seconds <= 60:
        raise ValueError("step_seconds must be from 1 through 60")
    start = datetime.fromisoformat(prediction["aos"]["at"]) - timedelta(
        seconds=int(watch["lead_seconds"])
    )
    stop = datetime.fromisoformat(prediction["los"]["at"]) + timedelta(
        seconds=int(watch["trail_seconds"])
    )
    count = int(np.ceil((stop - start).total_seconds() / step_seconds)) + 1
    moments = [min(start + timedelta(seconds=index * step_seconds), stop)
               for index in range(count)]
    ts = load.timescale(builtin=True)
    satellite = EarthSatellite(watch["tle_line1"], watch["tle_line2"],
                               watch["satellite_name"], ts)
    observer = wgs84.latlon(watch["latitude_deg"], watch["longitude_deg"],
                            elevation_m=watch["elevation_m"])
    output = []
    for moment in moments:
        time_value = ts.from_datetime(moment)
        topocentric = (satellite - observer).at(time_value)
        altitude, azimuth, _ = topocentric.altaz()
        _, _, distance, _, _, range_rate = topocentric.frame_latlon_and_rates(observer)
        rate_m_s = float(range_rate.km_per_s) * 1000.0
        shift = -float(watch["frequency_hz"]) * rate_m_s / SPEED_OF_LIGHT_M_S
        output.append({
            "at": moment.isoformat(),
            "azimuth_deg": round(float(azimuth.degrees), 2),
            "elevation_deg": round(float(altitude.degrees), 2),
            "range_km": round(float(distance.km), 3),
            "range_rate_km_s": round(float(range_rate.km_per_s), 6),
            "doppler_shift_hz": round(shift, 2),
            "corrected_receive_frequency_hz": round(float(watch["frequency_hz"]) + shift, 2),
        })
    return output


def save_doppler_plot(pass_record: dict) -> str:
    plan = pass_record.get("doppler_plan") or []
    if not plan:
        raise ValueError("Satellite pass has no persisted Doppler plan")
    from .plotting import pyplot

    plt = pyplot()

    ensure_data_dirs()
    times = [datetime.fromisoformat(item["at"]) for item in plan]
    shifts = [item["doppler_shift_hz"] for item in plan]
    elevations = [item["elevation_deg"] for item in plan]
    figure, first = plt.subplots(figsize=(10, 5))
    first.plot(times, shifts, color="#f36d2e", linewidth=2, label="Doppler shift")
    first.axhline(0, color="gray", linewidth=0.8)
    first.set_ylabel("Doppler shift (Hz)", color="#f36d2e")
    first.set_xlabel("UTC")
    first.grid(alpha=0.25)
    second = first.twinx()
    second.plot(times, elevations, color="#2878b5", linewidth=1.5, label="Elevation")
    second.set_ylabel("Elevation (degrees)", color="#2878b5")
    first.set_title(
        f"{pass_record['satellite_name']} Doppler — {pass_record['aos_at']}"
    )
    figure.autofmt_xdate()
    figure.tight_layout()
    path = PLOT_DIR / f"{pass_record['pass_id']}-doppler.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path.resolve())


def refresh_satellite_tle(catalog, watch: dict, *, now: datetime | None = None,
                          fetcher: Callable[[int], dict] = fetch_celestrak_tle,
                          raise_errors: bool = True) -> dict:
    now = (now or utc_now()).astimezone(timezone.utc)
    if watch.get("tle_source") != "celestrak":
        raise ValueError("Satellite watch tle_source must be celestrak for managed refresh")
    try:
        downloaded = fetcher(int(watch["norad_id"]))
        normalized = normalize_satellite_watch(
            name=watch["name"], satellite_name=watch["satellite_name"],
            norad_id=watch["norad_id"], tle_line1=downloaded["tle_line1"],
            tle_line2=downloaded["tle_line2"], latitude_deg=watch["latitude_deg"],
            longitude_deg=watch["longitude_deg"], elevation_m=watch["elevation_m"],
            frequency_hz=watch["frequency_hz"], receiver_mode=watch["receiver_mode"],
            minimum_elevation_deg=watch["minimum_elevation_deg"],
            lead_seconds=watch["lead_seconds"], trail_seconds=watch["trail_seconds"],
            notify_before_seconds=watch["notify_before_seconds"], tle_source="celestrak",
            auto_refresh=watch["auto_refresh"],
            refresh_interval_seconds=watch["refresh_interval_seconds"],
            doppler_correction_mode=watch.get("doppler_correction_mode", "off"),
            doppler_step_seconds=watch.get("doppler_step_seconds", 10),
            downlinks=watch.get("downlinks"),
            downlink_selection_policy=watch.get("downlink_selection_policy", "priority"),
            enabled=watch["enabled"],
        )
        return catalog.record_satellite_tle_refresh(
            watch["watch_id"], status="succeeded",
            tle_line1=normalized["tle_line1"], tle_line2=normalized["tle_line2"],
            tle_epoch_at=normalized["tle_epoch_at"], now=now,
        )
    except Exception as exc:
        failed = catalog.record_satellite_tle_refresh(
            watch["watch_id"], status="failed",
            error=f"{type(exc).__name__}: {exc}", now=now,
        )
        if raise_errors:
            raise RuntimeError(f"TLE refresh failed; last-known-good elements retained: {exc}") from exc
        return failed


class SatellitePassScheduler:
    def __init__(self, catalog, launch_receiver: Callable[..., dict],
                 receiver_busy: Callable[[], bool], *, poll_seconds: float = 5.0,
                 tle_fetcher: Callable[[int], dict] = fetch_celestrak_tle) -> None:
        self.catalog, self.launch_receiver, self.receiver_busy = catalog, launch_receiver, receiver_busy
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.tle_fetcher = tle_fetcher
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._last_prediction_at: str | None = None
        self._last_tle_refresh_at: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="SDR-MCP-satellite", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def status(self) -> dict:
        watches = self.catalog.list_satellite_watches(enabled=True, limit=200)
        planned = self.catalog.list_satellite_passes(state="planned", limit=200)
        failed_tles = [item for item in watches if item.get("last_tle_refresh_status") == "failed"]
        return {"running": bool(self._thread and self._thread.is_alive()),
                "enabled_watch_count": len(watches), "planned_pass_count": len(planned),
                "next_start_at": planned[0]["start_at"] if planned else None,
                "last_prediction_at": self._last_prediction_at,
                "last_tle_refresh_at": self._last_tle_refresh_at,
                "tle_refresh_failure_count": len(failed_tles),
                "last_error": self._last_error}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.poll_seconds)

    def refresh(self, now: datetime | None = None) -> list[dict]:
        now = (now or utc_now()).astimezone(timezone.utc)
        for watch in self.catalog.due_satellite_tle_refreshes(now.isoformat(), limit=20):
            refresh_satellite_tle(
                self.catalog, watch, now=now, fetcher=self.tle_fetcher,
                raise_errors=False,
            )
            self._last_tle_refresh_at = now.isoformat()
        results = []
        for watch in self.catalog.list_satellite_watches(enabled=True, limit=200):
            passes = predict_passes(watch, start=now, hours=48, limit=20)
            enabled_downlinks = [item for item in watch.get("downlinks", []) if item["enabled"]]
            for index, predicted in enumerate(passes):
                if watch.get("downlink_selection_policy") == "round_robin":
                    downlink = enabled_downlinks[index % len(enabled_downlinks)]
                else:
                    downlink = enabled_downlinks[0]
                downlink_watch = {**watch, "frequency_hz": downlink["frequency_hz"],
                                  "receiver_mode": downlink["receiver_mode"]}
                with_track = dict(predicted)
                with_track["selected_downlink"] = downlink
                with_track["frequency_hz"] = downlink["frequency_hz"]
                with_track["doppler_track"] = build_doppler_plan(downlink_watch, predicted)
                results.append(self.catalog.save_satellite_pass(watch, with_track))
        self._last_prediction_at = now.isoformat()
        return results

    def tick(self, now: datetime | None = None) -> list[dict]:
        now = (now or utc_now()).astimezone(timezone.utc)
        if self._last_prediction_at is None or (
            now - datetime.fromisoformat(self._last_prediction_at)
        ) >= timedelta(minutes=30):
            self.refresh(now)
        outcomes = []
        for item in self.catalog.due_satellite_pass_notifications(now.isoformat(), limit=20):
            event = self.catalog.record_satellite_pass_event(
                item["pass_id"], event_kind="prepass"
            )
            deliveries = self.catalog.enqueue_webhook_deliveries(event)
            event["webhook_delivery_count"] = len(deliveries)
            outcomes.append(event)
        for item in self.catalog.list_satellite_passes(state="launched", limit=100):
            if not item.get("job_id"):
                continue
            try:
                job = self.catalog.get_job(item["job_id"])
            except ValueError:
                continue
            if job["state"] in {"completed", "stopped", "failed", "interrupted"}:
                finished = self.catalog.record_satellite_pass(
                    item["pass_id"], state=job["state"], job_id=item["job_id"],
                    error=job.get("error"))
                outcomes.append(finished)
                event = self.catalog.record_satellite_pass_event(
                    item["pass_id"], event_kind="outcome"
                )
                self.catalog.enqueue_webhook_deliveries(event)
        for item in self.catalog.due_satellite_passes(now.isoformat(), limit=10):
            watch = self.catalog.get_satellite_watch(item["watch_id"])
            if not watch["enabled"]:
                outcomes.append(self.catalog.record_satellite_pass(
                    item["pass_id"], state="superseded", error="Satellite watch is disabled"))
                continue
            if self.receiver_busy():
                finished = self.catalog.record_satellite_pass(
                    item["pass_id"], state="skipped_busy",
                    error="Airspy receiver is occupied by another long-running job")
                outcomes.append(finished)
                self.catalog.enqueue_webhook_deliveries(
                    self.catalog.record_satellite_pass_event(item["pass_id"], event_kind="outcome"))
                continue
            remaining = (datetime.fromisoformat(item["stop_at"]) - now).total_seconds()
            if remaining < 30:
                finished = self.catalog.record_satellite_pass(
                    item["pass_id"], state="missed", error="Pass window ended before launch")
                outcomes.append(finished)
                self.catalog.enqueue_webhook_deliveries(
                    self.catalog.record_satellite_pass_event(item["pass_id"], event_kind="outcome"))
                continue
            try:
                launched = self.launch_receiver(
                    watch=watch, pass_record=item, duration_seconds=remaining,
                )
                outcomes.append(self.catalog.record_satellite_pass(
                    item["pass_id"], state="launched", job_id=launched.get("job_id")))
            except Exception as exc:
                finished = self.catalog.record_satellite_pass(
                    item["pass_id"], state="failed", error=f"{type(exc).__name__}: {exc}")
                outcomes.append(finished)
                self.catalog.enqueue_webhook_deliveries(
                    self.catalog.record_satellite_pass_event(item["pass_id"], event_kind="outcome"))
        return outcomes
