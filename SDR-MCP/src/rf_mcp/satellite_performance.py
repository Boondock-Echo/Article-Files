from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import PLOT_DIR, SATELLITE_DIR, ensure_data_dirs


PACKET_MODES = {"ax25_afsk1200", "ax25_g3ruh9600"}


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def score_satellite_pass(pass_record: dict, observations: list[dict],
                         telemetry_value_count: int = 0,
                         sstv_images: list[dict] | None = None) -> dict:
    sstv_images = list(sstv_images or [])
    outcome_scores = {"completed": 1.0, "stopped": 0.65, "launched": 0.35,
                      "skipped_busy": 0.0, "missed": 0.0, "failed": 0.0,
                      "interrupted": 0.1, "superseded": 0.0}
    outcome_score = outcome_scores.get(pass_record.get("state"), 0.0)
    total_packets = sum(int(item.get("packet_count", 0)) for item in observations)
    valid_packets = sum(int(item.get("valid_packet_count", 0)) for item in observations)
    packet_observations = [item for item in observations if item.get("mode") in PACKET_MODES]
    rms_values = [float(item.get("details", {}).get("rms", 0) or 0)
                  for item in observations if float(item.get("details", {}).get("rms", 0) or 0) > 0]
    peak_values = [float(item.get("details", {}).get("peak", 0) or 0)
                   for item in observations if float(item.get("details", {}).get("peak", 0) or 0) > 0]
    rms = float(np.mean(rms_values)) if rms_values else 0.0
    rms_dbfs = 20 * math.log10(rms) if rms > 0 else None
    signal_score = _clamp(((rms_dbfs or -100.0) + 80.0) / 60.0)
    duration = sum(float(item.get("duration_seconds", 0)) for item in observations)
    expected = max(float(pass_record.get("prediction", {}).get("duration_seconds", 0)), 1.0)
    duration_score = _clamp(duration / expected)
    components = {"outcome": outcome_score, "signal_level": signal_score,
                  "duration_coverage": duration_score}
    selected_mode = (pass_record.get("selected_downlink") or {}).get("mode")
    if selected_mode == "sstv":
        image_yield = _clamp(len(sstv_images))
        quality_values = [float(item["quality"]) for item in sstv_images
                          if item.get("quality") is not None]
        image_quality = float(np.mean(quality_values)) if quality_values else 0.0
        components.update({"image_yield": image_yield, "image_quality": image_quality})
        score = 100 * (0.30 * outcome_score + 0.20 * signal_score
                       + 0.20 * duration_score + 0.15 * image_yield
                       + 0.15 * image_quality)
        confidence = "high" if sstv_images else "medium" if observations else "low"
    elif packet_observations:
        packet_yield = _clamp(total_packets / 10.0)
        fcs_rate = valid_packets / total_packets if total_packets else 0.0
        telemetry_yield = _clamp(telemetry_value_count / 20.0)
        components.update({"packet_yield": packet_yield, "valid_fcs_rate": fcs_rate,
                           "telemetry_yield": telemetry_yield})
        score = 100 * (0.20 * outcome_score + 0.15 * signal_score
                       + 0.10 * duration_score + 0.20 * packet_yield
                       + 0.25 * fcs_rate + 0.10 * telemetry_yield)
        confidence = "high" if total_packets >= 10 else "medium" if total_packets else "low"
    else:
        score = 100 * (0.40 * outcome_score + 0.30 * signal_score + 0.30 * duration_score)
        confidence = "medium" if observations else "low"
    downlink = pass_record.get("selected_downlink") or {}
    return {
        "schema": "rf-mcp.satellite-pass-performance.v1",
        "pass_id": pass_record["pass_id"], "watch_id": pass_record.get("watch_id"),
        "satellite_name": pass_record["satellite_name"], "state": pass_record["state"],
        "aos_at": pass_record["aos_at"],
        "maximum_elevation_deg": pass_record["maximum_elevation_deg"],
        "downlink_id": downlink.get("downlink_id"), "downlink_label": downlink.get("label"),
        "mode": downlink.get("mode"), "frequency_hz": downlink.get("frequency_hz"),
        "observation_count": len(observations), "captured_duration_seconds": round(duration, 3),
        "packet_count": total_packets, "valid_packet_count": valid_packets,
        "valid_fcs_rate": round(valid_packets / total_packets, 4) if total_packets else None,
        "telemetry_value_count": int(telemetry_value_count), "mean_rms": round(rms, 8),
        "sstv_image_count": len(sstv_images),
        "mean_sstv_quality": (round(float(np.mean(
            [item["quality"] for item in sstv_images if item.get("quality") is not None])), 4)
            if any(item.get("quality") is not None for item in sstv_images) else None),
        "mean_rms_dbfs": round(rms_dbfs, 2) if rms_dbfs is not None else None,
        "peak": round(max(peak_values), 8) if peak_values else None,
        "performance_score": round(score, 1), "score_confidence": confidence,
        "score_components": {key: round(value, 4) for key, value in components.items()},
    }


def build_pass_report(catalog, item: dict) -> dict:
    observations = catalog.list_satellite_observations(pass_id=item["pass_id"], limit=100)
    telemetry = catalog.list_satellite_telemetry_values(pass_id=item["pass_id"], limit=5000)
    images = catalog.list_sstv_images(source_satellite_pass_id=item["pass_id"], limit=100)
    if not observations and item.get("job_id"):
        try:
            job = catalog.get_job(item["job_id"])
            if job.get("job_type") == "sstv_watch":
                observations = [{"mode": "sstv", "packet_count": 0,
                                 "valid_packet_count": 0,
                                 "duration_seconds": job.get("config", {}).get(
                                     "watch_duration_seconds", 0), "details": {}}]
        except ValueError:
            pass
    return score_satellite_pass(item, observations, len(telemetry), images)


def build_pass_reports(catalog, *, watch_id: str | None = None, limit: int = 200) -> list[dict]:
    passes = catalog.list_satellite_passes(
        watch_id=watch_id, limit=limit, newest_first=True
    )
    reports = []
    for item in passes:
        if item["state"] in {"planned", "superseded"}:
            continue
        reports.append(build_pass_report(catalog, item))
    return sorted(reports, key=lambda item: item["aos_at"], reverse=True)


def summarize_pass_performance(reports: list[dict]) -> dict:
    groups = defaultdict(list)
    for report in reports:
        groups[(report.get("downlink_id") or "unknown", report.get("downlink_label") or "Unknown",
                report.get("mode") or "unknown")].append(report)
    downlinks = []
    for (identifier, label, mode), items in groups.items():
        packets = sum(item["packet_count"] for item in items)
        valid = sum(item["valid_packet_count"] for item in items)
        downlinks.append({
            "downlink_id": identifier, "downlink_label": label, "mode": mode,
            "pass_count": len(items),
            "mean_performance_score": round(float(np.mean(
                [item["performance_score"] for item in items])), 1),
            "median_performance_score": round(float(np.median(
                [item["performance_score"] for item in items])), 1),
            "packet_count": packets, "valid_packet_count": valid,
            "valid_fcs_rate": round(valid / packets, 4) if packets else None,
            "telemetry_value_count": sum(item["telemetry_value_count"] for item in items),
        })
    downlinks.sort(key=lambda item: (-item["mean_performance_score"], -item["pass_count"]))
    return {"pass_count": len(reports),
            "mean_performance_score": round(float(np.mean(
                [item["performance_score"] for item in reports])), 1) if reports else None,
            "by_downlink": downlinks,
            "recommendation": (downlinks[0] if downlinks else None)}


def save_pass_performance_plot(reports: list[dict], *, title: str) -> str:
    if not reports:
        raise ValueError("No attempted satellite passes are available to plot")
    from .plotting import pyplot

    plt = pyplot()
    ensure_data_dirs()
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    labels = sorted({item.get("downlink_label") or "Unknown" for item in reports})
    colors = {label: plt.cm.tab10(index % 10) for index, label in enumerate(labels)}
    chronological = sorted(reports, key=lambda item: item["aos_at"])
    for label in labels:
        items = [item for item in chronological if (item.get("downlink_label") or "Unknown") == label]
        axes[0].plot([datetime.fromisoformat(item["aos_at"]) for item in items],
                     [item["performance_score"] for item in items], marker="o",
                     label=label, color=colors[label])
        axes[1].scatter([item["maximum_elevation_deg"] for item in items],
                        [item["performance_score"] for item in items], label=label,
                        color=colors[label], alpha=0.8)
    axes[0].set_title(title); axes[0].set_ylabel("Performance score"); axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Maximum elevation (degrees)"); axes[1].set_ylabel("Performance score")
    axes[1].grid(alpha=0.25); axes[0].legend(); axes[1].legend(); figure.autofmt_xdate()
    path = PLOT_DIR / f"satellite-performance-{uuid4().hex[:12]}.png"
    figure.savefig(path, dpi=150); plt.close(figure)
    return str(path.resolve())


def export_pass_performance(reports: list[dict], *, output_format: str) -> str:
    output_format = str(output_format).strip().lower()
    if output_format not in {"json", "csv"}:
        raise ValueError("output_format must be json or csv")
    ensure_data_dirs()
    path = SATELLITE_DIR / f"pass-performance-{uuid4().hex[:12]}.{output_format}"
    if output_format == "json":
        path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    else:
        fields = [key for key in reports[0] if key != "score_components"] if reports else [
            "pass_id", "satellite_name", "aos_at", "downlink_id", "performance_score"
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(reports)
    return str(path.resolve())
