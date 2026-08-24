from __future__ import annotations

import csv
import json
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4

import numpy as np

from .config import PLOT_DIR, RESULT_DIR, ensure_data_dirs


RF_BANDS = {
    "160m": {"label": "160 meter amateur", "start_hz": 1_800_000, "stop_hz": 2_000_000},
    "80m": {"label": "80 meter amateur", "start_hz": 3_500_000, "stop_hz": 4_000_000},
    "40m": {"label": "40 meter amateur", "start_hz": 7_000_000, "stop_hz": 7_300_000},
    "30m": {"label": "30 meter amateur", "start_hz": 10_100_000, "stop_hz": 10_150_000},
    "20m": {"label": "20 meter amateur", "start_hz": 14_000_000, "stop_hz": 14_350_000},
    "17m": {"label": "17 meter amateur", "start_hz": 18_068_000, "stop_hz": 18_168_000},
    "15m": {"label": "15 meter amateur", "start_hz": 21_000_000, "stop_hz": 21_450_000},
    "12m": {"label": "12 meter amateur", "start_hz": 24_890_000, "stop_hz": 24_990_000},
    "10m": {"label": "10 meter amateur", "start_hz": 28_000_000, "stop_hz": 29_700_000},
    "fm_broadcast": {"label": "FM broadcast", "start_hz": 88_000_000, "stop_hz": 108_000_000},
    "airband": {"label": "VHF civil airband", "start_hz": 118_000_000, "stop_hz": 137_000_000},
    "2m": {"label": "2 meter amateur", "start_hz": 144_000_000, "stop_hz": 148_000_000},
    "noaa_weather": {"label": "NOAA weather radio", "start_hz": 162_390_000, "stop_hz": 162_560_000},
}


def _cluster_signals(runs: list[dict], tolerance_hz: float) -> list[dict]:
    records = []
    for run_index, run in enumerate(runs):
        for signal_index, signal in enumerate(run.get("signals") or []):
            records.append((float(signal["frequency_hz"]), run_index, signal_index,
                            run.get("created_at"),
                            float(signal.get("above_noise_db", 0.0))))
    # The secondary keys make the sweep (and therefore timestamp handling) stable
    # even when callers supply runs and signals in an arbitrary order.
    records.sort(key=lambda record: (record[0], record[1], record[2]))

    clusters: list[dict] = []
    for frequency, run_index, _signal_index, created_at, power in records:
        match = clusters[-1] if clusters else None
        if (match is None
                or frequency - match["frequency_sum_hz"] / match["observation_count"]
                > tolerance_hz):
            match = {
                "frequency_sum_hz": 0.0,
                "observation_count": 0,
                "minimum_frequency_hz": frequency,
                "maximum_frequency_hz": frequency,
                "maximum_above_noise_db": power,
                "run_indices": set(),
                "first_seen_at": created_at,
                "last_seen_at": created_at,
            }
            clusters.append(match)
        match["frequency_sum_hz"] += frequency
        match["observation_count"] += 1
        match["maximum_frequency_hz"] = frequency
        match["maximum_above_noise_db"] = max(match["maximum_above_noise_db"], power)
        match["run_indices"].add(run_index)
        if created_at is not None:
            if match["first_seen_at"] is None or created_at < match["first_seen_at"]:
                match["first_seen_at"] = created_at
            if match["last_seen_at"] is None or created_at > match["last_seen_at"]:
                match["last_seen_at"] = created_at
    output = []
    for item in clusters:
        mean_frequency_hz = item["frequency_sum_hz"] / item["observation_count"]
        output.append({
            "mean_frequency_hz": round(mean_frequency_hz, 1),
            "minimum_frequency_hz": round(item["minimum_frequency_hz"], 1),
            "maximum_frequency_hz": round(item["maximum_frequency_hz"], 1),
            "observation_count": item["observation_count"],
            "run_count": len(item["run_indices"]),
            "detection_rate": round(len(item["run_indices"]) / max(1, len(runs)), 4),
            "maximum_above_noise_db": round(item["maximum_above_noise_db"], 2),
            "first_seen_at": item["first_seen_at"], "last_seen_at": item["last_seen_at"],
        })
    return sorted(output, key=lambda item: (-item["run_count"], item["mean_frequency_hz"]))


def summarize_activity_runs(runs: list[dict], *, frequency_tolerance_hz: float = 1500,
                            noise_anomaly_db: float = 6.0,
                            occupancy_anomaly_fraction: float = 0.05) -> dict:
    if not runs:
        raise ValueError("No completed activity-monitor runs were found")
    runs = sorted(runs, key=lambda item: item.get("created_at") or "")
    rows = []
    for item in runs:
        rows.append({
            "job_id": item["job_id"], "created_at": item.get("created_at"),
            "completed_at": item.get("completed_at"),
            "digital_noise_floor_dbfs_hz": item.get("digital_noise_floor_dbfs_hz"),
            "relative_noise_floor_db": item.get("relative_noise_floor_db"),
            "occupied_bin_fraction": item.get("occupied_bin_fraction"),
            "signal_count": int(item.get("signal_count", 0)),
            "overload_suspected": bool((item.get("overload") or {}).get("suspected", False)),
        })
    latest = rows[-1]
    baseline_rows = rows[:-1]
    noise_values = [float(row["digital_noise_floor_dbfs_hz"]) for row in baseline_rows
                    if row["digital_noise_floor_dbfs_hz"] is not None]
    occupancy_values = [float(row["occupied_bin_fraction"]) for row in baseline_rows
                        if row["occupied_bin_fraction"] is not None]
    baseline_noise = median(noise_values) if noise_values else None
    baseline_occupancy = median(occupancy_values) if occupancy_values else None
    noise_delta = (float(latest["digital_noise_floor_dbfs_hz"]) - baseline_noise
                   if baseline_noise is not None and latest["digital_noise_floor_dbfs_hz"] is not None else None)
    occupancy_delta = (float(latest["occupied_bin_fraction"]) - baseline_occupancy
                       if baseline_occupancy is not None and latest["occupied_bin_fraction"] is not None else None)
    clusters = _cluster_signals(runs, float(frequency_tolerance_hz))
    latest_frequencies = [float(item["frequency_hz"]) for item in runs[-1].get("signals") or []]
    historical_frequencies = sorted(float(signal["frequency_hz"]) for run in runs[:-1]
                                    for signal in run.get("signals") or [])
    new_signals = []
    if runs[:-1]:
        for frequency in latest_frequencies:
            insertion_point = bisect_left(historical_frequencies, frequency)
            previous_is_near = (insertion_point > 0 and
                                frequency - historical_frequencies[insertion_point - 1]
                                <= frequency_tolerance_hz)
            next_is_near = (insertion_point < len(historical_frequencies) and
                            historical_frequencies[insertion_point] - frequency
                            <= frequency_tolerance_hz)
            if not previous_is_near and not next_is_near:
                new_signals.append(frequency)
    anomalies = []
    if noise_delta is not None and noise_delta >= noise_anomaly_db:
        anomalies.append({"type": "raised_noise_floor", "delta_db": round(noise_delta, 2)})
    if occupancy_delta is not None and occupancy_delta >= occupancy_anomaly_fraction:
        anomalies.append({"type": "raised_occupancy", "delta_fraction": round(occupancy_delta, 4)})
    if new_signals:
        anomalies.append({"type": "new_signals", "count": len(new_signals),
                          "frequencies_hz": new_signals})
    if latest["overload_suspected"]:
        anomalies.append({"type": "receiver_overload", "job_id": latest["job_id"]})
    return {
        "run_count": len(runs), "runs": rows, "latest_run": latest,
        "baseline": {"run_count": len(baseline_rows),
                     "median_digital_noise_floor_dbfs_hz": baseline_noise,
                     "median_occupied_bin_fraction": baseline_occupancy},
        "latest_vs_baseline": {"noise_floor_delta_db": noise_delta,
                               "occupied_bin_fraction_delta": occupancy_delta},
        "cluster_count": len(clusters), "signal_clusters": clusters,
        "new_signal_count": len(new_signals), "new_signal_frequencies_hz": new_signals,
        "anomaly_count": len(anomalies), "anomalies": anomalies,
        "measurement_warning": "Digital levels are dBFS/Hz, not calibrated RF input power.",
    }


def build_activity_dashboard(catalog, preset: dict, *, run_limit: int = 24,
                             frequency_tolerance_hz: float = 1500,
                             noise_anomaly_db: float = 6.0,
                             occupancy_anomaly_fraction: float = 0.05) -> tuple[dict, list[dict]]:
    run_limit = max(1, min(int(run_limit), 100))
    matching = []
    for job in catalog.list_jobs(state="completed", limit=200):
        if job["job_type"] not in {"band_scan", "band_survey"}:
            continue
        if (job.get("config") or {}).get("source_preset_id") != preset["preset_id"]:
            continue
        full = catalog.get_job(job["job_id"])
        if isinstance(full.get("result"), dict):
            matching.append(full["result"])
        if len(matching) >= run_limit:
            break
    matching.reverse()
    summary = summarize_activity_runs(
        matching, frequency_tolerance_hz=frequency_tolerance_hz,
        noise_anomaly_db=noise_anomaly_db,
        occupancy_anomaly_fraction=occupancy_anomaly_fraction,
    )
    summary.update({"profile_id": preset["preset_id"], "profile_name": preset["name"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "config": preset["config"]})
    return summary, matching


def save_activity_plot(summary: dict, runs: list[dict]) -> Path:
    from .plotting import pyplot

    plt = pyplot()
    ensure_data_dirs()
    identifier = f"activity-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    usable = [run for run in runs if (run.get("spectrum_summary") or {}).get("frequency_hz")]
    if usable:
        starts = [min(run["spectrum_summary"]["frequency_hz"]) for run in usable]
        stops = [max(run["spectrum_summary"]["frequency_hz"]) for run in usable]
        grid = np.linspace(max(starts), min(stops), 500)
        matrix = np.asarray([
            np.interp(grid, run["spectrum_summary"]["frequency_hz"],
                      run["spectrum_summary"]["relative_power_db"])
            for run in usable
        ])
        image = axes[0].imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                               extent=[grid[0] / 1e6, grid[-1] / 1e6, 1, len(usable)])
        figure.colorbar(image, ax=axes[0], label="Relative power (dB)")
        axes[0].set_ylabel("Survey sequence")
    else:
        axes[0].text(0.5, 0.5, "Compact spectra become available after upgrading to v0.37",
                     ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title(f"{summary['profile_name']} — frequency/time activity heatmap")
    axes[0].set_xlabel("Frequency (MHz)")
    rows = summary["runs"]
    x = np.arange(1, len(rows) + 1)
    occupancy = [row["occupied_bin_fraction"] for row in rows]
    signals = [row["signal_count"] for row in rows]
    first = axes[1]
    if any(value is not None for value in occupancy):
        first.plot(x, [np.nan if value is None else value * 100 for value in occupancy],
                   marker="o", color="#f36d2e", label="Occupied bins")
    first.set_ylabel("Occupied bins (%)", color="#f36d2e")
    first.set_xlabel("Survey sequence")
    first.grid(alpha=0.25)
    second = first.twinx()
    second.plot(x, signals, marker=".", color="#1677b8", label="Signals")
    second.set_ylabel("Detected signals", color="#1677b8")
    path = PLOT_DIR / f"{identifier}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_activity_exports(summary: dict) -> tuple[Path, Path]:
    ensure_data_dirs()
    stem = f"activity-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    json_path, csv_path = RESULT_DIR / f"{stem}.json", RESULT_DIR / f"{stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ["job_id", "created_at", "completed_at", "digital_noise_floor_dbfs_hz",
              "relative_noise_floor_db", "occupied_bin_fraction", "signal_count",
              "overload_suspected"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary["runs"])
    return json_path, csv_path
