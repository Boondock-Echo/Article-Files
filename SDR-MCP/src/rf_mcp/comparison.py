from __future__ import annotations

from bisect import bisect_left, bisect_right
import heapq
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np


SUPPORTED_JOB_TYPES = {"band_scan", "band_survey"}


def _compatible_digital_power(baseline: dict, comparison: dict) -> bool:
    left_scale = baseline.get("digital_power_scale", {})
    right_scale = comparison.get("digital_power_scale", {})
    if left_scale.get("scale") != "digital_dbfs_per_hz_v1" or left_scale != right_scale:
        return False
    left_profile = baseline.get("receiver_profile", {})
    right_profile = comparison.get("receiver_profile", {})
    comparable_keys = ("sample_rate_hz", "agc", "attenuation_steps", "lna")
    if any(left_profile.get(key) != right_profile.get(key) for key in comparable_keys):
        return False
    if left_profile.get("agc") is not False:
        return False
    signals = list(baseline.get("signals", [])) + list(comparison.get("signals", []))
    return all("digital_power_dbfs_10khz" in item for item in signals)


def _range(result: dict) -> tuple[float, float]:
    scanned = result.get("scanned_range_hz")
    if scanned and len(scanned) == 2:
        return float(scanned[0]), float(scanned[1])
    config = result.get("config", {})
    try:
        return float(config["start_frequency_hz"]), float(config["stop_frequency_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved job does not contain a band scan range") from exc


def _classifications(result: dict, signals: list[dict], tolerance_hz: float) -> dict[int, dict]:
    completed = sorted(
        (
            (float(item["frequency_hz"]), index, item)
            for index, item in enumerate(result.get("classifications", []))
            if item.get("status") == "completed" and item.get("frequency_hz") is not None
        ),
        key=lambda entry: (entry[0], entry[1]),
    )
    completed_frequencies = [entry[0] for entry in completed]
    sorted_signals = sorted(
        ((float(signal["frequency_hz"]), index) for index, signal in enumerate(signals)),
        key=lambda entry: (entry[0], entry[1]),
    )
    mapping: dict[int, dict] = {}
    for frequency, signal_index in sorted_signals:
        start = bisect_left(completed_frequencies, frequency - tolerance_hz)
        stop = bisect_right(completed_frequencies, frequency + tolerance_hz)
        if start < stop:
            # Preserve stable-min behavior explicitly with the original classification index.
            _, _, item = min(
                islice(completed, start, stop),
                key=lambda entry: (abs(entry[0] - frequency), entry[1]),
            )
            mapping[signal_index] = item
    return mapping


def _frequency_candidate_pairs(
    left_signals: list[dict], right_signals: list[dict], tolerance_hz: float
) -> Iterator[tuple[float, int, int]]:
    """Yield the old greedy sort order without storing the frequency cross product."""
    left = sorted(
        ((float(signal["frequency_hz"]), index) for index, signal in enumerate(left_signals)),
        key=lambda entry: (entry[0], entry[1]),
    )
    sorted_right = sorted(
        ((float(signal["frequency_hz"]), index) for index, signal in enumerate(right_signals)),
        key=lambda entry: (entry[0], entry[1]),
    )
    right_groups: list[tuple[float, list[int]]] = []
    for frequency, index in sorted_right:
        if not right_groups or right_groups[-1][0] != frequency:
            right_groups.append((frequency, []))
        right_groups[-1][1].append(index)
    right_frequencies = [frequency for frequency, _ in right_groups]

    def candidates(left_frequency: float, left_index: int, start: int, stop: int):
        # Merge the two frequency fronts; each duplicate group is already index ordered.
        insertion = bisect_left(right_frequencies, left_frequency, start, stop)
        lower = insertion - 1
        upper = insertion
        while lower >= start or upper < stop:
            lower_distance = (
                left_frequency - right_frequencies[lower]
                if lower >= start
                else float("inf")
            )
            upper_distance = (
                right_frequencies[upper] - left_frequency
                if upper < stop
                else float("inf")
            )
            distance = min(lower_distance, upper_distance)
            groups = []
            if lower_distance == distance:
                groups.append(right_groups[lower][1])
                lower -= 1
            if upper_distance == distance:
                groups.append(right_groups[upper][1])
                upper += 1
            # Two equidistant groups require an explicit original-index merge.
            for right_index in heapq.merge(*groups):
                yield distance, left_index, right_index

    streams = []
    queue: list[tuple[float, int, int, int]] = []
    for left_frequency, left_index in left:
        start = bisect_left(right_frequencies, left_frequency - tolerance_hz)
        stop = bisect_right(right_frequencies, left_frequency + tolerance_hz)
        if start == stop:
            continue
        stream_index = len(streams)
        stream = candidates(left_frequency, left_index, start, stop)
        streams.append(stream)
        distance, candidate_left, candidate_right = next(stream)
        heapq.heappush(queue, (distance, candidate_left, candidate_right, stream_index))
    while queue:
        distance, left_index, right_index, stream_index = heapq.heappop(queue)
        yield distance, left_index, right_index
        try:
            next_distance, next_left, next_right = next(streams[stream_index])
        except StopIteration:
            continue
        heapq.heappush(queue, (next_distance, next_left, next_right, stream_index))


def compare_survey_results(
    baseline: dict,
    comparison: dict,
    *,
    frequency_tolerance_hz: float = 1_500,
    power_change_threshold_db: float = 6.0,
    frequency_shift_threshold_hz: float = 250,
) -> dict:
    """Compare two saved scan/survey result dictionaries without using the receiver."""
    frequency_tolerance_hz = float(frequency_tolerance_hz)
    power_change_threshold_db = float(power_change_threshold_db)
    frequency_shift_threshold_hz = float(frequency_shift_threshold_hz)
    if not 50 <= frequency_tolerance_hz <= 100_000:
        raise ValueError("frequency_tolerance_hz must be from 50 through 100000")
    if not 0.5 <= power_change_threshold_db <= 60:
        raise ValueError("power_change_threshold_db must be from 0.5 through 60")
    if not 1 <= frequency_shift_threshold_hz <= frequency_tolerance_hz:
        raise ValueError("frequency_shift_threshold_hz must be from 1 through the match tolerance")

    baseline_range = _range(baseline)
    comparison_range = _range(comparison)
    overlap_low = max(baseline_range[0], comparison_range[0])
    overlap_high = min(baseline_range[1], comparison_range[1])
    if overlap_high <= overlap_low:
        raise ValueError("Band surveys do not overlap")
    union_span = max(baseline_range[1], comparison_range[1]) - min(
        baseline_range[0], comparison_range[0]
    )
    overlap_fraction = (overlap_high - overlap_low) / union_span
    if overlap_fraction < 0.9:
        raise ValueError("Band surveys must overlap by at least 90 percent")

    baseline_signals = [
        dict(item)
        for item in baseline.get("signals", [])
        if overlap_low <= float(item["frequency_hz"]) <= overlap_high
    ]
    comparison_signals = [
        dict(item)
        for item in comparison.get("signals", [])
        if overlap_low <= float(item["frequency_hz"]) <= overlap_high
    ]
    baseline_classes = _classifications(baseline, baseline_signals, frequency_tolerance_hz)
    comparison_classes = _classifications(
        comparison, comparison_signals, frequency_tolerance_hz
    )
    use_digital_power = _compatible_digital_power(baseline, comparison)
    power_field = "digital_power_dbfs_10khz" if use_digital_power else "relative_power_db"
    power_scale = "digital_power_dbfs_10khz" if use_digital_power else "legacy_relative_db"
    power_change_label = "digital_power_changed" if use_digital_power else "relative_power_changed"

    used_left: set[int] = set()
    used_right: set[int] = set()
    matched: list[dict] = []
    for _, left_index, right_index in _frequency_candidate_pairs(
        baseline_signals, comparison_signals, frequency_tolerance_hz
    ):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        left = baseline_signals[left_index]
        right = comparison_signals[right_index]
        frequency_delta = float(right["frequency_hz"]) - float(left["frequency_hz"])
        baseline_power = float(left[power_field])
        comparison_power = float(right[power_field])
        power_delta = comparison_power - baseline_power
        relative_power_delta = float(right["relative_power_db"]) - float(
            left["relative_power_db"]
        )
        left_class = baseline_classes.get(left_index)
        right_class = comparison_classes.get(right_index)
        left_label = left_class.get("best_label") if left_class else None
        right_label = right_class.get("best_label") if right_class else None
        changes = []
        if abs(frequency_delta) >= frequency_shift_threshold_hz:
            changes.append("frequency_shifted")
        if abs(power_delta) >= power_change_threshold_db:
            changes.append(power_change_label)
        if left_label and right_label and left_label != right_label:
            changes.append("classification_changed")
        matched.append(
            {
                "baseline_frequency_hz": float(left["frequency_hz"]),
                "comparison_frequency_hz": float(right["frequency_hz"]),
                "frequency_delta_hz": frequency_delta,
                "baseline_relative_power_db": float(left["relative_power_db"]),
                "comparison_relative_power_db": float(right["relative_power_db"]),
                "relative_power_delta_db": relative_power_delta,
                "baseline_power_db": baseline_power,
                "comparison_power_db": comparison_power,
                "power_delta_db": power_delta,
                "power_scale": power_scale,
                "baseline_classification": left_label,
                "comparison_classification": right_label,
                "baseline_classification_ambiguous": (
                    bool(left_class.get("ambiguous")) if left_class else None
                ),
                "comparison_classification_ambiguous": (
                    bool(right_class.get("ambiguous")) if right_class else None
                ),
                "changes": changes,
                "changed": bool(changes),
            }
        )

    disappeared = [
        baseline_signals[index] for index in range(len(baseline_signals)) if index not in used_left
    ]
    new = [
        comparison_signals[index]
        for index in range(len(comparison_signals))
        if index not in used_right
    ]
    matched.sort(key=lambda item: item["comparison_frequency_hz"])
    new.sort(key=lambda item: item["frequency_hz"])
    disappeared.sort(key=lambda item: item["frequency_hz"])
    changed = [item for item in matched if item["changed"]]
    return {
        "comparison_method": "nearest_unique_frequency_match_v1",
        "power_comparison_scale": power_scale,
        "measurement_caveat": (
            "Power deltas use fixed-gain 10 kHz integrated digital dBFS. They are repeatable "
            "digital-domain measurements, not calibrated RF input power or dBm."
            if use_digital_power
            else "Power deltas compare within-scan normalized relative levels, not calibrated "
            "dBm. A change in the strongest carrier can shift every reported relative level."
        ),
        "overload_warning": bool(
            baseline.get("overload", {}).get("suspected")
            or comparison.get("overload", {}).get("suspected")
        ),
        "baseline_range_hz": list(baseline_range),
        "comparison_range_hz": list(comparison_range),
        "compared_overlap_hz": [overlap_low, overlap_high],
        "overlap_fraction": overlap_fraction,
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "frequency_shift_threshold_hz": frequency_shift_threshold_hz,
        "power_change_threshold_db": power_change_threshold_db,
        "baseline_signal_count": len(baseline_signals),
        "comparison_signal_count": len(comparison_signals),
        "matched_count": len(matched),
        "stable_count": len(matched) - len(changed),
        "changed_count": len(changed),
        "new_count": len(new),
        "disappeared_count": len(disappeared),
        "matched_signals": matched,
        "changed_signals": changed,
        "new_signals": new,
        "disappeared_signals": disappeared,
    }


def save_comparison_plot(path: Path, result: dict) -> None:
    from .plotting import pyplot

    plt = pyplot()
    matched = result["matched_signals"]
    new = result["new_signals"]
    disappeared = result["disappeared_signals"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    for item in matched:
        color = "#f36d2e" if item["changed"] else "#888888"
        axes[0].plot(
            [item["baseline_frequency_hz"] / 1e6, item["comparison_frequency_hz"] / 1e6],
            [1, 0],
            color=color,
            alpha=0.7,
            linewidth=1,
        )
    if matched:
        axes[0].scatter(
            [item["baseline_frequency_hz"] / 1e6 for item in matched],
            np.ones(len(matched)),
            marker="x",
            color="#1677b8",
            label="Baseline",
        )
        axes[0].scatter(
            [item["comparison_frequency_hz"] / 1e6 for item in matched],
            np.zeros(len(matched)),
            marker="o",
            facecolors="none",
            edgecolors="#f36d2e",
            label="Comparison",
        )
    if new:
        axes[0].scatter(
            [item["frequency_hz"] / 1e6 for item in new],
            np.zeros(len(new)),
            marker="^",
            color="#2f9e44",
            label="New",
        )
    if disappeared:
        axes[0].scatter(
            [item["frequency_hz"] / 1e6 for item in disappeared],
            np.ones(len(disappeared)),
            marker="v",
            color="#c92a2a",
            label="Disappeared",
        )
    axes[0].set_yticks([0, 1], ["Comparison", "Baseline"])
    axes[0].set_xlabel("Frequency (MHz)")
    axes[0].set_title("Band Change Map")
    axes[0].grid(axis="x", alpha=0.25)
    if matched or new or disappeared:
        axes[0].legend(loc="upper center", ncol=4)

    if matched:
        frequencies = [item["comparison_frequency_hz"] / 1e6 for item in matched]
        deltas = [item["power_delta_db"] for item in matched]
        colors = ["#f36d2e" if item["changed"] else "#1677b8" for item in matched]
        axes[1].scatter(frequencies, deltas, c=colors, s=28)
    threshold = result["power_change_threshold_db"]
    axes[1].axhline(threshold, color="#777", linestyle="--", linewidth=0.9)
    axes[1].axhline(-threshold, color="#777", linestyle="--", linewidth=0.9)
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].set_xlabel("Comparison frequency (MHz)")
    axes[1].set_ylabel("Power delta (dB)")
    scale_label = (
        "10 kHz integrated digital dBFS"
        if result.get("power_comparison_scale") == "digital_power_dbfs_10khz"
        else "legacy within-scan relative power"
    )
    axes[1].set_title(f"Matched-Signal Power Changes — {scale_label}")
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
