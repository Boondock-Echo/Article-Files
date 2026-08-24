from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .signal_analysis import _complex_lowpass


CLASS_LABELS = ("am", "usb", "lsb", "cw", "nfm", "digital_or_unknown")


@dataclass(frozen=True)
class ClassificationFeatures:
    analysis_bandwidth_hz: int
    bin_width_hz: float
    peak_above_median_db: float
    carrier_prominence_db: float
    sideband_imbalance_db: float
    occupied_bandwidth_hz: float
    envelope_coefficient_of_variation: float
    instantaneous_frequency_std_hz: float
    spectral_entropy: float
    significant_peak_count: int
    dominant_offset_hz: float


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def extract_features(
    baseband_iq: np.ndarray,
    sample_rate_hz: int,
    analysis_bandwidth_hz: int,
    fft_size: int = 16_384,
) -> tuple[ClassificationFeatures, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    analysis_bandwidth_hz = int(analysis_bandwidth_hz)
    if not 2_000 <= analysis_bandwidth_hz <= 50_000:
        raise ValueError("analysis_bandwidth_hz must be from 2000 through 50000")
    if fft_size < 4096 or fft_size > 65_536 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two from 4096 through 65536")
    block_count = len(baseband_iq) // fft_size
    if block_count < 1:
        raise ValueError("Capture is too short for classification FFT size")

    blocks = baseband_iq[: block_count * fft_size].reshape(block_count, fft_size)
    window = np.blackman(fft_size)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / sample_rate_hz))
    mask = np.abs(frequencies) <= analysis_bandwidth_hz / 2
    frequencies = frequencies[mask]
    power = power[mask]
    power = np.maximum(power, np.finfo(float).tiny)
    power_db = 10 * np.log10(power)
    power_db -= np.max(power_db)
    bin_width_hz = float(np.median(np.diff(frequencies)))
    median_db = float(np.median(power_db))
    peak_index = int(np.argmax(power_db))
    dominant_offset_hz = float(frequencies[peak_index])

    carrier_mask = np.abs(frequencies) <= max(150.0, 2 * bin_width_hz)
    carrier_peak_db = float(np.max(power_db[carrier_mask])) if np.any(carrier_mask) else median_db
    # Zero means the carrier region contains the strongest spectral component;
    # negative values mean a sideband or offset component is stronger.
    carrier_prominence_db = carrier_peak_db - float(np.max(power_db))

    guard_hz = max(250.0, 3 * bin_width_hz)
    positive = (frequencies >= guard_hz) & (frequencies <= analysis_bandwidth_hz / 2)
    negative = (frequencies <= -guard_hz) & (frequencies >= -analysis_bandwidth_hz / 2)
    positive_power = float(np.sum(power[positive]))
    negative_power = float(np.sum(power[negative]))
    sideband_imbalance_db = float(
        10 * np.log10((positive_power + 1e-30) / (negative_power + 1e-30))
    )

    order = np.argsort(frequencies)
    sorted_frequency = frequencies[order]
    cumulative = np.cumsum(power[order])
    lower = int(np.searchsorted(cumulative, cumulative[-1] * 0.005))
    upper = min(
        len(sorted_frequency) - 1,
        int(np.searchsorted(cumulative, cumulative[-1] * 0.995)),
    )
    occupied_bandwidth_hz = float(sorted_frequency[upper] - sorted_frequency[lower])

    filtered = _complex_lowpass(
        baseband_iq,
        sample_rate_hz,
        analysis_bandwidth_hz / 2,
    )
    filtered = filtered[512:]
    envelope = np.abs(filtered)
    envelope_mean = float(np.mean(envelope))
    envelope_cv = float(np.std(envelope) / max(envelope_mean, 1e-12))
    instantaneous_frequency = (
        np.angle(filtered[1:] * np.conj(filtered[:-1])) * sample_rate_hz / (2 * np.pi)
    )
    if envelope.size > 1:
        reliable = envelope[1:] >= np.percentile(envelope[1:], 20)
        instantaneous_frequency_std_hz = float(np.std(instantaneous_frequency[reliable]))
    else:
        instantaneous_frequency_std_hz = 0.0

    probability = power / np.sum(power)
    spectral_entropy = float(
        -np.sum(probability * np.log(probability + 1e-30)) / np.log(len(probability))
    )
    indices, _ = find_peaks(
        power_db,
        height=median_db + 8,
        prominence=4,
        distance=max(1, round(200 / bin_width_hz)),
    )
    features = ClassificationFeatures(
        analysis_bandwidth_hz=analysis_bandwidth_hz,
        bin_width_hz=bin_width_hz,
        peak_above_median_db=float(np.max(power_db) - median_db),
        carrier_prominence_db=carrier_prominence_db,
        sideband_imbalance_db=sideband_imbalance_db,
        occupied_bandwidth_hz=occupied_bandwidth_hz,
        envelope_coefficient_of_variation=envelope_cv,
        instantaneous_frequency_std_hz=instantaneous_frequency_std_hz,
        spectral_entropy=spectral_entropy,
        significant_peak_count=int(len(indices)),
        dominant_offset_hz=dominant_offset_hz,
    )
    display_count = min(len(filtered), sample_rate_hz // 5)
    display_time = np.arange(display_count) / sample_rate_hz
    return (
        features,
        frequencies,
        power_db,
        display_time,
        instantaneous_frequency[:display_count],
    )


def classify_features(features: ClassificationFeatures) -> list[dict]:
    strength = _clip01((features.peak_above_median_db - 5) / 20)
    carrier = _clip01((features.carrier_prominence_db + 12) / 12)
    symmetry = float(np.exp(-abs(features.sideband_imbalance_db) / 5))
    upper_dominance = _clip01((features.sideband_imbalance_db - 3) / 15)
    lower_dominance = _clip01((-features.sideband_imbalance_db - 3) / 15)
    narrow = _clip01((1_500 - features.occupied_bandwidth_hz) / 1_300)
    voice_width = _clip01((features.occupied_bandwidth_hz - 500) / 2_000) * _clip01(
        (6_000 - features.occupied_bandwidth_hz) / 3_000
    )
    fm_width = _clip01((features.occupied_bandwidth_hz - 3_000) / 8_000) * _clip01(
        (30_000 - features.occupied_bandwidth_hz) / 15_000
    )
    envelope_variation = _clip01((features.envelope_coefficient_of_variation - 0.04) / 0.30)
    constant_envelope = _clip01((0.22 - features.envelope_coefficient_of_variation) / 0.18)
    frequency_motion = _clip01((features.instantaneous_frequency_std_hz - 250) / 2_500)
    low_entropy = _clip01((0.55 - features.spectral_entropy) / 0.35)
    high_entropy = _clip01((features.spectral_entropy - 0.35) / 0.45)
    multi_peak = _clip01((features.significant_peak_count - 2) / 8)

    raw_scores = {
        "cw": strength * (0.5 * narrow + 0.3 * carrier + 0.2 * low_entropy),
        "am": strength
        * (0.35 * carrier + 0.30 * symmetry + 0.25 * envelope_variation + 0.10 * (1 - narrow)),
        "usb": strength * (0.55 * upper_dominance + 0.30 * voice_width + 0.15 * (1 - carrier)),
        "lsb": strength * (0.55 * lower_dominance + 0.30 * voice_width + 0.15 * (1 - carrier)),
        "nfm": strength
        * (0.40 * constant_envelope + 0.30 * frequency_motion + 0.25 * fm_width + 0.05 * symmetry),
        "digital_or_unknown": (
            0.15 + 0.30 * high_entropy + 0.20 * multi_peak + 0.35 * (1 - strength)
        ),
    }
    total = sum(max(score, 0.001) for score in raw_scores.values())
    ranking = [
        {
            "label": label,
            "confidence": float(max(score, 0.001) / total),
            "raw_score": float(score),
        }
        for label, score in raw_scores.items()
    ]
    ranking.sort(key=lambda item: item["confidence"], reverse=True)
    return ranking


def save_classification_plot(
    path: Path,
    frequency_hz: int,
    frequencies: np.ndarray,
    power_db: np.ndarray,
    time_axis: np.ndarray,
    instantaneous_frequency: np.ndarray,
    ranking: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(frequencies / 1000, power_db, color="#1677b8", linewidth=0.8)
    axes[0].axvline(0, color="#555", linestyle="--", linewidth=0.8)
    axes[0].set_title(
        f"Classification Spectrum — {frequency_hz / 1e6:g} MHz — "
        f"best: {ranking[0]['label'].upper()} ({ranking[0]['confidence']:.0%})"
    )
    axes[0].set_xlabel("Offset from requested frequency (kHz)")
    axes[0].set_ylabel("Relative power (dB)")
    axes[0].grid(alpha=0.25)

    count = min(len(time_axis), len(instantaneous_frequency))
    axes[1].plot(
        time_axis[:count] * 1000,
        instantaneous_frequency[:count],
        color="#f36d2e",
        linewidth=0.6,
    )
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Instantaneous frequency (Hz)")
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_dict(features: ClassificationFeatures) -> dict:
    return asdict(features)
