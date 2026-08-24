from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


PSD_SCALE = "digital_dbfs_per_hz_v1"
DEFAULT_POWER_MEASUREMENT_BANDWIDTH_HZ = 10_000


@dataclass(frozen=True)
class Peak:
    frequency_hz: float
    relative_power_db: float
    above_noise_db: float
    prominence_db: float


def load_complex_float32(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 2:
        raise ValueError("IQ file contains no samples")
    if raw.size % 2:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def averaged_spectrum(
    iq: np.ndarray,
    center_frequency_hz: int,
    sample_rate_hz: int,
    fft_size: int = 16_384,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if fft_size < 1024 or fft_size > 262_144 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two from 1024 through 262144")
    block_count = len(iq) // fft_size
    if block_count < 1:
        raise ValueError(f"Capture is too short for fft_size={fft_size}")

    blocks = iq[: block_count * fft_size].reshape(block_count, fft_size)
    window = np.blackman(fft_size).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    if normalize:
        power_db -= np.max(power_db)
    frequencies = center_frequency_hz + np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1 / sample_rate_hz)
    )
    return frequencies, power_db


def averaged_psd_dbfs_per_hz(
    iq: np.ndarray,
    center_frequency_hz: int,
    sample_rate_hz: int,
    fft_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a window-corrected digital PSD referenced to complex magnitude 1.

    This is a repeatable digital-domain scale, not calibrated RF input power.
    Integrating the linear PSD over frequency approximates mean |IQ| squared.
    """
    if fft_size < 1024 or fft_size > 262_144 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two from 1024 through 262144")
    block_count = len(iq) // fft_size
    if block_count < 1:
        raise ValueError(f"Capture is too short for fft_size={fft_size}")
    blocks = iq[: block_count * fft_size].reshape(block_count, fft_size)
    window = np.blackman(fft_size).astype(np.float64)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    normalization = float(sample_rate_hz * np.sum(window**2))
    psd = np.mean(np.abs(spectra) ** 2, axis=0) / normalization
    psd_dbfs_hz = 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))
    frequencies = center_frequency_hz + np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1 / sample_rate_hz)
    )
    return frequencies, psd_dbfs_hz


def iq_level_metrics(iq: np.ndarray, clip_threshold: float = 0.999) -> dict:
    """Report digital headroom indicators for complex float IQ samples."""
    if iq.size == 0:
        raise ValueError("IQ array is empty")
    components = np.concatenate((np.abs(np.real(iq)), np.abs(np.imag(iq))))
    max_component = float(np.max(components))
    rms_magnitude = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    peak_magnitude = float(np.max(np.abs(iq)))
    clipped_fraction = float(np.mean(components >= clip_threshold))
    crest_factor_db = float(
        20 * np.log10(max(peak_magnitude, 1e-30) / max(rms_magnitude, 1e-30))
    )
    return {
        "reference": "complex_float_full_scale_component_1.0",
        "max_component_abs": max_component,
        "rms_magnitude": rms_magnitude,
        "peak_magnitude": peak_magnitude,
        "crest_factor_db": crest_factor_db,
        "clip_threshold": float(clip_threshold),
        "clipped_component_fraction": clipped_fraction,
        "overload_suspected": bool(clipped_fraction >= 1e-5),
    }


def integrate_psd_dbfs(
    frequencies_hz: np.ndarray,
    psd_dbfs_hz: np.ndarray,
    center_frequency_hz: float,
    bandwidth_hz: float,
) -> float:
    """Integrate a digital PSD over a fixed bandwidth and return dBFS."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if len(frequencies_hz) < 2 or len(frequencies_hz) != len(psd_dbfs_hz):
        raise ValueError("PSD arrays must have matching lengths of at least two")
    mask = np.abs(frequencies_hz - center_frequency_hz) <= bandwidth_hz / 2
    if not np.any(mask):
        raise ValueError("No PSD bins fall inside the integration bandwidth")
    bin_width_hz = float(np.median(np.diff(frequencies_hz)))
    power = float(np.sum(10 ** (psd_dbfs_hz[mask] / 10)) * abs(bin_width_hz))
    return float(10 * np.log10(max(power, np.finfo(float).tiny)))


def valid_passband_mask(
    frequencies: np.ndarray,
    center_frequency_hz: int,
    sample_rate_hz: int,
    edge_fraction: float = 0.12,
    dc_exclusion_hz: float = 1_500,
) -> np.ndarray:
    usable_half_width = (sample_rate_hz / 2) * (1 - edge_fraction)
    offsets = frequencies - center_frequency_hz
    return (np.abs(offsets) <= usable_half_width) & (np.abs(offsets) >= dc_exclusion_hz)


def analyze_peaks(
    frequencies: np.ndarray,
    power_db: np.ndarray,
    valid_mask: np.ndarray,
    threshold_above_noise_db: float = 8.0,
    minimum_spacing_hz: float = 500.0,
    max_peaks: int = 20,
) -> tuple[float, list[Peak]]:
    if not np.any(valid_mask):
        raise ValueError("No valid spectrum bins remain after passband filtering")
    noise_floor_db = float(np.median(power_db[valid_mask]))
    bin_width_hz = float(abs(frequencies[1] - frequencies[0]))
    minimum_bins = max(1, round(minimum_spacing_hz / bin_width_hz))
    candidate_power = power_db.copy()
    # A -inf mask makes SciPy report infinite prominence for peaks adjacent to
    # the usable-passband boundary. Keep masked bins finite and safely below
    # the detection threshold so every exported measurement remains finite.
    finite_floor = float(np.min(power_db[valid_mask]))
    candidate_power[~valid_mask] = finite_floor - 100.0
    indices, properties = find_peaks(
        candidate_power,
        height=noise_floor_db + threshold_above_noise_db,
        prominence=max(3.0, threshold_above_noise_db / 2),
        distance=minimum_bins,
    )
    prominences = properties.get("prominences", np.zeros(indices.size))
    peaks = [
        Peak(
            frequency_hz=float(frequencies[index]),
            relative_power_db=float(power_db[index]),
            above_noise_db=float(power_db[index] - noise_floor_db),
            prominence_db=float(prominence),
        )
        for index, prominence in zip(indices, prominences, strict=True)
    ]
    peaks.sort(key=lambda peak: peak.relative_power_db, reverse=True)
    return noise_floor_db, peaks[:max_peaks]


def save_plot(
    path: Path,
    frequencies: np.ndarray,
    power_db: np.ndarray,
    valid_mask: np.ndarray,
    center_frequency_hz: int,
    noise_floor_db: float,
    peaks: list[Peak],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frequencies / 1e6, power_db, linewidth=0.8, color="#1677b8")
    ax.plot(
        frequencies[valid_mask] / 1e6,
        power_db[valid_mask],
        linewidth=0.8,
        color="#f36d2e",
        label="Analyzed passband",
    )
    ax.axhline(noise_floor_db, color="#555", linestyle="--", linewidth=1, label="Median noise")
    if peaks:
        ax.scatter(
            [peak.frequency_hz / 1e6 for peak in peaks],
            [peak.relative_power_db for peak in peaks],
            s=18,
            color="#d62728",
            zorder=3,
            label="Detected peaks",
        )
    ax.set_title(f"Airspy HF+ Spectrum Centered at {center_frequency_hz / 1e6:g} MHz")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Relative power (dB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def peak_dicts(peaks: list[Peak]) -> list[dict]:
    return [asdict(peak) for peak in peaks]
