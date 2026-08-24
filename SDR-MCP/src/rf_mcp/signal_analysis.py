from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, firwin, lfilter, resample_poly, sosfilt


AUDIO_SAMPLE_RATE = 48_000
SUPPORTED_MODES = ("am", "usb", "lsb", "cw", "nfm")
DEFAULT_BANDWIDTHS_HZ = {
    "am": 10_000,
    "usb": 3_000,
    "lsb": 3_000,
    "cw": 500,
    "nfm": 12_500,
}
BANDWIDTH_LIMITS_HZ = {
    "am": (2_000, 20_000),
    "usb": (1_000, 6_000),
    "lsb": (1_000, 6_000),
    "cw": (100, 2_000),
    "nfm": (5_000, 30_000),
}


@dataclass(frozen=True)
class SignalMetrics:
    relative_peak_db: float
    relative_noise_floor_db: float
    estimated_snr_db: float
    dominant_frequency_hz: float
    dominant_offset_hz: float
    occupied_bandwidth_hz: float
    signal_present: bool
    signal_confidence: float
    duty_cycle_percent: float


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of: {', '.join(SUPPORTED_MODES)}")
    return normalized


def validate_bandwidth(mode: str, bandwidth_hz: int | None) -> int:
    mode = normalize_mode(mode)
    bandwidth_hz = DEFAULT_BANDWIDTHS_HZ[mode] if bandwidth_hz is None else int(bandwidth_hz)
    low, high = BANDWIDTH_LIMITS_HZ[mode]
    if not low <= bandwidth_hz <= high:
        raise ValueError(f"{mode} bandwidth must be between {low:,} and {high:,} Hz")
    return bandwidth_hz


def downconvert(
    iq: np.ndarray,
    sample_rate_hz: int,
    offset_hz: float,
) -> np.ndarray:
    phase = -2j * np.pi * offset_hz * np.arange(len(iq), dtype=np.float64) / sample_rate_hz
    return iq * np.exp(phase)


def _complex_lowpass(iq: np.ndarray, sample_rate_hz: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = min(float(cutoff_hz), sample_rate_hz * 0.45)
    taps = firwin(513, cutoff_hz, fs=sample_rate_hz, window="blackman")
    return lfilter(taps, 1.0, iq)


def _single_sideband(iq: np.ndarray, sample_rate_hz: int, bandwidth_hz: int, upper: bool) -> np.ndarray:
    spectrum = np.fft.fft(iq)
    frequencies = np.fft.fftfreq(len(iq), d=1 / sample_rate_hz)
    low_audio_hz = min(250.0, bandwidth_hz / 4)
    if upper:
        mask = (frequencies >= low_audio_hz) & (frequencies <= bandwidth_hz)
    else:
        mask = (frequencies <= -low_audio_hz) & (frequencies >= -bandwidth_hz)
    # A short cosine transition reduces ringing at the selected sideband edges.
    selected = np.zeros_like(spectrum)
    selected[mask] = spectrum[mask]
    return np.fft.ifft(selected)


def _resample_audio(audio: np.ndarray, input_rate_hz: int) -> np.ndarray:
    divisor = gcd(int(input_rate_hz), AUDIO_SAMPLE_RATE)
    return resample_poly(audio, AUDIO_SAMPLE_RATE // divisor, input_rate_hz // divisor)


def _condition_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    audio -= np.mean(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = 0.85 * audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def demodulate(
    baseband_iq: np.ndarray,
    sample_rate_hz: int,
    mode: str,
    bandwidth_hz: int,
    cw_tone_hz: int = 700,
) -> np.ndarray:
    mode = normalize_mode(mode)
    bandwidth_hz = validate_bandwidth(mode, bandwidth_hz)

    if mode == "am":
        filtered = _complex_lowpass(baseband_iq, sample_rate_hz, bandwidth_hz / 2)
        audio = np.abs(filtered)
    elif mode == "nfm":
        filtered = _complex_lowpass(baseband_iq, sample_rate_hz, bandwidth_hz / 2)
        audio = np.empty(len(filtered), dtype=np.float64)
        audio[0] = 0
        audio[1:] = np.angle(filtered[1:] * np.conj(filtered[:-1]))
    elif mode == "usb":
        audio = np.real(_single_sideband(baseband_iq, sample_rate_hz, bandwidth_hz, True))
    elif mode == "lsb":
        audio = np.real(_single_sideband(baseband_iq, sample_rate_hz, bandwidth_hz, False))
    else:  # CW: move a carrier at the requested frequency to an audible tone.
        filtered = _complex_lowpass(baseband_iq, sample_rate_hz, bandwidth_hz / 2)
        oscillator = np.exp(
            2j * np.pi * cw_tone_hz * np.arange(len(filtered), dtype=np.float64) / sample_rate_hz
        )
        audio = np.real(filtered * oscillator)

    return _condition_audio(_resample_audio(audio, sample_rate_hz))


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    wavfile.write(path, AUDIO_SAMPLE_RATE, pcm)


def _deemphasis(audio: np.ndarray, sample_rate_hz: int, microseconds: int) -> np.ndarray:
    if microseconds not in {50, 75}:
        raise ValueError("deemphasis_us must be 50 or 75")
    coefficient = float(np.exp(-1 / (sample_rate_hz * microseconds * 1e-6)))
    return lfilter([1 - coefficient], [1, -coefficient], audio, axis=0)


def demodulate_broadcast_fm(
    baseband_iq: np.ndarray,
    sample_rate_hz: int,
    *,
    deemphasis_us: int = 75,
    stereo: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """Demodulate an FM broadcast multiplex into 48 kHz mono or stereo audio."""
    if deemphasis_us not in {50, 75}:
        raise ValueError("deemphasis_us must be 50 or 75")
    composite_rate = 192_000
    filtered = _complex_lowpass(baseband_iq, sample_rate_hz, 100_000)
    discriminator = np.concatenate(
        ([0.0], np.angle(filtered[1:] * np.conj(filtered[:-1])))
    )
    composite = resample_poly(discriminator, composite_rate, int(sample_rate_hz))
    composite -= np.mean(composite)
    time_axis = np.arange(len(composite), dtype=np.float64) / composite_rate

    pilot_reference = np.exp(-2j * np.pi * 19_000 * time_axis)
    pilot_phasor = np.mean(composite * pilot_reference)
    pilot_amplitude = float(2 * np.abs(pilot_phasor))
    composite_rms = float(np.sqrt(np.mean(composite**2)) + 1e-15)
    pilot_to_composite_db = float(20 * np.log10((pilot_amplitude + 1e-15) / composite_rms))
    pilot_detected = bool(pilot_to_composite_db >= -25)

    audio_filter = butter(8, 15_000, btype="lowpass", fs=composite_rate, output="sos")
    mono = sosfilt(audio_filter, composite)
    stereo_used = bool(stereo and pilot_detected)
    if stereo_used:
        pilot_phase = float(np.angle(pilot_phasor))
        subcarrier = 2 * np.cos(2 * np.pi * 38_000 * time_axis + 2 * pilot_phase)
        difference = sosfilt(audio_filter, composite * subcarrier)
        multiplex_audio = np.column_stack((mono + difference, mono - difference))
    else:
        multiplex_audio = mono[:, None]

    multiplex_audio = _deemphasis(multiplex_audio, composite_rate, deemphasis_us)
    audio = resample_poly(multiplex_audio, AUDIO_SAMPLE_RATE, composite_rate, axis=0)
    audio -= np.mean(audio, axis=0, keepdims=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = 0.92 * audio / peak
    audio = np.clip(audio, -1, 1).astype(np.float32)

    frequencies = np.fft.rfftfreq(len(composite), 1 / composite_rate)
    spectrum = np.abs(np.fft.rfft(composite * np.hanning(len(composite)))) ** 2
    rds_mask = (frequencies >= 54_000) & (frequencies <= 60_000)
    multiplex_mask = frequencies <= 76_000
    rds_fraction = float(
        np.sum(spectrum[rds_mask]) / max(np.sum(spectrum[multiplex_mask]), 1e-30)
    )
    rds_band_relative_db = float(10 * np.log10(max(rds_fraction, 1e-30)))
    metrics = {
        "audio_channels": int(audio.shape[1]),
        "stereo_requested": bool(stereo),
        "stereo_detected": pilot_detected,
        "stereo_used": stereo_used,
        "pilot_frequency_hz": 19_000,
        "pilot_amplitude_relative": pilot_amplitude,
        "pilot_to_composite_rms_db": pilot_to_composite_db,
        "rds_subcarrier_frequency_hz": 57_000,
        "rds_band_power_relative_db": rds_band_relative_db,
        "rds_candidate_detected": bool(pilot_detected and rds_band_relative_db >= -35),
        "deemphasis_us": deemphasis_us,
        "audio_sample_rate_hz": AUDIO_SAMPLE_RATE,
    }
    diagnostic = {
        "composite_sample_rate_hz": composite_rate,
        "composite": composite,
        "frequencies_hz": frequencies,
        "spectrum_power": spectrum,
    }
    return audio, metrics, diagnostic


def save_broadcast_fm_plot(path: Path, center_frequency_hz: int, diagnostic: dict) -> None:
    rate = diagnostic["composite_sample_rate_hz"]
    composite = diagnostic["composite"]
    frequencies = diagnostic["frequencies_hz"]
    spectrum = diagnostic["spectrum_power"]
    spectrum_db = 10 * np.log10(spectrum + 1e-30)
    spectrum_db -= np.max(spectrum_db)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    display = min(len(composite), rate // 5)
    axes[0].plot(np.arange(display) / rate * 1_000, composite[:display], linewidth=0.6)
    axes[0].set_title(f"Broadcast FM multiplex — {center_frequency_hz / 1e6:.3f} MHz")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("FM discriminator")
    axes[0].grid(alpha=0.25)
    mask = frequencies <= 80_000
    axes[1].plot(frequencies[mask] / 1_000, spectrum_db[mask], linewidth=0.7)
    for frequency, label in ((19_000, "pilot"), (38_000, "stereo"), (57_000, "RDS")):
        axes[1].axvline(frequency / 1_000, linestyle="--", linewidth=0.8, label=label)
    axes[1].set_xlabel("Multiplex frequency (kHz)")
    axes[1].set_ylabel("Relative power (dB)")
    axes[1].set_ylim(-100, 3)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def measure_signal(
    frequencies_hz: np.ndarray,
    relative_power_db: np.ndarray,
    target_frequency_hz: int,
    bandwidth_hz: int,
    baseband_iq: np.ndarray,
    sample_rate_hz: int,
) -> SignalMetrics:
    offsets = frequencies_hz - target_frequency_hz
    signal_mask = np.abs(offsets) <= bandwidth_hz / 2
    noise_mask = (np.abs(offsets) >= bandwidth_hz * 1.5) & (np.abs(offsets) <= bandwidth_hz * 4)
    if not np.any(signal_mask):
        raise ValueError("FFT resolution does not contain the requested signal bandwidth")
    if not np.any(noise_mask):
        noise_mask = ~signal_mask

    signal_values = relative_power_db[signal_mask]
    signal_frequencies = frequencies_hz[signal_mask]
    peak_index = int(np.argmax(signal_values))
    peak_db = float(signal_values[peak_index])
    noise_floor_db = float(np.median(relative_power_db[noise_mask]))
    snr_db = max(0.0, peak_db - noise_floor_db)

    linear = 10 ** (signal_values / 10)
    order = np.argsort(signal_frequencies)
    sorted_frequency = signal_frequencies[order]
    cumulative = np.cumsum(linear[order])
    if cumulative[-1] > 0:
        lower_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.005))
        upper_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.995))
        upper_index = min(upper_index, len(sorted_frequency) - 1)
        occupied_bandwidth_hz = float(sorted_frequency[upper_index] - sorted_frequency[lower_index])
    else:
        occupied_bandwidth_hz = 0.0

    block_size = max(1, sample_rate_hz // 20)
    block_count = len(baseband_iq) // block_size
    if block_count:
        blocks = baseband_iq[: block_count * block_size].reshape(block_count, block_size)
        block_power_db = 10 * np.log10(np.mean(np.abs(blocks) ** 2, axis=1) + 1e-30)
        spread_db = float(np.percentile(block_power_db, 90) - np.percentile(block_power_db, 10))
        if spread_db < 3.0:
            duty_cycle = 100.0 if snr_db >= 6.0 else 0.0
        else:
            threshold = float(np.percentile(block_power_db, 20) + 3.0)
            duty_cycle = float(100 * np.mean(block_power_db > threshold))
    else:
        duty_cycle = 0.0

    confidence = float(np.clip((snr_db - 3) / 17, 0, 1))
    dominant_frequency_hz = float(signal_frequencies[peak_index])
    return SignalMetrics(
        relative_peak_db=peak_db,
        relative_noise_floor_db=noise_floor_db,
        estimated_snr_db=snr_db,
        dominant_frequency_hz=dominant_frequency_hz,
        dominant_offset_hz=dominant_frequency_hz - target_frequency_hz,
        occupied_bandwidth_hz=occupied_bandwidth_hz,
        signal_present=snr_db >= 6.0,
        signal_confidence=confidence,
        duty_cycle_percent=duty_cycle,
    )


def save_audio_spectrum(path: Path, audio: np.ndarray, mode: str, frequency_hz: int) -> None:
    fft_size = min(16_384, 2 ** int(np.floor(np.log2(max(1024, len(audio))))))
    block = audio[:fft_size]
    window = np.hanning(len(block))
    spectrum = np.fft.rfft(block * window)
    frequencies = np.fft.rfftfreq(len(block), d=1 / AUDIO_SAMPLE_RATE)
    level = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    level -= np.max(level)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frequencies, level, color="#f36d2e", linewidth=0.8)
    ax.set_xlim(0, min(12_000, AUDIO_SAMPLE_RATE / 2))
    ax.set_ylim(-100, 5)
    ax.set_title(f"{mode.upper()} Audio Spectrum — {frequency_hz / 1e6:g} MHz")
    ax.set_xlabel("Audio frequency (Hz)")
    ax.set_ylabel("Relative level (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
