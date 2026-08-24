from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .plotting import lazy_pyplot as plt
import numpy as np
from .lazy_imports import resample_poly


MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6",
    "--...": "7", "---..": "8", "----.": "9", ".-.-.-": ".",
    "--..--": ",", "..--..": "?", "-..-.": "/", "-....-": "-",
    ".----.": "'", "-.--.": "(", "-.--.-": ")", ".-..-.": '"',
    "---...": ":", "-.-.-.": ";", "-...-": "=", ".-.-.": "+",
    "..--.-": "_", "...-..-": "$", ".--.-.": "@",
}

ITA2_LETTERS = {
    0: "", 1: "E", 2: "\n", 3: "A", 4: " ", 5: "S", 6: "I", 7: "U",
    8: "\r", 9: "D", 10: "R", 11: "J", 12: "N", 13: "F", 14: "C",
    15: "K", 16: "T", 17: "Z", 18: "L", 19: "W", 20: "H", 21: "Y",
    22: "P", 23: "Q", 24: "O", 25: "B", 26: "G", 28: "M", 29: "X", 30: "V",
}
ITA2_FIGURES = {
    0: "", 1: "3", 2: "\n", 3: "-", 4: " ", 5: "'", 6: "8", 7: "7",
    8: "\r", 9: "$", 10: "4", 11: "'", 12: ",", 13: "!", 14: ":",
    15: "(", 16: "5", 17: '"', 18: ")", 19: "2", 20: "#", 21: "6",
    22: "0", 23: "1", 24: "9", 25: "?", 26: "&", 28: ".", 29: "/", 30: ";",
}

PSK31_VARICODE = {
    "1": " ", "111111111": "!", "101011111": '"', "111110101": "#",
    "111011011": "$", "1011010101": "%", "1010111011": "&",
    "101111111": "'", "11111011": "(", "11110111": ")",
    "101101111": "*", "111011111": "+", "1110101": ",", "110101": "-",
    "1010111": ".", "110101111": "/", "10110111": "0", "10111101": "1",
    "11101101": "2", "11111111": "3", "101110111": "4", "101011011": "5",
    "101101011": "6", "110101101": "7", "110101011": "8", "110110111": "9",
    "11110101": ":", "110111101": ";", "111101101": "<", "1010101": "=",
    "111010111": ">", "1010101111": "?", "1010111101": "@",
    "1111101": "A", "11101011": "B", "10101101": "C", "10110101": "D",
    "1110111": "E", "11011011": "F", "11111101": "G", "101010101": "H",
    "1111111": "I", "111111101": "J", "101111101": "K", "11010111": "L",
    "10111011": "M", "11011101": "N", "10101011": "O", "11010101": "P",
    "111011101": "Q", "10101111": "R", "1101111": "S", "1101101": "T",
    "101010111": "U", "110110101": "V", "101011101": "W", "101110101": "X",
    "101111011": "Y", "1010101101": "Z", "1011": "a", "1011111": "b",
    "101111": "c", "101101": "d", "11": "e", "111101": "f",
    "1011011": "g", "101011": "h", "1101": "i", "111101011": "j",
    "10111111": "k", "11011": "l", "111011": "m", "1111": "n",
    "111": "o", "111111": "p", "110111111": "q", "10101": "r",
    "10111": "s", "101": "t", "110111": "u", "1111011": "v",
    "1101011": "w", "11011111": "x", "1011101": "y", "111010101": "z",
    "11101": "\n", "11111": "\r",
}
LTRS = 31
FIGS = 27


def _runs(values: np.ndarray) -> list[tuple[bool, int]]:
    if not len(values):
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(values)]))
    return [(bool(values[start]), int(stop - start)) for start, stop in zip(bounds[:-1], bounds[1:])]


def decode_cw(iq: np.ndarray, sample_rate_hz: int, *, wpm: float | None = None) -> dict:
    if wpm is not None and not 5 <= float(wpm) <= 60:
        raise ValueError("CW speed must be from 5 through 60 WPM")
    target_rate = 2_000
    envelope = np.abs(resample_poly(iq, target_rate, int(sample_rate_hz)))
    window = max(1, round(target_rate * 0.01))
    envelope = np.convolve(envelope, np.ones(window) / window, mode="same")
    low, high = np.percentile(envelope, [20, 90])
    threshold = float(low + 0.45 * (high - low))
    keyed = envelope > threshold
    runs = _runs(keyed)
    mark_seconds = np.array([length / target_rate for state, length in runs if state])
    if not len(mark_seconds) or high <= low * 1.05:
        return {
            "text": "", "tokens": [], "confidence": 0.0, "dot_seconds": None,
            "estimated_wpm": None, "unknown_count": 0, "threshold": threshold,
            "diagnostic": {"sample_rate_hz": target_rate, "envelope": envelope, "keyed": keyed},
        }
    if wpm is None:
        ordered = np.sort(mark_seconds)
        shortest = ordered[: max(1, (len(ordered) + 1) // 2)]
        dot = float(np.median(shortest))
    else:
        dot = 1.2 / float(wpm)
    dot = max(0.015, min(dot, 0.25))
    tokens: list[str] = []
    symbol = ""
    for state, length in runs:
        units = length / target_rate / dot
        if state:
            symbol += "." if units < 2.0 else "-"
        elif symbol:
            if units >= 6:
                tokens.extend([symbol, "/"])
                symbol = ""
            elif units >= 2:
                tokens.append(symbol)
                symbol = ""
    if symbol:
        tokens.append(symbol)
    decoded = []
    unknown = 0
    for token in tokens:
        if token == "/":
            if decoded and decoded[-1] != " ":
                decoded.append(" ")
        else:
            character = MORSE.get(token, "�")
            unknown += character == "�"
            decoded.append(character)
    character_count = sum(token != "/" for token in tokens)
    timing_quality = min(1.0, float((high - low) / max(high, 1e-12)))
    confidence = timing_quality * (1 - unknown / max(character_count, 1))
    return {
        "text": "".join(decoded).strip(), "tokens": tokens,
        "confidence": round(float(confidence), 4), "dot_seconds": dot,
        "estimated_wpm": 1.2 / dot, "unknown_count": unknown,
        "threshold": threshold,
        "diagnostic": {"sample_rate_hz": target_rate, "envelope": envelope, "keyed": keyed},
    }


def _decode_ita2(codes: list[int]) -> str:
    figures = False
    output = []
    for code in codes:
        if code == LTRS:
            figures = False
        elif code == FIGS:
            figures = True
        else:
            output.append((ITA2_FIGURES if figures else ITA2_LETTERS).get(code, "�"))
    return "".join(output).replace("\r", "").strip()


def _rtty_candidate(bits: np.ndarray, samples_per_symbol: float) -> tuple[list[int], int]:
    minimum_stable = max(1, round(samples_per_symbol * 0.55))
    edges = np.flatnonzero((bits[:-1] == 1) & (bits[1:] == 0)) + 1
    codes = []
    framing_errors = 0
    next_allowed = 0
    for edge in edges:
        if edge < next_allowed or edge < minimum_stable:
            continue
        if not np.all(bits[max(0, edge - minimum_stable):edge] == 1):
            continue
        sample_points = [round(edge + samples_per_symbol * offset) for offset in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)]
        if sample_points[-1] >= len(bits):
            break
        data = [int(bits[index]) for index in sample_points[:5]]
        stop = int(bits[sample_points[5]])
        if stop != 1:
            framing_errors += 1
            continue
        codes.append(sum(bit << index for index, bit in enumerate(data)))
        next_allowed = round(edge + 7.0 * samples_per_symbol)
    return codes, framing_errors


def decode_rtty(
    iq: np.ndarray,
    sample_rate_hz: int,
    *,
    baud: float = 45.45,
    shift_hz: float = 170.0,
    polarity: str = "auto",
) -> dict:
    baud = float(baud)
    shift_hz = float(shift_hz)
    polarity = polarity.strip().lower()
    if not 40 <= baud <= 300:
        raise ValueError("RTTY baud must be from 40 through 300")
    if not 80 <= shift_hz <= 1_000:
        raise ValueError("RTTY shift must be from 80 through 1000 Hz")
    if polarity not in {"auto", "normal", "reverse"}:
        raise ValueError("polarity must be auto, normal, or reverse")
    target_rate = max(2_000, round(baud * 64))
    baseband = resample_poly(iq, target_rate, int(sample_rate_hz))
    phase = np.angle(baseband[1:] * np.conj(baseband[:-1]))
    instantaneous_hz = np.concatenate(([0.0], phase * target_rate / (2 * np.pi)))
    window = max(1, round(target_rate / baud * 0.2))
    frequency = np.convolve(instantaneous_hz, np.ones(window) / window, mode="same")
    space_estimate, mark_estimate = np.percentile(frequency, [15, 85])
    center = float((space_estimate + mark_estimate) / 2)
    estimated_shift = float(mark_estimate - space_estimate)
    normal_bits = frequency >= center
    candidates = []
    choices = ("normal", "reverse") if polarity == "auto" else (polarity,)
    for choice in choices:
        bits = normal_bits if choice == "normal" else ~normal_bits
        codes, errors = _rtty_candidate(bits, target_rate / baud)
        text = _decode_ita2(codes)
        printable = sum(character.isprintable() or character in "\r\n" for character in text)
        score = printable - errors * 0.5
        candidates.append((score, choice, codes, errors, text, bits))
    _, selected, codes, errors, text, bits = max(candidates, key=lambda item: item[0])
    valid = len(codes)
    framing_confidence = valid / max(valid + errors, 1)
    shift_confidence = max(0.0, 1 - abs(estimated_shift - shift_hz) / max(shift_hz, 1))
    confidence = framing_confidence * (0.5 + 0.5 * shift_confidence)
    return {
        "text": text, "codes": codes, "confidence": round(float(confidence), 4),
        "character_count": len(text), "framing_error_count": errors,
        "baud": baud, "shift_hz": shift_hz, "selected_polarity": selected,
        "estimated_center_offset_hz": center, "estimated_shift_hz": estimated_shift,
        "diagnostic": {"sample_rate_hz": target_rate, "frequency_hz": frequency, "bits": bits},
    }


def decode_bpsk31(iq: np.ndarray, sample_rate_hz: int) -> dict:
    symbol_rate = 31.25
    target_rate = 2_000
    signal = resample_poly(iq, target_rate, int(sample_rate_hz))
    squared = signal * signal
    residual_hz = float(
        np.angle(np.sum(squared[1:] * np.conj(squared[:-1])))
        * target_rate / (4 * np.pi)
    )
    signal *= np.exp(-2j * np.pi * residual_hz * np.arange(len(signal)) / target_rate)
    samples_per_symbol = round(target_rate / symbol_rate)
    candidates = []
    for offset in range(samples_per_symbol):
        count = (len(signal) - offset) // samples_per_symbol
        if count < 3:
            continue
        symbols = signal[offset:offset + count * samples_per_symbol].reshape(
            count, samples_per_symbol
        ).mean(axis=1)
        differential = symbols[1:] * np.conj(symbols[:-1])
        bits = (np.real(differential) >= 0).astype(np.uint8)
        bit_text = "".join(str(int(bit)) for bit in bits)
        codes = [code for code in re.split("0{2,}", bit_text) if code and len(code) <= 10]
        decoded = [PSK31_VARICODE.get(code, "�") for code in codes]
        printable = sum(character != "�" for character in decoded)
        unknown = len(decoded) - printable
        magnitude = float(np.mean(np.abs(symbols)))
        candidates.append((printable - unknown * 1.5, magnitude, offset, bits, codes, decoded))
    if not candidates:
        return {
            "text": "", "varicode": [], "confidence": 0.0, "unknown_count": 0,
            "symbol_rate": symbol_rate, "estimated_frequency_offset_hz": residual_hz,
            "diagnostic": {"sample_rate_hz": target_rate, "phase_bits": np.array([])},
        }
    _, _, offset, bits, codes, decoded = max(candidates, key=lambda item: (item[1], item[0]))
    unknown = sum(character == "�" for character in decoded)
    confidence = (len(decoded) - unknown) / max(len(decoded), 1)
    return {
        "text": "".join(decoded).strip("\x00"), "varicode": codes,
        "confidence": round(float(confidence), 4), "unknown_count": unknown,
        "symbol_rate": symbol_rate, "timing_offset_samples": offset,
        "estimated_frequency_offset_hz": residual_hz,
        "diagnostic": {"sample_rate_hz": symbol_rate, "phase_bits": bits},
    }


def ax25_fcs(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        for bit_index in range(8):
            bit = (byte >> bit_index) & 1
            crc = (crc >> 1) ^ (0x8408 if (crc & 1) != bit else 0)
    return crc ^ 0xFFFF


def _destuff(bits: list[int]) -> list[int] | None:
    output = []
    ones = 0
    index = 0
    while index < len(bits):
        bit = bits[index]
        output.append(bit)
        if bit:
            ones += 1
            if ones > 5:
                return None
        else:
            if ones == 5:
                output.pop()
            ones = 0
        index += 1
    return output


def _parse_ax25_frame(frame: bytes, fcs_valid: bool) -> dict:
    addresses = []
    index = 0
    while index + 7 <= len(frame) and len(addresses) < 10:
        chunk = frame[index:index + 7]
        callsign = "".join(chr(byte >> 1) for byte in chunk[:6]).strip()
        ssid = (chunk[6] >> 1) & 0x0F
        addresses.append(f"{callsign}-{ssid}" if ssid else callsign)
        index += 7
        if chunk[6] & 1:
            break
    control = frame[index] if index < len(frame) else None
    pid = frame[index + 1] if index + 1 < len(frame) else None
    info = frame[index + 2:] if index + 2 <= len(frame) else b""
    return {
        "destination": addresses[0] if addresses else None,
        "source": addresses[1] if len(addresses) > 1 else None,
        "digipeaters": addresses[2:], "control": control, "pid": pid,
        "information_text": info.decode("latin-1", errors="replace"),
        "information_hex": info.hex(), "frame_hex": frame.hex(), "fcs_valid": fcs_valid,
    }


def _decode_hdlc_frames(bits: np.ndarray) -> tuple[list[dict], int]:
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    values = list(map(int, bits))
    positions = [
        index for index in range(max(0, len(values) - 7))
        if values[index:index + 8] == flag
    ]
    frames = []
    for start, stop in zip(positions[:-1], positions[1:]):
        stuffed = values[start + 8:stop]
        if not stuffed:
            continue
        clean = _destuff(stuffed)
        if clean is None or len(clean) < 24 or len(clean) % 8:
            continue
        octets = bytes(
            sum(clean[index + bit] << bit for bit in range(8))
            for index in range(0, len(clean), 8)
        )
        payload, received_fcs = octets[:-2], int.from_bytes(octets[-2:], "little")
        frames.append(_parse_ax25_frame(payload, ax25_fcs(payload) == received_fcs))
    return frames, len(positions)


def decode_ax25_afsk1200(iq: np.ndarray, sample_rate_hz: int) -> dict:
    target_rate = 9_600
    baseband = resample_poly(iq, target_rate, int(sample_rate_hz))
    audio = np.concatenate(
        ([0.0], np.angle(baseband[1:] * np.conj(baseband[:-1])))
    )
    samples_per_symbol = target_rate // 1_200
    time = np.arange(samples_per_symbol) / target_rate
    mark = np.exp(-2j * np.pi * 1_200 * time)
    space = np.exp(-2j * np.pi * 2_200 * time)
    candidates = []
    for offset in range(samples_per_symbol):
        count = (len(audio) - offset) // samples_per_symbol
        chunks = audio[offset:offset + count * samples_per_symbol].reshape(
            count, samples_per_symbol
        )
        tone = (np.abs(chunks @ space) > np.abs(chunks @ mark)).astype(np.uint8)
        bits = np.ones(len(tone) - 1, dtype=np.uint8)
        bits[tone[1:] != tone[:-1]] = 0
        _, flags = _decode_hdlc_frames(bits)
        candidates.append((flags, offset, bits, tone))
    flag_count, offset, bits, tone = max(candidates, key=lambda item: item[0])
    frames, flag_count = _decode_hdlc_frames(bits)
    valid = sum(frame["fcs_valid"] for frame in frames)
    return {
        "frames": frames, "frame_count": len(frames), "valid_fcs_count": valid,
        "confidence": round(valid / max(len(frames), 1), 4),
        "baud": 1_200, "mark_hz": 1_200, "space_hz": 2_200,
        "timing_offset_samples": offset, "flag_count": flag_count,
        "diagnostic": {"sample_rate_hz": 1_200, "symbol_bits": bits, "tone_state": tone},
    }


def decode_ax25_g3ruh9600(iq: np.ndarray, sample_rate_hz: int) -> dict:
    """Decode scrambled NRZI AX.25 carried as 9600-baud direct FSK.

    This implements the common G3RUH self-synchronizing x^17+x^12+1
    descrambler and searches all four symbol phases at 38.4 ksample/s.
    """
    target_rate, baud = 38_400, 9_600
    baseband = resample_poly(iq, target_rate, int(sample_rate_hz))
    discriminator = np.concatenate(
        ([0.0], np.angle(baseband[1:] * np.conj(baseband[:-1])))
    )
    discriminator -= np.mean(discriminator) if len(discriminator) else 0
    samples_per_symbol = target_rate // baud
    candidates = []
    for offset in range(samples_per_symbol):
        count = (len(discriminator) - offset) // samples_per_symbol
        if count < 32:
            continue
        symbols = discriminator[
            offset:offset + count * samples_per_symbol
        ].reshape(count, samples_per_symbol).mean(axis=1)
        for polarity in (1, -1):
            scrambled = ((symbols * polarity) >= 0).astype(np.uint8)
            levels = scrambled.copy()
            if len(levels) > 17:
                levels[17:] = (
                    scrambled[17:] ^ scrambled[5:-12] ^ scrambled[:-17]
                )
            bits = np.ones(max(0, len(levels) - 1), dtype=np.uint8)
            if len(levels) > 1:
                bits[levels[1:] != levels[:-1]] = 0
            frames, flags = _decode_hdlc_frames(bits)
            valid = sum(frame["fcs_valid"] for frame in frames)
            candidates.append((valid, len(frames), flags, offset, polarity,
                               bits, scrambled, frames))
    if not candidates:
        return {
            "frames": [], "frame_count": 0, "valid_fcs_count": 0,
            "confidence": 0.0, "baud": baud, "flag_count": 0,
            "diagnostic": {"sample_rate_hz": baud,
                           "symbol_bits": np.array([], dtype=np.uint8),
                           "tone_state": np.array([], dtype=np.uint8)},
        }
    valid, _, flags, offset, polarity, bits, scrambled, frames = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return {
        "frames": frames, "frame_count": len(frames), "valid_fcs_count": valid,
        "confidence": round(valid / max(len(frames), 1), 4), "baud": baud,
        "line_code": "NRZI", "scrambler": "x^17+x^12+1",
        "timing_offset_samples": offset, "polarity": polarity,
        "flag_count": flags,
        "diagnostic": {"sample_rate_hz": baud, "symbol_bits": bits,
                       "tone_state": scrambled[1:]},
    }


def save_decode_plot(path: Path, mode: str, result: dict) -> None:
    diagnostic = result["diagnostic"]
    rate = diagnostic["sample_rate_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    if mode == "cw":
        values = diagnostic["envelope"]
        axes[0].plot(np.arange(len(values)) / rate, values, linewidth=0.7)
        axes[0].axhline(result["threshold"], color="#f36d2e", linestyle="--")
        axes[0].set_ylabel("CW envelope")
        bits = diagnostic["keyed"]
    elif mode == "rtty":
        values = diagnostic["frequency_hz"]
        axes[0].plot(np.arange(len(values)) / rate, values, linewidth=0.6)
        axes[0].axhline(result["estimated_center_offset_hz"], color="#f36d2e", linestyle="--")
        axes[0].set_ylabel("Instantaneous Hz")
        bits = diagnostic["bits"]
    elif mode == "bpsk31":
        bits = diagnostic["phase_bits"]
        values = bits
        axes[0].step(np.arange(len(bits)) / rate, bits, where="post")
        axes[0].set_ylabel("Differential bit")
    else:
        bits = diagnostic["symbol_bits"]
        values = diagnostic["tone_state"]
        axes[0].step(np.arange(len(values)) / rate, values, where="post")
        axes[0].set_ylabel("AFSK tone state")
    axes[0].set_title(f"{mode.upper()} decoder diagnostic")
    axes[0].grid(alpha=0.25)
    axes[1].step(np.arange(len(bits)) / rate, bits.astype(int), where="post")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Key/symbol state")
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)
