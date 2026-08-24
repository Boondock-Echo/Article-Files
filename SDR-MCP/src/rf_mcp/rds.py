from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt


RDS_BIT_RATE = 1_187.5
RDS_GENERATOR = 0x5B9
OFFSET_WORDS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C_prime": 0x350, "D": 0x1B4}


def rds_remainder(value: int, bit_count: int = 26) -> int:
    working = int(value)
    for bit in range(bit_count - 1, 9, -1):
        if working & (1 << bit):
            working ^= RDS_GENERATOR << (bit - 10)
    return working & 0x3FF


def make_rds_block(data_word: int, offset_name: str) -> int:
    data_word &= 0xFFFF
    base = data_word << 10
    check = rds_remainder(base) ^ OFFSET_WORDS[offset_name]
    return base | check


def _block_at(bits: np.ndarray, start: int) -> tuple[int, str | None]:
    value = 0
    for bit in bits[start:start + 26]:
        value = (value << 1) | int(bit)
    remainder = rds_remainder(value)
    name = next((key for key, offset in OFFSET_WORDS.items() if offset == remainder), None)
    return value >> 10, name


def _expected_block(bits: np.ndarray, start: int, expected: set[str]) -> tuple[int, str | None, int]:
    data, name = _block_at(bits, start)
    if name in expected:
        return data, name, 0
    original = bits[start:start + 26].copy()
    for position in range(26):
        corrected = original.copy()
        corrected[position] ^= 1
        value = 0
        for bit in corrected:
            value = (value << 1) | int(bit)
        remainder = rds_remainder(value)
        name = next((key for key in expected if OFFSET_WORDS[key] == remainder), None)
        if name:
            return value >> 10, name, 1
    return data, None, 0


def _find_groups(bits: np.ndarray) -> list[dict]:
    groups = []
    index = 0
    while index + 104 <= len(bits):
        a, name_a = _block_at(bits, index)
        if name_a != "A":
            index += 1
            continue
        b, name_b, corrected_b = _expected_block(bits, index + 26, {"B"})
        c, name_c, corrected_c = _expected_block(bits, index + 52, {"C", "C_prime"})
        d, name_d, corrected_d = _expected_block(bits, index + 78, {"D"})
        if name_b == "B" and name_c in {"C", "C_prime"} and name_d == "D":
            groups.append(
                {
                    "bit_offset": index,
                    "block_names": [name_a, name_b, name_c, name_d],
                    "blocks": [a, b, c, d],
                    "corrected_bit_count": corrected_b + corrected_c + corrected_d,
                }
            )
            index += 104
        else:
            index += 1
    return groups


def _mjd_to_date(mjd: int) -> datetime:
    return datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=mjd)


def parse_rds_groups(groups: list[dict]) -> dict:
    ps = [" "] * 8
    ps_segments = set()
    radiotext = [" "] * 64
    rt_segments = set()
    rt_ab = None
    ptyn = [" "] * 8
    ptyn_segments = set()
    alternative_frequencies = set()
    parsed_groups = []
    station: dict = {}
    for group in groups:
        a, b, c, d = group["blocks"]
        group_type = (b >> 12) & 0xF
        version = "B" if (b >> 11) & 1 else "A"
        parsed = {
            **group,
            "group_type": f"{group_type}{version}",
            "pi_code": f"{a:04X}",
            "traffic_program": bool((b >> 10) & 1),
            "program_type": (b >> 5) & 0x1F,
        }
        station.update(
            {
                "pi_code": parsed["pi_code"],
                "traffic_program": parsed["traffic_program"],
                "program_type": parsed["program_type"],
            }
        )
        if group_type == 0:
            segment = b & 0x3
            characters = [chr((d >> 8) & 0xFF), chr(d & 0xFF)]
            ps[segment * 2:segment * 2 + 2] = characters
            ps_segments.add(segment)
            parsed.update(
                {
                    "traffic_announcement": bool((b >> 4) & 1),
                    "music_speech": bool((b >> 3) & 1),
                    "decoder_information_bit": bool((b >> 2) & 1),
                    "program_service_segment": segment,
                    "program_service_characters": "".join(characters),
                }
            )
            station["traffic_announcement"] = parsed["traffic_announcement"]
            station["music_speech"] = parsed["music_speech"]
            if version == "A":
                for code in ((c >> 8) & 0xFF, c & 0xFF):
                    if 1 <= code <= 204:
                        alternative_frequencies.add(round(87.5 + code / 10, 1))
        elif group_type == 2:
            ab = (b >> 4) & 1
            if rt_ab is not None and ab != rt_ab:
                radiotext = [" "] * 64
                rt_segments.clear()
            rt_ab = ab
            segment = b & 0xF
            if version == "A":
                characters = [chr((c >> 8) & 0xFF), chr(c & 0xFF), chr((d >> 8) & 0xFF), chr(d & 0xFF)]
                position = segment * 4
            else:
                characters = [chr((d >> 8) & 0xFF), chr(d & 0xFF)]
                position = segment * 2
            radiotext[position:position + len(characters)] = characters
            rt_segments.add(segment)
            parsed.update(
                {
                    "text_ab_flag": ab,
                    "radiotext_segment": segment,
                    "radiotext_characters": "".join(characters),
                }
            )
        elif group_type == 4 and version == "A":
            mjd = ((b & 0x3) << 15) | (c >> 1)
            hour = ((c & 1) << 4) | ((d >> 12) & 0xF)
            minute = (d >> 6) & 0x3F
            offset_half_hours = d & 0x1F
            offset_minutes = offset_half_hours * 30 * (-1 if (d >> 5) & 1 else 1)
            if hour < 24 and minute < 60:
                utc_time = _mjd_to_date(mjd).replace(hour=hour, minute=minute)
                parsed["clock_time_utc"] = utc_time.isoformat()
                parsed["local_offset_minutes"] = offset_minutes
                station["clock_time_utc"] = parsed["clock_time_utc"]
                station["local_offset_minutes"] = offset_minutes
        elif group_type == 10 and version == "A":
            segment = b & 1
            characters = [chr((c >> 8) & 0xFF), chr(c & 0xFF), chr((d >> 8) & 0xFF), chr(d & 0xFF)]
            ptyn[segment * 4:segment * 4 + 4] = characters
            ptyn_segments.add(segment)
            parsed["program_type_name_segment"] = segment
            parsed["program_type_name_characters"] = "".join(characters)
        parsed_groups.append(parsed)
    station["program_service"] = "".join(ps).rstrip()
    station["program_service_complete"] = len(ps_segments) == 4
    station["radiotext"] = "".join(radiotext).split("\r", 1)[0].rstrip()
    station["radiotext_segments_received"] = sorted(rt_segments)
    station["alternative_frequencies_mhz"] = sorted(alternative_frequencies)
    station["program_type_name"] = "".join(ptyn).rstrip()
    station["program_type_name_complete"] = len(ptyn_segments) == 2
    return {"station": station, "groups": parsed_groups}


def decode_rds(composite: np.ndarray, sample_rate_hz: int) -> dict:
    sample_rate_hz = int(sample_rate_hz)
    time_axis = np.arange(len(composite), dtype=np.float64) / sample_rate_hz
    pilot_phasor = np.mean(composite * np.exp(-2j * np.pi * 19_000 * time_axis))
    pilot_phase = float(np.angle(pilot_phasor))
    bandpass = butter(6, [54_000, 60_000], btype="bandpass", fs=sample_rate_hz, output="sos")
    rds_band = sosfilt(bandpass, composite)
    baseband = rds_band * 2 * np.cos(2 * np.pi * 57_000 * time_axis + 3 * pilot_phase)
    lowpass = butter(6, 2_400, btype="lowpass", fs=sample_rate_hz, output="sos")
    baseband = sosfilt(lowpass, baseband)
    target_rate = 19_000
    chips = resample_poly(baseband, target_rate, sample_rate_hz)
    samples_per_bit = 16
    candidates = []
    for timing in range(samples_per_bit):
        count = (len(chips) - timing) // samples_per_bit
        if count < 105:
            continue
        symbols = chips[timing:timing + count * samples_per_bit].reshape(count, samples_per_bit)
        biphase = symbols[:, :8].mean(axis=1) - symbols[:, 8:].mean(axis=1)
        encoded = biphase >= 0
        differential = np.logical_xor(encoded[1:], encoded[:-1]).astype(np.uint8)
        for inverted in (False, True):
            bits = 1 - differential if inverted else differential
            groups = _find_groups(bits)
            candidates.append((len(groups), float(np.mean(np.abs(biphase))), timing, inverted, bits, groups))
    if not candidates:
        return {
            "group_count": 0, "valid_block_count": 0, "block_error_count": 0,
            "confidence": 0.0, "station": {}, "groups": [],
        }
    _, _, timing, inverted, bits, groups = max(candidates, key=lambda item: (item[0], item[1]))
    parsed = parse_rds_groups(groups)
    return {
        "group_count": len(groups),
        "valid_block_count": len(groups) * 4,
        "block_error_count": sum(group["corrected_bit_count"] for group in groups),
        "confidence": (
            round(1 - sum(group["corrected_bit_count"] for group in groups) / (len(groups) * 104), 4)
            if groups else 0.0
        ),
        "bit_rate": RDS_BIT_RATE,
        "timing_offset_samples": timing,
        "differential_polarity_inverted": inverted,
        "station": parsed["station"],
        "groups": parsed["groups"],
        "raw_bit_count": len(bits),
    }
