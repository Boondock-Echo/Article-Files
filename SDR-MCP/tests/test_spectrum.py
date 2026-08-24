import numpy as np
import json
from scipy.signal import resample_poly
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rf_mcp.airspyhf import stream_iq_chunks, validate_duration, validate_frequency
import rf_mcp.airspyhf as airspyhf_module
from rf_mcp.spectrum import (
    analyze_peaks,
    averaged_psd_dbfs_per_hz,
    averaged_spectrum,
    integrate_psd_dbfs,
    iq_level_metrics,
    valid_passband_mask,
)
from rf_mcp.signal_analysis import (
    demodulate,
    demodulate_broadcast_fm,
    downconvert,
    measure_signal,
    validate_bandwidth,
)
from rf_mcp.monitoring import MonitorManager, MonitorSample
from rf_mcp.scanning import ScanJob, ScanManager, ScanSegment, plan_centers
from rf_mcp.catalog import Catalog
from rf_mcp.classification import classify_features, extract_features
from rf_mcp.comparison import compare_survey_results, save_comparison_plot
from rf_mcp.activity import summarize_activity_runs
import rf_mcp.propagation as propagation_module
from rf_mcp.propagation import (
    band_for_frequency, fetch_space_weather, space_weather_interpretation,
    summarize_local_propagation,
)
import rf_mcp.signal_library as signal_library_module
from rf_mcp.signal_library import (
    add_exemplar, delete_fingerprint, list_fingerprints, match_fingerprints,
    save_fingerprint,
)
import rf_mcp.recording_workspace as recording_workspace_module
from rf_mcp.recording_workspace import (
    add_annotation, add_bookmark, compare_wav, create_session, delete_session,
    export_session, extract_wav_clip, get_session, search_sessions,
)
from rf_mcp.web import RfWebApp, authorized, validate_api_token
from rf_mcp.presets import normalize_preset
from rf_mcp.scheduling import SchedulerManager, normalize_schedule
from rf_mcp.satellite import (
    SatellitePassScheduler,
    build_doppler_plan,
    fetch_celestrak_tle,
    normalize_satellite_downlinks,
    normalize_satellite_watch,
    parse_coordinate,
    parse_tle_response,
    predict_passes,
    refresh_satellite_tle,
    save_doppler_plot,
    tle_checksum_valid,
)
import rf_mcp.satellite as satellite_module
from rf_mcp.satellite_catalog import get_catalog_entry, search_catalog, selected_downlinks
import rf_mcp.satellite_catalog as satellite_catalog_module
import rf_mcp.satellite_planner as satellite_planner_module
from rf_mcp.satellite_planner import (
    delete_location, get_location, list_locations, plan_observations, save_location,
)
from rf_mcp.satellite_receiver import SatelliteReceiverManager, export_satellite_telemetry
import rf_mcp.satellite_receiver as satellite_receiver_module
from rf_mcp.satellite_performance import (
    export_pass_performance,
    save_pass_performance_plot,
    score_satellite_pass,
    summarize_pass_performance,
)
import rf_mcp.satellite_performance as satellite_performance_module
from rf_mcp.satellite_telemetry import (
    decode_observation_telemetry,
    decode_payload as decode_telemetry_payload,
    export_decoded_telemetry,
    normalize_telemetry_schema,
    save_telemetry_plot,
)
import rf_mcp.satellite_telemetry as satellite_telemetry_module
from rf_mcp.satellite_telemetry_alerts import (
    evaluate_telemetry_alert_rule,
    normalize_telemetry_alert_rule,
)
from rf_mcp.alerts import AlertEvaluator, evaluate_rule, normalize_alert_rule
from rf_mcp.notifications import normalize_webhook_destination, signed_headers
import rf_mcp.notifications as notifications
from rf_mcp.rds import decode_rds, make_rds_block, parse_rds_groups, rds_remainder
from rf_mcp.fm_survey import (
    compare_fm_survey_results,
    fm_candidate_score,
    fm_channel_plan,
    station_record,
)
from rf_mcp.weak_signal import (
    analyze_weak_audio,
    iq_cycle_to_audio,
    parse_jt9_output,
    parse_wsprd_output,
    seconds_to_next_period,
)
from rf_mcp.fldigi_bridge import (
    extract_text_entities,
    normalize_fldigi_mode,
    playback_command,
)
from rf_mcp.sstv import (
    HEADLESS_DECODER_BOOTSTRAP,
    SSTV_MODES,
    detect_vis,
    decoder_indicates_no_signal,
    image_fingerprint,
    iq_to_nfm_audio,
    run_sstv_decoder,
)
import rf_mcp.sstv as sstv_module
from rf_mcp.sstv_watcher import (
    SSTVStreamDetector,
    SSTVWatcherManager,
    StreamingDemodulator,
    WatchJob,
    doppler_frequency_at,
)
import rf_mcp.sstv_watcher as sstv_watcher_module
from rf_mcp.sstv_alerts import (
    evaluate_sstv_image,
    evaluate_sstv_rule,
    normalize_sstv_alert_rule,
)
from rf_mcp.digital_decode import (
    FIGS,
    ITA2_FIGURES,
    ITA2_LETTERS,
    LTRS,
    MORSE,
    PSK31_VARICODE,
    ax25_fcs,
    decode_ax25_afsk1200,
    decode_ax25_g3ruh9600,
    decode_bpsk31,
    decode_cw,
    decode_rtty,
)


def test_frequency_ranges():
    assert validate_frequency(10_000_000) == 10_000_000
    assert validate_frequency(118_500_000) == 118_500_000
    for invalid in (1_000, 40_000_000, 300_000_000):
        try:
            validate_frequency(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid frequency {invalid}")


def test_duration_range():
    assert validate_duration(2) == 2


def test_detects_synthetic_tone():
    sample_rate = 768_000
    center = 10_000_000
    offset = 25_000
    count = 768_000
    rng = np.random.default_rng(1234)
    time = np.arange(count) / sample_rate
    iq = np.exp(2j * np.pi * offset * time) + 0.05 * (
        rng.normal(size=count) + 1j * rng.normal(size=count)
    )
    frequencies, power = averaged_spectrum(iq, center, sample_rate, 16_384)
    mask = valid_passband_mask(frequencies, center, sample_rate)
    noise, peaks = analyze_peaks(frequencies, power, mask, max_peaks=5)
    assert noise < -20
    assert peaks
    assert abs(peaks[0].frequency_hz - (center + offset)) < 100


def test_peak_adjacent_to_valid_passband_boundary_has_finite_prominence():
    frequencies = np.arange(32, dtype=float)
    power = np.full(32, -60.0)
    mask = np.zeros(32, dtype=bool)
    mask[8:24] = True
    power[9] = -10.0
    _, peaks = analyze_peaks(frequencies, power, mask, max_peaks=5)
    assert peaks
    assert np.isfinite(peaks[0].prominence_db)


def test_digital_psd_integrates_to_tone_power_across_fft_sizes():
    sample_rate = 65_536
    tone_hz = 1_024
    amplitude = 0.25
    time = np.arange(sample_rate) / sample_rate
    iq = amplitude * np.exp(2j * np.pi * tone_hz * time)
    expected_dbfs = 20 * np.log10(amplitude)
    measurements = []
    for fft_size in (4_096, 8_192):
        frequencies, psd = averaged_psd_dbfs_per_hz(iq, 0, sample_rate, fft_size)
        measurements.append(integrate_psd_dbfs(frequencies, psd, tone_hz, 512))
    assert all(abs(value - expected_dbfs) < 0.15 for value in measurements)
    assert abs(measurements[0] - measurements[1]) < 0.1


def test_iq_level_metrics_flags_full_scale_clipping():
    safe = np.full(10_000, 0.25 + 0.25j)
    clipped = safe.copy()
    clipped[:10] = 1.0 + 1.0j
    assert not iq_level_metrics(safe)["overload_suspected"]
    assert iq_level_metrics(clipped)["overload_suspected"]


def test_am_demodulation_contains_modulation_tone():
    sample_rate = 768_000
    duration = 1.0
    time = np.arange(int(sample_rate * duration)) / sample_rate
    modulation_hz = 1_000
    iq = (1 + 0.5 * np.sin(2 * np.pi * modulation_hz * time)).astype(np.complex64)
    audio = demodulate(iq, sample_rate, "am", 10_000)
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / 48_000)
    dominant = frequencies[np.argmax(spectrum[1:]) + 1]
    assert abs(dominant - modulation_hz) < 5


def test_downconversion_moves_offset_tone_to_baseband():
    sample_rate = 768_000
    offset = -50_000
    time = np.arange(sample_rate // 10) / sample_rate
    iq = np.exp(2j * np.pi * offset * time)
    baseband = downconvert(iq, sample_rate, offset)
    phase_step = np.angle(np.mean(baseband[1:] * np.conj(baseband[:-1])))
    assert abs(phase_step) < 1e-6


def test_bandwidth_defaults_and_limits():
    assert validate_bandwidth("am", None) == 10_000
    assert validate_bandwidth("nfm", 12_500) == 12_500


def test_nfm_demodulation_contains_modulation_tone():
    sample_rate = 768_000
    time = np.arange(sample_rate) / sample_rate
    modulation_hz = 1_000
    deviation_hz = 2_500
    beta = deviation_hz / modulation_hz
    iq = np.exp(1j * beta * np.sin(2 * np.pi * modulation_hz * time))
    audio = demodulate(iq, sample_rate, "nfm", 12_500)
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / 48_000)
    dominant = frequencies[np.argmax(spectrum[1:]) + 1]
    assert abs(dominant - modulation_hz) < 5


def test_monitor_groups_consecutive_activity_samples():
    manager = MonitorManager()

    def sample(elapsed, present, snr):
        return MonitorSample(
            captured_at="2026-08-10T00:00:00+00:00",
            elapsed_seconds=elapsed,
            relative_peak_db=-10,
            relative_noise_floor_db=-30,
            estimated_snr_db=snr,
            dominant_frequency_hz=10_000_000,
            dominant_offset_hz=0,
            occupied_bandwidth_hz=5_000,
            signal_present=present,
            signal_confidence=0.9 if present else 0.0,
            duty_cycle_percent=100 if present else 0,
        )

    samples = [sample(0, False, 2), sample(5, True, 12), sample(10, True, 18), sample(15, False, 3)]
    events = manager._events(samples, 5)
    assert len(events) == 1
    assert events[0]["start_elapsed_seconds"] == 5
    assert events[0]["end_elapsed_seconds"] == 15
    assert events[0]["peak_snr_db"] == 18


def test_band_scan_center_plan_covers_requested_range():
    centers = plan_centers(14_000_000, 16_000_000, 0.15)
    usable_half_width = (768_000 / 2) * (1 - 0.12)
    assert centers[0] - usable_half_width <= 14_000_000
    assert centers[-1] + usable_half_width >= 16_000_000
    assert all(right > left for left, right in zip(centers, centers[1:]))


def test_band_survey_ranks_candidates_and_finishes_at_100_percent():
    manager = ScanManager()
    frequencies = np.arange(14_000_000, 14_100_001, 100.0)
    power = np.full(frequencies.size, -50.0)
    power[np.argmin(abs(frequencies - 14_025_000))] = -5.0
    power[np.argmin(abs(frequencies - 14_075_000))] = -12.0
    config = {
        "minimum_signal_spacing_hz": 1_000,
        "threshold_above_noise_db": 8.0,
        "max_signals": 10,
        "classify_top_signals": 2,
    }
    job = ScanJob("survey-test", config, [14_050_000], state="completed", phase="finished")
    job.segments.append(ScanSegment(14_050_000, "2026-08-10T00:00:00+00:00", -50, -5))
    job.classification_total = 2
    job.classifications.extend([{"status": "completed"}, {"status": "completed"}])
    manager._jobs[job.job_id] = job

    _, signals = manager._detect_signals(frequencies, power, job)
    assert [round(item["frequency_hz"]) for item in signals[:2]] == [14_025_000, 14_075_000]
    psd = power - 80
    _, digital_signals = manager._detect_signals(frequencies, power, job, psd)
    assert all("digital_peak_psd_dbfs_hz" in item for item in digital_signals)
    assert all("digital_power_dbfs_10khz" in item for item in digital_signals)
    assert manager.status(job.job_id)["progress_percent"] == 100.0


def test_band_scan_checkpoint_persists_live_dashboard_progress(monkeypatch):
    manager = ScanManager()
    job = ScanJob(
        "scan-checkpoint",
        {"classify_top_signals": 0, "planned_steps": 2},
        [14_100_000, 14_200_000],
        state="running",
    )
    job.segments.append(ScanSegment(14_100_000, "2026-08-10T00:00:00+00:00", -40, 0))
    calls = []
    monkeypatch.setattr("rf_mcp.scanning.catalog.upsert_job", lambda *args, **kwargs: calls.append((args, kwargs)))

    manager._checkpoint(job, current_frequency_hz=14_100_000)

    summary = calls[0][1]["summary"]
    assert summary["completed_steps"] == 1
    assert summary["planned_steps"] == 2
    assert summary["current_frequency_hz"] == 14_100_000
    assert summary["progress_percent"] == 50.0


def test_band_survey_comparison_detects_changes():
    def signal(frequency, power):
        return {"frequency_hz": frequency, "relative_power_db": power, "above_noise_db": 20}

    baseline = {
        "scanned_range_hz": [14_000_000, 14_100_000],
        "signals": [
            signal(14_010_000, -3),
            signal(14_020_000, -10),
            signal(14_030_000, -20),
        ],
        "classifications": [
            {"status": "completed", "frequency_hz": 14_010_000, "best_label": "am", "ambiguous": False},
            {"status": "completed", "frequency_hz": 14_020_000, "best_label": "usb", "ambiguous": False},
        ],
    }
    current = {
        "scanned_range_hz": [14_000_000, 14_100_000],
        "signals": [
            signal(14_010_100, -11),
            signal(14_020_000, -9),
            signal(14_040_000, -5),
        ],
        "classifications": [
            {"status": "completed", "frequency_hz": 14_010_100, "best_label": "am", "ambiguous": False},
            {"status": "completed", "frequency_hz": 14_020_000, "best_label": "lsb", "ambiguous": False},
        ],
    }
    result = compare_survey_results(
        baseline,
        current,
        frequency_tolerance_hz=200,
        frequency_shift_threshold_hz=50,
        power_change_threshold_db=6,
    )
    assert result["matched_count"] == 2
    assert result["changed_count"] == 2
    assert result["new_count"] == 1
    assert result["disappeared_count"] == 1
    changes = {change for item in result["changed_signals"] for change in item["changes"]}
    assert changes == {
        "frequency_shifted",
        "relative_power_changed",
        "classification_changed",
    }
    with tempfile.TemporaryDirectory() as temporary:
        plot_path = Path(temporary) / "comparison.png"
        save_comparison_plot(plot_path, result)
        assert plot_path.stat().st_size > 0


def test_fm_channel_plan_and_candidate_score():
    assert fm_channel_plan(87_900_000, 88_300_000, 200_000) == [
        87_900_000, 88_100_000, 88_300_000
    ]
    sample_rate = 768_000
    center = 88_250_000
    target = 88_100_000
    time_axis = np.arange(sample_rate // 2) / sample_rate
    rng = np.random.default_rng(11)
    iq = 0.02 * (rng.normal(size=time_axis.size) + 1j * rng.normal(size=time_axis.size))
    iq += np.exp(2j * np.pi * (target - center) * time_axis)
    score = fm_candidate_score(iq, center, sample_rate, target)
    assert score["discovery_score_db"] > 20


def test_fm_station_record_and_survey_comparison():
    metrics = {"stereo_detected": True, "pilot_to_composite_rms_db": -18.0}
    rds = {"group_count": 12, "station": {
        "pi_code": "1234", "program_service": "TESTFM", "program_type": 5,
        "pty_name": "Rock", "program_type_name": "ROCKFM",
        "radiotext": "HELLO", "traffic_program": True,
        "traffic_announcement": False, "music_speech": True,
        "alternative_frequencies_mhz": [88.5],
    }}
    record = station_record(88_100_000, metrics, rds)
    assert record["ps"] == "TESTFM"
    assert record["pty"] == 5
    assert record["music_speech"] == "music"
    assert record["alternative_frequencies_hz"] == [88_500_000]
    assert record["rds_group_count"] == 12
    baseline = {"job_id": "one", "stations": [record]}
    changed = dict(record, radiotext="UPDATED")
    added = dict(record, frequency_hz=88_300_000, ps="OTHER")
    result = compare_fm_survey_results(
        baseline, {"job_id": "two", "stations": [changed, added]}
    )
    assert result["new_count"] == 1
    assert result["changed_count"] == 1
    assert result["changed_stations"][0]["changes"]["radiotext"]["after"] == "UPDATED"


def test_catalog_accumulates_fm_station_directory():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        observed = "2026-08-10T20:00:00+00:00"
        station = {
            "frequency_hz": 88_100_000, "pi_code": "1234", "ps": "TESTFM",
            "pty": 5, "pty_name": "Rock", "ptyn": None, "radiotext": "FIRST",
            "tp": True, "ta": False, "music_speech": "music",
            "alternative_frequencies_hz": [88_500_000], "stereo_detected": True,
            "estimated_snr_db": 22.5, "pilot_to_composite_rms_db": -18.0,
            "rds_group_count": 8,
        }
        local_catalog.upsert_fm_station(station, job_id="survey-one", observed_at=observed)
        updated = dict(station, radiotext=None, rds_group_count=2)
        result = local_catalog.upsert_fm_station(
            updated, job_id="survey-two", observed_at="2026-08-11T20:00:00+00:00"
        )
        assert result["radiotext"] == "FIRST"
        assert result["observation_count"] == 2
        assert local_catalog.list_fm_stations(rds_only=True)[0]["pi_code"] == "1234"


def test_parse_ft8_and_ft4_decoder_output():
    output = """123015 -18  0.2  1274 ~  CQ K1ABC FN42
123015 -07 -0.1  1850 ~  W1XYZ K1ABC -12
decoder summary line
"""
    spots = parse_jt9_output(
        output, mode="ft8", dial_frequency_hz=14_074_000,
        captured_at="2026-08-10T12:30:00+00:00",
    )
    assert len(spots) == 2
    assert spots[0]["callsign"] == "K1ABC"
    assert spots[0]["grid"] == "FN42"
    assert spots[0]["is_cq"]
    assert spots[0]["rf_frequency_hz"] == 14_075_274
    assert spots[1]["snr_db"] == -7


def test_parse_wspr_decoder_output():
    output = "1230 -27  0.4  14.097124  1 K1ABC FN42 37\n"
    spots = parse_wsprd_output(
        output, dial_frequency_hz=14_095_600,
        captured_at="2026-08-10T12:30:00+00:00",
    )
    assert len(spots) == 1
    assert spots[0]["callsign"] == "K1ABC"
    assert spots[0]["power_dbm"] == 37
    assert spots[0]["drift_hz_per_minute"] == 1
    assert abs(spots[0]["audio_frequency_hz"] - 1524) < 0.01


def test_weak_signal_usb_streaming_audio_conversion():
    sample_rate = 768_000
    duration = 0.25
    offset_hz = -10_000
    audio_hz = 1_500
    time_axis = np.arange(round(sample_rate * duration)) / sample_rate
    iq = np.exp(2j * np.pi * (offset_hz + audio_hz) * time_axis).astype("complex64")
    interleaved = np.empty(iq.size * 2, dtype="<f4")
    interleaved[0::2], interleaved[1::2] = iq.real, iq.imag
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "test.iq"
        interleaved.tofile(path)
        audio = iq_cycle_to_audio(
            path, first_sample=0, sample_count=iq.size,
            sample_rate_hz=sample_rate, offset_hz=offset_hz,
        )
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / 12_000)
    assert abs(frequencies[np.argmax(spectrum)] - audio_hz) < 5


def test_catalog_persists_and_filters_weak_signal_spots():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        spot = {
            "mode": "ft8", "dial_frequency_hz": 14_074_000,
            "audio_frequency_hz": 1274.0, "rf_frequency_hz": 14_075_274.0,
            "utc_text": "123015", "snr_db": -18.0, "time_offset_seconds": 0.2,
            "drift_hz_per_minute": None, "message": "CQ K1ABC FN42",
            "callsign": "K1ABC", "grid": "FN42", "power_dbm": None,
            "is_cq": True, "captured_at": "2026-08-10T12:30:00+00:00",
            "raw_line": "123015 -18 0.2 1274 ~ CQ K1ABC FN42",
        }
        inserted = local_catalog.add_weak_signal_spots([spot], job_id="weak-one")
        assert inserted[0]["spot_id"].startswith("spot-")
        result = local_catalog.list_weak_signal_spots(callsign="k1abc")
        assert len(result) == 1 and result[0]["is_cq"]


def test_weak_signal_utc_period_alignment():
    assert seconds_to_next_period(15, now=30.0) == 0
    assert seconds_to_next_period(15, now=31.0) == 14


def test_weak_signal_audio_diagnostics_identify_tone_energy():
    time_axis = np.arange(12_000 * 2) / 12_000
    audio = 0.2 * np.sin(2 * np.pi * 1_500 * time_axis)
    result = analyze_weak_audio(audio)
    assert result["classification"] == "concentrated_tone_energy"
    assert -18 < result["rms_dbfs"] < -16
    assert result["spectral_contrast_db"] > 12


def test_weak_signal_audio_diagnostics_identify_low_level():
    result = analyze_weak_audio(np.zeros(12_000))
    assert result["classification"] == "very_low_level"
    assert result["peak_dbfs"] <= -200


def test_fldigi_mode_normalization_and_aliases():
    assert normalize_fldigi_mode("Olivia") == ("olivia-8-500", "OLIVIA-8-500")
    assert normalize_fldigi_mode("MFSK16") == ("mfsk16", "MFSK16")
    assert normalize_fldigi_mode("Hellschreiber") == ("feldhell", "HELL")
    try:
        normalize_fldigi_mode("made-up-mode")
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an unsupported Fldigi mode")


def test_fldigi_playback_command_is_argument_safe(monkeypatch):
    monkeypatch.setenv(
        "RF_MCP_FLDIGI_PLAYBACK", "aplay -q -D plughw:Loopback,0,0 {wav}"
    )
    path = Path("/tmp/audio with spaces.wav")
    command = playback_command(path)
    assert command[-1] == str(path)
    assert command[0] == "aplay"


def test_fldigi_text_entity_extraction():
    decoded = extract_text_entities(
        "CQ CQ DE K1ABC K1ABC FN42\nHello    from Olivia!\x00\n\n\n"
    )
    assert decoded["callsigns"] == ["K1ABC"]
    assert decoded["grids"] == ["FN42"]
    assert "Hello from Olivia!" in decoded["text"]


def test_catalog_persists_fldigi_decode():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        decoded = local_catalog.add_fldigi_decode({
            "job_id": "fldigi-one", "mode": "olivia-8-500",
            "fldigi_modem": "OLIVIA-8-500", "dial_frequency_hz": 14_106_500,
            "carrier_audio_hz": 1500, "text": "CQ K1ABC FN42", "quality": 87.5,
            "callsigns": ["K1ABC"], "grids": ["FN42"],
            "captured_at": "2026-08-10T12:30:00+00:00", "duration_seconds": 30,
        })
        assert decoded["decode_id"].startswith("fldigi-")
        result = local_catalog.list_fldigi_decodes(mode="olivia-8-500")
        assert result[0]["callsigns"] == ["K1ABC"]
        assert result[0]["quality"] == 87.5


def test_band_comparison_prefers_compatible_digital_power():
    scale = {
        "scale": "digital_dbfs_per_hz_v1",
        "calibrated_rf_input_power": False,
        "psd_units": "dBFS/Hz",
        "integrated_power_units": "dBFS",
        "integration_bandwidth_hz": 10_000,
        "reference": "complex_float_full_scale_component_1.0",
    }
    profile = {
        "sample_rate_hz": 768_000,
        "agc": False,
        "attenuation_steps": 1,
        "attenuation_db": 6,
        "lna": False,
        "fft_size": 8_192,
        "window": "blackman",
    }
    baseline = {
        "scanned_range_hz": [14_000_000, 14_100_000],
        "digital_power_scale": scale,
        "receiver_profile": profile,
        "signals": [
            {
                "frequency_hz": 14_050_000,
                "relative_power_db": -3,
                "digital_power_dbfs_10khz": -40,
            }
        ],
    }
    current = {
        **baseline,
        "signals": [
            {
                "frequency_hz": 14_050_000,
                "relative_power_db": -3,
                "digital_power_dbfs_10khz": -31,
            }
        ],
    }
    result = compare_survey_results(baseline, current, power_change_threshold_db=6)
    assert result["power_comparison_scale"] == "digital_power_dbfs_10khz"
    assert result["changed_signals"][0]["changes"] == ["digital_power_changed"]
    assert result["changed_signals"][0]["power_delta_db"] == 9


def test_persistent_catalog_and_guarded_cleanup():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        catalog = Catalog(root, index_existing=False)
        job_id = "test-job"
        catalog.upsert_job(job_id, "test", "completed", config={"frequency_hz": 10_000_000})
        artifact_path = root / "old-result.json"
        artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
        os.utime(artifact_path, (old_time, old_time))
        artifact = catalog.register_artifact(artifact_path, "result_json", job_id=job_id)

        assert catalog.get_job(job_id)["job_id"] == job_id
        assert catalog.get_artifact(artifact["artifact_id"])["filename"] == artifact_path.name
        assert catalog.cleanup(
            older_than_days=30, kinds=["result_json"], max_delete=10, dry_run=True
        )["candidate_count"] == 1
        assert artifact_path.exists()

        catalog.set_pinned(artifact["artifact_id"], True)
        assert catalog.cleanup(
            older_than_days=30, kinds=["result_json"], max_delete=10, dry_run=False
        )["deleted_count"] == 0
        catalog.set_pinned(artifact["artifact_id"], False)
        assert catalog.cleanup(
            older_than_days=30, kinds=["result_json"], max_delete=10, dry_run=False
        )["deleted_count"] == 1
        assert not artifact_path.exists()


def test_persistent_presets_preserve_identity_on_replace():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        created = catalog.save_preset(
            name="20 Meter Survey",
            preset_type="band_survey",
            description="Morning baseline",
            config={"start_frequency_hz": 14_000_000},
        )
        assert catalog.get_preset("20 meter survey")["preset_id"] == created["preset_id"]
        assert catalog.list_presets(preset_type="band_survey")[0]["name"] == "20 Meter Survey"
        replaced = catalog.save_preset(
            name="20 Meter Survey",
            preset_type="band_survey",
            description="Updated",
            config={"start_frequency_hz": 14_100_000},
            replace_existing=True,
        )
        assert replaced["preset_id"] == created["preset_id"]
        assert replaced["description"] == "Updated"
        deleted = catalog.delete_preset(created["preset_id"])
        assert deleted["preset_id"] == created["preset_id"]
        assert catalog.list_presets() == []


def test_persistent_schedules_preserve_identity_and_protect_presets():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        preset = catalog.save_preset(
            name="20m Survey",
            preset_type="band_survey",
            description="",
            config={"start_frequency_hz": 14_000_000},
        )
        created = catalog.save_schedule(
            name="Hourly 20m",
            preset_id=preset["preset_id"],
            interval_seconds=3600,
            enabled=True,
            next_run_at="2026-08-11T12:00:00+00:00",
        )
        assert catalog.get_schedule("hourly 20M")["schedule_id"] == created["schedule_id"]
        replaced = catalog.save_schedule(
            name="Hourly 20m",
            preset_id=preset["preset_id"],
            interval_seconds=7200,
            enabled=False,
            next_run_at="2026-08-11T13:00:00+00:00",
            replace_existing=True,
        )
        assert replaced["schedule_id"] == created["schedule_id"]
        assert replaced["interval_seconds"] == 7200
        try:
            catalog.delete_preset(preset["preset_id"])
        except ValueError:
            pass
        else:
            raise AssertionError("Deleted a preset still referenced by a schedule")
        catalog.delete_schedule(created["schedule_id"])
        catalog.delete_preset(preset["preset_id"])


def test_scheduler_runs_one_catchup_and_skips_when_busy():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        preset = catalog.save_preset(
            name="Survey", preset_type="band_survey", description="", config={}
        )
        schedule = catalog.save_schedule(
            name="Recurring Survey",
            preset_id=preset["preset_id"],
            interval_seconds=300,
            enabled=True,
            next_run_at="2026-08-10T12:00:00+00:00",
        )
        launched = []
        busy = [False]

        def launch(preset_id, schedule_id):
            launched.append((preset_id, schedule_id))
            return {"job_id": f"job-{len(launched)}"}

        manager = SchedulerManager(catalog, launch, lambda: busy[0])
        now = datetime(2026, 8, 10, 18, tzinfo=timezone.utc)
        result = manager.tick(now)[0]
        assert result["last_status"] == "launched"
        assert result["next_run_at"] == (now + timedelta(seconds=300)).isoformat()
        assert len(launched) == 1
        assert manager.tick(now) == []

        later = now + timedelta(hours=3)
        manager.tick(later)
        assert len(launched) == 2
        busy[0] = True
        busy_at = later + timedelta(minutes=5)
        skipped = manager.tick(busy_at)[0]
        assert skipped["last_status"] == "skipped_busy"
        assert len(launched) == 2
        assert skipped["schedule_id"] == schedule["schedule_id"]


def test_scheduler_marks_station_memory_scan_completed():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        preset = catalog.save_preset(
            name="Memory round", preset_type="station_memory_scan", description="", config={}
        )
        catalog.save_schedule(
            name="Memory hourly", preset_id=preset["preset_id"], interval_seconds=3600,
            enabled=True, next_run_at="2026-08-10T12:00:00+00:00",
        )
        manager = SchedulerManager(catalog, lambda *args: {"job_id": "memory-job"}, lambda: False)
        result = manager.tick(datetime(2026, 8, 10, 13, tzinfo=timezone.utc))[0]
        assert result["last_status"] == "completed"
        assert result["last_job_id"] == "memory-job"


def test_schedule_validation_requires_timezone_and_bounds_interval():
    now = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
    normalized = normalize_schedule(
        name="Daily", interval_seconds=3600, start_at=None, enabled=True, now=now
    )
    assert normalized[3] == (now + timedelta(hours=1)).isoformat()
    for kwargs in (
        {"name": "Daily", "interval_seconds": 59, "start_at": None, "enabled": True},
        {"name": "Daily", "interval_seconds": 60, "start_at": "2026-08-11T12:00:00", "enabled": True},
    ):
        try:
            normalize_schedule(**kwargs, now=now)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an invalid schedule")


def test_alert_rule_validation_and_matching():
    rule = normalize_alert_rule(
        name="Strong AM",
        condition_type="classification_is",
        entry_label="WWV 10 MHz",
        classification_label="am",
        min_confidence=0.6,
        threshold_db=None,
        enabled=True,
    )
    rule["rule_id"] = "rule-test"
    observation = {
        "label": "WWV 10 MHz",
        "frequency_hz": 10_000_000,
        "status": "completed",
        "best_label": "am",
        "best_confidence": 0.72,
    }
    matched, message = evaluate_rule(rule, observation)
    assert matched is True
    assert "Strong AM" in message
    observation["best_confidence"] = 0.5
    assert evaluate_rule(rule, observation)[0] is False
    try:
        normalize_alert_rule(
            name="Bad",
            condition_type="classification_is",
            entry_label=None,
            classification_label="telepathy",
            min_confidence=0.5,
            threshold_db=None,
            enabled=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an invalid classification label")


def test_persistent_alert_events_survive_rule_deletion_and_acknowledge():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        preset = catalog.save_preset(
            name="WWV",
            preset_type="watchlist",
            description="",
            config={"entries": [{"frequency_hz": 10_000_000, "label": "WWV 10"}]},
        )
        schedule = catalog.save_schedule(
            name="WWV Hourly",
            preset_id=preset["preset_id"],
            interval_seconds=3600,
            enabled=True,
            next_run_at="2026-08-11T12:00:00+00:00",
        )
        rule = catalog.save_alert_rule(
            name="AM Found",
            schedule_id=schedule["schedule_id"],
            entry_label="WWV 10",
            condition_type="classification_is",
            classification_label="am",
            min_confidence=0.5,
            threshold_db=None,
            enabled=True,
        )
        destination = catalog.save_webhook_destination(
            name="Alert Receiver",
            url="https://example.com/rf-alerts",
            signing_secret="correct-horse-battery-staple",
            all_rules=False,
            rule_id=rule["rule_id"],
            sstv_rule_id=None,
            satellite_watch_id=None,
            enabled=True,
        )
        assert destination["has_signing_secret"] is True
        assert "signing_secret" not in destination
        events = AlertEvaluator(catalog).evaluate_watchlist(
            schedule["schedule_id"],
            {
                "job_id": None,
                "observations": [
                    {
                        "label": "WWV 10",
                        "frequency_hz": 10_000_000,
                        "status": "completed",
                        "best_label": "am",
                        "best_confidence": 0.8,
                    }
                ],
            },
        )
        assert len(events) == 1
        event_id = events[0]["event_id"]
        delivery = catalog.list_webhook_deliveries(event_id=event_id)[0]
        assert delivery["state"] == "pending"
        due_at = datetime.fromisoformat(delivery["next_attempt_at"]) + timedelta(seconds=1)
        due = catalog.due_webhook_deliveries(due_at.isoformat())[0]
        assert due["signing_secret"] == "correct-horse-battery-staple"
        assert due["payload"]["event"]["event_id"] == event_id
        assert catalog.list_alert_events(acknowledged=False)[0]["event_id"] == event_id
        acknowledged = catalog.acknowledge_alert_event(event_id)
        assert acknowledged["acknowledged"] is True
        first_ack = acknowledged["acknowledged_at"]
        assert catalog.acknowledge_alert_event(event_id)["acknowledged_at"] == first_ack
        catalog.delete_webhook_destination(destination["destination_id"])
        assert catalog.get_webhook_delivery(delivery["delivery_id"])["state"] == "cancelled"
        catalog.delete_alert_rule(rule["rule_id"])
        retained = catalog.get_alert_event(event_id)
        assert retained["rule_id"] is None
        assert retained["rule_name"] == "AM Found"


def test_webhook_validation_and_signature_are_deterministic():
    normalized = normalize_webhook_destination(
        name="IFTTT",
        url="https://example.com/hook",
        signing_secret="0123456789abcdef",
        enabled=True,
        resolve_host=False,
    )
    assert normalized["url"] == "https://example.com/hook"
    first = signed_headers(b'{"ok":true}', normalized["signing_secret"], "2026-08-10T00:00:00+00:00", "alert-1")
    second = signed_headers(b'{"ok":true}', normalized["signing_secret"], "2026-08-10T00:00:00+00:00", "alert-1")
    assert first["X-RF-MCP-Signature-256"] == second["X-RF-MCP-Signature-256"]
    assert first["X-RF-MCP-Signature-256"].startswith("sha256=")
    try:
        normalize_webhook_destination(
            name="Unsafe",
            url="http://example.com/hook",
            signing_secret=None,
            enabled=True,
            resolve_host=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an insecure webhook without explicit opt-in")


def test_webhook_dispatcher_marks_successful_delivery():
    recorded = {}

    class FakeCatalog:
        def record_webhook_delivery_attempt(self, delivery_id, **kwargs):
            recorded.update({"delivery_id": delivery_id, **kwargs})
            return dict(recorded)

    class FakeResponse:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeOpener:
        def open(self, request, timeout):
            assert request.full_url == "https://example.com/hook"
            assert timeout == 5
            return FakeResponse()

    original_opener = notifications.build_opener
    original_validate = notifications.validate_webhook_url
    notifications.build_opener = lambda *args: FakeOpener()
    notifications.validate_webhook_url = lambda url: url
    try:
        result = notifications.WebhookDispatcher(FakeCatalog())._deliver(
            {
                "delivery_id": "delivery-1",
                "event_id": "alert-1",
                "destination_url": "https://example.com/hook",
                "payload": {"event": {"event_id": "alert-1"}},
                "signing_secret": "0123456789abcdef",
                "attempt_count": 0,
            },
            datetime(2026, 8, 10, 12, tzinfo=timezone.utc),
        )
    finally:
        notifications.build_opener = original_opener
        notifications.validate_webhook_url = original_validate
    assert result["state"] == "delivered"
    assert result["attempt_count"] == 1
    assert result["http_status"] == 204


def test_cw_decoder_recovers_synthetic_morse_iq():
    sample_rate = 8_000
    dot = 0.06
    reverse_morse = {value: key for key, value in MORSE.items() if value.isalnum()}
    states = [(False, 7)]
    for word_index, word in enumerate("SOS TEST".split()):
        if word_index:
            states.append((False, 7))
        for char_index, character in enumerate(word):
            if char_index:
                states.append((False, 3))
            for symbol_index, symbol in enumerate(reverse_morse[character]):
                if symbol_index:
                    states.append((False, 1))
                states.append((True, 1 if symbol == "." else 3))
    states.append((False, 7))
    keyed = np.concatenate(
        [np.full(round(units * dot * sample_rate), state) for state, units in states]
    )
    rng = np.random.default_rng(10)
    iq = keyed.astype(float) * np.exp(2j * np.pi * 80 * np.arange(len(keyed)) / sample_rate)
    iq += 0.015 * (rng.normal(size=len(iq)) + 1j * rng.normal(size=len(iq)))
    decoded = decode_cw(iq.astype(np.complex64), sample_rate)
    assert decoded["text"] == "SOS TEST"
    assert decoded["confidence"] > 0.8
    assert 17 <= decoded["estimated_wpm"] <= 23


def test_rtty_decoder_recovers_synthetic_baudot_iq():
    sample_rate = 8_000
    baud = 45.45
    reverse_letters = {value: key for key, value in ITA2_LETTERS.items() if value}
    reverse_figures = {value: key for key, value in ITA2_FIGURES.items() if value}
    codes = [reverse_letters[c] for c in "TEST "]
    codes += [FIGS] + [reverse_figures[c] for c in "123"] + [LTRS]
    bits = [1] * round(3 * sample_rate / baud)
    accumulator = float(len(bits))

    def append_symbol(bit, symbols=1.0):
        nonlocal accumulator
        accumulator += symbols * sample_rate / baud
        count = round(accumulator) - len(bits)
        bits.extend([bit] * count)

    for code in codes:
        append_symbol(0)
        for index in range(5):
            append_symbol((code >> index) & 1)
        append_symbol(1, 1.5)
    bits.extend([1] * round(3 * sample_rate / baud))
    bits = np.asarray(bits)
    frequency = np.where(bits == 1, 85.0, -85.0)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    rng = np.random.default_rng(11)
    iq = np.exp(1j * phase) + 0.02 * (
        rng.normal(size=len(bits)) + 1j * rng.normal(size=len(bits))
    )
    decoded = decode_rtty(iq.astype(np.complex64), sample_rate, baud=baud, shift_hz=170)
    assert "TEST 123" in decoded["text"]
    assert decoded["confidence"] > 0.8


def test_bpsk31_decoder_recovers_synthetic_varicode_iq():
    sample_rate = 2_000
    samples_per_symbol = 64
    reverse = {character: code for code, character in PSK31_VARICODE.items()}
    bit_text = "0" * 20 + "00".join(reverse[char] for char in "test") + "00" + "1" * 12
    phase = 0.0
    signal = []
    for bit in map(int, bit_text):
        if bit == 0:
            phase += np.pi
        signal.extend(np.exp(1j * phase) * np.ones(samples_per_symbol))
    signal = np.asarray(signal)
    signal *= np.exp(2j * np.pi * 7 * np.arange(len(signal)) / sample_rate)
    rng = np.random.default_rng(12)
    signal += 0.02 * (rng.normal(size=len(signal)) + 1j * rng.normal(size=len(signal)))
    decoded = decode_bpsk31(signal.astype(np.complex64), sample_rate)
    assert "test" in decoded["text"]
    assert decoded["confidence"] > 0.8
    assert abs(decoded["estimated_frequency_offset_hz"] - 7) < 0.5


def test_ax25_afsk1200_decoder_recovers_valid_packet():
    def address(callsign, ssid, final):
        padded = callsign.ljust(6)
        return bytes([ord(char) << 1 for char in padded]) + bytes(
            [0x60 | (ssid << 1) | int(final)]
        )

    payload = (
        address("APRS", 0, False)
        + address("N0CALL", 1, True)
        + bytes([0x03, 0xF0])
        + b">RF-MCP TEST"
    )
    frame = payload + ax25_fcs(payload).to_bytes(2, "little")
    data_bits = [(byte >> bit) & 1 for byte in frame for bit in range(8)]
    stuffed = []
    ones = 0
    for bit in data_bits:
        stuffed.append(bit)
        if bit:
            ones += 1
            if ones == 5:
                stuffed.append(0)
                ones = 0
        else:
            ones = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    bits = flag * 8 + stuffed + flag * 4
    tone_state = 0
    states = []
    for bit in bits:
        if bit == 0:
            tone_state ^= 1
        states.append(tone_state)
    sample_rate = 9_600
    phase = 0.0
    audio = []
    for state in states:
        frequency = 2_200 if state else 1_200
        for _ in range(8):
            phase += 2 * np.pi * frequency / sample_rate
            audio.append(np.sin(phase))
    audio = np.asarray(audio)
    fm_phase = np.cumsum(audio * 0.9)
    iq = np.exp(1j * fm_phase).astype(np.complex64)
    decoded = decode_ax25_afsk1200(iq, sample_rate)
    assert decoded["valid_fcs_count"] >= 1
    frame_result = next(frame for frame in decoded["frames"] if frame["fcs_valid"])
    assert frame_result["source"] == "N0CALL-1"
    assert frame_result["destination"] == "APRS"
    assert frame_result["information_text"] == ">RF-MCP TEST"


def test_ax25_g3ruh9600_decoder_recovers_valid_packet():
    def address(callsign, ssid, final):
        return bytes([ord(char) << 1 for char in callsign.ljust(6)]) + bytes(
            [0x60 | (ssid << 1) | int(final)]
        )

    payload = (address("CQ", 0, False) + address("N0CALL", 2, True)
               + bytes([0x03, 0xF0]) + b"G3RUH TEST")
    frame = payload + ax25_fcs(payload).to_bytes(2, "little")
    raw = [(byte >> bit) & 1 for byte in frame for bit in range(8)]
    stuffed, ones = [], 0
    for bit in raw:
        stuffed.append(bit)
        if bit:
            ones += 1
            if ones == 5:
                stuffed.append(0)
                ones = 0
        else:
            ones = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    hdlc = flag * 8 + stuffed + flag * 5
    level, nrzi = 0, []
    for bit in hdlc:
        if bit == 0:
            level ^= 1
        nrzi.append(level)
    scrambled = []
    for index, bit in enumerate(nrzi):
        delayed_12 = scrambled[index - 12] if index >= 12 else 0
        delayed_17 = scrambled[index - 17] if index >= 17 else 0
        scrambled.append(bit ^ delayed_12 ^ delayed_17)
    discriminator = np.repeat(np.where(np.asarray(scrambled), 0.45, -0.45), 4)
    iq = np.exp(1j * np.cumsum(discriminator)).astype(np.complex64)
    decoded = decode_ax25_g3ruh9600(iq, 38_400)
    assert decoded["valid_fcs_count"] >= 1
    recovered = next(item for item in decoded["frames"] if item["fcs_valid"])
    assert recovered["source"] == "N0CALL-2"
    assert recovered["information_text"] == "G3RUH TEST"


def test_broadcast_fm_recovers_synthetic_stereo_multiplex():
    sample_rate = 192_000
    duration = 1.0
    time_axis = np.arange(round(sample_rate * duration)) / sample_rate
    left = 0.5 * np.sin(2 * np.pi * 1_000 * time_axis)
    right = 0.5 * np.sin(2 * np.pi * 2_000 * time_axis)
    mono = (left + right) / 2
    difference = (left - right) / 2
    composite = (
        mono
        + difference * np.cos(2 * np.pi * 38_000 * time_axis)
        + 0.1 * np.cos(2 * np.pi * 19_000 * time_axis)
    )
    phase = np.cumsum(2 * np.pi * 60_000 * composite / sample_rate)
    iq = np.exp(1j * phase).astype(np.complex64)
    audio, metrics, _ = demodulate_broadcast_fm(
        iq, sample_rate, deemphasis_us=75, stereo=True
    )
    assert metrics["stereo_detected"] is True
    assert metrics["stereo_used"] is True
    assert metrics["audio_channels"] == 2
    frequencies = np.fft.rfftfreq(len(audio), 1 / 48_000)
    left_spectrum = np.abs(np.fft.rfft(audio[:, 0]))
    right_spectrum = np.abs(np.fft.rfft(audio[:, 1]))
    index_1k = int(np.argmin(np.abs(frequencies - 1_000)))
    index_2k = int(np.argmin(np.abs(frequencies - 2_000)))
    assert left_spectrum[index_1k] > left_spectrum[index_2k] * 5
    assert right_spectrum[index_2k] > right_spectrum[index_1k] * 5


def test_broadcast_fm_falls_back_to_mono_without_pilot():
    sample_rate = 192_000
    time_axis = np.arange(sample_rate // 2) / sample_rate
    composite = 0.5 * np.sin(2 * np.pi * 1_000 * time_axis)
    phase = np.cumsum(2 * np.pi * 60_000 * composite / sample_rate)
    audio, metrics, _ = demodulate_broadcast_fm(
        np.exp(1j * phase).astype(np.complex64), sample_rate, stereo=True
    )
    assert metrics["stereo_detected"] is False
    assert metrics["stereo_used"] is False
    assert metrics["audio_channels"] == 1
    assert audio.shape[1] == 1


def _synthetic_rds_groups():
    pi = 0x1234
    groups = []
    ps_name = "TESTFM  "
    for segment in range(4):
        b = (1 << 10) | (5 << 5) | (1 << 4) | (1 << 3) | segment
        c = (10 << 8) | 20
        pair = ps_name[segment * 2:segment * 2 + 2]
        d = (ord(pair[0]) << 8) | ord(pair[1])
        groups.append([pi, b, c, d])
    text = "HELLO RDS\r".ljust(12)
    for segment in range(3):
        b = (2 << 12) | (1 << 10) | segment
        chars = text[segment * 4:segment * 4 + 4]
        c = (ord(chars[0]) << 8) | ord(chars[1])
        d = (ord(chars[2]) << 8) | ord(chars[3])
        groups.append([pi, b, c, d])
    date = datetime(2026, 8, 10, tzinfo=timezone.utc)
    mjd = (date - datetime(1858, 11, 17, tzinfo=timezone.utc)).days
    hour, minute = 19, 45
    b = (4 << 12) | ((mjd >> 15) & 0x3)
    c = ((mjd & 0x7FFF) << 1) | ((hour >> 4) & 1)
    d = ((hour & 0xF) << 12) | (minute << 6) | 14
    groups.append([pi, b, c, d])
    return groups


def test_rds_block_crc_and_metadata_parsing():
    raw_groups = _synthetic_rds_groups()
    groups = []
    for words in raw_groups:
        blocks = [
            make_rds_block(words[0], "A"),
            make_rds_block(words[1], "B"),
            make_rds_block(words[2], "C"),
            make_rds_block(words[3], "D"),
        ]
        assert [rds_remainder(block) for block in blocks] == [0x0FC, 0x198, 0x168, 0x1B4]
        groups.append({"bit_offset": 0, "block_names": ["A", "B", "C", "D"], "blocks": words})
    parsed = parse_rds_groups(groups)
    station = parsed["station"]
    assert station["pi_code"] == "1234"
    assert station["program_service"] == "TESTFM"
    assert station["program_service_complete"] is True
    assert station["radiotext"] == "HELLO RDS"
    assert station["alternative_frequencies_mhz"] == [88.5, 89.5]
    assert station["clock_time_utc"].startswith("2026-08-10T19:45")
    assert station["local_offset_minutes"] == 420


def test_rds_decoder_recovers_synthetic_57khz_groups():
    words_list = _synthetic_rds_groups()
    data_bits = []
    for words in words_list:
        for word, offset in zip(words, ("A", "B", "C", "D")):
            block = make_rds_block(word, offset)
            data_bits.extend((block >> bit) & 1 for bit in range(25, -1, -1))
    data_bits[31] ^= 1  # exercise bounded one-bit correction in the first B block
    encoded = [0]
    for bit in data_bits:
        encoded.append(encoded[-1] ^ bit)
    chips = []
    for state in encoded:
        chips.extend(([1.0] * 8 + [-1.0] * 8) if state else ([-1.0] * 8 + [1.0] * 8))
    baseband = resample_poly(np.asarray(chips), 192_000, 19_000)
    time_axis = np.arange(len(baseband)) / 192_000
    composite = (
        0.15 * np.sin(2 * np.pi * 1_000 * time_axis)
        + 0.1 * np.cos(2 * np.pi * 19_000 * time_axis)
        + 0.04 * baseband * np.cos(2 * np.pi * 57_000 * time_axis)
    )
    decoded = decode_rds(composite, 192_000)
    assert decoded["group_count"] >= len(words_list) - 1
    assert decoded["station"]["pi_code"] == "1234"
    assert decoded["station"]["program_service"] == "TESTFM"
    assert decoded["station"]["radiotext"] == "HELLO RDS"
    assert decoded["block_error_count"] >= 1


def test_preset_validation_normalizes_survey_and_watchlist():
    _, survey_type, _, survey = normalize_preset(
        name="20m",
        preset_type="band_survey",
        description="",
        config={"start_frequency_hz": 14_000_000, "stop_frequency_hz": 14_350_000},
    )
    assert survey_type == "band_survey"
    assert survey["attenuation_steps"] == 1
    assert survey["classify_top_signals"] == 10
    _, activity_type, _, activity = normalize_preset(
        name="20m activity", preset_type="activity_monitor", description="",
        config={"start_frequency_hz": 14_000_000, "stop_frequency_hz": 14_350_000},
    )
    assert activity_type == "activity_monitor"
    assert activity["classify_top_signals"] == 10

    _, watch_type, _, watchlist = normalize_preset(
        name="Time stations",
        preset_type="watchlist",
        description="",
        config={
            "entries": [
                {"frequency_hz": 5_000_000, "label": "WWV 5"},
                {"frequency_hz": 10_000_000, "label": "WWV 10", "enabled": False},
            ]
        },
    )
    assert watch_type == "watchlist"
    assert watchlist["duration_seconds"] == 2
    assert watchlist["entries"][0]["enabled"] is True

    try:
        normalize_preset(
            name="Bad duplicate",
            preset_type="watchlist",
            description="",
            config={
                "entries": [
                    {"frequency_hz": 10_000_000},
                    {"frequency_hz": 10_000_000},
                ]
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted duplicate watchlist frequencies")


def test_rf_activity_dashboard_baseline_clusters_and_anomalies():
    runs = [
        {"job_id": "run-1", "created_at": "2026-08-11T01:00:00+00:00",
         "completed_at": "2026-08-11T01:01:00+00:00",
         "digital_noise_floor_dbfs_hz": -120.0, "relative_noise_floor_db": -35.0,
         "occupied_bin_fraction": 0.02, "signal_count": 1,
         "signals": [{"frequency_hz": 14_074_000, "above_noise_db": 15}],
         "overload": {"suspected": False}},
        {"job_id": "run-2", "created_at": "2026-08-11T02:00:00+00:00",
         "completed_at": "2026-08-11T02:01:00+00:00",
         "digital_noise_floor_dbfs_hz": -112.0, "relative_noise_floor_db": -34.0,
         "occupied_bin_fraction": 0.10, "signal_count": 2,
         "signals": [{"frequency_hz": 14_074_400, "above_noise_db": 18},
                     {"frequency_hz": 14_230_000, "above_noise_db": 12}],
         "overload": {"suspected": False}},
    ]
    summary = summarize_activity_runs(runs, frequency_tolerance_hz=1000,
                                      noise_anomaly_db=6,
                                      occupancy_anomaly_fraction=0.05)
    assert summary["cluster_count"] == 2
    assert summary["signal_clusters"][0]["run_count"] == 2
    assert summary["new_signal_frequencies_hz"] == [14_230_000.0]
    assert {item["type"] for item in summary["anomalies"]} == {
        "raised_noise_floor", "raised_occupancy", "new_signals",
    }


def test_hf_propagation_local_evidence_and_space_weather(monkeypatch):
    assert band_for_frequency(14_074_000) == "20m"
    local = summarize_local_propagation(
        spots=[{"rf_frequency_hz": 14_074_000, "callsign": "K1ABC", "snr_db": -10}],
        activity={"20m": {"latest_run": {"occupied_bin_fraction": 0.06,
                                          "signal_count": 7},
                           "latest_vs_baseline": {"noise_floor_delta_db": 1.5}}},
        time_stations=[{"label": "WWV 10 MHz", "detected": True}], hours=24,
    )
    band = next(item for item in local["bands"] if item["band_name"] == "20m")
    assert band["evidence_rating"] == "strong_local_evidence"
    assert local["time_station_detection_count"] == 1

    with tempfile.TemporaryDirectory() as temporary:
        old_result = propagation_module.RESULT_DIR
        propagation_module.RESULT_DIR = Path(temporary)
        responses = {
            "10cm-flux": {"TimeStamp": "2026-08-11 20:00", "Flux": "145"},
            "planetary-k": [["time_tag", "Kp"], ["2026-08-11 18:00", "5.33"]],
            "noaa-scales": {"0": {"DateStamp": "2026-08-11", "TimeStamp": "20:00:00",
                                           "R": {"Scale": "1", "Text": "minor"},
                                           "S": {"Scale": "0", "Text": "none"},
                                           "G": {"Scale": "1", "Text": "minor"}}},
        }
        monkeypatch.setattr(
            propagation_module, "_download_json",
            lambda url: next(value for key, value in responses.items() if key in url),
        )
        try:
            snapshot = fetch_space_weather(force_refresh=True)
            assert snapshot["solar_flux_10_7_cm"]["value_sfu"] == 145
            assert snapshot["planetary_k_index"]["value"] == 5.33
            assert snapshot["noaa_scales"]["R"]["scale"] == 1
            factors = {item["factor"] for item in space_weather_interpretation(snapshot)}
            assert factors == {"planetary_k_index", "10_7_cm_solar_flux", "NOAA_R_scale"}
        finally:
            propagation_module.RESULT_DIR = old_result


def test_station_local_signal_fingerprint_library():
    def observation(frequency=10_000_000, bandwidth=2800, entropy=0.42):
        return {
            "job_id": "classify-test", "requested_frequency_hz": frequency,
            "best_label": "am", "best_confidence": 0.7,
            "started_at": "2026-08-11T20:00:00+00:00",
            "features": {
                "carrier_prominence_db": -1.0, "sideband_imbalance_db": 0.4,
                "occupied_bandwidth_hz": bandwidth,
                "envelope_coefficient_of_variation": 0.22,
                "instantaneous_frequency_std_hz": 180,
                "spectral_entropy": entropy, "significant_peak_count": 3,
                "dominant_offset_hz": 20,
            },
        }
    with tempfile.TemporaryDirectory() as temporary:
        old_data = signal_library_module.DATA_DIR
        signal_library_module.DATA_DIR = Path(temporary)
        try:
            saved = save_fingerprint(name="WWV-like local reference",
                                     observation=observation(), frequency_tolerance_hz=2000)
            assert saved["exemplar_count"] == 1
            updated = add_exemplar(saved["fingerprint_id"], observation(bandwidth=3000))
            assert updated["exemplar_count"] == 2
            matched = match_fingerprints(observation(bandwidth=2900), minimum_similarity=0.7)
            assert matched["accepted"] is True
            assert matched["best_match"]["name"] == "WWV-like local reference"
            outside = match_fingerprints(observation(frequency=10_010_000))
            assert outside["match_count"] == 0
            assert len(list_fingerprints()) == 1
            delete_fingerprint(saved["fingerprint_id"])
            assert list_fingerprints() == []
        finally:
            signal_library_module.DATA_DIR = old_data


def test_recording_workspace_sessions_clips_search_and_comparison():
    from scipy.io import wavfile
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        old = (recording_workspace_module.DATA_DIR, recording_workspace_module.AUDIO_DIR,
               recording_workspace_module.PLOT_DIR, recording_workspace_module.RESULT_DIR)
        recording_workspace_module.DATA_DIR = root
        recording_workspace_module.AUDIO_DIR = root / "audio"
        recording_workspace_module.PLOT_DIR = root / "plots"
        recording_workspace_module.RESULT_DIR = root / "results"
        for path in (recording_workspace_module.AUDIO_DIR, recording_workspace_module.PLOT_DIR,
                     recording_workspace_module.RESULT_DIR):
            path.mkdir(parents=True)
        rate = 12_000
        time_axis = np.arange(rate * 2) / rate
        first_path, second_path = root / "first.wav", root / "second.wav"
        wavfile.write(first_path, rate, np.int16(np.sin(2*np.pi*1000*time_axis)*12000))
        wavfile.write(second_path, rate, np.int16(np.sin(2*np.pi*1200*time_axis)*10000))
        first_artifact = {"artifact_id": "art-first", "job_id": "job-1", "kind": "audio",
                          "filename": "first.wav", "mime_type": "audio/wav", "path": str(first_path)}
        try:
            session = create_session(name="Evening net", description="Interesting dispatch",
                                     tags=["vhf"], artifacts=[first_artifact])
            note = add_annotation(session["session_id"], text="Strong station identification",
                                  artifact_id="art-first", start_seconds=.2, end_seconds=.8,
                                  tags=["callsign"])
            assert note["annotation_id"].startswith("note-")
            bookmark = add_bookmark(session["session_id"], artifact=first_artifact,
                                    position_seconds=.5, label="ID begins")
            assert bookmark["position_seconds"] == .5
            assert search_sessions("dispatch callsign")[0]["session_id"] == session["session_id"]
            clip, info = extract_wav_clip(first_path, start_seconds=.25,
                                          duration_seconds=.5, label="id")
            assert clip.stat().st_size > 100 and abs(info["actual_duration_seconds"]-.5) < .01
            metrics, plot = compare_wav(first_path, second_path)
            assert metrics["second_spectral_centroid_hz"] > metrics["first_spectral_centroid_hz"]
            assert plot.stat().st_size > 1000
            exported = export_session(get_session(session["session_id"]))
            assert all(path.stat().st_size > 0 for path in exported)
            delete_session(session["session_id"])
            assert search_sessions("dispatch") == []
        finally:
            (recording_workspace_module.DATA_DIR, recording_workspace_module.AUDIO_DIR,
             recording_workspace_module.PLOT_DIR, recording_workspace_module.RESULT_DIR) = old


def test_catalog_marks_unfinished_jobs_interrupted():
    with tempfile.TemporaryDirectory() as temporary:
        catalog = Catalog(Path(temporary), index_existing=False)
        catalog.upsert_job("running-job", "monitor", "running")
        assert catalog.mark_interrupted_jobs() == 1
        assert catalog.get_job("running-job")["state"] == "interrupted"


def test_catalog_records_formal_schema_version_and_migration_history():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        status = local_catalog.schema_status()
        assert status["current_version"] == status["supported_version"] == 2
        assert status["up_to_date"] is True
        assert status["migrations"][0]["name"] == "baseline_v067"


def test_catalog_restart_recovery_includes_stopping_jobs():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        local_catalog.upsert_job("stopping-job", "band_scan", "stopping")
        assert local_catalog.mark_interrupted_jobs() == 1
        recovered = local_catalog.get_job("stopping-job")
        assert recovered["state"] == "interrupted"
        assert "service restarted" in recovered["error"]


def test_classifier_ranks_synthetic_modulations():
    sample_rate = 192_000
    time = np.arange(sample_rate) / sample_rate
    signals = {
        "am": (1 + 0.5 * np.sin(2 * np.pi * 1_000 * time)).astype(complex),
        "cw": np.ones(sample_rate, dtype=complex),
        "nfm": np.exp(1j * 2.5 * np.sin(2 * np.pi * 1_000 * time)),
        "usb": sum(
            amplitude * np.exp(2j * np.pi * frequency * time)
            for frequency, amplitude in ((500, 0.5), (1_200, 0.8), (2_200, 0.4))
        ),
        "lsb": sum(
            amplitude * np.exp(-2j * np.pi * frequency * time)
            for frequency, amplitude in ((500, 0.5), (1_200, 0.8), (2_200, 0.4))
        ),
    }
    for expected, iq in signals.items():
        features, *_ = extract_features(iq, sample_rate, 30_000, 8_192)
        ranking = classify_features(features)
        assert ranking[0]["label"] == expected


def test_bearer_auth_validation():
    token = "a" * 32
    assert validate_api_token(None) is None
    assert validate_api_token(token) == token
    assert authorized({"headers": [(b"authorization", f"Bearer {token}".encode())]}, token)
    assert not authorized({"headers": [(b"authorization", b"Bearer incorrect")]}, token)
    try:
        validate_api_token("too-short")
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a short API token")


def test_web_health_and_auth_boundary():
    import asyncio

    downstream_calls = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def request(app, path, headers=None):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await app(
            {"type": "http", "path": path, "method": "GET", "headers": headers or []},
            receive,
            send,
        )
        return messages

    token = "b" * 32
    app = RfWebApp(downstream, None, token, "0.40.0")
    health = asyncio.run(request(app, "/healthz"))
    denied = asyncio.run(request(app, "/mcp"))
    allowed = asyncio.run(
        request(app, "/mcp", [(b"authorization", f"Bearer {token}".encode())])
    )
    assert health[0]["status"] == 200
    assert denied[0]["status"] == 401
    assert allowed[0]["status"] == 204
    assert downstream_calls == ["/mcp"]


def test_authenticated_artifact_download_streams_catalog_file():
    import asyncio

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "result.json"
        path.write_bytes(b'{"result":true}\n')

        class FakeCatalog:
            def get_artifact(self, artifact_id):
                assert artifact_id == "art-deadbeef"
                return {
                    "artifact_id": artifact_id,
                    "path": str(path),
                    "filename": path.name,
                    "mime_type": "application/json",
                }

        async def downstream(scope, receive, send):
            raise AssertionError("Artifact request was incorrectly forwarded to MCP")

        async def request():
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            token = "c" * 32
            app = RfWebApp(downstream, FakeCatalog(), token, "0.40.0")
            await app(
                {
                    "type": "http",
                    "path": "/artifacts/art-deadbeef",
                    "method": "GET",
                    "headers": [(b"authorization", f"Bearer {token}".encode())],
                },
                receive,
                send,
            )
            return messages

        messages = asyncio.run(request())
        assert messages[0]["status"] == 200
        assert b"".join(item.get("body", b"") for item in messages[1:]) == path.read_bytes()


def _sstv_tone(frequency_hz, seconds, sample_rate_hz=12_000):
    samples = round(seconds * sample_rate_hz)
    time = np.arange(samples) / sample_rate_hz
    return np.sin(2 * np.pi * frequency_hz * time)


def _synthetic_vis(code, *, corrupt_parity=False):
    data_bits = [(code >> index) & 1 for index in range(7)]
    parity = sum(data_bits) % 2
    if corrupt_parity:
        parity ^= 1
    return np.concatenate([
        _sstv_tone(1900, 0.300), _sstv_tone(1200, 0.010),
        _sstv_tone(1900, 0.300), _sstv_tone(1200, 0.030),
        *[_sstv_tone(1100 if bit else 1300, 0.030) for bit in data_bits],
        _sstv_tone(1100 if parity else 1300, 0.030),
        _sstv_tone(1200, 0.030), np.zeros(2400),
    ])


def test_sstv_vis_detection_and_parity_validation():
    detected = detect_vis(_synthetic_vis(44))
    assert detected["detected"] is True
    assert detected["vis_code"] == 44
    assert detected["mode"] == SSTV_MODES[44] == "Martin M1"
    assert detected["parity_valid"] is True

    corrupted = detect_vis(_synthetic_vis(44, corrupt_parity=True))
    assert corrupted["detected"] is True
    assert corrupted["vis_code"] == 44
    assert corrupted["parity_valid"] is False


def test_sstv_headless_bootstrap_replaces_terminal_size_call(monkeypatch):
    assert "sstv.common.get_terminal_size" in HEADLESS_DECODER_BOOTSTRAP
    captured = {}

    def fake_run(command, **options):
        captured["command"], captured["options"] = command, options
        return object()

    monkeypatch.setattr(sstv_module, "sstv_decoder_path", lambda: "/venv/bin/sstv")
    monkeypatch.setattr(sstv_module.subprocess, "run", fake_run)
    run_sstv_decoder(Path("capture.wav"), Path("image.png"))
    assert captured["command"][:3] == [
        sstv_module.sys.executable, "-c", HEADLESS_DECODER_BOOTSTRAP,
    ]
    assert captured["command"][-4:] == ["-d", "capture.wav", "-o", "image.png"]
    assert captured["options"]["capture_output"] is True


def test_sstv_nfm_demodulation_recovers_audio_tone():
    sample_rate_hz = 768_000
    duration_seconds = 0.25
    count = round(sample_rate_hz * duration_seconds)
    time = np.arange(count) / sample_rate_hz
    audio_hz, deviation_hz, offset_hz = 1900, 2500, -10_000
    phase = (2 * np.pi * offset_hz * time +
             (deviation_hz / audio_hz) * np.sin(2 * np.pi * audio_hz * time))
    iq = np.exp(1j * phase).astype(np.complex64)
    interleaved = np.empty(count * 2, dtype="<f4")
    interleaved[0::2], interleaved[1::2] = iq.real, iq.imag
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "capture.iq"
        interleaved.tofile(path)
        audio = iq_to_nfm_audio(
            path, sample_count=count, sample_rate_hz=sample_rate_hz,
            offset_hz=offset_hz,
        )
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), 1 / 12_000)
    dominant = frequencies[np.argmax(spectrum[1:]) + 1]
    assert abs(dominant - audio_hz) < 10


def test_sstv_gallery_catalog_round_trip():
    from PIL import Image

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image_path = root / "sstv" / "job-1" / "sstv-image.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (320, 256), "navy").save(image_path)
        local_catalog = Catalog(root, index_existing=False)
        stored = local_catalog.add_sstv_image({
            "job_id": "job-1", "frequency_hz": 14_230_000,
            "receiver_mode": "usb", "sstv_mode": "Martin M1", "vis_code": 44,
            "vis_parity_valid": True, "width": 320, "height": 256,
            "quality": 0.75, "image_path": str(image_path), "audio_path": None,
            "captured_at": "2026-08-10T19:40:07+00:00", "duration_seconds": 130,
            "decoder_output": "ok",
        })
        assert stored["vis_parity_valid"] is True
        assert Path(stored["image_path"]) == image_path.resolve()
        assert local_catalog.get_sstv_image(stored["image_id"])["width"] == 320
        assert local_catalog.list_sstv_images(
            frequency_hz=14_230_000, sstv_mode="martin m1"
        )[0]["image_id"] == stored["image_id"]


def test_sstv_preset_normalization_and_no_signal_outcome():
    name, preset_type, _, config = normalize_preset(
        name="ISS SSTV watcher", preset_type="sstv", description="",
        config={
            "frequency_hz": 145_800_000, "duration_seconds": 180,
            "receiver_mode": "nfm", "retain_audio": False,
        },
    )
    assert name == "ISS SSTV watcher"
    assert preset_type == "sstv"
    assert config == {
        "frequency_hz": 145_800_000, "duration_seconds": 180.0,
        "receiver_mode": "nfm", "retain_audio": False,
        "retain_iq": False, "deduplicate": True,
    }
    assert decoder_indicates_no_signal(
        "[sstv] Couldn't find SSTV header in the given audio file",
        {"detected": False},
    ) is True
    assert decoder_indicates_no_signal("decoder crashed", {"detected": False}) is False


def test_sstv_duplicate_detection_and_activity_summary():
    from PIL import Image

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        gallery = root / "sstv"
        gallery.mkdir()
        first_path = gallery / "first.png"
        second_path = gallery / "second.png"
        image = Image.new("RGB", (320, 256), "navy")
        image.save(first_path)
        image.save(second_path)
        fingerprint = image_fingerprint(image)
        assert len(fingerprint) == 64
        local_catalog = Catalog(root, index_existing=False)

        def record(job_id, path, duplicate_of=None):
            return local_catalog.add_sstv_image({
                "job_id": job_id, "frequency_hz": 145_800_000,
                "receiver_mode": "nfm", "sstv_mode": "Robot 36", "vis_code": 8,
                "vis_parity_valid": True, "width": 320, "height": 256,
                "quality": 0.8, "image_path": str(path), "audio_path": None,
                "captured_at": "2026-08-11T04:22:26+00:00", "duration_seconds": 180,
                "decoder_output": "ok", "image_hash": fingerprint,
                "duplicate_of": duplicate_of,
            })

        first = record("job-1", first_path)
        match = local_catalog.find_sstv_duplicate(
            fingerprint, frequency_hz=145_800_000
        )
        assert match["image_id"] == first["image_id"]
        assert match["hash_distance"] == 0
        record("job-2", second_path, first["image_id"])
        assert len(local_catalog.list_sstv_images(include_duplicates=False)) == 1
        summary = local_catalog.sstv_activity_summary()
        assert summary["images"] == 2
        assert summary["unique_images"] == 1
        assert summary["duplicates"] == 1


def test_sstv_gallery_schema_migrates_from_v023():
    import sqlite3

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        database = root / "rf-mcp.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("""
                CREATE TABLE sstv_images (
                    image_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
                    frequency_hz INTEGER NOT NULL, receiver_mode TEXT NOT NULL,
                    sstv_mode TEXT, vis_code INTEGER, vis_parity_valid INTEGER,
                    width INTEGER NOT NULL, height INTEGER NOT NULL, quality REAL,
                    image_path TEXT NOT NULL UNIQUE, audio_path TEXT,
                    captured_at TEXT NOT NULL, duration_seconds REAL NOT NULL,
                    decoder_output TEXT NOT NULL DEFAULT ''
                )
            """)
            connection.execute("""
                CREATE TABLE alert_events (
                    event_id TEXT PRIMARY KEY, rule_id TEXT, rule_name TEXT NOT NULL,
                    schedule_id TEXT, job_id TEXT, observation_label TEXT,
                    frequency_hz INTEGER, message TEXT NOT NULL,
                    details_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    acknowledged_at TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE webhook_destinations (
                    destination_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE, url TEXT NOT NULL,
                    signing_secret TEXT, all_rules INTEGER NOT NULL DEFAULT 1,
                    rule_id TEXT, enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE satellite_watches (
                    watch_id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    satellite_name TEXT NOT NULL, norad_id INTEGER NOT NULL,
                    tle_line1 TEXT NOT NULL, tle_line2 TEXT NOT NULL,
                    latitude_deg REAL NOT NULL, longitude_deg REAL NOT NULL,
                    elevation_m REAL NOT NULL DEFAULT 0, frequency_hz INTEGER NOT NULL,
                    receiver_mode TEXT NOT NULL, minimum_elevation_deg REAL NOT NULL DEFAULT 10,
                    lead_seconds INTEGER NOT NULL DEFAULT 60,
                    trail_seconds INTEGER NOT NULL DEFAULT 30,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE satellite_passes (
                    pass_id TEXT PRIMARY KEY, watch_id TEXT,
                    satellite_name TEXT NOT NULL, norad_id INTEGER NOT NULL,
                    aos_at TEXT NOT NULL, tca_at TEXT NOT NULL, los_at TEXT NOT NULL,
                    start_at TEXT NOT NULL, stop_at TEXT NOT NULL,
                    maximum_elevation_deg REAL NOT NULL, prediction_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'planned', job_id TEXT, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(watch_id, aos_at)
                )
            """)
        Catalog(root, index_existing=False)
        with sqlite3.connect(database) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sstv_images)")
            }
            event_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(alert_events)")
            }
            destination_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(webhook_destinations)")
            }
            satellite_watch_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(satellite_watches)")
            }
            satellite_pass_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(satellite_passes)")
            }
        assert {"image_hash", "duplicate_of", "source_preset_id",
                "source_schedule_id", "source_watch_id",
                "source_satellite_pass_id"} <= columns
        assert {"event_type", "sstv_rule_id", "satellite_watch_id",
                "satellite_pass_id"} <= event_columns
        assert {"sstv_rule_id", "satellite_watch_id"} <= destination_columns
        assert {"auto_refresh", "tle_source", "refresh_interval_seconds",
                "last_tle_refresh_status", "next_tle_refresh_at",
                "notify_before_seconds", "doppler_correction_mode",
                "doppler_step_seconds"} <= satellite_watch_columns
        assert {"notify_at", "prepass_event_id", "outcome_event_id",
                "doppler_plan_json", "doppler_plot_path",
                "doppler_artifact_id"} <= satellite_pass_columns


def test_streaming_sstv_detector_triggers_completes_and_rejects_bad_parity():
    detector = SSTVStreamDetector(
        pre_trigger_seconds=3, mode_capture_seconds={8: 2.0}, rearm_seconds=0.1
    )
    audio = np.concatenate((_synthetic_vis(8), np.zeros(24_000)))
    events = []
    for start in range(0, len(audio), 3_000):
        events.extend(detector.feed(audio[start:start + 3_000]))
    assert [event["event"] for event in events] == ["triggered", "complete"]
    complete = events[-1]
    assert complete["vis"]["vis_code"] == 8
    assert len(complete["audio"]) == 24_000

    rejected = SSTVStreamDetector(mode_capture_seconds={8: 2.0})
    bad_events = []
    bad_audio = np.concatenate((_synthetic_vis(8, corrupt_parity=True), np.zeros(12_000)))
    for start in range(0, len(bad_audio), 3_000):
        bad_events.extend(rejected.feed(bad_audio[start:start + 3_000]))
    assert any(event["event"] == "rejected" for event in bad_events)
    assert rejected.false_triggers == 1


def test_streaming_nfm_demodulator_recovers_tone_across_chunks():
    sample_rate_hz = 768_000
    count = sample_rate_hz
    time = np.arange(count) / sample_rate_hz
    audio_hz, deviation_hz, offset_hz = 1900, 2500, -10_000
    phase = (2 * np.pi * offset_hz * time +
             (deviation_hz / audio_hz) * np.sin(2 * np.pi * audio_hz * time))
    iq = np.exp(1j * phase).astype(np.complex64)
    interleaved = np.empty(count * 2, dtype="<f4")
    interleaved[0::2], interleaved[1::2] = iq.real, iq.imag
    demodulator = StreamingDemodulator(mode="nfm", offset_hz=offset_hz)
    midpoint = len(interleaved) // 2
    audio = np.concatenate((
        demodulator.process(interleaved[:midpoint]),
        demodulator.process(interleaved[midpoint:]),
    ))
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), 1 / 12_000)
    dominant = frequencies[np.argmax(spectrum[1:]) + 1]
    assert abs(dominant - audio_hz) < 5


def test_sstv_watcher_preset_normalization():
    _, preset_type, _, config = normalize_preset(
        name="Live ISS SSTV", preset_type="sstv_watch", description="",
        config={"frequency_hz": 145_800_000, "watch_duration_seconds": 3600},
    )
    assert preset_type == "sstv_watch"
    assert config == {
        "frequency_hz": 145_800_000, "receiver_mode": "nfm",
        "watch_duration_seconds": 3600.0, "rearm": True,
        "retain_audio": True, "deduplicate": True,
    }


def test_airspy_stream_uses_binary_stdout_and_bounded_chunks(monkeypatch):
    import io
    import threading

    sample_count = round(768_000 * 0.1)
    payload = np.zeros(sample_count * 2, dtype="<f4").tobytes()
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(payload)
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **options):
        captured["command"], captured["options"] = command, options
        return FakeProcess()

    monkeypatch.setattr(airspyhf_module.shutil, "which", lambda _: "/usr/bin/airspyhf_rx")
    monkeypatch.setattr(airspyhf_module.subprocess, "Popen", fake_popen)
    chunks = list(stream_iq_chunks(
        145_810_000, duration_seconds=10, stop_event=threading.Event(),
        chunk_seconds=0.1,
    ))
    assert captured["command"][1:3] == ["-r", "stdout"]
    assert captured["options"]["bufsize"] == 0
    assert len(chunks) == 1
    assert len(chunks[0]) == sample_count * 2


def test_sstv_watcher_session_decodes_trigger_and_persists_gallery(monkeypatch):
    from PIL import Image
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        local_catalog = Catalog(root, index_existing=False)
        watcher_rule = local_catalog.save_sstv_alert_rule(
            name="Watcher image", frequency_hz=145_800_000, sstv_mode="Robot 36",
            minimum_quality=0.4, unique_only=True, enabled=True,
        )
        local_catalog.save_webhook_destination(
            name="Watcher webhook", url="https://example.com/sstv-watcher",
            signing_secret=None, all_rules=False, rule_id=None,
            sstv_rule_id=watcher_rule["rule_id"], satellite_watch_id=None,
            enabled=True,
        )
        audio = np.concatenate((_synthetic_vis(8), np.zeros(45 * 12_000)))
        audio_chunks = [audio[index:index + 6000] for index in range(0, len(audio), 6000)]

        class FakeDemodulator:
            def __init__(self, **kwargs):
                self.index = 0

            def process(self, _values, **_kwargs):
                chunk = audio_chunks[self.index]
                self.index += 1
                return chunk

        def fake_stream(*args, **kwargs):
            for _ in audio_chunks:
                yield np.zeros(2, dtype="<f4")

        def fake_decoder(_wav_path, png_path):
            Image.new("RGB", (320, 240), "navy").save(png_path)
            return SimpleNamespace(returncode=0, stdout="decoded", stderr="")

        monkeypatch.setattr(sstv_watcher_module, "catalog", local_catalog)
        monkeypatch.setattr(sstv_watcher_module, "SSTV_DIR", root / "sstv")
        monkeypatch.setattr(sstv_watcher_module, "StreamingDemodulator", FakeDemodulator)
        monkeypatch.setattr(sstv_watcher_module, "stream_iq_chunks", fake_stream)
        monkeypatch.setattr(sstv_watcher_module, "run_sstv_decoder", fake_decoder)
        job = WatchJob("sstv-watch-test", {
            "frequency_hz": 145_800_000, "receiver_mode": "nfm",
            "watch_duration_seconds": 60, "rearm": True,
            "retain_audio": False, "deduplicate": True,
            "pre_trigger_seconds": 3.0,
            "source_satellite_watch_id": "satwatch-test",
            "source_satellite_pass_id": "satpass-test",
        })
        local_catalog.upsert_job(
            job.job_id, "sstv_watch", "queued", config=job.config,
            created_at=job.created_at,
        )
        SSTVWatcherManager()._run(job)
        assert job.state == "completed"
        assert job.trigger_count == 1
        assert job.decode_failure_count == 0
        assert len(job.images) == 1
        assert job.images[0]["outcome"] == "decoded"
        assert job.images[0]["alert_event_count"] == 1
        assert local_catalog.list_sstv_images()[0]["source_watch_id"] == job.job_id
        assert local_catalog.list_sstv_images()[0]["source_satellite_pass_id"] == "satpass-test"
        assert len(local_catalog.list_alert_events(event_type="sstv_image")) == 1
        assert len(local_catalog.list_webhook_deliveries()) == 1


def test_sstv_alert_rule_validation_and_matching():
    rule = normalize_sstv_alert_rule(
        name="New ISS image", frequency_hz=145_800_000,
        sstv_mode="robot 36", minimum_quality=0.4,
        unique_only=True, enabled=True,
    )
    assert rule["sstv_mode"] == "Robot 36"
    matched, message = evaluate_sstv_rule(rule, {
        "frequency_hz": 145_800_000, "sstv_mode": "Robot 36",
        "quality": 0.8, "duplicate_of": None,
    })
    assert matched is True
    assert "New ISS image" in message
    assert evaluate_sstv_rule(rule, {
        "frequency_hz": 145_800_000, "sstv_mode": "Robot 36",
        "quality": 0.8, "duplicate_of": "sstv-original",
    })[0] is False
    for kwargs in (
        {**rule, "minimum_quality": 1.1},
        {**rule, "sstv_mode": "Unsupported 99"},
    ):
        try:
            normalize_sstv_alert_rule(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an invalid SSTV alert rule")


def test_sstv_alert_event_queues_filtered_signed_webhook_and_is_acknowledged():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        local_catalog.upsert_job("sstv-child", "sstv_decode", "completed")
        rule = local_catalog.save_sstv_alert_rule(
            name="Unique ISS", frequency_hz=145_800_000, sstv_mode=None,
            minimum_quality=0.5, unique_only=True, enabled=True,
        )
        destination = local_catalog.save_webhook_destination(
            name="SSTV Receiver", url="https://example.com/sstv",
            signing_secret="correct-horse-battery-staple", all_rules=False,
            rule_id=None, sstv_rule_id=rule["rule_id"], satellite_watch_id=None,
            enabled=True,
        )
        assert destination["sstv_rule_id"] == rule["rule_id"]
        image = {
            "image_id": "sstv-image-1", "job_id": "sstv-child",
            "source_watch_id": "sstv-watch-1", "frequency_hz": 145_800_000,
            "receiver_mode": "nfm", "sstv_mode": "Robot 36", "vis_code": 8,
            "vis_parity_valid": True, "width": 320, "height": 240,
            "quality": 0.8, "captured_at": "2026-08-11T20:00:00+00:00",
            "duration_seconds": 45, "duplicate_of": None,
            "image_artifact_id": "art-image", "image_download_path": "/artifacts/art-image",
            "decoder_output": "this must not enter the webhook",
        }
        events = evaluate_sstv_image(local_catalog, image)
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "sstv_image"
        assert event["sstv_rule_id"] == rule["rule_id"]
        assert event["webhook_delivery_count"] == 1
        delivery = local_catalog.list_webhook_deliveries(event_id=event["event_id"])[0]
        assert delivery["payload"]["schema"] == "rf-mcp.sstv-alert.v1"
        payload_image = delivery["payload"]["event"]["details"]["image"]
        assert payload_image["image_download_path"] == "/artifacts/art-image"
        assert "decoder_output" not in payload_image
        assert local_catalog.list_alert_events(event_type="sstv_image")[0]["event_id"] == event["event_id"]
        acknowledged = local_catalog.acknowledge_alert_event(event["event_id"])
        assert acknowledged["acknowledged"] is True
        deleted = local_catalog.delete_sstv_alert_rule(rule["rule_id"])
        assert deleted["name"] == "Unique ISS"
        assert local_catalog.get_webhook_destination(destination["destination_id"])[
            "sstv_rule_id"
        ] is None
        assert local_catalog.get_alert_event(event["event_id"])["rule_name"] == "Unique ISS"


ISS_TLE_1 = "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082"
ISS_TLE_2 = "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473"


def _satellite_watch_values():
    return normalize_satellite_watch(
        name="ISS SSTV", satellite_name="ISS (ZARYA)", norad_id=25544,
        tle_line1=ISS_TLE_1, tle_line2=ISS_TLE_2,
        latitude_deg=40.8939, longitude_deg=-83.8917, elevation_m=250,
        frequency_hz=145_800_000, receiver_mode="nfm",
        minimum_elevation_deg=30, lead_seconds=60, trail_seconds=30,
        enabled=True,
    )


def test_coordinate_text_and_assisted_satellite_catalog(monkeypatch):
    assert parse_coordinate("33.96", axis="latitude") == 33.96
    assert abs(parse_coordinate("117 57 0 W", axis="longitude") + 117.95) < 1e-9
    transmitter_json = json.dumps([
        {"uuid": "packet-id", "description": "APRS AFSK telemetry",
         "downlink_low": 145_825_000, "downlink_high": 145_825_000,
         "mode": "AFSK", "baud": 1200, "service": "Amateur", "status": "active"},
        {"uuid": "s-band-id", "description": "S-band", "downlink_low": 2_400_000_000,
         "mode": "GMSK", "baud": 9600, "status": "active"},
    ])

    def fake_download(url, **kwargs):
        if "satnogs" in url:
            return transmitter_json
        return "ISS (ZARYA)\n" + ISS_TLE_1 + "\n" + ISS_TLE_2 + "\n"

    monkeypatch.setattr(satellite_catalog_module, "_download_text", fake_download)
    found = search_catalog(query="ISS", limit=10)
    assert found["satellites"][0]["norad_id"] == 25544
    entry = get_catalog_entry("25544")
    assert entry["compatible_transmitter_count"] == 1
    assert entry["suggested_downlinks"][0]["mode"] == "ax25_afsk1200"
    assert entry["transmitters"][1]["decoder_support"] == "unavailable"
    chosen = selected_downlinks(entry, ["packet-id"])
    assert chosen[0]["frequency_hz"] == 145_825_000
    assert "catalog_transmitter_id" not in chosen[0]


def test_saved_observer_locations_and_opportunity_planner(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        old_directory = satellite_planner_module.SATELLITE_DIR
        satellite_planner_module.SATELLITE_DIR = Path(temporary)
        try:
            home = save_location(
                name="MiniRackDisplay", latitude_deg="33 57 36 N",
                longitude_deg="117.95 W", elevation_m=120, make_default=True,
            )
            assert abs(home["latitude_deg"] - 33.96) < 1e-9
            assert abs(home["longitude_deg"] + 117.95) < 1e-9
            assert get_location()["name"] == "MiniRackDisplay"
            assert list_locations()[0]["is_default"] is True
            monkeypatch.setattr(satellite_planner_module, "catalog_orbital_records",
                                lambda **kwargs: [{"name": "TESTSAT", "norad_id": 12345,
                                                   "tle_line1": "line1", "tle_line2": "line2"}])
            monkeypatch.setattr(satellite_planner_module, "predict_passes", lambda *args, **kwargs: [{
                "aos": {"at": "2026-08-12T01:00:00+00:00"},
                "tca": {"at": "2026-08-12T01:05:00+00:00"},
                "los": {"at": "2026-08-12T01:10:00+00:00"},
                "maximum_elevation_deg": 62.5, "duration_seconds": 600,
            }])
            monkeypatch.setattr(satellite_planner_module, "get_catalog_entry", lambda value: {
                "suggested_downlinks": [{"frequency_hz": 145825000,
                                          "mode": "ax25_afsk1200"}],
            })
            result = plan_observations(location=home, category="amateur")
            assert result["opportunity_count"] == 1
            assert result["opportunities"][0]["rank"] == 1
            assert result["opportunities"][0]["best_downlink"]["frequency_hz"] == 145825000
            assert delete_location(home["location_id"])["deleted"] is True
            assert list_locations() == []
        finally:
            satellite_planner_module.SATELLITE_DIR = old_directory


def test_satellite_watch_validation_and_skyfield_pass_prediction():
    watch = _satellite_watch_values()
    passes = predict_passes(
        watch, start=datetime(2014, 1, 23, tzinfo=timezone.utc), hours=24, limit=10
    )
    assert len(passes) == 2
    assert passes[0]["aos"]["at"].startswith("2014-01-23T06:25")
    assert passes[0]["maximum_elevation_deg"] >= 30
    assert passes[0]["duration_seconds"] > 120
    assert passes[0]["tle_stale"] is False
    assert passes[0]["aos"]["doppler_shift_hz"] > 0
    assert passes[0]["los"]["doppler_shift_hz"] < 0
    assert passes[0]["aos"]["corrected_receive_frequency_hz"] > 145_800_000
    plan = build_doppler_plan(watch, passes[0], step_seconds=10)
    assert len(plan) > 10
    assert plan[0]["doppler_shift_hz"] > 0
    assert plan[-1]["doppler_shift_hz"] < 0
    midpoint = datetime.fromisoformat(plan[len(plan) // 2]["at"])
    interpolated = doppler_frequency_at(plan, midpoint)
    assert interpolated == plan[len(plan) // 2]["corrected_receive_frequency_hz"]
    try:
        normalize_satellite_watch(**{
            **{key: value for key, value in watch.items() if key != "tle_epoch_at"},
            "norad_id": 99999,
        })
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Accepted a NORAD ID that does not match the TLE")


def test_satellite_multi_downlinks_and_observation_catalog():
    downlinks = normalize_satellite_downlinks([
        {"downlink_id": "voice", "label": "Voice", "frequency_hz": 145_800_000,
         "mode": "nfm_audio", "receiver_mode": "nfm", "priority": 20,
         "enabled": True, "retain_audio": True},
        {"downlink_id": "packet", "label": "Packet", "frequency_hz": 145_825_000,
         "mode": "ax25_afsk1200", "receiver_mode": "nfm", "priority": 10,
         "enabled": True, "retain_audio": False},
    ])
    assert [item["downlink_id"] for item in downlinks] == ["packet", "voice"]
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        result_path = Path(temporary) / "observation.json"
        result_path.write_text("{}")
        stored = local_catalog.add_satellite_observation({
            "job_id": "sat-rx-test", "pass_id": "pass-test", "watch_id": "watch-test",
            "satellite_name": "TESTSAT", "downlink_id": "packet",
            "downlink_label": "Packet", "mode": "ax25_afsk1200",
            "nominal_frequency_hz": 145_825_000, "outcome": "completed",
            "packet_count": 2, "valid_packet_count": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 60, "result_json_path": result_path,
            "details": {"ax25": {"frame_count": 2, "frames": [{
                "source": "N0CALL", "destination": "CQ", "digipeaters": [],
                "control": 3, "pid": 240, "fcs_valid": True,
                "information_text": "TEST", "information_hex": "54455354",
                "frame_hex": "001122",
            }]}},
        })
        assert stored["details"]["ax25"]["frame_count"] == 2
        summary = local_catalog.satellite_activity_summary(watch_id="watch-test")
        assert summary["observations"] == 1
        assert summary["packet_count"] == 2
        monkeypatch_path = satellite_receiver_module.SATELLITE_DIR
        satellite_receiver_module.SATELLITE_DIR = Path(temporary)
        try:
            export = Path(export_satellite_telemetry([stored], output_format="jsonl"))
            exported = json.loads(export.read_text().strip())
            assert exported["frame_hex"] == "001122"
            assert exported["fcs_valid"] is True
        finally:
            satellite_receiver_module.SATELLITE_DIR = monkeypatch_path


def test_satellite_capture_only_receiver_job(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        monkeypatch.setattr(satellite_receiver_module, "catalog", local_catalog)
        monkeypatch.setattr(satellite_receiver_module, "SATELLITE_DIR", Path(temporary) / "sat")
        monkeypatch.setattr(
            satellite_receiver_module, "stream_iq_chunks",
            lambda *args, **kwargs: iter([np.zeros(7_680, dtype="<f4")]),
        )
        manager = SatelliteReceiverManager()
        downlink = normalize_satellite_downlinks([{
            "downlink_id": "beacon", "label": "Beacon", "frequency_hz": 145_900_000,
            "mode": "capture_only", "receiver_mode": "nfm", "enabled": True,
        }])[0]
        started = manager.start(
            watch={"watch_id": "watch-test", "satellite_name": "TESTSAT",
                   "doppler_correction_mode": "off"},
            pass_record={"pass_id": "pass-test", "doppler_plan": []},
            downlink=downlink, duration_seconds=30,
        )
        manager._jobs[started["job_id"]].thread.join(timeout=5)
        result = manager.results(started["job_id"])
        assert result["state"] == "completed"
        assert result["result"]["mode"] == "capture_only"
        assert result["result"]["details"]["complex_sample_count"] == 60


def test_satellite_pass_performance_scoring_summary_plot_and_export():
    pass_record = {
        "pass_id": "pass-good", "watch_id": "watch-1", "satellite_name": "TESTSAT",
        "state": "completed", "aos_at": "2026-08-11T01:00:00+00:00",
        "maximum_elevation_deg": 62.0, "prediction": {"duration_seconds": 300},
        "selected_downlink": {"downlink_id": "packet", "label": "Packet 9600",
                              "mode": "ax25_g3ruh9600", "frequency_hz": 145_900_000},
    }
    observations = [{"mode": "ax25_g3ruh9600", "packet_count": 10,
                     "valid_packet_count": 9, "duration_seconds": 300,
                     "details": {"rms": 0.1, "peak": 0.4}}]
    good = score_satellite_pass(pass_record, observations, telemetry_value_count=20)
    assert good["performance_score"] == 97.5
    assert good["valid_fcs_rate"] == 0.9
    weak = score_satellite_pass(
        {**pass_record, "pass_id": "pass-weak", "aos_at": "2026-08-11T02:00:00+00:00",
         "state": "failed", "maximum_elevation_deg": 20.0}, [], 0
    )
    summary = summarize_pass_performance([good, weak])
    assert summary["pass_count"] == 2
    assert summary["recommendation"]["downlink_id"] == "packet"
    sstv = score_satellite_pass(
        {**pass_record, "pass_id": "pass-sstv",
         "selected_downlink": {"downlink_id": "sstv", "label": "SSTV", "mode": "sstv",
                               "frequency_hz": 145_800_000}},
        [{"mode": "sstv", "duration_seconds": 300, "packet_count": 0,
          "valid_packet_count": 0, "details": {}}], 0,
        [{"quality": 0.8}],
    )
    assert sstv["sstv_image_count"] == 1
    assert sstv["score_components"]["image_quality"] == 0.8
    with tempfile.TemporaryDirectory() as temporary:
        old_plot, old_sat = satellite_performance_module.PLOT_DIR, satellite_performance_module.SATELLITE_DIR
        satellite_performance_module.PLOT_DIR = Path(temporary)
        satellite_performance_module.SATELLITE_DIR = Path(temporary)
        try:
            assert Path(save_pass_performance_plot([good, weak], title="Passes")).stat().st_size > 1000
            exported = Path(export_pass_performance([good, weak], output_format="csv"))
            assert "performance_score" in exported.read_text()
        finally:
            satellite_performance_module.PLOT_DIR = old_plot
            satellite_performance_module.SATELLITE_DIR = old_sat


def test_satellite_telemetry_schema_decode_persist_plot_and_export():
    normalized = normalize_telemetry_schema(
        name="Test beacon", satellite_name="TESTSAT",
        match={"source": "N0CALL-2", "pid": 240, "payload_prefix_hex": "aa"},
        fields=[
            {"name": "battery_voltage", "label": "Battery", "type": "uint16",
             "offset": 1, "byte_order": "big", "scale": 0.001, "unit": "V"},
            {"name": "temperature", "type": "int8", "offset": 3, "unit": "C"},
            {"name": "status", "type": "ascii", "offset": 4, "length": 2},
        ],
    )
    sample = bytes.fromhex("aa0ce4fb4f4b")
    fields = decode_telemetry_payload(normalized, sample)
    assert abs(fields[0]["numeric_value"] - 3.3) < 1e-9
    assert fields[1]["numeric_value"] == -5
    assert fields[2]["text_value"] == "OK"
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        schema = local_catalog.save_satellite_telemetry_schema(**normalized)
        alert_rule = normalize_telemetry_alert_rule(
            catalog=local_catalog, name="Battery high", schema_id_or_name=schema["schema_id"],
            field_name="battery_voltage", condition_type="above", threshold_high=3.2,
            cooldown_seconds=3600, enabled=True,
        )
        rule = local_catalog.save_satellite_telemetry_alert_rule(**alert_rule)
        local_catalog.save_webhook_destination(
            name="All telemetry", url="https://example.com/telemetry", signing_secret=None,
            all_rules=True, rule_id=None, sstv_rule_id=None, satellite_watch_id=None,
            enabled=True,
        )
        result_path = Path(temporary) / "packet.json"
        result_path.write_text("{}")
        observation = local_catalog.add_satellite_observation({
            "job_id": "sat-rx-telemetry", "pass_id": "pass-1", "watch_id": "watch-1",
            "satellite_name": "TESTSAT", "downlink_id": "packet",
            "downlink_label": "Packet", "mode": "ax25_g3ruh9600",
            "nominal_frequency_hz": 145_900_000, "outcome": "completed",
            "packet_count": 1, "valid_packet_count": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": 30,
            "result_json_path": result_path, "details": {"ax25": {"frames": [{
                "source": "N0CALL-2", "destination": "CQ", "pid": 240,
                "fcs_valid": True, "information_hex": sample.hex(),
            }]}},
        })
        decoded = decode_observation_telemetry(local_catalog, observation)
        assert decoded["matched_frame_count"] == 1
        assert decoded["value_count"] == 3
        assert decoded["telemetry_alert_count"] == 1
        events = local_catalog.list_alert_events(event_type="satellite_telemetry")
        assert len(events) == 1
        assert events[0]["telemetry_rule_id"] == rule["rule_id"]
        assert len(local_catalog.list_webhook_deliveries(event_id=events[0]["event_id"])) == 1
        repeated = decode_observation_telemetry(local_catalog, observation)
        assert repeated["telemetry_alert_count"] == 0
        assert any(item.get("suppressed_by_cooldown")
                   for item in repeated["alert_evaluations"])
        values = local_catalog.list_satellite_telemetry_values(schema_id=schema["schema_id"])
        assert {item["field_name"] for item in values} == {
            "battery_voltage", "temperature", "status"
        }
        old_plot, old_sat = satellite_telemetry_module.PLOT_DIR, satellite_telemetry_module.SATELLITE_DIR
        satellite_telemetry_module.PLOT_DIR = Path(temporary)
        satellite_telemetry_module.SATELLITE_DIR = Path(temporary)
        try:
            assert Path(save_telemetry_plot(values, title="Test telemetry")).stat().st_size > 1000
            exported = Path(export_decoded_telemetry(values, output_format="csv"))
            assert "battery_voltage" in exported.read_text()
        finally:
            satellite_telemetry_module.PLOT_DIR = old_plot
            satellite_telemetry_module.SATELLITE_DIR = old_sat

    change = {"name": "Jump", "condition_type": "absolute_change",
              "change_threshold": 2.0}
    evaluation = evaluate_telemetry_alert_rule(
        change, {"field_name": "temperature", "numeric_value": 12, "unit": "C"},
        {"numeric_value": 9},
    )
    assert evaluation["matched"] is True
    assert evaluation["metric"] == 3


def test_satellite_pass_scheduler_persists_and_launches_pass_aware_watcher():
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        watch_values = _satellite_watch_values()
        watch_values["doppler_correction_mode"] = "digital"
        watch_values["doppler_step_seconds"] = 10
        watch = local_catalog.save_satellite_watch(**watch_values)
        now = datetime(2014, 1, 23, 6, 24, 37, tzinfo=timezone.utc)
        predicted = predict_passes(watch, start=now, hours=1, limit=1)[0]
        predicted["doppler_track"] = build_doppler_plan(watch, predicted)
        pass_record = local_catalog.save_satellite_pass(watch, predicted)
        local_catalog.set_satellite_watch_enabled(watch["watch_id"], False)
        assert local_catalog.get_satellite_pass(pass_record["pass_id"])["state"] == "superseded"
        watch = local_catalog.set_satellite_watch_enabled(watch["watch_id"], True)
        pass_record = local_catalog.save_satellite_pass(watch, predicted)
        assert pass_record["state"] == "planned"
        monkeypatch_plot = satellite_module.PLOT_DIR
        satellite_module.PLOT_DIR = Path(temporary)
        try:
            plot_path = Path(save_doppler_plot(pass_record))
            assert plot_path.exists() and plot_path.stat().st_size > 1000
            artifact = local_catalog.register_artifact(plot_path, "satellite_doppler_plot")
            pass_record = local_catalog.set_satellite_doppler_plot(
                pass_record["pass_id"], path=str(plot_path),
                artifact_id=artifact["artifact_id"],
            )
            assert pass_record["doppler_artifact_id"] == artifact["artifact_id"]
        finally:
            satellite_module.PLOT_DIR = monkeypatch_plot
        local_catalog.save_webhook_destination(
            name="ISS pass receiver", url="https://example.com/iss-pass",
            signing_secret="correct-horse-battery-staple", all_rules=False,
            rule_id=None, sstv_rule_id=None, satellite_watch_id=watch["watch_id"],
            enabled=True,
        )
        launched = {}

        def fake_launch(**kwargs):
            launched.update(kwargs)
            local_catalog.upsert_job("sstv-watch-pass", "sstv_watch", "queued")
            return {"job_id": "sstv-watch-pass"}

        manager = SatellitePassScheduler(
            local_catalog, fake_launch, lambda: False, poll_seconds=1
        )
        manager._last_prediction_at = now.isoformat()
        outcomes = manager.tick(now)
        pass_outcome = next(item for item in outcomes if item.get("state") == "launched")
        assert pass_outcome["job_id"] == "sstv-watch-pass"
        assert launched["watch"]["watch_id"] == watch["watch_id"]
        assert launched["pass_record"]["pass_id"] == pass_record["pass_id"]
        assert launched["watch"]["doppler_correction_mode"] == "digital"
        assert len(launched["pass_record"]["doppler_plan"]) > 10
        assert launched["duration_seconds"] >= 30
        assert local_catalog.get_satellite_pass(pass_record["pass_id"])["state"] == "launched"
        alerts = local_catalog.list_alert_events(event_type="satellite_pass")
        assert len(alerts) == 1
        assert alerts[0]["details"]["event_kind"] == "prepass"
        deliveries = local_catalog.list_webhook_deliveries(event_id=alerts[0]["event_id"])
        assert deliveries[0]["payload"]["schema"] == "rf-mcp.satellite-pass.v1"

        local_catalog.upsert_job(
            "sstv-watch-pass", "sstv_watch", "completed",
            completed_at=(now + timedelta(minutes=5)).isoformat(),
        )
        manager.tick(now + timedelta(minutes=5))
        alerts = local_catalog.list_alert_events(event_type="satellite_pass")
        assert len(alerts) == 2
        assert {item["details"]["event_kind"] for item in alerts} == {"prepass", "outcome"}
        assert len(local_catalog.list_webhook_deliveries()) == 2


def test_managed_tle_refresh_validates_and_retains_last_known_good_on_failure():
    assert tle_checksum_valid(ISS_TLE_1)
    assert tle_checksum_valid(ISS_TLE_2)
    parsed = parse_tle_response(
        "ISS (ZARYA)\n" + ISS_TLE_1 + "\n" + ISS_TLE_2 + "\n",
        norad_id=25544,
    )
    assert parsed == (ISS_TLE_1, ISS_TLE_2, "ISS (ZARYA)")
    with tempfile.TemporaryDirectory() as temporary:
        local_catalog = Catalog(Path(temporary), index_existing=False)
        values = _satellite_watch_values()
        values.update({
            "tle_source": "celestrak", "auto_refresh": True,
            "refresh_interval_seconds": 86400,
        })
        watch = local_catalog.save_satellite_watch(**values)
        original = watch["tle_line1"]

        def successful(_norad_id):
            return {"tle_line1": ISS_TLE_1, "tle_line2": ISS_TLE_2,
                    "satellite_name": "ISS (ZARYA)"}

        refreshed = refresh_satellite_tle(
            local_catalog, watch,
            now=datetime(2014, 1, 23, tzinfo=timezone.utc), fetcher=successful,
        )
        assert refreshed["last_tle_refresh_status"] == "succeeded"
        assert refreshed["last_tle_refresh_error"] is None
        assert refreshed["next_tle_refresh_at"].startswith("2014-01-24")

        def failed(_norad_id):
            raise RuntimeError("upstream unavailable")

        try:
            refresh_satellite_tle(
                local_catalog, refreshed,
                now=datetime(2014, 1, 24, tzinfo=timezone.utc), fetcher=failed,
            )
        except RuntimeError as exc:
            assert "last-known-good" in str(exc)
        else:
            raise AssertionError("Managed TLE failure did not propagate")
        retained = local_catalog.get_satellite_watch(watch["watch_id"])
        assert retained["tle_line1"] == original
        assert retained["last_tle_refresh_status"] == "failed"
        assert "upstream unavailable" in retained["last_tle_refresh_error"]
        assert retained["next_tle_refresh_at"].startswith("2014-01-24T01:00")


def test_celestrak_fetch_is_catalog_scoped_and_size_bounded(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, count):
            captured["read_count"] = count
            return ("ISS (ZARYA)\n" + ISS_TLE_1 + "\n" + ISS_TLE_2 + "\n").encode("ascii")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["agent"] = request.headers.get("User-agent")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(satellite_module, "urlopen", fake_urlopen)
    result = fetch_celestrak_tle(25544)
    assert result["tle_line1"] == ISS_TLE_1
    assert captured["url"].endswith("CATNR=25544&FORMAT=TLE")
    assert captured["agent"] == "rf-mcp-tle/0.50"
    assert captured["read_count"] == 8193
    try:
        fetch_celestrak_tle(100000)
    except ValueError as exc:
        assert "99999" in str(exc)
    else:
        raise AssertionError("Accepted a catalog number that TLE cannot represent")
