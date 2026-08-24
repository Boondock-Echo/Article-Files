from __future__ import annotations

import csv
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .plotting import lazy_pyplot as plt
import numpy as np
from .lazy_imports import wavfile
from .lazy_imports import resample_poly

from .config import AUDIO_DIR, DATA_DIR, PLOT_DIR, RESULT_DIR, ensure_data_dirs


_LOCK = threading.RLock()


def _path() -> Path:
    return DATA_DIR / "recording-sessions.json"


def _load() -> list[dict]:
    ensure_data_dirs()
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read recording sessions: {exc}") from exc


def _write(items: list[dict]) -> None:
    path = _path(); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _text(value: str, field: str, maximum: int) -> str:
    value = str(value).strip()
    if not value or len(value) > maximum or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
        raise ValueError(f"{field} must contain 1 through {maximum} printable characters")
    return value


def _item(artifact: dict, label: str | None = None) -> dict:
    return {"artifact_id": artifact["artifact_id"], "job_id": artifact.get("job_id"),
            "label": str(label or artifact["filename"])[:120], "kind": artifact["kind"],
            "filename": artifact["filename"], "mime_type": artifact["mime_type"],
            "added_at": datetime.now(timezone.utc).isoformat()}


def create_session(*, name: str, description: str = "", tags: list[str] | None = None,
                   artifacts: list[dict] | None = None) -> dict:
    name = _text(name, "name", 80)
    description = str(description).strip()
    if len(description) > 2000:
        raise ValueError("description must contain no more than 2000 characters")
    tags = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
    if len(tags) > 20 or any(len(tag) > 40 for tag in tags):
        raise ValueError("tags must contain at most 20 values of no more than 40 characters")
    now = datetime.now(timezone.utc).isoformat()
    record = {"session_id": f"session-{uuid4().hex[:16]}", "name": name,
              "description": description, "tags": sorted(set(tags)),
              "items": [_item(value) for value in (artifacts or [])],
              "annotations": [], "bookmarks": [], "created_at": now, "updated_at": now}
    with _LOCK:
        items = _load()
        if any(item["name"].casefold() == name.casefold() for item in items):
            raise ValueError("A recording session with this name already exists")
        items.append(record); _write(items)
    return record


def list_sessions() -> list[dict]:
    with _LOCK:
        return sorted(_load(), key=lambda item: item["updated_at"], reverse=True)


def get_session(identifier: str) -> dict:
    text = str(identifier).strip().casefold()
    match = next((item for item in list_sessions()
                  if item["session_id"].casefold() == text or item["name"].casefold() == text), None)
    if not match:
        raise ValueError(f"Recording session not found: {identifier}")
    return match


def _replace(record: dict) -> dict:
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    items = [record if item["session_id"] == record["session_id"] else item for item in _load()]
    _write(items); return record


def add_artifacts(identifier: str, artifacts: list[dict]) -> dict:
    with _LOCK:
        record = get_session(identifier)
        known = {item["artifact_id"] for item in record["items"]}
        record["items"].extend(_item(value) for value in artifacts
                               if value["artifact_id"] not in known)
        return _replace(record)


def add_annotation(identifier: str, *, text: str, artifact_id: str | None = None,
                   start_seconds: float | None = None, end_seconds: float | None = None,
                   tags: list[str] | None = None) -> dict:
    text = _text(text, "text", 2000)
    if (start_seconds is None) != (end_seconds is None):
        raise ValueError("Provide both annotation start_seconds and end_seconds or neither")
    if start_seconds is not None and not 0 <= float(start_seconds) < float(end_seconds):
        raise ValueError("Annotation times must satisfy 0 <= start_seconds < end_seconds")
    with _LOCK:
        record = get_session(identifier)
        if artifact_id and artifact_id not in {item["artifact_id"] for item in record["items"]}:
            raise ValueError("artifact_id is not attached to this session")
        annotation = {"annotation_id": f"note-{uuid4().hex[:16]}", "text": text,
                      "artifact_id": artifact_id, "start_seconds": start_seconds,
                      "end_seconds": end_seconds,
                      "tags": sorted({str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}),
                      "created_at": datetime.now(timezone.utc).isoformat()}
        record["annotations"].append(annotation); _replace(record)
        return annotation


def wav_info(path: Path) -> dict:
    sample_rate, data = wavfile.read(path, mmap=True)
    return {"sample_rate_hz": int(sample_rate), "frame_count": int(data.shape[0]),
            "channel_count": 1 if data.ndim == 1 else int(data.shape[1]),
            "duration_seconds": float(data.shape[0] / sample_rate), "dtype": str(data.dtype)}


def add_bookmark(identifier: str, *, artifact: dict, position_seconds: float,
                 label: str, notes: str = "") -> dict:
    if artifact["mime_type"] not in {"audio/wav", "audio/x-wav"} and not artifact["filename"].lower().endswith(".wav"):
        raise ValueError("Bookmarks currently support WAV audio artifacts only")
    info = wav_info(Path(artifact["path"]))
    position = float(position_seconds)
    if not 0 <= position <= info["duration_seconds"]:
        raise ValueError("position_seconds is outside the WAV duration")
    with _LOCK:
        record = get_session(identifier)
        if artifact["artifact_id"] not in {item["artifact_id"] for item in record["items"]}:
            record["items"].append(_item(artifact))
        bookmark = {"bookmark_id": f"bookmark-{uuid4().hex[:16]}",
                    "artifact_id": artifact["artifact_id"], "position_seconds": position,
                    "label": _text(label, "label", 120), "notes": str(notes).strip()[:1000],
                    "created_at": datetime.now(timezone.utc).isoformat()}
        record["bookmarks"].append(bookmark); _replace(record)
        return bookmark


def delete_session(identifier: str) -> dict:
    with _LOCK:
        target = get_session(identifier)
        _write([item for item in _load() if item["session_id"] != target["session_id"]])
        return target


def search_sessions(query: str) -> list[dict]:
    terms = [term.casefold() for term in str(query).split() if term]
    if not terms:
        raise ValueError("query must contain at least one search term")
    output = []
    for session in list_sessions():
        haystack = " ".join([
            session["name"], session["description"], *session["tags"],
            *(item["label"] for item in session["items"]),
            *(item["filename"] for item in session["items"]),
            *(item["text"] for item in session["annotations"]),
            *(" ".join(item["tags"]) for item in session["annotations"]),
            *(item["label"] + " " + item["notes"] for item in session["bookmarks"]),
        ]).casefold()
        if all(term in haystack for term in terms):
            output.append(session)
    return output


def extract_wav_clip(source: Path, *, start_seconds: float,
                     duration_seconds: float, label: str | None = None) -> tuple[Path, dict]:
    sample_rate, data = wavfile.read(source)
    start, duration = float(start_seconds), float(duration_seconds)
    total = data.shape[0] / sample_rate
    if not 0 <= start < total:
        raise ValueError("start_seconds is outside the WAV duration")
    if not 0.05 <= duration <= 600:
        raise ValueError("duration_seconds must be from 0.05 through 600")
    stop = min(total, start + duration)
    first, last = round(start * sample_rate), round(stop * sample_rate)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(label or "clip")).strip("-")[:48] or "clip"
    path = AUDIO_DIR / f"{source.stem}-{safe_label}-{uuid4().hex[:8]}.wav"
    ensure_data_dirs(); wavfile.write(path, sample_rate, data[first:last])
    return path, {"source_duration_seconds": total, "start_seconds": start,
                  "requested_duration_seconds": duration,
                  "actual_duration_seconds": (last - first) / sample_rate,
                  "sample_rate_hz": int(sample_rate), "frame_count": last - first}


def _audio(path: Path, maximum_seconds: float = 120) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    data = np.asarray(data[:round(rate * maximum_seconds)])
    original_dtype = data.dtype
    if data.ndim > 1:
        data = np.mean(data.astype(float), axis=1)
    if np.issubdtype(original_dtype, np.integer):
        scale = max(abs(np.iinfo(original_dtype).min), np.iinfo(original_dtype).max)
        data = data.astype(float) / scale
    else:
        data = data.astype(float)
    return int(rate), data


def compare_wav(first: Path, second: Path) -> tuple[dict, Path]:
    ensure_data_dirs()
    rate_a, a = _audio(first); rate_b, b = _audio(second)
    common = min(rate_a, rate_b)
    if rate_a != common: a = resample_poly(a, common, rate_a)
    if rate_b != common: b = resample_poly(b, common, rate_b)
    count = min(len(a), len(b)); a, b = a[:count], b[:count]
    if count < 100:
        raise ValueError("Audio clips are too short to compare")
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    db = lambda value: float(20 * np.log10(max(value, 1e-12)))
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    def centroid(x):
        power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freq = np.fft.rfftfreq(len(x), 1 / common)
        return float(np.sum(freq * power) / max(np.sum(power), 1e-30))
    result = {"compared_duration_seconds": count / common, "sample_rate_hz": common,
              "first_rms_dbfs": db(rms(a)), "second_rms_dbfs": db(rms(b)),
              "rms_difference_db": db(rms(b)) - db(rms(a)),
              "waveform_correlation": correlation,
              "first_spectral_centroid_hz": centroid(a),
              "second_spectral_centroid_hz": centroid(b),
              "difference_rms_dbfs": db(rms(a - b)),
              "warning": "Correlation is alignment-sensitive and does not authenticate a source."}
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    display = min(count, common * 5); time = np.arange(display) / common
    axes[0].plot(time, a[:display], label="First", alpha=.8)
    axes[0].plot(time, b[:display], label="Second", alpha=.65)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude"); axes[0].legend(); axes[0].grid(alpha=.25)
    for values, label in ((a, "First"), (b, "Second")):
        spectrum = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(values * np.hanning(count))), 1e-12))
        axes[1].plot(np.fft.rfftfreq(count, 1/common), spectrum, label=label, alpha=.8)
    axes[1].set_xlim(0, min(common/2, 20_000)); axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative magnitude (dB)"); axes[1].legend(); axes[1].grid(alpha=.25)
    path = PLOT_DIR / f"audio-compare-{uuid4().hex[:12]}.png"; figure.savefig(path, dpi=150); plt.close(figure)
    return result, path


def export_session(session: dict) -> tuple[Path, Path]:
    ensure_data_dirs(); stem = f"{session['session_id']}-{uuid4().hex[:8]}"
    json_path, csv_path = RESULT_DIR / f"{stem}.json", RESULT_DIR / f"{stem}-annotations.csv"
    json_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    fields = ["annotation_id", "artifact_id", "start_seconds", "end_seconds", "text", "tags", "created_at"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in session["annotations"]:
            writer.writerow({**item, "tags": ";".join(item["tags"])})
    return json_path, csv_path
