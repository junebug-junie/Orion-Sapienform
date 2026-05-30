# services/orion-hub/scripts/voice_stt_errors.py
from __future__ import annotations

from typing import Any, Dict, Optional

ALLOWED_CLIENT_AUDIO_META_KEYS = frozenset(
    {
        "source_sample_rate",
        "target_sample_rate",
        "duration_sec",
        "peak",
        "rms",
        "pcm_samples",
        "chunk_count",
        "client_peak_threshold",
        "client_gate",
    }
)


def sanitize_client_audio_meta(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return {k: raw[k] for k in ALLOWED_CLIENT_AUDIO_META_KEYS if k in raw}


def build_audio_debug(
    *,
    stt_meta: Optional[Dict[str, Any]] = None,
    client_audio_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    if stt_meta:
        out["stt"] = stt_meta
    if client_audio_meta:
        out["client"] = client_audio_meta
    return out or None


def empty_transcript_error_message(
    *,
    client_audio_meta: Optional[Dict[str, Any]],
    stt_meta: Dict[str, Any],
    audio_byte_len: int,
) -> str:
    client_chunks = None
    client_gate = None
    if isinstance(client_audio_meta, dict):
        client_chunks = client_audio_meta.get("chunk_count")
        client_gate = client_audio_meta.get("client_gate")

    if (
        client_chunks == 0
        or client_gate == "empty"
        or (not isinstance(client_audio_meta, dict) and audio_byte_len == 0)
    ):
        return "Browser did not capture microphone frames."

    silence_gate = stt_meta.get("silence_gate")
    peak_val = stt_meta.get("peak")
    peak_threshold = stt_meta.get("peak_threshold")
    try:
        peak_num = float(peak_val) if peak_val is not None else None
    except (TypeError, ValueError):
        peak_num = None
    try:
        threshold_num = float(peak_threshold) if peak_threshold is not None else None
    except (TypeError, ValueError):
        threshold_num = None

    if silence_gate == "rejected" and peak_num is not None and threshold_num is not None:
        return (
            f"STT rejected near-silent audio: peak={int(peak_num)}, "
            f"threshold={int(threshold_num)}."
        )

    if client_gate == "low_peak_warn" and silence_gate != "rejected":
        return (
            "Low microphone level captured. Audio was recorded but too quiet for STT."
        )

    return "No speech detected in recording."
