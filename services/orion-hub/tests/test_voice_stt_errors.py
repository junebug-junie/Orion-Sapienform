from __future__ import annotations

from scripts.voice_stt_errors import (
    build_audio_debug,
    empty_transcript_error_message,
    sanitize_client_audio_meta,
)


def test_sanitize_client_audio_meta_allowlist() -> None:
    raw = {
        "peak": 0.001,
        "chunk_count": 12,
        "client_gate": "low_peak_warn",
        "evil": "drop me",
    }
    out = sanitize_client_audio_meta(raw)
    assert out == {
        "peak": 0.001,
        "chunk_count": 12,
        "client_gate": "low_peak_warn",
    }


def test_empty_transcript_browser_no_chunks() -> None:
    msg = empty_transcript_error_message(
        client_audio_meta={"chunk_count": 0, "client_gate": "empty"},
        stt_meta={"peak": 0, "silence_gate": "rejected", "peak_threshold": 50},
        audio_byte_len=1000,
    )
    assert msg == "Browser did not capture microphone frames."


def test_empty_transcript_stt_rejected() -> None:
    msg = empty_transcript_error_message(
        client_audio_meta={"chunk_count": 40, "client_gate": "low_peak_warn"},
        stt_meta={"peak": 8, "silence_gate": "rejected", "peak_threshold": 50},
        audio_byte_len=5000,
    )
    assert "STT rejected near-silent audio" in msg
    assert "peak=8" in msg


def test_empty_transcript_low_peak_warn_when_stt_passed() -> None:
    msg = empty_transcript_error_message(
        client_audio_meta={"chunk_count": 40, "client_gate": "low_peak_warn"},
        stt_meta={"peak": 120, "silence_gate": "passed", "peak_threshold": 50},
        audio_byte_len=5000,
    )
    assert "Low microphone level captured" in msg


def test_empty_transcript_no_speech_default() -> None:
    msg = empty_transcript_error_message(
        client_audio_meta={"chunk_count": 40, "client_gate": "passed"},
        stt_meta={"peak": 120, "silence_gate": "passed", "peak_threshold": 50},
        audio_byte_len=5000,
    )
    assert msg == "No speech detected in recording."


def test_build_audio_debug() -> None:
    dbg = build_audio_debug(
        stt_meta={"peak": 10},
        client_audio_meta={"chunk_count": 3},
    )
    assert dbg == {"stt": {"peak": 10}, "client": {"chunk_count": 3}}
