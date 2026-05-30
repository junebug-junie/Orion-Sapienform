from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.schemas.tts import TTSResultPayload

from scripts import websocket_handler as wh


class _OkTTSClient:
    async def speak(self, _request):
        return TTSResultPayload(
            audio_b64="YWJj",
            content_type="audio/wav",
            duration_sec=1.5,
            metadata={"backend": "coqui"},
        )


class _FailingTTSClient:
    async def speak(self, _request):
        raise RuntimeError("synth blew up")


class _SlowTTSClient:
    async def speak(self, _request):
        await asyncio.sleep(60)
        return None


@pytest.mark.asyncio
async def test_run_tts_remote_enqueues_speaking_payload(monkeypatch) -> None:
    monkeypatch.setattr(wh.settings, "HUB_TTS_TIMEOUT_SEC", 5.0, raising=False)
    queue: asyncio.Queue = asyncio.Queue()
    await wh.run_tts_remote("hello", _OkTTSClient(), queue)
    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert msg["audio_response"] == "YWJj"
    assert msg["state"] == "speaking"
    assert msg.get("text") is None
    assert msg["tts_source_text"] == "hello"
    assert msg["tts_meta"]["content_type"] == "audio/wav"
    assert msg["tts_meta"]["duration_sec"] == 1.5


@pytest.mark.asyncio
async def test_run_tts_remote_enqueues_tts_error_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(wh.settings, "HUB_TTS_TIMEOUT_SEC", 5.0, raising=False)
    queue: asyncio.Queue = asyncio.Queue()
    await wh.run_tts_remote("hello", _FailingTTSClient(), queue)
    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert msg["tts_error"] == "synth blew up"
    assert msg["text"] == "hello"
    assert msg["state"] == "idle"


@pytest.mark.asyncio
async def test_run_tts_remote_enqueues_tts_error_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(wh.settings, "HUB_TTS_TIMEOUT_SEC", 0.05, raising=False)
    queue: asyncio.Queue = asyncio.Queue()
    await wh.run_tts_remote("hello", _SlowTTSClient(), queue)
    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert "timed out" in msg["tts_error"].lower()
    assert msg["state"] == "idle"


def test_start_recording_audio_payload_includes_disable_tts() -> None:
    hub_root = Path(__file__).resolve().parents[1]
    app_js = (hub_root / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false" in app_js
    assert "d.tts_error" in app_js
    assert "[voice] audio_debug" in app_js
    assert "client_audio_meta" in app_js
    assert "[tts] audio_response received" in app_js
    assert "[tts] playback started" in app_js
    assert "audioContext.resume" in app_js
    assert "d.audio_response && !d.llm_response" in app_js


def test_hub_voice_timeout_settings_defaults() -> None:
    from app.settings import Settings

    s = Settings()
    assert s.HUB_STT_TIMEOUT_SEC == 120.0
    assert s.HUB_TTS_TIMEOUT_SEC == 180.0
