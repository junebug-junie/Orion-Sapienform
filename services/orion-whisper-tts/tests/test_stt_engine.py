from __future__ import annotations

import base64
import shutil
import struct
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub heavy deps before stt.py is imported.
sys.modules.setdefault("whisper", MagicMock())
_torch = MagicMock()
_torch.cuda.is_available = MagicMock(return_value=False)
sys.modules.setdefault("torch", _torch)

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _write_pcm16_wav(path: Path, samples: list[int], sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(frames)


def _load_stt_module():
    import importlib.util
    import sys

    path = SERVICE_ROOT / "app" / "stt.py"
    spec = importlib.util.spec_from_file_location("whisper_stt", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisper_stt_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mock_canonicalize(_self, src_path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    shutil.copy(src_path, out.name)
    return out.name


@pytest.fixture
def stt_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _load_stt_module()
    monkeypatch.setattr(mod, "whisper", MagicMock())
    monkeypatch.setattr(mod.torch, "cuda", MagicMock(is_available=lambda: False))
    mod.whisper.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setenv("STT_NEAR_SILENT_PEAK_INT16", "50")
    monkeypatch.setattr(mod, "_peak_threshold", lambda: 50)
    engine = mod.STTEngine.__new__(mod.STTEngine)
    engine.model = MagicMock()
    engine.model.transcribe = MagicMock(return_value={"text": "hello world"})
    monkeypatch.setattr(mod.STTEngine, "_canonicalize_wav", _mock_canonicalize)
    return engine, mod


def test_measure_wav_levels_silent_vs_tone(tmp_path: Path, stt_engine) -> None:
    engine, _mod = stt_engine
    silent_path = tmp_path / "silent.wav"
    tone_path = tmp_path / "tone.wav"
    _write_pcm16_wav(silent_path, [0] * 1600)
    _write_pcm16_wav(tone_path, [0] * 800 + [12000] * 800)

    silent_rms, silent_peak = engine._measure_wav_levels(str(silent_path))
    tone_rms, tone_peak = engine._measure_wav_levels(str(tone_path))

    assert silent_peak == 0
    assert silent_rms == 0.0
    assert tone_peak >= 10000
    assert tone_rms > 100.0


def test_transcribe_near_silent_skips_whisper(tmp_path: Path, stt_engine) -> None:
    engine, mod = stt_engine
    wav_path = tmp_path / "quiet.wav"
    _write_pcm16_wav(wav_path, [0] * 3200)
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()

    text, meta = engine.transcribe(audio_b64, language="en", audio_format="wav")

    assert text == ""
    assert meta["peak"] < meta["peak_threshold"]
    assert meta["silence_gate"] == "rejected"
    assert meta["duration_sec"] >= 0
    assert meta["input_bytes"] > 0
    assert meta["wav_bytes"] > 0
    engine.model.transcribe.assert_not_called()


def test_transcribe_low_amplitude_passes_with_lower_threshold(
    tmp_path: Path, stt_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, mod = stt_engine
    monkeypatch.setattr(mod, "_peak_threshold", lambda: 40)
    wav_path = tmp_path / "low.wav"
    _write_pcm16_wav(wav_path, [60] * 3200)
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()

    text, meta = engine.transcribe(audio_b64, language="en", audio_format="wav")

    assert text == "hello world"
    assert meta["peak"] == 60
    assert meta["peak_threshold"] == 40
    assert meta["silence_gate"] == "passed"
    engine.model.transcribe.assert_called_once()


def test_transcribe_tone_calls_whisper(tmp_path: Path, stt_engine) -> None:
    engine, _mod = stt_engine
    wav_path = tmp_path / "tone.wav"
    _write_pcm16_wav(wav_path, [0] * 400 + [14000] * 1200)
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()

    text, meta = engine.transcribe(audio_b64, language="en", audio_format="wav")

    assert text == "hello world"
    assert meta["peak"] >= meta["peak_threshold"]
    assert meta["silence_gate"] == "passed"
    engine.model.transcribe.assert_called_once()


def test_transcribe_empty_payload(stt_engine) -> None:
    engine, _mod = stt_engine
    text, meta = engine.transcribe("", language="en", audio_format="wav")
    assert text == ""
    assert meta["peak"] == 0
    assert meta["silence_gate"] == "rejected"
    assert meta["peak_threshold"] == 50
    engine.model.transcribe.assert_not_called()


def test_stt_worker_publishes_typed_envelope() -> None:
    worker_src = (SERVICE_ROOT / "app" / "stt_worker.py").read_text(encoding="utf-8")
    assert 'kind="stt.transcribe.result"' in worker_src
    assert "derive_child" in worker_src
    assert 'kind="system.error"' in worker_src
    assert 'request.format or "wav"' in worker_src
