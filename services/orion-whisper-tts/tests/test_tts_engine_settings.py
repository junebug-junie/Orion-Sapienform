from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load_settings_module():
    path = SERVICE_ROOT / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("whisper_tts_settings", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_whisper_tts_timeout_settings_defaults() -> None:
    mod = _load_settings_module()
    s = mod.Settings()
    assert s.tts_model_name == "tts_models/en/ljspeech/vits"
    assert s.whisper_tts_stt_timeout_sec == 90.0
    assert s.whisper_tts_synth_timeout_sec == 120.0


def test_tts_engine_reads_settings_not_hardcoded_gpu() -> None:
    tts_py = (SERVICE_ROOT / "app" / "tts.py").read_text(encoding="utf-8")
    settings_py = (SERVICE_ROOT / "app" / "settings.py").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "settings.tts_use_gpu" in tts_py
    assert "gpu=True" not in tts_py
    assert '"tts_models/en/ljspeech/vits"' in settings_py
    assert "tts_models/en/ljspeech/vits" in compose
    assert "WHISPER_TTS_STT_TIMEOUT_SEC" in compose
    assert "WHISPER_TTS_SYNTH_TIMEOUT_SEC" in compose


def test_stt_and_tts_workers_use_wait_for() -> None:
    stt_worker = (SERVICE_ROOT / "app" / "stt_worker.py").read_text(encoding="utf-8")
    tts_worker = (SERVICE_ROOT / "app" / "tts_worker.py").read_text(encoding="utf-8")
    assert "asyncio.wait_for" in stt_worker
    assert "whisper_tts_stt_timeout_sec" in stt_worker
    assert "asyncio.wait_for" in tts_worker
    assert "whisper_tts_synth_timeout_sec" in tts_worker


def test_stt_engine_exports_near_silent_constant() -> None:
    stt_py = (SERVICE_ROOT / "app" / "stt.py").read_text(encoding="utf-8")
    assert "STT_NEAR_SILENT_PEAK_INT16" in stt_py
    assert "_measure_wav_levels" in stt_py
