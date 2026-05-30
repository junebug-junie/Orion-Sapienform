from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.tts import resolve_synthesis_plan  # noqa: E402


def _settings(**overrides):
    class S:
        tts_backend = "coqui"
        tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts_default_language = "en"
        tts_default_speaker = None
        tts_default_speaker_wav = None
        tts_split_sentences = True
        tts_voice_profile_dir = "/models/voices"

    for k, v in overrides.items():
        setattr(S, k, v)
    return S()


def test_options_speaker_wav_wins(tmp_path) -> None:
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(tts_voice_profile_dir=str(tmp_path)),
        voice_id="ignored",
        language=None,
        options={"speaker_wav": str(wav), "language": "fr"},
    )
    assert plan.kwargs["speaker_wav"] == str(wav.resolve())
    assert plan.kwargs["language"] == "fr"
    assert plan.metadata["speaker_wav_used"] is True


def test_voice_id_resolves_under_profile_dir(tmp_path) -> None:
    wav = tmp_path / "orion_reference.wav"
    wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(tts_voice_profile_dir=str(tmp_path)),
        voice_id="orion_reference.wav",
        language="en",
        options=None,
    )
    assert plan.kwargs["speaker_wav"] == str(wav.resolve())
    assert plan.metadata["speaker_wav_used"] is True


def test_voice_id_passed_as_coqui_speaker_when_not_a_file() -> None:
    plan = resolve_synthesis_plan(
        _settings(tts_default_speaker="Claribel Dervla"),
        voice_id="Ana Florence",
        language="en",
        options=None,
    )
    assert plan.kwargs.get("speaker") == "Ana Florence"
    assert "speaker_wav" not in plan.kwargs
    assert plan.metadata["speaker_wav_used"] is False


def test_default_speaker_wav_from_settings(tmp_path) -> None:
    wav = tmp_path / "orion_reference.wav"
    wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_reference.wav",
        ),
        voice_id=None,
        language=None,
        options=None,
    )
    assert plan.kwargs["speaker_wav"] == str(wav.resolve())


def test_missing_speaker_wav_raises_loud(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="speaker_wav"):
        resolve_synthesis_plan(
            _settings(tts_voice_profile_dir=str(tmp_path)),
            voice_id="missing.wav",
            language="en",
            options=None,
        )
