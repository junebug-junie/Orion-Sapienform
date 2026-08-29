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


def test_default_speaker_wav_beats_voice_id(tmp_path) -> None:
    """The live production config: TTS_DEFAULT_SPEAKER_WAV set AND a per-request
    voice_id supplied. The env reference wins -- resolve order is
    options.speaker_wav > tts_default_speaker_wav > voice_id >
    tts_default_speaker (app/tts.py:85-110), so the env key outranks voice_id
    and not merely tts_default_speaker.

    Every other case in this file sets exactly one of the two, so nothing
    covered the combination that actually ships. Added 2026-08-29 after review
    found README payload examples documenting behavior this config makes
    impossible.
    """
    wav = tmp_path / "orion_reference_v2.wav"
    wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_reference_v2.wav",
            tts_default_speaker="Ana Florence",
        ),
        voice_id="Claribel Dervla",
        language="en",
        options=None,
    )
    assert plan.kwargs["speaker_wav"] == str(wav.resolve())
    # The built-in speaker is not merely deprioritised -- XTTS is never told
    # about it at all, so a caller passing voice_id gets silence on that field.
    assert "speaker" not in plan.kwargs
    assert plan.metadata["speaker"] is None
    assert plan.metadata["speaker_wav_used"] is True
    # voice_id is still echoed for traceability, which is exactly why metadata
    # alone cannot tell you whether it was honoured.
    assert plan.metadata["voice_id"] == "Claribel Dervla"


def test_request_speaker_wav_beats_default_speaker_wav(tmp_path) -> None:
    """options.speaker_wav is the only thing that outranks the env default --
    this is the override the rollback/control procedure in README.md relies on.
    """
    default_wav = tmp_path / "orion_reference_v2.wav"
    default_wav.write_bytes(b"RIFF")
    override_wav = tmp_path / "orion_reference.wav"
    override_wav.write_bytes(b"RIFF")
    plan = resolve_synthesis_plan(
        _settings(
            tts_voice_profile_dir=str(tmp_path),
            tts_default_speaker_wav="orion_reference_v2.wav",
        ),
        voice_id=None,
        language="en",
        options={"speaker_wav": "orion_reference.wav"},
    )
    assert plan.kwargs["speaker_wav"] == str(override_wav.resolve())
    assert plan.metadata["speaker_wav_basename"] == "orion_reference.wav"


def test_missing_speaker_wav_raises_loud(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="speaker_wav"):
        resolve_synthesis_plan(
            _settings(tts_voice_profile_dir=str(tmp_path)),
            voice_id="missing.wav",
            language="en",
            options=None,
        )


def test_speaker_wav_rejects_path_outside_profile_dir(tmp_path) -> None:
    root = tmp_path / "voices"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="must be under"):
        resolve_synthesis_plan(
            _settings(tts_voice_profile_dir=str(root)),
            voice_id=None,
            language="en",
            options={"speaker_wav": str(outside)},
        )


def test_xtts_requires_speaker() -> None:
    with pytest.raises(ValueError, match="XTTS requires a speaker"):
        resolve_synthesis_plan(
            _settings(),
            voice_id=None,
            language="en",
            options=None,
        )
