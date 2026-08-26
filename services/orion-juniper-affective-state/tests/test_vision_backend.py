"""The vision backend, and the gates that decide whether Orion hears about it.

Several tests here are regression tests for one specific live failure: chat
turn `ddddfe40` (2026-08-26), where AffectGPT returned "it is not possible to
infer the character's emotional state from the subtitle content", the pipeline
recorded `ok=True`, and that sentence was mirrored verbatim into Juniper's chat
prompt. Each of the three independent things that had to be true for that to
happen gets its own test below.
"""
from __future__ import annotations

import pytest

from app.main import (
    _mirror_decision,
    _render_affect_summary,
    _vision_backend_selected,
    settings,
)
from app.vision_backend import _RESPONSE_SCHEMA, _SYSTEM_PROMPT, _build_prompt
from orion.schemas.affectgpt import AffectReadV1, JuniperMultimodalAffectV1

from datetime import datetime, timezone


def _event(**kw) -> JuniperMultimodalAffectV1:
    base = dict(
        observed_at=datetime.now(timezone.utc),
        ok=True,
        backend="vision",
        source="vision",
    )
    base.update(kw)
    return JuniperMultimodalAffectV1(**base)


def _affect(**kw) -> AffectReadV1:
    base = dict(
        valence=-0.2,
        arousal=0.3,
        primary_affect="tired, subdued",
        cues=["half-lidded eyes", "mouth closed in a flat line"],
        confidence=0.7,
        cannot_tell=["cause"],
    )
    base.update(kw)
    return AffectReadV1(**base)


# ── The prompt no longer has a subtitle branch to collapse into ───────────


def test_prompt_never_mentions_a_subtitle_when_there_is_no_transcript():
    """THE regression test for turn ddddfe40.

    AffectGPT bailed to "cannot infer from the subtitle content" because its
    prompt template had a text slot that was empty. The fix is structural: with
    no transcript, nothing in the prompt refers to speech at all, so there is
    no empty slot for the model to anchor on.
    """
    prompt = _build_prompt(None, 5)
    lowered = prompt.lower()
    for banned in ("subtitle", "caption", "transcript", "said", "speech", "audio"):
        assert banned not in lowered, f"empty-transcript prompt mentions {banned!r}"


def test_prompt_treats_whitespace_only_transcript_as_absent():
    # A stray " " must not count as real text -- the same whitespace hole that
    # was found and fixed in the retired backend's resolve_subtitle().
    assert _build_prompt("   \n ", 5) == _build_prompt(None, 5)


def test_prompt_includes_real_transcript_as_context_only():
    prompt = _build_prompt("I am so tired of this", 5)
    assert "I am so tired of this" in prompt
    # Included, but explicitly subordinate to what the model can see.
    assert "context" in prompt.lower()


def test_prompt_states_the_frame_count_it_was_actually_given():
    assert "3 webcam stills" in _build_prompt(None, 3)
    assert "5 webcam stills" in _build_prompt(None, 5)


# ── Identity inference is banned at the source ───────────────────────────


def test_system_prompt_forbids_gender_and_identity_inference():
    """AffectGPT called Juniper "the man" in 3 of 3 reads that committed to
    anything. This is the structural fix, not a hope that a better model
    happens to get it right."""
    lowered = _SYSTEM_PROMPT.lower()
    assert "do not infer or state the person's gender" in lowered
    assert "never use gendered words" in lowered


def test_system_prompt_forbids_claiming_audio_evidence():
    """The retired backend cited "the acoustic characteristics of the voice"
    from a track measured at -49.2 dB peak. This backend is handed no audio at
    all, so any such claim would be pure confabulation."""
    assert "You have NO audio" in _SYSTEM_PROMPT
    assert "Never describe voice, tone, speech, or sound." in _SYSTEM_PROMPT


def test_response_schema_requires_the_fields_the_gates_read():
    # confidence gates the mirror write; cues is what makes a read auditable.
    # If either stopped being required, the gate would silently start passing
    # reads it cannot evaluate.
    assert "confidence" in _RESPONSE_SCHEMA["required"]
    assert "cues" in _RESPONSE_SCHEMA["required"]


# ── Backend selection fails closed to vision ─────────────────────────────


@pytest.mark.parametrize(
    "value,expected_vision",
    [
        ("vision", True),
        ("affectgpt", False),
        ("AFFECTGPT", False),
        ("  affectgpt  ", False),
        # A typo must NOT select the backend that misgendered her.
        ("affectgtp", True),
        ("", True),
        (None, True),
    ],
)
def test_backend_selection_fails_closed_to_vision(monkeypatch, value, expected_vision):
    monkeypatch.setattr(settings, "AFFECT_BACKEND", value, raising=False)
    assert _vision_backend_selected() is expected_vision


def test_default_backend_is_vision():
    """Asserted on the real unpinned settings object, deliberately: every test
    in test_capture_and_assess.py pins AFFECT_BACKEND="affectgpt", so without
    this the default could flip back and that whole file would still pass."""
    from app.settings import Settings

    assert Settings().AFFECT_BACKEND == "vision"


# ── The mirror gate ──────────────────────────────────────────────────────


def test_failed_read_is_never_mirrored():
    allowed, reason = _mirror_decision(_event(ok=False, error="boom", affect=None))
    assert allowed is False
    assert reason == "not_ok"


def test_low_confidence_read_is_published_but_not_mirrored(monkeypatch):
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.35, raising=False)
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_DETECTION_RATE", 0.15, raising=False)
    allowed, reason = _mirror_decision(
        _event(affect=_affect(confidence=0.2), face_detection={"detection_rate": 1.0})
    )
    assert allowed is False
    assert reason.startswith("low_confidence:")


def test_low_detection_rate_read_is_not_mirrored(monkeypatch):
    """The 2026-08-26 capture that scored detection_rate=0.052 -- 170 of 231
    frames with no detectable face -- and still produced a confident
    "anger, frustration, or sadness" from the retired backend."""
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.35, raising=False)
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_DETECTION_RATE", 0.15, raising=False)
    allowed, reason = _mirror_decision(
        _event(affect=_affect(confidence=0.9), face_detection={"detection_rate": 0.052})
    )
    assert allowed is False
    assert reason.startswith("low_detection_rate:")


def test_good_read_is_mirrored(monkeypatch):
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.35, raising=False)
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_DETECTION_RATE", 0.15, raising=False)
    # The other 2026-08-26 capture: 231/231 frames detected.
    allowed, reason = _mirror_decision(
        _event(affect=_affect(confidence=0.7), face_detection={"detection_rate": 1.0})
    )
    assert allowed is True
    assert reason == "ok"


def test_thresholds_are_read_live_not_captured_at_import(monkeypatch):
    """These are env keys precisely so they can be retuned without a redeploy;
    a value captured at import time would make that a lie."""
    good = _event(affect=_affect(confidence=0.5), face_detection={"detection_rate": 1.0})
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.35, raising=False)
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_DETECTION_RATE", 0.15, raising=False)
    assert _mirror_decision(good)[0] is True
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.8, raising=False)
    assert _mirror_decision(good)[0] is False


def test_legacy_affectgpt_row_without_structured_read_still_mirrors():
    """The rollback path must keep working -- neither gate is evaluable when
    there is no structured read, so it falls back to the pre-cutover rule."""
    allowed, reason = _mirror_decision(
        _event(backend="affectgpt", source="affectgpt", affect=None, raw_response="sad")
    )
    assert allowed is True
    assert reason == "legacy_no_structured_read"


def test_legacy_row_with_empty_response_is_not_mirrored():
    allowed, _ = _mirror_decision(
        _event(backend="affectgpt", source="affectgpt", affect=None, raw_response="")
    )
    assert allowed is False


def test_missing_detection_rate_does_not_block_a_confident_read(monkeypatch):
    # Absent telemetry is not evidence of a bad capture. Blocking here would
    # make any future producer that omits the field silently mute the feature.
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_DETECTION_RATE", 0.15, raising=False)
    monkeypatch.setattr(settings, "AFFECT_MIRROR_MIN_CONFIDENCE", 0.35, raising=False)
    assert _mirror_decision(_event(affect=_affect(confidence=0.7), face_detection=None))[0] is True


# ── The rendered prompt line ─────────────────────────────────────────────


def test_summary_is_short_and_carries_confidence():
    """The whole failure was a 400-character essay landing in the prompt."""
    summary = _render_affect_summary(_affect())
    assert len(summary) < 200, summary
    assert "tired, subdued" in summary
    assert "confidence 0.70" in summary


def test_summary_surfaces_cues_and_uncertainty():
    summary = _render_affect_summary(_affect())
    assert "half-lidded eyes" in summary
    assert "can't tell" in summary


def test_summary_survives_an_empty_cue_list():
    summary = _render_affect_summary(_affect(cues=[], cannot_tell=[]))
    assert "tired, subdued" in summary
    assert "confidence" in summary


def test_summary_caps_cue_spam():
    """A model returning twenty cues must not rebuild the essay through the
    side door -- the cap is what keeps this a prompt line."""
    summary = _render_affect_summary(_affect(cues=[f"cue number {i}" for i in range(20)]))
    assert "cue number 2" not in summary
    assert len(summary) < 200
