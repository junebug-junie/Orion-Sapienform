"""2026-08-25: Juniper's facial+vocal affect read in the situation brief.

Closes a real gap: orion-affectgpt-worker's real inference (circe GPU1,
AffectGPT + Whisper) was already published on `orion:affectgpt:assessment`,
but nothing downstream of `orion-juniper-affective-state` ever consumed it
except a manual debug CLI -- Orion's own chat turns never found out. This
file exercises the reader side of the seam that fixes that
(`orion/situational/juniper_affect_state.py` write side has its own tests
in `orion/situational/tests/test_juniper_affect_state.py`).

The two properties that matter more than the happy path, mirroring
`test_situation_perception_context.py`'s own framing: a stale read must
never reach a prompt as a current mood, and the privacy contract in
`AffectContextV1`'s docstring (excerpt only, never the verbatim transcript)
is enforced by what this module is given, not by good intentions alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.situational import context as situation_mod
from orion.situational.context import (
    SituationSettings,
    _build_affect_context,
    _build_prompt_fragment,
    settings_from_runtime,
)
from orion.situational.juniper_affect_state import JuniperAffectState
from orion.schemas.situation import AffectContextV1, SituationBriefV1

NOW = datetime.now(timezone.utc)


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _diag():
    from orion.schemas.situation import SituationDiagnosticsV1

    return SituationDiagnosticsV1()


def _state(**overrides) -> JuniperAffectState:
    base = dict(summary=None, observed_at=None, trigger=None, subtitle_source=None, ok=True)
    base.update(overrides)
    return JuniperAffectState(**base)


# --- provider states ---------------------------------------------------


def test_enabled_by_default() -> None:
    # Unlike perception, capture is already an explicit Juniper action --
    # see settings_from_runtime's own comment.
    assert settings_from_runtime(SimpleNamespace()).affect_enabled is True


@pytest.mark.asyncio
async def test_disabled_yields_unavailable_not_an_error(monkeypatch) -> None:
    diag = _diag()
    ctx = await _build_affect_context(_cfg(affect_enabled=False), diag)
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["affect"] == "disabled"


@pytest.mark.asyncio
async def test_no_capture_is_unavailable(monkeypatch) -> None:
    async def _fake_read():
        return _state()

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    ctx = await _build_affect_context(_cfg(affect_enabled=True), _diag())
    assert ctx.available is False
    assert ctx.source == "unavailable"
    assert ctx.summary is None


@pytest.mark.asyncio
async def test_fresh_capture_is_available(monkeypatch) -> None:
    async def _fake_read():
        return _state(
            summary="Juniper appears focused, leaning toward the screen.",
            observed_at=NOW,
            trigger="manual",
            subtitle_source="transcribed",
        )

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    ctx = await _build_affect_context(_cfg(affect_enabled=True), _diag())
    assert ctx.available is True
    assert ctx.source == "live"
    assert ctx.summary == "Juniper appears focused, leaning toward the screen."
    assert ctx.trigger == "manual"
    assert ctx.subtitle_source == "transcribed"
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds < 60


@pytest.mark.asyncio
async def test_stale_capture_is_withheld_entirely(monkeypatch) -> None:
    """Core honesty gate, same as perception's: an old mood read presented
    as current is a confabulation with a real referent -- so the summary is
    not merely flagged stale, it is not carried at all."""
    old = NOW - timedelta(minutes=20)

    async def _fake_read():
        return _state(summary="Juniper seemed tense.", observed_at=old, trigger="ambient")

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    ctx = await _build_affect_context(_cfg(affect_enabled=True, affect_max_age_seconds=300), _diag())
    assert ctx.available is False
    assert ctx.source == "stale"
    assert ctx.summary is None, "a stale summary must not ride along in the payload"
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds > 300


@pytest.mark.asyncio
async def test_age_boundary_is_inclusive_of_the_threshold(monkeypatch) -> None:
    at_limit = NOW - timedelta(seconds=300)

    async def _fake_read():
        return _state(summary="calm", observed_at=at_limit, trigger="manual")

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    ctx = await _build_affect_context(_cfg(affect_enabled=True, affect_max_age_seconds=300), _diag())
    assert ctx.available is True, "exactly at the threshold is still fresh"


@pytest.mark.asyncio
async def test_reader_exception_fails_open(monkeypatch) -> None:
    async def _boom():
        raise RuntimeError("redis gone")

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _boom)
    diag = _diag()
    ctx = await _build_affect_context(_cfg(affect_enabled=True), diag)
    assert ctx.available is False
    assert ctx.source == "error"
    assert "redis gone" in diag.provider_errors["affect"]


@pytest.mark.asyncio
async def test_read_not_ok_is_an_error_not_unavailable(monkeypatch) -> None:
    """state.ok=False means the read itself could not be trusted (unbound
    bus, Redis error) -- distinct from a genuinely-confirmed-empty read."""

    async def _fake_read():
        return _state(ok=False)

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    diag = _diag()
    ctx = await _build_affect_context(_cfg(affect_enabled=True), diag)
    assert ctx.available is False
    assert ctx.source == "error"
    assert diag.provider_status["affect"] == "error"


@pytest.mark.asyncio
async def test_empty_summary_is_not_a_capture(monkeypatch) -> None:
    async def _fake_read():
        return _state(summary="", observed_at=NOW, trigger="manual")

    monkeypatch.setattr(situation_mod, "read_latest_juniper_affect", _fake_read)
    ctx = await _build_affect_context(_cfg(affect_enabled=True), _diag())
    assert ctx.available is False


# --- prompt rendering ----------------------------------------------------


def _brief(affect: AffectContextV1) -> SituationBriefV1:
    """Same technique test_situation_perception_context.py's own _brief()
    uses: build via the production helpers, swap in the sub-context under
    test, so this breaks if the real builders' shape drifts."""
    cfg = _cfg(affect_enabled=False)
    diag = _diag()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    import asyncio

    return SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        affect=affect,
    )


def test_available_capture_renders_with_its_age() -> None:
    brief = _brief(
        AffectContextV1(
            available=True,
            source="live",
            summary="Juniper appears relaxed and smiling.",
            observation_age_seconds=120,
            trigger="manual",
            subtitle_source="transcribed",
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Juniper's affect (captured 2 min ago): Juniper appears relaxed and smiling." in text


def test_no_speech_detected_is_noted_inline() -> None:
    brief = _brief(
        AffectContextV1(
            available=True,
            source="live",
            summary="Juniper appears neutral.",
            observation_age_seconds=30,
            trigger="ambient",
            subtitle_source="none",
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "(no speech detected)" in text


def test_unavailable_renders_as_no_recent_capture_not_a_guess() -> None:
    text = _build_prompt_fragment(_brief(AffectContextV1()), 4000).compact_text
    assert "no recent capture" in text.lower()


def test_stale_summary_never_reaches_the_prompt() -> None:
    stale = AffectContextV1(available=False, source="stale", observation_age_seconds=1800)
    text = _build_prompt_fragment(_brief(stale), 4000).compact_text
    assert "no recent capture" in text.lower()


def test_caution_line_present_against_overinterpreting() -> None:
    text = _build_prompt_fragment(_brief(AffectContextV1()), 4000).compact_text
    assert "not a diagnosis" in text


# --- privacy contract ------------------------------------------------------


def test_schema_exposes_no_transcript_or_raw_response_field() -> None:
    """The exposed-field list IS the privacy contract, same role
    PerceptionContextV1's own test plays. `extra='forbid'` means a future
    caller cannot smuggle the verbatim transcript in without changing this
    schema, which is the point."""
    fields = set(AffectContextV1.model_fields)
    for banned in ("transcript", "raw_response", "video_path", "audio_path", "input_ref"):
        assert banned not in fields, f"{banned} must not be exposed to the prompt"


def test_affect_is_session_only_by_default() -> None:
    assert AffectContextV1().privacy_mode == "session_only"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(Exception):
        AffectContextV1(transcript="verbatim spoken words")
