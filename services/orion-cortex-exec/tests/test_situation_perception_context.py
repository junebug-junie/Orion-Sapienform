"""P4: the camera percept in the situation brief.

The two properties that matter more than the happy path: a stale percept must
never reach a prompt as a current observation, and the privacy contract in
PerceptionContextV1's docstring must be enforced by the schema rather than by
good intentions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app import situation as situation_mod
from app.situation import (
    SituationSettings,
    _build_perception_context,
    _build_prompt_fragment,
    settings_from_runtime,
)
from orion.schemas.situation import PerceptionContextV1, SituationBriefV1

NOW = datetime.now(timezone.utc)


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _diag():
    from orion.schemas.situation import SituationDiagnosticsV1

    return SituationDiagnosticsV1()


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """Never touch a database from these tests."""
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: None)


# --- provider states -------------------------------------------------------


def test_disabled_by_default() -> None:
    # Camera-derived content about a private home is opt-in.
    assert settings_from_runtime(SimpleNamespace()).perception_enabled is False


def test_disabled_yields_unavailable_not_an_error() -> None:
    diag = _diag()
    ctx = _build_perception_context(_cfg(perception_enabled=False), diag)
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["perception"] == "disabled"


def test_no_percept_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: None)
    ctx = _build_perception_context(_cfg(perception_enabled=True), _diag())
    assert ctx.available is False
    assert ctx.source == "unavailable"
    assert ctx.scene_summary is None


def test_fresh_percept_is_available(monkeypatch) -> None:
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "Three chairs and a door are visible.", "observed_at": NOW},
    )
    ctx = _build_perception_context(_cfg(perception_enabled=True), _diag())
    assert ctx.available is True
    assert ctx.source == "live"
    assert ctx.scene_summary == "Three chairs and a door are visible."
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds < 60


def test_stale_percept_is_withheld_entirely(monkeypatch) -> None:
    """
    The core honesty gate. An old percept presented as current is a
    confabulation with a real referent, which is worse than saying nothing --
    so the summary is not merely flagged stale, it is not carried at all.
    """
    old = NOW - timedelta(hours=2)
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "A person is at the desk.", "observed_at": old},
    )
    ctx = _build_perception_context(_cfg(perception_enabled=True, perception_max_age_seconds=900), _diag())
    assert ctx.available is False
    assert ctx.source == "stale"
    assert ctx.scene_summary is None, "a stale summary must not ride along in the payload"
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds > 900


def test_age_boundary_is_inclusive_of_the_threshold(monkeypatch) -> None:
    at_limit = NOW - timedelta(seconds=900)
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "A door.", "observed_at": at_limit},
    )
    ctx = _build_perception_context(_cfg(perception_enabled=True, perception_max_age_seconds=900), _diag())
    assert ctx.available is True, "exactly at the threshold is still fresh"


def test_reader_exception_fails_open(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _boom)
    diag = _diag()
    ctx = _build_perception_context(_cfg(perception_enabled=True), diag)
    assert ctx.available is False
    assert ctx.source == "error"
    assert "db gone" in diag.provider_errors["perception"]


def test_empty_narrative_is_not_a_percept(monkeypatch) -> None:
    monkeypatch.setattr(
        situation_mod, "fetch_latest_percept", lambda: {"scene_summary": "", "observed_at": NOW}
    )
    ctx = _build_perception_context(_cfg(perception_enabled=True), _diag())
    assert ctx.available is False


# --- prompt rendering ------------------------------------------------------


def _brief(perception: PerceptionContextV1) -> SituationBriefV1:
    """Build a real brief via the production helpers, then swap in the percept.

    Hand-rolling the sub-contexts means guessing their required fields; using
    the real builders means this test breaks if their shape drifts, which is
    the more useful failure.
    """
    cfg = _cfg(perception_enabled=False)
    diag = _diag()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    return SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        # _build_conversation_phase is now async (Redis-backed session turn
        # state, see session_turn_phase.py) -- these tests only care about
        # perception rendering, so no bus is bound here and the call
        # fails open to phase="unknown" (an unbound-bus WARNING is expected
        # and harmless in this file's test output).
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        perception=perception,
    )


def test_available_percept_renders_with_its_age() -> None:
    brief = _brief(
        PerceptionContextV1(
            available=True,
            source="live",
            scene_summary="Three chairs and a door are visible.",
            observation_age_seconds=120,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Room (seen 2 min ago): Three chairs and a door are visible." in text


def test_unavailable_renders_as_not_having_looked_not_as_an_empty_room() -> None:
    """
    "I haven't seen anything" and "there is nothing there" are different
    claims, and only the first is true when the camera is stale or off.
    """
    text = _build_prompt_fragment(_brief(PerceptionContextV1()), 4000).compact_text
    assert "haven't seen anything recently" in text
    for wrong in ("room is empty", "nothing in the room", "the room is quiet"):
        assert wrong not in text.lower()


def test_stale_percept_summary_never_reaches_the_prompt() -> None:
    stale = PerceptionContextV1(
        available=False, source="stale", observation_age_seconds=7200
    )
    text = _build_prompt_fragment(_brief(stale), 4000).compact_text
    assert "haven't seen anything recently" in text


# --- privacy contract ------------------------------------------------------


def test_schema_exposes_no_identity_or_raw_frame_fields() -> None:
    """
    The exposed-field list IS the privacy contract. `extra="forbid"` means a
    future caller cannot smuggle one of these in without changing this schema,
    which is the point.
    """
    fields = set(PerceptionContextV1.model_fields)
    for banned in (
        "entities",
        "faces",
        "identities",
        "objects",
        "boxes",
        "bounding_boxes",
        "frame_path",
        "image_path",
        "embedding",
        "detections",
    ):
        assert banned not in fields, f"{banned} must not be exposed to the prompt"


def test_percepts_are_session_only_by_default() -> None:
    assert PerceptionContextV1().privacy_mode == "session_only"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(Exception):
        PerceptionContextV1(frame_path="/mnt/telemetry/vision/frames/x.jpg")
