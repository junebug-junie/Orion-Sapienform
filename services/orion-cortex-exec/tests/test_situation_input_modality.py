"""2026-08-25: a SPOKEN turn is distinguishable from a typed one in the prompt.

`SurfaceContextV1.input_modality` has carried a `"spoken"` literal since the
situation brief was first built, and Hub's browser has been sending
`surface_context: {input_modality: "spoken"}` on every microphone payload --
but `_build_prompt_fragment` never rendered the field, so it reached no
prompt. Orion received Juniper's transcribed words with no way to know they
had been said out loud. This file covers the render side; the plumbing that
finally supplies the value on the unified-turn path is covered by
`services/orion-hub/tests/test_unified_turn_surface_context.py`.

The truncation case below is the one that actually matters. The live cap is
`orion_situation_prompt_max_chars=1200`, and a line that gets silently sliced
off the end by that cap is a feature that does not exist -- exactly the
interaction PR #1865 hit when its affect line pushed a fixture past its own
boundary.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from orion.situational import context as situation_mod
from orion.situational.context import (
    SituationSettings,
    _build_prompt_fragment,
    settings_from_runtime,
)
from orion.schemas.situation import SituationBriefV1, SurfaceContextV1

NOW = datetime.now(timezone.utc)

# The real production default (orion/situational/context.py's
# settings_from_runtime fallback), not a roomy test-only number -- the whole
# point of test_spoken_line_survives_the_live_prompt_cap is to exercise the
# cap Hub actually applies.
LIVE_MAX_CHARS = 1200


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _brief(input_modality: str) -> SituationBriefV1:
    """Built through the production helpers, same technique as
    test_situation_affect_context.py's own _brief(), so this breaks if the
    real builders' shape drifts rather than passing against a stale copy."""
    cfg = _cfg()
    diag = SimpleNamespace(provider_status={})
    time_ctx = situation_mod._build_time_context(cfg, diag)
    return SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        conversation_phase=asyncio.run(
            situation_mod._build_conversation_phase({}, time_ctx, NOW)
        ),
        place=situation_mod._build_place_context(cfg),
        surface=SurfaceContextV1(surface="hub_desktop", input_modality=input_modality),
    )


def test_spoken_turn_is_announced_as_spoken() -> None:
    text = _build_prompt_fragment(_brief("spoken"), 4000).compact_text
    assert "SPOKE this turn aloud" in text
    # The behavioural half: it is not enough to say "spoken", the prompt has
    # to say what Orion should DO differently about it.
    assert "transcription artifacts" in text


def test_typed_turn_says_nothing_at_all() -> None:
    """A line on every typed turn announcing that nothing special happened
    is pure prompt noise -- typed is the overwhelming majority case."""
    text = _build_prompt_fragment(_brief("typed"), 4000).compact_text
    assert "Input modality" not in text
    assert "SPOKE" not in text


def test_unknown_modality_stays_silent_like_typed() -> None:
    text = _build_prompt_fragment(_brief("unknown"), 4000).compact_text
    assert "Input modality" not in text


def test_other_known_modality_is_named_plainly() -> None:
    text = _build_prompt_fragment(_brief("external_room"), 4000).compact_text
    assert "Input modality: external_room." in text


def test_spoken_line_survives_the_live_prompt_cap_under_real_truncation() -> None:
    """Regression guard for the real failure mode, with the truncation
    actually happening rather than assumed.

    `_build_prompt_fragment` slices the tail (`compact[: max_chars - 1] + "…"`),
    so whether a line survives is purely a question of where it sits in the
    list. The spoken line is inserted 4th, immediately after Presence and
    ahead of weather/lab/room/affect -- deliberately, so the cap eats the
    trailing caution lines long before it reaches this one.

    A minimal brief comes in at ~989 chars, comfortably under the live
    1200 cap, which would make a naive assertion here pass without ever
    exercising the cap at all -- a false green. So this fills the affect
    slot with a summary at its real 300-char ceiling to push the fragment
    PAST the cap, asserts truncation genuinely occurred, and only then
    asserts the spoken line is still present.
    """
    from orion.schemas.situation import AffectContextV1

    brief = _brief("spoken")
    brief.affect = AffectContextV1(
        available=True,
        summary="x" * 300,
        observed_at=NOW,
        observation_age_seconds=5.0,
        source="live",
    )
    fragment = _build_prompt_fragment(brief, LIVE_MAX_CHARS)

    assert len(fragment.compact_text) <= LIVE_MAX_CHARS
    assert fragment.compact_text.endswith("…"), (
        "fixture no longer overflows the cap, so this test would pass "
        "without exercising truncation at all -- lengthen the fixture"
    )
    assert "SPOKE this turn aloud" in fragment.compact_text, (
        "spoken-modality line was truncated away at the live "
        f"{LIVE_MAX_CHARS}-char cap; it must stay ahead of the "
        "weather/lab/room/affect lines in _build_prompt_fragment"
    )
