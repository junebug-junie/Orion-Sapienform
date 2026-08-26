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


def test_nothing_is_truncated_at_the_live_cap_even_with_a_max_affect_summary() -> None:
    """The actual fix for the review finding, asserted as an outcome rather
    than as a truncation policy.

    Before: with a 300-char affect summary (the real
    _AFFECT_SUMMARY_MAX_CHARS ceiling) plus a 246-char spoken-modality line,
    the fragment hit exactly the 1200 cap and cut the affect privacy caution
    mid-sentence. Now the spoken line is short enough that the whole
    fragment fits with room left, so nothing is lost at all.

    If a future line pushes this back over the cap, this test fails FIRST,
    before the caution-dropping fallback silently starts eating guards.
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
    assert "…" not in fragment.compact_text, (
        "the fragment is back to overflowing the live cap; shorten a line "
        "or raise ORION_SITUATION_PROMPT_MAX_CHARS before a caution is lost"
    )
    assert "SPOKE this turn aloud" in fragment.compact_text
    for caution in fragment.caution_lines:
        assert caution in fragment.compact_text


def test_a_caution_is_never_emitted_half_written() -> None:
    """The invariant that has to hold at ANY cap, including artificially
    small ones: a caution is included whole or not at all. A half-sentence
    instruction is worse than a missing one -- it reads as complete.

    Body keeps first claim on the budget (an over-corrected earlier draft
    reserved the entire caution block and starved the body down to
    boilerplate at the 400-char cap test_situation_provider's fixture uses).
    """
    for cap in (200, 400, 700, 1000, LIVE_MAX_CHARS, 4000):
        fragment = _build_prompt_fragment(_brief("spoken"), cap)
        assert len(fragment.compact_text) <= cap, f"cap {cap} exceeded"
        for caution in fragment.caution_lines:
            # Either wholly present, or wholly absent -- never a prefix.
            if caution in fragment.compact_text:
                continue
            for cut in range(20, len(caution)):
                assert caution[:cut] not in fragment.compact_text, (
                    f"cap {cap}: partial caution emitted -- {caution[:cut]!r}"
                )


def test_the_affect_guard_is_the_last_caution_to_be_dropped() -> None:
    """Of the three cautions only one changes what Orion may SAY about a
    real reading of Juniper's face and voice; the other two are style. So
    when the budget cannot fit all three, that one must be the survivor."""
    brief = _brief("spoken")
    full = _build_prompt_fragment(brief, 4000)
    assert full.caution_lines[0].startswith("Juniper's affect read")

    # Derive the untruncated body length from the full fragment by
    # subtracting the caution suffix it appended -- NOT by rendering at a
    # small cap, which would truncate the body and give a wrong number.
    caution_suffix = "".join("\n- " + c for c in full.caution_lines)
    body_len = len(full.compact_text) - len(caution_suffix)

    # A cap with room for the body plus exactly one caution.
    cap = body_len + len("\n- ") + len(full.caution_lines[0])
    fragment = _build_prompt_fragment(brief, cap)
    kept = [c for c in full.caution_lines if c in fragment.compact_text]
    assert kept == [full.caution_lines[0]], (
        f"expected only the affect guard to survive at cap={cap}, got {len(kept)}"
    )


def test_cache_key_separates_spoken_from_typed() -> None:
    """The brief is cached per session for 300s. Without input_modality in
    the key, a spoken turn's cached brief is replayed on the next TYPED
    turn -- telling Orion "Juniper SPOKE this turn aloud" about something
    she typed (and vice versa, suppressing the line on a real spoken turn).
    Invisible before 2026-08-26 only because the value was a constant.
    """
    from orion.situational.context import _situation_cache_key

    cfg = _cfg()
    spoken_ctx = {
        "session_id": "s1",
        "metadata": {"surface_context": {"surface": "hub_desktop", "input_modality": "spoken"}},
    }
    typed_ctx = {
        "session_id": "s1",
        "metadata": {"surface_context": {"surface": "hub_desktop", "input_modality": "typed"}},
    }
    assert _situation_cache_key(spoken_ctx, cfg) != _situation_cache_key(typed_ctx, cfg)
    # Same modality + same session must still share a key, or the cache
    # stops working entirely.
    assert _situation_cache_key(spoken_ctx, cfg) == _situation_cache_key(dict(spoken_ctx), cfg)
