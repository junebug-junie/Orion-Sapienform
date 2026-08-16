"""Orion's own thinking yields the lane to Juniper's (ROADMAP A3).

Until this, a metacog tick, a reverie or a dream competed for atlas's 4 `quick` slots on exactly
equal terms with an interactive chat turn. A2 measured that lane over 27.74 h: completely full
**4.01%** of the time, blocking **174x** more than Poisson arrivals would at the same offered
load, because work arrives in batches rather than steadily.

`quick_background` is the same upstream and the same model. The only difference is the gateway's
admission gate (`priority_admission.py`), which holds `reserved_free_slots` free for foreground
callers -- so a background request waits instead of making Juniper wait. Measured: that
reservation is already enforced 4.84% of the time.

The classification of "Orion's own work" is NOT invented here. `resolve_llm_lane_for_step`
already sorts every step into chat/spark/background/agent with a priority, and `low` is exactly
the Orion-initiated set. These tests pin that reuse, and pin the three places the redirect must
NOT reach.
"""
from __future__ import annotations

import pytest

from app.llm_lane import resolve_llm_lane_for_step


class _Step:
    def __init__(self, verb_name: str, step_name: str = "s"):
        self.verb_name = verb_name
        self.step_name = step_name


class _Settings:
    exec_lane = "legacy"


def _priority(verb: str, ctx=None) -> str:
    return resolve_llm_lane_for_step(
        step=_Step(verb), ctx=ctx or {}, settings=_Settings()
    )["priority"]


def _decide(route, priority, *, override=None, enabled=True):
    """The REAL decision function from executor.py -- not a copy of it.

    Calling the shipped code matters here: an earlier draft of this file reimplemented the
    three-clause condition, which would have passed while the executor diverged.
    """
    from app.executor import _apply_autonomous_background_route

    return _apply_autonomous_background_route(
        route=route, override=override, lane_priority=priority, enabled=enabled
    )


# ------------------------------------------------------------- what gets backgrounded

@pytest.mark.parametrize("verb", [
    "dream_cycle", "dream_synthesis", "log_orion_metacognition", "reverie_narrate",
])
def test_orions_background_verbs_are_low_priority(verb):
    """These are _BACKGROUND_VERBS -- Orion dreaming and journalling, nobody waiting on them."""
    assert _priority(verb) == "low"


def test_introspect_spark_is_low_priority():
    """Orion's internal analysis of its own state. Also nobody waiting."""
    assert _priority("introspect_spark") == "low"


@pytest.mark.parametrize("verb", [
    "dream_cycle", "dream_synthesis", "log_orion_metacognition", "reverie_narrate",
    "introspect_spark",
])
def test_low_priority_quick_work_moves_to_the_background_lane(verb):
    route, moved = _decide("quick", _priority(verb))
    assert route == "quick_background"
    assert moved is True


# ------------------------------------------------------------- what must NOT be touched

def test_an_interactive_chat_turn_keeps_its_lane():
    """chat_general is Juniper talking. Demoting it would invert the entire point."""
    assert _priority("chat_general") == "high"
    assert _decide("quick", _priority("chat_general")) == ("quick", False)


def test_the_chat_route_is_never_demoted_even_at_low_priority():
    """Only `quick` is redirected. A step routed to the deep lane stays there regardless --
    `chat` runs on circe and has its own contention profile."""
    assert _decide("chat", "low") == ("chat", False)


def test_the_metacog_route_is_untouched():
    """A2 measured metacog at 0.00% all-busy over 27.74 h, never exceeding 2 of 4 slots. It
    never fills, so moving it would be ceremony -- the roadmap's own kill gate for that lane."""
    assert _decide("metacog", "low") == ("metacog", False)


def test_an_explicit_override_always_wins():
    """A caller that named a lane is not second-guessed, even for low-priority work."""
    assert _decide("quick", "low", override="quick") == ("quick", False)
    assert _decide("quick", "low", override="chat") == ("quick", False)


def test_normal_priority_agent_work_is_not_backgrounded():
    """`agent` is priority normal, not low -- deliberately outside the redirect."""
    assert _decide("quick", "normal") == ("quick", False)


def test_a_none_route_stays_none():
    """Steps with no default mapping fall through to the gateway's own choice."""
    assert _decide(None, "low") == (None, False)


# ------------------------------------------------------------- the kill gate

def test_the_flag_off_restores_the_previous_behaviour_exactly():
    """The roadmap's kill gate: one env key, and the fallback is what shipped before."""
    for verb in ("dream_cycle", "introspect_spark", "reverie_narrate"):
        assert _decide("quick", _priority(verb), enabled=False) == ("quick", False)


# ------------------------------------------------------------- the classification is reused

def test_the_execution_lane_can_also_mark_work_background():
    """`execution_lane == "background"` reaches low priority without naming a verb, so the
    redirect follows the existing classifier rather than a hardcoded verb list of its own."""
    assert _priority("some_new_verb", ctx={"execution_lane": "background"}) == "low"
    assert _decide("quick", _priority("some_new_verb", ctx={"execution_lane": "background"})) == (
        "quick_background", True)


def test_an_explicit_llm_lane_still_drives_priority():
    """A caller setting llm_lane=chat gets high priority and keeps the foreground lane, even on
    a verb that would otherwise be background."""
    assert _priority("dream_cycle", ctx={"options": {"llm_lane": "chat"}}) == "high"
    assert _decide("quick", _priority("dream_cycle", ctx={"options": {"llm_lane": "chat"}})) == (
        "quick", False)


def test_quick_background_is_an_accepted_route_value():
    """The redirect must produce a route the override validator recognises, or a later
    round-trip through _resolve_llm_route_override would silently drop it."""
    from app.executor import _ACCEPTED_LLM_ROUTE_OVERRIDES, _resolve_llm_route_override

    assert "quick_background" in _ACCEPTED_LLM_ROUTE_OVERRIDES
    assert _resolve_llm_route_override({"options": {"llm_route": "quick_background"}}) == (
        "quick_background", "quick_background")
