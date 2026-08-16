"""ai-town (and any other external world) never reaches the human governor queue.

Regression cover for the 2026-08-14 finding: the live governor queue held 621
proposals, 610 of which were ai-town NPC dialogue -- 98.2% noise against 11 real
conversations with Juniper. No existing score could separate them (salience is
pinned at exactly 1.0 for this entire path by construction), so the gate keys off
the one honest discriminator: the turn's source platform.

The gate must hold in BOTH directions, which is what most of these tests are
about. Suppressing NPC chatter is the easy half; the half that would actually
hurt is a window containing Juniper's own words being auto-activated without her
ever seeing it.
"""
from __future__ import annotations

import pytest

from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.formation_executor import GovernorPathRequired, auto_activate
from orion.memory.crystallization.formation_policy import (
    FormationPolicy,
    resolve_formation_policy,
)
from orion.memory.crystallization.intake_consolidation_window import (
    _window_source_platform,
    build_crystallization_from_window,
)
from orion.memory.crystallization.salience import apply_salience


def _turn(corr: str, platform: str | None, prompt: str = "hi") -> dict:
    return {
        "correlation_id": corr,
        "prompt": prompt,
        "response": "ok",
        "spark_meta": {},
        "source_platform": platform,
    }


def _stance_window(turns: list[dict], window_id: str = "win-1"):
    return build_crystallization_from_window(
        memory_window_id=window_id,
        turns=turns,
        gate=ConsolidationGateResult(
            action="propose", dominant_shift="STANCE", grammar_event_ids=[]
        ),
    )


# --------------------------------------------------------------------------
# _window_source_platform: unanimity in both directions
# --------------------------------------------------------------------------


def test_all_aitown_window_is_aitown():
    turns = [_turn("c1", "aitown"), _turn("c2", "aitown"), _turn("c3", "aitown")]
    assert _window_source_platform(turns) == "aitown"


def test_all_direct_window_is_none():
    turns = [_turn("c1", None), _turn("c2", None)]
    assert _window_source_platform(turns) is None


def test_mixed_window_is_none_not_aitown():
    """The 26 live mixed windows: one real turn is enough to demand review."""
    turns = [_turn("c1", "aitown"), _turn("c2", "aitown"), _turn("c3", None, "hey, I'm back")]
    assert _window_source_platform(turns) is None


def test_two_different_platforms_is_none():
    turns = [_turn("c1", "aitown"), _turn("c2", "discord")]
    assert _window_source_platform(turns) is None


def test_empty_window_is_none():
    assert _window_source_platform([]) is None


def test_missing_key_treated_as_direct():
    """Turn dicts written before this field existed have no source_platform at
    all -- they must read as "not external", never as unanimous-with-whatever."""
    turns = [{"correlation_id": "c1", "prompt": "p", "response": "r"}, _turn("c2", "aitown")]
    assert _window_source_platform(turns) is None


def test_empty_string_platform_treated_as_direct():
    turns = [_turn("c1", ""), _turn("c2", "")]
    assert _window_source_platform(turns) is None


# --------------------------------------------------------------------------
# provenance carries it
# --------------------------------------------------------------------------


def test_provenance_records_platform():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    assert crys.provenance["source_platform"] == "aitown"


def test_provenance_records_none_for_mixed():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", None)])
    assert crys.provenance["source_platform"] is None


# --------------------------------------------------------------------------
# formation policy routing
# --------------------------------------------------------------------------


def test_aitown_stance_auto_activates_instead_of_queueing():
    """The whole point: a stance is a GATED_KIND and would normally queue."""
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    assert crys.kind == "stance"
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.AUTO_ACTIVATE
    assert reasons == ["external_platform:aitown"]


def test_direct_stance_still_queues():
    crys = _stance_window([_turn("c1", None, "bruh you are Orion")])
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.GOVERNOR_QUEUE
    assert reasons == ["gated_kind:stance"]


def test_mixed_window_stance_still_queues():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", None, "hey, I'm back")])
    policy, _ = resolve_formation_policy(crys)
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_unlisted_platform_still_queues():
    crys = _stance_window([_turn("c1", "discord"), _turn("c2", "discord")])
    policy, _ = resolve_formation_policy(crys)
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_empty_platform_allowlist_disables_the_gate():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    policy, _ = resolve_formation_policy(crys, auto_activate_platforms=frozenset())
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_duplicate_still_wins_over_platform_gate():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    policy, reasons = resolve_formation_policy(crys, duplicate_id="dup-1")
    assert policy == FormationPolicy.REINFORCE_EXISTING
    assert reasons == ["duplicate:dup-1"]


# --------------------------------------------------------------------------
# only "stance" is bypassable
# --------------------------------------------------------------------------


def test_other_gated_kinds_are_not_bypassed_by_the_platform_gate():
    """An allowlisted platform must not push a contradiction/decision/attractor/
    failure_mode straight to active. Unreachable from this producer today
    (_KIND_FOR_SHIFT cannot emit them), which is exactly why it is pinned: the
    hazard would arrive silently with the next shift mapping, and fictional NPC
    roleplay writing a "contradiction" into active memory unreviewed is not a
    failure anyone would notice from the outside."""
    for kind in ("contradiction", "decision", "attractor", "failure_mode"):
        crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
        crys.kind = kind
        policy, reasons = resolve_formation_policy(crys)
        assert policy == FormationPolicy.GOVERNOR_QUEUE, kind
        assert reasons == [f"gated_kind:{kind}"], kind


def test_non_gated_kinds_are_unaffected_by_the_platform_gate():
    """semantic/episode/open_loop auto-activate on their own; the platform gate
    must not be what is doing the work for them."""
    for kind in ("semantic", "episode", "open_loop", "procedure"):
        crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
        crys.kind = kind
        policy, reasons = resolve_formation_policy(crys, auto_activate_platforms=frozenset())
        assert policy == FormationPolicy.AUTO_ACTIVATE, kind
        assert reasons == [], kind


# --------------------------------------------------------------------------
# privacy boundaries outrank the convenience gate
#
# HONESTY NOTE: both tests below hand-mutate the object after construction,
# because build_crystallization_from_window() hardcodes sensitivity="private"
# and scope=["memory_window:<id>"] -- so neither guard can actually fire for the
# only producer that currently sets source_platform. These pin the ORDERING for
# a future producer that can set them; they are not evidence of a live rail.
# --------------------------------------------------------------------------


def test_intimate_aitown_window_still_queues():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    crys.governance.sensitivity = "intimate"  # not reachable via this producer
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.GOVERNOR_QUEUE
    assert reasons == ["intimate_sensitivity"]


def test_identity_scoped_aitown_window_still_queues():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    crys.scope = ["identity:orion"]  # not reachable via this producer
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.GOVERNOR_QUEUE
    assert reasons == ["identity_scope"]


def test_window_producer_cannot_actually_reach_those_guards():
    """Pins the honesty note above, so it cannot quietly become false. If this
    starts failing, the two tests above became real rails and the note should go."""
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    assert crys.governance.sensitivity == "private"
    assert all(not s.startswith("identity:") for s in crys.scope)


# --------------------------------------------------------------------------
# formation_executor.auto_activate re-resolves policy on its own -- prove the
# allowlist actually reaches that second, decisive resolution rather than being
# silently dropped between the caller's check and the executor's.
# --------------------------------------------------------------------------


def test_executor_activates_aitown_stance():
    crys = apply_salience(_stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")]))
    activated, history = auto_activate(crys)
    assert activated.status == "active"
    assert activated.governance.approval_mode == "auto_policy"
    assert activated.governance.requires_manual_review is False
    assert history["reasons"] == ["external_platform:aitown"]


def test_executor_rejects_direct_stance():
    crys = apply_salience(_stance_window([_turn("c1", None, "bruh you are Orion")]))
    with pytest.raises(GovernorPathRequired, match="gated_kind:stance"):
        auto_activate(crys)


def test_executor_honors_caller_supplied_empty_allowlist():
    crys = apply_salience(_stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")]))
    with pytest.raises(GovernorPathRequired, match="gated_kind:stance"):
        auto_activate(crys, auto_activate_platforms=frozenset())
