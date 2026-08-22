"""ai-town (and any other external world) never becomes a crystallization at all.

Regression cover for the 2026-08-14 finding: the live governor queue held 621
proposals, 610 of which were ai-town NPC dialogue -- 98.2% noise against 11 real
conversations with Juniper. No existing score could separate them (salience is
pinned at exactly 1.0 for this entire path by construction), so the gate keys off
the one honest discriminator: the turn's source platform.

2026-08-16, Juniper direct: "I do want it stored in whatever sql table, I just
don't want it on the graphs and crystalizations and such." This superseded the
first cut of the gate (AUTO_ACTIVATE, stance-only via
EXTERNAL_PLATFORM_BYPASSABLE_KINDS -- a stance still became a full active memory,
still projected, just skipped review). Now: DISCARD, every kind -- an allowlisted
platform never gets a memory_crystallizations row.

The gate must hold in BOTH directions, which is what most of these tests are
about. Suppressing NPC chatter is the easy half; the half that would actually
hurt is a window containing Juniper's own words being discarded/auto-activated
without her ever seeing it.
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


def test_aitown_stance_discards_instead_of_queueing():
    """The whole point: a stance is a GATED_KIND and would normally queue."""
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    assert crys.kind == "stance"
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.DISCARD
    assert reasons == ["discard_platform:aitown"]


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
    policy, _ = resolve_formation_policy(crys, discard_platforms=frozenset())
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_platform_gate_wins_over_duplicate():
    """Changed 2026-08-16: discard is checked FIRST, ahead of duplicate
    detection. The old ordering let an ai-town window that Jaccard-matched a
    real crystallization REINFORCE it with NPC-dialogue evidence -- worse than
    a queued proposal, since reinforcement writes straight into an already-
    active memory with no review step at all."""
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    policy, reasons = resolve_formation_policy(crys, duplicate_id="dup-1")
    assert policy == FormationPolicy.DISCARD
    assert reasons == ["discard_platform:aitown"]


# --------------------------------------------------------------------------
# discard now applies to every kind, not just "stance"
#
# This is the actual live-leak fix: before 2026-08-16, an ai-town window whose
# kind was semantic/episode/open_loop/procedure hit AUTO_ACTIVE_KINDS
# unconditionally (platform-agnostic) and became a full active memory
# regardless of platform. EXTERNAL_PLATFORM_BYPASSABLE_KINDS only ever covered
# the GATED_KINDS side (stance), so it never touched this path at all.
# --------------------------------------------------------------------------


def test_all_gated_kinds_are_discarded_for_an_allowlisted_platform():
    for kind in ("stance", "contradiction", "decision", "attractor", "failure_mode"):
        crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
        crys.kind = kind
        policy, reasons = resolve_formation_policy(crys)
        assert policy == FormationPolicy.DISCARD, kind
        assert reasons == ["discard_platform:aitown"], kind


def test_all_auto_active_kinds_are_discarded_for_an_allowlisted_platform():
    """The live leak this closes: these kinds used to AUTO_ACTIVATE for ai-town
    regardless of platform, becoming real active memories with nobody ever
    reviewing them."""
    for kind in ("semantic", "episode", "open_loop", "procedure"):
        crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
        crys.kind = kind
        policy, reasons = resolve_formation_policy(crys)
        assert policy == FormationPolicy.DISCARD, kind
        assert reasons == ["discard_platform:aitown"], kind


def test_non_gated_kinds_are_unaffected_by_the_platform_gate_when_disabled():
    """semantic/episode/open_loop auto-activate on their own when the platform
    gate is disabled; the platform gate must not be what is doing the work for
    them in that case."""
    for kind in ("semantic", "episode", "open_loop", "procedure"):
        crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
        crys.kind = kind
        policy, reasons = resolve_formation_policy(crys, discard_platforms=frozenset())
        assert policy == FormationPolicy.AUTO_ACTIVATE, kind
        assert reasons == [], kind


# --------------------------------------------------------------------------
# discard now outranks the privacy boundaries too -- deliberately: DISCARD is
# strictly MORE restrictive than GOVERNOR_QUEUE (nothing is persisted at all),
# so there is no privacy regression in letting it win.
#
# HONESTY NOTE: both tests below hand-mutate the object after construction,
# because build_crystallization_from_window() hardcodes sensitivity="private"
# and scope=["memory_window:<id>"] -- so neither guard can actually fire for the
# only producer that currently sets source_platform. These pin the ORDERING for
# a future producer that can set them; they are not evidence of a live rail.
# --------------------------------------------------------------------------


def test_intimate_aitown_window_is_still_discarded():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    crys.governance.sensitivity = "intimate"  # not reachable via this producer
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.DISCARD
    assert reasons == ["discard_platform:aitown"]


def test_identity_scoped_aitown_window_is_still_discarded():
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    crys.scope = ["identity:orion"]  # not reachable via this producer
    policy, reasons = resolve_formation_policy(crys)
    assert policy == FormationPolicy.DISCARD
    assert reasons == ["discard_platform:aitown"]


def test_window_producer_cannot_actually_reach_those_guards():
    """Pins the honesty note above, so it cannot quietly become false. If this
    starts failing, the two tests above became real rails and the note should go."""
    crys = _stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")])
    assert crys.governance.sensitivity == "private"
    assert all(not s.startswith("identity:") for s in crys.scope)


# --------------------------------------------------------------------------
# formation_executor.auto_activate re-resolves policy on its own -- prove the
# discard set actually reaches that second, decisive resolution rather than
# being silently dropped between the caller's check and the executor's. In the
# real pipeline, intake_pipeline short-circuits before ever calling
# auto_activate() for a discard-platform window -- these tests cover the
# executor's own defense-in-depth for callers that don't (bulk scripts,
# smoke replay, any future caller).
# --------------------------------------------------------------------------


def test_executor_rejects_aitown_stance_now_discarded():
    crys = apply_salience(_stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")]))
    with pytest.raises(GovernorPathRequired, match="discard_platform:aitown"):
        auto_activate(crys)


def test_executor_rejects_aitown_semantic_now_discarded():
    """The live leak, from the executor's side: this kind used to activate
    unconditionally regardless of platform."""
    crys = apply_salience(_stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")]))
    crys.kind = "semantic"
    with pytest.raises(GovernorPathRequired, match="discard_platform:aitown"):
        auto_activate(crys)


def test_executor_rejects_direct_stance():
    crys = apply_salience(_stance_window([_turn("c1", None, "bruh you are Orion")]))
    with pytest.raises(GovernorPathRequired, match="gated_kind:stance"):
        auto_activate(crys)


def test_executor_honors_caller_supplied_empty_discard_set():
    crys = apply_salience(_stance_window([_turn("c1", "aitown"), _turn("c2", "aitown")]))
    with pytest.raises(GovernorPathRequired, match="gated_kind:stance"):
        auto_activate(crys, discard_platforms=frozenset())
