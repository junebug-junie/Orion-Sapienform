from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.endogenous_curiosity import (
    HARD_BUDGET_CEILING,
    EndogenousCuriosityConfig,
    _PREDICTION_ERROR_DECAY_HORIZON_SECONDS,
    endogenous_curiosity_candidates,
)
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator

_NOW = datetime(2026, 7, 16, 0, 0, 0, tzinfo=timezone.utc)


def _node(node_id: str, prediction_error: float, *, observed_at: datetime | None = None) -> SimpleNamespace:
    """``observed_at`` defaults to ``None`` (no ``.temporal``), matching every
    pre-existing test in this file -- those nodes are treated as unaged (decay
    factor 1.0) by ``_prediction_error_staleness_decay``, so this default keeps
    all of them behaving exactly as before the staleness fix. Tests that care
    about age pass ``observed_at`` explicitly and also pass ``now=_NOW`` to
    ``endogenous_curiosity_candidates`` so the two are measured consistently."""
    node = SimpleNamespace(node_id=node_id, metadata={"prediction_error": prediction_error})
    if observed_at is not None:
        node.temporal = SimpleNamespace(observed_at=observed_at)
    return node


def _enabled(**overrides) -> EndogenousCuriosityConfig:
    return EndogenousCuriosityConfig(enabled=True, **overrides)


def test_disabled_by_default_env(monkeypatch) -> None:
    monkeypatch.delenv("ORION_ENDOGENOUS_CURIOSITY_ENABLED", raising=False)
    monkeypatch.delenv("ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH", raising=False)
    assert endogenous_curiosity_candidates(nodes=[_node("node:a", 0.9)]) == []


def test_kill_switch_beats_enable() -> None:
    config = EndogenousCuriosityConfig(enabled=True, kill_switch=True)
    assert (
        endogenous_curiosity_candidates(nodes=[_node("node:a", 0.9)], config=config) == []
    )


def test_sustained_prediction_error_seeds_candidates() -> None:
    candidates = endogenous_curiosity_candidates(
        nodes=[_node("node:hot", 0.8), _node("node:calm", 0.1)],
        config=_enabled(),
    )
    assert len(candidates) == 1
    seed = candidates[0]
    assert seed.signal_type == "curiosity_candidate"
    assert seed.target_zone == "concept_graph"
    assert seed.focal_node_refs == ["node:hot"]
    assert seed.signal_strength == 0.8
    assert "endogenous_seed" in seed.notes


def test_missing_prediction_error_never_seeds_even_at_zero_threshold() -> None:
    """Locks in the `raw_error <= 0.0: continue` short-circuit (review finding):
    a node with no/zero `prediction_error` must not seed a spurious "sustained
    prediction error" candidate at signal_strength=0.0, even when the threshold
    itself is configured at 0.0 (which would otherwise let `0.0 < 0.0 == False`
    pass through, as it did before this fix)."""
    no_error = _node("node:quiet", 0.0)
    missing = SimpleNamespace(node_id="node:no-key", metadata={})
    candidates = endogenous_curiosity_candidates(
        nodes=[no_error, missing], config=_enabled(min_prediction_error=0.0)
    )
    assert candidates == []


def test_stale_prediction_error_decays_and_is_not_sustained() -> None:
    """Regression for the sibling of PR #1061's salience decay-bypass bug:
    `metadata["prediction_error"]` never decays on its own (it's a raw upsert
    snapshot), so an unguarded read let a node surprising once, days ago, stay
    labeled "sustained prediction error" at full strength forever -- live-
    confirmed 2026-07-16 (node:substrate.transport pinned at signal_strength=1.0
    across 1,428 consecutive persisted candidate sets). A node last observed
    well past the decay horizon must score strictly lower than an
    identically-seeded node observed right now, and must not clear the default
    threshold once decayed below it."""
    fresh = _node("node:fresh", 0.9, observed_at=_NOW)
    stale = _node(
        "node:stale",
        0.9,
        observed_at=_NOW - timedelta(seconds=_PREDICTION_ERROR_DECAY_HORIZON_SECONDS * 4),
    )

    candidates = endogenous_curiosity_candidates(
        nodes=[fresh, stale], config=_enabled(min_prediction_error=0.0), now=_NOW
    )
    by_node = {c.focal_node_refs[0]: c for c in candidates if c.notes and "source:prediction_error" in c.notes}

    assert by_node["node:fresh"].signal_strength == 0.9
    assert by_node["node:stale"].signal_strength == 0.0
    assert by_node["node:stale"].signal_strength < by_node["node:fresh"].signal_strength

    # At the default (non-zero) threshold, the decayed-to-zero stale node must
    # not surface as a candidate at all -- it should not win any share of the
    # bounded per-cycle budget just because it was surprising once, long ago.
    thresholded = endogenous_curiosity_candidates(nodes=[fresh, stale], config=_enabled(), now=_NOW)
    assert all(c.focal_node_refs != ["node:stale"] for c in thresholded)


def test_prediction_error_staleness_decay_is_linear_within_horizon() -> None:
    half_horizon = _NOW - timedelta(seconds=_PREDICTION_ERROR_DECAY_HORIZON_SECONDS / 2)
    node = _node("node:half-decayed", 1.0, observed_at=half_horizon)
    candidates = endogenous_curiosity_candidates(
        nodes=[node], config=_enabled(min_prediction_error=0.0), now=_NOW
    )
    assert len(candidates) == 1
    assert abs(candidates[0].signal_strength - 0.5) < 1e-6


def test_budget_cap_and_hard_ceiling() -> None:
    nodes = [_node(f"node:{i}", 0.6 + i * 0.01) for i in range(20)]
    capped = endogenous_curiosity_candidates(nodes=nodes, config=_enabled(budget=2))
    assert len(capped) == 2
    # strongest first
    assert capped[0].signal_strength >= capped[1].signal_strength

    runaway = endogenous_curiosity_candidates(nodes=nodes, config=_enabled(budget=999))
    assert len(runaway) == HARD_BUDGET_CEILING


def test_repair_pressure_at_moderate_level_seeds_when_threshold_lowered() -> None:
    appraisal = SimpleNamespace(
        dimensions={"level": 0.28},
        causal_molecule_ids=["mol:chat"],
        summary="moderate chat repair pressure",
        confidence=0.65,
    )
    candidates = endogenous_curiosity_candidates(
        repair_appraisal=appraisal,
        config=_enabled(min_repair_level=0.25),
    )
    assert len(candidates) == 1
    assert candidates[0].signal_strength == 0.28
    assert candidates[0].evidence_summary == "moderate chat repair pressure"


def test_min_repair_level_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ENDOGENOUS_CURIOSITY_MIN_REPAIR_LEVEL", "0.25")
    config = EndogenousCuriosityConfig.from_env()
    assert config.min_repair_level == 0.25


def test_repair_pressure_appraisal_seeds_candidate() -> None:
    appraisal = SimpleNamespace(
        dimensions={"level": 0.75},
        causal_molecule_ids=["mol:1", "mol:2"],
        summary="trust rupture cluster",
        confidence=0.7,
    )
    candidates = endogenous_curiosity_candidates(repair_appraisal=appraisal, config=_enabled())
    assert len(candidates) == 1
    assert candidates[0].evidence_summary == "trust rupture cluster"
    assert candidates[0].focal_node_refs == ["mol:1", "mol:2"]

    calm = SimpleNamespace(dimensions={"level": 0.2}, causal_molecule_ids=[], summary="", confidence=0.7)
    assert endogenous_curiosity_candidates(repair_appraisal=calm, config=_enabled()) == []


def test_attention_open_loops_seed_candidates() -> None:
    loop = SimpleNamespace(
        id="open-loop-1",
        description="surprising transport batch",
        already_known=False,
        novelty=0.7,
        confidence=0.8,
        source_refs=["node:transport"],
    )
    known = SimpleNamespace(
        id="open-loop-2",
        description="known thing",
        already_known=True,
        novelty=0.9,
        confidence=0.8,
        source_refs=[],
    )
    frame = SimpleNamespace(open_loops=[loop, known], deferred_items=["open-loop-1"])
    candidates = endogenous_curiosity_candidates(attention_frame=frame, config=_enabled())
    assert len(candidates) == 1
    assert candidates[0].focal_node_refs == ["node:transport"]
    assert "source:attention_open_loop" in candidates[0].notes


def test_candidates_never_target_strict_or_autonomy_zones() -> None:
    nodes = [_node(f"node:{i}", 0.9) for i in range(5)]
    appraisal = SimpleNamespace(dimensions={"level": 0.9}, causal_molecule_ids=[], summary="s", confidence=0.9)
    candidates = endogenous_curiosity_candidates(
        nodes=nodes, repair_appraisal=appraisal, config=_enabled(budget=8)
    )
    assert candidates
    assert all(c.target_zone == "concept_graph" for c in candidates)


def test_evaluator_decides_over_endogenous_signals_without_invocation_authority() -> None:
    """Endogenous seeds ride the existing decision policy: a strong seed can
    reach 'invoke' (which downstream is still proposal-governed), and the
    strict-zone guardrails in _decide are untouched."""
    evaluator = FrontierCuriosityEvaluator(store=MagicMock())
    seeds = endogenous_curiosity_candidates(nodes=[_node("node:hot", 0.85)], config=_enabled())
    decision = evaluator._decide(signals=seeds)
    assert decision.outcome == "invoke"
    assert decision.target_zone == "concept_graph"

    weak_seeds = endogenous_curiosity_candidates(
        nodes=[_node("node:mild", 0.45)], config=_enabled(min_prediction_error=0.4)
    )
    weak_decision = evaluator._decide(signals=weak_seeds)
    assert weak_decision.outcome == "noop"


def test_world_coverage_gap_passes_through_as_curiosity_seed() -> None:
    gap = FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="gpu section empty",
        confidence=0.65,
        notes=["run_id:wp-1"],
    )
    candidates = endogenous_curiosity_candidates(
        coverage_gap_signals=[gap],
        config=_enabled(),
    )
    assert len(candidates) == 1
    assert candidates[0].signal_type == "curiosity_candidate"
    assert "hardware_compute_gpu" in candidates[0].focal_node_refs[0]
    assert "world_coverage_gap" in candidates[0].notes


# --- Repair-pressure replay fix (2026-10-10) -------------------------------
# Live bug: the worker took the all-time max repair_pressure_level across every
# chat-projection turn, so one 0.913 HIGH turn was replayed as a candidate on
# every tick for days. These tests build real ChatTurnStateV1 rows.

from orion.schemas.chat_projection import ChatTurnStateV1  # noqa: E402
from orion.substrate.endogenous_curiosity import repair_appraisal_from_chat_turns  # noqa: E402

_LIVE_SPIKE = 0.9129342275597288
_LIVE_REST = 0.087


def _turn(level: float, observed_at: datetime, *, turn_id: str = "t") -> ChatTurnStateV1:
    return ChatTurnStateV1(
        trace_id=f"node:{turn_id}",
        turn_id=turn_id,
        session_id="s",
        node_id="node",
        observed_at=observed_at,
        repair_pressure_level=level,
        repair_pressure_confidence=0.65,
        has_repair_signal=level >= 0.6,
        evidence_event_ids=[f"ev-{turn_id}"],
        last_updated_at=observed_at,
    )


def _repair_seeds(turns, *, now=_NOW):
    appraisal = repair_appraisal_from_chat_turns(turns, now=now)
    return [
        s
        for s in endogenous_curiosity_candidates(repair_appraisal=appraisal, config=_enabled(), now=now)
        if "source:repair_pressure" in s.notes
    ]


def test_fresh_repair_spike_still_produces_candidate() -> None:
    seeds = _repair_seeds([_turn(_LIVE_SPIKE, _NOW - timedelta(seconds=60))])
    assert len(seeds) == 1
    assert seeds[0].focal_node_refs == ["ev-t"]
    assert 0.6 < seeds[0].signal_strength < _LIVE_SPIKE


def test_old_repair_spike_decays_below_min_and_stops_emitting() -> None:
    # 0.913 * (1 - age/1800) < 0.6 once age > ~617s.
    assert _repair_seeds([_turn(_LIVE_SPIKE, _NOW - timedelta(seconds=700))]) == []
    assert repair_appraisal_from_chat_turns(
        [_turn(_LIVE_SPIKE, _NOW - timedelta(seconds=_PREDICTION_ERROR_DECAY_HORIZON_SECONDS + 1))],
        now=_NOW,
    ) is None


def test_live_shape_old_spike_plus_calm_turns_emits_no_repair_candidate() -> None:
    """Regression: the exact live shape -- a 0.913 spike four days old followed
    by many calm 0.087 turns. Old code returned level=0.913 forever."""
    spike_at = _NOW - timedelta(days=4)
    turns = [_turn(_LIVE_SPIKE, spike_at, turn_id="spike")]
    turns += [
        _turn(_LIVE_REST, spike_at + timedelta(hours=h), turn_id=f"calm{h}")
        for h in range(1, 96)
    ]
    turns.append(_turn(_LIVE_REST, _NOW - timedelta(minutes=5), turn_id="calm_now"))
    appraisal = repair_appraisal_from_chat_turns(turns, now=_NOW)
    # The surviving reading is a recent calm turn, not the replayed spike.
    assert appraisal is not None
    assert appraisal.dimensions["level"] < _LIVE_REST + 1e-9
    assert appraisal.causal_molecule_ids == ["ev-calm_now"]
    assert _repair_seeds(turns) == []


def test_turn_without_timestamp_is_not_treated_as_fresh() -> None:
    stale = SimpleNamespace(repair_pressure_level=_LIVE_SPIKE, observed_at=None)
    assert repair_appraisal_from_chat_turns([stale], now=_NOW) is None


def test_recent_spike_beats_older_bigger_spike() -> None:
    turns = [
        _turn(0.99, _NOW - timedelta(seconds=1500), turn_id="old"),
        _turn(0.8, _NOW - timedelta(seconds=10), turn_id="new"),
    ]
    appraisal = repair_appraisal_from_chat_turns(turns, now=_NOW)
    assert appraisal.causal_molecule_ids == ["ev-new"]


def test_future_dated_turn_is_treated_as_age_zero_and_naive_is_utc() -> None:
    future = _turn(0.8, _NOW + timedelta(seconds=120), turn_id="future")
    assert repair_appraisal_from_chat_turns([future], now=_NOW).dimensions["level"] == 0.8
    naive = _turn(0.8, (_NOW - timedelta(seconds=900)).replace(tzinfo=None), turn_id="naive")
    level = repair_appraisal_from_chat_turns([naive], now=_NOW.replace(tzinfo=None)).dimensions["level"]
    assert abs(level - 0.4) < 1e-9
