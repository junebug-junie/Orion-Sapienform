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
