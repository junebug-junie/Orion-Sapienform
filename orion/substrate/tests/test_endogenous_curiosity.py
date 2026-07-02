from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from orion.substrate.endogenous_curiosity import (
    HARD_BUDGET_CEILING,
    EndogenousCuriosityConfig,
    endogenous_curiosity_candidates,
)
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator


def _node(node_id: str, prediction_error: float) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, metadata={"prediction_error": prediction_error})


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


def test_budget_cap_and_hard_ceiling() -> None:
    nodes = [_node(f"node:{i}", 0.6 + i * 0.01) for i in range(20)]
    capped = endogenous_curiosity_candidates(nodes=nodes, config=_enabled(budget=2))
    assert len(capped) == 2
    # strongest first
    assert capped[0].signal_strength >= capped[1].signal_strength

    runaway = endogenous_curiosity_candidates(nodes=nodes, config=_enabled(budget=999))
    assert len(runaway) == HARD_BUDGET_CEILING


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
