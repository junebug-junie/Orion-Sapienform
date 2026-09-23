"""Unit tests for rung-5 endogenous curiosity tick wiring in substrate-runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker
from app.store import EndogenousCuriosityPersistResult


def _make_worker(
    monkeypatch,
    *,
    enabled: bool = True,
    kill_switch: bool = False,
) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "ORION_ENDOGENOUS_CURIOSITY_ENABLED", "true" if enabled else "false"
    )
    monkeypatch.setenv(
        "ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH", "true" if kill_switch else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    worker._store.load_attention_broadcast.return_value = None
    worker._store.load_chat_session_projection.return_value = None
    worker._store.load_latest_system_one_appraisal.return_value = None
    worker._store.save_endogenous_curiosity_candidates.return_value = (
        EndogenousCuriosityPersistResult(
            candidate_set_id="curiosity-test",
            gate_lineage_persisted=True,
        )
    )
    return worker


def _graph_node(node_id: str, prediction_error: float) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, metadata={"prediction_error": prediction_error})


def test_endogenous_curiosity_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=False)
    with patch("orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates") as seeds:
        worker._endogenous_curiosity_tick()
    seeds.assert_not_called()


def test_kill_switch_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True, kill_switch=True)
    with patch("orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates") as seeds:
        worker._endogenous_curiosity_tick()
    seeds.assert_not_called()


def test_endogenous_curiosity_routes_seeds_through_evaluator(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:hot": _graph_node("node:hot", 0.85)}
    )
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="evidence_gap_scan")
    run_result = SimpleNamespace(signals=[seed], decision=decision)

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ) as seeds_fn, patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        worker._endogenous_curiosity_tick()

    seeds_fn.assert_called_once()
    evaluator_cls.return_value.evaluate.assert_called_once()
    kwargs = evaluator_cls.return_value.evaluate.call_args.kwargs
    assert kwargs["operator_requested"] is False
    assert kwargs["endogenous_signals"] == [seed]


def test_endogenous_curiosity_fails_open_on_evaluator_error(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.side_effect = RuntimeError("boom")
        worker._endogenous_curiosity_tick()  # must not raise


def test_endogenous_curiosity_noop_tick_persists_empty_heartbeat(monkeypatch):
    """When no seeds qualify, tick still writes an empty candidate set for observability."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[],
    ):
        worker._endogenous_curiosity_tick()

    worker._store.save_endogenous_curiosity_candidates.assert_called_once()
    args, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
    assert args == ([],)
    assert kwargs.get("retention_hours") == worker._settings.endogenous_curiosity_candidate_retention_hours


def test_endogenous_curiosity_noop_persist_failure_does_not_break_tick(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    worker._store.save_endogenous_curiosity_candidates.side_effect = RuntimeError("db down")

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[],
    ):
        worker._endogenous_curiosity_tick()  # must not raise


def test_endogenous_curiosity_persists_bounded_candidate_set(monkeypatch):
    """Evaluator signals are persisted endogenous-first, capped at 8."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:hot": _graph_node("node:hot", 0.85)}
    )
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    endogenous = [
        SimpleNamespace(signal_type="t", notes=["endogenous_seed"], signal_strength=0.9, confidence=0.7)
        for _ in range(5)
    ]
    exogenous = [
        SimpleNamespace(signal_type="t", notes=[], signal_strength=0.5, confidence=0.6)
        for _ in range(5)
    ]
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="evidence_gap_scan")
    run_result = SimpleNamespace(signals=endogenous + exogenous, decision=decision)

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()

    worker._store.save_endogenous_curiosity_candidates.assert_called_once()
    persisted = worker._store.save_endogenous_curiosity_candidates.call_args.args[0]
    assert len(persisted) == 8
    assert persisted[:5] == endogenous  # endogenous seeds ranked first


def test_endogenous_curiosity_persist_failure_does_not_break_tick(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:hot": _graph_node("node:hot", 0.85)}
    )
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="evidence_gap_scan")
    run_result = SimpleNamespace(signals=[seed], decision=decision)
    worker._store.save_endogenous_curiosity_candidates.side_effect = RuntimeError("db down")

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()  # must not raise


def test_seed_sources_logged_for_visibility(monkeypatch, caplog):
    """2026-07-26 visibility fix: which source/node won a budget slot this
    tick must be logged. This is the generic node-iteration property that let
    bus_synaptic join this consumer automatically (PR #1377) with zero code
    change here -- and the same property that let the old broken transport
    domain go unnoticed winning slots for weeks. A new signal source joining
    silently should at least be loggable, not just silently invisible."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={"node:substrate.bus_synaptic": _graph_node("node:substrate.bus_synaptic", 0.7)}
    )
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed", "source:prediction_error"],
        focal_node_refs=["node:substrate.bus_synaptic"],
        signal_strength=0.7,
        confidence=0.7,
    )
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="evidence_gap_scan")
    run_result = SimpleNamespace(signals=[seed], decision=decision)

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        with caplog.at_level("INFO"):
            worker._endogenous_curiosity_tick()

    assert any(
        "substrate_endogenous_curiosity_seed_sources" in r.message
        and "prediction_error=node:substrate.bus_synaptic" in r.message
        for r in caplog.records
    )


def test_seed_source_logging_failure_does_not_break_tick(monkeypatch):
    """A malformed/mocked signal missing focal_node_refs (as several existing
    fixtures in this file do) must not crash the tick -- the visibility log
    is best-effort."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(signal_type="curiosity_candidate", notes=["endogenous_seed"])
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="evidence_gap_scan")
    run_result = SimpleNamespace(signals=[seed], decision=decision)

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()  # must not raise


def _curiosity_frame(level: str):
    from datetime import datetime, timedelta, timezone
    from orion.schemas.system_one_appraisal import (
        SystemOneAnswerV1,
        SystemOneAppraisalFrameV1,
        SystemOneInputStateV1,
    )
    from orion.substrate.system_one_appraisal import QUESTION_SET_ID, SYSTEM_ONE_QUESTIONS

    now = datetime.now(timezone.utc)
    probs_by_level = {
        "0": {"0": 0.7, "1": 0.2, "2": 0.1},
        "1": {"0": 0.2, "1": 0.55, "2": 0.25},
        "2": {"0": 0.1, "1": 0.2, "2": 0.7},
    }
    answers = {}
    for qid, question in SYSTEM_ONE_QUESTIONS.items():
        probs = probs_by_level[level] if qid == "curiosity_pull" else {"0": 0.7, "1": 0.2, "2": 0.1}
        answers[qid] = SystemOneAnswerV1(
            question_id=qid,
            type="score",
            score=sum(int(k) * float(v) for k, v in probs.items()),
            confidence=0.6,
            probabilities=probs,
        )
    return SystemOneAppraisalFrameV1(
        frame_id=f"frame-level-{level}",
        question_set_id=QUESTION_SET_ID,
        generated_at=now,
        expires_at=now + timedelta(seconds=90),
        provider="kev",
        model_id="kev-latest",
        source_refs=["attention.broadcast:x"],
        input_state=SystemOneInputStateV1(
            source_broadcast_projection_id="broadcast-1",
            source_broadcast_generated_at=now,
            selected_action_type="reflect",
            coalition_stability_score=0.5,
        ),
        questions=dict(SYSTEM_ONE_QUESTIONS),
        answers=answers,
    )


def test_system_one_level_0_skips_evaluator_preserves_seeds(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed", "source:repair_pressure"],
        focal_node_refs=["gev_1"],
        signal_strength=0.85,
        confidence=0.7,
    )
    worker._store.load_latest_system_one_appraisal.return_value = _curiosity_frame("0")

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        worker._endogenous_curiosity_tick()

    evaluator_cls.assert_not_called()
    worker._store.save_endogenous_curiosity_candidates.assert_called_once()
    args, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
    assert args[0] == [seed]
    assert kwargs["gate"]["gate_result"] == "system_one_curiosity_noop"
    assert kwargs["gate"]["selected_level"] == "0"
    assert kwargs["gate"]["frame_id"] == "frame-level-0"
    assert kwargs["require_gate_lineage"] is True
    # Authority boundary: level-0 still persists endogenous seeds for Hub
    # readers (curiosity_hint / endogenous_outreach); it only skips the evaluator.
    assert seed in args[0]


def test_system_one_level_0_without_lineage_fails_open_to_evaluator(monkeypatch):
    """Causal veto requires durable gate_json; missing lineage → legacy admit."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    decision = SimpleNamespace(
        outcome="invoke",
        chosen_task_type="evidence_gap_scan",
        decision_id="dec-legacy",
        bounded_context_reason="invoke based on curiosity_candidate",
    )
    run_result = SimpleNamespace(signals=[seed], decision=decision)
    worker._store.load_latest_system_one_appraisal.return_value = _curiosity_frame("0")
    worker._store.save_endogenous_curiosity_candidates.return_value = (
        EndogenousCuriosityPersistResult(
            candidate_set_id="curiosity-legacy",
            gate_lineage_persisted=False,
        )
    )

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()

    evaluator_cls.return_value.evaluate.assert_called_once()
    # First persist attempted the noop with require_gate_lineage; second is admit path.
    assert worker._store.save_endogenous_curiosity_candidates.call_count >= 2
    first_kwargs = worker._store.save_endogenous_curiosity_candidates.call_args_list[0].kwargs
    assert first_kwargs["require_gate_lineage"] is True
    last_kwargs = worker._store.save_endogenous_curiosity_candidates.call_args_list[-1].kwargs
    assert last_kwargs["gate"]["fallback_reason"] == "lineage_store_unavailable"
    assert last_kwargs["gate"]["gate_result"] == "system_one_unavailable_fallback"


def test_system_one_level_1_and_2_admit_evaluator(monkeypatch):
    for level in ("1", "2"):
        worker = _make_worker(monkeypatch, enabled=True)
        fake_store = MagicMock()
        fake_store.snapshot.return_value = SimpleNamespace(nodes={})
        seed = SimpleNamespace(
            signal_type="curiosity_candidate",
            notes=["endogenous_seed"],
            signal_strength=0.85,
            confidence=0.7,
        )
        decision = SimpleNamespace(
            outcome="invoke",
            chosen_task_type="evidence_gap_scan",
            decision_id="dec-1",
            bounded_context_reason="invoke based on curiosity_candidate",
        )
        run_result = SimpleNamespace(signals=[seed], decision=decision)
        worker._store.load_latest_system_one_appraisal.return_value = _curiosity_frame(level)

        with patch(
            "orion.substrate.graphdb_store.build_substrate_store_from_env",
            return_value=fake_store,
        ), patch(
            "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
            return_value=[seed],
        ), patch(
            "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
        ) as evaluator_cls:
            evaluator_cls.return_value.evaluate.return_value = run_result
            worker._endogenous_curiosity_tick()

        evaluator_cls.return_value.evaluate.assert_called_once()
        _, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
        assert kwargs["gate"]["gate_result"] == "system_one_curiosity_admit"
        assert kwargs["gate"]["selected_level"] == level
        assert kwargs["gate"]["evaluator_outcome"] == "invoke"


def test_system_one_missing_frame_falls_back_to_legacy(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    decision = SimpleNamespace(outcome="noop", chosen_task_type=None, decision_id=None)
    run_result = SimpleNamespace(signals=[seed], decision=decision)
    worker._store.load_latest_system_one_appraisal.return_value = None

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()

    evaluator_cls.return_value.evaluate.assert_called_once()
    _, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
    assert kwargs["gate"]["gate_result"] == "system_one_unavailable_fallback"
    assert kwargs["gate"]["fallback_reason"] == "no_frame"


def test_system_one_curiosity_gate_kill_switch_falls_back(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_SYSTEM_ONE_CURIOSITY_GATE_KILL_SWITCH", "true")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = _make_worker(monkeypatch, enabled=True)
    worker._settings = settings_mod.get_settings()
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    seed = SimpleNamespace(
        signal_type="curiosity_candidate",
        notes=["endogenous_seed"],
        signal_strength=0.85,
        confidence=0.7,
    )
    decision = SimpleNamespace(outcome="invoke", chosen_task_type="ontology_expand", decision_id="d")
    run_result = SimpleNamespace(signals=[seed], decision=decision)
    # Even with a level-0 frame, kill switch must admit evaluator.
    worker._store.load_latest_system_one_appraisal.return_value = _curiosity_frame("0")

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[seed],
    ), patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        evaluator_cls.return_value.evaluate.return_value = run_result
        worker._endogenous_curiosity_tick()

    evaluator_cls.return_value.evaluate.assert_called_once()
    _, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
    assert kwargs["gate"]["fallback_reason"] == "consumer_kill_switch"


def test_system_one_cannot_mint_candidates(monkeypatch):
    """Gate only sees seeds from endogenous_curiosity_candidates."""
    worker = _make_worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes={})
    worker._store.load_latest_system_one_appraisal.return_value = _curiosity_frame("2")

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch(
        "orion.substrate.endogenous_curiosity.endogenous_curiosity_candidates",
        return_value=[],
    ) as seeds_fn, patch(
        "orion.substrate.frontier_curiosity.FrontierCuriosityEvaluator"
    ) as evaluator_cls:
        worker._endogenous_curiosity_tick()

    seeds_fn.assert_called_once()
    evaluator_cls.assert_not_called()
    worker._store.save_endogenous_curiosity_candidates.assert_called_once()
    args, kwargs = worker._store.save_endogenous_curiosity_candidates.call_args
    assert args == ([],)
    assert "retention_hours" in kwargs
