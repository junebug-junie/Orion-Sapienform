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
