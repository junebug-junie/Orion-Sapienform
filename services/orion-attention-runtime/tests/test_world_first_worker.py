"""World-first wiring in the attention-runtime worker (spec 2026-10-07 section A).

The worker has no magnitudes of its own: it reads the substrate's stored
prediction-error history and Juniper's chat turns, and every source it cannot
read becomes an ABSENT candidate -- never calm, never a fallback to the old
ranking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.schemas.field_state import FieldStateV1

from app.store import AttentionRuntimeStore
from app.worker import AttentionRuntimeWorker, _camera_absent_reason

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _history(node_id: str, calm: float, current: float, n: int = 400):
    rows = [(node_id, NOW - timedelta(minutes=10 * (n - i)), calm) for i in range(n)]
    rows.append((node_id, NOW - timedelta(seconds=20), current))
    return rows


def _worker(*, pe_rows=None, pe_error=None, turns=None, chat_error=None) -> AttentionRuntimeWorker:
    w = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    w._store = MagicMock(spec=AttentionRuntimeStore)
    w._pe_history = NodePeHistoryCache()
    w._settings = SimpleNamespace(attention_world_first_enabled=True)
    if pe_error is not None:
        w._store.fetch_node_prediction_error_history.side_effect = pe_error
    else:
        w._store.fetch_node_prediction_error_history.side_effect = lambda since: [
            r for r in (pe_rows or []) if r[1] >= since
        ]
    if chat_error is not None:
        w._store.fetch_chat_turn_times.side_effect = chat_error
    else:
        w._store.fetch_chat_turn_times.return_value = turns or []
    return w


def _field(staleness: float | None = 0.0) -> FieldStateV1:
    vectors = {"node:substrate.execution": {"prediction_error": 0.9}}
    if staleness is not None:
        vectors["node:substrate.vision_organ"] = {"vision_frame_staleness": staleness}
    return FieldStateV1(tick_id="t", generated_at=NOW, node_vectors=vectors)


def _by_id(cands):
    return {c.source_id: c for c in cands}


def test_candidates_cover_every_stored_node_plus_chat() -> None:
    rows = _history("node:substrate.execution", 0.0, 0.9) + _history("node:substrate.biometrics", 0.02, 0.02)
    cands = _by_id(_worker(pe_rows=rows)._world_first_candidates(_field(), NOW))
    assert set(cands) == {"node:substrate.execution", "node:substrate.biometrics", "world:chat"}
    ex = cands["node:substrate.execution"]
    assert ex.source_kind == "internal" and ex.unusualness.band == "unusual"
    assert cands["node:substrate.biometrics"].unusualness.percentile_now == 0.0


def test_unreadable_history_makes_body_nodes_absent_not_calm() -> None:
    cands = _by_id(_worker(pe_error=RuntimeError("db down"))._world_first_candidates(_field(), NOW))
    ex = cands["node:substrate.execution"]
    assert ex.absent and "unreadable" in ex.absent_reason


def test_unreadable_chat_log_is_absent() -> None:
    cands = _by_id(_worker(chat_error=RuntimeError("db down"))._world_first_candidates(_field(), NOW))
    assert cands["world:chat"].absent


def test_camera_absence_comes_from_frame_staleness() -> None:
    assert _camera_absent_reason(_field(0.0)) is None
    assert "stale" in _camera_absent_reason(_field(1.0))
    assert "unmeasured" in _camera_absent_reason(_field(None))
    rows = _history("node:substrate.perception", 0.0, 0.0)
    cands = _by_id(_worker(pe_rows=rows)._world_first_candidates(_field(1.0), NOW))
    p = cands["node:substrate.perception"]
    assert p.source_kind == "external" and p.absent


def test_tick_saves_a_world_first_frame_and_logs_no_winner(monkeypatch) -> None:
    from pathlib import Path

    from orion.attention.field_attention.policy import load_attention_policy

    w = _worker(pe_rows=_history("node:substrate.execution", 0.0, 0.0))
    w._settings = SimpleNamespace(
        attention_world_first_enabled=True, enable_attention_runtime=True,
        prediction_error_history_limit=200, enable_goal_provenance_producer=False,
    )
    w._policy = load_attention_policy(
        Path(__file__).resolve().parents[3] / "config" / "attention" / "field_attention_policy.v1.yaml"
    )
    w._bus = None
    w._last_field = None
    w._node_streak = None
    w._store.load_latest_field.return_value = _field()
    w._store.load_attention_frame_for_field_tick.return_value = None
    w._store.load_latest_attention_frame.return_value = None
    w._store.advance_node_prediction_error_baseline.return_value = MagicMock(
        observation_count=0, ewma=0.0, variance=0.0, last_value=None, last_observed_at=None
    )
    w._tick()
    frame = w._store.save_attention_frame.call_args[0][0]
    assert frame.dominant_targets == [] and frame.warnings == ["world_first_no_winner"]
