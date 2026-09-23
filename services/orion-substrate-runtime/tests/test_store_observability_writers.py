"""Unit tests for the self-observability store writers (curiosity + dwell)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1

from app.store import BiometricsSubstrateStore


def _store_with_conn() -> tuple[BiometricsSubstrateStore, MagicMock]:
    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    store._engine = engine
    return store, conn


def _signal(signal_id: str, strength: float = 0.8) -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_id=signal_id,
        signal_type="ontology_sparse_region",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="world_ontology",
        task_type_candidate="ontology_expand",
        signal_strength=strength,
        confidence=0.7,
        evidence_summary="gap",
    )


def _projection() -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        generated_at=datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc),
        frame=AttentionFrameV1(),
        attended_node_ids=["node:a", "node:b"],
        dwell_ticks=4,
        coalition_stability_score=0.9,
    )


def test_save_curiosity_candidates_inserts_json_array_and_prunes():
    store, conn = _store_with_conn()
    result = store.save_endogenous_curiosity_candidates(
        [_signal("sig-1"), _signal("sig-2")],
        retention_hours=720.0,
    )

    assert result.candidate_set_id.startswith("curiosity-")
    assert result.gate_lineage_persisted is True
    assert conn.execute.call_count == 2  # insert + prune
    insert_params = conn.execute.call_args_list[0].args[1]
    assert insert_params["candidate_set_id"].startswith("curiosity-")
    candidates = insert_params["candidates_json"].adapted
    assert [c["signal_id"] for c in candidates] == ["sig-1", "sig-2"]
    assert insert_params["gate_json"] is None
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_endogenous_curiosity_candidates" in prune_sql
    assert "720.0 hours" in prune_sql


def test_save_curiosity_candidates_empty_persists_heartbeat():
    store, conn = _store_with_conn()
    result = store.save_endogenous_curiosity_candidates([], retention_hours=48.0)

    assert result.gate_lineage_persisted is True
    assert conn.execute.call_count == 2  # insert + prune
    insert_params = conn.execute.call_args_list[0].args[1]
    assert insert_params["candidate_set_id"].startswith("curiosity-")
    assert insert_params["candidates_json"].adapted == []
    assert insert_params["gate_json"] is None
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_endogenous_curiosity_candidates" in prune_sql
    assert "48.0 hours" in prune_sql


def test_save_curiosity_candidates_persists_gate_json():
    store, conn = _store_with_conn()
    gate = {
        "gate_result": "system_one_curiosity_admit",
        "selected_level": "1",
        "frame_id": "frame-1",
        "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
    }
    result = store.save_endogenous_curiosity_candidates(
        [_signal("sig-1")],
        gate=gate,
        retention_hours=720.0,
    )

    assert result.candidate_set_id.startswith("curiosity-")
    assert result.gate_lineage_persisted is True
    insert_params = conn.execute.call_args_list[0].args[1]
    assert insert_params["gate_json"].adapted["gate_result"] == "system_one_curiosity_admit"
    assert insert_params["gate_json"].adapted["candidate_set_id"] == result.candidate_set_id
    assert "gate_json" in str(conn.execute.call_args_list[0].args[0])


def test_save_curiosity_candidates_reports_lineage_unavailable_on_missing_column():
    store, conn = _store_with_conn()

    def execute(stmt, params=None):
        sql = str(stmt)
        if "gate_json" in sql and "INSERT" in sql:
            raise RuntimeError('column "gate_json" of relation does not exist')
        return MagicMock()

    conn.execute.side_effect = execute
    result = store.save_endogenous_curiosity_candidates(
        [_signal("sig-1")],
        gate={"gate_result": "system_one_curiosity_noop", "selected_level": "0"},
        require_gate_lineage=True,
        retention_hours=720.0,
    )
    assert result.gate_lineage_persisted is False
    assert result.candidate_set_id.startswith("curiosity-")


def test_save_coalition_dwell_row_shape_and_prune():
    store, conn = _store_with_conn()
    store.save_coalition_dwell(_projection())

    assert conn.execute.call_count == 2  # insert + prune
    params = conn.execute.call_args_list[0].args[1]
    assert params["dwell_id"].startswith("dwell-")
    assert params["coalition_ids"].adapted == ["node:a", "node:b"]
    assert params["dwell_ticks"] == 4
    assert params["active"] is True
    assert params["salience_trend"] == 0.9
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_coalition_dwell_log" in prune_sql


def test_save_coalition_dwell_inactive_when_zero_ticks():
    store, conn = _store_with_conn()
    projection = _projection().model_copy(update={"dwell_ticks": 0})
    store.save_coalition_dwell(projection)
    params = conn.execute.call_args_list[0].args[1]
    assert params["active"] is False


def test_save_attention_broadcast_history_row_shape_and_prune():
    store, conn = _store_with_conn()
    store.save_attention_broadcast_history(_projection(), retention_hours=168.0)

    assert conn.execute.call_count == 2  # insert + prune
    insert_sql = str(conn.execute.call_args_list[0].args[0])
    assert "INSERT INTO substrate_attention_broadcast_log" in insert_sql
    assert "ON CONFLICT (log_id) DO NOTHING" in insert_sql
    params = conn.execute.call_args_list[0].args[1]
    assert params["log_id"].startswith("broadcast-")
    assert params["generated_at"] == _projection().generated_at
    assert params["projection_json"].adapted["projection_id"] == _projection().projection_id
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_attention_broadcast_log" in prune_sql
    assert "168.0 hours" in prune_sql


def test_save_attention_broadcast_history_idempotent_digest():
    """Same generated_at + projection_id always produces the same log_id, so a
    re-delivered event is a no-op via ON CONFLICT DO NOTHING rather than a
    duplicate row."""
    store, conn = _store_with_conn()
    store.save_attention_broadcast_history(_projection(), retention_hours=168.0)
    first_id = conn.execute.call_args_list[0].args[1]["log_id"]

    store2, conn2 = _store_with_conn()
    store2.save_attention_broadcast_history(_projection(), retention_hours=168.0)
    second_id = conn2.execute.call_args_list[0].args[1]["log_id"]

    assert first_id == second_id
