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
    store.save_endogenous_curiosity_candidates([_signal("sig-1"), _signal("sig-2")])

    assert conn.execute.call_count == 2  # insert + prune
    insert_params = conn.execute.call_args_list[0].args[1]
    assert insert_params["candidate_set_id"].startswith("curiosity-")
    candidates = insert_params["candidates_json"].adapted
    assert [c["signal_id"] for c in candidates] == ["sig-1", "sig-2"]
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_endogenous_curiosity_candidates" in prune_sql


def test_save_curiosity_candidates_empty_persists_heartbeat():
    store, conn = _store_with_conn()
    store.save_endogenous_curiosity_candidates([])

    assert conn.execute.call_count == 2  # insert + prune
    insert_params = conn.execute.call_args_list[0].args[1]
    assert insert_params["candidate_set_id"].startswith("curiosity-")
    assert insert_params["candidates_json"].adapted == []
    prune_sql = str(conn.execute.call_args_list[1].args[0])
    assert "DELETE FROM substrate_endogenous_curiosity_candidates" in prune_sql


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
