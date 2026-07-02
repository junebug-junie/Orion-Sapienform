from __future__ import annotations

import json
from datetime import datetime, timezone

from orion.substrate.relational.adapters.episodes_ctx import map_episode_ctx_to_substrate


def _episode(**overrides):
    payload = {
        "schema_version": "substrate.episode_summary.v1",
        "episode_id": "ep_abc123",
        "status": "proposal",
        "window_start": "2026-07-02T10:00:00+00:00",
        "window_end": "2026-07-02T10:15:00+00:00",
        "window_seconds": 900,
        "receipt_refs": ["r1", "r2"],
        "receipt_count_total": 42,
        "organ_counts": {"biometrics_pressure": 30, "execution": 8, "transport": 4},
        "warning_count": 1,
        "notes": ["transport backlog spike"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(overrides)
    return payload


def test_maps_episode_to_single_proposal_marked_node():
    record = map_episode_ctx_to_substrate({"episode_summary": _episode()})
    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "episode:latest"
    assert node.metadata["status"] == "proposal"
    assert node.metadata["episode_id"] == "ep_abc123"
    assert node.metadata["receipt_count_total"] == 42
    assert node.metadata["window_end"] == "2026-07-02T10:15:00+00:00"


def test_accepts_json_string_payload():
    record = map_episode_ctx_to_substrate({"episode_summary": json.dumps(_episode())})
    assert record is not None
    assert record.nodes[0].metadata["episode_id"] == "ep_abc123"


def test_salience_scales_with_receipt_count_and_is_clamped():
    quiet = map_episode_ctx_to_substrate(
        {"episode_summary": _episode(receipt_count_total=0)}
    )
    busy = map_episode_ctx_to_substrate(
        {"episode_summary": _episode(receipt_count_total=500)}
    )
    assert quiet is not None and busy is not None
    assert quiet.nodes[0].signals.salience < busy.nodes[0].signals.salience
    assert busy.nodes[0].signals.salience <= 1.0


def test_returns_none_on_missing_or_garbage_ctx():
    assert map_episode_ctx_to_substrate({}) is None
    assert map_episode_ctx_to_substrate({"episode_summary": "not json"}) is None
    assert map_episode_ctx_to_substrate(None) is None
