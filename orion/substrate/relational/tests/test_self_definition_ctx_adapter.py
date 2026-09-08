from __future__ import annotations

from datetime import datetime, timezone

from orion.substrate.relational.adapters.self_definition_ctx import (
    CTX_KEY,
    NODE_ID,
    SNAPSHOT_SOURCE,
    map_self_definition_ctx_to_substrate,
)


def _payload(**over):
    base = {
        "entry_id": "e1",
        "content": "I am a mesh of services.",
        "version": 2,
        "evidence_refs": ["README.md", "dreams: 17 rows"],
        "created_at": "2026-09-08T04:00:00+00:00",
    }
    base.update(over)
    return base


def test_maps_one_snapshot_anchored_to_orion() -> None:
    record = map_self_definition_ctx_to_substrate({CTX_KEY: _payload()})
    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.node_id == NODE_ID
    assert node.snapshot_source == SNAPSHOT_SOURCE
    assert node.metadata["content"] == "I am a mesh of services."
    assert node.metadata["version"] == 2
    assert node.metadata["evidence_refs"] == ["README.md", "dreams: 17 rows"]
    assert node.dimensions["evidence_count"] == 2.0
    assert node.provenance.tier_rank == 2
    assert node.temporal.observed_at == datetime(2026, 9, 8, 4, 0, tzinfo=timezone.utc)


def test_none_for_missing_or_blank_content() -> None:
    assert map_self_definition_ctx_to_substrate({}) is None
    assert map_self_definition_ctx_to_substrate({CTX_KEY: None}) is None
    assert map_self_definition_ctx_to_substrate({CTX_KEY: "not a dict"}) is None
    assert map_self_definition_ctx_to_substrate({CTX_KEY: _payload(content="  ")}) is None


def test_tolerates_bad_version_and_timestamp() -> None:
    record = map_self_definition_ctx_to_substrate({CTX_KEY: _payload(version="x", created_at="garbage")})
    assert record is not None
    assert record.nodes[0].metadata["version"] == 1
    assert record.nodes[0].temporal.observed_at.tzinfo is not None


def test_datetime_created_at_is_accepted() -> None:
    when = datetime(2026, 9, 1, tzinfo=timezone.utc)
    record = map_self_definition_ctx_to_substrate({CTX_KEY: _payload(created_at=when)})
    assert record is not None
    assert record.nodes[0].temporal.observed_at == when
