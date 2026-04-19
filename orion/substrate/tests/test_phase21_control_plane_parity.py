from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_review_queue import GraphReviewCycleBudgetV1, GraphReviewQueueItemV1
from orion.substrate.review_queue import GraphReviewQueue


def _queue_item() -> GraphReviewQueueItemV1:
    return GraphReviewQueueItemV1(
        focal_node_refs=["n1"],
        focal_edge_refs=["e1"],
        anchor_scope="orion",
        subject_ref="orion",
        target_zone="concept_graph",
        originating_decision_id="d1",
        originating_request_id="r1",
        reason_for_revisit="phase21",
        priority=80,
        next_review_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        cycle_budget=GraphReviewCycleBudgetV1(cycle_count=0, max_cycles=3, remaining_cycles=3, no_change_cycles=0),
    )


def test_queue_sqlite_persistence_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite3"
    queue = GraphReviewQueue(sql_db_path=str(db_path), max_items=20)
    queue.upsert(_queue_item())

    reloaded = GraphReviewQueue(sql_db_path=str(db_path), max_items=20)
    snap = reloaded.snapshot(limit=10)
    assert len(snap.queue_items) == 1
    assert reloaded.source_kind() == "sqlite"


def test_queue_postgres_preferred_with_explicit_fallback(tmp_path) -> None:
    queue = GraphReviewQueue(
        postgres_url="postgresql://invalid:invalid@localhost:1/nope",
        sql_db_path=str(tmp_path / "queue.sqlite3"),
        max_items=20,
    )
    queue.upsert(_queue_item())
    assert queue.degraded() is True
    assert queue.last_error()
