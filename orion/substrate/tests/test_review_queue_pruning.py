"""The review queue's absorbing state, and the guard that really keeps a zone out.

Both facts here are load-bearing for the unattended review tick in
``api_routes.execute_substrate_review_scheduled_cycle`` and neither can be
covered with a fake: the first is a property of ``GraphReviewQueue``'s own
suppression/upsert interaction, and the second lives in ``GraphReviewScheduler``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_consolidation import (
    GraphConsolidationDecisionV1,
    GraphConsolidationResultV1,
)
from orion.core.schemas.substrate_review_queue import (
    GraphReviewCycleBudgetV1,
    GraphReviewQueueItemV1,
)
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_schedule import GraphReviewScheduler


def _item(
    *,
    node: str,
    zone: str = "concept_graph",
    suppressed: bool = False,
    terminated: bool = False,
    remaining: int = 3,
    next_review_at: datetime | None = None,
    last_review_at: datetime | None = None,
    created_at: datetime | None = None,
) -> GraphReviewQueueItemV1:
    now = datetime.now(timezone.utc)
    return GraphReviewQueueItemV1(
        focal_node_refs=[node],
        anchor_scope="orion",
        subject_ref="orion",
        target_zone=zone,
        originating_decision_id=f"d-{node}",
        originating_request_id=f"r-{node}",
        reason_for_revisit="test",
        priority=50,
        next_review_at=next_review_at or (now - timedelta(seconds=1)),
        cycle_budget=GraphReviewCycleBudgetV1(
            cycle_count=0, max_cycles=3, remaining_cycles=remaining, no_change_cycles=0
        ),
        suppression_state=suppressed,
        termination_state=terminated,
        last_review_at=last_review_at,
        created_at=created_at or now,
    )


def test_suppressed_queue_is_non_empty_but_undrainable() -> None:
    """The exact shape that made an emptiness-gated reseed an absorbing state."""
    queue = GraphReviewQueue(max_items=20)
    old = datetime.now(timezone.utc) - timedelta(days=1)
    for n in ("a", "b", "c"):
        queue.upsert(_item(node=n, suppressed=True, last_review_at=old))

    assert len(queue.snapshot(limit=50).queue_items) == 3  # not empty
    assert queue.list_eligible(limit=50) == []  # nothing due
    assert queue.usable_items(limit=50) == []  # and nothing can ever be due


def test_upsert_resurrects_suppression_so_reseeding_alone_cannot_recover() -> None:
    """Why pruning is required rather than just re-running bootstrap.

    ``upsert`` matches on the region key and copies ``suppression_state``
    forward, so seeding the same region again returns a still-suppressed item.
    """
    queue = GraphReviewQueue(max_items=20)
    old = datetime.now(timezone.utc) - timedelta(days=1)
    queue.upsert(_item(node="a", suppressed=True, last_review_at=old))

    queue.upsert(_item(node="a", suppressed=False))  # a fresh bootstrap seed

    assert queue.usable_items(limit=50) == []
    assert queue.snapshot(limit=50).queue_items[0].suppression_state is True


def test_prune_finished_then_reseed_recovers_the_queue() -> None:
    queue = GraphReviewQueue(max_items=20)
    old = datetime.now(timezone.utc) - timedelta(days=1)
    for n in ("a", "b", "c"):
        queue.upsert(_item(node=n, suppressed=True, last_review_at=old))

    assert queue.prune_finished(older_than_sec=3600.0) == 3
    assert queue.snapshot(limit=50).queue_items == []

    queue.upsert(_item(node="a"))
    assert len(queue.usable_items(limit=50)) == 1


def test_prune_respects_the_cutoff_so_it_is_not_a_churn_loop() -> None:
    """A just-suppressed item must rest; pruning it immediately would reseed it next tick."""
    queue = GraphReviewQueue(max_items=20)
    queue.upsert(
        _item(node="a", suppressed=True, last_review_at=datetime.now(timezone.utc))
    )

    assert queue.prune_finished(older_than_sec=21600.0) == 0
    assert len(queue.snapshot(limit=50).queue_items) == 1


def test_prune_never_touches_an_active_item() -> None:
    queue = GraphReviewQueue(max_items=20)
    old = datetime.now(timezone.utc) - timedelta(days=7)
    queue.upsert(_item(node="live", last_review_at=old))
    queue.upsert(_item(node="dead", suppressed=True, last_review_at=old))

    assert queue.prune_finished(older_than_sec=1.0) == 1
    remaining = queue.snapshot(limit=50).queue_items
    assert [i.focal_node_refs[0] for i in remaining] == ["live"]


def test_prune_ages_an_unreviewed_item_from_created_at() -> None:
    """last_review_at is None for an item suppressed before it was ever reviewed."""
    queue = GraphReviewQueue(max_items=20)
    old = datetime.now(timezone.utc) - timedelta(days=1)
    queue.upsert(_item(node="a", terminated=True, last_review_at=None, created_at=old))

    assert queue.prune_finished(older_than_sec=3600.0) == 1


def test_usable_items_excludes_an_exhausted_budget() -> None:
    queue = GraphReviewQueue(max_items=20)
    queue.upsert(_item(node="a", remaining=0))

    assert len(queue.snapshot(limit=50).queue_items) == 1
    assert queue.usable_items(limit=50) == []


def test_self_relationship_zone_never_enters_the_queue() -> None:
    """The guard that actually contains that zone -- review_schedule.py:84.

    The unattended tick runs under ``invocation_surface="operator_review"``,
    which SATISFIES the zone gate in ``review_runtime._select_item``. Nothing
    downstream would stop the loop selecting such an item; what protects it is
    that the scheduler refuses to enqueue one at all, and this scheduler is the
    only ``queue.upsert`` caller in the repo.
    """
    queue = GraphReviewQueue(max_items=20)
    scheduler = GraphReviewScheduler(queue=queue)

    result = scheduler.apply_consolidation_result(
        consolidation_result=GraphConsolidationResultV1(
            request_id="r-self",
            decisions=[
                GraphConsolidationDecisionV1(
                    target_refs=["node-self"],
                    outcome="requeue_review",
                    reason="self relationship drift",
                    confidence=0.9,
                    zone="self_relationship_graph",
                    priority=90,
                )
            ],
            outcome_counts={"requeue_review": 1},
            regions_reviewed=["self"],
            unresolved_regions=[],
            confidence=0.9,
        ),
        anchor_scope="orion",
        subject_ref="orion",
        invocation_surface="operator_review",
    )

    assert result.enqueued_items == []
    assert queue.snapshot(limit=50).queue_items == []
    assert result.schedule_decisions[0].outcome == "operator_only"
    assert "strict_zone_guardrail" in result.schedule_decisions[0].notes
