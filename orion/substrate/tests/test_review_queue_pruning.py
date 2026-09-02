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


def test_reseed_after_suppression_preserves_the_prune_clock() -> None:
    """The live 2026-09-02 failure: a re-seed reset the clock the prune reads.

    Ordering is the whole point. ``test_prune_finished_then_reseed_recovers_the_queue``
    prunes and *then* re-seeds; the scheduler does the reverse -- it re-seeds
    whenever nothing is usable, which is exactly when every item is suppressed --
    so the re-seed lands first and the prune runs against whatever it left behind.
    Suppression is driven through the real ``mark_reviewed``/``apply_cycle_feedback``
    path rather than injected, because the bug was that a re-seed discarded the
    history those two calls had recorded.
    """
    queue = GraphReviewQueue(max_items=20)
    seeded_at = datetime.now(timezone.utc) - timedelta(days=2)

    queue.upsert(_item(node="a", created_at=seeded_at, next_review_at=seeded_at))
    item_id = queue.snapshot(limit=50).queue_items[0].queue_item_id
    queue.mark_reviewed(item_id, reviewed_at=seeded_at)
    queue.apply_cycle_feedback(item_id, no_change=True)
    queue.apply_cycle_feedback(item_id, no_change=True)
    assert queue.usable_items(limit=50) == []  # suppressed, as the scheduler finds it

    queue.upsert(_item(node="a"))  # the tick's bootstrap re-seeds the same region

    revived = queue.snapshot(limit=50).queue_items[0]
    assert revived.created_at == seeded_at
    assert revived.last_review_at == seeded_at
    assert queue.prune_finished(older_than_sec=21600.0) == 1
    assert queue.snapshot(limit=50).queue_items == []


def test_reseed_still_refreshes_the_incoming_proposal_details() -> None:
    """Preserving lifecycle must not freeze the item against a newer proposal."""
    queue = GraphReviewQueue(max_items=20)
    queue.upsert(_item(node="a"))
    original_id = queue.snapshot(limit=50).queue_items[0].queue_item_id

    fresh = _item(node="a").model_copy(
        update={
            "priority": 91,
            "reason_for_revisit": "raised again",
            "notes": ["second"],
            "focal_edge_refs": ["e-new"],
            "originating_decision_id": "d-second",
            "originating_request_id": "r-second",
        }
    )
    queue.upsert(fresh)

    merged = queue.snapshot(limit=50).queue_items[0]
    assert merged.queue_item_id == original_id
    assert merged.priority == 91
    assert merged.reason_for_revisit == "raised again"
    assert merged.notes == ["second"]
    assert merged.focal_edge_refs == ["e-new"]
    assert merged.originating_decision_id == "d-second"
    assert merged.originating_request_id == "r-second"


def test_reseed_preserves_created_at_for_an_item_never_reviewed() -> None:
    """The live row shape: suppressed with ``last_review_at`` still None.

    All eight rows stuck in production on 2026-09-02 had ``last_review_at``
    NULL, so ``created_at`` was the only clock they had. The companion test
    above ages an item whose two timestamps are both set, which means its prune
    assertion passes if *either* survives the merge. This one removes that
    redundancy: refresh ``created_at`` and nothing else, and the prune goes back
    to returning 0.
    """
    queue = GraphReviewQueue(max_items=20)
    seeded_at = datetime.now(timezone.utc) - timedelta(days=2)

    queue.upsert(
        _item(node="a", suppressed=True, last_review_at=None, created_at=seeded_at)
    )
    queue.upsert(_item(node="a"))  # the tick's bootstrap re-seeds the same region

    revived = queue.snapshot(limit=50).queue_items[0]
    assert revived.last_review_at is None
    assert revived.created_at == seeded_at
    assert queue.prune_finished(older_than_sec=21600.0) == 1


def test_eviction_drops_a_dead_item_before_a_live_one() -> None:
    """``created_at`` no longer resets, so age alone must not decide liveness.

    Before the merge fix a re-seeded item looked young and was shielded from the
    age tiebreak. With the clock preserved, the oldest item wins that tiebreak --
    which would evict a long-surviving active region in favour of a younger
    already-suppressed one that ``prune_finished`` is about to reap anyway.
    """
    queue = GraphReviewQueue(max_items=2)
    old = datetime.now(timezone.utc) - timedelta(days=7)
    queue.upsert(_item(node="live-and-old", created_at=old))
    queue.upsert(_item(node="dead-but-young", suppressed=True))

    queue.upsert(_item(node="forces-eviction"))

    survivors = {i.focal_node_refs[0] for i in queue.snapshot(limit=50).queue_items}
    assert survivors == {"live-and-old", "forces-eviction"}


def test_reseed_refresh_list_partitions_every_schema_field() -> None:
    """Adding a field to the schema must force an explicit refresh/preserve call.

    The original merge listed what to *preserve*, so a new field silently
    defaulted to being overwritten by the re-seed -- which is how ``created_at``
    and ``last_review_at`` came to be wiped. Listing what to *refresh* makes the
    safe direction the default; this pins the list against the schema so the
    choice cannot be skipped by omission.
    """
    from orion.substrate.review_queue import _RESEED_REFRESHED_FIELDS

    refreshed = set(_RESEED_REFRESHED_FIELDS)
    preserved = {
        "queue_item_id",
        "cycle_budget",
        "suppression_state",
        "termination_state",
        "last_review_at",
        "created_at",
    }

    assert refreshed.isdisjoint(preserved)
    assert refreshed | preserved == set(GraphReviewQueueItemV1.model_fields)
