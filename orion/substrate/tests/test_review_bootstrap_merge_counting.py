"""``items_merged`` -- telling a re-seed of a known region apart from seeding nothing.

``items_enqueued`` counts newly minted ``queue_item_id``s. A seed landing on an
already-queued region key merges into the existing item and mints none, so once
the queue has warmed up the field reads 0 forever and is indistinguishable from
a bootstrap that found nothing at all. Live on 2026-09-02 every scheduler tick
logged ``bootstrapped: true, items_enqueued: 0`` for hours while it was in fact
re-seeding -- and resurrecting -- eight suppressed items each time; separating
the two required a Postgres query rather than a log line.

Only the semantic store is stubbed here. The bootstrapper, scheduler and queue
are the real classes, because the count is a property of how they interact.
"""
from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.review_bootstrap import GraphReviewBootstrapper
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.store import SubstrateNeighborhoodSliceV1, SubstrateQueryResultV1


class _FixedRegionStore:
    """Returns the same one-node region for every query, as a settled graph does."""

    def __init__(self) -> None:
        self.node = ConceptNodeV1(
            node_id="node-fixed",
            label="fixed",
            anchor_scope="orion",
            subject_ref="orion",
            temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
            provenance=SubstrateProvenanceV1(
                authority="local_inferred",
                source_kind="test",
                source_channel="test",
                producer="test",
            ),
            metadata={"dynamic_pressure": 0.9, "resolved": False},
        )

    def _result(self, query_kind: str) -> SubstrateQueryResultV1:
        return SubstrateQueryResultV1(
            query_kind=query_kind,
            slice=SubstrateNeighborhoodSliceV1(nodes=[self.node], edges=[]),
            source_kind="cache",
        )

    def query_contradiction_region(self, **_: object) -> SubstrateQueryResultV1:
        return self._result("contradiction_region")

    def query_hotspot_region(self, **_: object) -> SubstrateQueryResultV1:
        return self._result("hotspot_region")

    def query_concept_region(self, **_: object) -> SubstrateQueryResultV1:
        return self._result("concept_region")


def _bootstrapper() -> tuple[GraphReviewBootstrapper, GraphReviewQueue]:
    queue = GraphReviewQueue(max_items=50)
    scheduler = GraphReviewScheduler(queue=queue)
    return GraphReviewBootstrapper(scheduler=scheduler, semantic_store=_FixedRegionStore()), queue


def test_first_bootstrap_reports_enqueues_and_no_merges() -> None:
    bootstrapper, queue = _bootstrapper()

    execution = bootstrapper.bootstrap(now=datetime.now(timezone.utc))

    assert execution.items_enqueued > 0
    assert execution.items_merged == 0
    assert len(queue.snapshot(limit=50).queue_items) == execution.items_enqueued


def test_reseeding_a_known_region_reports_merges_not_enqueues() -> None:
    """The live steady state: the queue stops growing but work is still happening."""
    bootstrapper, queue = _bootstrapper()
    first = bootstrapper.bootstrap(now=datetime.now(timezone.utc))
    size_after_first = len(queue.snapshot(limit=50).queue_items)

    second = bootstrapper.bootstrap(now=datetime.now(timezone.utc))

    assert second.items_enqueued == 0  # what the old log line showed, alone
    assert second.items_merged == first.items_enqueued  # what it could not show
    assert len(queue.snapshot(limit=50).queue_items) == size_after_first
