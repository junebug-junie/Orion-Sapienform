"""Regression cover for the substrate mutation pipeline's five-week starvation.

Live finding 2026-08-30: `substrate_review_telemetry` held 1,358 rows and the
mutation scheduler consumed 0 of them, every 30 seconds, for five weeks, because
its two filter conditions were each individually satisfiable and never jointly:

    operator_review      / concept_graph   1,356 rows  <- wrong zone
    chat_reflective_lane / autonomy_graph      2 rows  <- wrong surface

`query()` reported that as an ordinary empty list. These tests pin the seam that
now tells the two apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.substrate_review_telemetry import (
    GraphReviewTelemetryQueryV1,
    GraphReviewTelemetryRecordV1,
)
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder


def _record(*, surface: str, zone: str, at: datetime) -> GraphReviewTelemetryRecordV1:
    return GraphReviewTelemetryRecordV1(
        invocation_surface=surface,  # type: ignore[arg-type]
        target_zone=zone,  # type: ignore[arg-type]
        anchor_scope="orion",
        subject_ref="entity:orion",
        selection_reason="attrition-test",
        selected_priority=50,
        execution_outcome="executed",
        runtime_duration_ms=1,
        selected_at=at,
    )


def _live_shaped_store() -> GraphReviewTelemetryRecorder:
    """The live histogram in miniature: 6 operator_review/concept_graph rows and
    2 chat_reflective_lane/autonomy_graph rows. Same disjointness, small enough
    to count by hand."""
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(6):
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=i)))
    for i in range(2):
        store.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base + timedelta(minutes=10 + i)))
    return store


def test_query_result_is_unchanged_by_the_attrition_refactor() -> None:
    store = _live_shaped_store()
    query = GraphReviewTelemetryQueryV1(limit=32, invocation_surface="operator_review")

    records, attrition = store.query_with_attrition(query)

    assert store.query(query) == records
    # 8 rows in, 2 dropped by the surface filter, 6 returned.
    assert attrition["total_records"] == 8
    assert attrition["dropped_by"] == {"invocation_surface": 2}
    assert attrition["matched"] == 6
    assert attrition["returned"] == 6


def test_starved_is_false_when_the_filter_matches_something() -> None:
    store = _live_shaped_store()
    _, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=32, invocation_surface="operator_review")
    )
    assert attrition["starved"] is False


def test_the_live_disjoint_filter_pair_reports_starved_not_empty() -> None:
    """The exact live failure: both conditions individually satisfiable, jointly never."""
    store = _live_shaped_store()
    records, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(
            limit=32, invocation_surface="operator_review", target_zone="autonomy_graph"
        )
    )

    assert records == []
    assert attrition["starved"] is True
    assert attrition["total_records"] == 8
    assert attrition["matched"] == 0
    # Which filter did which damage is the whole point: 2 rows lost to surface,
    # the remaining 6 lost to zone.
    assert attrition["dropped_by"] == {"invocation_surface": 2, "target_zone": 6}
    # The histograms are over the store as a whole, so they show the operator the
    # values that DO exist -- the fix is readable straight off the report.
    assert attrition["surface_histogram"] == {"operator_review": 6, "chat_reflective_lane": 2}
    assert attrition["zone_histogram"] == {"concept_graph": 6, "autonomy_graph": 2}


def test_an_empty_store_is_not_reported_as_starved() -> None:
    """`starved` must mean "rows exist and my filter rejects all of them", not
    "no rows". Conflating the two is the bug this seam exists to prevent."""
    store = GraphReviewTelemetryRecorder()
    records, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=32, invocation_surface="operator_review")
    )
    assert records == []
    assert attrition["total_records"] == 0
    assert attrition["matched"] == 0
    assert attrition["starved"] is False
    assert attrition["dropped_by"] == {}


def test_histograms_are_bounded() -> None:
    """A high-cardinality store must not grow the scheduler's log payload without
    limit; the cap is on keys, and the report stays truthful about the total."""
    from orion.substrate import review_telemetry

    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    # target_zone is a closed literal type, so cardinality has to come from a
    # field the schema leaves open; subject_ref cannot drive the histograms, so
    # assert the cap against the constant rather than fabricating 50 zones.
    for i in range(6):
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=i)))
    _, attrition = store.query_with_attrition(GraphReviewTelemetryQueryV1(limit=32))
    assert len(attrition["zone_histogram"]) <= review_telemetry._ATTRITION_HISTOGRAM_MAX_KEYS
    assert len(attrition["surface_histogram"]) <= review_telemetry._ATTRITION_HISTOGRAM_MAX_KEYS
    assert attrition["total_records"] == 6


def test_limit_slice_is_reported_separately_from_filter_drops() -> None:
    """Rows lost to the limit are not "dropped by a filter" -- widening the
    filter would not recover them, so they must not be attributed to one."""
    store = _live_shaped_store()
    _, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=2, invocation_surface="operator_review")
    )
    assert attrition["matched"] == 6
    assert attrition["returned"] == 2
    assert attrition["limit"] == 2
    assert attrition["dropped_by"] == {"invocation_surface": 2}
