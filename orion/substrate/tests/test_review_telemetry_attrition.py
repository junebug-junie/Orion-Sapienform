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


def _record(
    *, surface: str, zone: str, at: datetime, telemetry_id: str | None = None
) -> GraphReviewTelemetryRecordV1:
    kwargs = {"telemetry_id": telemetry_id} if telemetry_id else {}
    return GraphReviewTelemetryRecordV1(
        **kwargs,  # type: ignore[arg-type]
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


def test_query_returns_the_newest_matching_rows_sorted_newest_first() -> None:
    """Pins sort direction AND that the limit slice happens AFTER the sort.

    Asserting `query(q) == query_with_attrition(q)[0]` proves nothing: `query()`
    delegates to `query_with_attrition()`, so that comparison holds for any
    implementation, including one that slices before sorting and hands the
    mutation scheduler the OLDEST rows instead of the newest.
    """
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    # Recorded oldest-first, so insertion order is the reverse of the expectation.
    for i in range(5):
        store.record(
            _record(
                surface="operator_review",
                zone="concept_graph",
                at=base + timedelta(minutes=i),
                telemetry_id=f"row-{i}",
            )
        )

    records, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=2, invocation_surface="operator_review")
    )

    # Newest two, newest first. A slice-before-sort returns row-0/row-1; an
    # ascending sort returns row-0/row-1 as well. Both are excluded here.
    assert [r.telemetry_id for r in records] == ["row-4", "row-3"]
    assert attrition["matched"] == 5
    assert attrition["returned"] == 2


def test_filter_attribution_survives_a_reordering_of_the_filters() -> None:
    """`dropped_by` is the headline product of this seam, so the counts must be
    asymmetric -- equal counts would make a swapped filter order indistinguishable."""
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(5):  # wrong surface only
        store.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base + timedelta(minutes=i)))
    for i in range(2):  # wrong zone only
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=10 + i)))
    for i in range(3):  # wrong on BOTH -- whichever filter runs first claims these
        store.record(_record(surface="chat_reflective_lane", zone="concept_graph", at=base + timedelta(minutes=20 + i)))

    _, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=32, invocation_surface="operator_review", target_zone="autonomy_graph")
    )

    # Rows failing both filters are the discriminator. Surface first: it claims
    # 5 + 3 = 8, leaving 2 for the zone filter. Zone first would give
    # {"target_zone": 5, "invocation_surface": 5}. Without the both-wrong rows the
    # two orders produce the SAME dict and the assertion proves nothing.
    assert attrition["dropped_by"] == {"invocation_surface": 8, "target_zone": 2}


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


def test_matched_zone_histogram_is_pre_slice_not_whole_store() -> None:
    """The signature that separates limit truncation from a real zone mismatch.

    The whole-store histogram cannot do this job: it counts rows that failed the
    surface filter, so on live data it shows autonomy_graph rows that the
    consumer can never actually use.
    """
    base = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    store = GraphReviewTelemetryRecorder()
    for i in range(4):  # usable, but oldest -- falls off a small limit
        store.record(_record(surface="operator_review", zone="autonomy_graph", at=base + timedelta(minutes=i)))
    for i in range(3):  # newest, wrong zone -- wins the limit slice
        store.record(_record(surface="operator_review", zone="concept_graph", at=base + timedelta(minutes=10 + i)))
    for i in range(2):  # wrong surface entirely
        store.record(_record(surface="chat_reflective_lane", zone="autonomy_graph", at=base + timedelta(minutes=20 + i)))

    records, attrition = store.query_with_attrition(
        GraphReviewTelemetryQueryV1(limit=3, invocation_surface="operator_review")
    )

    assert [r.target_zone for r in records] == ["concept_graph"] * 3
    # Pre-slice, surface-filtered: the 2 wrong-surface autonomy_graph rows are gone.
    assert attrition["matched_zone_histogram"] == {"concept_graph": 3, "autonomy_graph": 4}
    # Whole-store, for contrast: it still counts them, which is why it cannot be
    # used to decide whether usable rows exist.
    assert attrition["zone_histogram"] == {"concept_graph": 3, "autonomy_graph": 6}
