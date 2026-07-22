from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import ActiveNodePressureProjectionV1, ActiveNodePressureStateV1
from orion.substrate.biometrics_loop.candidate_events import build_pressure_candidate_events
from orion.substrate.biometrics_loop.pressure_reducer import reduce_node_pressure_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


@pytest.fixture
def empty_pressure() -> ActiveNodePressureProjectionV1:
    return ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={},
    )


def _candidate(role: str, *, node_id: str = "atlas", confidence: float = 0.75) -> list:
    return build_pressure_candidate_events(
        node_id=node_id,
        semantic_role=role,
        evidence_event_ids=["gev_sample_1"],
        confidence=confidence,
        observed_at=FIXED_TS,
    )


def test_valid_pressure_candidate_accepted(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    projection, receipt = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_detected")],
        projection=empty_pressure,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    assert projection.nodes["atlas"].active_pressures == ["strain"]


def test_missing_evidence_rejected(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    trace = build_pressure_candidate_events(
        node_id="atlas",
        semantic_role="node_pressure_detected",
        evidence_event_ids=[],
        confidence=0.75,
        observed_at=FIXED_TS,
    )
    _, receipt = reduce_node_pressure_candidates(
        candidates=[trace],
        projection=empty_pressure,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert receipt.rejected_event_ids
    assert not receipt.accepted_event_ids


def test_duplicate_pressure_merged(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    first = _candidate("node_pressure_detected")
    second = _candidate("node_pressure_detected")
    projection, receipt = reduce_node_pressure_candidates(
        candidates=[first, second],
        projection=empty_pressure,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS + timedelta(seconds=30),
    )
    assert receipt.merged_event_ids
    assert projection.nodes["atlas"].active_pressures == ["strain"]


def test_suppression_updates_suppressed_pressures(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    active = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            "circe": ActiveNodePressureStateV1(
                node_id="circe",
                availability_status="offline_expected",
                active_pressures=["strain"],
                last_updated_at=FIXED_TS,
            )
        },
    )
    projection, receipt = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_suppressed", node_id="circe")],
        projection=active,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    state = projection.nodes["circe"]
    assert "strain" in state.suppressed_pressures
    assert "strain" not in state.active_pressures


def test_projection_rebuilds_deterministically(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    candidates = [_candidate("node_pressure_detected")]
    p1, r1 = reduce_node_pressure_candidates(
        candidates=candidates,
        projection=empty_pressure,
        catalog=catalog,
        now=FIXED_TS,
    )
    p2, r2 = reduce_node_pressure_candidates(
        candidates=candidates,
        projection=empty_pressure,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert p1.model_dump(mode="json") == p2.model_dump(mode="json")
    assert r1.accepted_event_ids == r2.accepted_event_ids
    assert r1.receipt_id == r2.receipt_id
    assert [d.delta_id for d in r1.state_deltas] == [d.delta_id for d in r2.state_deltas]


def test_low_confidence_rejected(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    _, receipt = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_detected", confidence=0.4)],
        projection=empty_pressure,
        catalog=catalog,
        min_confidence=0.60,
        now=FIXED_TS,
    )
    assert receipt.rejected_event_ids
    assert not receipt.accepted_event_ids


def test_availability_recovered_clears_the_flag(
    catalog: NodeCatalog,
) -> None:
    """The actual live bug this fix closes: "availability" was a one-way
    ratchet before node_availability_recovered existed -- node_pressure_decayed
    only ever clears "strain" (ROLE_TO_PRESSURE_KIND), never "availability"."""
    active = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            "atlas": ActiveNodePressureStateV1(
                node_id="atlas",
                availability_status="stale",
                active_pressures=["strain", "availability"],
                pressure_score=0.8,
                last_updated_at=FIXED_TS,
            )
        },
    )
    projection, receipt = reduce_node_pressure_candidates(
        candidates=[_candidate("node_availability_recovered")],
        projection=active,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    state = projection.nodes["atlas"]
    assert "availability" not in state.active_pressures
    # "strain" is a separate pressure kind -- recovery must not touch it.
    assert "strain" in state.active_pressures
    assert state.availability_status == "online"


def test_merge_window_dedup_persists_across_separate_reduce_calls(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    """The actual live bug this fix closes: reduce_node_pressure_candidates()
    is called once per trigger event (orion/substrate/biometrics_loop/
    pipeline.py's per-event loop), so a function-local dedup dict could
    never accumulate history across calls -- merge_window_sec=300 was a
    complete no-op in production (confirmed live: node:atlas accepted 767
    "reinforce" deltas in 2 hours instead of the ~24 a working 5-minute
    window would allow). The dedup must now live on the durable projection
    (last_accepted_at) so it survives across separate calls, matching how
    this function is actually invoked."""
    projection, receipt1 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_detected")],
        projection=empty_pressure,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS,
    )
    assert receipt1.accepted_event_ids

    # A SEPARATE call (not the same batch), 30s later -- well within the
    # 300s merge window. Must be merged, not accepted again.
    projection, receipt2 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_reinforced")],
        projection=projection,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS + timedelta(seconds=30),
    )
    assert receipt2.merged_event_ids
    assert not receipt2.accepted_event_ids


def test_merge_window_dedup_expires_after_window_across_separate_calls(
    catalog: NodeCatalog, empty_pressure: ActiveNodePressureProjectionV1
) -> None:
    """Sanity check: the dedup fix must not become a permanent block --
    a candidate arriving after merge_window_sec has elapsed must still be
    accepted."""
    projection, receipt1 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_detected")],
        projection=empty_pressure,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS,
    )
    assert receipt1.accepted_event_ids

    projection, receipt2 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_reinforced")],
        projection=projection,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS + timedelta(seconds=301),
    )
    assert receipt2.accepted_event_ids
    assert not receipt2.merged_event_ids


def test_merge_window_dedup_is_independent_per_pressure_kind(
    catalog: NodeCatalog,
) -> None:
    """last_accepted_at is keyed by pressure_kind, not a single per-node
    timestamp -- reinforcing "strain" must not suppress a fresh
    "availability" recovery candidate arriving moments later for the same
    node (and vice versa). A single shared timestamp would have made this
    fail even though the two pressure kinds are unrelated."""
    active = ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            "atlas": ActiveNodePressureStateV1(
                node_id="atlas",
                availability_status="stale",
                active_pressures=["strain", "availability"],
                last_updated_at=FIXED_TS,
            )
        },
    )
    projection, receipt1 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_pressure_reinforced")],
        projection=active,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS,
    )
    assert receipt1.accepted_event_ids

    # Different pressure_kind ("availability"), 5s later -- must NOT be
    # suppressed by "strain"'s just-set last_accepted_at.
    projection, receipt2 = reduce_node_pressure_candidates(
        candidates=[_candidate("node_availability_recovered")],
        projection=projection,
        catalog=catalog,
        merge_window_sec=300,
        now=FIXED_TS + timedelta(seconds=5),
    )
    assert receipt2.accepted_event_ids
    assert not receipt2.merged_event_ids
