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
