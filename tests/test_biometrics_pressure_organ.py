from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.candidate_events import build_pressure_candidate_events
from orion.substrate.biometrics_loop.emission_validator import validate_organ_emission
from orion.substrate.biometrics_loop.pressure_organ import invoke_biometrics_pressure

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)

ALLOWED_ROLES = {
    "node_pressure_detected",
    "node_pressure_reinforced",
    "node_pressure_decayed",
    "node_availability_concern",
    "node_pressure_suppressed",
    "node_capability_impact",
}


def _group_traces(events: list[GrammarEventV1]) -> list[list[GrammarEventV1]]:
    grouped: dict[str, list[GrammarEventV1]] = defaultdict(list)
    order: list[str] = []
    for event in events:
        if event.trace_id not in grouped:
            order.append(event.trace_id)
        grouped[event.trace_id].append(event)
    return [grouped[trace_id] for trace_id in order]


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _trigger_event(node_id: str = "atlas") -> GrammarEventV1:
    return GrammarEventV1(
        event_id="gev_trigger",
        event_kind="atom_emitted",
        trace_id=f"biometrics.node:{node_id}:2026-05-24T12:00:00Z",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        provenance=GrammarProvenanceV1(
            source_service="orion-biometrics",
            source_component="biometrics_grammar_emit",
        ),
    )


def _node_bio(
    *,
    node_id: str,
    availability_status: str = "online",
    last_seen_at: datetime | None = FIXED_TS,
    pressure_hints: dict | None = None,
    expected_online: bool | None = True,
    capabilities: list[str] | None = None,
) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="proj_node_bio",
        generated_at=FIXED_TS,
        nodes={
            node_id: NodeBiometricsStateV1(
                node_id=node_id,
                expected_online=expected_online,
                availability_status=availability_status,  # type: ignore[arg-type]
                last_seen_at=last_seen_at,
                pressure_hints=pressure_hints or {},
                capabilities=capabilities or [],
            )
        },
    )


def _active_pressure(
    *,
    node_id: str,
    active_pressures: list[str] | None = None,
    suppressed_pressures: list[str] | None = None,
) -> ActiveNodePressureProjectionV1:
    return ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure",
        generated_at=FIXED_TS,
        nodes={
            node_id: ActiveNodePressureStateV1(
                node_id=node_id,
                availability_status="online",
                active_pressures=active_pressures or [],
                suppressed_pressures=suppressed_pressures or [],
                last_updated_at=FIXED_TS,
            )
        },
    )


def test_circe_expected_offline_emits_suppression_candidate(catalog: NodeCatalog) -> None:
    stale_ts = FIXED_TS - timedelta(seconds=300)
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("circe"),
        node_bio=_node_bio(
            node_id="circe",
            availability_status="offline_expected",
            last_seen_at=stale_ts,
            expected_online=False,
        ),
        active_pressure=_active_pressure(node_id="circe"),
        catalog=catalog,
        stale_after_sec=180,
        now=FIXED_TS,
    )
    roles = {
        ev.atom.semantic_role
        for trace in _group_traces(emission.candidate_events)
        for ev in trace
        if ev.atom
    }
    assert "node_pressure_suppressed" in roles


def test_missing_expected_atlas_emits_availability_concern(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(
            node_id="atlas",
            availability_status="stale",
            last_seen_at=FIXED_TS - timedelta(seconds=300),
            expected_online=True,
        ),
        active_pressure=_active_pressure(node_id="atlas"),
        catalog=catalog,
        stale_after_sec=180,
        now=FIXED_TS,
    )
    roles = {
        ev.atom.semantic_role
        for trace in _group_traces(emission.candidate_events)
        for ev in trace
        if ev.atom
    }
    assert "node_availability_concern" in roles


def test_organ_emits_only_allowlisted_roles(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(
            node_id="atlas",
            pressure_hints={"strain": 0.8, "gpu": 0.7},
            capabilities=["local_llm_heavy"],
        ),
        active_pressure=_active_pressure(node_id="atlas", active_pressures=["strain"]),
        catalog=catalog,
        now=FIXED_TS,
    )
    for trace in _group_traces(emission.candidate_events):
        for ev in trace:
            if ev.atom:
                assert ev.atom.semantic_role in ALLOWED_ROLES


def test_organ_does_not_emit_state_deltas(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(node_id="atlas", pressure_hints={"strain": 0.9}),
        active_pressure=_active_pressure(node_id="atlas"),
        catalog=catalog,
        now=FIXED_TS,
    )
    assert "state_delta" not in emission.model_dump_json()


def test_pressure_reinforced_when_prior_active(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(node_id="atlas", pressure_hints={"strain": 0.75}),
        active_pressure=_active_pressure(node_id="atlas", active_pressures=["strain"]),
        catalog=catalog,
        now=FIXED_TS,
    )
    roles = {
        ev.atom.semantic_role
        for trace in _group_traces(emission.candidate_events)
        for ev in trace
        if ev.atom
    }
    assert "node_pressure_reinforced" in roles


def test_pressure_decayed_when_hints_empty(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(node_id="atlas", pressure_hints={}),
        active_pressure=_active_pressure(node_id="atlas", active_pressures=["strain"]),
        catalog=catalog,
        now=FIXED_TS,
    )
    roles = {
        ev.atom.semantic_role
        for trace in _group_traces(emission.candidate_events)
        for ev in trace
        if ev.atom
    }
    assert "node_pressure_decayed" in roles


def test_capability_impact_for_llm_gpu(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(
            node_id="atlas",
            pressure_hints={"gpu": 0.85},
            capabilities=["local_llm_heavy"],
        ),
        active_pressure=_active_pressure(node_id="atlas"),
        catalog=catalog,
        now=FIXED_TS,
    )
    roles = {
        ev.atom.semantic_role
        for trace in _group_traces(emission.candidate_events)
        for ev in trace
        if ev.atom
    }
    assert "node_capability_impact" in roles


def test_emission_validator_accepts_valid_emission(catalog: NodeCatalog) -> None:
    emission = invoke_biometrics_pressure(
        trigger_event=_trigger_event("atlas"),
        node_bio=_node_bio(node_id="atlas", pressure_hints={"strain": 0.9}),
        active_pressure=_active_pressure(node_id="atlas"),
        catalog=catalog,
        now=FIXED_TS,
    )
    validate_organ_emission(emission, max_events=8)


def test_candidate_events_have_pressure_provenance() -> None:
    trace = build_pressure_candidate_events(
        node_id="atlas",
        semantic_role="node_pressure_detected",
        evidence_event_ids=["gev_sample"],
        confidence=0.8,
        observed_at=FIXED_TS,
    )
    assert trace[0].event_kind == "trace_started"
    assert trace[-1].event_kind == "trace_ended"
    atom_event = next(ev for ev in trace if ev.atom)
    assert atom_event.provenance.source_service == "orion-substrate-organs"
    assert atom_event.provenance.source_component == "biometrics_pressure"
