from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.grammar_extract import extract_node_state_from_events
from orion.substrate.biometrics_loop.ids import parse_biometrics_trace_id, parse_pressure_trace_id

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _provenance() -> GrammarProvenanceV1:
    return GrammarProvenanceV1(
        source_service="orion-biometrics",
        source_component="biometrics_grammar_emit",
    )


def _atom_event(
    *,
    trace_id: str,
    event_id: str,
    role: str,
    observed_at: datetime,
    payload_ref: str | None = None,
    salience: float | None = None,
    summary: str = "test",
    text_value: str | None = None,
) -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{role}",
        trace_id=trace_id,
        atom_type="signal",
        semantic_role=role,
        layer="biometrics",
        summary=summary,
        text_value=text_value,
        salience=salience,
        payload_ref=payload_ref,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=observed_at,
        observed_at=observed_at,
        atom=atom,
        provenance=_provenance(),
    )


def test_parse_biometrics_trace_id() -> None:
    assert parse_biometrics_trace_id("biometrics.node:atlas:2026-05-24T12:00:00Z") == "atlas"
    assert parse_biometrics_trace_id("substrate.pressure:atlas:2026-05-24T12:00:00Z") is None
    assert parse_biometrics_trace_id("biometrics.node:atlas") is None


def test_parse_pressure_trace_id() -> None:
    assert parse_pressure_trace_id("substrate.pressure:atlas:2026-05-24T12:00:00Z") == "atlas"
    assert parse_pressure_trace_id("biometrics.node:atlas:2026-05-24T12:00:00Z") is None


def test_prometheous_resolves_via_catalog(catalog: NodeCatalog) -> None:
    assert catalog.resolve("prometheous").node_id == "prometheus"


def test_extract_sets_last_seen_and_event_ids(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_sample",
            role="telemetry_sample",
            observed_at=FIXED_TS,
            payload_ref="biometrics.sample:atlas:2026-05-24T12:00:00Z",
        ),
        _atom_event(
            trace_id=trace_id,
            event_id="gev_body",
            role="body_state",
            observed_at=FIXED_TS,
            payload_ref="biometrics.summary:atlas:2026-05-24T12:00:00Z",
            salience=0.55,
        ),
        _atom_event(
            trace_id=trace_id,
            event_id="gev_induction",
            role="body_state",
            observed_at=FIXED_TS,
            payload_ref="biometrics.induction:atlas:2026-05-24T12:00:00Z",
            salience=0.4,
        ),
    ]
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=FIXED_TS)
    assert state.node_id == "atlas"
    assert state.last_seen_at == FIXED_TS
    assert state.latest_sample_event_id == "gev_sample"
    assert state.latest_summary_event_id == "gev_body"
    assert state.latest_induction_event_id == "gev_induction"
    assert state.latest_payload_ref == "biometrics.induction:atlas:2026-05-24T12:00:00Z"
    assert state.pressure_hints.get("strain") == 0.4


def test_extract_availability_online_when_fresh(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_avail",
            role="node_availability",
            observed_at=FIXED_TS,
            summary="atlas telemetry status OK (expected online, known node)",
        ),
    ]
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=FIXED_TS)
    assert state.availability_status == "online"


def test_extract_circe_offline_expected(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:circe:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_avail",
            role="node_availability",
            observed_at=FIXED_TS,
            summary="circe telemetry status OK (expected offline, known node)",
        ),
    ]
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=FIXED_TS)
    assert state.availability_status == "offline_expected"


def test_extract_stale_when_observed_too_old(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_sample",
            role="telemetry_sample",
            observed_at=FIXED_TS,
            payload_ref="biometrics.sample:atlas:2026-05-24T12:00:00Z",
        ),
    ]
    now = FIXED_TS + timedelta(seconds=200)
    state = extract_node_state_from_events(events, catalog, stale_after_sec=180, now=now)
    assert state.availability_status == "stale"


def test_extract_no_debug_trace_usage(catalog: NodeCatalog) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_sample",
            role="telemetry_sample",
            observed_at=FIXED_TS,
            payload_ref="biometrics.sample:atlas:2026-05-24T12:00:00Z",
        ),
    ]
    state = extract_node_state_from_events(events, catalog)
    dumped = state.model_dump_json()
    assert "debug_trace" not in dumped
    assert "gpu_util" not in dumped
