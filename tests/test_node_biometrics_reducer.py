from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1, NodeBiometricsStateV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.node_reducer import reduce_biometrics_node_event

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


@pytest.fixture
def empty_projection() -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="proj_node_bio",
        generated_at=FIXED_TS,
        nodes={},
    )


def _bio_provenance() -> GrammarProvenanceV1:
    return GrammarProvenanceV1(
        source_service="orion-biometrics",
        source_component="biometrics_grammar_emit",
    )


def _atom_event(
    *,
    trace_id: str,
    event_id: str,
    role: str,
    observed_at: datetime = FIXED_TS,
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
        provenance=_bio_provenance(),
    )


def test_atlas_trace_updates_last_seen(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    event = _atom_event(
        trace_id=trace_id,
        event_id="gev_sample",
        role="telemetry_sample",
        payload_ref="biometrics.sample:atlas:2026-05-24T12:00:00Z",
    )
    projection, receipt = reduce_biometrics_node_event(
        event=event,
        projection=empty_projection,
        catalog=catalog,
        stale_after_sec=180,
        now=FIXED_TS,
    )
    assert receipt.noop_event_ids == []
    assert projection.nodes["atlas"].last_seen_at == FIXED_TS
    assert projection.nodes["atlas"].latest_sample_event_id == "gev_sample"


def test_prometheous_alias_resolves_to_prometheus(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    trace_id = "biometrics.node:prometheous:2026-05-24T12:00:00Z"
    event = _atom_event(
        trace_id=trace_id,
        event_id="gev_ctx",
        role="node_context",
        text_value="prometheous",
    )
    projection, _ = reduce_biometrics_node_event(
        event=event,
        projection=empty_projection,
        catalog=catalog,
        now=FIXED_TS,
    )
    assert "prometheus" in projection.nodes
    assert projection.nodes["prometheus"].node_id == "prometheus"


def test_circe_expected_offline_preserved(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    trace_id = "biometrics.node:circe:2026-05-24T12:00:00Z"
    event = _atom_event(
        trace_id=trace_id,
        event_id="gev_avail",
        role="node_availability",
        summary="circe telemetry status OK (expected offline, known node)",
    )
    projection, _ = reduce_biometrics_node_event(
        event=event,
        projection=empty_projection,
        catalog=catalog,
        now=FIXED_TS,
    )
    state = projection.nodes["circe"]
    assert state.expected_online is False
    assert state.availability_status == "offline_expected"


def test_payload_ref_stored_no_blob_copy(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    event = _atom_event(
        trace_id=trace_id,
        event_id="gev_body",
        role="body_state",
        payload_ref="biometrics.summary:atlas:2026-05-24T12:00:00Z",
        salience=0.42,
    )
    projection, _ = reduce_biometrics_node_event(
        event=event,
        projection=empty_projection,
        catalog=catalog,
        now=FIXED_TS,
    )
    state = projection.nodes["atlas"]
    assert "gpu_util" not in str(state.model_dump())
    assert state.latest_payload_ref.startswith("biometrics.")


def test_sample_summary_induction_event_ids_attach(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    trace_id = "biometrics.node:atlas:2026-05-24T12:00:00Z"
    events = [
        _atom_event(
            trace_id=trace_id,
            event_id="gev_sample",
            role="telemetry_sample",
            payload_ref="biometrics.sample:atlas:2026-05-24T12:00:00Z",
        ),
        _atom_event(
            trace_id=trace_id,
            event_id="gev_summary",
            role="body_state",
            payload_ref="biometrics.summary:atlas:2026-05-24T12:00:00Z",
        ),
        _atom_event(
            trace_id=trace_id,
            event_id="gev_induction",
            role="body_state",
            payload_ref="biometrics.induction:atlas:2026-05-24T12:00:00Z",
        ),
    ]
    projection = empty_projection
    for event in events:
        projection, _ = reduce_biometrics_node_event(
            event=event,
            projection=projection,
            catalog=catalog,
            now=FIXED_TS,
        )
    state = projection.nodes["atlas"]
    assert state.latest_sample_event_id == "gev_sample"
    assert state.latest_summary_event_id == "gev_summary"
    assert state.latest_induction_event_id == "gev_induction"


def test_non_biometrics_source_is_noop(
    catalog: NodeCatalog, empty_projection: NodeBiometricsProjectionV1
) -> None:
    event = GrammarEventV1(
        event_id="gev_other",
        event_kind="atom_emitted",
        trace_id="biometrics.node:atlas:2026-05-24T12:00:00Z",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        provenance=GrammarProvenanceV1(source_service="other-service"),
    )
    projection, receipt = reduce_biometrics_node_event(
        event=event,
        projection=empty_projection,
        catalog=catalog,
    )
    assert projection == empty_projection
    assert receipt.noop_event_ids == ["gev_other"]
