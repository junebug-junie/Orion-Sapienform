from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import ActiveNodePressureProjectionV1, NodeBiometricsProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.pipeline import process_biometrics_grammar_events

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _biometrics_event(node_id: str = "atlas", *, salience: float = 0.8) -> GrammarEventV1:
    trace_id = f"biometrics.node:{node_id}:2026-05-24T12:00:00Z"
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:body_state",
        trace_id=trace_id,
        atom_type="signal",
        semantic_role="body_state",
        layer="biometrics",
        summary="body state",
        payload_ref=f"biometrics.summary:{node_id}:2026-05-24T12:00:00Z",
        salience=salience,
    )
    return GrammarEventV1(
        event_id="gev_body_1",
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-biometrics",
            source_component="biometrics_grammar_emit",
        ),
    )


def test_pipeline_runs_closed_loop_without_error(catalog: NodeCatalog) -> None:
    state = {
        "node_bio": NodeBiometricsProjectionV1(
            projection_id="proj_node_bio",
            generated_at=FIXED_TS,
            nodes={},
        ),
        "pressure": ActiveNodePressureProjectionV1(
            projection_id="proj_active_pressure",
            generated_at=FIXED_TS,
            nodes={},
        ),
        "receipts": [],
        "emissions": [],
        "published": [],
    }

    stats = process_biometrics_grammar_events(
        events=[_biometrics_event()],
        catalog=catalog,
        load_node_bio=lambda: state["node_bio"],
        save_node_bio=lambda p: state.update(node_bio=p),
        load_pressure=lambda: state["pressure"],
        save_pressure=lambda p: state.update(pressure=p),
        save_receipt=lambda r: state["receipts"].append(r),
        save_emission=lambda e: state["emissions"].append(e),
        publish_accepted=lambda evs: state["published"].extend(evs),
        now=FIXED_TS,
    )

    assert stats["events"] == 1
    assert stats["receipts"] == 2
    assert stats["emissions"] == 1
    assert len(state["receipts"]) == 2
    assert len(state["emissions"]) == 1
    assert state["emissions"][0].candidate_events
    assert "atlas" in state["node_bio"].nodes
