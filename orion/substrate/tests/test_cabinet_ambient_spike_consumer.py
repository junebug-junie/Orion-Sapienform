"""Tests for cabinet ambient spike cognition consumer."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1, NodeBiometricsStateV1
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1
from orion.substrate.biometrics_loop.constants import NODE_BIOMETRICS_PROJECTION_ID
from orion.substrate.biometrics_loop.grammar_extract import extract_node_state_from_events
from orion.substrate.cabinet_ambient_spike_consumer import (
    REDUCER_ID,
    apply_cabinet_ambient_spike_bump,
    build_cabinet_ambient_spike_grammar_events,
    spike_trace_id,
)

TS = datetime(2026, 8, 29, 4, 46, 54, tzinfo=timezone.utc)


def _spike(*, activity: float = 0.34) -> CabinetAmbientSpikeV1:
    return CabinetAmbientSpikeV1(
        spike_id="spike-abc",
        node="athena",
        timestamp=TS,
        activity=activity,
        rms=6500.0,
        peak=12000.0,
        activity_threshold=0.30,
        consecutive_ticks=2,
        source_service="orion-biometrics",
        source_node="athena",
    )


def test_build_spike_grammar_trace_has_spike_signal_atom() -> None:
    spike = _spike()
    events = build_cabinet_ambient_spike_grammar_events(spike)
    assert len(events) == 3
    assert events[0].event_kind == "trace_started"
    assert events[-1].event_kind == "trace_ended"
    atom_event = events[1]
    assert atom_event.atom is not None
    assert atom_event.atom.semantic_role == "cabinet_ambient_audio_spike_signal"
    assert atom_event.atom.salience == 0.34
    assert spike_trace_id(spike) == "cabinet.ambient.spike:athena:spike-abc"


def test_grammar_extract_resolves_spike_trace_node_id() -> None:
    spike = _spike(activity=0.37)
    events = build_cabinet_ambient_spike_grammar_events(spike)
    catalog = NodeCatalog.load("config/biometrics/node_catalog.yaml")
    state = extract_node_state_from_events(events, catalog, now=TS)
    assert state.node_id == "athena"


def test_grammar_extract_maps_spike_signal_to_activity_hint() -> None:
    spike = _spike(activity=0.37)
    events = build_cabinet_ambient_spike_grammar_events(spike)
    catalog = NodeCatalog.load("config/biometrics/node_catalog.yaml")
    state = extract_node_state_from_events(events, catalog, now=TS)
    assert state.pressure_hints["cabinet_ambient_audio_activity"] == 0.37


def test_apply_spike_bump_merges_existing_hint_upward() -> None:
    spike = _spike(activity=0.34)
    events = build_cabinet_ambient_spike_grammar_events(spike)
    catalog = NodeCatalog.load("config/biometrics/node_catalog.yaml")
    projection = NodeBiometricsProjectionV1(
        projection_id=NODE_BIOMETRICS_PROJECTION_ID,
        generated_at=TS,
        nodes={
            "athena": NodeBiometricsStateV1(
                node_id="athena",
                pressure_hints={"cabinet_ambient_audio_activity": 0.20},
            )
        },
    )

    updated, receipt = apply_cabinet_ambient_spike_bump(
        projection=projection,
        spike=spike,
        catalog=catalog,
        grammar_events=events,
        now=TS,
    )

    assert updated.nodes["athena"].pressure_hints["cabinet_ambient_audio_activity"] == 0.34
    assert receipt.state_deltas[0].reducer_id == REDUCER_ID
    assert receipt.state_deltas[0].target_kind == "node_biometrics"
    assert receipt.state_deltas[0].after["pressure_hints"]["cabinet_ambient_audio_activity"] == 0.34


def test_apply_spike_bump_does_not_downgrade_existing_hint() -> None:
    spike = _spike(activity=0.31)
    events = build_cabinet_ambient_spike_grammar_events(spike)
    catalog = NodeCatalog.load("config/biometrics/node_catalog.yaml")
    projection = NodeBiometricsProjectionV1(
        projection_id=NODE_BIOMETRICS_PROJECTION_ID,
        generated_at=TS,
        nodes={
            "athena": NodeBiometricsStateV1(
                node_id="athena",
                pressure_hints={"cabinet_ambient_audio_activity": 0.40},
            )
        },
    )

    updated, _ = apply_cabinet_ambient_spike_bump(
        projection=projection,
        spike=spike,
        catalog=catalog,
        grammar_events=events,
        now=TS,
    )

    assert updated.nodes["athena"].pressure_hints["cabinet_ambient_audio_activity"] == 0.40
