from __future__ import annotations

from datetime import datetime, timezone


def test_brain_frame_schema_roundtrips_and_defaults():
    from orion.schemas.brain_frame import (
        SUBSTRATE_BRAIN_FRAME_KIND,
        BrainRegionV1,
        SubstrateBrainFrameV1,
    )

    assert SUBSTRATE_BRAIN_FRAME_KIND == "substrate.brain_frame.v1"

    region = BrainRegionV1(
        dimension="node_kind",
        region_id="node_kind:tension",
        label="Tension",
        intensity=0.9,
        state="firing",
        node_count=3,
        as_of=datetime(2026, 7, 7, tzinfo=timezone.utc),
        stale=False,
    )
    frame = SubstrateBrainFrameV1(
        frame_id="abc123",
        generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        tick_seq=1,
        phase="warming",
        regions=[region],
    )
    dumped = frame.model_dump(mode="json")
    again = SubstrateBrainFrameV1.model_validate(dumped)
    assert again.phase == "warming"
    assert again.regions[0].state == "firing"
    assert again.spotlight is None
    assert again.nodes == [] and again.edges == []
    assert again.schema_version == "substrate.brain_frame.v1"


from types import SimpleNamespace


def _node(node_id, kind, activation, pressure=0.0, dormant=False):
    return SimpleNamespace(
        node_id=node_id,
        node_kind=kind,
        label=f"{kind}:{node_id}",
        activation=activation,
        metadata={"dynamic_pressure": pressure, "dormant": dormant},
    )


def _settings():
    return SimpleNamespace(
        brain_frame_sample_nodes=40,
        brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5,
        brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0,
        brain_frame_spotlight_cadence_sec=30.0,
    )


def test_producer_yields_firing_and_starving_regions_and_samples():
    from datetime import datetime, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
    nodes = [
        _node("t1", "tension", activation=0.95, pressure=0.9),
        _node("t2", "tension", activation=0.8, pressure=0.7),
        _node("c1", "concept", activation=0.02, pressure=0.0, dormant=True),
    ]
    lane_health = {
        "cursor_lag_by_reducer": {"execution_trajectory": 1.0, "transport_bus": 400.0},
        "pending_backlog_by_reducer": {"execution_trajectory": 12, "transport_bus": 0},
        "quarantine_by_reducer": {},
    }
    frame = assemble_brain_frame(
        nodes=nodes,
        edges=[],
        lane_health=lane_health,
        self_state=None,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=7,
    )
    assert frame.phase == "live"  # real activation present
    kinds = {r.region_id: r for r in frame.regions if r.dimension == "node_kind"}
    assert kinds["node_kind:tension"].state == "firing"
    assert kinds["node_kind:concept"].state == "starving"
    lanes = {r.region_id: r for r in frame.regions if r.dimension == "lane"}
    # Fresh lane (low lag, backlog) fires; badly-lagged lane starves.
    assert lanes["lane:execution_trajectory"].state in {"firing", "steady"}
    assert lanes["lane:transport_bus"].state == "starving"
    assert len(frame.nodes) >= 1  # non-empty decoration
    assert frame.tick_seq == 7


def test_producer_warming_when_graph_dead():
    from datetime import datetime, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    frame = assemble_brain_frame(
        nodes=[_node("c1", "concept", activation=0.0)],
        edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=None,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=0,
    )
    assert frame.phase == "warming"


def test_self_state_region_marked_stale_when_old():
    from datetime import datetime, timedelta, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
    old = (now - timedelta(seconds=120)).isoformat()
    self_state = {
        "generated_at": old,
        "dimensions": {
            "execution_pressure": {"score": 0.8, "confidence": 0.7},
            "coherence": {"score": 0.4, "confidence": 0.6},
        },
    }
    frame = assemble_brain_frame(
        nodes=[_node("t1", "tension", 0.9, 0.9)],
        edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=self_state,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=3,
    )
    ss = {r.region_id: r for r in frame.regions if r.dimension == "self_state"}
    assert ss["self_state:execution_pressure"].stale is True
    assert ss["self_state:execution_pressure"].intensity == 0.8
