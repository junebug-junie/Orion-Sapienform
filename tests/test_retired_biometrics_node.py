"""Retired hosts (node_catalog.yaml `retired_nodes:`) leave the live biometrics loop.

Regression cover for atlas, 2026-10-16: decommissioned 2026-08-21, last biometrics
report 2026-08-20T22:43:26Z, yet still a node in `substrate_node_biometrics_projection`
(cached `expected_online: true`, 8 frozen `pressure_hints`) and in
`substrate_active_node_pressure_projection` (stale `availability` pressure, `strain`
re-reinforced on every tick). Its frozen keys added zero deltas to every
`biometrics_prediction_error()` mean. Nothing rewrote the entry because a host that
no longer exists never sends the event that would.
"""

from __future__ import annotations

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
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.biometrics_loop.candidate_events import build_pressure_candidate_events
from orion.substrate.biometrics_loop.node_reducer import reduce_biometrics_node_event
from orion.substrate.biometrics_loop.pipeline import (
    process_biometrics_grammar_events,
    prune_retired_nodes,
)
from orion.substrate.biometrics_loop.pressure_organ import (
    invoke_biometrics_pressure,
    sweep_absent_nodes,
    sweep_suppressible_nodes,
)
from orion.substrate.biometrics_loop.pressure_reducer import reduce_node_pressure_candidates
from orion.substrate.prediction_error import biometrics_prediction_error

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
NOW = datetime(2026, 10, 16, 3, 3, 22, tzinfo=timezone.utc)
ATLAS_LAST_SEEN = datetime(2026, 8, 20, 22, 43, 26, tzinfo=timezone.utc)

# atlas's pressure_hints verbatim from substrate_node_biometrics_projection, 2026-10-16.
ATLAS_FROZEN_HINTS = {
    "gpu": 0.0,
    "strain": 0.06884440810864358,
    "fan_pressure": 0.72,
    "disk_pressure": 0.00013899738798999628,
    "power_pressure": 0.3567925057485332,
    "memory_pressure": 0.11267778854128259,
    "thermal_pressure": 0.0,
    "disk_capacity_pressure": 0.3837,
}


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _state(node_id: str, *, hints: dict, last_seen_at: datetime, expected_online=True):
    return NodeBiometricsStateV1(
        node_id=node_id,
        expected_online=expected_online,
        availability_status="online",
        last_seen_at=last_seen_at,
        pressure_hints=dict(hints),
        capabilities=["local_llm_heavy"],
    )


def _bio(nodes: dict) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(projection_id="proj_node_bio", generated_at=NOW, nodes=nodes)


def _pressure(nodes: dict) -> ActiveNodePressureProjectionV1:
    return ActiveNodePressureProjectionV1(
        projection_id="proj_active_pressure", generated_at=NOW, nodes=nodes
    )


def _atlas_pressure() -> ActiveNodePressureStateV1:
    return ActiveNodePressureStateV1(
        node_id="atlas",
        availability_status="stale",
        last_updated_at=NOW,
        active_pressures=["availability"],
        pressure_score=0.9,
    )


def _live_shape() -> tuple[NodeBiometricsProjectionV1, ActiveNodePressureProjectionV1]:
    bio = _bio(
        {
            "atlas": _state("atlas", hints=ATLAS_FROZEN_HINTS, last_seen_at=ATLAS_LAST_SEEN),
            "circe": _state("circe", hints={"strain": 0.29, "gpu": 0.24}, last_seen_at=NOW),
        }
    )
    pressure = _pressure(
        {
            "atlas": _atlas_pressure(),
            "circe": ActiveNodePressureStateV1(
                node_id="circe", availability_status="online", last_updated_at=NOW,
                    active_pressures=["strain"],
            ),
        }
    )
    return bio, pressure


def _event(node_id: str) -> GrammarEventV1:
    trace_id = f"biometrics.node:{node_id}:2026-10-16T03:03:22Z"
    return GrammarEventV1(
        event_id=f"gev_body_{node_id}",
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=NOW,
        observed_at=NOW,
        atom=GrammarAtomV1(
            atom_id=f"{trace_id}:body_state",
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="body_state",
            layer="biometrics",
            summary="body state",
            payload_ref=f"biometrics.summary:{node_id}:2026-10-16T03:03:22Z",
            salience=0.8,
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-biometrics", source_component="biometrics_grammar_emit"
        ),
    )


# ------------------------------------------------------------------ catalog


def test_atlas_is_retired_and_out_of_the_live_roster(catalog: NodeCatalog) -> None:
    assert "atlas" not in catalog.profiles
    assert catalog.is_retired("atlas")
    assert catalog.is_retired("ATLAS.tail348bbe.ts.net")
    profile = catalog.resolve("atlas.tail348bbe.ts.net")
    assert profile.node_id == "atlas"
    assert profile.known is True
    assert profile.retired is True
    assert profile.expected_online is False


def test_live_hosts_are_not_retired(catalog: NodeCatalog) -> None:
    for node in ("athena", "circe", "hecate", "prometheus"):
        assert node in catalog.profiles
        assert not catalog.is_retired(node)
        assert catalog.resolve(node).retired is False
    assert not catalog.is_retired("brand-new-box")
    assert catalog.resolve("brand-new-box").known is False


def test_a_node_cannot_be_both_live_and_retired(tmp_path) -> None:
    path = tmp_path / "node_catalog.yaml"
    path.write_text("nodes:\n  atlas: {}\nretired_nodes:\n  atlas: {}\n")
    with pytest.raises(ValueError, match="both"):
        NodeCatalog.load(path)


# ------------------------------------------------------------------ prune


def test_prune_drops_retired_host_from_both_projections(catalog: NodeCatalog) -> None:
    bio, pressure = _live_shape()
    new_bio, new_pressure, receipt = prune_retired_nodes(
        node_bio=bio, pressure=pressure, catalog=catalog, now=NOW
    )
    assert set(new_bio.nodes) == {"circe"}
    assert set(new_pressure.nodes) == {"circe"}
    assert set(bio.nodes) == {"atlas", "circe"}, "input projection must not be mutated"
    assert receipt is not None
    removed = {(d.target_kind, d.target_id) for d in receipt.state_deltas}
    assert removed == {("node_biometrics", "atlas"), ("active_node_pressure", "atlas")}
    assert all(d.after is None and d.before for d in receipt.state_deltas)


def test_prune_is_a_noop_when_nothing_is_retired(catalog: NodeCatalog) -> None:
    bio = _bio({"circe": _state("circe", hints={"strain": 0.3}, last_seen_at=NOW)})
    pressure = _pressure({})
    new_bio, new_pressure, receipt = prune_retired_nodes(
        node_bio=bio, pressure=pressure, catalog=catalog, now=NOW
    )
    assert receipt is None
    assert new_bio is bio and new_pressure is pressure


def test_pipeline_persists_the_prune_on_an_ordinary_tick(catalog: NodeCatalog) -> None:
    """End to end on the live shape: one ordinary circe report removes atlas from
    both stored projections, and nothing in the same tick puts it back."""
    bio, pressure = _live_shape()
    store = {"bio": bio, "pressure": pressure, "receipts": []}
    process_biometrics_grammar_events(
        events=[_event("circe")],
        catalog=catalog,
        load_node_bio=lambda: store["bio"],
        save_node_bio=lambda p: store.update(bio=p),
        load_pressure=lambda: store["pressure"],
        save_pressure=lambda p: store.update(pressure=p),
        save_receipt=store["receipts"].append,
        save_emission=lambda _e: None,
        publish_accepted=lambda _evs: None,
        now=NOW,
    )
    assert "atlas" not in store["bio"].nodes
    assert "atlas" not in store["pressure"].nodes
    assert "circe" in store["bio"].nodes
    assert any(
        d.reducer_id == "biometrics_retired_node_prune"
        for r in store["receipts"]
        for d in r.state_deltas
    )


# --------------------------------------------- nothing can bring it back


def test_node_reducer_refuses_a_retired_host_event(catalog: NodeCatalog) -> None:
    projection, receipt = reduce_biometrics_node_event(
        event=_event("atlas"), projection=_bio({}), catalog=catalog, now=NOW
    )
    assert projection.nodes == {}
    assert receipt.noop_event_ids == ["gev_body_atlas"]


def test_organ_emits_nothing_for_a_retired_host(catalog: NodeCatalog) -> None:
    bio, pressure = _live_shape()
    emission = invoke_biometrics_pressure(
        trigger_event=_event("atlas"), node_bio=bio, active_pressure=pressure,
        catalog=catalog, now=NOW,
    )
    assert emission.candidate_events == []


def test_pressure_reducer_rejects_a_retired_host_candidate(catalog: NodeCatalog) -> None:
    candidate = build_pressure_candidate_events(
        node_id="atlas", semantic_role="node_pressure_reinforced",
        evidence_event_ids=["gev_x"], confidence=0.9, observed_at=NOW,
    )
    projection, receipt = reduce_node_pressure_candidates(
        candidates=[candidate], projection=_pressure({}), catalog=catalog, now=NOW
    )
    assert projection.nodes == {}
    assert receipt.rejected_event_ids and not receipt.accepted_event_ids


def test_sweeps_never_target_a_retired_host(catalog: NodeCatalog) -> None:
    """Even with the stale cached `expected_online: true` that live atlas carried."""
    bio, pressure = _live_shape()
    assert "atlas" not in sweep_absent_nodes(node_bio=bio, catalog=catalog, now=NOW)
    assert "atlas" not in sweep_suppressible_nodes(
        node_bio=bio, active_pressure=pressure, catalog=catalog, now=NOW
    )


# ------------------------------------------------ the metric it was diluting


def test_frozen_retired_keys_no_longer_dilute_biometrics_prediction_error(
    catalog: NodeCatalog,
) -> None:
    """With atlas present, its 8 frozen keys add 8 zero deltas to the mean."""
    prev, _ = _live_shape()
    curr = prev.model_copy(deep=True)
    curr.nodes["circe"].pressure_hints = {"strain": 0.59, "gpu": 0.84}  # +0.30, +0.60

    diluted = biometrics_prediction_error(prev, curr)
    pruned_prev, _, _ = prune_retired_nodes(
        node_bio=prev, pressure=_pressure({}), catalog=catalog, now=NOW
    )
    pruned_curr, _, _ = prune_retired_nodes(
        node_bio=curr, pressure=_pressure({}), catalog=catalog, now=NOW
    )
    clean = biometrics_prediction_error(pruned_prev, pruned_curr)

    # mean(|d|) = 0.90/10 = 0.09 -> 0.30 with atlas; 0.90/2 = 0.45 -> saturates without.
    assert diluted == pytest.approx(0.09 / 0.30)
    assert clean == pytest.approx(1.0)
    assert ATLAS_LAST_SEEN < NOW - timedelta(days=50)


def test_absence_sweep_path_does_not_claim_a_node_bio_prune_it_never_saves(
    catalog: NodeCatalog,
) -> None:
    """The runtime's absence sweep runs with enable_node_reducer=False and a no-op
    node-bio saver. The prune receipt must only name what that call persists."""
    bio, pressure = _live_shape()
    store = {"pressure": pressure, "receipts": []}
    process_biometrics_grammar_events(
        events=[],
        catalog=catalog,
        load_node_bio=lambda: bio,
        save_node_bio=lambda _p: None,
        load_pressure=lambda: store["pressure"],
        save_pressure=lambda p: store.update(pressure=p),
        save_receipt=store["receipts"].append,
        save_emission=lambda _e: None,
        publish_accepted=None,
        enable_node_reducer=False,
        now=NOW,
    )
    assert "atlas" not in store["pressure"].nodes
    kinds = {d.target_kind for r in store["receipts"] for d in r.state_deltas}
    assert kinds == {"active_node_pressure"}


def test_cabinet_spike_cannot_recreate_a_retired_host(catalog: NodeCatalog) -> None:
    from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1
    from orion.substrate.cabinet_ambient_spike_consumer import (
        apply_cabinet_ambient_spike_bump,
        build_cabinet_ambient_spike_grammar_events,
    )

    spike = CabinetAmbientSpikeV1(
        spike_id="spike-atlas",
        node="atlas",
        timestamp=NOW,
        activity=0.5,
        rms=6500.0,
        peak=12000.0,
        activity_threshold=0.30,
        consecutive_ticks=2,
        source_service="orion-biometrics",
        source_node="atlas",
    )
    projection, receipt = apply_cabinet_ambient_spike_bump(
        projection=_bio({}),
        spike=spike,
        catalog=catalog,
        grammar_events=build_cabinet_ambient_spike_grammar_events(spike),
        now=NOW,
    )
    assert projection.nodes == {}
    assert receipt.noop_event_ids and not receipt.state_deltas
