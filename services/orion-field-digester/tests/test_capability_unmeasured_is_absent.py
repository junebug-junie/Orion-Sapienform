"""An unmeasured capability channel is ABSENT, never a perfect reading.

Before 2026-10-07 (fix/field-capability-unmeasured-fallback), once every input
to a capability channel expired (decay.py EXPIRING_NODE_CHANNELS), apply_diffusion
wrote pressure 0.0 and the derived fallback wrote confidence 1.0 and
available_capacity 1.0, with no provenance: a capability nobody was measuring
read as perfectly healthy to every generic consumer. The worst shape is
"outage reads as recovery": the eye reporting staleness 1.0 (no camera), then
the router dying, flipped capability:vision from alarm straight to perfect.

These run the production order (reconcile -> digestion tick, same as
app/worker.py) against the live topology, so reconcile's default re-seeding is
exercised too: it refills the keys every tick, and diffusion must drop them
again before the tick is saved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.attention.field_attention.selectors import _current_pressure_proxy
from orion.field.pressure import collect_field_channel_pressures
from orion.schemas.field_state import FieldStateV1

from app.digestion.perturbation import apply_perturbations
from app.graph.lattice import load_lattice
from app.ingest.state_deltas import Perturbation
from app.tensor.field_state import empty_field_state
from app.tensor.reconcile import reconcile_field_state_with_lattice
from app.tensor.update_rules import run_digestion_tick

NOW = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
VISION_NODE = "node:substrate.vision_organ"
STORAGE_NODE = "node:substrate.storage_write"
RPC_NODE = "node:substrate.rpc_delivery"
DERIVED = ("confidence", "available_capacity")


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


class _NoHistoryStore:
    """significance reads recent rows; none exist in a unit test."""

    def load_recent_field_json(self, *, window_seconds: float) -> list:
        return []


class _Field:
    """The worker's per-tick loop, minus Postgres and the bus."""

    def __init__(self) -> None:
        self.lattice = _lattice()
        self.state = empty_field_state(lattice=self.lattice, now=NOW, tick_id="t0")

    def write(self, node: str, channels: dict[str, float], at: datetime) -> None:
        self.state.generated_at = at
        apply_perturbations(
            self.state,
            [Perturbation(node_id=node, channel=ch, intensity=v, label="test", mode="replace") for ch, v in channels.items()],
            now=at,
        )

    def tick(self, at: datetime) -> FieldStateV1:
        self.state = reconcile_field_state_with_lattice(self.state, lattice=self.lattice)
        self.state.generated_at = at
        run_digestion_tick(
            self.state,
            perturbations=[],
            decay_rate=0.92,
            diffusion_rate=1.0,
            staleness_threshold_sec=90.0,
            store=_NoHistoryStore(),
            significance_window_seconds=60.0,
            significance_check_interval_sec=1e9,
        )
        # Round-trip through the stored form, as save_field/load would.
        self.state = FieldStateV1.model_validate_json(self.state.model_dump_json())
        return self.state


def test_vision_outage_after_alarm_does_not_read_as_recovery() -> None:
    f = _Field()
    f.write(VISION_NODE, {"vision_frame_staleness": 1.0, "vision_processing_failure_pressure": 0.0}, NOW)
    s = f.tick(NOW + timedelta(seconds=5))
    vision = s.capability_vectors["capability:vision"]
    assert vision["pressure"] == 0.85
    assert s.capability_provenance["capability:vision"]["pressure"] == VISION_NODE

    # Router dies: no write for > 300 s, both channels expire.
    s = f.tick(NOW + timedelta(seconds=301))
    vision = s.capability_vectors["capability:vision"]
    for ch in ("pressure", "reliability_pressure", *DERIVED):
        assert ch not in vision, f"{ch} fabricated for an unmeasured eye: {vision}"
    assert s.capability_provenance.get("capability:vision", {}) == {}

    # Still absent on the next tick, after reconcile re-seeded the defaults.
    s = f.tick(NOW + timedelta(seconds=303))
    assert "pressure" not in s.capability_vectors["capability:vision"]
    assert "confidence" not in s.capability_vectors["capability:vision"]


def test_vision_reading_returns_when_the_router_does() -> None:
    f = _Field()
    s = f.tick(NOW)
    assert "pressure" not in s.capability_vectors["capability:vision"]  # never reported yet
    f.write(VISION_NODE, {"vision_frame_staleness": 0.0}, NOW + timedelta(seconds=10))
    s = f.tick(NOW + timedelta(seconds=11))
    vision = s.capability_vectors["capability:vision"]
    assert vision["pressure"] == 0.0  # a measured calm, attributed
    assert s.capability_provenance["capability:vision"]["pressure"] == VISION_NODE
    assert vision["confidence"] == 1.0 and vision["available_capacity"] == 1.0
    # reliability has no reading (nothing dispatched): absent, not 0.0
    assert "reliability_pressure" not in vision


def test_storage_writer_outage_drops_reliability_but_keeps_measured_load() -> None:
    f = _Field()
    f.write("node:athena", {"disk_pressure": 0.4}, NOW)
    f.write(STORAGE_NODE, {"write_failure_pressure": 0.0}, NOW)
    s = f.tick(NOW + timedelta(seconds=5))
    storage = s.capability_vectors["capability:storage"]
    assert storage["reliability_pressure"] == 0.0
    assert s.capability_provenance["capability:storage"]["reliability_pressure"] == STORAGE_NODE

    f.write("node:athena", {"disk_pressure": 0.4}, NOW + timedelta(seconds=180))
    s = f.tick(NOW + timedelta(seconds=181))
    storage = s.capability_vectors["capability:storage"]
    assert "reliability_pressure" not in storage
    # athena still measures disk, so load and its derived channels stay real.
    assert abs(storage["pressure"] - 0.3) < 1e-9
    assert abs(storage["available_capacity"] - 0.7) < 1e-9
    assert s.capability_provenance["capability:storage"]["pressure"] == "node:athena"


def test_rpc_outage_leaves_transport_reliability_to_the_remaining_source() -> None:
    """Partial coverage, pinned on purpose: transport reliability is a max over
    two sources. When the RPC bridge expires, node:athena's observer channel
    still measures it, so the channel stays present and attributed to athena --
    a lower bound, honestly labelled, not an unmeasured value."""
    f = _Field()
    f.write("node:athena", {"observer_failure_pressure": 0.0}, NOW)
    f.write(RPC_NODE, {"rpc_timeout_pressure": 0.6}, NOW)
    s = f.tick(NOW + timedelta(seconds=5))
    assert s.capability_provenance["capability:transport"]["reliability_pressure"] == RPC_NODE

    f.write("node:athena", {"observer_failure_pressure": 0.0}, NOW + timedelta(seconds=120))
    s = f.tick(NOW + timedelta(seconds=121))
    assert s.capability_vectors["capability:transport"]["reliability_pressure"] == 0.0
    assert s.capability_provenance["capability:transport"]["reliability_pressure"] == "node:athena"


def test_unmeasured_capability_is_not_read_as_alarm_or_as_perfect_by_generic_consumers() -> None:
    """The two over/under-index traps, checked on the real generic readers.

    Every measured capability sits at a calm pressure 0.0 (confidence 1.0), so
    on main the eye's fabricated confidence 1.0 TIED them in the merged min()
    and could be named as its source. Under-index: the dark eye must carry no
    confidence and never be the merged source. Over-index (option b, rejected):
    the attention pressure proxy must not jump for it.
    """
    calm = {
        "node:athena": {"cpu_pressure": 0.0, "disk_pressure": 0.0, "memory_pressure": 0.0},
        "node:circe": {"gpu_pressure": 0.0, "memory_pressure": 0.0},
        "node:prometheus": {"cpu_pressure": 0.0},
        "node:substrate.bus_synaptic": {"prediction_error": 0.0},
    }
    f = _Field()
    f.write(VISION_NODE, {"vision_frame_staleness": 0.0}, NOW)
    later = NOW + timedelta(seconds=301)
    for node, chans in calm.items():
        f.write(node, chans, later)
    s = f.tick(later)
    vision = s.capability_vectors["capability:vision"]
    assert "confidence" not in vision and "available_capacity" not in vision
    _, provenance = collect_field_channel_pressures(s)
    for ch in ("confidence", "available_capacity", "pressure"):
        assert provenance.get(ch) != "capability:vision", ch
    assert _current_pressure_proxy(vision) == 0.0


def test_unmeasured_upstream_capability_is_not_a_measured_zero_downstream() -> None:
    """cap->cap edge (llm_inference -> orchestration pressure): reconcile
    re-seeds llm_inference.pressure to 0.0 after diffusion dropped it, so key
    presence must not count as a measurement of the upstream capability."""
    from app.digestion.diffusion import apply_diffusion
    from orion.schemas.field_state import FieldEdgeV1

    state = FieldStateV1(
        generated_at=NOW,
        tick_id="t",
        node_vectors={"node:circe": {}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.0}},  # reconcile seed
        capability_provenance={"capability:llm_inference": {}},
        edges=[
            FieldEdgeV1(source_id="node:circe", target_id="capability:llm_inference", edge_type="node_capability",
                        weight=0.85, channel_map={"gpu_pressure": "pressure"}),
            FieldEdgeV1(source_id="capability:llm_inference", target_id="capability:orchestration",
                        edge_type="capability_capability", weight=0.6, channel_map={"pressure": "pressure"}),
        ],
    )
    apply_diffusion(state, diffusion_rate=1.0)
    assert "pressure" not in state.capability_vectors["capability:orchestration"]
    assert "pressure" not in state.capability_provenance.get("capability:orchestration", {})


def test_direct_measured_zero_confidence_survives_unmeasured_pressure() -> None:
    from app.digestion.diffusion import apply_diffusion
    from orion.schemas.field_state import FieldEdgeV1

    state = FieldStateV1(
        generated_at=NOW,
        tick_id="t",
        node_vectors={"node:x": {"conf_src": 0.0}},
        edges=[FieldEdgeV1(source_id="node:x", target_id="capability:c", edge_type="node_capability", weight=1.0,
                           channel_map={"press_src": "pressure", "conf_src": "confidence"})],
    )
    apply_diffusion(state, diffusion_rate=1.0)
    cap = state.capability_vectors["capability:c"]
    assert "pressure" not in cap
    assert cap["confidence"] == 0.0
    assert state.capability_provenance["capability:c"]["confidence"] == "node:x"
