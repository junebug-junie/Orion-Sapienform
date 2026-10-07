"""Static gate: every transport-lattice policy row names a real reading.

config/substrate-lattice/transport_lattice_policy.v1.yaml row ids are
lane-local labels (and the key of the shared EWMA threshold state); each row's
`source:` block says which real reading it is. Readers (hub lattice routes)
resolve values only through `source`, so this test is what keeps the label and
the metric from drifting apart the way `bus_synaptic_pressure` (policy) and
`pressure` (field) did before 2026-10-07, when the hub kept its own hand-made
mapping between the two.

Plain YAML plus one pydantic model import; runs from the repo root in
orion-static-gates CI.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.transport_projection import (
    RETIRED_TRANSPORT_BUS_STATE_FIELDS,
    TransportBusStateV1,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config/substrate-lattice/transport_lattice_policy.v1.yaml"
TOPOLOGY = REPO / "config/field/orion_field_topology.v1.yaml"
GLOSSARY = REPO / "config/field/field_channel_glossary.v1.yaml"

M4_VECTOR = "capability:transport"
# Derived by apply_diffusion() from `pressure` for any edge target; not written
# by a channel_map entry (see tests/test_field_topology_edges.py).
DERIVED_CAPABILITY_CHANNELS = {"confidence", "available_capacity"}


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _channels() -> dict[str, dict]:
    return _yaml(POLICY)["channels"]


def _channels_written_into(target: str) -> dict[str, list[tuple[str, str, float]]]:
    """channel -> [(source_id, source_channel, weight)] for edges into target."""
    out: dict[str, list[tuple[str, str, float]]] = {}
    for edge in _yaml(TOPOLOGY).get("edges") or []:
        if edge.get("target_id") != target:
            continue
        for src_ch, dst_ch in (edge.get("channel_map") or {}).items():
            out.setdefault(dst_ch, []).append(
                (edge["source_id"], src_ch, float(edge.get("weight", 1.0)))
            )
    return out


def resolve_problems(channels: dict[str, dict]) -> list[str]:
    topo = _yaml(TOPOLOGY)
    cap_channels = set(topo.get("capability_channels") or [])
    written = _channels_written_into(M4_VECTOR)
    glossary_cap = {
        e["channel"]
        for e in _yaml(GLOSSARY).get("channels") or []
        if "node" not in e and "capability" in (e.get("level") or [])
    }
    bus_fields = set(TransportBusStateV1.model_fields)
    problems: list[str] = []
    for row_id, row in channels.items():
        src = (row or {}).get("source")
        if not isinstance(src, dict):
            problems.append(f"{row_id}: no source block")
            continue
        layer = src.get("layer")
        if layer == "m4":
            ch = src.get("channel")
            if src.get("vector") != M4_VECTOR:
                problems.append(f"{row_id}: m4 source must be on {M4_VECTOR}, got {src.get('vector')!r}")
            elif ch not in cap_channels:
                problems.append(f"{row_id}: {ch!r} is not a capability channel in the topology")
            elif ch not in written and ch not in DERIVED_CAPABILITY_CHANNELS:
                problems.append(f"{row_id}: nothing in the topology writes {M4_VECTOR}.{ch}")
            elif ch not in glossary_cap:
                problems.append(f"{row_id}: {ch!r} has no capability-level glossary entry")
        elif layer == "m3":
            field = src.get("field")
            if field in RETIRED_TRANSPORT_BUS_STATE_FIELDS:
                problems.append(f"{row_id}: m3 field {field!r} is retired")
            elif field not in bus_fields:
                problems.append(f"{row_id}: {field!r} is not a TransportBusStateV1 field")
        else:
            problems.append(f"{row_id}: unknown source layer {layer!r}")
    return problems


def test_every_policy_row_resolves_to_a_real_reading() -> None:
    assert resolve_problems(_channels()) == []


def test_gate_catches_a_drifted_source() -> None:
    bad = {
        "a": {"source": {"layer": "m4", "vector": M4_VECTOR, "channel": "not_a_channel"}},
        "b": {"source": {"layer": "m3", "field": "contract_pressure"}},
        "c": {"watch_at": 0.5},
        "d": {"source": {"layer": "m3", "field": "made_up"}},
    }
    problems = resolve_problems(bad)
    assert len(problems) == 4, problems


def test_bus_synaptic_row_is_capability_transport_pressure_from_bus_synaptic() -> None:
    """The row id is a label; the metric is capability:transport.pressure, fed
    only by node:substrate.bus_synaptic prediction_error."""
    src = _channels()["bus_synaptic_pressure"]["source"]
    assert src == {"layer": "m4", "vector": M4_VECTOR, "channel": "pressure"}
    feeders = _channels_written_into(M4_VECTOR)["pressure"]
    assert [(s, c) for s, c, _ in feeders] == [("node:substrate.bus_synaptic", "prediction_error")]


def test_retired_contract_pressure_row_is_gone() -> None:
    assert "contract_pressure" not in _channels()
