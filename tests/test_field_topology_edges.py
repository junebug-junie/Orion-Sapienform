"""Static gate on config/field/orion_field_topology.v1.yaml.

Plain YAML, no service imports, so it runs from the repo root in
orion-static-gates CI. (The test it replaces, test_field_topology_config.py,
imported `app.graph.lattice` and could never be collected from the root, so
it guarded nothing for months.)

Why each check exists:

- A capability->capability edge reads its source channel off the SOURCE
  capability's vector. Capability vectors only get real values from edges
  pointing into them (services/orion-field-digester/app/digestion/
  diffusion.py), so a cap->cap edge whose source channel no inbound edge
  writes reads a seeded 0.0 forever. That is exactly the
  capability:transport -> capability:orchestration edge deleted 2026-09-25
  (stream_backlog_pressure, 0.0 on 121,114 of 121,114 live ticks).
- confidence/available_capacity are the one exception: apply_diffusion()
  derives them from `pressure` for every capability that is the target of any
  edge (reconcile always seeds `pressure`), so any inbound edge counts.
- One hop only: a channel written by an edge whose own source is dead still
  passes. That case needs live data, not YAML.
- The biometrics_lattice.yaml alias was deleted the same day. It fell three
  edges behind the canonical file without anything noticing; a second copy of
  the topology is how that happens again.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TOPOLOGY = REPO / "config" / "field" / "orion_field_topology.v1.yaml"

# Written by apply_diffusion()'s derived fallback from `pressure`, not by an edge.
_DERIVED_FROM_PRESSURE = {"confidence", "available_capacity"}


def _edges() -> list[dict]:
    return yaml.safe_load(TOPOLOGY.read_text(encoding="utf-8"))["edges"]


def _written_channels(edges: list[dict]) -> dict[str, set[str]]:
    written: dict[str, set[str]] = {}
    for edge in edges:
        written.setdefault(edge["target_id"], set()).update(edge["channel_map"].values())
    for channels in written.values():
        channels.update(_DERIVED_FROM_PRESSURE)
    return written


def dead_cap_cap_channels(edges: list[dict]) -> list[str]:
    written = _written_channels(edges)
    dead = []
    for edge in edges:
        if edge["edge_type"] != "capability_capability":
            continue
        for src_ch in edge["channel_map"]:
            if src_ch not in written.get(edge["source_id"], set()):
                dead.append(f"{edge['source_id']}.{src_ch} -> {edge['target_id']}")
    return dead


def test_every_cap_cap_edge_reads_a_channel_something_writes() -> None:
    assert dead_cap_cap_channels(_edges()) == []


def test_gate_catches_the_edge_it_was_written_for() -> None:
    """Mutation check: re-adding the deleted edge must fail the gate."""
    edges = _edges() + [
        {
            "source_id": "capability:transport",
            "target_id": "capability:orchestration",
            "edge_type": "capability_capability",
            "weight": 0.70,
            "channel_map": {"stream_backlog_pressure": "stream_backlog_pressure"},
        }
    ]
    assert dead_cap_cap_channels(edges) == [
        "capability:transport.stream_backlog_pressure -> capability:orchestration"
    ]


def test_gate_allows_a_live_channel_and_derived_confidence() -> None:
    edges = _edges() + [
        {
            "source_id": "capability:transport",
            "target_id": "capability:orchestration",
            "edge_type": "capability_capability",
            "weight": 0.70,
            "channel_map": {"pressure": "pressure", "confidence": "confidence"},
        }
    ]
    assert dead_cap_cap_channels(edges) == []


def test_orchestration_edge_keeps_its_live_channels_and_drops_the_census() -> None:
    athena_orch = [
        e
        for e in _edges()
        if e["source_id"] == "node:athena" and e["target_id"] == "capability:orchestration"
    ]
    assert len(athena_orch) == 1
    channel_map = athena_orch[0]["channel_map"]
    assert channel_map["cortex_exec_step_load"] == "execution_pressure"
    assert channel_map["cpu_pressure"] == "pressure"
    assert "stream_backlog_pressure" not in channel_map


def test_no_transport_to_orchestration_edge_on_the_dead_channel() -> None:
    for edge in _edges():
        if edge["source_id"] == "capability:transport":
            assert "stream_backlog_pressure" not in edge["channel_map"]


def test_topology_alias_is_gone_and_unreferenced() -> None:
    assert not (REPO / "config" / "field" / "biometrics_lattice.yaml").exists()
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "biometrics_lattice.yaml",
            "--",
            "*.py",
            "*.yaml",
            "*.yml",
            "*.env_example",
            ":!docs/**",
            ":!tests/test_field_topology_edges.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    # 0 = matches, 1 = no matches; anything else (e.g. 128, no work tree)
    # must not pass silently as "no references".
    assert proc.returncode in (0, 1), proc.stderr
    assert proc.stdout.split() == []
