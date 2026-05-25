from pathlib import Path

from app.graph.lattice import load_lattice

REPO = Path(__file__).resolve().parents[1]


def test_canonical_and_alias_lattice_load_same_edges() -> None:
    canonical = load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")
    alias = load_lattice(REPO / "config" / "field" / "biometrics_lattice.yaml")
    assert len(canonical.edges) == len(alias.edges)
    athena_orch = [
        e
        for e in canonical.edges
        if e.source_id == "node:athena" and e.target_id == "capability:orchestration"
    ]
    assert len(athena_orch) == 1
    assert "execution_load" in athena_orch[0].channel_map
    assert athena_orch[0].channel_map["execution_load"] == "execution_pressure"
