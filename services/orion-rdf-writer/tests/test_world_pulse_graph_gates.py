from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GRAPHDB_URL", "http://example.test")
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

RDF_BUILDER_SPEC = importlib.util.spec_from_file_location("rdf_writer_builder", SERVICE_ROOT / "app" / "rdf_builder.py")
assert RDF_BUILDER_SPEC and RDF_BUILDER_SPEC.loader
rdf_builder_mod = importlib.util.module_from_spec(RDF_BUILDER_SPEC)
RDF_BUILDER_SPEC.loader.exec_module(rdf_builder_mod)
build_triples_from_envelope = rdf_builder_mod.build_triples_from_envelope


def _delta_payload(*, include_policy_stamp: bool) -> dict:
    payload = {
        "graph_delta_id": "graph:run-1",
        "run_id": "run-1",
        "graph_name": "http://conjourney.net/graph/world-pulse",
        "triples": "<urn:orion:world-pulse:item:i1> <http://conjourney.net/orion#title> \"x\" .",
        "summary": "1 item deltas",
        "triple_count": 1,
        "dry_run": True,
        "created_at": "2026-04-26T00:00:00+00:00",
    }
    if include_policy_stamp:
        payload["policy_stamp"] = {"graph_write_only_for_trust_tier_lte_3": True}
    return payload


def test_graph_emit_blocked_when_world_pulse_graph_disabled() -> None:
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_ENABLED = False
    triples, graph_name = build_triples_from_envelope("world.pulse.graph.upsert.v1", _delta_payload(include_policy_stamp=True))
    assert triples is None
    assert graph_name is None


def test_graph_emit_requires_policy_stamp_when_enabled() -> None:
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_ENABLED = True
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_REQUIRE_POLICY_STAMP = True
    triples, graph_name = build_triples_from_envelope("world.pulse.graph.upsert.v1", _delta_payload(include_policy_stamp=False))
    assert triples is None
    assert graph_name is None


def test_graph_dry_run_payload_remains_inspectable() -> None:
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_ENABLED = True
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_REQUIRE_POLICY_STAMP = True
    rdf_builder_mod.settings.WORLD_PULSE_GRAPH_DRY_RUN = True
    triples, graph_name = build_triples_from_envelope("world.pulse.graph.upsert.v1", _delta_payload(include_policy_stamp=True))
    assert isinstance(triples, str)
    assert "world-pulse:item" in triples
    assert graph_name == "http://conjourney.net/graph/world-pulse"
