from __future__ import annotations

import sys
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GRAPHDB_URL", "http://example.test")
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVICE_ROOT))
RDF_BUILDER_SPEC = importlib.util.spec_from_file_location("rdf_writer_builder", SERVICE_ROOT / "app" / "rdf_builder.py")
SETTINGS_SPEC = importlib.util.spec_from_file_location("rdf_writer_settings", SERVICE_ROOT / "app" / "settings.py")
assert RDF_BUILDER_SPEC and RDF_BUILDER_SPEC.loader
assert SETTINGS_SPEC and SETTINGS_SPEC.loader
rdf_builder_mod = importlib.util.module_from_spec(RDF_BUILDER_SPEC)
settings_mod = importlib.util.module_from_spec(SETTINGS_SPEC)
RDF_BUILDER_SPEC.loader.exec_module(rdf_builder_mod)
SETTINGS_SPEC.loader.exec_module(settings_mod)
build_triples_from_envelope = rdf_builder_mod.build_triples_from_envelope

"""
Regression guards for the 2026-07-18 Fuseki-kill audit: 4 channels/kinds
found to have no real live traffic (either zero producers anywhere in the
repo, or a real producer whose data is already durably owned elsewhere --
Postgres for identity/goals, or config-disabled for world-pulse-graph).
None of these have a Falkor migration counterpart because there is nothing
to migrate. See docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md
Phase 1 findings.
"""


def _channels() -> list[str]:
    return settings_mod.Settings(ORION_BUS_URL="redis://example.test/0").get_all_subscribe_channels()


def test_rdf_worker_channel_not_subscribed():
    """orion:rdf:worker: channels.yaml claims orion-cortex-exec as producer
    but no code anywhere actually publishes to it."""
    assert "orion:rdf:worker" not in _channels()


def test_cortex_telemetry_channel_not_subscribed():
    """orion:cortex:telemetry: zero producers anywhere, not even in
    channels.yaml's registry."""
    assert "orion:cortex:telemetry" not in _channels()


def test_memory_identity_snapshot_channel_not_subscribed():
    """orion:memory:identity:snapshot: zero live Fuseki triples ever
    recorded; Postgres (orion-self-state-runtime's SelfStateRuntimeStore)
    already durably owns identity snapshots."""
    assert "orion:memory:identity:snapshot" not in _channels()


def test_memory_goals_proposed_channel_not_subscribed():
    """orion:memory:goals:proposed: zero live Fuseki triples ever recorded;
    orion-substrate-runtime's goal_context_listener.py already consumes
    goal proposals for live active-goal state."""
    assert "orion:memory:goals:proposed" not in _channels()


def test_world_pulse_graph_channel_not_subscribed():
    """orion:world_pulse:graph:upsert: WORLD_PULSE_GRAPH_ENABLED was false
    in the live .env -- fully inert despite real producer/dispatch code
    existing."""
    assert "orion:world_pulse:graph:upsert" not in _channels()


def test_drives_audit_channel_not_in_default_subscriptions():
    """rdf-writer must not subscribe to orion:memory:drives:audit by
    default (2026-07-15 kill, unrelated to this session's changes -- kept
    as a standing regression guard)."""
    assert "orion:memory:drives:audit" not in _channels()


def test_cortex_worker_rdf_build_kind_is_quiet_noop():
    """cortex.worker.rdf_build (would-be CognitiveStepExecution) must not
    resurrect a dispatch branch -- unknown kind falls through to (None, None)."""
    nt, graph_name = build_triples_from_envelope("cortex.worker.rdf_build", {"correlation_id": "c-1", "verb": "test"})
    assert nt is None
    assert graph_name is None


def test_identity_snapshot_kind_is_quiet_noop():
    nt, graph_name = build_triples_from_envelope("memory.identity.snapshot.v1", {"artifact_id": "identity-1"})
    assert nt is None
    assert graph_name is None


def test_goals_proposed_kind_is_quiet_noop():
    nt, graph_name = build_triples_from_envelope("memory.goals.proposed.v1", {"artifact_id": "goal-1"})
    assert nt is None
    assert graph_name is None


def test_world_pulse_graph_upsert_kind_is_quiet_noop():
    nt, graph_name = build_triples_from_envelope(
        "world.pulse.graph.upsert.v1",
        {
            "graph_delta_id": "graph:run-1",
            "run_id": "run-1",
            "graph_name": "http://conjourney.net/graph/world-pulse",
            "triples": "",
            "summary": "0 item deltas",
            "created_at": "2026-03-19T12:00:00+00:00",
        },
    )
    assert nt is None
    assert graph_name is None
