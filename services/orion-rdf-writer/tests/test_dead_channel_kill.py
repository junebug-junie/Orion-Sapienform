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


def test_tags_chat_enriched_channel_not_subscribed():
    """orion:tags:chat:enriched (2026-07-18, same day, later patch): chat/
    social-turn tag+entity enrichment is FalkorDB-only now (orion-meta-tags'
    Phase 2 writer). The Fuseki `chat_tagging` copy this channel fed was a
    redundant second materialization of the same data -- unlike the other 4
    channels above, this one HAD real live traffic (confirmed via a live
    GRAPH.QUERY + matching rdf-writer enqueue log for the same
    correlation_id right before this kill), it just became redundant once
    Falkor covered the same data. Do not re-add."""
    assert "orion:tags:chat:enriched" not in _channels()


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


def test_chat_tagging_enrichment_falls_through_to_generic_event_uri():
    """2026-07-18 (same day, later patch): the dedicated
    enrichment_type=="chat_tagging" subject_uri branch (chatTurn/{id}) was
    removed since orion:tags:chat:enriched is no longer subscribed (see
    test_tags_chat_enriched_channel_not_subscribed above), so this shape can
    no longer reach build_triples_from_envelope via the live bus. This test
    locks in what happens if it ever does anyway (dead-letter replay, a
    stale queued message from before the cutover, a reintroduced producer):
    it now falls through to the generic event/{id} URI scheme rather than
    being rejected -- not a no-op like the other kinds in this file, a real
    behavior change worth having a test name for."""
    nt, graph_name = build_triples_from_envelope(
        "tags.enriched",
        {
            "service_name": "orion-meta-tags",
            "service_version": "0.2.0",
            "id": "turn-1",
            "collapse_id": "turn-1",
            "enrichment_type": "chat_tagging",
            "tags": ["sentiment:neutral"],
            "entities": ["Circe"],
        },
    )
    assert nt is not None
    assert graph_name == "orion:enrichment"
    assert "http://conjourney.net/orion/chatTurn/" not in nt
    assert "http://conjourney.net/event/turn-1" in nt


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
