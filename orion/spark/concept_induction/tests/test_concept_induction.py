from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import patch

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.clusterer import ConceptClusterer
from orion.spark.concept_induction.drives import DriveEngine
from orion.spark.concept_induction.embedder import EmbeddingClient
from orion.spark.concept_induction.extractor import SpacyConceptExtractor
from orion.spark.concept_induction.inducer import ConceptInducer, WindowEvent
from orion.spark.concept_induction.settings import ConceptSettings
from orion.spark.concept_induction.store import LocalProfileStore
from orion.spark.concept_induction.tensions import extract_tensions
from orion.spark.concept_induction.bus_worker import ConceptWorker


def _env_with_text(text: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.message",
        source=ServiceRef(name="test", version="0.0.0"),
        payload={"content": text},
    )


class FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


class FailOnKindBus(FakeBus):
    def __init__(self, *, fail_channel: str, fail_kind: str):
        super().__init__()
        self.fail_channel = fail_channel
        self.fail_kind = fail_kind

    async def publish(self, channel, env):
        self.published.append((channel, env))
        if channel == self.fail_channel and env.kind == self.fail_kind:
            raise RuntimeError("graph write unavailable")


class StubInducer:
    def __init__(self, *, profile, save_hook=None):
        self.profile = profile
        self.save_hook = save_hook

    async def run(self, *, subject: str, window):
        if self.save_hook:
            self.save_hook(subject, self.profile)
        return type("Result", (), {"profile": self.profile, "delta": None})()


class ConceptInductionTests(unittest.TestCase):
    def test_spacy_extraction_basic(self):
        extractor = SpacyConceptExtractor(model_name="en_core_web_sm")
        res = extractor.extract(["Orion talked with Juniper about concept induction."])
        self.assertTrue(any("orion" in c for c in res.candidates))

    def test_embedding_client_mock(self):
        calls = []

        def _fake_post(url, json, timeout):
            calls.append({"url": url, "payload": json, "timeout": timeout})

            class Resp:
                def raise_for_status(self): ...

                def json(self):
                    return {"embedding": [1.0, 0.0], "embedding_model": "test", "embedding_dim": 2}

            return Resp()

        with patch("requests.post", _fake_post):
            client = EmbeddingClient("http://fake")
            resp = client.embed(["alpha", "beta"])
            self.assertEqual(resp.embeddings["alpha"], [1.0, 0.0])
            self.assertEqual(resp.embeddings["beta"], [1.0, 0.0])
            self.assertEqual(calls[0]["url"], "http://fake/embedding")
            self.assertEqual(calls[0]["payload"]["text"], "alpha")
            self.assertTrue(calls[0]["payload"]["doc_id"].startswith("concept-"))
            self.assertEqual(calls[0]["payload"]["embedding_profile"], "default")
            self.assertFalse(calls[0]["payload"]["include_latent"])

    def test_clusterer_deterministic(self):
        clusterer = ConceptClusterer(threshold=0.5)
        res = clusterer.cluster(["orion self", "orion self", "juniper friend"], embeddings=None)
        self.assertEqual(len(res.clusters), 2)

    def test_profile_and_delta_generation(self):
        settings = ConceptSettings()
        inducer = ConceptInducer(settings)
        now = datetime.now(timezone.utc)
        window = [
            WindowEvent(text="Orion reflected with Juniper about the lake.", timestamp=now, envelope=_env_with_text("x"), intake_channel="orion:chat:history:log"),
            WindowEvent(text="Juniper shared new plans for Orion", timestamp=now + timedelta(seconds=1), envelope=_env_with_text("y"), intake_channel="orion:chat:history:log"),
        ]
        result = asyncio.run(inducer.run(subject="relationship", window=window))
        self.assertTrue(result.profile.concepts)
        result2 = asyncio.run(inducer.run(subject="relationship", window=window))
        self.assertGreaterEqual(result2.profile.revision, result.profile.revision)

    def test_store_round_trip_without_hash_field(self):
        settings = ConceptSettings()
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "store.json"
            store = LocalProfileStore(store_path)
            inducer = ConceptInducer(settings, store_loader=store.load, store_saver=store.save)
            now = datetime.now(timezone.utc)
            window = [
                WindowEvent(text="Orion reflected with Juniper about the lake.", timestamp=now, envelope=_env_with_text("x"), intake_channel="orion:chat:history:log"),
            ]
            result = asyncio.run(inducer.run(subject="orion", window=window))
            self.assertTrue(result.profile.concepts)
            reloaded = store.load("orion")
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.subject, "orion")
            self.assertEqual(store.load_hash("orion"), store.load_hash("orion"))

    def test_provenance_channel_preserved_in_evidence(self):
        settings = ConceptSettings()
        inducer = ConceptInducer(settings)
        env = BaseEnvelope(
            kind="chat.message",
            source=ServiceRef(name="test", version="0.0.0"),
            payload={"content": "Orion reflected."},
            trace={"trace_id": "trace-123"},
        )
        now = datetime.now(timezone.utc)
        result = asyncio.run(
            inducer.run(
                subject="orion",
                window=[WindowEvent(text="Orion reflected.", timestamp=now, envelope=env, intake_channel="orion:collapse:sql-write")],
            )
        )
        ev = result.profile.concepts[0].evidence[0]
        self.assertEqual(ev.channel, "orion:collapse:sql-write")
        self.assertEqual(ev.trace_id, "trace-123")

    def test_turn_effect_tension_mapping(self):
        env = BaseEnvelope(
            kind="spark.telemetry",
            source=ServiceRef(name="test", version="0.0.0"),
            payload={
                "spark_meta": {
                    "turn_effect": {
                        "turn": {
                            "coherence": -0.4,
                            "valence": -0.2,
                            "novelty": 0.7,
                            "energy": -0.6,
                        }
                    }
                }
            },
        )
        events = extract_tensions(
            envelope=env,
            intake_channel="orion:spark:telemetry",
            subject="orion",
            model_layer="self-model",
            entity_id="self:orion",
        )
        kinds = {e.kind for e in events}
        self.assertEqual(kinds, {
            "tension.contradiction.v1",
            "tension.distress.v1",
            "tension.identity_drift.v1",
            "tension.cognitive_load.v1",
        })
        self.assertTrue(all(event.provenance.source_event_refs for event in events))
        self.assertTrue(all(event.correlation_id for event in events))

    def test_drive_update_is_deterministic_and_decays(self):
        engine = DriveEngine()
        tension_env = BaseEnvelope(
            kind="spark.telemetry",
            source=ServiceRef(name="test", version="0.0.0"),
            payload={"turn_effect": {"turn": {"coherence": -0.5}}},
        )
        tensions = extract_tensions(
            envelope=tension_env,
            intake_channel="orion:spark:telemetry",
            subject="orion",
            model_layer="self-model",
            entity_id="self:orion",
        )
        now = datetime.now(timezone.utc)
        p1, a1 = engine.update(previous_pressures={}, previous_activations={}, tensions=tensions, now=now, previous_ts=None)
        p2, a2 = engine.update(previous_pressures={}, previous_activations={}, tensions=tensions, now=now, previous_ts=None)
        self.assertEqual(p1, p2)
        self.assertEqual(a1, a2)
        p3, _ = engine.update(previous_pressures=p1, previous_activations=a1, tensions=[], now=now + timedelta(hours=2), previous_ts=now)
        self.assertLess(p3["coherence"], p1["coherence"])

    def test_restart_persistence_for_drive_state(self):
        with tempfile.TemporaryDirectory() as td:
            store = LocalProfileStore(str(Path(td) / "store.json"))
            now = datetime.now(timezone.utc)
            store.save_drive_state("orion", pressures={"coherence": 0.8}, activations={"coherence": True}, updated_at=now)
            reloaded = LocalProfileStore(str(Path(td) / "store.json")).load_drive_state("orion")
            self.assertEqual(reloaded.get("pressures", {}).get("coherence"), 0.8)

    def test_world_identity_hardening_avoids_generic_world_anchor(self):
        worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False))
        env = BaseEnvelope(
            kind="system.health",
            source=ServiceRef(name="telemetry-hub", version="0.0.0"),
            payload={"subject": "world", "service": "Auth API"},
        )
        subject = worker._detect_subject(env, "orion:system:health")
        model_layer = worker._model_layer(subject, "orion:system:health")
        entity_id = worker._entity_id(subject, model_layer)
        self.assertEqual(subject, "service:auth-api")
        self.assertEqual(model_layer, "world-model")
        self.assertEqual(entity_id, "world:service:auth-api")
        self.assertNotEqual(subject, "world")

    def test_juniper_identity_typing_and_world_layer_no_collapse(self):
        worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False))
        env = BaseEnvelope(
            kind="chat.message",
            source=ServiceRef(name="test", version="0.0.0"),
            payload={"user": "Juniper"},
        )
        self.assertEqual(worker._detect_subject(env), "juniper")
        self.assertEqual(worker._model_layer("juniper", "orion:spark:telemetry"), "user-model")
        self.assertEqual(worker._model_layer("service:vector-host", "orion:system:health"), "world-model")

    def test_memory_drives_state_publication_and_join_ref_hardening(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ConceptWorker(
                ConceptSettings(
                    orion_bus_enabled=False,
                    use_cortex_orch=False,
                    store_path=str(Path(td) / "state.json"),
                    goal_proposal_cooldown_minutes=180,
                )
            )
            fake_bus = FakeBus()
            worker.bus = fake_bus
            corr_id = uuid4()
            env = BaseEnvelope(
                kind="metacognition.tick.v1",
                source=ServiceRef(name="test", version="0.0.0"),
                correlation_id=corr_id,
                trace={"trace_id": "trace-xyz"},
                payload={
                    "id": "turn-123",
                    "subject": "orion",
                    "turn_effect": {"turn": {"coherence": -0.3, "novelty": 0.5}},
                    "summary": "Detected instability in recent turn.",
                },
            )
            asyncio.run(worker.handle_envelope(env, "orion:metacognition:tick"))

            by_kind = {published_env.kind: published_env.payload for _, published_env in fake_bus.published}
            self.assertIn("memory.drives.state.v1", by_kind)
            self.assertIn("memory.drives.audit.v1", by_kind)
            self.assertIn("memory.identity.snapshot.v1", by_kind)
            self.assertIn("memory.goals.proposed.v1", by_kind)
            self.assertIn("debug.turn.dossier.v1", by_kind)

            drive_audit = by_kind["memory.drives.audit.v1"]
            self.assertEqual(drive_audit["correlation_id"], str(corr_id))
            self.assertEqual(drive_audit["trace_id"], "trace-xyz")
            self.assertEqual(drive_audit["turn_id"], "turn-123")
            self.assertTrue(drive_audit["provenance"]["source_event_refs"])
            self.assertTrue(drive_audit["provenance"]["tension_refs"])
            self.assertTrue(drive_audit["evidence_items"])

            identity_snapshot = by_kind["memory.identity.snapshot.v1"]
            self.assertTrue(identity_snapshot["anchor_strategy"])
            self.assertTrue(identity_snapshot["provenance"]["source_event_refs"])
            self.assertTrue(identity_snapshot["provenance"]["tension_refs"])

            goal = by_kind["memory.goals.proposed.v1"]
            self.assertEqual(goal["correlation_id"], str(corr_id))
            self.assertTrue(goal["proposal_signature"])
            self.assertTrue(goal["provenance"]["tension_refs"])
            self.assertTrue(goal["source_event_refs"])

            dossier = by_kind["debug.turn.dossier.v1"]
            self.assertEqual(dossier["trace_id"], "trace-xyz")
            self.assertEqual(dossier["turn_id"], "turn-123")
            self.assertTrue(dossier["drive_audit_ref"])
            self.assertTrue(dossier["identity_snapshot_ref"])

            self.assertFalse(any("exec" in published_env.kind for _, published_env in fake_bus.published))

    def test_goal_proposal_dedupe_cooldown_suppresses_repeats(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ConceptWorker(
                ConceptSettings(
                    orion_bus_enabled=False,
                    use_cortex_orch=False,
                    store_path=str(Path(td) / "state.json"),
                    goal_proposal_cooldown_minutes=180,
                )
            )
            fake_bus = FakeBus()
            worker.bus = fake_bus
            payload = {
                "subject": "orion",
                "turn_effect": {"turn": {"coherence": -0.45}},
                "summary": "Repeated coherence drop.",
            }
            env1 = BaseEnvelope(kind="metacognition.tick.v1", source=ServiceRef(name="test", version="0.0.0"), correlation_id=uuid4(), payload=payload)
            env2 = BaseEnvelope(kind="metacognition.tick.v1", source=ServiceRef(name="test", version="0.0.0"), correlation_id=uuid4(), payload=payload)
            asyncio.run(worker.handle_envelope(env1, "orion:metacognition:tick"))
            first_goal_count = sum(1 for _, published_env in fake_bus.published if published_env.kind == "memory.goals.proposed.v1")
            asyncio.run(worker.handle_envelope(env2, "orion:metacognition:tick"))
            second_goal_count = sum(1 for _, published_env in fake_bus.published if published_env.kind == "memory.goals.proposed.v1")
            dossier_payloads = [published_env.payload for _, published_env in fake_bus.published if published_env.kind == "debug.turn.dossier.v1"]
            self.assertEqual(first_goal_count, 1)
            self.assertEqual(second_goal_count, 1)
            self.assertTrue(dossier_payloads[-1]["suppressed_goal_signatures"])
            cooldowns = worker.store.load_goal_cooldown(dossier_payloads[-1]["suppressed_goal_signatures"][0])
            self.assertGreaterEqual(cooldowns.get("suppressed_count", 0), 1)

    def test_run_for_subject_invokes_graph_materialization(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False, store_path=str(Path(td) / "state.json")))
            profile = asyncio.run(
                ConceptInducer(ConceptSettings()).run(
                    subject="orion",
                    window=[
                        WindowEvent(
                            text="Orion keeps coherence.",
                            timestamp=datetime.now(timezone.utc),
                            envelope=_env_with_text("Orion keeps coherence."),
                            intake_channel="orion:chat:history:log",
                        )
                    ],
                )
            ).profile
            worker.inducer = StubInducer(profile=profile)
            worker.bus = FakeBus()
            worker.window["orion"] = [
                WindowEvent(
                    text="Orion keeps coherence.",
                    timestamp=datetime.now(timezone.utc),
                    envelope=_env_with_text("Orion keeps coherence."),
                    intake_channel="orion:chat:history:log",
                )
            ]
            asyncio.run(worker.run_for_subject("orion"))

            published = worker.bus.published
            assert any(channel == worker.cfg.forward_rdf_channel and env.kind == "rdf.write.request" for channel, env in published)
            assert any(env.kind == "memory.concepts.profile.v1" for _, env in published)

    def test_graph_materialization_failure_isolated_from_local_write(self):
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "state.json"
            worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False, store_path=str(store_path)))
            profile = asyncio.run(
                ConceptInducer(ConceptSettings()).run(
                    subject="orion",
                    window=[
                        WindowEvent(
                            text="Orion forms concepts.",
                            timestamp=datetime.now(timezone.utc),
                            envelope=_env_with_text("Orion forms concepts."),
                            intake_channel="orion:chat:history:log",
                        )
                    ],
                )
            ).profile

            def _save_local(subject: str, p):
                worker.store.save(subject, p, "hash-local-test")

            worker.inducer = StubInducer(profile=profile, save_hook=_save_local)
            worker.bus = FailOnKindBus(fail_channel=worker.cfg.forward_rdf_channel, fail_kind="rdf.write.request")
            worker.window["orion"] = [
                WindowEvent(
                    text="Orion forms concepts.",
                    timestamp=datetime.now(timezone.utc),
                    envelope=_env_with_text("Orion forms concepts."),
                    intake_channel="orion:chat:history:log",
                )
            ]

            asyncio.run(worker.run_for_subject("orion"))

            reloaded = worker.store.load("orion")
            assert reloaded is not None
            assert reloaded.subject == "orion"
            assert any(env.kind == "memory.concepts.profile.v1" for _, env in worker.bus.published)

    def test_trigger_source_mapping(self):
        worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False))
        env_chat = BaseEnvelope(kind="chat.message", source=ServiceRef(name="test", version="0.0.0"), payload={"content": "hello"})
        env_dream = BaseEnvelope(kind="dream.result.v1", source=ServiceRef(name="dream", version="0.0.0"), payload={"summary": "dream"})
        env_journal = BaseEnvelope(kind="journal.entry.created.v1", source=ServiceRef(name="journal", version="0.0.0"), payload={"body": "entry"})
        self.assertEqual(worker._source_kind(env_chat, "orion:chat:history:log"), "chat_turn")
        self.assertEqual(worker._source_kind(env_dream, "orion:dream:complete"), "dream_result")
        self.assertEqual(worker._source_kind(env_journal, "orion:journal:created"), "journal_write")

    def test_deterministic_subject_selection(self):
        worker = ConceptWorker(ConceptSettings(orion_bus_enabled=False))
        env = BaseEnvelope(
            kind="chat.message",
            source=ServiceRef(name="orion-cortex", version="0.0.0"),
            payload={"role": "user", "user": "Juniper", "content": "Orion, let's reflect together"},
        )
        subjects = worker._select_trigger_subjects(
            env,
            "orion:chat:social:stored",
            "chat_turn",
            "Orion and Juniper reflected together",
        )
        self.assertEqual(subjects, ["orion", "juniper", "relationship"])

    def test_cooldown_and_dedupe_suppression(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ConceptWorker(
                ConceptSettings(
                    orion_bus_enabled=False,
                    store_path=str(Path(td) / "state.json"),
                    concept_trigger_cooldown_sec=600,
                    concept_trigger_dedupe_sec=120,
                )
            )
            worker.bus = FakeBus()
            env = BaseEnvelope(
                kind="chat.message",
                source=ServiceRef(name="test", version="0.0.0"),
                payload={"content": "Orion and Juniper discussed plans.", "role": "assistant"},
            )
            asyncio.run(worker.handle_envelope(env, "orion:chat:history:log"))
            asyncio.run(worker.handle_envelope(env, "orion:chat:history:log"))
            decisions = [d["decision"] for d in worker.trigger_decisions]
            self.assertIn("triggered", decisions)
            self.assertIn("coalesced", decisions)

    def test_trigger_invokes_existing_run_for_subject(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ConceptWorker(
                ConceptSettings(orion_bus_enabled=False, store_path=str(Path(td) / "state.json"))
            )
            called = []

            async def _fake_run(subject: str, corr_id=None):
                called.append((subject, corr_id))

            worker.run_for_subject = _fake_run  # type: ignore[method-assign]
            env = BaseEnvelope(
                kind="metacognition.tick.v1",
                source=ServiceRef(name="test", version="0.0.0"),
                correlation_id=uuid4(),
                payload={"subject": "orion", "summary": "self check"},
            )
            asyncio.run(worker.handle_envelope(env, "orion:metacognition:tick"))
            self.assertEqual(called[0][0], "orion")

    def test_observability_status_surface_contains_decisions(self):
        worker = ConceptWorker(
            ConceptSettings(
                orion_bus_enabled=False,
                concept_trigger_recent_decisions=5,
            )
        )
        worker._record_trigger_decision({"decision": "triggered", "subject": "orion"})
        status = worker.trigger_status()
        self.assertEqual(status["cooldown_sec"], worker.cfg.concept_trigger_cooldown_sec)
        self.assertTrue(status["recent_decisions"])


if __name__ == "__main__":
    unittest.main()
