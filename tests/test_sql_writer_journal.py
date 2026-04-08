from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer"
sys.path.insert(0, str(SERVICE_ROOT))

from sqlalchemy.exc import IntegrityError  # noqa: E402

from app.models.journal_entry import JournalEntrySQL  # noqa: E402
from app.models.social_room_turn import SocialRoomTurnSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import _write_row, handle_envelope  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))


class _FakeSession:
    def __init__(self, *, duplicate: bool = False) -> None:
        self.duplicate = duplicate
        self.add_calls = []
        self.merge_calls = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def add(self, obj):
        self.add_calls.append(obj)
        if self.duplicate:
            class _Orig:
                pgcode = "23505"
            raise IntegrityError("insert", {}, _Orig())

    def merge(self, obj):
        self.merge_calls.append(obj)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class TestSqlWriterJournal(unittest.TestCase):
    def test_route_map_and_channels_include_journal(self):
        settings = Settings()
        self.assertEqual(DEFAULT_ROUTE_MAP["journal.entry.write.v1"], "JournalEntrySQL")
        self.assertIn("orion:journal:write", settings.effective_subscribe_channels)
        self.assertEqual(DEFAULT_ROUTE_MAP["social.turn.v1"], "SocialRoomTurnSQL")
        self.assertIn("orion:chat:social:turn", settings.effective_subscribe_channels)
        self.assertEqual(DEFAULT_ROUTE_MAP["external.room.message.v1"], "ExternalRoomMessageSQL")
        self.assertEqual(DEFAULT_ROUTE_MAP["external.room.participant.v1"], "ExternalRoomParticipantSQL")
        self.assertIn("orion:bridge:social:room:intake", settings.effective_subscribe_channels)
        self.assertIn("orion:bridge:social:participant", settings.effective_subscribe_channels)
        self.assertTrue(settings.sql_writer_emit_social_turn_stored)
        self.assertEqual(settings.sql_writer_social_turn_stored_channel, "orion:chat:social:stored")

    def test_journal_routing_keeps_schema_payload_clean_and_moves_db_metadata_to_extra_fields(self):
        import app.worker as worker_mod  # noqa: E402

        captured = []

        async def fake_write(sql_model, schema_model, payload, extra_fields):
            captured.append((sql_model.__tablename__, schema_model.__name__, payload, extra_fields))
            return True

        original_write = worker_mod._write
        worker_mod._write = fake_write
        try:
            env = BaseEnvelope(
                kind="journal.entry.write.v1",
                id="00000000-0000-0000-0000-000000000010",
                source=ServiceRef(name="orion-actions", node="writer-node"),
                correlation_id="00000000-0000-0000-0000-000000000011",
                payload={
                    "entry_id": "entry-clean",
                    "created_at": "2026-03-21T12:00:00+00:00",
                    "author": "orion",
                    "mode": "daily",
                    "title": "Morning",
                    "body": "Keep payload strict.",
                    "source_kind": "scheduler",
                    "source_ref": "2026-03-21",
                    "correlation_id": "00000000-0000-0000-0000-000000000011",
                },
            )
            asyncio.run(handle_envelope(env, bus=_FakeBus()))
        finally:
            worker_mod._write = original_write

        self.assertEqual(captured[0][0], "journal_entries")
        self.assertEqual(captured[0][1], "JournalEntryWriteV1")
        self.assertNotIn("node", captured[0][2])
        self.assertNotIn("source_message_id", captured[0][2])
        self.assertEqual(captured[0][3]["node"], "writer-node")
        self.assertEqual(captured[0][3]["source_message_id"], "00000000-0000-0000-0000-000000000010")

    def test_journal_insert_success_emits_created_event(self):
        import app.worker as worker_mod  # noqa: E402

        captured = []

        async def fake_write(sql_model, schema_model, payload, extra_fields):
            captured.append((sql_model.__tablename__, schema_model.__name__, payload, extra_fields))
            return True

        original_write = worker_mod._write
        worker_mod._write = fake_write
        try:
            env = BaseEnvelope(
                kind="journal.entry.write.v1",
                source=ServiceRef(name="orion-actions"),
                correlation_id="00000000-0000-0000-0000-000000000001",
                payload={
                    "entry_id": "entry-1",
                    "created_at": "2026-03-21T12:00:00+00:00",
                    "author": "orion",
                    "mode": "daily",
                    "title": "Morning",
                    "body": "I noticed the system settling.",
                    "source_kind": "scheduler",
                    "source_ref": "2026-03-21",
                    "correlation_id": "00000000-0000-0000-0000-000000000001",
                },
            )
            bus = _FakeBus()
            asyncio.run(handle_envelope(env, bus=bus))
        finally:
            worker_mod._write = original_write

        self.assertEqual(captured[0][0], "journal_entries")
        self.assertEqual(captured[0][1], "JournalEntryWriteV1")
        self.assertEqual(len(bus.published), 1)
        channel, created_env = bus.published[0]
        self.assertEqual(channel, Settings().sql_writer_journal_created_channel)
        self.assertEqual(created_env.kind, "journal.entry.created.v1")
        self.assertEqual(created_env.payload["entry_id"], "entry-1")

    def test_failed_insert_does_not_emit_journal_created_event(self):
        import app.worker as worker_mod  # noqa: E402

        async def fake_write(*args, **kwargs):
            raise RuntimeError("db down")

        original_write = worker_mod._write
        original_fallback = worker_mod._write_fallback
        worker_mod._write = fake_write
        worker_mod._write_fallback = lambda *args, **kwargs: None
        try:
            env = BaseEnvelope(
                kind="journal.entry.write.v1",
                source=ServiceRef(name="orion-actions"),
                correlation_id="00000000-0000-0000-0000-000000000002",
                payload={
                    "entry_id": "entry-2",
                    "created_at": "2026-03-21T12:00:00+00:00",
                    "author": "orion",
                    "mode": "daily",
                    "title": None,
                    "body": "Body",
                    "source_kind": "scheduler",
                    "source_ref": "2026-03-21",
                    "correlation_id": "00000000-0000-0000-0000-000000000002",
                },
            )
            bus = _FakeBus()
            asyncio.run(handle_envelope(env, bus=bus))
        finally:
            worker_mod._write = original_write
            worker_mod._write_fallback = original_fallback

        self.assertEqual(bus.published, [])

    def test_collapse_stored_event_emits_after_successful_persistence(self):
        import app.worker as worker_mod  # noqa: E402

        async def fake_write(*args, **kwargs):
            return True

        original_write = worker_mod._write
        worker_mod._write = fake_write
        try:
            env = BaseEnvelope(
                kind="collapse.mirror",
                source=ServiceRef(name="orion-collapse-mirror"),
                correlation_id="00000000-0000-0000-0000-000000000003",
                payload={
                    "id": "mirror-1",
                    "summary": "Dense mirror",
                    "trigger": "prompt",
                    "is_causally_dense": True,
                    "what_changed_summary": "attention sharpened",
                    "mantra": "stay steady",
                },
            )
            bus = _FakeBus()
            asyncio.run(handle_envelope(env, bus=bus))
        finally:
            worker_mod._write = original_write

        self.assertEqual(len(bus.published), 1)
        channel, stored_env = bus.published[0]
        self.assertEqual(channel, "orion:collapse:stored")
        self.assertEqual(stored_env.kind, "collapse.mirror.stored.v1")
        self.assertEqual(stored_env.payload["mirror_id"], "mirror-1")
        self.assertTrue(stored_env.payload["is_causally_dense"])

    def test_append_only_journal_path_uses_insert_not_merge(self):
        import app.worker as worker_mod  # noqa: E402

        fake_session = _FakeSession()
        original_get_session = worker_mod.get_session
        original_remove_session = worker_mod.remove_session
        worker_mod.get_session = lambda: fake_session
        worker_mod.remove_session = lambda: None
        try:
            result = _write_row(
                JournalEntrySQL,
                {
                    "entry_id": "entry-3",
                    "created_at": "2026-03-21T12:00:00+00:00",
                    "author": "orion",
                    "mode": "manual",
                    "title": None,
                    "body": "Append only",
                },
            )
        finally:
            worker_mod.get_session = original_get_session
            worker_mod.remove_session = original_remove_session

        self.assertTrue(result)
        self.assertEqual(len(fake_session.add_calls), 1)
        self.assertEqual(fake_session.merge_calls, [])

    def test_social_turn_insert_emits_post_commit_stored_event(self):
        import app.worker as worker_mod  # noqa: E402

        async def fake_write(*args, **kwargs):
            return True

        original_write = worker_mod._write
        worker_mod._write = fake_write
        try:
            env = BaseEnvelope(
                kind="social.turn.v1",
                source=ServiceRef(name="hub"),
                correlation_id="00000000-0000-0000-0000-000000000004",
                payload={
                    "turn_id": "social-1",
                    "correlation_id": "00000000-0000-0000-0000-000000000004",
                    "session_id": "sid-social",
                    "source": "hub_ws",
                    "profile": "social_room",
                    "prompt": "Hey, stay close to the conversation.",
                    "response": "I’m here with you.",
                    "text": "User: Hey, stay close to the conversation.\nOrion: I’m here with you.",
                    "recall_profile": "social.room.v1",
                    "tags": ["social_room"],
                    "concept_evidence": [],
                    "grounding_state": {"profile": "social_room", "identity_label": "Oríon", "relationship_frame": "peer", "self_model_hint": "distributed social presence", "stance": "warm, direct, grounded"},
                    "redaction": {"prompt_score": 0.0, "response_score": 0.0, "memory_score": 0.0, "overall_score": 0.0, "recall_safe": True, "redaction_level": "low", "reasons": []},
                    "client_meta": {"chat_profile": "social_room"},
                },
            )
            bus = _FakeBus()
            asyncio.run(handle_envelope(env, bus=bus))
        finally:
            worker_mod._write = original_write

        self.assertEqual(len(bus.published), 1)
        channel, stored_env = bus.published[0]
        self.assertEqual(channel, Settings().sql_writer_social_turn_stored_channel)
        self.assertEqual(stored_env.kind, "social.turn.stored.v1")
        self.assertEqual(stored_env.payload["turn_id"], "social-1")
        self.assertEqual(stored_env.payload["profile"], "social_room")

    def test_disabled_social_turn_emit_skips_post_commit_event(self):
        import app.worker as worker_mod  # noqa: E402
        import app.settings as settings_mod  # noqa: E402

        async def fake_write(*args, **kwargs):
            return True

        original_write = worker_mod._write
        original_settings = settings_mod.settings
        worker_mod._write = fake_write
        settings_mod.settings = Settings(SQL_WRITER_EMIT_SOCIAL_TURN_STORED=False)
        worker_mod.settings = settings_mod.settings
        try:
            env = BaseEnvelope(
                kind="social.turn.v1",
                source=ServiceRef(name="hub"),
                correlation_id="00000000-0000-0000-0000-000000000005",
                payload={
                    "turn_id": "social-3",
                    "correlation_id": "00000000-0000-0000-0000-000000000005",
                    "session_id": "sid-social",
                    "source": "hub_ws",
                    "profile": "social_room",
                    "prompt": "still here?",
                    "response": "Yes.",
                    "text": "User: still here?\nOrion: Yes.",
                },
            )
            bus = _FakeBus()
            asyncio.run(handle_envelope(env, bus=bus))
        finally:
            worker_mod._write = original_write
            settings_mod.settings = original_settings
            worker_mod.settings = original_settings

        self.assertEqual(bus.published, [])

    def test_append_only_social_turn_uses_insert_not_merge(self):
        import app.worker as worker_mod  # noqa: E402

        fake_session = _FakeSession()
        original_get_session = worker_mod.get_session
        original_remove_session = worker_mod.remove_session
        worker_mod.get_session = lambda: fake_session
        worker_mod.remove_session = lambda: None
        try:
            result = _write_row(
                SocialRoomTurnSQL,
                {
                    "turn_id": "social-2",
                    "correlation_id": "corr-social-2",
                    "session_id": "sid-social",
                    "source": "hub_ws",
                    "profile": "social_room",
                    "prompt": "hi",
                    "response": "hello",
                    "text": "User: hi\nOrion: hello",
                },
            )
        finally:
            worker_mod.get_session = original_get_session
            worker_mod.remove_session = original_remove_session

        self.assertTrue(result)
        self.assertEqual(len(fake_session.add_calls), 1)
        self.assertEqual(fake_session.merge_calls, [])


if __name__ == "__main__":
    unittest.main()
