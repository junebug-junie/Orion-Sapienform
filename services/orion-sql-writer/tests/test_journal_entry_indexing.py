from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.dialects import postgresql

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import JournalEntryWriteV1, JournalTriggerV1, build_journal_entry_index_payload
from orion.schemas.chat_stance import ChatStanceBrief

from app.journal_index_repository import build_journal_entry_index_select

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_journal_index_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _source() -> ServiceRef:
    return ServiceRef(name="test-producer", version="0.0.1", node="local")


def _base_write() -> JournalEntryWriteV1:
    return JournalEntryWriteV1(
        entry_id=str(uuid4()),
        created_at=datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc),
        author="orion",
        mode="manual",
        title="Journal title",
        body="Journal body",
        source_kind="manual",
        source_ref="src-1",
        correlation_id="corr-1",
    )


def test_index_payload_builds_from_thin_write_only() -> None:
    payload = build_journal_entry_index_payload(_base_write())
    assert payload["entry_id"]
    assert payload["trigger_kind"] is None
    assert payload["conversation_frame"] is None
    assert payload["active_identity_facets"] is None


def test_index_payload_builds_with_trigger() -> None:
    trigger = JournalTriggerV1(
        trigger_kind="manual",
        source_kind="manual",
        source_ref="manual-ref",
        summary="manual summary",
    )
    payload = build_journal_entry_index_payload(_base_write(), trigger=trigger)
    assert payload["trigger_kind"] == "manual"
    assert payload["trigger_summary"] == "manual summary"


def test_index_payload_builds_with_trigger_and_chat_stance() -> None:
    trigger = JournalTriggerV1(
        trigger_kind="notify_summary",
        source_kind="notify",
        source_ref="n-1",
        summary="notify summary",
    )
    stance = ChatStanceBrief(
        conversation_frame="reflective",
        task_mode="reflective_dialogue",
        identity_salience="high",
        user_intent="reflect",
        self_relevance="high",
        juniper_relevance="high",
        active_identity_facets=["identity continuity"],
        active_growth_axes=["stability"],
        active_relationship_facets=["trust"],
        social_posture=["warm"],
        reflective_themes=["integration"],
        active_tensions=["speed_vs_depth"],
        dream_motifs=["bridge"],
        response_priorities=["direct"],
        response_hazards=["overgeneralization"],
        answer_strategy="concise_reflection",
        stance_summary="brief reflective frame",
    )
    payload = build_journal_entry_index_payload(_base_write(), trigger=trigger, chat_stance=stance)
    assert payload["conversation_frame"] == "reflective"
    assert payload["answer_strategy"] == "concise_reflection"
    assert payload["active_identity_facets"] == ["identity continuity"]


def test_index_payload_gracefully_handles_missing_optional_stance_fields() -> None:
    payload = build_journal_entry_index_payload(
        _base_write(),
        stance_metadata={"conversation_frame": "technical"},
    )
    assert payload["conversation_frame"] == "technical"
    assert payload["task_mode"] is None
    assert payload["response_hazards"] is None


def test_journal_write_populates_journal_entry_index(monkeypatch) -> None:
    writes: list[tuple[str, dict]] = []

    def _fake_write_row(sql_model_cls, data):
        writes.append((sql_model_cls.__tablename__, data))
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    write = _base_write().model_dump(mode="json")
    env = BaseEnvelope(
        kind="journal.entry.write.v1",
        source=_source(),
        payload={
            "schema_id": "orion.envelope",
            "schema_version": "2.0.0",
            "kind": "journal.entry.write.v1",
            "payload": write,
            "journal_trigger": {
                "trigger_kind": "manual",
                "source_kind": "manual",
                "summary": "manual trigger",
            },
            "chat_stance_brief": {
                "conversation_frame": "technical",
                "task_mode": "direct_response",
                "identity_salience": "medium",
                "user_intent": "help",
                "self_relevance": "x",
                "juniper_relevance": "y",
                "active_identity_facets": ["identity"],
                "active_growth_axes": [],
                "active_relationship_facets": [],
                "social_posture": [],
                "reflective_themes": [],
                "active_tensions": [],
                "dream_motifs": [],
                "response_priorities": ["direct"],
                "response_hazards": ["drift"],
                "answer_strategy": "plain",
                "stance_summary": "short",
            },
        },
    )

    asyncio.run(worker.handle_envelope(env, bus=None))

    tables = [table for table, _ in writes]
    assert "journal_entries" in tables
    assert "journal_entry_index" in tables
    assert "evidence_units" in tables
    index_row = [row for table, row in writes if table == "journal_entry_index"][0]
    assert index_row["trigger_kind"] == "manual"
    assert index_row["conversation_frame"] == "technical"


def test_simple_filtered_retrieval_sql_shape() -> None:
    stmt = build_journal_entry_index_select(
        mode="manual",
        source_kind="manual",
        trigger_kind="manual",
        author="orion",
        source_ref="src-1",
        correlation_id="corr-1",
        facets={"active_identity_facets": ["identity continuity"]},
        text_query="reflective bridge",
        limit=20,
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "FROM journal_entry_index" in sql
    assert "journal_entry_index.mode =" in sql
    assert "journal_entry_index.source_kind =" in sql
    assert "ILIKE" in sql
    assert "ORDER BY journal_entry_index.created_at DESC" in sql
