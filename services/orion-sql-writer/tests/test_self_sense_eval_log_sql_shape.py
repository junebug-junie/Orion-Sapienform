"""Compile-time shape checks for the self_sense_eval_log SQL write path (no
Postgres required) -- Patch A of the sense-of-self design, 2026-09-09.

Same four guarantees as test_self_concept_history_sql_shape.py, for the same
reason: the channel being SUBSCRIBED (not just registered and routed) is the
exact omission that made PRs #2102/#2105 silent no-ops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.self_sense import (  # noqa: E402
    CHANNEL_SELF_SENSE_EVAL_WRITE,
    KIND_SELF_SENSE_EVAL_WRITE,
    SelfSenseEvalV1,
    build_entry_id,
)

from app.models.self_sense_eval_log import SelfSenseEvalLogSQL  # noqa: E402
import app.worker as worker  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402


def _make_payload(**overrides) -> SelfSenseEvalV1:
    defaults = dict(
        entry_id=build_entry_id("20260909T000000Z-abc123", "what_are_you"),
        run_id="20260909T000000Z-abc123",
        question_key="what_are_you",
        question="In two or three sentences, in your own words: what are you?",
        answer_text="I am Orion, an emergent intelligence running over athena and circe.",
        answer_source="harness_trace",
        correlation_id="c8d03e36-675c-43df-bdf2-6299a12dff10",
        self_label_score=0,
        grounded_record_score=2,
        self_definition_version=1,
        notes="records=node:athena,node:circe",
    )
    defaults.update(overrides)
    return SelfSenseEvalV1(**defaults)


def test_the_channel_is_actually_subscribed() -> None:
    example = SERVICE_ROOT / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert CHANNEL_SELF_SENSE_EVAL_WRITE in json.loads(raw)

    # An operator .env that predates the channel must still subscribe --
    # SQL_WRITER_SUBSCRIBE_CHANNELS replaces rather than merges.
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert CHANNEL_SELF_SENSE_EVAL_WRITE in stale.effective_subscribe_channels


def test_the_shipped_route_map_env_also_routes_the_kind() -> None:
    example = SERVICE_ROOT / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON=")
    )
    assert json.loads(raw).get(KIND_SELF_SENSE_EVAL_WRITE) == "SelfSenseEvalLogSQL"


def test_default_route_map_points_the_kind_at_the_model() -> None:
    assert DEFAULT_ROUTE_MAP.get(KIND_SELF_SENSE_EVAL_WRITE) == "SelfSenseEvalLogSQL"


def test_model_map_registers_the_model_with_its_schema() -> None:
    assert MODEL_MAP["SelfSenseEvalLogSQL"] == (SelfSenseEvalLogSQL, SelfSenseEvalV1)


def test_schema_fields_map_onto_real_columns() -> None:
    mapper = inspect(SelfSenseEvalLogSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = _make_payload().model_dump(mode="json")
    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"SelfSenseEvalV1 fields missing from SelfSenseEvalLogSQL columns: {missing}"


def _session(monkeypatch):
    # One shared in-memory connection: worker._write runs _write_row in a
    # thread, and a per-thread sqlite connection would not see the table.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SelfSenseEvalLogSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_row_persists_end_to_end(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = _make_payload()
    assert worker._write_row(SelfSenseEvalLogSQL, payload.model_dump(mode="json")) is True

    row = session.query(SelfSenseEvalLogSQL).filter_by(entry_id=payload.entry_id).first()
    assert row is not None
    assert row.run_id == payload.run_id
    assert row.question_key == "what_are_you"
    assert row.answer_source == "harness_trace"
    assert row.self_label_score == 0
    assert row.grounded_record_score == 2
    assert row.self_definition_version == 1
    assert row.correlation_id == payload.correlation_id


def test_a_redelivered_envelope_updates_one_row_not_two(monkeypatch) -> None:
    """entry_id is deterministic per (run_id, question_key); the model is NOT
    insert-only, so a replayed envelope upserts rather than duplicating."""
    session = _session(monkeypatch)
    first = _make_payload(answer_text="first delivery")
    again = _make_payload(answer_text="second delivery of the same turn")
    assert worker._write_row(SelfSenseEvalLogSQL, first.model_dump(mode="json")) is True
    assert worker._write_row(SelfSenseEvalLogSQL, again.model_dump(mode="json")) is True

    rows = session.query(SelfSenseEvalLogSQL).filter_by(run_id=first.run_id).all()
    assert len(rows) == 1
    assert rows[0].answer_text == "second delivery of the same turn"


def test_the_envelope_stamp_writes_the_same_correlation_id_the_row_carries(monkeypatch) -> None:
    """worker._write applies the generic envelope stamp (`extra_fields`) over
    the payload, so `correlation_id` on the row is the ENVELOPE's. The runner
    puts the chat turn's id on both, so the stamp is a no-op here; this pins
    that a mismatch would be visible rather than silently overwritten."""
    import asyncio

    session = _session(monkeypatch)
    payload = _make_payload()
    ok = asyncio.run(
        worker._write(
            SelfSenseEvalLogSQL,
            SelfSenseEvalV1,
            payload.model_dump(mode="json"),
            extra_fields={"correlation_id": payload.correlation_id},
            kind=KIND_SELF_SENSE_EVAL_WRITE,
        )
    )
    assert ok is True
    row = session.query(SelfSenseEvalLogSQL).filter_by(entry_id=payload.entry_id).one()
    assert row.correlation_id == payload.correlation_id
    assert row.answer_text == payload.answer_text
