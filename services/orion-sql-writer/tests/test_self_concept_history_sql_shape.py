"""Compile-time shape checks for the self_concept_history SQL write path
(no Postgres required) -- self-model rebuild arc, Patch 3, 2026-09-05.
Real wiring: `SelfConceptHistorySQL` is registered in `MODEL_MAP` under its
own route key, keyed off kind `self_concept.history.write.v1`, every field
on the real `SelfConceptHistoryV1` producer schema maps onto a real column,
AND -- learned the hard way twice already this session on sibling
channels (chat_stance:belief:write, self_study:items:write both shipped
without this and were caught by review, one after already merging) --
the channel is actually subscribed, not just registered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.self_concept_history import SelfConceptHistoryV1  # noqa: E402

from app.models.self_concept_history import SelfConceptHistorySQL  # noqa: E402
import app.worker as worker  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402


def _make_payload(**overrides) -> SelfConceptHistoryV1:
    defaults = dict(
        concept_id="self-concept-physical-topology",
        version=1,
        content="Orion's physical mesh spans athena, circe, and prometheus.",
        evidence_refs=["self-item-abc123"],
        produced_by="layer3_reflect",
    )
    defaults.update(overrides)
    return SelfConceptHistoryV1(**defaults)


def test_the_channel_is_actually_subscribed() -> None:
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:self_concept:history:write" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:self_concept:history:write" in stale.effective_subscribe_channels


def test_default_route_map_points_self_concept_history_write_v1_at_self_concept_history_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("self_concept.history.write.v1") == "SelfConceptHistorySQL"


def test_model_map_registers_self_concept_history_sql_with_its_schema() -> None:
    assert MODEL_MAP["SelfConceptHistorySQL"] == (SelfConceptHistorySQL, SelfConceptHistoryV1)


def test_self_concept_history_v1_fields_map_onto_real_columns() -> None:
    mapper = inspect(SelfConceptHistorySQL)
    valid_keys = {attr.key for attr in mapper.attrs}

    payload = _make_payload()
    data = payload.model_dump(mode="json")

    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"SelfConceptHistoryV1 fields missing from SelfConceptHistorySQL columns: {missing}"


def test_write_row_persists_self_concept_history_end_to_end(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    SelfConceptHistorySQL.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    payload = _make_payload()
    data = payload.model_dump(mode="json")

    ok = worker._write_row(SelfConceptHistorySQL, data)
    assert ok is True

    row = session.query(SelfConceptHistorySQL).filter_by(entry_id=payload.entry_id).first()
    assert row is not None
    assert row.concept_id == payload.concept_id
    assert row.version == payload.version
    assert row.content == payload.content
    assert row.evidence_refs == payload.evidence_refs
    assert row.produced_by == payload.produced_by


def test_two_versions_of_same_concept_both_persist_latest_by_created_at(monkeypatch) -> None:
    """'Current' is defined as the latest row per concept_id by created_at,
    not an upsert -- confirm two versions of the same concept_id coexist as
    two distinct rows (append-only, never updated in place)."""
    engine = create_engine("sqlite://")
    SelfConceptHistorySQL.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    v1 = _make_payload(version=1, content="First understanding.")
    v2 = _make_payload(version=2, content="Revised understanding.")
    assert worker._write_row(SelfConceptHistorySQL, v1.model_dump(mode="json")) is True
    assert worker._write_row(SelfConceptHistorySQL, v2.model_dump(mode="json")) is True

    rows = session.query(SelfConceptHistorySQL).filter_by(concept_id=v1.concept_id).all()
    assert len(rows) == 2
    contents = {row.content for row in rows}
    assert contents == {"First understanding.", "Revised understanding."}
