"""Compile-time shape checks for the self_knowledge_items SQL write path
(no Postgres required) -- self-model rebuild arc, Patch 2, 2026-09-05.
Real wiring: `SelfKnowledgeItemLogSQL` is registered in `MODEL_MAP` under
its own route key, keyed off kind `self_study.items.write.v1`, and every
field on the real `SelfKnowledgeItemLogV1` producer schema maps onto a real
column on `SelfKnowledgeItemLogSQL`.
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

from orion.schemas.self_knowledge_item_log import SelfKnowledgeItemLogV1  # noqa: E402

from app.models.self_knowledge_item import SelfKnowledgeItemLogSQL  # noqa: E402
import app.worker as worker  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402


def _make_payload(**overrides) -> SelfKnowledgeItemLogV1:
    defaults = dict(
        item_id="self-item-abc123",
        run_id="self-run-1",
        category="service",
        name="orion-cortex-exec",
        trust_tier="authoritative",
        observed_at="2026-09-05T00:00:00Z",
        source_path="services/orion-cortex-exec/app/settings.py",
        symbol_name=None,
        metadata_text="has_app_main=True",
    )
    defaults.update(overrides)
    return SelfKnowledgeItemLogV1(**defaults)


def test_the_channel_is_actually_subscribed() -> None:
    """Review finding: registering a channel/route/model is not the same as
    subscribing to it. SQL_WRITER_SUBSCRIBE_CHANNELS REPLACES the Python
    default wholesale rather than merging (this repo's own established
    failure shape, see settings.py's effective_subscribe_channels comments)
    -- both the .env_example list and a code-default-only Settings instance
    must carry the new channel."""
    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:self_study:items:write" in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:self_study:items:write" in stale.effective_subscribe_channels


def test_default_route_map_points_self_study_items_write_v1_at_self_knowledge_item_log_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("self_study.items.write.v1") == "SelfKnowledgeItemLogSQL"


def test_model_map_registers_self_knowledge_item_log_sql_with_its_schema() -> None:
    assert MODEL_MAP["SelfKnowledgeItemLogSQL"] == (SelfKnowledgeItemLogSQL, SelfKnowledgeItemLogV1)


def test_self_knowledge_item_log_v1_fields_map_onto_real_columns() -> None:
    mapper = inspect(SelfKnowledgeItemLogSQL)
    valid_keys = {attr.key for attr in mapper.attrs}

    payload = _make_payload()
    data = payload.model_dump(mode="json")

    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"SelfKnowledgeItemLogV1 fields missing from SelfKnowledgeItemLogSQL columns: {missing}"


def test_self_knowledge_item_log_v1_data_constructs_sql_row_without_raising() -> None:
    payload = _make_payload()
    data = payload.model_dump(mode="json")

    row = SelfKnowledgeItemLogSQL(**data)

    assert row.entry_id == payload.entry_id
    assert row.item_id == payload.item_id
    assert row.run_id == payload.run_id
    assert row.category == payload.category
    assert row.name == payload.name
    assert row.trust_tier == payload.trust_tier
    assert row.observed_at == payload.observed_at
    assert row.source_path == payload.source_path
    assert row.metadata_text == payload.metadata_text


def test_write_row_persists_self_knowledge_item_log_end_to_end(monkeypatch) -> None:
    """Review finding: the tests above never exercise the real write path
    (`worker._write_row`) -- this repo's own test_journal_entry_trigger_kind_
    filtering.py already shows the correct pattern for that. Real proof: an
    in-memory sqlite table, a real `_write_row` call, a query-back."""
    engine = create_engine("sqlite://")
    SelfKnowledgeItemLogSQL.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    payload = _make_payload()
    data = payload.model_dump(mode="json")

    ok = worker._write_row(SelfKnowledgeItemLogSQL, data)
    assert ok is True

    row = session.query(SelfKnowledgeItemLogSQL).filter_by(entry_id=payload.entry_id).first()
    assert row is not None
    assert row.item_id == payload.item_id
    assert row.category == payload.category
    assert row.metadata_text == payload.metadata_text
