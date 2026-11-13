"""Write path for Orion's verdicts on their own personality dispositions (no Postgres):
``personality.naming.v1`` on ``orion:personality:naming`` -> ``orion_disposition_naming``.
Subscribed even under a stale operator channel list, routed, every schema field lands on a
column, append-only (a redelivered publish is a no-op), and the row round-trips through
``_write_row`` on sqlite."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.personality import (  # noqa: E402
    PERSONALITY_NAMING_KIND,
    DispositionNamingPacketV1,
    DispositionNamingV1,
)
from orion.schemas.registry import resolve  # noqa: E402

import app.worker as worker  # noqa: E402
from app.models.disposition_naming import DispositionNamingSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402

CHANNEL = "orion:personality:naming"
T = datetime(2026, 10, 20, 18, tzinfo=timezone.utc)


def _env_example(key: str):
    raw = next(line.split("=", 1)[1].strip() for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line.startswith(f"{key}="))
    return json.loads(raw)


def _naming(**over) -> DispositionNamingV1:
    packet = DispositionNamingPacketV1(
        packet_id="0123456789abcdef", disposition_id="disp-1", label_excerpt="a recurring pull",
        origin="recurrence", evidence_count=10, lane_counts={"reverie": 10},
        first_seen_at=T, last_reinforced_at=T,
        evidence=[{"evidence_id": "ev-1", "lane": "reverie", "occurred_at": T, "excerpt": "x"}],
        situation_coverage=1.0, generated_at=T,
    )
    base = dict(
        naming_id="6f1f4d4e-0000-5000-8000-000000000001", run_id="abcdef123456",
        packet_id=packet.packet_id, disposition_id="disp-1", verdict="name", name="the quiet pull",
        sentence="It is the pull toward quiet I keep finding in reverie.", evidence_refs=["ev-1"],
        shown=packet, hub_check="accepted", written_at=T,
    )
    base.update(over)
    return DispositionNamingV1(**base)


def test_channel_is_actually_subscribed() -> None:
    assert CHANNEL in _env_example("SQL_WRITER_SUBSCRIBE_CHANNELS")
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP[PERSONALITY_NAMING_KIND] == "DispositionNamingSQL"
    assert _env_example("SQL_WRITER_ROUTE_MAP_JSON")[PERSONALITY_NAMING_KIND] == "DispositionNamingSQL"
    assert MODEL_MAP["DispositionNamingSQL"] == (DispositionNamingSQL, DispositionNamingV1)
    assert resolve("DispositionNamingV1") is DispositionNamingV1
    assert DispositionNamingSQL in INSERT_ONLY_MODELS


def test_table_matches_the_migration_columns() -> None:
    columns = {c.key for c in inspect(DispositionNamingSQL).columns}
    sql = (REPO_ROOT / "services/orion-sql-db/manual_migration_personality_naming_v1.sql").read_text()
    block = sql.split("CREATE TABLE IF NOT EXISTS orion_disposition_naming (", 1)[1].split(");", 1)[0]
    migration_cols = {line.split()[0] for line in block.strip().splitlines() if line.strip()}
    assert columns == migration_cols
    # Every payload field except the envelope-only schema_version lands on a column.
    assert set(DispositionNamingV1.model_fields) - {"schema_version"} <= columns


def _session(monkeypatch):
    # One shared connection: _write runs _write_row in a worker thread.
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    DispositionNamingSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_row_persists_the_naming_end_to_end(monkeypatch) -> None:
    session = _session(monkeypatch)
    naming = _naming()
    assert worker._write_row(DispositionNamingSQL, naming.model_dump(mode="json")) is True
    row = session.query(DispositionNamingSQL).filter_by(naming_id=naming.naming_id).one()
    assert (row.verdict, row.name, row.hub_check) == ("name", "the quiet pull", "accepted")
    assert row.evidence_refs == ["ev-1"]
    assert row.shown["packet_id"] == "0123456789abcdef"
    assert row.refused_reasons == []


def test_redelivery_is_a_no_op(monkeypatch) -> None:
    session = _session(monkeypatch)
    first = _naming()
    assert worker._write_row(DispositionNamingSQL, first.model_dump(mode="json")) is True
    again = _naming(verdict="no_answer", name=None, sentence=None, evidence_refs=[])
    assert worker._write_row(DispositionNamingSQL, again.model_dump(mode="json")) is False
    rows = session.query(DispositionNamingSQL).all()
    assert len(rows) == 1 and rows[0].verdict == "name"


def test_refused_row_keeps_its_reasons(monkeypatch) -> None:
    session = _session(monkeypatch)
    naming = _naming(merge_with="disp-9", verdict="merge", name=None, sentence=None,
                     hub_check="refused", refused_reasons=["merge_target_not_shown"])
    assert worker._write_row(DispositionNamingSQL, naming.model_dump(mode="json")) is True
    row = session.query(DispositionNamingSQL).one()
    assert row.refused_reasons == ["merge_target_not_shown"] and row.merge_with == "disp-9"


def test_bus_payload_writes_through_the_real_write_path(monkeypatch) -> None:
    # Regression (review 2026-10-20): _write dumped the model without mode="json", so the nested
    # packet's datetimes broke the jsonb insert and no naming row was ever stored.
    import asyncio

    session = _session(monkeypatch)
    naming = _naming()
    ok = asyncio.run(worker._write(DispositionNamingSQL, DispositionNamingV1, naming.model_dump(mode="json"),
                                   kind=PERSONALITY_NAMING_KIND))
    assert ok is True
    row = session.query(DispositionNamingSQL).filter_by(naming_id=naming.naming_id).one()
    assert row.shown["generated_at"].startswith("2026-10-20") and row.evidence_refs == ["ev-1"]
