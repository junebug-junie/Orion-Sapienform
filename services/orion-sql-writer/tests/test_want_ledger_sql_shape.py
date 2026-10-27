"""Write path for the bid ledger (body economy Patch C1, no Postgres):
``physiology.want.decision.v1`` on ``orion:physiology:want:decision`` -> ``want_ledger``.

Subscribed even under a stale operator channel list (the silent-no-op incident class), routed,
catalogued, the migration matches the model and fits the post-pull auto-apply allowlist, and the
full ``_write`` path lands one row per decision. Redelivery is a no-op. Also: the strain dial's new
``self_spent`` field (consumer-first) is accepted on physiology_state."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(REPO_ROOT / "scripts")]

from orion.schemas.physiology_want import (  # noqa: E402
    WANT_DECISION_CHANNEL,
    WANT_DECISION_KIND,
    WantDecisionV1,
    WantV1,
)
from orion.schemas.registry import resolve  # noqa: E402

import app.worker as worker  # noqa: E402
from app.models.want_ledger import WantLedgerSQL, want_ledger_row  # noqa: E402

T0 = datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc)
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_want_ledger_v1.sql"


def _decision(**kw) -> WantDecisionV1:
    want = WantV1(want_id="w1", family="durable.self_inquiry", producer="test", verb="ask", tier=2,
                  correlation_id="corr-1")
    base = dict(decision_id=f"w1@{T0.isoformat()}", want_id="w1", family="durable.self_inquiry", producer="test",
                verb="ask", tier=2, decided_at=T0, outcome="deferred", reason="price", lane="agent",
                prices_seen={"lane:agent": 0.8}, cost_frac={"lane:agent": 0.5}, charge=0.4, net=-0.1,
                binding_constraint="lane:agent", retry_after_sec=900.0, want=want)
    base.update(kw)
    return WantDecisionV1(**base)


def test_subscribed_routed_and_catalogued() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = dict(line.split("=", 1) for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    assert WANT_DECISION_CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert WANT_DECISION_CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[WANT_DECISION_KIND] == "WantLedgerSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[WANT_DECISION_KIND] == "WantLedgerSQL"
    # The live operator map overrides keys but keeps defaults (route_map merges): the route works
    # even before the .env JSON is hand-edited.
    assert Settings(SQL_WRITER_ROUTE_MAP_JSON="{}").route_map[WANT_DECISION_KIND] == "WantLedgerSQL"
    assert worker.MODEL_MAP["WantLedgerSQL"] == (WantLedgerSQL, WantDecisionV1)
    assert WantLedgerSQL in worker.INSERT_ONLY_MODELS
    assert resolve("WantDecisionV1") is WantDecisionV1 and resolve("WantV1") is WantV1
    import yaml

    catalog = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text())
    entries = catalog if isinstance(catalog, list) else catalog["channels"]
    ch = next(c for c in entries if c["name"] == WANT_DECISION_CHANNEL)
    assert ch["schema_id"] == "WantDecisionV1" and ch["message_kind"] == WANT_DECISION_KIND
    assert ch["producer_services"] == ["orion-durable-runs"] and "orion-sql-writer" in ch["consumer_services"]


def test_migration_columns_match_the_model_and_fit_the_auto_apply_allowlist() -> None:
    text = MIGRATION.read_text()
    block = re.search(r"CREATE TABLE IF NOT EXISTS want_ledger \((.*?)\n\);", text, re.S).group(1)
    migration_cols = {line.split()[0] for line in block.strip().splitlines() if line.strip()}
    assert {c.name for c in inspect(WantLedgerSQL).columns} == migration_cols
    assert text.startswith("-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer")
    import orion.sql_migration_drift as drift

    assert drift.auto_apply_refusals(text, set()) == [], "migration must fit the post-pull additive allowlist"
    rb = drift.parse_required_by(text)
    assert rb is not None and not rb.errors


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    WantLedgerSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_path_persists_one_row_per_decision_and_redelivery_is_a_no_op(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = json.loads(_decision().model_dump_json())
    assert asyncio.run(worker._write(WantLedgerSQL, WantDecisionV1, payload, kind=WANT_DECISION_KIND))
    assert not asyncio.run(worker._write(WantLedgerSQL, WantDecisionV1, payload, kind=WANT_DECISION_KIND))
    row = session.query(WantLedgerSQL).one()
    assert (row.outcome, row.reason, row.lane, row.binding_constraint, row.shadow) == (
        "deferred", "price", "agent", "lane:agent", True)
    assert row.correlation_id == "corr-1" and row.decision["prices_seen"] == {"lane:agent": 0.8}
    assert WantDecisionV1.model_validate(row.decision) == _decision()


def test_row_is_json_safe() -> None:
    row = want_ledger_row(_decision())
    json.dumps({k: v for k, v in row.items() if k != "decided_at"})


def test_physiology_state_with_the_self_spent_strain_field_is_accepted() -> None:
    """Consumer-first for Patch C1's strain fix: PhysiologyStrainV1.self_spent validates here."""
    from orion.schemas.physiology import PhysiologyStateV1

    st = PhysiologyStateV1.model_validate({
        "tick_id": "physiology:x", "generated_at": T0.isoformat(), "constraints": [],
        "strain": {"value": 0.0, "self_spent": ["attention:juniper"], "why": "strain 0"}})
    assert st.strain.self_spent == ["attention:juniper"]
