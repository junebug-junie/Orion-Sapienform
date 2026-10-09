"""The existing broadcast becomes small, idempotent history with honest freshness."""
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from orion.schemas.gpu_pool import GpuPoolStateV1
from app.models.gpu_pool_state import GpuPoolStateSQL, state_history_row


def payload(**updates):
    return dict(generated_at="2026-10-09T08:00:00Z", host="athena", mode="enforce",
                config_digest="test-config", cards=[], roles=[], backlog_depth={"agent": 2}, queue_depth={}, **updates)


def test_snapshot_is_minimal_and_redelivery_has_same_identity():
    state = GpuPoolStateV1(**payload())
    row = state_history_row(state)
    assert row == state_history_row(GpuPoolStateV1(**payload()))
    assert row["backlog_depth"] == {"agent": 2}
    assert "leases" not in row and "cards" not in row


@pytest.mark.parametrize("missing", ["generated_at", "host", "backlog_depth", "queue_depth"])
def test_schema_defaults_cannot_invent_fresh_empty_readings(missing):
    raw = payload()
    del raw[missing]
    with pytest.raises(ValueError):
        state_history_row(GpuPoolStateV1(**raw))


def test_subscription_route_retention_and_catalog_agree():
    from app.settings import Settings, DEFAULT_ROUTE_MAP
    from app.worker import MODEL_MAP, INSERT_ONLY_MODELS
    from app.grammar_retention_loop import retention_days_for
    from app.grammar_truth import GRAMMAR_RETENTION_TABLES
    root = Path(__file__).resolve().parents[1]
    env = dict(line.split("=", 1) for line in (root/".env_example").read_text().splitlines() if line and not line.startswith("#") and "=" in line)
    settings = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[])
    assert "orion:gpu_pool:state" in settings.effective_subscribe_channels
    assert "orion:gpu_pool:state" in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP["gpu_pool.state.v1"] == "GpuPoolStateSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])["gpu_pool.state.v1"] == "GpuPoolStateSQL"
    assert MODEL_MAP["GpuPoolStateSQL"] == (GpuPoolStateSQL, GpuPoolStateV1)
    assert GpuPoolStateSQL in INSERT_ONLY_MODELS
    assert retention_days_for(settings)["gpu_pool_state_history"] == 30
    assert "gpu_pool_state_history" in dict(GRAMMAR_RETENTION_TABLES)
    from app.grammar_truth import _other_retention_truth_blocks
    assert "gpu_pool_state_history" in _other_retention_truth_blocks(settings)
    assert settings.gpu_pool_state_history_retention_days == 30
    import yaml
    catalog = yaml.safe_load((root.parents[1]/"orion/bus/channels.yaml").read_text())
    entries = catalog if isinstance(catalog, list) else catalog["channels"]
    channel = next(c for c in entries if c["name"] == "orion:gpu_pool:state")
    assert "orion-sql-writer" in channel["consumer_services"]


def test_actual_persist_path_postgres_and_retention(monkeypatch):
    uri = os.environ.get("REGULATION_HISTORY_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("requires isolated regulation_history_test Postgres")
    assert urlparse(uri).path == "/regulation_history_test", "refuse any other database"
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from app import worker, grammar_truth
    engine = create_engine(uri)
    migration = Path(__file__).resolve().parents[3]/"services/orion-sql-db/manual_migration_regulation_history.sql"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(migration.read_text())
    monkeypatch.setattr(worker, "get_session", sessionmaker(bind=engine))
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    monkeypatch.setattr(grammar_truth, "default_engine", engine)
    assert asyncio.run(worker._write(GpuPoolStateSQL, GpuPoolStateV1, payload(), kind="gpu_pool.state.v1"))
    assert not asyncio.run(worker._write(GpuPoolStateSQL, GpuPoolStateV1, payload(), kind="gpu_pool.state.v1"))
    identity = {"id": state_history_row(GpuPoolStateV1(**payload()))["snapshot_id"]}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT backlog_depth,generated_at FROM gpu_pool_state_history WHERE snapshot_id=:id"), identity).one()
        assert row[0] == {"agent": 2} and row[1].isoformat() == "2026-10-09T08:00:00+00:00"
        conn.execute(text("UPDATE gpu_pool_state_history SET created_at=now()-interval '31 days' WHERE snapshot_id=:id"), identity)
    grammar_truth.apply_gpu_pool_state_retention(30, max_batches=1, max_elapsed_sec=5)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM gpu_pool_state_history WHERE snapshot_id=:id"), identity).scalar() == 0
    engine.dispose()
