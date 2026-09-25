"""Stage 4.4: ``durable_demand_pending`` counts durable runs waiting for a GPU across the cutover
from durable-runs' own broker (durable_resource_demands) to GPU pool holds (gpu_pool_leases
kind='hold') -- no gap, no double count -- and ``gpu_pool_waiting`` stops counting holds.

Runs the store's real SQL on a throwaway Postgres (GPU_POOL_TEST_POSTGRES_URI, the pool CI
service; never the live 55432), in a private schema, with gpu_pool_leases created by the real
migration file. Skips without a DSN.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import quote

import pytest

DSN = os.environ.get("GPU_POOL_TEST_POSTGRES_URI") or os.environ.get("FIELD_DIGESTER_TEST_POSTGRES_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="needs GPU_POOL_TEST_POSTGRES_URI (throwaway Postgres)")

ROOT = Path(__file__).resolve().parents[3]
POOL_MIGRATION = ROOT / "services" / "orion-sql-db" / "manual_migration_gpu_pool_v1.sql"


@pytest.fixture
def store():
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from app.store import FieldDigesterStore

    schema = f"fd_s44_{uuid.uuid4().hex[:10]}"
    admin = sqlalchemy.create_engine(DSN)
    with admin.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DSN else "?"
    scoped = f"{DSN}{sep}options={quote(f'-csearch_path={schema}')}"
    st = FieldDigesterStore(scoped)
    with st._engine.begin() as conn:
        conn.exec_driver_sql(POOL_MIGRATION.read_text())
        # Only the columns the reader touches (the full admission migration drags in its FKs).
        conn.exec_driver_sql(
            "CREATE TABLE durable_resource_demands (demand_id text PRIMARY KEY, run_id text NOT NULL UNIQUE, "
            "created_at timestamptz NOT NULL, status text NOT NULL)")
    try:
        yield st
    finally:
        st._engine.dispose()
        with admin.begin() as conn:
            conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
        admin.dispose()


def demand(st, run: str, status: str = "pending", age_sec: int = 60) -> None:
    with st._engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO durable_resource_demands VALUES (%s, %s, now() - make_interval(secs => %s), %s)",
            (f"{run}:turn", run, age_sec, status))


def lease(st, lease_id: str, *, kind: str, status: str, holder: str, age_sec: int = 30,
          queued_age_sec: int | None = None) -> None:
    with st._engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO gpu_pool_leases (lease_id, request_id, holder, work_class, priority, kind, status, "
            "created_at, queued_since) VALUES (%s, %s, %s, 'agent', 'background', %s, %s, "
            "now() - make_interval(secs => %s), "
            "CASE WHEN %s::int IS NULL THEN NULL ELSE now() - make_interval(secs => %s::int) END)",
            (lease_id, f"req-{lease_id}", holder, kind, status, age_sec, queued_age_sec, queued_age_sec))


def hold(st, run: str, status: str, **kw) -> None:
    lease(st, f"hold-{run}", kind="hold", status=status, holder=f"durable-runs:{run}", **kw)


def test_before_cutover_it_is_exactly_the_legacy_queue(store):
    demand(store, "a", age_sec=4000)
    demand(store, "b", age_sec=100)
    demand(store, "c", status="granted", age_sec=9000)
    lease(store, "r1", kind="request", status="queued", holder="http:anthropic", queued_age_sec=5)
    assert store.count_durable_demand_pending() == 2
    assert 3990 <= store.oldest_durable_demand_pending_age_sec() <= 4100
    assert store.count_gpu_pool_waiting() == 1


def test_during_cutover_a_run_with_both_counts_once(store):
    # a: frozen legacy demand AND a re-registered queued hold -> one waiting run, not two.
    demand(store, "a", age_sec=380000)
    hold(store, "a", "queued", queued_age_sec=50)
    # b: legacy demand still pending but its hold is already granted -> running, not waiting.
    demand(store, "b", age_sec=200000)
    hold(store, "b", "granted")
    # c: only a hold, backlogged (no role can serve it yet) -> waiting.
    hold(store, "c", "backlogged", age_sec=70)
    # d: only a hold, cooling down after an expiry -> not waiting (same rule as gpu_pool_waiting).
    hold(store, "d", "retry_wait")
    # e: legacy only, its old hold finished (released) -> still the legacy wait.
    demand(store, "e", age_sec=1000)
    hold(store, "e", "released")
    assert store.count_durable_demand_pending() == 3  # a (hold), c (hold), e (legacy)
    # a's 380000s-old demand is dropped in favour of its hold; oldest is e's legacy demand.
    assert 990 <= store.oldest_durable_demand_pending_age_sec() <= 1100


def test_after_migration_it_is_exactly_the_waiting_holds(store):
    demand(store, "a", status="withdrawn", age_sec=380000)
    hold(store, "a", "queued", age_sec=900, queued_age_sec=600)
    hold(store, "b", "queued", queued_age_sec=10)
    assert store.count_durable_demand_pending() == 2
    assert 590 <= store.oldest_durable_demand_pending_age_sec() <= 700  # queued_since, not created_at


def test_waiting_holds_never_enter_gpu_pool_waiting(store):
    hold(store, "a", "queued", queued_age_sec=5000)
    hold(store, "b", "backlogged")
    lease(store, "r1", kind="request", status="backlogged", holder="http:anthropic")
    lease(store, "r2", kind="request", status="queued", holder="orion-llm-gateway", queued_age_sec=7)
    assert store.count_gpu_pool_waiting() == 2
    # The 5000s hold would pin the 60s-anchored pool age; it belongs to the durable source.
    assert 5 <= store.oldest_gpu_pool_waiting_age_sec() <= 60
    assert store.count_durable_demand_pending() == 2


def test_empty_is_zero_not_a_failure(store):
    assert store.count_durable_demand_pending() == 0
    assert store.oldest_durable_demand_pending_age_sec() == 0.0
