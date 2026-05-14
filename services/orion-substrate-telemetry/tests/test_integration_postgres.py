"""Optional integration checks (Postgres round-trip; not full Redis→worker→HTTP E2E)."""

from __future__ import annotations

import os
from uuid import uuid4

import asyncpg
import pytest

from app import db


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_insert_and_fetch_latest_roundtrip() -> None:
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("RUN_INTEGRATION=1 not set")
    dsn = (os.environ.get("POSTGRES_URI") or "").strip()
    if not dsn:
        pytest.skip("POSTGRES_URI required for integration")

    correlation_id = uuid4()
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await db.ensure_schema(conn)
        await db.insert_event(
            conn,
            correlation_id=correlation_id,
            envelope_kind="substrate.tier_outcomes.v1",
            generated_at="2026-05-14T12:00:00+00:00",
            cold_anchors=["a1"],
            tier_outcomes={"a1": ["operator_static_protected:2"]},
            degraded_producers=["p1"],
            source_service="orion-cortex-exec",
            source_node="n1",
        )
        row = await db.fetch_latest(conn, correlation_id=correlation_id)
        assert row is not None
        assert row["tier_outcomes"] == {"a1": ["operator_static_protected:2"]}
        await conn.execute(
            "DELETE FROM substrate_tier_outcomes_events WHERE correlation_id = $1::uuid",
            correlation_id,
        )
    finally:
        await conn.close()
