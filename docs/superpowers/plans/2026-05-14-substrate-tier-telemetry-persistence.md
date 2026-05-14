# Substrate tier telemetry persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `orion-substrate-telemetry` to subscribe to `orion:substrate:tier_outcomes`, append tier-outcome rows to Postgres, expose correlation-keyed HTTP reads, and have `orion-cortex-orch` optionally merge persisted telemetry into `MindRunRequestV1.snapshot_inputs.facets.substrate_telemetry` before calling Mind, with catalog and deployment wiring aligned to the approved design.

**Architecture:** Cortex-exec’s existing synchronous Redis publish stays untouched; a new FastAPI + `BaseChassis` worker mirrors `orion-state-journaler`: decode `OrionCodec` frames on the tier-outcomes channel, validate `kind == substrate.tier_outcomes.v1`, insert append-only rows via asyncpg, prune excess rows per correlation after each insert, and run a slow periodic global age purge. Orch performs a short `httpx` GET to the telemetry service (unless tests inlined a facet via `context.metadata`) and passes the decoded JSON into an extended `build_mind_run_request`; Hub remains bus-free and calls the same HTTP API (directly or via a thin Hub proxy route) for UI.

**Tech Stack:**

- Python 3.12, FastAPI, uvicorn, pydantic v2, pydantic-settings
- `asyncpg` for Postgres; `redis[hiredis]` + `orion.core.bus` (`BaseChassis`, `OrionBusAsync`, `OrionCodec`)
- `httpx` async client in `orion-cortex-orch` (already used for Mind)
- Existing schemas: `orion.schemas.substrate_telemetry.SubstrateTierOutcomesPayloadV1`, `orion.core.bus.bus_schemas.BaseEnvelope`

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-substrate-telemetry/` | New service root (Dockerfile, `docker-compose.yml`, `requirements.txt`, `.env_example`, `README` optional — skip README unless user asks). |
| `services/orion-substrate-telemetry/app/settings.py` | Env-driven bus URL, Postgres DSN, port, channel name, table name, retention knobs, optional read API bearer token. |
| `services/orion-substrate-telemetry/app/db.py` | DDL `CREATE TABLE IF NOT EXISTS`, insert row, select latest/history, per-correlation prune SQL, global age-delete SQL. |
| `services/orion-substrate-telemetry/app/service.py` | `SubstrateTelemetryChassis(BaseChassis)` — `start_background`/`_run` subscribe loop, decode, validate kind, persist, never crash loop on bad frames. |
| `services/orion-substrate-telemetry/app/main.py` | FastAPI lifespan starts/stops chassis; GET `/v1/substrate/tier-outcomes/latest` and `/history`; optional auth dependency. |
| `services/orion-substrate-telemetry/Dockerfile` | Same pattern as state-journaler: copy `orion/` + service tree, `python -m app.main`. |
| `services/orion-substrate-telemetry/docker-compose.yml` | Per-service compose: build context `../..`, external `app-net`, env passthrough. |
| `services/orion-substrate-telemetry/tests/test_decode_persist_map.py` | Unit: codec bytes → envelope + payload → dict suitable for INSERT. |
| `services/orion-substrate-telemetry/tests/test_db_queries.py` | Unit: SQL row roundtrip using mocked asyncpg or lightweight integration if CI has Postgres (prefer unit with `asyncpg` testcontainer only if repo already uses it; otherwise mock `conn.fetchrow`). |
| `orion/bus/channels.yaml` | `orion:substrate:tier_outcomes`: real consumers only (`orion-substrate-telemetry`); drop aspirational `orion-mind` / `orion-hub` / `*`. |
| `services/orion-cortex-orch/app/settings.py` | `ORION_SUBSTRATE_TELEMETRY_BASE_URL`, timeout, optional `ORION_SUBSTRATE_TELEMETRY_READ_TOKEN` header value for orch→telemetry. |
| `services/orion-cortex-orch/app/mind_runtime.py` | `fetch_substrate_telemetry_facet_for_mind`, extend `build_mind_run_request` to merge `facets.substrate_telemetry`. |
| `services/orion-cortex-orch/app/orchestrator.py` | Before `build_mind_run_request` / `call_orion_mind_http`, resolve inline vs HTTP fetch using `correlation_id` (string UUID). |
| `services/orion-cortex-orch/docker-compose.yml` | Wire new env vars for local stacks. |
| `services/orion-cortex-orch/tests/test_mind_orch.py` (or new `test_substrate_telemetry_orch.py`) | Assert facet merge and that HTTP fetch helper is skipped when metadata inline dict present. |
| `services/orion-hub/scripts/api_routes.py` (small addition) | Hub-only HTTP: optional proxy GET e.g. `/api/substrate/tier-outcomes/latest` forwarding to `SUBSTRATE_TELEMETRY_BASE_URL` with server-side token (no Redis). |
| `services/orion-hub/docker-compose.yml` | Document `SUBSTRATE_TELEMETRY_BASE_URL` + optional token for the proxy. |

**Design locks (from parent brief):**

- **Idempotency:** Append-only `INSERT` for every delivered bus message; duplicate deliveries create multiple rows; dedupe is explicitly deferred.
- **Orch data path:** Extend `build_mind_run_request(..., substrate_telemetry_facet: dict[str, Any] | None = None)`; orchestrator performs async `httpx` GET when base URL is set and inline override absent, then passes the result into that parameter (single clear merge point).
- **Correlation id:** Query param `correlation_id` is the string form of orch’s `correlation_id` variable (UUID string).
- **Retention:** (1) After each successful insert for a `correlation_id`, delete older rows for that correlation keeping the **100** newest by `(generated_at DESC NULLS LAST, received_at_utc DESC, id DESC)`. (2) Background loop every **3600** s: `DELETE FROM … WHERE received_at_utc < NOW() AT TIME ZONE 'UTC' - INTERVAL '7 days'` (parameterize `RETENTION_DAYS` default 7).

---

### Task 1: Postgres DDL + repository module (new service)

**Files:**

- Create: `services/orion-substrate-telemetry/app/db.py`
- Create: `services/orion-substrate-telemetry/app/settings.py` (minimal placeholders; filled in Task 2)
- Test: `services/orion-substrate-telemetry/tests/test_db_queries.py`

- [ ] **Step 1: Add failing unit test for table DDL string and prune SQL shape**

Create `services/orion-substrate-telemetry/tests/test_db_queries.py`:

```python
from __future__ import annotations

import re

from app.db import LATEST_SQL, PRUNE_CORRELATION_SQL, GLOBAL_RETENTION_SQL, CREATE_TABLE_SQL


def test_sql_contains_expected_fragments() -> None:
    assert "substrate_tier_outcomes_events" in CREATE_TABLE_SQL
    assert "correlation_id" in CREATE_TABLE_SQL
    assert "generated_at" in CREATE_TABLE_SQL.lower()
    assert "ORDER BY" in PRUNE_CORRELATION_SQL
    assert re.search(r"OFFSET\s+(\$2::int|100)", PRUNE_CORRELATION_SQL, re.I)
    assert "received_at_utc" in GLOBAL_RETENTION_SQL
    assert "WHERE correlation_id" in LATEST_SQL.lower()
```

- [ ] **Step 2: Run pytest and confirm failure**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry
PYTHONPATH=. pytest tests/test_db_queries.py::test_sql_contains_expected_fragments -v
```

Expected: `ERROR collecting` or `ModuleNotFoundError: app.db` until Step 3.

- [ ] **Step 3: Implement `app/db.py` with constants and helpers**

Create `services/orion-substrate-telemetry/app/db.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

# Table name matches settings.substrate_telemetry_table in Task 2
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS substrate_tier_outcomes_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID NOT NULL,
    envelope_kind TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    cold_anchors JSONB NOT NULL DEFAULT '[]'::jsonb,
    tier_outcomes JSONB NOT NULL DEFAULT '{}'::jsonb,
    degraded_producers JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_service TEXT,
    source_node TEXT,
    received_at_utc TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
);
CREATE INDEX IF NOT EXISTS idx_substrate_tier_corr_received
  ON substrate_tier_outcomes_events (correlation_id, generated_at DESC, received_at_utc DESC);
"""

INSERT_SQL = """
INSERT INTO substrate_tier_outcomes_events (
  correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
  degraded_producers, source_service, source_node
) VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8)
RETURNING id, received_at_utc;
"""

PRUNE_CORRELATION_SQL = """
DELETE FROM substrate_tier_outcomes_events a
USING (
  SELECT id FROM substrate_tier_outcomes_events
  WHERE correlation_id = $1::uuid
  ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
  OFFSET $2::int
) AS doomed
WHERE a.id = doomed.id;
"""

GLOBAL_RETENTION_SQL = """
DELETE FROM substrate_tier_outcomes_events
WHERE received_at_utc < (NOW() AT TIME ZONE 'utc' - $1::interval);
"""

LATEST_SQL = """
SELECT id, correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
       degraded_producers, source_service, source_node, received_at_utc
FROM substrate_tier_outcomes_events
WHERE correlation_id = $1::uuid
ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
LIMIT 1;
"""

HISTORY_SQL = """
SELECT id, correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
       degraded_producers, source_service, source_node, received_at_utc
FROM substrate_tier_outcomes_events
WHERE correlation_id = $1::uuid
ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
LIMIT $2::int;
"""


async def ensure_schema(conn: asyncpg.Connection, *, table: str = "substrate_tier_outcomes_events") -> None:
    # If settings allow overriding table name, substitute safely in real impl — Task 2 can require fixed name for YAGNI.
    assert table == "substrate_tier_outcomes_events"
    await conn.execute(CREATE_TABLE_SQL)


async def insert_event(
    conn: asyncpg.Connection,
    *,
    correlation_id: UUID,
    envelope_kind: str,
    generated_at: str,
    cold_anchors: list[Any],
    tier_outcomes: dict[str, Any],
    degraded_producers: list[Any],
    source_service: str | None,
    source_node: str | None,
) -> asyncpg.Record:
    return await conn.fetchrow(
        INSERT_SQL,
        correlation_id,
        envelope_kind,
        generated_at,
        cold_anchors,
        tier_outcomes,
        degraded_producers,
        source_service,
        source_node,
    )


async def prune_correlation(
    conn: asyncpg.Connection, *, correlation_id: UUID, keep_newest: int
) -> None:
    await conn.execute(PRUNE_CORRELATION_SQL, correlation_id, int(keep_newest))


async def global_retention_sweep(conn: asyncpg.Connection, *, max_age_days: int) -> str:
    return await conn.execute(GLOBAL_RETENTION_SQL, f"{int(max_age_days)} days")


async def fetch_latest(conn: asyncpg.Connection, *, correlation_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow(LATEST_SQL, correlation_id)


async def fetch_history(conn: asyncpg.Connection, *, correlation_id: UUID, limit: int) -> list[asyncpg.Record]:
    return await conn.fetch(HISTORY_SQL, correlation_id, limit)
```

- [ ] **Step 4: Run pytest**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry
PYTHONPATH=. pytest tests/test_db_queries.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
cd /mnt/scripts/Orion-Sapienform
git add services/orion-substrate-telemetry/app/db.py services/orion-substrate-telemetry/tests/test_db_queries.py
git commit -m "feat(substrate-telemetry): add Postgres DDL and query helpers"
```

---

### Task 2: Service scaffold — settings, requirements, Dockerfile, compose, package init

**Files:**

- Create: `services/orion-substrate-telemetry/app/__init__.py` (empty)
- Create: `services/orion-substrate-telemetry/app/settings.py`
- Create: `services/orion-substrate-telemetry/requirements.txt`
- Create: `services/orion-substrate-telemetry/Dockerfile`
- Create: `services/orion-substrate-telemetry/docker-compose.yml`
- Create: `services/orion-substrate-telemetry/.env_example`
- Modify: `services/orion-substrate-telemetry/app/db.py` — if you parameterized table name, align with `settings.substrate_telemetry_table` (optional; fixed name is acceptable).

- [ ] **Step 1: Add `requirements.txt` (match state-journaler stack)**

Create `services/orion-substrate-telemetry/requirements.txt`:

```
fastapi==0.115.9
uvicorn[standard]==0.30.6
pydantic==2.11.5
pydantic_settings==2.5.2
asyncpg==0.29.0
redis[hiredis]==5.0.7
httpx==0.27.2
pytest==8.3.4
pytest-asyncio==0.25.0
```

- [ ] **Step 2: Add `app/settings.py`**

```python
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("substrate-telemetry", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8395, alias="PORT")

    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    channel_substrate_tier_outcomes: str = Field(
        "orion:substrate:tier_outcomes",
        alias="CHANNEL_SUBSTRATE_TIER_OUTCOMES",
    )

    postgres_uri: str = Field(
        "postgresql://postgres:postgres@localhost:5432/conjourney",
        alias="POSTGRES_URI",
    )
    substrate_telemetry_table: str = Field(
        "substrate_tier_outcomes_events",
        alias="SUBSTRATE_TELEMETRY_TABLE",
    )

    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    retention_days: int = Field(7, alias="SUBSTRATE_TELEMETRY_RETENTION_DAYS")
    retention_scan_interval_sec: float = Field(3600.0, alias="SUBSTRATE_TELEMETRY_RETENTION_SCAN_SEC")
    per_correlation_row_cap: int = Field(100, alias="SUBSTRATE_TELEMETRY_PER_CORR_CAP")

    read_api_token: str | None = Field(None, alias="SUBSTRATE_TELEMETRY_READ_API_TOKEN")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

- [ ] **Step 3: Dockerfile (clone state-journaler pattern)**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-substrate-telemetry/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY orion /app/orion
COPY services/orion-substrate-telemetry /app

CMD ["python", "-m", "app.main"]
```

- [ ] **Step 4: `docker-compose.yml`**

```yaml
services:
  substrate-telemetry:
    build:
      context: ../..
      dockerfile: services/orion-substrate-telemetry/Dockerfile
    container_name: ${PROJECT}-substrate-telemetry
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${SUBSTRATE_TELEMETRY_PORT:-8395}:${SUBSTRATE_TELEMETRY_PORT:-8395}"
    environment:
      - SERVICE_NAME=${SERVICE_NAME:-substrate-telemetry}
      - SERVICE_VERSION=${SERVICE_VERSION:-0.1.0}
      - NODE_NAME=${NODE_NAME}
      - PORT=${SUBSTRATE_TELEMETRY_PORT:-8395}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED}
      - ORION_BUS_ENFORCE_CATALOG=${ORION_BUS_ENFORCE_CATALOG}
      - CHANNEL_SUBSTRATE_TIER_OUTCOMES=${CHANNEL_SUBSTRATE_TIER_OUTCOMES:-orion:substrate:tier_outcomes}
      - POSTGRES_URI=${POSTGRES_URI}
      - SUBSTRATE_TELEMETRY_READ_API_TOKEN=${SUBSTRATE_TELEMETRY_READ_API_TOKEN:-}

networks:
  app-net:
    name: ${NET}
    external: true
```

- [ ] **Step 5: `.env_example`** — copy all `Settings` fields with comments mirroring `orion-state-journaler/.env` style if present; else minimal documented keys from Step 2.

- [ ] **Step 6: Commit**

```bash
git add services/orion-substrate-telemetry/requirements.txt services/orion-substrate-telemetry/Dockerfile services/orion-substrate-telemetry/docker-compose.yml services/orion-substrate-telemetry/.env_example services/orion-substrate-telemetry/app/settings.py services/orion-substrate-telemetry/app/__init__.py
git commit -m "chore(substrate-telemetry): scaffold service package, Docker, and settings"
```

---

### Task 3: `BaseChassis` subscriber — decode, validate, insert, prune, retention loop

**Files:**

- Create: `services/orion-substrate-telemetry/app/service.py`
- Modify: `services/orion-substrate-telemetry/app/db.py` — use `settings.per_correlation_row_cap` in prune if you replace hardcoded `OFFSET 100` (recommended: f-string is forbidden for SQL injection; use two statements or `OFFSET $2`).

- [ ] **Step 1: Failing unit test for message handler (mock bus)**

Create `services/orion-substrate-telemetry/tests/test_decode_persist_map.py`:

```python
from __future__ import annotations

from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1


def test_codec_roundtrip_tier_outcomes_envelope() -> None:
    corr = uuid4()
    payload = SubstrateTierOutcomesPayloadV1(
        generated_at="2026-05-14T12:00:00+00:00",
        cold_anchors=["a1"],
        tier_outcomes={"a1": ["operator_static_protected:2"]},
        degraded_producers=["p1"],
    )
    env = BaseEnvelope(
        kind="substrate.tier_outcomes.v1",
        source=ServiceRef(name="orion-cortex-exec", node="n1"),
        correlation_id=corr,
        payload=payload.model_dump(mode="json"),
    )
    raw = OrionCodec().encode(env)
    dec = OrionCodec().decode(raw)
    assert dec.ok
    assert dec.envelope.kind == "substrate.tier_outcomes.v1"
    body = SubstrateTierOutcomesPayloadV1.model_validate(dec.envelope.payload)
    assert body.cold_anchors == ["a1"]
```

- [ ] **Step 2: Run pytest**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-substrate-telemetry:orion pytest services/orion-substrate-telemetry/tests/test_decode_persist_map.py -v
```

(Adjust `PYTHONPATH` to include repo `orion` package root: `PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry`.)

Expected: pass once imports resolve.

- [ ] **Step 3: Implement `app/service.py`**

Key implementation sketch (complete in repo):

```python
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg
from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.codec import OrionCodec
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1

from . import db
from .settings import settings

logger = logging.getLogger("orion.substrate.telemetry")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SubstrateTelemetryService(BaseChassis):
    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
            )
        )
        self.codec = OrionCodec()
        self.bus.codec = self.codec

    async def _retention_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn = await asyncpg.connect(dsn=settings.postgres_uri)
                try:
                    await db.ensure_schema(conn)
                    await db.global_retention_sweep(conn, max_age_days=settings.retention_days)
                finally:
                    await conn.close()
            except Exception as exc:
                logger.warning("substrate_telemetry_retention_sweep_failed err=%s", exc)
            try:
                await asyncio.sleep(float(settings.retention_scan_interval_sec))
            except asyncio.CancelledError:
                break

    async def _handle_message(self, data: bytes | None) -> None:
        if not data:
            return
        decoded = self.codec.decode(data)
        if not decoded.ok or decoded.envelope is None:
            logger.debug("substrate_telemetry_decode_skip")
            return
        env = decoded.envelope
        if env.kind != "substrate.tier_outcomes.v1":
            return
        try:
            payload = SubstrateTierOutcomesPayloadV1.model_validate(env.payload)
        except Exception as exc:
            logger.warning("substrate_telemetry_payload_invalid err=%s", exc)
            return
        src = env.source
        conn = await asyncpg.connect(dsn=settings.postgres_uri)
        try:
            await db.ensure_schema(conn)
            await db.insert_event(
                conn,
                correlation_id=env.correlation_id,
                envelope_kind=env.kind,
                generated_at=payload.generated_at,
                cold_anchors=list(payload.cold_anchors),
                tier_outcomes=dict(payload.tier_outcomes),
                degraded_producers=list(payload.degraded_producers),
                source_service=src.name if src else None,
                source_node=src.node if src else None,
            )
            await db.prune_correlation(
                conn,
                correlation_id=env.correlation_id,
                keep_newest=settings.per_correlation_row_cap,
            )
        finally:
            await conn.close()

    async def _run(self) -> None:
        ret_task = asyncio.create_task(self._retention_loop(), name="substrate-telemetry-retention")
        try:
            async with self.bus.subscribe(settings.channel_substrate_tier_outcomes) as pubsub:
                async for msg in self.bus.iter_messages(pubsub):
                    if self._stop.is_set():
                        break
                    try:
                        await self._handle_message(msg.get("data"))
                    except Exception as exc:
                        logger.warning("substrate_telemetry_message_handler_err err=%s", exc)
        finally:
            ret_task.cancel()
            try:
                await ret_task
            except Exception:
                pass
```

- [ ] **Step 4: Run pytest**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry pytest services/orion-substrate-telemetry/tests/test_decode_persist_map.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/orion-substrate-telemetry/app/service.py services/orion-substrate-telemetry/app/db.py services/orion-substrate-telemetry/tests/test_decode_persist_map.py
git commit -m "feat(substrate-telemetry): subscribe, validate, and persist tier outcome events"
```

---

### Task 4: FastAPI app — lifespan, GET latest/history, optional read token

**Files:**

- Create: `services/orion-substrate-telemetry/app/main.py`
- Test: extend `tests/test_db_queries.py` or add `tests/test_api_contract.py` with `httpx.AsyncClient` + `lifespan` mocks (optional); minimum: manual curl steps in plan verified in Task 7 integration.

- [ ] **Step 1: Implement `app/main.py`**

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from .service import SubstrateTelemetryService
from .settings import settings
from . import db

svc = SubstrateTelemetryService()


def _optional_auth(x_telemetry_token: str | None = Header(default=None, alias="X-Telemetry-Token")) -> None:
    expected = settings.read_api_token
    if expected:
        if (x_telemetry_token or "").strip() != expected.strip():
            raise HTTPException(status_code=401, detail="unauthorized")


def _row_to_json(r: asyncpg.Record) -> dict[str, Any]:
    return {
        "id": str(r["id"]),
        "correlation_id": str(r["correlation_id"]),
        "envelope_kind": r["envelope_kind"],
        "generated_at": r["generated_at"],
        "cold_anchors": r["cold_anchors"],
        "tier_outcomes": r["tier_outcomes"],
        "degraded_producers": r["degraded_producers"],
        "source_service": r["source_service"],
        "source_node": r["source_node"],
        "received_at_utc": r["received_at_utc"].isoformat() if r["received_at_utc"] else None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start_background()
    try:
        yield
    finally:
        await svc.stop()


app = FastAPI(title="orion-substrate-telemetry", lifespan=lifespan)


@app.get("/v1/substrate/tier-outcomes/latest")
async def latest(
    correlation_id: str = Query(..., description="UUID string matching bus envelope"),
    _: None = Depends(_optional_auth),
):
    try:
        cid = UUID(correlation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_correlation_id")
    conn = await asyncpg.connect(dsn=settings.postgres_uri)
    try:
        await db.ensure_schema(conn)
        row = await db.fetch_latest(conn, correlation_id=cid)
    finally:
        await conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    return JSONResponse(_row_to_json(row))


@app.get("/v1/substrate/tier-outcomes/history")
async def history(
    correlation_id: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
    _: None = Depends(_optional_auth),
):
    try:
        cid = UUID(correlation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_correlation_id")
    conn = await asyncpg.connect(dsn=settings.postgres_uri)
    try:
        await db.ensure_schema(conn)
        rows = await db.fetch_history(conn, correlation_id=cid, limit=limit)
    finally:
        await conn.close()
    return JSONResponse({"correlation_id": str(cid), "items": [_row_to_json(r) for r in rows]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
```

- [ ] **Step 2: Smoke run (developer machine)**

```bash
cd /mnt/scripts/Orion-Sapienform
export PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry
export POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/conjourney
# Requires local Postgres or skip if unavailable
python -m uvicorn app.main:app --app-dir services/orion-substrate-telemetry --port 8395 &
sleep 2
curl -sS "http://127.0.0.1:8395/v1/substrate/tier-outcomes/latest?correlation_id=00000000-0000-0000-0000-000000000001" | head
```

Expected: `404` JSON or FastAPI detail until a row exists.

- [ ] **Step 3: Commit**

```bash
git add services/orion-substrate-telemetry/app/main.py
git commit -m "feat(substrate-telemetry): expose HTTP read API for tier outcomes"
```

---

### Task 5: Bus catalog — real consumers only

**Files:**

- Modify: `orion/bus/channels.yaml` (block near `orion:substrate:tier_outcomes`, lines 1744–1751)

- [ ] **Step 1: Patch `consumer_services`**

Replace:

```yaml
    consumer_services: ["orion-mind", "orion-hub", "*"]
```

with:

```yaml
    consumer_services: ["orion-substrate-telemetry"]
```

- [ ] **Step 2: Commit**

```bash
git add orion/bus/channels.yaml
git commit -m "docs(bus): list orion-substrate-telemetry as tier outcomes consumer"
```

---

### Task 6: `orion-cortex-orch` — fetch + `build_mind_run_request` merge

**Files:**

- Modify: `services/orion-cortex-orch/app/mind_runtime.py`
- Modify: `services/orion-cortex-orch/app/settings.py`
- Modify: `services/orion-cortex-orch/app/orchestrator.py` (mind block ~540)
- Modify: `services/orion-cortex-orch/docker-compose.yml`

- [ ] **Step 1: Extend settings**

In `services/orion-cortex-orch/app/settings.py`, after `orion_mind_max_response_bytes`:

```python
    orion_substrate_telemetry_base_url: str = Field("", alias="ORION_SUBSTRATE_TELEMETRY_BASE_URL")
    orion_substrate_telemetry_timeout_sec: float = Field(2.0, alias="ORION_SUBSTRATE_TELEMETRY_TIMEOUT_SEC")
    orion_substrate_telemetry_read_token: str = Field("", alias="ORION_SUBSTRATE_TELEMETRY_READ_TOKEN")
```

- [ ] **Step 2: Add fetch helper + extend builder in `mind_runtime.py`**

Add near top after imports:

```python
async def fetch_substrate_telemetry_facet_for_mind(correlation_id: str) -> dict[str, Any] | None:
    """GET latest row from telemetry service; return None on 404/misconfig so Mind omits facet."""
    s = get_settings()
    base = (s.orion_substrate_telemetry_base_url or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/v1/substrate/tier-outcomes/latest"
    params = {"correlation_id": correlation_id}
    headers = {}
    tok = (s.orion_substrate_telemetry_read_token or "").strip()
    if tok:
        headers["X-Telemetry-Token"] = tok
    timeout = httpx.Timeout(min(10.0, float(s.orion_substrate_telemetry_timeout_sec)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        row = resp.json()
    return {
        "status": "present",
        "generated_at": row.get("generated_at"),
        "cold_anchors": row.get("cold_anchors"),
        "tier_outcomes": row.get("tier_outcomes"),
        "degraded_producers": row.get("degraded_producers"),
        "source_service": row.get("source_service"),
        "source_node": row.get("source_node"),
        "received_at_utc": row.get("received_at_utc"),
        "row_id": row.get("id"),
    }
```

Change `build_mind_run_request` signature and body (preserve existing logic, then merge):

```python
def build_mind_run_request(
    client_request: CortexClientRequest,
    plan_request: PlanExecutionRequest,
    correlation_id: str,
    *,
    substrate_telemetry_facet: dict[str, Any] | None = None,
) -> MindRunRequestV1:
    # ... existing snapshot construction into `snapshot` ...
    meta = client_request.context.metadata if isinstance(client_request.context.metadata, dict) else {}
    if substrate_telemetry_facet is not None:
        facets: dict[str, Any] = dict(snapshot.get("facets") or {}) if isinstance(snapshot.get("facets"), dict) else {}
        facets["substrate_telemetry"] = substrate_telemetry_facet
        snapshot["facets"] = facets
    # ... return MindRunRequestV1(..., snapshot_inputs=snapshot, ...)
```

- [ ] **Step 3: Orchestrator wiring**

Inside `call_verb_runtime`, in the block `if _mind_enabled_exact(cr_meta):` **before** `build_mind_run_request`:

```python
        substrate_facet: dict[str, Any] | None = None
        inline = cr_meta.get("substrate_telemetry_facet")
        if isinstance(inline, dict):
            substrate_facet = inline
        else:
            substrate_facet = await fetch_substrate_telemetry_facet_for_mind(correlation_id)
        mind_req = build_mind_run_request(
            client_request,
            plan_request,
            correlation_id,
            substrate_telemetry_facet=substrate_facet,
        )
```

Note: when `ORION_SUBSTRATE_TELEMETRY_BASE_URL` is empty, `fetch_substrate_telemetry_facet_for_mind` returns `None` and facet is omitted (neutral semantics).

- [ ] **Step 4: Add unit test**

In `services/orion-cortex-orch/tests/test_mind_orch.py` (new test function):

```python
def test_build_mind_run_request_merges_substrate_telemetry_facet() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request
    from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientContext
    from orion.schemas.cortex.schemas import PlanExecutionRequest
    from orion.schemas.cortex.types import ExecutionPlan, ExecutionStep

    cr = CortexClientRequest(
        verb="chat",
        mode="brain",
        context=CortexClientContext(
            session_id="s",
            trace_id="t",
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    plan = ExecutionPlan(
        verb_name="chat",
        steps=[
            ExecutionStep(
                verb_name="chat",
                step_name="noop",
                order=0,
                services=[],
            )
        ],
    )
    pr = PlanExecutionRequest(plan=plan, context={})
    req = build_mind_run_request(
        cr,
        pr,
        "550e8400-e29b-41d4-a716-446655440000",
        substrate_telemetry_facet={"status": "absent"},
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("substrate_telemetry") == {"status": "absent"}
```

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-cortex-orch pytest tests/test_mind_orch.py::test_build_mind_run_request_merges_substrate_telemetry_facet -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-orch/app/settings.py services/orion-cortex-orch/app/mind_runtime.py services/orion-cortex-orch/app/orchestrator.py services/orion-cortex-orch/tests/test_mind_orch.py services/orion-cortex-orch/docker-compose.yml
git commit -m "feat(cortex-orch): merge persisted substrate tier telemetry into Mind snapshot"
```

Add to `docker-compose.yml` under `environment:`:

```yaml
      ORION_SUBSTRATE_TELEMETRY_BASE_URL: ${ORION_SUBSTRATE_TELEMETRY_BASE_URL:-}
      ORION_SUBSTRATE_TELEMETRY_TIMEOUT_SEC: ${ORION_SUBSTRATE_TELEMETRY_TIMEOUT_SEC:-2.0}
      ORION_SUBSTRATE_TELEMETRY_READ_TOKEN: ${ORION_SUBSTRATE_TELEMETRY_READ_TOKEN:-}
```

---

### Task 7: Optional integration test (Redis → DB → GET)

**Files:**

- Create: `services/orion-substrate-telemetry/tests/test_integration_redis_http.py` (mark `@pytest.mark.integration` and skip unless `RUN_INTEGRATION=1`)

- [ ] **Step 1: Implement gated test** that publishes one `OrionCodec` message to real `ORION_BUS_URL`, waits (`asyncio.sleep(2)`), GETs latest — **only** when env set; default skip.

- [ ] **Step 2: Run default CI path**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry
PYTHONPATH=/mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform/services/orion-substrate-telemetry pytest tests/ -v -m "not integration"
```

Expected: all non-integration tests passed.

- [ ] **Step 3: Commit**

```bash
git add services/orion-substrate-telemetry/tests/test_integration_redis_http.py
git commit -m "test(substrate-telemetry): add optional Redis+HTTP integration coverage"
```

---

### Task 8: Hub thin HTTP proxy (no bus)

**Files:**

- Modify: `services/orion-hub/scripts/api_routes.py` — add `@router.get("/api/substrate/tier-outcomes/latest")` calling `httpx.get` with `settings.substrate_telemetry_base_url` (new setting in hub settings module if one exists; else read `os.environ` in route handler for YAGNI).
- Modify: `services/orion-hub/docker-compose.yml` — `SUBSTRATE_TELEMETRY_BASE_URL`, `SUBSTRATE_TELEMETRY_READ_TOKEN`.

Example route body:

```python
@router.get("/api/substrate/tier-outcomes/latest")
def api_substrate_tier_outcomes_latest(correlation_id: str = Query(...)) -> dict[str, Any]:
    import os
    import httpx

    base = (os.environ.get("SUBSTRATE_TELEMETRY_BASE_URL") or "").rstrip("/")
    if not base:
        return {"ok": False, "error": "substrate_telemetry_unconfigured"}
    headers = {}
    tok = (os.environ.get("SUBSTRATE_TELEMETRY_READ_TOKEN") or "").strip()
    if tok:
        headers["X-Telemetry-Token"] = tok
    url = f"{base}/v1/substrate/tier-outcomes/latest"
    r = httpx.get(url, params={"correlation_id": correlation_id}, headers=headers, timeout=5.0)
    if r.status_code == 404:
        return {"ok": True, "status": "absent"}
    r.raise_for_status()
    return {"ok": True, "data": r.json()}
```

- [ ] **Step 1: Implement route + compose env**

- [ ] **Step 2: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/docker-compose.yml
git commit -m "feat(hub): proxy substrate tier telemetry HTTP for UI consumers"
```

---

### Task 9: Align auth header between services

**Files:**

- Modify: `services/orion-substrate-telemetry/app/main.py` — use same header name as orch (`X-Telemetry-Token`) in `_optional_auth`.

Ensure `orion-cortex-orch` `fetch_substrate_telemetry_facet_for_mind` uses identical header.

- [ ] **Step 1: Commit**

```bash
git commit -am "fix(substrate-telemetry): align read token header with cortex-orch client"
```

(If policy prefers single commit with Task 4/6, fold header name into those tasks instead.)

---

### Task 10: Mind facet ordering note (no code change required unless policies added)

**Files:**

- None required for v1 deterministic engine (it does not yet branch on `substrate_telemetry` content). If future Mind policies read this facet, keep key **`substrate_telemetry`** (not `substrate`) to avoid collision with `_FACET_ORDER`’s `"substrate"` slot in `services/orion-mind/app/engine.py`.

- [ ] **Step 1: Verify** `grep -n "substrate_telemetry" services/orion-mind/app/engine.py` returns no hits post-change (expected); no commit unless Mind starts consuming the facet.

---

## Self-review

**Spec coverage mapping**

- **Keep existing publish path unchanged** — Task list never edits `orion/substrate/tier_outcomes_bus.py` or `publish_substrate_tier_outcomes_sync`; verified by omission.
- **New service subscribes + persists + HTTP read** — Tasks 2–4, 3.
- **cortex-orch / Mind obtain data via pull + inline override** — Task 6 (`substrate_telemetry_facet` param + metadata inline + GET).
- **Hub HTTP only, no bus** — Task 8.
- **Channel catalog reflects real subscribers** — Task 5.
- **Testing: unit decode/map, optional integration, contract** — Tasks 1, 3, 7; OpenAPI auto from FastAPI (`/openapi.json`) for Hub/orch consumers — document in runbooks; static fixture optional in Task 7.
- **Verification scenarios (cold path GET, warm 404, Mind no error)** — Task 7 manual checklist: after cold-path chat, `GET .../latest?correlation_id=<orch corr>` returns `200`; warm-only returns `404` and orch leaves facet absent; Mind tests unchanged aside from new facet optional data.

**Placeholder scan:** none found.

**Type consistency:** `BaseEnvelope.correlation_id` is `UUID`; HTTP query uses `str`; Postgres column `UUID`; orch `correlation_id` variable matches string passed to Mind request and query param. Header token name aligned in Task 9. `MindRunRequestV1.snapshot_inputs` remains `dict[str, Any]` per `orion/mind/v1.py`.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-14-substrate-tier-telemetry-persistence.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task; review between tasks.
2. **Inline Execution** — use executing-plans with checkpoints.

Which approach?
