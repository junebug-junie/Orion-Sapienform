# Mesh Bus Resilience + Auto-Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect half-dead bus consumers on the chat critical path, surface incidents in Hub Pending Attention, and auto-remediate via tiered docker compose actions — backed by shared bus resilience in core and `/ready` probes on rostered services.

**Architecture:** Phase A lifts `publish_with_reconnect` and chassis heartbeat reconnect into `orion/core/bus/`, adds `GET /ready` on P0 chat-path services, and seeds `config/mesh_remediation_roster.yaml` plus an `orion-mesh-guardian` skeleton. Phase B runs equilibrium + active probes through a per-service state machine and fires `notify.attention_request` (observe-only by default). Phase C executes tier-1/tier-2 compose remediation with Redis-backed cooldown state.

**Tech Stack:** Python 3.12, FastAPI, Pydantic Settings, Redis pub/sub (`PUBSUB NUMSUB`), `EquilibriumSnapshotV1`, `NotifyClient`, docker compose via `/var/run/docker.sock`, pytest.

**Phasing:** Each phase ships independently testable software. Stop after Phase B for observe-only production; enable Phase C only after live-stack proof.

---

## File structure

| File | Responsibility |
|------|----------------|
| `orion/core/bus/resilience.py` | Shared `publish_with_reconnect` |
| `orion/core/bus/tests/test_resilience.py` | Unit tests for reconnect publish |
| `orion/core/bus/bus_service_chassis.py` | Heartbeat reconnect on publish failure |
| `tests/test_chassis_heartbeat_reconnect.py` | Heartbeat reconnect regression |
| `services/orion-landing-pad/app/bus_resilience.py` | Thin re-export from core |
| `services/orion-landing-pad/app/main.py` | `GET /ready` with RPC smoke |
| `services/orion-landing-pad/tests/test_ready.py` | Landing pad readiness tests |
| `services/orion-cortex-gateway/app/main.py` | `GET /ready` using intake bus redis |
| `services/orion-cortex-gateway/tests/test_ready.py` | Gateway readiness tests |
| `services/orion-cortex-orch/app/health_http.py` | Background FastAPI `/health` + `/ready` |
| `services/orion-cortex-orch/tests/test_ready.py` | Orch readiness tests |
| `services/orion-cortex-exec/app/health_http.py` | Background FastAPI `/health` + `/ready` |
| `services/orion-cortex-exec/tests/test_ready.py` | Exec readiness tests |
| `config/mesh_remediation_roster.yaml` | v1 chat-path roster |
| `services/orion-mesh-guardian/**` | Guardian service (watch, decide, act, notify) |
| `scripts/smoke_mesh_guardian.sh` | Optional live-stack acceptance helper |

Reference implementations already in repo:
- `/ready` pattern: `services/orion-llm-gateway/app/main.py`, `services/orion-recall/app/main.py`
- Notify attention: `services/orion-actions/app/main.py` `_send_pending_attention`
- Compose env-file pattern: `services/orion-hub/scripts/service_logs.py` `build_compose_logs_command`
- Consumer readiness: `orion/bus/consumer_readiness.py`

---

# Phase A — Foundation

---

### Task 1: Lift `publish_with_reconnect` to core

**Files:**
- Create: `orion/core/bus/resilience.py`
- Create: `orion/core/bus/tests/test_resilience.py`

- [ ] **Step 1: Write the failing test**

Create `orion/core/bus/tests/test_resilience.py`:

```python
from __future__ import annotations

import pytest

from orion.core.bus.resilience import publish_with_reconnect


class _FlakyBus:
    def __init__(self) -> None:
        self.publish_calls = 0
        self.reconnect_calls = 0
        self.last_channel = ""

    async def publish(self, channel: str, msg: object) -> None:
        self.publish_calls += 1
        if self.publish_calls == 1:
            raise TimeoutError("Timeout connecting to server")
        self.last_channel = channel

    async def reconnect(self) -> None:
        self.reconnect_calls += 1


@pytest.mark.asyncio
async def test_publish_with_reconnect_retries_after_transport_error() -> None:
    bus = _FlakyBus()
    await publish_with_reconnect(bus, "orion:pad:stats", {"ok": True}, log_label="test")
    assert bus.reconnect_calls == 1
    assert bus.publish_calls == 2
    assert bus.last_channel == "orion:pad:stats"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/core/bus/tests/test_resilience.py -v
```
Expected: FAIL — `ModuleNotFoundError: orion.core.bus.resilience`

- [ ] **Step 3: Write minimal implementation**

Create `orion/core/bus/resilience.py` (copy from `services/orion-landing-pad/app/bus_resilience.py`, change logger to stdlib or keep loguru — match `async_service.py` style in same package; loguru is fine if already a root dep):

```python
from __future__ import annotations

from typing import Any

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope


async def publish_with_reconnect(
    bus: OrionBusAsync,
    channel: str,
    msg: BaseEnvelope | dict[str, Any],
    *,
    log_label: str = "bus_publish",
) -> None:
    """Publish once; on transport failure reconnect the command client and retry."""
    try:
        await bus.publish(channel, msg)
    except Exception as exc:
        logger.warning("{} failed channel={} err={}; reconnecting", log_label, channel, exc)
        await bus.reconnect()
        await bus.publish(channel, msg)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/core/bus/tests/test_resilience.py -v
```
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add orion/core/bus/resilience.py orion/core/bus/tests/test_resilience.py
git commit -m "feat(bus): lift publish_with_reconnect to core"
```

---

### Task 2: Chassis heartbeat reconnect on publish failure

**Files:**
- Modify: `orion/core/bus/bus_service_chassis.py:120-144`
- Create: `tests/test_chassis_heartbeat_reconnect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chassis_heartbeat_reconnect.py`:

```python
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name="test-rabbit",
        service_version="0.1.0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
        heartbeat_interval_sec=0.05,
    )


@pytest.mark.asyncio
async def test_heartbeat_reconnects_and_retries_publish() -> None:
    rabbit = Rabbit(_cfg(), request_channel="orion:test:request", handler=lambda _env: None)
    rabbit.bus = MagicMock()
    rabbit.bus.publish = AsyncMock(side_effect=[RuntimeError("connection lost"), None])
    rabbit.bus.reconnect = AsyncMock()
    rabbit._stop.set()

    with patch.object(rabbit, "_source", return_value=MagicMock()):
        await rabbit._heartbeat_loop()

    assert rabbit.bus.reconnect.await_count == 1
    assert rabbit.bus.publish.await_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_chassis_heartbeat_reconnect.py -v
```
Expected: FAIL — `reconnect.await_count == 0`

- [ ] **Step 3: Write minimal implementation**

In `orion/core/bus/bus_service_chassis.py`, replace the except block in `_heartbeat_loop`:

```python
            except Exception as e:
                logger.warning("Heartbeat publish failed: %s; reconnecting", e)
                try:
                    await self.bus.reconnect()
                    await self.bus.publish(self.cfg.health_channel, v1_env)
                except Exception as retry_exc:
                    logger.warning("Heartbeat publish retry failed: %s", retry_exc)
```

Note: `v1_env` must remain in scope inside the except — it is built in the try block above; if refactor needed, build envelope before publish attempt.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_chassis_heartbeat_reconnect.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/core/bus/bus_service_chassis.py tests/test_chassis_heartbeat_reconnect.py
git commit -m "fix(chassis): reconnect bus client on heartbeat publish failure"
```

---

### Task 3: Landing pad imports core resilience

**Files:**
- Modify: `services/orion-landing-pad/app/bus_resilience.py`
- Modify: `services/orion-landing-pad/tests/test_bus_resilience.py`

- [ ] **Step 1: Replace landing pad module with re-export**

Replace contents of `services/orion-landing-pad/app/bus_resilience.py`:

```python
from orion.core.bus.resilience import publish_with_reconnect

__all__ = ["publish_with_reconnect"]
```

- [ ] **Step 2: Simplify landing pad test to import from app module**

Replace `services/orion-landing-pad/tests/test_bus_resilience.py` loader hack with:

```python
from __future__ import annotations

import pytest

from app.bus_resilience import publish_with_reconnect


class _FlakyBus:
    ...
```

(keep `_FlakyBus` and assertion body unchanged from current file)

- [ ] **Step 3: Run tests**

Run:
```bash
./scripts/test_service.sh orion-landing-pad services/orion-landing-pad/tests/test_bus_resilience.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add services/orion-landing-pad/app/bus_resilience.py services/orion-landing-pad/tests/test_bus_resilience.py
git commit -m "refactor(landing-pad): import publish_with_reconnect from core"
```

---

### Task 4: Landing pad `GET /ready`

**Files:**
- Modify: `services/orion-landing-pad/app/main.py`
- Create: `services/orion-landing-pad/tests/test_ready.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-landing-pad/tests/test_ready.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ready_503_when_bus_not_started(client: TestClient) -> None:
    with patch("app.main.service") as mock_service:
        mock_service.bus = MagicMock()
        mock_service.bus.enabled = False
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


def test_ready_200_when_consumer_ready(client: TestClient) -> None:
    fake_redis = MagicMock()
    with patch("app.main.service") as mock_service, patch(
        "app.main.check_bus_consumer_readiness",
        new=AsyncMock(
            return_value=MagicMock(
                ok=True,
                bus_consumer_ready=True,
                subscriber_count=1,
                intake_channel="orion:pad:rpc:request",
                dependency_status="available",
                error=None,
                heartbeat_fresh=True,
                rpc_smoke_ok=True,
            )
        ),
    ), patch("app.main.bus_consumer_readiness_v1") as mock_v1:
        mock_service.bus = MagicMock(enabled=True, redis=fake_redis)
        mock_service.settings.pad_rpc_request_channel = "orion:pad:rpc:request"
        mock_service.settings.app_name = "landing-pad"
        mock_service.settings.orion_health_channel = "orion:system:health"
        mock_service.settings.heartbeat_interval_sec = 10
        mock_v1.return_value = MagicMock(ok=True, model_dump=lambda mode: {"ok": True})
        resp = client.get("/ready")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test — expect FAIL**

Run:
```bash
./scripts/test_service.sh orion-landing-pad services/orion-landing-pad/tests/test_ready.py -v
```
Expected: FAIL — `/ready` route missing

- [ ] **Step 3: Implement `/ready` in main.py**

Add imports and endpoint to `services/orion-landing-pad/app/main.py`:

```python
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1


@app.get("/ready")
async def ready() -> JSONResponse:
    if not service.bus.enabled:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=service.settings.pad_rpc_request_channel,
            subscriber_count=0,
            dependency_status="unavailable",
            error="bus not connected",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(service.bus, "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=service.settings.pad_rpc_request_channel,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    async def _rpc_smoke() -> bool:
        return service._tasks and not service._tasks[0].done()

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=service.settings.pad_rpc_request_channel,
        service_name=service.settings.app_name,
        health_channel=service.settings.orion_health_channel,
        heartbeat_ttl_sec=float(service.settings.heartbeat_interval_sec) * 3.0,
        check_heartbeat=True,
        rpc_smoke_fn=_rpc_smoke,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)
```

Adjust `_rpc_smoke` to check an explicit `service._rpc_server_running` flag if cleaner — add `bool` property on `LandingPadService` that returns whether RPC task is alive.

- [ ] **Step 4: Run tests — expect PASS**

Run:
```bash
./scripts/test_service.sh orion-landing-pad services/orion-landing-pad/tests/test_ready.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-landing-pad/app/main.py services/orion-landing-pad/tests/test_ready.py
git commit -m "feat(landing-pad): expose bus consumer /ready endpoint"
```

---

### Task 5: Cortex gateway `GET /ready`

**Files:**
- Modify: `services/orion-cortex-gateway/app/main.py`
- Create: `services/orion-cortex-gateway/tests/test_ready.py`

- [ ] **Step 1: Write failing test** (mirror llm-gateway pattern; mock `bus_client._intake_bus.redis`)

- [ ] **Step 2: Run test — FAIL**

Run:
```bash
./scripts/test_service.sh orion-cortex-gateway services/orion-cortex-gateway/tests/test_ready.py -v
```

- [ ] **Step 3: Add `/ready` endpoint**

Add to `services/orion-cortex-gateway/app/main.py`:

```python
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1


@app.get("/ready")
async def ready() -> JSONResponse:
    intake_bus = getattr(bus_client, "_intake_bus", None)
    if intake_bus is None or not getattr(intake_bus, "enabled", False):
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_gateway_request,
            subscriber_count=0,
            dependency_status="unavailable",
            error="intake bus not connected",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(intake_bus, "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_gateway_request,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=settings.channel_gateway_request,
        service_name=settings.service_name,
        check_heartbeat=False,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)
```

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(cortex-gateway): expose bus consumer /ready endpoint"
```

---

### Task 6: Cortex orch background health HTTP + `/ready`

**Files:**
- Create: `services/orion-cortex-orch/app/health_http.py`
- Modify: `services/orion-cortex-orch/app/main.py`
- Create: `services/orion-cortex-orch/tests/test_ready.py`
- Modify: `services/orion-cortex-orch/docker-compose.yml` (ensure `app-net`, port 8072 published — already present)

Bus-first orch runs `python -m app.main`; add a background uvicorn task.

- [ ] **Step 1: Write failing test for health_http module**

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.health_http import create_health_app


def test_ready_503_without_redis() -> None:
    app = create_health_app(redis_getter=lambda: None, intake_channel="orion:cortex:request")
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503
```

- [ ] **Step 2: Run test — FAIL**

- [ ] **Step 3: Implement `health_http.py`**

```python
from __future__ import annotations

from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1


def create_health_app(
    *,
    redis_getter: Callable[[], object | None],
    intake_channel: str,
    service_name: str,
    service_version: str,
) -> FastAPI:
    app = FastAPI(title="cortex-orch-health", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": service_name, "version": service_version}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        redis = redis_getter()
        if redis is None:
            body = BusConsumerReadinessV1(
                ok=False,
                http_alive=True,
                bus_consumer_ready=False,
                intake_channel=intake_channel,
                subscriber_count=0,
                dependency_status="unavailable",
                error="redis unavailable",
            )
            return JSONResponse(body.model_dump(mode="json"), status_code=503)

        result = await check_bus_consumer_readiness(
            redis,
            intake_channel=intake_channel,
            service_name=service_name,
            check_heartbeat=False,
        )
        body = bus_consumer_readiness_v1(result, http_alive=True)
        status_code = 200 if body.ok else 503
        return JSONResponse(body.model_dump(mode="json"), status_code=status_code)

    return app
```

Add helper in same file:

```python
import asyncio

import uvicorn


async def start_health_server(*, app: FastAPI, host: str, port: int) -> asyncio.Task:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    return asyncio.create_task(server.serve(), name="orch-health-http")
```

- [ ] **Step 4: Wire into `main.py` `main()`**

After `await svc.bus.connect()`:

```python
    from .health_http import create_health_app, start_health_server

    health_app = create_health_app(
        redis_getter=lambda: getattr(svc.bus, "redis", None),
        intake_channel=s.channel_cortex_request,
        service_name=s.service_name,
        service_version=s.service_version,
    )
    health_task = await start_health_server(
        app=health_app,
        host=s.api_host,
        port=int(s.api_port),
    )
```

Include `health_task` in `asyncio.gather(...)` or cancel on shutdown.

- [ ] **Step 5: Run tests + compile**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m compileall services/orion-cortex-orch
./scripts/test_service.sh orion-cortex-orch services/orion-cortex-orch/tests/test_ready.py -v
```

- [ ] **Step 6: Commit**

---

### Task 7: Cortex exec background health HTTP + `/ready`

**Files:**
- Create: `services/orion-cortex-exec/app/health_http.py` (same pattern as orch; intake = `settings.channel_exec_request`)
- Modify: `services/orion-cortex-exec/app/main.py`
- Modify: `services/orion-cortex-exec/app/settings.py` — add `health_http_port: int = Field(8070, alias="HEALTH_HTTP_PORT")`
- Modify: `services/orion-cortex-exec/docker-compose.yml` — publish `8070:8070`, pass `HEALTH_HTTP_PORT`
- Modify: `services/orion-cortex-exec/.env_example` — document `HEALTH_HTTP_PORT=8070`
- Create: `services/orion-cortex-exec/tests/test_ready.py`

Roster ready URL: `http://${PROJECT}-cortex-exec:8070/ready`

After `.env_example` change run:
```bash
python scripts/sync_local_env_from_example.py
```

- [ ] Follow same TDD steps as Task 6
- [ ] Commit: `feat(cortex-exec): expose bus consumer /ready on HEALTH_HTTP_PORT`

---

### Task 8: Verify recall `/ready` (no code unless gap found)

**Files:**
- Read: `services/orion-recall/app/main.py` (already has `/ready`)
- Optional: `services/orion-recall/tests/test_ready.py` if missing coverage

- [ ] **Step 1: Run existing recall tests**

```bash
./scripts/test_service.sh orion-recall -q -k ready
```

- [ ] **Step 2: Manual curl against running stack** (document in commit message if no code change)

```bash
curl -sf "http://${PROJECT}-recall:8260/ready" | jq .
```

If PORT differs, read `services/orion-recall/.env_example` for `PORT`.

- [ ] **Step 3: Commit only if tests added**

---

### Task 9: Seed roster YAML

**Files:**
- Create: `config/mesh_remediation_roster.yaml`

- [ ] **Step 1: Add v1 chat-path roster**

```yaml
services:
  - id: landing-pad
    heartbeat_name: landing-pad
    compose_dir: orion-landing-pad
    compose_service: orion-landing-pad
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:pad:rpc:request
      ready_url: "http://${PROJECT}-landing-pad:8370/ready"
      service_name: landing-pad

  - id: cortex-gateway
    heartbeat_name: cortex-gateway
    compose_dir: orion-cortex-gateway
    compose_service: cortex-gateway
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:cortex:gateway:request
      ready_url: "http://${PROJECT}-cortex-gateway:8022/ready"
      service_name: cortex-gateway

  - id: cortex-orch
    heartbeat_name: cortex-orch
    compose_dir: orion-cortex-orch
    compose_service: cortex-orchestrator
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:cortex:request
      ready_url: "http://${PROJECT}-cortex-orch:8072/ready"
      service_name: cortex-orch

  - id: cortex-exec
    heartbeat_name: cortex-exec
    compose_dir: orion-cortex-exec
    compose_service: cortex-exec
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:cortex:exec:request
      ready_url: "http://${PROJECT}-cortex-exec:8070/ready"
      service_name: cortex-exec

  - id: recall
    heartbeat_name: recall
    compose_dir: orion-recall
    compose_service: recall
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:exec:request:RecallService
      ready_url: "http://${PROJECT}-recall:8260/ready"
      service_name: recall

  - id: llm-gateway
    heartbeat_name: llm-gateway
    compose_dir: orion-llm-gateway
    compose_service: llm-gateway
    include_bus_env: false
    auto_remediate: true
    probe:
      mode: redis_and_http
      intake_channels:
        - orion:exec:request:LLMGatewayService
      ready_url: "http://${PROJECT}-llm-gateway:8210/ready"
      service_name: llm-gateway

  - id: notify
    heartbeat_name: notify
    compose_dir: orion-notify
    compose_service: notify
    include_bus_env: false
    auto_remediate: false
    probe:
      mode: http
      intake_channels: []
      ready_url: "http://${PROJECT}-notify:7140/health"
      service_name: notify

  - id: equilibrium
    heartbeat_name: orion-equilibrium-service
    compose_dir: orion-equilibrium-service
    compose_service: equilibrium-service
    include_bus_env: true
    auto_remediate: false
    probe:
      mode: redis
      intake_channels: []
      ready_url: null
      service_name: orion-equilibrium-service
```

**Port verification before commit:** cross-check every `ready_url` against the service's `docker-compose.yml` and `.env_example` (known values: gateway `8022`, orch `8072`, exec `8070`, landing-pad `8370`, recall `8260`, llm-gateway `8210`). Equilibrium is bus-only (no HTTP); guardian observes it via equilibrium snapshot layer only — set `probe.mode: redis` with empty channels so active probe is a no-op and equilibrium layer drives status.

- [ ] **Step 2: Commit**

```bash
git add config/mesh_remediation_roster.yaml
git commit -m "feat(mesh-guardian): seed v1 chat-path remediation roster"
```

---

### Task 10: Mesh guardian skeleton (Docker + `/health`)

**Files:**
- Create: `services/orion-mesh-guardian/Dockerfile`
- Create: `services/orion-mesh-guardian/docker-compose.yml`
- Create: `services/orion-mesh-guardian/requirements.txt`
- Create: `services/orion-mesh-guardian/.env_example`
- Create: `services/orion-mesh-guardian/app/settings.py`
- Create: `services/orion-mesh-guardian/app/main.py`
- Create: `services/orion-mesh-guardian/tests/test_health.py`

- [ ] **Step 1: Create minimal FastAPI app**

`services/orion-mesh-guardian/app/settings.py`:

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("orion-mesh-guardian", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    project: str = Field("orion", alias="PROJECT")
    node_name: str = Field("unknown", alias="NODE_NAME")
    orion_repo_root: str = Field("/repo", alias="ORION_REPO_ROOT")
    orion_bus_url: str = Field("redis://bus-core:6379/0", alias="ORION_BUS_URL")
    notify_base_url: str = Field("http://orion-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    roster_path: str = Field("/repo/config/mesh_remediation_roster.yaml", alias="MESH_GUARDIAN_ROSTER_PATH")
    enabled: bool = Field(True, alias="MESH_GUARDIAN_ENABLED")
    auto_remediate: bool = Field(False, alias="MESH_GUARDIAN_AUTO_REMEDIATE")
    remediation_cooldown_sec: int = Field(300, alias="MESH_GUARDIAN_REMEDIATION_COOLDOWN_SEC")
    max_attempts_per_hour: int = Field(3, alias="MESH_GUARDIAN_MAX_ATTEMPTS_PER_HOUR")
    probe_interval_sec: int = Field(15, alias="MESH_GUARDIAN_PROBE_INTERVAL_SEC")
    post_remediate_grace_sec: int = Field(60, alias="MESH_GUARDIAN_POST_REMEDIATE_GRACE_SEC")
    consecutive_probe_fails: int = Field(2, alias="MESH_GUARDIAN_CONSECUTIVE_PROBE_FAILS")
    equilibrium_grace_sec: int = Field(30, alias="MESH_GUARDIAN_EQUILIBRIUM_GRACE_SEC")
    channel_equilibrium_snapshot: str = Field("orion:equilibrium:snapshot", alias="CHANNEL_EQUILIBRIUM_SNAPSHOT")
```

`services/orion-mesh-guardian/app/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .settings import Settings

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Guardian loops wired in Phase B
    yield


app = FastAPI(title="orion-mesh-guardian", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "enabled": settings.enabled,
        "auto_remediate": settings.auto_remediate,
    }
```

- [ ] **Step 2: Dockerfile** (copy pattern from `services/orion-notify-digest/Dockerfile`; port 7160; COPY `orion/`)

- [ ] **Step 3: docker-compose.yml** (from design spec — `app-net`, docker.sock, `/repo:ro`, env vars)

- [ ] **Step 4: `.env_example`** — all `MESH_GUARDIAN_*` keys; `MESH_GUARDIAN_AUTO_REMEDIATE=false` for Phase B default

Run:
```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 5: Test + compile**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest services/orion-mesh-guardian/tests/test_health.py -v
python -m compileall services/orion-mesh-guardian
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(mesh-guardian): add service skeleton and compose wiring"
```

**Phase A checkpoint:** Core resilience, `/ready` on P0 services, roster file, guardian container builds.

---

# Phase B — Observe

---

### Task 11: Roster loader + validation

**Files:**
- Create: `services/orion-mesh-guardian/app/roster.py`
- Create: `services/orion-mesh-guardian/tests/test_roster_load.py`

- [ ] **Step 1: Write failing tests** for YAML load, `${PROJECT}` substitution, required fields, `auto_remediate` default

- [ ] **Step 2: Implement `roster.py`**

Use Pydantic models:

```python
from enum import Enum
from pydantic import BaseModel, Field


class ProbeMode(str, Enum):
    redis = "redis"
    http = "http"
    redis_and_http = "redis_and_http"


class ProbeConfig(BaseModel):
    mode: ProbeMode
    intake_channels: list[str] = Field(default_factory=list)
    ready_url: str | None = None
    service_name: str | None = None


class RosterEntry(BaseModel):
    id: str
    heartbeat_name: str
    compose_dir: str
    compose_service: str
    include_bus_env: bool = False
    auto_remediate: bool = True
    probe: ProbeConfig


class RosterDocument(BaseModel):
    services: list[RosterEntry]


def load_roster(path: str, *, project: str, node_name: str) -> RosterDocument:
    ...
```

Hard-coded exclusions in separate constant:

```python
NEVER_REMEDIATE_IDS = frozenset({"mesh-guardian", "hub", "orion-hub", "notify", "orion-notify"})
```

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 12: Active probe module

**Files:**
- Create: `services/orion-mesh-guardian/app/probe.py`
- Create: `services/orion-mesh-guardian/tests/test_probe_logic.py`

- [ ] **Step 1: Write failing tests**

Cases:
- Redis PING failure → `probe_bad`
- NUMSUB == 0 on intake channel → `probe_bad`
- HTTP `/ready` non-200 or JSON `ok: false` → `probe_bad`
- All pass → `probe_ok`

- [ ] **Step 2: Implement**

```python
from dataclasses import dataclass
from typing import Literal

import httpx

from orion.bus.consumer_readiness import redis_pubsub_numsub

from .roster import ProbeConfig, ProbeMode


@dataclass(frozen=True)
class ProbeResult:
    status: Literal["probe_ok", "probe_bad"]
    reason: str | None = None
    subscriber_counts: dict[str, int] | None = None
    http_status: int | None = None


async def run_probe(*, redis, entry_probe: ProbeConfig) -> ProbeResult:
    ...
```

Use `httpx.AsyncClient(timeout=5.0)` for HTTP probes.

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 13: Per-service state machine

**Files:**
- Create: `services/orion-mesh-guardian/app/state_machine.py`
- Create: `services/orion-mesh-guardian/tests/test_state_machine.py`

- [ ] **Step 1: Write failing tests for every transition in design spec**

Implement pure function:

```python
from dataclasses import dataclass, field
from enum import Enum
from time import time


class ServicePhase(str, Enum):
    healthy = "healthy"
    suspect = "suspect"
    unhealthy_confirmed = "unhealthy_confirmed"
    remediating_tier1 = "remediating_tier1"
    remediating_tier2 = "remediating_tier2"
    post_check_grace = "post_check_grace"
    attention_only = "attention_only"


@dataclass
class ServiceState:
    phase: ServicePhase = ServicePhase.healthy
    consecutive_probe_fails: int = 0
    last_remediate_ts: float | None = None
    attempts_this_hour: int = 0
    hour_window_start_ts: float = field(default_factory=time)
    post_grace_until_ts: float | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class TransitionInput:
    equilibrium_bad: bool
    probe_status: Literal["probe_ok", "probe_bad"]
    auto_remediate: bool
    now: float
    cooldown_sec: int
    max_attempts_per_hour: int
    consecutive_probe_fails_threshold: int
    post_grace_sec: int


@dataclass(frozen=True)
class TransitionOutput:
    new_state: ServiceState
    attention_events: list[dict]
    should_remediate_tier1: bool = False
    should_remediate_tier2: bool = False


def transition(state: ServiceState, inp: TransitionInput) -> TransitionOutput:
    ...
```

Test matrix (minimum):
1. `healthy` + equilibrium_bad → `suspect`
2. `healthy` + probe_bad → `suspect`
3. `suspect` + equilibrium_bad + probe_bad → `unhealthy_confirmed` + attention event
4. `suspect` + 2x probe_bad (equilibrium ok) → `unhealthy_confirmed`
5. `unhealthy_confirmed` + cooldown clear + auto_remediate → `remediating_tier1`
6. `unhealthy_confirmed` + auto_remediate false → stays, attention only (no docker)
7. `remediating_tier1` → immediate `post_check_grace`
8. `post_check_grace` + probe_ok + not equilibrium_bad → `healthy` + recovery attention
9. `post_check_grace` + still bad → `remediating_tier2`
10. tier2 post-grace still bad → `attention_only`
11. max attempts/hour → `attention_only`, no further docker

- [ ] **Step 2: Implement `transition()`**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 14: Equilibrium snapshot watcher

**Files:**
- Create: `services/orion-mesh-guardian/app/equilibrium_watch.py`
- Create: `services/orion-mesh-guardian/tests/test_equilibrium_watch.py`

- [ ] **Step 1: Write failing test** parsing `EquilibriumSnapshotV1` and mapping `heartbeat_name` → status

```python
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1, EquilibriumServiceState


def equilibrium_status_for_service(
    snapshot: EquilibriumSnapshotV1 | None,
    *,
    heartbeat_name: str,
    grace_sec: float,
    now_ts: float,
) -> tuple[bool, str | None]:
    """Return (equilibrium_bad, reason)."""
    ...
```

Treat as bad when:
- `status == "down"`
- `status == "degraded"` and `down_for_ms > grace_sec * 1000`
- service absent from snapshot but expected

- [ ] **Step 2: Implement subscriber loop** `async def watch_equilibrium(bus, channel, out_queue)` — decode envelope kind, validate payload to `EquilibriumSnapshotV1`, push to asyncio queue

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 15: Pending Attention wrapper

**Files:**
- Create: `services/orion-mesh-guardian/app/attention.py`
- Create: `services/orion-mesh-guardian/tests/test_attention.py`

- [ ] **Step 1: Write failing test** verifying `NotifyClient.attention_request` called with spec context fields

- [ ] **Step 2: Implement**

```python
from uuid import uuid4

from orion.notify.client import NotifyClient

from .settings import Settings


class AttentionPublisher:
    def __init__(self, settings: Settings) -> None:
        self._client = NotifyClient(
            base_url=settings.notify_base_url,
            api_token=settings.notify_api_token,
            timeout=10,
        )
        self._source = settings.service_name

    def publish_transition(self, *, service_id: str, heartbeat_name: str, event: dict) -> None:
        severity = event.get("severity", "error")
        message = event.get("message", f"mesh health: {service_id}")
        context = {
            "source_service": self._source,
            "event_kind": "orion.mesh.health.attention.v1",
            "service_id": service_id,
            "heartbeat_name": heartbeat_name,
            "correlation_id": event.get("correlation_id") or str(uuid4()),
            **event.get("context", {}),
        }
        self._client.attention_request(
            message=message,
            severity=severity,
            require_ack=True,
            context=context,
        )
```

Map transitions from state machine `attention_events` list.

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 16: Guardian orchestration loop (observe-only)

**Files:**
- Create: `services/orion-mesh-guardian/app/service.py`
- Modify: `services/orion-mesh-guardian/app/main.py`

- [ ] **Step 1: Implement `MeshGuardianService`**

Responsibilities:
- Connect `OrionBusAsync` for equilibrium subscribe + Redis probes
- Load roster on startup
- Maintain `dict[str, ServiceState]` in memory; optionally persist to Redis hash `mesh-guardian:state` (JSON per field)
- Every `probe_interval_sec`: run probes for all roster entries
- On equilibrium message: update equilibrium map
- Call `transition()`; call `AttentionPublisher` on attention events
- **Do not** call remediator yet (`auto_remediate` ignored for docker)

`service.py` skeleton:

```python
class MeshGuardianService:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def _probe_loop(self) -> None: ...
    async def _equilibrium_loop(self) -> None: ...
```

Wire in `main.py` lifespan.

- [ ] **Step 2: Integration test with mocked redis + httpx**

- [ ] **Step 3: Run tests**

```bash
./scripts/test_service.sh orion-mesh-guardian -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(mesh-guardian): equilibrium watch, probes, and attention (observe-only)"
```

**Phase B checkpoint:** Guardian fires Pending Attention cards; `MESH_GUARDIAN_AUTO_REMEDIATE=false` — no docker commands.

---

# Phase C — Act

---

### Task 17: Compose command builder

**Files:**
- Create: `services/orion-mesh-guardian/app/remediator.py`
- Create: `services/orion-mesh-guardian/tests/test_compose_command.py`

- [ ] **Step 1: Write failing tests**

```python
from app.remediator import build_compose_command
from app.roster import RosterEntry, ProbeConfig, ProbeMode


def test_tier1_force_recreate_command() -> None:
    entry = RosterEntry(
        id="landing-pad",
        heartbeat_name="landing-pad",
        compose_dir="orion-landing-pad",
        compose_service="orion-landing-pad",
        include_bus_env=False,
        auto_remediate=True,
        probe=ProbeConfig(mode=ProbeMode.redis_and_http, intake_channels=["orion:pad:rpc:request"]),
    )
    cmd = build_compose_command(entry, repo_root="/repo", tier=1)
    assert cmd[:4] == ["docker", "compose", "--env-file", "/repo/.env"]
    assert "services/orion-landing-pad/.env" in cmd
    assert cmd[-3:] == ["up", "-d", "--force-recreate", "orion-landing-pad"][-3:]  # adjust exact tail assert


def test_tier2_build_then_up() -> None:
    ...
```

- [ ] **Step 2: Implement**

```python
from pathlib import Path

from .roster import RosterEntry


def build_compose_command(entry: RosterEntry, *, repo_root: str, tier: int) -> list[str]:
    root = Path(repo_root)
    cmd: list[str] = ["docker", "compose", "--env-file", str(root / ".env")]
    if entry.include_bus_env:
        cmd.extend(["--env-file", str(root / "services/orion-bus/.env")])
    cmd.extend(["--env-file", str(root / "services" / entry.compose_dir / ".env")])
    cmd.extend(["-f", str(root / "services" / entry.compose_dir / "docker-compose.yml")])
    if tier == 1:
        cmd.extend(["up", "-d", "--force-recreate", entry.compose_service])
    elif tier == 2:
        raise ValueError("tier 2 uses build_compose_build_command + build_compose_up_command")
    else:
        raise ValueError(f"unsupported tier {tier}")
    return cmd


def build_compose_build_command(entry: RosterEntry, *, repo_root: str) -> list[str]:
    ...


def build_compose_up_command(entry: RosterEntry, *, repo_root: str) -> list[str]:
    ...
```

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 18: Remediation executor

**Files:**
- Modify: `services/orion-mesh-guardian/app/remediator.py`
- Create: `services/orion-mesh-guardian/tests/test_remediator_exec.py`

- [ ] **Step 1: Write failing test** with `asyncio.create_subprocess_exec` mocked — logs command, captures exit code, returns structured result

```python
@dataclass(frozen=True)
class RemediationResult:
    ok: bool
    tier: int
    command: list[str]
    exit_code: int
    stderr_tail: str
```

- [ ] **Step 2: Implement `async def execute_remediation(entry, repo_root, tier) -> RemediationResult`**

- cwd = `ORION_REPO_ROOT`
- tier 2: run build, then up (sequential; fail fast)
- Never execute for `NEVER_REMEDIATE_IDS` or `auto_remediate=False`

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 19: Wire remediation into state machine + Redis persistence

**Files:**
- Modify: `services/orion-mesh-guardian/app/service.py`
- Modify: `services/orion-mesh-guardian/app/state_machine.py` (expose `should_remediate_tier1/tier2` flags — already in Task 13)
- Create: `services/orion-mesh-guardian/app/state_store.py`

- [ ] **Step 1: Redis hash persistence**

```python
STATE_HASH_KEY = "mesh-guardian:state"

async def load_all(redis) -> dict[str, ServiceState]: ...
async def save_one(redis, service_id: str, state: ServiceState) -> None: ...
```

- [ ] **Step 2: In probe loop**, when transition returns `should_remediate_tier1` or `should_remediate_tier2`:
  - Check `settings.auto_remediate` and `NEVER_REMEDIATE_IDS`
  - Call `execute_remediation`
  - On failure → attention card with stderr tail
  - On success → set phase `post_check_grace` with `post_grace_until_ts = now + post_remediate_grace_sec`

- [ ] **Step 3: Tests** for cooldown (no second recreate within 300s) and max attempts (3/hour → attention_only)

- [ ] **Step 4: Set default `MESH_GUARDIAN_AUTO_REMEDIATE=true` in `.env_example`** only after tests pass; sync local env

- [ ] **Step 5: Commit**

---

### Task 20: Guardian `/ready` + compile sweep

**Files:**
- Modify: `services/orion-mesh-guardian/app/main.py`

- [ ] **Step 1: Add `GET /ready`** — best-effort bus ping + equilibrium subscriber alive flag

- [ ] **Step 2: Run full unit suite**

```bash
./scripts/test_service.sh orion-mesh-guardian -q
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/core/bus/tests/test_resilience.py tests/test_chassis_heartbeat_reconnect.py -q
```

- [ ] **Step 3: Commit**

---

### Task 21: Smoke script (optional v1)

**Files:**
- Create: `scripts/smoke_mesh_guardian.sh`

- [ ] **Step 1: Script checks**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

curl -sf "http://${PROJECT:-orion}-mesh-guardian:7160/health" | jq -e '.ok == true'
curl -sf "http://${PROJECT:-orion}-mesh-guardian:7160/ready" || true  # may 503 before bus connect

PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-mesh-guardian/tests/test_state_machine.py \
  services/orion-mesh-guardian/tests/test_compose_command.py \
  services/orion-mesh-guardian/tests/test_roster_load.py -q

echo "smoke_mesh_guardian: unit checks passed"
```

- [ ] **Step 2: chmod +x and run**

```bash
chmod +x scripts/smoke_mesh_guardian.sh
./scripts/smoke_mesh_guardian.sh
```

- [ ] **Step 3: Commit**

---

### Task 22: Live-stack acceptance (manual — required before enabling auto-remediate in prod)

Run on host with chat stack up and guardian container deployed.

| # | Check | Command / action | Expected |
|---|-------|------------------|----------|
| 1 | Half-death detect | Stop landing pad bus subscriber or block Redis briefly | Pending Attention error card within ~30–45s |
| 2 | Tier-1 recover | Let guardian force-recreate landing pad | Post-grace probe passes; info recovery card |
| 3 | Tier-2 escalate | Mock persistent failure after tier-1 | build+up runs after grace |
| 4 | Cooldown | Repeated failures | ≤1 recreate per 5 min |
| 5 | Max attempts | 3 failures in 1 hour | attention_only; no docker |
| 6 | Kill switch | `MESH_GUARDIAN_AUTO_REMEDIATE=false` | Cards only, no docker |
| 7 | Notify exclusion | Stop notify | Card appears; notify container not recreated |
| 8 | Unit tests | `./scripts/smoke_mesh_guardian.sh` | exit 0 |

Document evidence (log lines + Hub `/api/attention?status=pending` JSON snippet) in PR test plan.

---

## Self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| Core `publish_with_reconnect` | Task 1, 3 |
| Chassis heartbeat reconnect | Task 2 |
| `/ready` on P0 chat-path services | Tasks 4–8 |
| Roster YAML v1 seed | Task 9 |
| Guardian on app-net + docker.sock | Task 10 |
| Equilibrium layer | Task 14 |
| Active probe (PING, NUMSUB, HTTP) | Task 12 |
| State machine transitions | Task 13 |
| Pending Attention via notify | Task 15 |
| Tier-1 force-recreate | Tasks 17–19 |
| Tier-2 build+up | Tasks 17–19 |
| Cooldown + max attempts | Task 13, 19 |
| Never remediate notify/hub/self | Task 11, 19 |
| Redis state hash | Task 19 |
| Acceptance checks | Task 22 |
| Hub no UI changes | N/A (notify path only) |

**Placeholder scan:** No TBD steps. All tasks include concrete paths and code entry points.

**Type consistency:** `ServicePhase`, `ProbeResult.status`, `RosterEntry`, and `NotifyClient.attention_request` context keys match design spec JSON example.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-mesh-bus-resilience-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints after each phase.

**Which approach?**
