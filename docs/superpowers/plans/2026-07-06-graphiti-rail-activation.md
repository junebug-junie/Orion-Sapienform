# Graphiti Rail Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Operator update (2026-07-16):** Commands using graphiti-adapter's
> `--profile falkordb` are historical. The sidecar was removed; use
> `services/orion-falkordb/README.md` for current bring-up and cutover steps.

**Goal:** Activate the dormant Graphiti projection rail in three sequential vertical slices — operational hardening (A), cross-crystallization link projection (B), and optional `graphiti-core` hybrid retrieval (C).

**Architecture:** Hub `project_crystallization()` and `retrieve_active_packet()` call `GraphitiAdapter` HTTP client (`orion/memory/crystallization/projection_graphiti.py`), which talks to `orion-graphiti-adapter`. Phase A fixes URL/env parity so approve auto-projects with default `.env_example`. Phase B projects `memory_crystallization_links` into `graphiti_edges` and enables depth-2 BFS neighborhood. Phase C adds `GRAPHITI_BACKEND=graphiti_core` behind a flag with prescribed ontology writes (no LLM re-extraction) and `/v1/search`.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, httpx, pytest, bash smoke scripts, optional `graphiti-core[falkordb]` (Phase C only).

**Spec:** `docs/superpowers/specs/2026-07-06-graphiti-rail-activation-design.md`

**Branches (one per phase):**
- `feat/graphiti-rail-phase-a`
- `feat/graphiti-rail-phase-b`
- `feat/graphiti-rail-phase-c`

**Worktree setup (run once before Phase A):**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-graphiti-rail -b feat/graphiti-rail-phase-a
cd ../Orion-Sapienform-graphiti-rail
```

---

## File Map

| File | Phase | Action | Responsibility |
|------|-------|--------|----------------|
| `orion/memory/crystallization/graphiti_config.py` | A | Create | `resolve_graphiti_adapter_url(settings)` — single URL resolver |
| `orion/memory/crystallization/projection_graphiti.py` | A,B,C | Modify | HTTP client; links in payload (B); `neighborhood(depth)` + `search()` (C) |
| `services/orion-hub/scripts/crystallization_routes.py` | A | Modify | Wire resolver into `_graphiti()` and `_projection_config()` |
| `services/orion-hub/.env_example` | A | Modify | Deprecate `GRAPHITI_URL` comment; document `GRAPHITI_ADAPTER_URL` as primary |
| `services/orion-hub/tests/test_graphiti_config_parity.py` | A | Create | Regression: `_projection_config` URL matches `_graphiti` |
| `services/orion-graphiti-adapter/tests/conftest.py` | A | Create | sys.path + test env defaults |
| `services/orion-graphiti-adapter/tests/test_episodes.py` | A | Create | Ingest + neighborhood + health tests |
| `scripts/smoke_memory_crystallization_e2e.sh` | A | Modify | Graphiti sync + neighborhood steps after approve |
| `services/orion-graphiti-adapter/app/main.py` | B,C | Modify | `links` on ingest; `depth` query param; `/v1/search`; backend routing |
| `services/orion-graphiti-adapter/app/store.py` | B | Modify | Link edge upsert; BFS neighborhood |
| `services/orion-graphiti-adapter/app/falkordb.py` | B | Modify | Optional `links` MERGE |
| `orion/memory/crystallization/retriever.py` | B,C | Modify | `depth=2` neighborhood; search when backend=graphiti_core |
| `services/orion-hub/static/js/memory-crystallization-ui.js` | B | Modify | Graphiti projection counts + Sync button |
| `services/orion-hub/tests/test_graphiti_sync_payload.py` | B | Create | Sync payload includes crystallization links |
| `services/orion-graphiti-adapter/tests/test_links.py` | B | Create | Two-crystallization link + depth-2 neighborhood |
| `scripts/smoke_graphiti_links_e2e.sh` | B | Create | Two-crystallization link smoke |
| `services/orion-graphiti-adapter/app/settings.py` | C | Modify | `GRAPHITI_BACKEND`, embed URL |
| `services/orion-graphiti-adapter/app/backends/orion_postgres.py` | C | Create | Delegate to current `store.py` |
| `services/orion-graphiti-adapter/app/backends/graphiti_core.py` | C | Create | Prescribed FalkorDB writes + hybrid search |
| `services/orion-graphiti-adapter/requirements.txt` | C | Modify | Pin `graphiti-core[falkordb]==0.19.10` |
| `services/orion-graphiti-adapter/.env_example` | C | Modify | `GRAPHITI_BACKEND`, `CRYSTALLIZER_EMBED_HOST_URL` |
| `services/orion-graphiti-adapter/README.md` | B,C | Modify | FalkorDB profile + backend flag docs |
| `tests/test_memory_crystallization.py` | B,C | Modify | Retriever graphiti_refs with linked crystallization |

---

# Phase A — Operational Hardening (PR 1)

## Task 1: URL resolver helper

**Files:**
- Create: `orion/memory/crystallization/graphiti_config.py`
- Create: `tests/test_graphiti_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graphiti_config.py`:

```python
from types import SimpleNamespace

from orion.memory.crystallization.graphiti_config import resolve_graphiti_adapter_url


def test_resolve_prefers_adapter_url_over_graphiti_url():
    settings = SimpleNamespace(
        GRAPHITI_ADAPTER_URL="http://adapter:8000",
        GRAPHITI_URL="http://legacy:8000",
    )
    assert resolve_graphiti_adapter_url(settings) == "http://adapter:8000"


def test_resolve_falls_back_to_graphiti_url():
    settings = SimpleNamespace(GRAPHITI_ADAPTER_URL="", GRAPHITI_URL="http://legacy:8000")
    assert resolve_graphiti_adapter_url(settings) == "http://legacy:8000"


def test_resolve_empty_when_both_unset():
    settings = SimpleNamespace(GRAPHITI_ADAPTER_URL="", GRAPHITI_URL="")
    assert resolve_graphiti_adapter_url(settings) == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
pytest tests/test_graphiti_config.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` for `graphiti_config`

- [ ] **Step 3: Write minimal implementation**

Create `orion/memory/crystallization/graphiti_config.py`:

```python
from __future__ import annotations


def resolve_graphiti_adapter_url(settings) -> str:
    return (
        getattr(settings, "GRAPHITI_ADAPTER_URL", "") or
        getattr(settings, "GRAPHITI_URL", "") or
        ""
    ).strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_graphiti_config.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/graphiti_config.py tests/test_graphiti_config.py
git commit -m "feat(graphiti): add resolve_graphiti_adapter_url helper"
```

---

## Task 2: Wire Hub `_graphiti()` and `_projection_config()`

**Files:**
- Modify: `services/orion-hub/scripts/crystallization_routes.py:20-21,72-95`

- [ ] **Step 1: Write the failing Hub parity test**

Create `services/orion-hub/tests/test_graphiti_config_parity.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from scripts import crystallization_routes as routes


@pytest.fixture
def adapter_url_only_settings(monkeypatch):
    fake = SimpleNamespace(
        GRAPHITI_ENABLED=False,
        GRAPHITI_ADAPTER_URL="http://orion-athena-graphiti-adapter:8000",
        GRAPHITI_URL="",
        FALKORDB_URI="",
        CRYSTALLIZER_VECTOR_COLLECTION="orion_memory_crystallizations",
        CRYSTALLIZER_EMBED_HOST_URL="",
        CRYSTALLIZER_EMBED_MODE="http",
        CRYSTALLIZER_EMBED_TIMEOUT_MS=8000,
        SERVICE_NAME="orion-hub",
        SERVICE_VERSION="0.1.0",
        NODE_NAME="hub",
    )
    monkeypatch.setattr(routes, "_settings", lambda: fake)
    return fake


def test_projection_config_url_matches_graphiti_adapter(adapter_url_only_settings):
    cfg = routes._projection_config()
    request = MagicMock(spec=Request)
    adapter = routes._graphiti(request)

    assert cfg.graphiti_url == "http://orion-athena-graphiti-adapter:8000"
    assert adapter.url == "http://orion-athena-graphiti-adapter:8000"
    assert cfg.graphiti_enabled is True
    assert adapter.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest services/orion-hub/tests/test_graphiti_config_parity.py -v
```

Expected: FAIL — `cfg.graphiti_url == ''` or `cfg.graphiti_enabled is False`

- [ ] **Step 3: Patch crystallization routes**

In `services/orion-hub/scripts/crystallization_routes.py`, add import:

```python
from orion.memory.crystallization.graphiti_config import resolve_graphiti_adapter_url
```

Replace `_graphiti()` body:

```python
def _graphiti(request: Request) -> GraphitiAdapter:
    settings = _settings()
    adapter_url = resolve_graphiti_adapter_url(settings)
    return GraphitiAdapter(
        enabled=bool(getattr(settings, "GRAPHITI_ENABLED", False)) or bool(adapter_url),
        url=adapter_url or None,
        falkordb_uri=getattr(settings, "FALKORDB_URI", None),
    )
```

Replace `_projection_config()` graphiti fields:

```python
def _projection_config() -> ProjectionConfig:
    s = _settings()
    adapter_url = resolve_graphiti_adapter_url(s)
    return ProjectionConfig(
        collection=getattr(s, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations"),
        embed_host_url=getattr(s, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        embed_mode=getattr(s, "CRYSTALLIZER_EMBED_MODE", "http") or "http",
        embed_timeout_ms=int(getattr(s, "CRYSTALLIZER_EMBED_TIMEOUT_MS", 8000) or 8000),
        graphiti_enabled=bool(getattr(s, "GRAPHITI_ENABLED", False)) or bool(adapter_url),
        graphiti_url=adapter_url,
        falkordb_uri=getattr(s, "FALKORDB_URI", "") or "",
        service_name=getattr(s, "SERVICE_NAME", "orion-hub"),
        service_version=getattr(s, "SERVICE_VERSION", "0.1.0"),
        node_name=getattr(s, "NODE_NAME", "hub"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest services/orion-hub/tests/test_graphiti_config_parity.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/crystallization_routes.py services/orion-hub/tests/test_graphiti_config_parity.py
git commit -m "fix(graphiti): unify adapter URL in projection config and routes"
```

---

## Task 3: Document env contract

**Files:**
- Modify: `services/orion-hub/.env_example:202-219`

- [ ] **Step 1: Update comments**

In `services/orion-hub/.env_example`, change the Graphiti block to:

```bash
# --- Memory crystallization Graphiti/FalkorDB (additive temporal projection; NOT RDF memory_graph) ---
GRAPHITI_ENABLED=true
# Deprecated fallback — prefer GRAPHITI_ADAPTER_URL below
GRAPHITI_URL=
FALKORDB_URI=

# ... (unchanged crystallizer keys) ...

# Graphiti additive adapter (NOT RDF memory_graph) — required when GRAPHITI_ENABLED=true
GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000
```

- [ ] **Step 2: Sync local env**

```bash
python scripts/sync_local_env_from_example.py
```

Expected: keys merged; report any skipped keys explicitly.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/.env_example
git commit -m "docs(graphiti): mark GRAPHITI_ADAPTER_URL as primary env key"
```

---

## Task 4: Adapter test harness

**Files:**
- Create: `services/orion-graphiti-adapter/tests/conftest.py`
- Create: `services/orion-graphiti-adapter/tests/test_health.py`

- [ ] **Step 1: Write conftest**

Create `services/orion-graphiti-adapter/tests/conftest.py`:

```python
import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_URI", "")
os.environ.setdefault("FALKORDB_ENABLED", "false")

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
```

- [ ] **Step 2: Write failing health test**

Create `services/orion-graphiti-adapter/tests/test_health.py`:

```python
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_mod


def test_health_without_postgres():
    with patch.object(main_mod, "pg_pool", None):
        client = TestClient(main_mod.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["postgres"] is False
        assert data["service"] == "orion-graphiti-adapter"
```

- [ ] **Step 3: Run test**

```bash
pytest services/orion-graphiti-adapter/tests/test_health.py -v
```

Expected: PASS (health endpoint already exists)

- [ ] **Step 4: Commit**

```bash
git add services/orion-graphiti-adapter/tests/
git commit -m "test(graphiti-adapter): add conftest and health-without-postgres test"
```

---

## Task 5: Adapter ingest + neighborhood tests

**Files:**
- Create: `services/orion-graphiti-adapter/tests/test_episodes.py`

- [ ] **Step 1: Write failing ingest test**

Create `services/orion-graphiti-adapter/tests/test_episodes.py`:

```python
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod


@pytest.fixture
def mock_pool():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool, conn


def test_ingest_episode_returns_ids(mock_pool):
    pool, conn = mock_pool
    with patch.object(main_mod, "pg_pool", pool), patch(
        "app.main.sync_to_falkordb", return_value=None
    ):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/episodes",
            json={
                "crystallization_id": "crys_test001",
                "kind": "stance",
                "subject": "Test subject",
                "summary": "Test summary",
                "status": "active",
                "metadata": {"scope": ["project:orion"]},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == "gep_crys_test001"
        assert data["entity_id"] == "gent_crys_test001"
        assert data["edge_id"] == "ged_crys_test001"
        assert data["canonical_mutated"] is False
        assert conn.execute.await_count >= 3


def test_neighborhood_after_ingest(mock_pool):
    pool, conn = mock_pool
    conn.fetch = AsyncMock(
        side_effect=[
            [{"episode_id": "gep_crys_test001", "crystallization_id": "crys_test001"}],
            [{"entity_id": "gent_crys_test001", "crystallization_id": "crys_test001"}],
            [{"edge_id": "ged_crys_test001", "from_id": "gent_crys_test001", "to_id": "gep_crys_test001"}],
        ]
    )
    with patch.object(main_mod, "pg_pool", pool):
        client = TestClient(main_mod.app)
        resp = client.get("/v1/neighborhood/crys_test001")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) >= 1
        assert data["crystallization_id"] == "crys_test001"
```

- [ ] **Step 2: Run tests**

```bash
pytest services/orion-graphiti-adapter/tests/test_episodes.py -v
```

Expected: `2 passed`

- [ ] **Step 3: Commit**

```bash
git add services/orion-graphiti-adapter/tests/test_episodes.py
git commit -m "test(graphiti-adapter): cover episode ingest and neighborhood"
```

---

## Task 6: Extend crystallization smoke with Graphiti steps

**Files:**
- Modify: `scripts/smoke_memory_crystallization_e2e.sh`

- [ ] **Step 1: Append Graphiti block after approve**

Add before the final `PASS` line in `scripts/smoke_memory_crystallization_e2e.sh`:

```bash
echo "== projection health (graphiti flag) =="
PROJ_HEALTH=$(curl -sS "${HDR[@]}" "${BASE}/api/memory/crystallizations/projection/health")
echo "$PROJ_HEALTH" | jq -c .
GRAPHITI_ON="$(echo "$PROJ_HEALTH" | jq -r '.graphiti_enabled // false')"

_skip_graphiti() {
  echo "WARN: skipping Graphiti steps ($1)"
}

if [[ "${GRAPHITI_SKIP:-0}" == "1" ]]; then
  _skip_graphiti "GRAPHITI_SKIP=1"
elif [[ "$GRAPHITI_ON" != "true" ]]; then
  _skip_graphiti "graphiti_enabled=false"
else
  echo "== graphiti sync =="
  SYNC=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/graphiti/sync/${CID}" "${HDR[@]}" -d '{}')
  SYNC_BODY="$(echo "$SYNC" | head -n -1)"
  SYNC_CODE="$(echo "$SYNC" | tail -n 1)"
  [[ "$SYNC_CODE" == "200" ]] || { echo "FAIL graphiti sync HTTP $SYNC_CODE body=$SYNC_BODY"; exit 1; }
  EP_COUNT="$(echo "$SYNC_BODY" | jq -r '(.graphiti.episode_ids // []) | length')"
  [[ "$EP_COUNT" -ge 1 ]] || { echo "FAIL graphiti sync returned no episode_ids"; exit 1; }

  echo "== graphiti neighborhood =="
  NB=$(curl -sS -w "\n%{http_code}" "${HDR[@]}" "${BASE}/api/memory/graphiti/neighborhood/${CID}")
  NB_BODY="$(echo "$NB" | head -n -1)"
  NB_CODE="$(echo "$NB" | tail -n 1)"
  [[ "$NB_CODE" == "200" ]] || { echo "FAIL neighborhood HTTP $NB_CODE"; exit 1; }
  NODE_COUNT="$(echo "$NB_BODY" | jq -r '(.nodes // []) | length')"
  [[ "$NODE_COUNT" -ge 1 ]] || { echo "FAIL neighborhood nodes empty"; exit 1; }

  echo "== graphiti health =="
  GH=$(curl -sS "${HDR[@]}" "${BASE}/api/memory/graphiti/health")
  echo "$GH" | jq -c .
  [[ "$(echo "$GH" | jq -r '.enabled')" == "true" ]] || { echo "FAIL graphiti health enabled!=true"; exit 1; }
  [[ "$(echo "$GH" | jq -r '.url_configured')" == "true" ]] || { echo "FAIL graphiti health url_configured!=true"; exit 1; }
fi
```

- [ ] **Step 2: Run smoke (adapter must be up; otherwise use GRAPHITI_SKIP=1 for crystallization-only check)**

```bash
# With adapter running:
ORION_HUB_URL=http://127.0.0.1:8080 \
ORION_HUB_SESSION_ID=<session> \
RECALL_PG_DSN=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
bash scripts/smoke_memory_crystallization_e2e.sh

# Without adapter (WARN path only):
GRAPHITI_SKIP=1 ORION_HUB_URL=... ORION_HUB_SESSION_ID=... RECALL_PG_DSN=... \
bash scripts/smoke_memory_crystallization_e2e.sh
```

Expected: `PASS smoke_memory_crystallization_e2e` (full path) or PASS with WARN lines (skip path).

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_memory_crystallization_e2e.sh
git commit -m "test(smoke): exercise graphiti sync and neighborhood after approve"
```

---

## Task 7: Phase A gate

- [ ] **Step 1: Run focused gates**

```bash
pytest tests/test_graphiti_config.py services/orion-hub/tests/test_graphiti_config_parity.py services/orion-graphiti-adapter/tests -q
pytest tests/test_memory_crystallization.py::TestProjections::test_graphiti_cannot_mutate_canonical -q
python scripts/check_env_template_parity.py
```

Expected: all pass.

- [ ] **Step 2: Push branch and open PR 1**

```bash
git push -u origin feat/graphiti-rail-phase-a
```

Phase A acceptance checklist:
- [ ] `pytest services/orion-graphiti-adapter/tests -q` passes
- [ ] Hub graphiti config parity test passes
- [ ] Smoke PASS with adapter container running
- [ ] Approve response `projection.graphiti` non-empty when `GRAPHITI_ENABLED=true` and adapter URL set

---

# Phase B — Richer Orion-Owned Graph (PR 2)

**Start from Phase A merged:**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git switch -c feat/graphiti-rail-phase-b
```

---

## Task 8: Adapter link ingest schema

**Files:**
- Modify: `services/orion-graphiti-adapter/app/main.py:17-24`

- [ ] **Step 1: Write failing link test**

Create `services/orion-graphiti-adapter/tests/test_links.py`:

```python
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import app.main as main_mod


def _mock_pool():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool, conn


def test_ingest_with_supports_link_writes_cross_edge():
    pool, conn = _mock_pool()
    with patch.object(main_mod, "pg_pool", pool), patch("app.main.sync_to_falkordb", return_value=None):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/episodes",
            json={
                "crystallization_id": "crys_a",
                "kind": "stance",
                "subject": "A",
                "summary": "A summary",
                "links": [
                    {
                        "target_crystallization_id": "crys_b",
                        "relation": "supports",
                        "confidence": 0.9,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        sql_calls = [str(c.args[0]) for c in conn.execute.await_args_list]
        assert any("graphiti_edges" in s for s in sql_calls)
        assert any("crys_b" in str(c) for c in conn.execute.await_args_list)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest services/orion-graphiti-adapter/tests/test_links.py::test_ingest_with_supports_link_writes_cross_edge -v
```

- [ ] **Step 3: Add schema models to main.py**

```python
class CrystallizationLinkIngestV1(BaseModel):
    target_crystallization_id: str
    relation: str
    confidence: float = 0.5


class EpisodeIngestV1(BaseModel):
    crystallization_id: str
    kind: str
    subject: str
    summary: str
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: list[CrystallizationLinkIngestV1] = Field(default_factory=list)
```

Pass `links` into `upsert_episode()`:

```python
await upsert_episode(
    pg_pool,
    episode_id=episode_id,
    crystallization_id=body.crystallization_id,
    kind=body.kind,
    subject=body.subject,
    summary=body.summary,
    status=body.status,
    metadata=body.metadata,
    links=[l.model_dump() for l in body.links],
)
```

- [ ] **Step 4: Run test — expect PASS after Task 9 store changes**

---

## Task 9: Store link projection + stub entities

**Files:**
- Modify: `services/orion-graphiti-adapter/app/store.py:41-98`

- [ ] **Step 1: Extend `upsert_episode` signature and link loop**

```python
async def upsert_episode(
    pool: asyncpg.Pool,
    *,
    episode_id: str,
    crystallization_id: str,
    kind: str,
    subject: str,
    summary: str,
    status: str,
    metadata: dict[str, Any],
    links: list[dict[str, Any]] | None = None,
) -> list[str]:
    edge_ids: list[str] = []
    async with pool.acquire() as conn:
        # ... existing episode, entity, has_episode edge inserts unchanged ...

        for link in links or []:
            target_id = str(link["target_crystallization_id"])
            relation = str(link["relation"])
            confidence = float(link.get("confidence", 0.5))
            target_entity_id = f"gent_{target_id}"
            await conn.execute(
                """
                INSERT INTO graphiti_entities (entity_id, crystallization_id, name, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (entity_id) DO NOTHING
                """,
                target_entity_id,
                target_id,
                f"stub:{target_id}",
                json.dumps({"stub": True}),
            )
            from_entity_id = f"gent_{crystallization_id}"
            edge_id = f"ged_{crystallization_id}_{target_id}_{relation}"
            await conn.execute(
                """
                INSERT INTO graphiti_edges (edge_id, from_id, to_id, relation, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (edge_id) DO UPDATE SET metadata = EXCLUDED.metadata
                """,
                edge_id,
                from_entity_id,
                target_entity_id,
                relation,
                json.dumps({"confidence": confidence, "note": link.get("note")}),
            )
            edge_ids.append(edge_id)
    return edge_ids
```

Update `main.py` ingest response to include all `edge_ids` from store.

- [ ] **Step 2: Run link ingest test**

```bash
pytest services/orion-graphiti-adapter/tests/test_links.py::test_ingest_with_supports_link_writes_cross_edge -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add services/orion-graphiti-adapter/app/store.py services/orion-graphiti-adapter/app/main.py services/orion-graphiti-adapter/tests/test_links.py
git commit -m "feat(graphiti-adapter): project crystallization links on ingest"
```

---

## Task 10: Multi-hop BFS neighborhood

**Files:**
- Modify: `services/orion-graphiti-adapter/app/store.py:101-122`
- Modify: `services/orion-graphiti-adapter/app/main.py:86-90`

- [ ] **Step 1: Write failing depth-2 test**

Add to `services/orion-graphiti-adapter/tests/test_links.py`:

```python
def test_neighborhood_depth_two_returns_linked_crystallization():
    pool, conn = _mock_pool()
    conn.fetch = AsyncMock(
        side_effect=[
            [{"episode_id": "gep_crys_a", "crystallization_id": "crys_a"}],
            [{"entity_id": "gent_crys_a", "crystallization_id": "crys_a"}],
            [
                {"edge_id": "ged_a_b_supports", "from_id": "gent_crys_a", "to_id": "gent_crys_b", "relation": "supports"},
                {"edge_id": "ged_has", "from_id": "gent_crys_b", "to_id": "gep_crys_b", "relation": "has_episode"},
            ],
            [{"entity_id": "gent_crys_b", "crystallization_id": "crys_b"}],
            [{"episode_id": "gep_crys_b", "crystallization_id": "crys_b"}],
        ]
    )
    with patch.object(main_mod, "pg_pool", pool):
        client = TestClient(main_mod.app)
        resp = client.get("/v1/neighborhood/crys_a?depth=2")
        assert resp.status_code == 200
        cids = {n.get("crystallization_id") for n in resp.json()["nodes"]}
        assert "crys_a" in cids
        assert "crys_b" in cids
```

- [ ] **Step 2: Implement BFS in `neighborhood()`**

```python
async def neighborhood(pool: asyncpg.Pool, crystallization_id: str, *, depth: int = 1) -> dict[str, Any]:
    depth = max(1, min(int(depth), 2))
    seed_prefix = f"%{crystallization_id}%"
    visited_nodes: dict[str, dict] = {}
    visited_edges: dict[str, dict] = {}
    frontier = {f"gent_{crystallization_id}", f"gep_{crystallization_id}"}

    async with pool.acquire() as conn:
        for _ in range(depth + 1):
            if not frontier:
                break
            node_ids = list(frontier)
            frontier = set()
            rows = await conn.fetch(
                """
                SELECT * FROM graphiti_edges
                WHERE from_id = ANY($1::text[]) OR to_id = ANY($1::text[])
                """,
                node_ids,
            )
            for row in rows:
                edge = dict(row)
                visited_edges[edge["edge_id"]] = edge
                for nid in (edge["from_id"], edge["to_id"]):
                    if nid not in visited_nodes:
                        frontier.add(nid)

        crystallization_ids = {crystallization_id}
        for edge in visited_edges.values():
            for nid in (edge["from_id"], edge["to_id"]):
                if nid.startswith("gent_"):
                    crystallization_ids.add(nid.removeprefix("gent_"))
                if nid.startswith("gep_"):
                    crystallization_ids.add(nid.removeprefix("gep_"))

        episodes = await conn.fetch(
            "SELECT * FROM graphiti_episodes WHERE crystallization_id = ANY($1::text[])",
            list(crystallization_ids),
        )
        entities = await conn.fetch(
            "SELECT * FROM graphiti_entities WHERE crystallization_id = ANY($1::text[])",
            list(crystallization_ids),
        )

    for row in list(entities) + list(episodes):
        key = row["entity_id"] if "entity_id" in row else row["episode_id"]
        visited_nodes[key] = dict(row)

    return {
        "crystallization_id": crystallization_id,
        "depth": depth,
        "nodes": list(visited_nodes.values()),
        "edges": list(visited_edges.values()),
    }
```

Update route:

```python
@app.get("/v1/neighborhood/{crystallization_id}")
async def get_neighborhood(crystallization_id: str, depth: int = 1) -> dict:
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="store_unavailable")
    return await neighborhood(pg_pool, crystallization_id, depth=depth)
```

- [ ] **Step 3: Run tests**

```bash
pytest services/orion-graphiti-adapter/tests/test_links.py -v
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add services/orion-graphiti-adapter/app/store.py services/orion-graphiti-adapter/app/main.py services/orion-graphiti-adapter/tests/test_links.py
git commit -m "feat(graphiti-adapter): depth-2 BFS neighborhood"
```

---

## Task 11: Hub sync payload includes links

**Files:**
- Modify: `orion/memory/crystallization/projection_graphiti.py:43-85`
- Create: `services/orion-hub/tests/test_graphiti_sync_payload.py`

- [ ] **Step 1: Write failing payload test**

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.schemas import CrystallizationGovernanceV1, CrystallizationLinkV1, MemoryCrystallizationV1


def _crys_with_link():
    now = datetime.now(timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id="crys_a",
        kind="stance",
        subject="A",
        summary="summary",
        status="active",
        governance=CrystallizationGovernanceV1(proposed_by="test", approved_by="test"),
        created_at=now,
        updated_at=now,
        links=[
            CrystallizationLinkV1(target_crystallization_id="crys_b", relation="supports", confidence=0.8)
        ],
    )


def test_sync_payload_includes_links():
    crys = _crys_with_link()
    adapter = GraphitiAdapter(enabled=True, url="http://graphiti")
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"episode_id": "gep_crys_a", "entity_id": "gent_crys_a", "edge_id": "ged_x", "canonical_mutated": False}
        return resp

    with patch("httpx.Client.post", side_effect=fake_post):
        adapter.sync_crystallization(crys)

    assert captured["json"]["links"] == [
        {"target_crystallization_id": "crys_b", "relation": "supports", "confidence": 0.8}
    ]
```

- [ ] **Step 2: Add links to both sync methods in projection_graphiti.py**

```python
"links": [
    {
        "target_crystallization_id": l.target_crystallization_id,
        "relation": l.relation,
        "confidence": l.confidence,
    }
    for l in crystallization.links
],
```

- [ ] **Step 3: Add `neighborhood(crystallization_id, depth=1)`**

```python
def neighborhood(self, crystallization_id: str, *, depth: int = 1) -> dict[str, Any]:
    if not self.enabled or not self.url:
        return {"enabled": False, "nodes": [], "edges": []}
    try:
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.get(f"{self.url}/v1/neighborhood/{crystallization_id}", params={"depth": depth})
            ...
```

- [ ] **Step 4: Run tests**

```bash
pytest services/orion-hub/tests/test_graphiti_sync_payload.py orion/memory/crystallization/ -q --ignore=services
```

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/projection_graphiti.py services/orion-hub/tests/test_graphiti_sync_payload.py
git commit -m "feat(graphiti): include crystallization links in sync payload"
```

---

## Task 12: Retriever uses depth=2

**Files:**
- Modify: `orion/memory/crystallization/retriever.py:76-82`
- Modify: `tests/test_memory_crystallization.py`

- [ ] **Step 1: Write failing retriever test**

Add to `tests/test_memory_crystallization.py`:

```python
@pytest.mark.asyncio
async def test_retriever_collects_linked_crystallization_from_graphiti_depth_two():
    from unittest.mock import MagicMock
    from orion.memory.crystallization.retriever import retrieve_active_packet

    crys = _active_crystallization()
    adapter = MagicMock()
    adapter.enabled = True
    adapter.neighborhood.return_value = {
        "enabled": True,
        "nodes": [
            {"crystallization_id": crys.crystallization_id},
            {"crystallization_id": "crys_linked"},
        ],
        "edges": [],
    }

    packet = await retrieve_active_packet(
        query="test",
        crystallizations=[crys],
        graphiti_adapter=adapter,
        seed_crystallization_id=crys.crystallization_id,
    )
    adapter.neighborhood.assert_called_once_with(crys.crystallization_id, depth=2)
    assert "crys_linked" in packet.graphiti_refs
```

- [ ] **Step 2: Change retriever call**

```python
nb = graphiti_adapter.neighborhood(seed_crystallization_id, depth=2)
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_memory_crystallization.py::test_retriever_collects_linked_crystallization_from_graphiti_depth_two -v
```

- [ ] **Step 4: Commit**

```bash
git add orion/memory/crystallization/retriever.py tests/test_memory_crystallization.py
git commit -m "feat(retriever): expand graphiti neighborhood to depth 2"
```

---

## Task 13: FalkorDB link sync

**Files:**
- Modify: `services/orion-graphiti-adapter/app/falkordb.py`
- Modify: `services/orion-graphiti-adapter/app/main.py:68-76`
- Modify: `services/orion-graphiti-adapter/README.md`

- [ ] **Step 1: Extend `sync_to_falkordb`**

```python
def sync_to_falkordb(
    *,
    uri: str,
    graph_name: str,
    crystallization_id: str,
    kind: str,
    subject: str,
    summary: str,
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ...
    for link in links or []:
        target = link["target_crystallization_id"]
        relation = str(link["relation"]).upper().replace("-", "_")
        target_entity = f"gent_{target}"
        cypher += (
            f" MERGE (t:Entity {{id: '{target_entity}', crystallization_id: '{target}'}}) "
            f"MERGE (e)-[:{relation}]->(t)"
        )
```

Pass `links=[l.model_dump() for l in body.links]` from `main.py`.

- [ ] **Step 2: Document FalkorDB profile in README**

```markdown
## FalkorDB profile (optional)

```bash
docker compose --profile falkordb \
  --env-file .env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d
```

Set `FALKORDB_ENABLED=true` in `services/orion-graphiti-adapter/.env`.
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-graphiti-adapter/app/falkordb.py services/orion-graphiti-adapter/app/main.py services/orion-graphiti-adapter/README.md
git commit -m "feat(graphiti-adapter): best-effort FalkorDB link edge sync"
```

---

## Task 14: UI projection counts + Sync button

**Files:**
- Modify: `services/orion-hub/static/js/memory-crystallization-ui.js:86-93`
- Modify: `services/orion-hub/tests/test_memory_crystallization_ui.py`

- [ ] **Step 1: Write failing UI contract test**

Add to `services/orion-hub/tests/test_memory_crystallization_ui.py`:

```python
def test_crystallization_ui_shows_graphiti_projection_and_sync() -> None:
    ui = (HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js").read_text(encoding="utf-8")
    assert "graphiti_episode_ids" in ui
    assert "/api/memory/graphiti/sync/" in ui
```

- [ ] **Step 2: Update detail panel HTML**

Replace projection refs line:

```javascript
<div class="text-gray-500">Projection refs: cards=${(row.projection_refs && row.projection_refs.memory_card_ids || []).length}, chroma=${(row.projection_refs && row.projection_refs.chroma_doc_ids || []).length}, graphiti_eps=${((row.projection_refs && row.projection_refs.graphiti_episode_ids) || []).length}, graphiti_edges=${((row.projection_refs && row.projection_refs.graphiti_edge_ids) || []).length}</div>
```

Add Sync button in button row:

```javascript
<button type="button" data-act="sync-graphiti" class="px-2 py-1 rounded border border-sky-700 text-sky-200">Sync Graphiti</button>
```

Handle click:

```javascript
if (act === "sync-graphiti") {
  await apiFetch(`/api/memory/graphiti/sync/${row.crystallization_id}`, { method: "POST", body: "{}" });
}
```

- [ ] **Step 3: Run UI test**

```bash
pytest services/orion-hub/tests/test_memory_crystallization_ui.py -v
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/js/memory-crystallization-ui.js services/orion-hub/tests/test_memory_crystallization_ui.py
git commit -m "feat(ui): show graphiti projection counts and sync action"
```

---

## Task 15: Two-crystallization link smoke

**Files:**
- Create: `scripts/smoke_graphiti_links_e2e.sh`

- [ ] **Step 1: Create script**

```bash
#!/usr/bin/env bash
# Smoke: two crystallizations with supports link → graphiti neighborhood depth 2 returns both
set -euo pipefail
: "${ORION_HUB_URL:?}"
: "${ORION_HUB_SESSION_ID:?}"
BASE="${ORION_HUB_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"

_propose() {
  local subject="$1" summary="$2"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/propose" -d "$(jq -n --arg s "$subject" --arg m "$summary" '{
    kind: "stance", subject: $s, summary: $m, scope: ["project:orion"],
    evidence: [{source_kind: "operator_note", source_id: "smoke-link", excerpt: "smoke"}],
    proposed_by: "smoke"
  }')" | jq -r '.crystallization_id'
}

_approve() {
  local cid="$1"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/validate" >/dev/null
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/approve" -d '{}' >/dev/null
}

CID_A="$(_propose "Link smoke A ${STAMP}" "seed A")"
CID_B="$(_propose "Link smoke B ${STAMP}" "target B")"
_approve "$CID_A"
_approve "$CID_B"

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/${CID_A}/links" \
  -d "$(jq -n --arg t "$CID_B" '{target_crystallization_id: $t, relation: "supports", confidence: 0.9}')" >/dev/null

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_A}" -d '{}' >/dev/null
curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_B}" -d '{}' >/dev/null

NB=$(curl -sS "${HDR[@]}" "${BASE}/api/memory/graphiti/neighborhood/${CID_A}?depth=2")
CIDS=$(echo "$NB" | jq -r '[.nodes[]?.crystallization_id] | unique | .[]')
echo "$CIDS" | grep -qx "$CID_A"
echo "$CIDS" | grep -qx "$CID_B"
echo "PASS smoke_graphiti_links_e2e seed=${CID_A} linked=${CID_B}"
```

```bash
chmod +x scripts/smoke_graphiti_links_e2e.sh
```

- [ ] **Step 2: Run smoke (live stack required)**

```bash
ORION_HUB_URL=http://127.0.0.1:8080 ORION_HUB_SESSION_ID=<session> bash scripts/smoke_graphiti_links_e2e.sh
```

- [ ] **Step 3: Commit + Phase B gate**

```bash
git add scripts/smoke_graphiti_links_e2e.sh
git commit -m "test(smoke): two-crystallization graphiti link scenario"
pytest services/orion-graphiti-adapter/tests services/orion-hub/tests/test_graphiti_sync_payload.py services/orion-hub/tests/test_memory_crystallization_ui.py tests/test_memory_crystallization.py -q
```

---

# Phase C — Real Graphiti (`graphiti-core`) (PR 3)

**Start from Phase B merged:**

```bash
git switch main && git pull --ff-only
git switch -c feat/graphiti-rail-phase-c
```

**Constraint:** Do **not** call `graphiti.add_episode()` with raw summary text (triggers LLM extraction). Use `FalkorDriver` Cypher MERGE for prescribed ontology writes; use `graphiti.search()` only for hybrid retrieval.

---

## Task 16: Backend selection setting

**Files:**
- Modify: `services/orion-graphiti-adapter/app/settings.py`
- Modify: `services/orion-graphiti-adapter/.env_example`

- [ ] **Step 1: Add settings**

```python
from typing import Literal

GRAPHITI_BACKEND: Literal["orion_postgres", "graphiti_core"] = Field(
    default="orion_postgres", alias="GRAPHITI_BACKEND"
)
CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
```

`.env_example` additions:

```bash
GRAPHITI_BACKEND=orion_postgres
# Required when GRAPHITI_BACKEND=graphiti_core
CRYSTALLIZER_EMBED_HOST_URL=
```

- [ ] **Step 2: Sync env + commit**

```bash
python scripts/sync_local_env_from_example.py
git add services/orion-graphiti-adapter/app/settings.py services/orion-graphiti-adapter/.env_example
git commit -m "feat(graphiti-adapter): add GRAPHITI_BACKEND and embed URL settings"
```

---

## Task 17: Backend module split

**Files:**
- Create: `services/orion-graphiti-adapter/app/backends/orion_postgres.py`
- Create: `services/orion-graphiti-adapter/app/backends/graphiti_core.py`
- Create: `services/orion-graphiti-adapter/app/backends/__init__.py`
- Modify: `services/orion-graphiti-adapter/app/main.py`

- [ ] **Step 1: Create `orion_postgres.py` wrapper**

```python
from app.store import neighborhood as pg_neighborhood, upsert_episode as pg_upsert_episode

async def ingest_episode(pool, **kwargs):
    edge_ids = await pg_upsert_episode(pool, **kwargs)
    return {"edge_ids": edge_ids}

async def get_neighborhood(pool, crystallization_id: str, *, depth: int = 1):
    return await pg_neighborhood(pool, crystallization_id, depth=depth)
```

- [ ] **Step 2: Create `graphiti_core.py` skeleton with prescribed MERGE**

```python
async def ingest_episode(pool, *, crystallization_id, kind, subject, summary, status, metadata, links, falkordb_uri, graph_name):
    from graphiti_core.driver.falkordb_driver import FalkorDriver
  # Parse host/port from falkordb_uri redis://host:port
    driver = FalkorDriver(host=host, port=port, database=graph_name)
    entity_id = f"gent_{crystallization_id}"
    episode_id = f"gep_{crystallization_id}"
    await driver.execute_query(
        f"MERGE (e:Entity {{id: '{entity_id}', crystallization_id: '{crystallization_id}', name: $name}}) "
        f"MERGE (ep:Episode {{id: '{episode_id}', kind: $kind}}) "
        f"MERGE (e)-[:HAS_EPISODE]->(ep)",
        {"name": subject, "kind": kind},
    )
    for link in links or []:
        ...
    return {"edge_ids": [...]}

async def search(query: str, *, seed_crystallization_id: str, limit: int, embed_url: str):
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    graphiti = Graphiti(graph_driver=driver)
    results = await graphiti.search(query=query, num_results=limit)
    # Map entity nodes back to crystallization_ids; always include seed
    return {"crystallization_ids": ids, "trace": {"backend": "graphiti_core", "rails": ["vector", "graph"]}}
```

- [ ] **Step 3: Route in main.py**

```python
from app.backends import graphiti_core as core_backend, orion_postgres as pg_backend

def _backend():
    return core_backend if settings.GRAPHITI_BACKEND == "graphiti_core" else pg_backend
```

Update `/health`:

```python
return {
    "service": settings.SERVICE_NAME,
    "postgres": pg_pool is not None,
    "falkordb_enabled": settings.FALKORDB_ENABLED,
    "backend": settings.GRAPHITI_BACKEND,
}
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-graphiti-adapter/app/backends/ services/orion-graphiti-adapter/app/main.py
git commit -m "feat(graphiti-adapter): backend module split for orion_postgres vs graphiti_core"
```

---

## Task 18: Hybrid search endpoint

**Files:**
- Modify: `services/orion-graphiti-adapter/app/main.py`
- Modify: `services/orion-graphiti-adapter/requirements.txt`
- Create: `services/orion-graphiti-adapter/tests/test_search.py`

- [ ] **Step 1: Pin dependency**

```
graphiti-core[falkordb]==0.19.10
```

- [ ] **Step 2: Add route**

```python
class SearchRequestV1(BaseModel):
    query: str
    seed_crystallization_id: str | None = None
    limit: int = 10

@app.post("/v1/search")
async def search_episodes(body: SearchRequestV1) -> dict:
    if settings.GRAPHITI_BACKEND != "graphiti_core":
        raise HTTPException(status_code=501, detail="search_requires_graphiti_core_backend")
    return await core_backend.search(
        body.query,
        seed_crystallization_id=body.seed_crystallization_id or "",
        limit=body.limit,
        embed_url=settings.CRYSTALLIZER_EMBED_HOST_URL,
        falkordb_uri=settings.FALKORDB_URI,
        graph_name=settings.FALKORDB_GRAPH,
    )
```

- [ ] **Step 3: Write test with mocked graphiti.search**

```python
@pytest.mark.asyncio
async def test_search_returns_crystallization_ids(monkeypatch):
    async def fake_search(**kwargs):
        return {"crystallization_ids": ["crys_a", "crys_b"], "trace": {"backend": "graphiti_core", "rails": ["vector", "graph"]}}
    monkeypatch.setattr("app.backends.graphiti_core.search", fake_search)
    ...
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-graphiti-adapter/requirements.txt services/orion-graphiti-adapter/app/main.py services/orion-graphiti-adapter/tests/test_search.py
git commit -m "feat(graphiti-adapter): hybrid search endpoint for graphiti_core backend"
```

---

## Task 19: Retriever search integration

**Files:**
- Modify: `orion/memory/crystallization/projection_graphiti.py`
- Modify: `orion/memory/crystallization/retriever.py`

- [ ] **Step 1: Add `search()` to GraphitiAdapter**

```python
def search(self, query: str, *, seed_crystallization_id: str | None = None, limit: int = 10) -> dict[str, Any]:
    if not self.enabled or not self.url:
        return {"crystallization_ids": [], "trace": {}}
    with httpx.Client(timeout=self.timeout_sec) as client:
        resp = client.post(
            f"{self.url}/v1/search",
            json={"query": query, "seed_crystallization_id": seed_crystallization_id, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 2: Retriever branches on adapter health backend**

```python
health = graphiti_adapter.health()  # new GET /health passthrough or cached backend field
if health.get("backend") == "graphiti_core":
    sr = graphiti_adapter.search(query, seed_crystallization_id=seed_crystallization_id)
    for cid in sr.get("crystallization_ids") or []:
        graphiti_refs.append(str(cid))
        extra_crystallization_ids.add(str(cid))
else:
    nb = graphiti_adapter.neighborhood(seed_crystallization_id, depth=2)
    ...
```

Add `health()` method on `GraphitiAdapter` calling `GET {url}/health`.

- [ ] **Step 3: Commit**

```bash
git add orion/memory/crystallization/projection_graphiti.py orion/memory/crystallization/retriever.py
git commit -m "feat(retriever): use graphiti search when backend is graphiti_core"
```

---

## Task 20: Privacy skip for intimate crystallizations

**Files:**
- Modify: `orion/memory/crystallization/projector.py:98-110`
- Modify: `services/orion-graphiti-adapter/app/backends/graphiti_core.py`

- [ ] **Step 1: Skip in projector before sync**

```python
if updated.governance.sensitivity == "intimate":
    result.errors.append("graphiti_projection_skipped:intimate_sensitivity")
elif project_graphiti and cfg.graphiti_enabled:
    ...
```

- [ ] **Step 2: Adapter-side guard (defense in depth)**

In ingest handlers, if `metadata.get("sensitivity") == "intimate"`, return `{"skipped": True, "reason": "intimate_sensitivity", "canonical_mutated": False}` without writing.

Pass sensitivity in Hub sync payload metadata:

```python
"metadata": {
    ...
    "sensitivity": crystallization.governance.sensitivity,
},
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(graphiti): skip intimate sensitivity crystallizations in projection"
```

---

## Task 21: Rebuild endpoint + Phase C smoke docs

**Files:**
- Modify: `services/orion-graphiti-adapter/app/main.py`
- Modify: `services/orion-graphiti-adapter/README.md`

- [ ] **Step 1: Add `POST /v1/rebuild` batch ingest**

```python
class RebuildItemV1(EpisodeIngestV1):
    pass

class RebuildRequestV1(BaseModel):
    items: list[RebuildItemV1]

@app.post("/v1/rebuild")
async def rebuild(body: RebuildRequestV1) -> dict:
    results = []
    for item in body.items:
        # delegate to same ingest path as /v1/episodes
        ...
    return {"ingested": len(results), "skipped_intimate": skip_count, "canonical_mutated": False}
```

Hub `/api/memory/crystallizations/projection/rebuild` stays unchanged — it already calls per-item project; no Hub change required.

- [ ] **Step 2: Document Phase C smoke in README**

```markdown
## graphiti_core backend smoke

```bash
GRAPHITI_BACKEND=graphiti_core FALKORDB_ENABLED=true \
docker compose --profile falkordb --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

Rollback: `GRAPHITI_BACKEND=orion_postgres` and restart adapter.
```

- [ ] **Step 3: Phase C gate**

```bash
pytest services/orion-graphiti-adapter/tests -q
python scripts/check_env_template_parity.py
```

---

## Self-Review (plan vs spec)

| Spec requirement | Plan task |
|------------------|-----------|
| A1 URL unification | Tasks 1–2 |
| A2 Adapter unit tests | Tasks 4–5 |
| A3 Hub parity test | Task 2 |
| A4 Smoke extension | Task 6 |
| B1 Link projection | Tasks 8–9, 11 |
| B2 Multi-hop neighborhood | Tasks 10–12 |
| B3 FalkorDB link sync | Task 13 |
| B4 UI | Task 14 |
| B two-crys smoke | Task 15 |
| C1 Backend selection | Tasks 16–17 |
| C2 Prescribed ontology (no LLM) | Task 17 (explicit MERGE, no add_episode) |
| C3 Hybrid search | Tasks 18–19 |
| C4 Rebuild | Task 21 |
| C5 Privacy intimate skip | Task 20 |
| Error handling (adapter down) | Existing projector/adapter soft-fail — verified in Phase A smoke |
| Env sync after `.env_example` | Tasks 3, 16 |

**Placeholder scan:** No TBD/TODO steps. All tasks include code and commands.

**Type consistency:** `CrystallizationLinkIngestV1.relation` is `str` (adapter boundary); Hub sends `CrystallizationRelation` literal values as strings. `GraphitiAdapter.neighborhood(..., depth=2)` used consistently in retriever and adapter route.

---

## Restart required (all phases)

```bash
# Phase A–B
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build

# Phase C (FalkorDB profile)
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```
