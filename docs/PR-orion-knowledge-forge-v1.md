# PR: Orion Knowledge Forge v1

**Branch:** `feat/orion-knowledge-forge-v1`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-orion-knowledge-forge-v1`  
**Base:** `feat/orion-knowledge-forge-v0` (v0 corpus + CLI)  
**Head:** `feat/orion-knowledge-forge-v1` (v1 service + Hub Forge tab)

## Summary

Ships **Knowledge Forge v1**: a thin FastAPI service plus an Orion Hub **Forge** tab. The git-tracked `orion-knowledge/` corpus remains the source of truth. Operators can inspect knowledge health, search claims/specs/sources, view pending reviews, and **compile task-specific context packs** for Cursor/Codex/Claude/Orion — without vectors, GraphDB, autonomous rewriting, or silent canonical mutation.

```text
v0: repo + CLI (lint, review, compile)
v1: repo + FastAPI service + Hub proxy + Forge tab
```

## What changed

### Knowledge Forge service (`services/orion-knowledge-forge/`)

| Piece | Role |
|-------|------|
| `app/main.py` | FastAPI on port **8630**, `/health` + `/v1/*` |
| `app/settings.py` | `KNOWLEDGE_FORGE_*` config (write off by default) |
| `app/api_schemas.py` | Status, claim/spec summaries, compile request/result |
| `app/service.py` | Corpus load with per-file warnings, search, compile orchestration |
| `app/routers/v1.py` | Full v1 API surface |
| `orion/knowledge_forge/api_compile.py` | Multi-spec/claim context pack compiler with lineage |

**API (service):**

```text
GET  /health
GET  /v1/status
GET  /v1/claims
GET  /v1/claims/search?q=...
GET  /v1/search?q=...          # claims + specs + sources
GET  /v1/specs
GET  /v1/specs/{spec_id}
GET  /v1/decisions
GET  /v1/context-packs
GET  /v1/context-packs/{pack_id}
POST /v1/context-packs/compile
GET  /v1/reviews/pending
GET  /v1/sources
POST /v1/reviews/{id}/accept|reject   # operator token optional
```

**Arsonist constraints enforced:**

- Read canonical knowledge; write only generated context packs (when `KNOWLEDGE_FORGE_WRITE_ENABLED=true`)
- Disputed/stale/superseded claims excluded from compile unless explicitly requested
- Malformed YAML → warnings, not process crash
- Generated packs include **Source lineage** section
- No embeddings, GraphDB, background organizers, or Hub-side editing of accepted claims

### Hub integration (`services/orion-hub/`)

| Piece | Role |
|-------|------|
| `app/settings.py` | `KNOWLEDGE_FORGE_BASE_URL`, `KNOWLEDGE_FORGE_PROXY_TIMEOUT_SEC` |
| `scripts/api_routes.py` | Proxy `/api/knowledge/*` → forge service |
| `templates/index.html` + `static/js/app.js` | **Forge** tab (status, search, claims, specs, reviews, compiler, debug JSON) |
| `docker-compose.yml` | Hub env + optional `orion-knowledge-forge` profile on `app-net` |

**Hub proxy routes:**

```text
GET  /api/knowledge/health
GET  /api/knowledge/status
GET  /api/knowledge/claims
GET  /api/knowledge/claims/search
GET  /api/knowledge/search
GET  /api/knowledge/specs
GET  /api/knowledge/specs/{spec_id}
GET  /api/knowledge/decisions
GET  /api/knowledge/context-packs
GET  /api/knowledge/context-packs/{pack_id}
POST /api/knowledge/context-packs/compile
GET  /api/knowledge/reviews/pending
GET  /api/knowledge/sources
```

### Env / Docker

| File | Keys |
|------|------|
| `services/orion-knowledge-forge/.env_example` + `.env` | `KNOWLEDGE_FORGE_*`, `PORT=8630`, `ORION_KNOWLEDGE_ROOT` |
| `services/orion-hub/.env_example` + `.env` | `KNOWLEDGE_FORGE_BASE_URL=http://127.0.0.1:8630`, `ORION_KNOWLEDGE_ROOT` |
| `services/orion-hub/docker-compose.yml` | Hub passes forge URL; optional forge service under profile `knowledge-forge` |

`.gitignore` exception: `!services/orion-knowledge-forge/.env` so committed local defaults match `.env_example`.

## Commits (v1 delta on v0)

1. `feat: add orion-knowledge-forge v1 FastAPI service`
2. `feat: wire Hub proxy routes for Knowledge Forge v1`
3. `fix: align Knowledge Forge v1 API schemas and search breadth`
4. `feat: add Hub Forge tab for Knowledge Forge v1`
5. `fix: align Forge compile write default with API and expand CI`

## Tests

| Suite | Command | Result |
|-------|---------|--------|
| v0 package | `PYTHONPATH=. pytest tests/test_knowledge_forge_*.py -q` | 17 passed |
| v1 service | `PYTHONPATH=services/orion-knowledge-forge pytest services/orion-knowledge-forge/tests/ -q` | 7 passed |
| Hub proxy + tab | `PYTHONPATH=services/orion-hub pytest services/orion-hub/tests/test_knowledge_forge_proxy_routes.py services/orion-hub/tests/test_forge_hub_tab.py -q` | 5 passed |

CI: `.github/workflows/orion-knowledge-forge-tests.yml` runs v0, v1 service, and Hub forge/proxy tests.

## E2E smoke (local)

```bash
# Knowledge Forge direct
curl http://127.0.0.1:8630/health
curl http://127.0.0.1:8630/v1/status
curl "http://127.0.0.1:8630/v1/claims/search?q=substrate"

curl -X POST http://127.0.0.1:8630/v1/context-packs/compile \
  -H "Content-Type: application/json" \
  -d '{
    "task": "implement substrate graph primitives v0",
    "target": "cursor",
    "spec_ids": ["spec:substrate:telemetry:v0"],
    "include_disputed": false,
    "include_stale": false,
    "write_file": false
  }'

# Hub proxy
curl http://127.0.0.1:8080/api/knowledge/status
curl http://127.0.0.1:8080/api/knowledge/reviews/pending
```

Start stack (example):

```bash
cd services/orion-knowledge-forge && docker compose --profile knowledge-forge up -d
cd services/orion-hub && docker compose up -d orion-hub
# Open Hub → Forge tab (#forge)
```

## Acceptance criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Service summarizes v0 corpus | ✅ |
| 2 | Hub has visible Forge tab | ✅ |
| 3 | Tab shows counts, warnings, claims, specs, reviews, packs | ✅ |
| 4 | Operator can search claims/specs/sources | ✅ |
| 5 | Operator can compile context pack from UI | ✅ |
| 6 | Disputed/stale excluded by default | ✅ |
| 7 | Malformed files → warnings, not crash | ✅ |
| 8 | Generated packs include lineage | ✅ |
| 9 | No silent canonical claim/spec/decision mutation | ✅ |
| 10 | Curl smoke + backend tests | ✅ |

## Non-goals (explicitly out of scope)

- Full CMS / ontology dashboard
- GraphDB / RDF / vector search
- Autonomous background rewriting or chat ingest
- Hub editor for accepted claims
- Review accept/reject in Hub UI (service endpoints exist, token-gated; UI deferred)

## Remaining risks

- Live Hub → Forge integration not smoke-tested in CI (unit/proxy tests only).
- Docker corpus mount is `:ro`; enabling writes requires writable mount + `KNOWLEDGE_FORGE_WRITE_ENABLED=true`.
- Operator review accept/reject on service has no dedicated tests yet.

## Reviewer notes

- Forge **Write file** checkbox defaults **off** to match API `write_file=false` and `KNOWLEDGE_FORGE_WRITE_ENABLED=false`.
- Primary operator action: **Compile Context Pack** — not “ingest everything.”
