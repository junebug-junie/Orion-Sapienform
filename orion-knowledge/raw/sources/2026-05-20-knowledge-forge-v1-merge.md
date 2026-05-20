# Source reference: Knowledge Forge v1 merge state

Imported from repo PR summary:
`docs/PR-orion-knowledge-forge-v1.md` (branch `feat/orion-knowledge-forge-v1`, base v0 corpus + CLI).

## Merged v1 summary

Knowledge Forge v1 adds a thin FastAPI service on port **8630** and an Orion Hub **Forge** tab. The git-tracked `orion-knowledge/` corpus remains the source of truth. Operators inspect knowledge health, search claims/specs/sources, view pending reviews, and compile task-specific context packs for Cursor/Codex/Claude/Orion.

```text
v0: repo + CLI (lint, review, compile)
v1: repo + FastAPI service + Hub proxy + Forge tab
```

## Service API (v1)

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

## Hub proxy (`/api/knowledge/*`)

Forge tab: status, search, claims, specs, reviews, context-pack compiler, debug JSON. Proxy routes mirror the service read/compile surface (review accept/reject deferred in Hub UI).

## Arsonist constraints (v1)

- Read canonical knowledge; write only generated context packs when `KNOWLEDGE_FORGE_WRITE_ENABLED=true`
- Disputed/stale/superseded claims excluded from compile unless explicitly requested
- Malformed YAML → warnings, not process crash
- Generated packs include **Source lineage**
- No embeddings, GraphDB, background organizers, or Hub-side editing of accepted claims
- No silent canonical claim/spec/decision mutation

## Explicit non-goals (v1)

- Full CMS / ontology dashboard
- GraphDB / RDF / vector search
- Autonomous background rewriting or chat ingest
- Hub editor for accepted claims
- Hub UI for review accept/reject (service endpoints exist, token-gated)

## Test evidence (v1 PR)

| Suite | Result |
|-------|--------|
| v0 package tests | 17 passed |
| v1 service tests | 7 passed |
| Hub proxy + Forge tab tests | 5 passed |

CI: `.github/workflows/orion-knowledge-forge-tests.yml`.
