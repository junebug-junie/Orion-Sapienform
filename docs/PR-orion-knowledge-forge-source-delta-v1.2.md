# PR: Orion Knowledge Forge — Source Delta Review v1.2

## Summary

Implements the **source-ingest → proposed delta review** spine for Knowledge Forge. Given a design doc path, Forge deterministically extracts claim candidates, registers the source under `raw/sources/`, and writes a human-review artifact to `reviews/pending/` — without mutating accepted claims, execution-ready specs, or decisions.

This is the first real HITL loop:

```text
source changed
  → ingest-source (CLI or API)
  → proposed claims/spec hints in pending review
  → human accepts/rejects later (existing review apply flow unchanged)
  → context packs compile from reviewed state
```

## Architecture

```text
design doc (path)
       ↓
orion/knowledge_forge/sources.py   parse + propose + write (allowlisted)
       ↓
CLI: python -m orion.knowledge_forge ingest-source
API: POST /v1/sources/ingest
Hub: POST /api/knowledge/sources/ingest  (proxy alias)
       ↓
orion-knowledge/raw/sources/{slug}.md
orion-knowledge/raw/sources/source-{slug}.yaml
orion-knowledge/reviews/pending/source-delta-{ts}-{slug}.md
```

**Read-only / never written by this slice:**
- `claims/accepted/`
- `specs/execution_ready/`
- `decisions/`

## Changes

| File | Change |
|------|--------|
| `orion/knowledge_forge/sources.py` | **New** — markdown parse, claim extraction, review builder, ingest orchestration, write allowlist |
| `orion/knowledge_forge/cli.py` | Add `ingest-source` subcommand |
| `services/orion-knowledge-forge/app/api_schemas.py` | `SourceIngestRequestV1`, `SourceIngestResultV1` |
| `services/orion-knowledge-forge/app/service.py` | `ingest_source()` service method |
| `services/orion-knowledge-forge/app/routers/v1.py` | `POST /v1/sources/ingest` + operator token gate on writes |
| `tests/test_knowledge_forge_source_ingest.py` | CLI/core unit + integration tests |
| `services/orion-knowledge-forge/tests/test_source_ingest_api.py` | API tests incl. write-disabled + operator token |
| `services/orion-hub/scripts/api_routes.py` | Hub proxy `POST /api/knowledge/sources/ingest` |
| `services/orion-hub/tests/test_knowledge_forge_proxy_routes.py` | Proxy alias coverage |

## New CLI

```bash
ORION_KNOWLEDGE_ROOT=$PWD/orion-knowledge \
PYTHONPATH=. python -m orion.knowledge_forge ingest-source \
  --path docs/some-design-doc.md \
  --source-id source:some-design-doc \
  --kind design_doc \
  --dry-run

ORION_KNOWLEDGE_ROOT=$PWD/orion-knowledge \
PYTHONPATH=. python -m orion.knowledge_forge ingest-source \
  --path docs/some-design-doc.md \
  --source-id source:some-design-doc \
  --kind design_doc \
  --write-review
```

CLI honors `KNOWLEDGE_FORGE_WRITE_ENABLED` (defaults to enabled when unset).

## New API

```http
POST /v1/sources/ingest
Content-Type: application/json

{
  "path": "/tmp/kf-test-design.md",
  "source_id": "source:test-design-doc",
  "kind": "design_doc",
  "write_review": true,
  "dry_run": false
}
```

Response fields: `source_id`, `status`, `source_path`, `review_path`, `proposed_claims`, `possibly_affected_specs`, `warnings`, `content`.

Write requires `KNOWLEDGE_FORGE_WRITE_ENABLED=true`. Operator token required when `KNOWLEDGE_FORGE_OPERATOR_TOKEN` is set and `write_review=true`.

## Hub proxy

Added: `POST /api/knowledge/sources/ingest` → `v1/sources/ingest` (explicit alias before catch-all).

No Hub UI panel — proxy only.

## Deterministic extractor (v1.2)

From markdown headings, extracts bullets under:
- Requirements, Decisions, Non-goals, Acceptance checks, Known traps, Implementation path

Outputs review artifact sections:
- Source / Source summary / Proposed claims (checkboxes) / Possibly affected specs / Suggested context packs / Human action needed

`possibly_affected_specs` uses lightweight keyword overlap against loaded specs (heuristic, not LLM).

## Configuration

No new env vars. Reuses existing Forge settings:

| Variable | Purpose |
|----------|---------|
| `ORION_KNOWLEDGE_ROOT` / `KNOWLEDGE_FORGE_REPO_ROOT` | Corpus root |
| `KNOWLEDGE_FORGE_WRITE_ENABLED` | Gate API/CLI writes |
| `KNOWLEDGE_FORGE_OPERATOR_TOKEN` | Gate authenticated writes |

## Test results

```bash
PYTHONPATH=. pytest tests/test_knowledge_forge_*.py -q
# 24 passed

PYTHONPATH=services/orion-knowledge-forge pytest services/orion-knowledge-forge/tests/ -q
# 21 passed

PYTHONPATH=. pytest services/orion-hub/tests/test_knowledge_forge_proxy_routes.py -q
# 3 passed
```

## Manual smoke

Dry-run (no corpus writes):

```bash
ORION_KNOWLEDGE_ROOT=$PWD/orion-knowledge \
PYTHONPATH=. python -m orion.knowledge_forge ingest-source \
  --path /tmp/kf-test-design.md \
  --source-id source:test-design-doc \
  --kind design_doc \
  --dry-run
# status=proposed, proposed_claim lines printed, no files under raw/sources or reviews/pending
```

Write mode:

```bash
ORION_KNOWLEDGE_ROOT=$PWD/orion-knowledge \
PYTHONPATH=. python -m orion.knowledge_forge ingest-source \
  --path /tmp/kf-test-design.md \
  --source-id source:test-design-doc \
  --kind design_doc \
  --write-review
# writes raw/sources/test-design-doc.md, source-test-design-doc.yaml, reviews/pending/source-delta-*.md
```

## Skipped / deferred

- LLM / frontier extraction (deterministic only)
- MCP, GraphDB, vector search, autonomous rewrite, chat ingestion
- Hub UI for source delta review
- Auto-accept of proposed claims (human review required)
- Numbered-list / `###` subsection parsing (future parser hardening)
- Source change detection / watch mode (ingest is explicit in v1.2)

## Branch

- **Worktree:** `.worktrees/feat-orion-knowledge-forge-source-delta-v1.2`
- **Branch:** `feat/orion-knowledge-forge-source-delta-v1.2`
- **Commits:** `c0c7aac7` (feature), follow-up (review hardening)

## Test plan

- [ ] CLI dry-run on a real design doc — verify no corpus mutation
- [ ] CLI `--write-review` — verify only `raw/sources/` + `reviews/pending/` change
- [ ] API dry-run with `write_review=false` — `review_path: null`, content populated
- [ ] API write with `KNOWLEDGE_FORGE_WRITE_ENABLED=true` — pending review created
- [ ] Hub proxy curl through Orion Hub when Forge running on 8630
- [ ] Existing lint/compile/ideation flows unchanged
