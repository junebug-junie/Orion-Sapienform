# Spec: Knowledge Forge ideation review v1 (v1.1)

**Id:** `spec:knowledge-forge-ideation-review-v1`  
**Status:** execution_ready  
**Component:** `orion-knowledge-forge`

## Goal

Add **proposal-only** Claude-backed ideation review so operators and agents can run structured critiques against corpus paths without mutating canonical knowledge. Claude proposes; humans accept.

## API

### `POST /v1/ideation/run`

**Request (example):**

```json
{
  "task": "Critique Knowledge Forge v1 and propose v1.1",
  "mode": "arsonist_review",
  "input_paths": ["services/orion-knowledge-forge"],
  "write_review": false
}
```

**Modes:** `arsonist_review`, `spec_critique`, `missing_questions`, `context_pack_review`, `implementation_plan_review`

**Response fields:** `run_id`, `provider`, `model`, `status`, `summary`, `content`, `artifact_path`, `warnings`, `usage`

## Providers

| Provider | Role |
|----------|------|
| `local` (default) | Deterministic noop/structured output for tests and offline runs |
| `anthropic` (optional) | Live Claude via `ANTHROPIC_API_KEY` and `KNOWLEDGE_FORGE_ANTHROPIC_MODEL` |

## Write policy

- Writes **only** to `reviews/pending/ideation-*.md` when `write_review=true` and `KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED=true` (plus operator token when required)
- **No** canonical mutation of `claims/accepted`, `specs/execution_ready`, or `decisions/`
- **No** MCP exposure in v1.1

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `KNOWLEDGE_FORGE_IDEATION_ENABLED` | `true` | 503 when false |
| `KNOWLEDGE_FORGE_IDEATION_PROVIDER` | `local` | `local` or `anthropic` |
| `KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED` | `false` | Gate filesystem writes |
| `KNOWLEDGE_FORGE_ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Anthropic model id |
| `ANTHROPIC_API_KEY` | empty | Required when provider=`anthropic` |

## Files likely to touch

- `services/orion-knowledge-forge/app/routers/ideation.py`
- `services/orion-knowledge-forge/app/ideation/*`
- `services/orion-knowledge-forge/app/providers/*`
- `services/orion-knowledge-forge/app/api_schemas.py`, `settings.py`, `main.py`
- `services/orion-knowledge-forge/tests/test_ideation_api.py`
- `services/orion-knowledge-forge/.env_example`
- `CLAUDE.md`, `services/orion-knowledge-forge/CLAUDE.md`
- `.claude/commands/forge-*.md`

## Acceptance checks

- [ ] `pytest services/orion-knowledge-forge/tests/ -q` passes (including `test_ideation_api.py`)
- [ ] Local provider returns deterministic structured sections
- [ ] `write_review=true` + write disabled → content + warning, `artifact_path=null`
- [ ] `write_review=true` + write enabled → file under `reviews/pending/` only
- [ ] Invalid mode → 422
- [ ] Anthropic without API key → settings validation error
- [ ] `../` traversal and out-of-corpus absolute paths rejected
- [ ] Accepted claims unchanged after ideation write
- [ ] `KNOWLEDGE_FORGE_IDEATION_ENABLED=false` → 503

## Source claims

- `claim:orion:knowledge-forge:0001` — v1 FastAPI surface
- `claim:orion:knowledge-forge:0002` — Hub Forge tab
- `claim:orion:knowledge-forge:0003` — git-tracked corpus authority
- `claim:orion:knowledge-forge:0004` — disputed/stale/superseded excluded by default
- `claim:orion:knowledge-forge:0005` — v1 non-goals (no GraphDB/vector/autonomous rewrite/silent mutation)

## Non-goals (v1.1)

- Hub “Run Ideation Review” button
- MCP server
- Prompt caching
- Service shells out to `claude -p`
