# PR: Orion Knowledge Forge — Ideation v1.1 (Claude / Anthropic seam)

## Summary

- **Track A (Claude Code cockpit):** Repo- and service-level `CLAUDE.md`, plus `.claude/commands/forge-{critique,spec,context-pack,arsonist}.md` for repeatable ideation from Claude Code without embedding Claude into the service runtime.
- **Track B (service seam):** `POST /v1/ideation/run` with `local` (default) and `anthropic` providers, prompt builder, and gated writes to `reviews/pending/ideation-*.md` only — proposals never mutate accepted claims/specs/decisions.
- **Hardening:** Input path sandboxing, operator token required when `write_review=true`, Settings-backed `ANTHROPIC_API_KEY`, lazy `anthropic` import for local-only boot.

## Architecture

```text
Claude Code (cockpit)          Knowledge Forge service (repeatable runs)
├── CLAUDE.md                  ├── POST /v1/ideation/run
├── .claude/commands/forge-*   ├── IdeationRunner
└── /memory verification     ├── providers: local | anthropic
                               └── writer → reviews/pending/ only
```

**Rule:** Claude proposes; humans accept. Ideation output is `status: proposed`, not canonical truth.

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

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `KNOWLEDGE_FORGE_IDEATION_ENABLED` | `true` | 503 when false |
| `KNOWLEDGE_FORGE_IDEATION_PROVIDER` | `local` | `local` or `anthropic` |
| `KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED` | `false` | Gate filesystem writes |
| `KNOWLEDGE_FORGE_ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Anthropic model id |
| `ANTHROPIC_API_KEY` | empty | Required when provider=`anthropic` |

Updated in: `.env`, `.env_example`, `docker-compose.yml`, `app/settings.py`.

## Files changed

- `CLAUDE.md`, `services/orion-knowledge-forge/CLAUDE.md`
- `.claude/commands/forge-*.md` (4 commands)
- `services/orion-knowledge-forge/app/ideation/*`
- `services/orion-knowledge-forge/app/providers/*`
- `services/orion-knowledge-forge/app/routers/ideation.py`
- `services/orion-knowledge-forge/app/api_schemas.py`, `settings.py`, `main.py`
- `services/orion-knowledge-forge/tests/test_ideation_api.py`
- `services/orion-knowledge-forge/requirements.txt` (+ `anthropic==0.49.0`)

## Test plan

- [x] `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-knowledge-forge/tests/ -q` — **16 passed**
- [x] Local provider returns deterministic structured sections
- [x] `write_review=true` + write disabled → content + warning, `artifact_path=null`
- [x] `write_review=true` + write enabled → file under `reviews/pending/`
- [x] Invalid mode → 422
- [x] Anthropic without API key → Settings validation error
- [x] Settings `ANTHROPIC_API_KEY` wired to provider
- [x] `../` traversal and out-of-corpus absolute paths rejected
- [x] Accepted claims unchanged after ideation write
- [x] `KNOWLEDGE_FORGE_IDEATION_ENABLED=false` → 503

### Smoke (local provider)

```bash
cd services/orion-knowledge-forge
# ensure KNOWLEDGE_FORGE_IDEATION_PROVIDER=local in .env
python app/main.py &
curl -s -X POST http://localhost:8630/v1/ideation/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Critique Knowledge Forge v1 and propose v1.1",
    "mode": "arsonist_review",
    "input_paths": ["claims/accepted"],
    "write_review": false
  }' | jq .
```

## Non-goals (this PR)

- Hub “Run Ideation Review” button (v1.2)
- MCP server exposure
- Prompt caching optimization
- Shelling out to `claude -p` from the service

## Commits

1. `feat: add Claude Code memory and forge ideation commands`
2. `feat: add Knowledge Forge ideation seam with Anthropic adapter`
3. `fix: harden ideation auth, path sandbox, and API key wiring`
