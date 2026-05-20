# PR: Orion Knowledge Forge — Ideation v1.2 (Hub proxy seam)

## Summary

- **Hub proxy route:** `POST /api/knowledge/ideation/run` forwards to Knowledge Forge `POST /v1/ideation/run` via the existing `_proxy_knowledge_forge_request` helper.
- **Why explicit alias:** The catch-all `/api/knowledge/{path:path}` forwards paths without the `v1/` prefix (e.g. `ideation/run`), which does not match Forge’s `/v1/ideation/run` router. The dedicated alias follows the same pattern as `context-packs/compile`.
- **Scope:** Hub-only. No Forge service changes, no Hub UI panel, no MCP/GraphDB/vectors/autonomous behavior.

## Architecture

```text
Hub client                    Orion Hub                         Knowledge Forge
POST /api/knowledge/          proxy_knowledge_forge_ideation_run   POST /v1/ideation/run
     ideation/run      →     _proxy_knowledge_forge_request   →   IdeationRunner
```

Headers (including `X-Knowledge-Forge-Token` for gated writes), body, and query params pass through unchanged.

## Changes

| File | Change |
|------|--------|
| `services/orion-hub/scripts/api_routes.py` | Add `POST /api/knowledge/ideation/run` → `v1/ideation/run` (before catch-all) |
| `services/orion-hub/tests/test_knowledge_forge_proxy_routes.py` | Extend alias-route test to cover ideation handler |

## Configuration

No new env vars. Existing wiring is sufficient:

| Variable | Location | Default |
|----------|----------|---------|
| `KNOWLEDGE_FORGE_BASE_URL` | `app/settings.py`, `.env_example`, `docker-compose.yml` | `http://127.0.0.1:8630` |

## Verification

```bash
cd .worktrees/feat-orion-knowledge-forge-ideation-v1.2
PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_knowledge_forge_proxy_routes.py -q
# 3 passed
```

**Manual smoke (with Forge running on 8630, Hub on 8080):**

```bash
curl -sS -X POST http://127.0.0.1:8080/api/knowledge/ideation/run \
  -H 'Content-Type: application/json' \
  -d '{"task":"Hub proxy smoke","mode":"arsonist_review","write_review":false}'
```

Expected: JSON ideation result (not 404).

Direct Forge route (unchanged):

```bash
curl -sS -X POST http://127.0.0.1:8630/v1/ideation/run \
  -H 'Content-Type: application/json' \
  -d '{"task":"Direct smoke","mode":"arsonist_review","write_review":false}'
```

## Commits

1. `feat(hub): proxy Knowledge Forge ideation run route`
2. `test(hub): align ideation proxy test body with Forge schema`

## Test plan

- [x] Unit: `test_knowledge_forge_proxy_routes.py` — alias forwards `v1/ideation/run`
- [ ] Integration: Hub proxy smoke with live Forge stack
- [ ] Regression: existing Knowledge Forge tab/proxy routes still work

## Follow-ups (out of scope)

- Hub UI panel for ideation runs
- Proxy timeout tuning for Anthropic-backed ideation (default 15s Hub proxy vs longer Forge provider timeout)
- Update `claim-knowledge-forge-v1-hub.yaml` to mention ideation proxy

## Branch

`feat/orion-knowledge-forge-ideation-v1.2`  
Worktree: `.worktrees/feat-orion-knowledge-forge-ideation-v1.2`
