# orion-harness-governor

Bus worker for unified Hub turns. Listens on `orion:harness:run:request`, runs fcc motor + three-beat finalize (5a/5b/5c), replies with `HarnessRunV1`, and publishes audit artifacts.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_HARNESS_RUN_REQUEST` | `orion:harness:run:request` | RPC intake from Hub |
| `CHANNEL_HARNESS_RESULT_PREFIX` | `orion:harness:run:result:` | Reply channel prefix |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Audit publish after each run |
| `CHANNEL_FINALIZE_APPRAISAL_REQUEST` | `orion:substrate:finalize_appraisal:request` | 5a draft molecule RPC |
| `CHANNEL_POST_TURN_CLOSURE` | `orion:substrate:post_turn_closure` | Step 7 learning closure |

## Flow

```text
LISTEN orion:harness:run:request
  → validate HarnessRunRequestV1 + thought disposition
  → HarnessRunner.run() — fcc motor + grammar receipts + draft_text
  → run_harness_finalize_chain() — 5a substrate / 5b reflect / 5c voice / 6b outcome
  → REPLY HarnessRunV1
  → PUBLISH orion:harness:run:artifact
  → emit_post_turn_closure (step 7)
```

## Local checks

```bash
PYTHONPATH=services/orion-harness-governor:. ./orion_dev/bin/python -m pytest services/orion-harness-governor/tests/ -v
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/harness/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml config
```

## Health

`GET http://localhost:7156/health`

## FCC MCP (Orion mode)

When `HARNESS_FCC_MCP_ENABLED=true`, harness turns spawn ephemeral MCP config (GitHub + Firecrawl; optional AI Town when `HARNESS_AITOWN_ENABLED=true`). The container image includes `docker`, `npx`, and the orion-aitown MCP package.

### Required secrets

Mount host `~/.fcc` (already wired in compose). In `~/.fcc/.env` (or path from `HARNESS_FCC_ENV_PATH`):

| Key | Used by |
|-----|---------|
| `GITHUB_PAT` | GitHub MCP (`docker run ghcr.io/github/github-mcp-server`) |
| `FIRECRAWL_API_KEY` | Firecrawl MCP (`npx firecrawl-mcp`) |

When `HARNESS_AITOWN_ENABLED=true`, also set `AITOWN_CONVEX_URL`, `AITOWN_ADMIN_KEY`, and `AITOWN_WORLD_ID` (optional: `AITOWN_ORION_AGENT_ID`, `AITOWN_ORION_PLAYER_ID`).

### Docker socket

GitHub MCP runs sibling containers via the host Docker daemon. Compose mounts `/var/run/docker.sock:/var/run/docker.sock` (same pattern as orion-hub).

### Enable and restart

```bash
# services/orion-harness-governor/.env
HARNESS_FCC_MCP_ENABLED=true
HARNESS_AITOWN_ENABLED=false   # optional

docker compose \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --build
```

Rebuild/restart after toggling MCP flags or changing `~/.fcc/.env` secrets.
