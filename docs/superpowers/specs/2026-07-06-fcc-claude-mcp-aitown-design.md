# fcc-claude MCP + mesh AI Town — design

**Date:** 2026-07-06  
**Status:** Approved for implementation planning  
**Problem:** Hub agent-claude (`fcc_claude_bridge`) spawns `claude -p` with FCC gateway only — no MCP tools, no world contact beyond the repo workspace. Operators want GitHub + Firecrawl + mesh-hosted AI Town with gameplay control and in-Hub visualization.

---

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| MCP config ownership | Orion-managed: checked-in template, bridge renders ephemeral config per turn |
| MCP availability | Always on for every agent-claude turn (when feature enabled) |
| Secrets | `~/.fcc/.env` alongside existing FCC keys |
| GitHub MCP | Docker stdio → `ghcr.io/github/github-mcp-server` |
| Firecrawl MCP | npx stdio → `firecrawl-mcp` |
| AI Town deployment | Self-hosted on Orion mesh (no Convex cloud) via `services/orion-ai-town/` |
| AI Town MCP | Custom `orion-aitown-mcp` (Python stdio) — **not** generic `convex@latest mcp` |
| Gameplay control | Embodied defaults + god-mode escape hatch (`aitown_send_input`) |
| Hub visualization | Same-origin reverse proxy + lazy iframe tab (not cross-origin raw `:5173`) |

---

## Goals

- Every agent-claude turn can use GitHub (PRs/issues/checks), Firecrawl (web search/scrape), and AI Town gameplay tools.
- AI Town runs as a mesh service with self-hosted Convex backend — no Convex cloud dependency.
- Orion controls AI Town simulation via `sendInput` and engine mutations through a typed MCP tool surface.
- Hub exposes an **AI Town** tab: proxied game view + native status strip (engine, counts, health).
- Deterministic preflight: missing secrets or unreachable AI Town fail before spawn with actionable error codes.

## Non-goals (v1)

- Per-turn or per-server MCP toggles in Hub UI
- Harness-governor MCP parity (Hub agent-claude only)
- Bus events from gameplay actions (`orion:aitown:action`) — defer to v2
- iframe `postMessage` bidirectional coupling between Hub chat and game UI
- Forking AI Town for Orion-specific characters (pin upstream; customize in follow-up)
- Custom AI Town MCP tools beyond the v1 table below
- Deploying AI Town LLM via Convex cloud APIs (mesh OpenAI-compatible endpoint only)

---

## Architecture

```text
┌─ orion-hub (agent-claude) ─────────────────────────────────────────┐
│ websocket_handler → fcc_claude_bridge.run_turn                     │
│   → fcc_mcp_config.render()  reads ~/.fcc/.env                     │
│   → claude -p … --mcp-config /tmp/orion-fcc-mcp/<cid>.json         │
│                    --dangerously-skip-permissions                  │
│                    --allowedTools "mcp__*"                         │
└────────────────────────────┬───────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   github (docker)    firecrawl (npx)    orion-aitown (python)
         │                   │                   │
         │                   │                   ▼
         │                   │         AITOWN_CONVEX_URL (mesh)
         │                   │         Convex HTTP API
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
┌─ orion-ai-town (mesh node) ┴─────────────────────────────────────┐
│ convex-backend :3210  ·  dashboard :6791  ·  frontend :5173      │
│ LLM → mesh OpenAI-compatible (FCC gateway or orion-ollama-host)    │
└────────────────────────────────────────────────────────────────────┘

Hub UI tab:
  GET /api/aitown/status  →  health + engine stats
  iframe /aitown/         →  reverse proxy → HUB_AITOWN_UI_URL
```

**Choke points:**

| File / service | Role |
|----------------|------|
| `services/orion-hub/scripts/fcc_claude_bridge.py` | Spawn argv, subprocess env, `--mcp-config` |
| `services/orion-hub/scripts/fcc_mcp_config.py` | Template render, secret injection, preflight |
| `services/orion-hub/config/fcc_claude_mcp.template.json` | Orion-owned MCP server definitions (no secrets) |
| `services/orion-ai-town/` | Mesh deploy of upstream ai-town + self-hosted Convex |
| `services/orion-ai-town/mcp/` | `orion-aitown-mcp` gameplay MCP server |
| `services/orion-hub/scripts/api_routes.py` | `/api/aitown/status`, `/aitown/{path}` proxy |
| `services/orion-hub/templates/index.html` + `static/js/aitown-panel.js` | AI Town tab UI |

---

## MCP wiring (Approach 1 — ephemeral config)

### Template + render

Checked-in `fcc_claude_mcp.template.json` defines three `mcpServers` entries. Before each turn, `fcc_mcp_config.render(correlation_id, fcc_env) -> Path`:

1. Load secrets from `load_fcc_env(expand_env_path(HUB_FCC_ENV_PATH))`.
2. Validate required keys (see preflight table).
3. Write `/tmp/orion-fcc-mcp/<correlation_id>.json` with secrets inlined into each server's `env` block.
4. Return path for `--mcp-config`.

Temp files are deleted in `run_turn` `finally` block.

### Spawn argv changes

```text
claude -p <prompt> \
  --output-format stream-json \
  --verbose \
  --model <tier> \
  --mcp-config /tmp/orion-fcc-mcp/<cid>.json \
  --allowedTools "mcp__*" \
  [--dangerously-skip-permissions]
```

`--dangerously-skip-permissions` remains for non-root (existing behavior). `--mcp-config` uses dynamic scope — avoids project `.mcp.json` trust/approval gate that breaks headless `claude -p`.

### MCP server definitions

**github** (stdio via Docker):

```json
{
  "type": "stdio",
  "command": "docker",
  "args": [
    "run", "-i", "--rm",
    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
    "ghcr.io/github/github-mcp-server"
  ],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "<from GITHUB_PAT>"
  }
}
```

**firecrawl** (stdio via npx — requires Node 20 in Hub image):

```json
{
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "firecrawl-mcp"],
  "env": {
    "FIRECRAWL_API_KEY": "<from ~/.fcc/.env>"
  }
}
```

**orion-aitown** (stdio via Python module):

```json
{
  "type": "stdio",
  "command": "python3",
  "args": ["-m", "orion_aitown_mcp"],
  "env": {
    "AITOWN_CONVEX_URL": "<mesh URL>",
    "AITOWN_ADMIN_KEY": "<admin key>",
    "AITOWN_WORLD_ID": "<world id>",
    "AITOWN_ORION_AGENT_ID": "<optional default agent>",
    "AITOWN_ORION_PLAYER_ID": "<optional default player>"
  }
}
```

Hub `Dockerfile` adds Node 20 (apt or nodesource) for Firecrawl npx. `orion-aitown-mcp` ships as installable package under `services/orion-ai-town/mcp/` and is on `PYTHONPATH` in Hub container (or installed via editable dep).

---

## Secrets (`~/.fcc/.env`)

| Key | Required | Purpose |
|-----|----------|---------|
| `GITHUB_PAT` | yes (when MCP enabled) | GitHub MCP Docker auth |
| `FIRECRAWL_API_KEY` | yes (when MCP enabled) | Firecrawl MCP |
| `AITOWN_CONVEX_URL` | yes | Self-hosted Convex HTTP base (e.g. `http://100.x.x.x:3210`) |
| `AITOWN_ADMIN_KEY` | yes | Convex admin key for mutations |
| `AITOWN_WORLD_ID` | yes after `init` | Target world for `sendInput` |
| `AITOWN_ORION_AGENT_ID` | optional | Default agent for embodied tools |
| `AITOWN_ORION_PLAYER_ID` | optional | Default player for embodied tools |

Document keys in `services/orion-hub/README.md` and `services/orion-ai-town/README.md`. Do not commit secrets.

---

## `services/orion-ai-town/` — mesh deployment

Wrap upstream [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town) self-hosted Docker Compose:

| Service | Image / build | Port (default) |
|---------|---------------|----------------|
| `convex-backend` | `ghcr.io/get-convex/convex-backend:latest` | 3210 (API), 3211 (HTTP actions) |
| `dashboard` | `ghcr.io/get-convex/convex-dashboard:latest` | 6791 |
| `frontend` | Build from pinned ai-town tag | 5173 |

### Operator bootstrap (documented in README)

1. `docker compose up -d` on mesh node (Athena default).
2. Generate admin key: `docker compose exec backend ./generate_admin_key.sh` → `AITOWN_ADMIN_KEY`.
3. `npm run predev` / deploy convex functions to self-hosted backend.
4. `npx convex run init` → capture `AITOWN_WORLD_ID`.
5. Point AI Town LLM at mesh endpoint:
   - `npx convex env set OLLAMA_HOST http://<mesh-ollama-host>:11434`, or
   - `npx convex env set LLM_API_URL http://<fcc-gateway>/v1` + `LLM_API_KEY` / model vars per ai-town docs.
6. Build frontend for production: `npm run build` + serve static (not dev Vite) for stable proxy.

### Mesh networking

- Publish Convex URL on Tailscale/LAN IP reachable from Hub container (`network_mode: host` on Hub simplifies `127.0.0.1` when co-located).
- `AITOWN_CONVEX_URL` in `~/.fcc/.env` must match what `orion-aitown-mcp` and Hub status API use.

---

## `orion-aitown-mcp` — gameplay tool surface (v1)

Calls self-hosted Convex HTTP API using admin key. Primary mutation: `aiTown/main:sendInput` with `{ worldId, name, args }`.

| Tool | Behavior |
|------|----------|
| `aitown_world_status` | Engine running, generation number, world id |
| `aitown_list_players` | Players with positions, activities |
| `aitown_list_agents` | Agents + character descriptions |
| `aitown_list_characters` | Available `Descriptions` indices/names |
| `aitown_move_player` | `sendInput("moveTo", { playerId, destination })` — defaults `playerId` to `AITOWN_ORION_PLAYER_ID` |
| `aitown_create_agent` | `sendInput("createAgent", { descriptionIndex })` |
| `aitown_stop_world` | `testing:stop` mutation |
| `aitown_resume_world` | `testing:resume` mutation |
| `aitown_kick_engine` | `testing:kick` mutation |
| `aitown_send_input` | God-mode: arbitrary `sendInput(worldId, name, args)` |

Embodiment: tools with `playerId` / `agentId` args default to env-bound Orion IDs when omitted; `aitown_send_input` always available for operator override.

Input names align with upstream `convex/aiTown/inputs.ts` (`moveTo`, `createAgent`, `join`, `leave`, conversation inputs, etc.).

---

## Hub AI Town tab (visualization)

### Pattern

Follow Substrate Inspector / Substrate Atlas: dedicated nav tab, lazy-loaded iframe, native status header, standalone link.

### Routes

| Route | Handler |
|-------|---------|
| `GET /api/aitown/status` | Proxy health: Convex `/version`, engine running, player/agent counts (via MCP module or direct Convex query helper) |
| `GET /aitown/{path:path}` | Reverse HTTP proxy → `HUB_AITOWN_UI_URL` (default `http://127.0.0.1:5173`) |

### UI (`#ai-town` panel)

- Status strip: `● Running / ○ Offline`, agent count, player count, engine generation.
- Buttons: **Refresh**, **Open standalone** (`HUB_AITOWN_UI_URL` or proxied `/aitown/`).
- iframe: `src="/aitown/"`, `loading="lazy"`, set `src` only on first tab visit.
- Offline fallback: message in panel when status API fails (iframe not loaded).

### Settings

| Key | Default | Purpose |
|-----|---------|---------|
| `HUB_AITOWN_ENABLED` | `false` | Gate tab + proxy + include aitown in MCP template |
| `HUB_AITOWN_UI_URL` | `http://127.0.0.1:5173` | Upstream for reverse proxy |
| `HUB_AGENT_CLAUDE_MCP_ENABLED` | `false` | Master gate for all MCP wiring on fcc-claude turns |

When `HUB_AGENT_CLAUDE_MCP_ENABLED=false`, bridge spawns claude without `--mcp-config` (current behavior).

### iframe risk mitigation

| Risk | Mitigation |
|------|------------|
| Cross-origin framing | Same-origin proxy `/aitown/` |
| Vite dev HMR websocket | Production frontend build in mesh deploy |
| Proxy failure | Standalone link always visible |
| Resource cost | Lazy iframe load on tab first visit |

---

## Preflight and error codes

| Check | When | Error code |
|-------|------|------------|
| `HUB_AGENT_CLAUDE_MCP_ENABLED` | turn start | skip MCP (not an error) |
| `GITHUB_PAT` non-empty | render | `fcc_mcp_github_missing` |
| `FIRECRAWL_API_KEY` non-empty | render | `fcc_mcp_firecrawl_missing` |
| `AITOWN_CONVEX_URL` reachable (`/version`) | render | `fcc_mcp_aitown_unreachable` |
| `AITOWN_ADMIN_KEY` + `AITOWN_WORLD_ID` set | render | `fcc_mcp_aitown_config` |
| Convex auth valid (probe query) | render | `fcc_mcp_aitown_auth` |
| Docker available for github MCP | render (warn if missing) | `fcc_mcp_docker_missing` |
| npx/node available | render | `fcc_mcp_node_missing` |
| MCP config write | render | `fcc_mcp_render_failed` |

Hub `/api/aitown/status` returns `{ ok, convex_reachable, engine_running, player_count, agent_count, generation, error? }` without failing agent-claude turns.

---

## Testing

| Test file | Coverage |
|-----------|----------|
| `services/orion-hub/tests/test_fcc_mcp_config.py` | Template render, secret injection, missing-key errors, temp cleanup |
| `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py` | argv includes `--mcp-config` when enabled; omitted when disabled |
| `services/orion-ai-town/mcp/tests/test_aitown_tools.py` | Mock Convex HTTP: sendInput payloads, defaults for embodied IDs |
| `services/orion-hub/tests/test_aitown_status_api.py` | Status endpoint shape, offline handling |
| `services/orion-hub/tests/test_aitown_proxy.py` | Proxy route forwards path (mock upstream) |
| `services/orion-hub/tests/test_hub_aitown_tab.py` | Template contains `#ai-town` tab when enabled |

Gate command:

```bash
pytest services/orion-hub/tests/test_fcc_mcp_config.py \
       services/orion-hub/tests/test_fcc_claude_bridge_mcp.py \
       services/orion-hub/tests/test_aitown_status_api.py \
       services/orion-hub/tests/test_aitown_proxy.py \
       services/orion-hub/tests/test_hub_aitown_tab.py -q
pytest services/orion-ai-town/mcp/tests/ -q
```

Smoke (manual / eval follow-up):

1. `HUB_AGENT_CLAUDE_MCP_ENABLED=true` + secrets in `~/.fcc/.env`.
2. Agent-claude turn: verify harness steps show `mcp__github__*` or `mcp__orion-aitown__*` tool calls.
3. Open Hub **AI Town** tab: game renders; status strip shows running engine.
4. Prompt: "move Orion to coordinates 10,15" → character moves in iframe view.

---

## Files likely to touch

### Create

- `docs/superpowers/specs/2026-07-06-fcc-claude-mcp-aitown-design.md` (this file)
- `services/orion-hub/config/fcc_claude_mcp.template.json`
- `services/orion-hub/scripts/fcc_mcp_config.py`
- `services/orion-hub/static/js/aitown-panel.js`
- `services/orion-ai-town/docker-compose.yml`
- `services/orion-ai-town/.env_example`
- `services/orion-ai-town/README.md`
- `services/orion-ai-town/mcp/` (package + tools)
- Hub + ai-town tests listed above

### Modify

- `services/orion-hub/scripts/fcc_claude_bridge.py`
- `services/orion-hub/app/settings.py` + `.env_example`
- `services/orion-hub/Dockerfile` (Node 20)
- `services/orion-hub/docker-compose.yml` (if env passthrough needed)
- `services/orion-hub/templates/index.html`
- `services/orion-hub/static/js/app.js` (tab routing)
- `services/orion-hub/scripts/api_routes.py`
- `services/orion-hub/README.md`

---

## Implementation phases (recommended PR order)

1. **MCP core** — `fcc_mcp_config`, template, bridge argv, GitHub + Firecrawl only, tests, Hub Dockerfile Node.
2. **orion-ai-town deploy** — compose, README bootstrap, mesh LLM wiring, production frontend.
3. **orion-aitown-mcp** — gameplay tools, embodied defaults, integration tests.
4. **Hub UI** — tab, proxy, status API, lazy iframe.

Phases 1–3 can ship MCP gameplay before UI tab; phase 4 completes visualization.

---

## Acceptance checks

- [ ] Agent-claude turn with MCP enabled spawns claude with `--mcp-config` and three servers in rendered JSON.
- [ ] Missing `GITHUB_PAT` fails preflight with `fcc_mcp_github_missing` before subprocess spawn.
- [ ] GitHub MCP tool call succeeds against a test repo (manual smoke).
- [ ] Firecrawl MCP returns scrape/search result (manual smoke).
- [ ] AI Town self-hosted Convex healthy on mesh without Convex cloud account.
- [ ] `aitown_move_player` mutation visible in game within one engine tick (manual smoke).
- [ ] Hub AI Town tab loads proxied iframe; status strip reflects engine state.
- [ ] `HUB_AGENT_CLAUDE_MCP_ENABLED=false` preserves pre-patch spawn behavior (no regression).

---

## Risks

| Severity | Concern | Mitigation |
|----------|---------|------------|
| Medium | Hub image size + Node dependency | Slim Node 20 install; only for npx MCP children |
| Medium | Convex self-hosted ops burden | Document bootstrap; pin images; health in status API |
| Low | Docker-in-Docker for GitHub MCP | Hub already mounts `docker.sock`; document requirement |
| Low | iframe proxy + WS edge cases | Production static frontend; standalone fallback |
| Low | AI Town upstream drift | Pin git tag in orion-ai-town build |

---

## Restart required (after implementation)

```bash
# AI Town (mesh node)
docker compose --env-file services/orion-ai-town/.env \
  -f services/orion-ai-town/docker-compose.yml up -d --build

# Hub
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Sync `~/.fcc/.env` with new keys after updating operator docs.
