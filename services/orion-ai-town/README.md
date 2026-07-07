# orion-ai-town

Mesh deployment wrapper for [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town) with self-hosted Convex (no Convex cloud).

**Upstream pin:** `AITOWN_UPSTREAM_REF=main` (override at clone time).

## Services

| Service | Port (default) | Role |
|---------|----------------|------|
| `backend` | 3210 (API), 3211 (HTTP actions) | Self-hosted Convex |
| `dashboard` | 6791 | Convex dashboard |
| `frontend` | 5173 | AI Town game UI |

## Bootstrap (mesh node)

### 1. Clone upstream

```bash
cd services/orion-ai-town
git clone --depth 1 --branch main https://github.com/a16z-infra/ai-town.git upstream
# Or pin: git clone --depth 1 --branch <tag> ...
bash scripts/apply_upstream_patches.sh
```

### 2. Env + compose

```bash
cp .env_example .env
# Set INSTANCE_SECRET to a long random string
docker compose --env-file .env up -d --build
```

### 3. Admin key

```bash
docker compose exec backend ./generate_admin_key.sh
```

Add to operator `~/.fcc/.env`:

```bash
AITOWN_CONVEX_URL=http://<mesh-tailscale-ip>:3210
AITOWN_ADMIN_KEY="<admin-key-from-script>"
```

Also add to `upstream/.env.local` for Convex CLI deploy:

```bash
CONVEX_SELF_HOSTED_URL=http://127.0.0.1:3210
CONVEX_SELF_HOSTED_ADMIN_KEY="<admin-key>"
```

### 4. Deploy Convex functions (one-time)

From `upstream/`:

```bash
npm install
npm run predev
```

Dashboard: http://127.0.0.1:6791 (use admin key).

### 5. Initialize world

```bash
npx convex run init
```

Capture `AITOWN_WORLD_ID` from output → `~/.fcc/.env`.

## Mesh LLM wiring

AI Town Convex actions call OpenAI-compatible HTTP. Point them at **orion-llm-gateway** so chat uses the same route table as cortex/FCC (→ Atlas llamacpp workers).

```bash
# After gateway OpenAI passthrough is enabled (default):
bash services/orion-ai-town/scripts/wire_llm_gateway.sh
```

Defaults:

| Convex env | Value | Meaning |
|------------|-------|---------|
| `LLM_API_URL` | `http://<mesh-ip>:8210` | Gateway base (no `/v1` suffix) |
| `LLM_MODEL` | `chat` | Route key in `LLM_GATEWAY_ROUTE_TABLE_JSON` |
| `LLM_EMBEDDING_MODEL` | `orion-vector-host` | Label only; gateway proxies to vector-host |
| `EMBEDDING_DIMENSION` | `1024` | Must match `VECTOR_HOST_EMBEDDING_MODEL` (bge-large-en-v1.5) |

Override: `AITOWN_LLM_GATEWAY_URL`, `AITOWN_LLM_CHAT_ROUTE`, `AITOWN_EMBEDDING_DIMENSION`.

**Requires:** `LLM_GATEWAY_OPENAI_PASSTHROUGH_ENABLED=true` and `ORION_VECTOR_HOST_URL` on orion-llm-gateway.

If you change vector-host embedding model/dimension, update `EMBEDDING_DIMENSION` and redeploy Convex (`npx convex dev --once`). Wipe AI Town memory tables if dimension changes on an existing world.

Legacy direct Ollama (bypasses gateway):

```bash
npx convex env set OLLAMA_HOST http://<mesh-ollama-host>:11434
```

### 6. Wire LLM gateway (recommended)

Point AI Town at mesh Ollama or FCC gateway (from `upstream/`):

```bash
# Ollama on mesh
npx convex env set OLLAMA_HOST http://<mesh-ollama-host>:11434

# Or OpenAI-compatible FCC gateway
npx convex env set LLM_API_URL http://<fcc-gateway>/v1
npx convex env set LLM_API_KEY <key>
```

### 7. Production frontend (stable proxy)

For Hub reverse proxy, prefer production build over dev Vite:

```bash
cd upstream
npm run build
# Serve dist/ via nginx or the frontend container production mode per upstream docs
```

Hub uses `HUB_AITOWN_UI_URL=http://127.0.0.1:5173` when co-located.

## Smoke

```bash
docker compose config
curl -fsS http://127.0.0.1:3210/version
```

## Orion embodiment

Orion's persistent body uses a dedicated `"Orion"` character slot added to AI Town's `Descriptions` by `patches/orion-character.patch` (applied by `scripts/apply_upstream_patches.sh` alongside the embed patch). The body itself is created/updated — and its persona projected from Orion's live self-model — by `services/orion-embodiment/scripts/bootstrap_orion_agent.py` (dry-run by default; `--write` persists `AITOWN_ORION_*` to `~/.fcc/.env`).

> Note: `patches/orion-character.patch` is generated from a real diff against the cloned `upstream/`. On a node where `upstream/` is not yet cloned, the apply script skips the patch (with a message) rather than failing; generate the patch on a node that has `upstream/` before relying on the character slot.

## MCP integration

Gameplay MCP lives in `mcp/orion_aitown_mcp/`. Hub fcc-claude includes it when `HUB_AITOWN_ENABLED=true` and MCP is enabled.

Secrets (`AITOWN_*`) live in `~/.fcc/.env`, not in this service `.env`.
