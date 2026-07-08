# orion-ai-town

Mesh deployment wrapper for [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town) with self-hosted Convex (no Convex cloud).

**Upstream pin:** `AITOWN_UPSTREAM_REF=7b242334bfbfef02f7718bded120d431e8f307df` — the a16z SHA the tracked `patches/` were generated against. The patches carry exact context and will not apply to a moved `main`; re-pin (and regenerate patches) intentionally when bumping upstream.

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
git clone https://github.com/a16z-infra/ai-town.git upstream
git -C upstream checkout 7b242334bfbfef02f7718bded120d431e8f307df  # AITOWN_UPSTREAM_REF
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

## Cast cards (source of truth)

The full character set — the 8 NPCs plus **Juniper Feld** (human) and **Orion** (external join) — lives as authored cards in `cards/town_cards.yaml`. This is the single source of truth for identities.

Regenerate the AI Town artifacts from the cards with the deterministic generator (run from repo root):

```bash
python services/orion-ai-town/scripts/generate_descriptions.py
```

It rewrites, in `upstream/`: the `Descriptions` array in `data/characters.ts` (rich, prompt-injected NPC identities read by `convex/agent/conversation.ts`), `DEFAULT_NAME` in `convex/constants.ts`, and Juniper's join description in `convex/world.ts`. It also emits `cards/generated/orion_town_card.txt` (Orion's full blurb, consumed by the embodiment bootstrap) and `cards/generated/juniper_description.txt` (reference copy). After regenerating, refresh the tracked patches:

```bash
git -C upstream diff -- data/characters.ts > patches/orion-character.patch
git -C upstream diff -- convex/constants.ts convex/world.ts > patches/orion-human-juniper.patch
```

## Orion embodiment

`patches/orion-character.patch` seeds the fresh 8-NPC town cast in AI Town's `Descriptions`: Mara Vale, Nico Sable, Dr. Elian Cross, Juno Park, Tessa Quinn, Vale Moreno, Sofia Bell, and Cam Lin (applied by `scripts/apply_upstream_patches.sh` alongside the embed patch). Orion is **not** in `Descriptions`; Orion joins externally — its body created/updated by `services/orion-embodiment/scripts/bootstrap_orion_agent.py` (dry-run by default; `--write` persists `AITOWN_ORION_*` to `~/.fcc/.env`). Orion joins with its **authored town card** (`cards/generated/orion_town_card.txt`, from `town_cards.yaml`); if that file is unreachable the bootstrap falls back to the live self-model projection, then a minimal safe blurb. Juniper Feld is the **human player**, wired via `patches/orion-human-juniper.patch` (sets `DEFAULT_NAME = 'Juniper Feld'` and her rich join description in `convex/world.ts`).

> Note: `patches/orion-character.patch` and `patches/orion-human-juniper.patch` are generated from real diffs against the cloned `upstream/`. On a node where `upstream/` is not yet cloned, the apply script skips a patch (with a message) rather than failing; generate the patches on a node that has `upstream/` before relying on the cast.

### Fresh game / reset

Reseed the town from scratch (destructive — wipes all world/memory tables). Operator-run:

```bash
cd services/orion-ai-town && bash scripts/apply_upstream_patches.sh
cd upstream && npx convex dev --once            # redeploy Convex functions
npx convex run testing:stop
npx convex run testing:wipeAllTables            # internalMutation; wipes all world/memory tables
npx convex run init                             # seeds the 8 NPCs from Descriptions
npx convex run testing:resume
# re-bootstrap Orion's external body:
cd ../../.. && python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write
```

### Engine recovery (`patches/orion-engine-recovery.patch`)

Adds two internal Convex functions to `convex/testing.ts`:

- `testing:debugEngineState` — dumps `processedInputNumber`, the pending (unactioned) input backlog by name, and each conversation's `lastMessage`. Read-only diagnostic.
- `testing:recoverFrozenEngine` — drops the unactioned input backlog and scrubs any malformed `lastMessage` from the stored world.

These recover a **frozen engine**: an externally-driven player (Orion) can enqueue a `finishSendingMessage` without a numeric `timestamp`, which builds `lastMessage={author}` and fails the `serializedConversation` validator in `saveWorld`. That crashes every `runStep`, so `processedInputNumber` never advances and the whole town freezes until the poisoned input is purged. Recover with:

```bash
cd upstream
npx convex run testing:debugEngineState      # inspect the backlog
npx convex run testing:recoverFrozenEngine    # purge stale inputs + scrub
npx convex run testing:stop && npx convex run testing:resume
```

The embodiment worker no longer sends `finishSendingMessage` itself (`messages:writeMessage` enqueues a well-formed one), so this poison cannot recur; the patch is kept for operator recovery and diagnosis.

## MCP integration

Gameplay MCP lives in `mcp/orion_aitown_mcp/`. Hub fcc-claude includes it when `HUB_AITOWN_ENABLED=true` and MCP is enabled.

Secrets (`AITOWN_*`) live in `~/.fcc/.env`, not in this service `.env`.
