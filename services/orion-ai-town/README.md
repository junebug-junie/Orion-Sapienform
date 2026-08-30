# orion-ai-town

Mesh deployment wrapper for [a16z-infra/ai-town](https://github.com/a16z-infra/ai-town) with self-hosted Convex (no Convex cloud).

**Upstream pin:** `AITOWN_UPSTREAM_REF=7b242334bfbfef02f7718bded120d431e8f307df` — the a16z SHA the tracked `patches/` were generated against. The patches carry exact context and will not apply to a moved `main`; re-pin (and regenerate patches) intentionally when bumping upstream.

**Deployment host: Circe (`100.112.254.99`, tailscale) as of 2026-08-29.** Relocated from deprecated Atlas — see `docs/superpowers/specs/2026-07-29-aitown-atlas-migration-runbook.md` for the prior Athena→Atlas history and **Atlas → Circe evacuation** below for the 2026-08-29 cutover.

```bash
mkdir -p ${TELEMETRY_ROOT:-/mnt/telemetry}/orion-circe/ai-town/convex-data
```

The daily compaction cron (`scripts/compact_convex_data.sh`, see "Maintenance" below) must run on whichever host actually holds these containers — move the crontab entry along with the deployment, don't leave it running against a stopped remote host.

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
| `LLM_MODEL` | `quick_background` | Route key in `LLM_GATEWAY_ROUTE_TABLE_JSON` |
| `LLM_EMBEDDING_MODEL` | `orion-vector-host` | Label only; gateway proxies to vector-host |
| `EMBEDDING_DIMENSION` | `1024` | Must match `VECTOR_HOST_EMBEDDING_MODEL` (bge-large-en-v1.5) |

Override: `AITOWN_LLM_GATEWAY_URL`, `AITOWN_LLM_CHAT_ROUTE`, `AITOWN_EMBEDDING_DIMENSION`.

> **`LLM_MODEL` must never be `chat` (circe-worker-1).** That lane is
> reserved for Juniper's direct deep/FCC turns (see
> `services/orion-llm-gateway/README.md`'s mesh-LLM-wiring note). Post-Atlas
> decommission, AI Town's NPC dialogue uses `quick_background` on
> **circe-worker-fast-1** — same physical host, different lane, background
> priority so town dialogue never blocks chat. A stale `LLM_MODEL=chat` caused
> a silent, town-wide 10+ hour dialogue outage on 2026-07-30.
> `wire_llm_gateway.sh` defaults to `quick_background`; run
> `python3 services/orion-ai-town/scripts/check_llm_route_not_circe.py` any
> time to verify the live value is not on the chat lane.
>
> **Why `quick_background`, not plain `quick` (2026-07-30):** `quick`
> (`circe-worker-fast-1` post-Atlas; was `atlas-worker-fast-1`) is also the
> default route for `orion-mind`, `orion-embodiment`'s hub-mode speech, and
> `orion-hub`'s memory-graph-suggest. `quick_background` shares the exact same
> upstream but waits for `/slots` slack before dispatching, so AI Town's
> dialogue never makes those other, snappier consumers wait behind it -- see
> `services/orion-llm-gateway/README.md`'s "Background-priority routes".

## Atlas → Circe evacuation (2026-08-29)

When moving off deprecated Atlas:

1. **Copy data** — rsync `${TELEMETRY_ROOT}/orion-atlas/ai-town/convex-data/` → `${TELEMETRY_ROOT}/orion-circe/ai-town/convex-data/` on Circe (`/mnt/telemetry`, not `/mnt/scripts`).
2. **Update compose bind mount** — `orion-circe/ai-town/convex-data` (this branch).
3. **Compact** — `bash scripts/compact_convex_data.sh --force` once the backend is healthy (revision bloat is normal after downtime).
4. **Hub cutover** — set `HUB_AITOWN_UI_URL=http://<circe-tailscale>:5173` and `HUB_AITOWN_CONVEX_URL=http://<circe-tailscale>:3210` in `services/orion-hub/.env`; recreate `hub-app` (`docker compose up -d`, not `restart`).
5. **Embodiment** — `AITOWN_CONVEX_URL` stays in `~/.fcc/.env` only (Orion's body), not Hub.

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

## Maintenance: Convex data compaction

The self-hosted Convex backend retains a full document-revision history
forever (every mutation writes a new version instead of overwriting), with
no built-in compaction. A continuously-ticking town accumulates this
without bound: confirmed live 2026-07-29, `db.sqlite3` reached 23.5GB after
~3 weeks even though logical/current data across every table added up to
only ~240MB. `VACUUM` cannot reclaim this — it only recovered ~5% (23.56GB →
22.24GB after a 14-minute run) because none of the retained revisions are
logically deleted from Convex's point of view.

`scripts/compact_convex_data.sh` fixes this by exporting current live data,
resetting the on-disk file, and reimporting — which starts a fresh history
from that point while preserving the game world exactly as it was (verified
live: 23.5GB → 240MB, all 216k+ documents round-tripped intact). It also
redeploys Convex functions and restores `npx convex env` variables, both of
which live in the same file and get wiped by the reset alongside the data,
and heartbeats the default world back to `running` so nobody has to reload
the frontend tab afterward.

```bash
# Report current db.sqlite3 size only, no changes:
bash scripts/compact_convex_data.sh --check

# Compact only if over the threshold (default 5GiB):
bash scripts/compact_convex_data.sh

# Compact regardless of current size:
bash scripts/compact_convex_data.sh --force
```

Env overrides: `AITOWN_COMPACT_THRESHOLD_BYTES` (default `5368709120` = 5GiB),
`AITOWN_COMPACT_HEALTH_TIMEOUT_SEC` (default `180`).

Each run writes a job dir under `/tmp/aitown-compact-<timestamp>/` containing
the pre-compact export, a raw `db.sqlite3` backup, and a `report.md` with
before/after sizes — keep these until you've confirmed the town looks right.

A host crontab entry runs this daily (threshold-gated, so it's a no-op most
days — see `crontab -l` for the exact line, installed 2026-07-29). There is
brief real downtime for AI Town while a compaction actually runs (stop →
reset → restart → reimport), typically well under a minute once the backend
is healthy again, though function redeploy/reindexing can add several
minutes on a large table set.

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

### Conversation proximity (`patches/orion-conversation-proximity.patch`)

Stock AI Town picks the **nearest free player on the entire map** as a conversation invitee, so NPCs lock each other (and humans) into cross-map chats. This patch:

- Adds `MAX_INVITE_DISTANCE = 6` (aligned with Orion embodiment `EMBODIMENT_SOCIAL_INITIATE_DISTANCE`)
- Filters `findConversationCandidate` to players within that range
- Rejects `Conversation.start` when initiator and invitee are too far apart
- Nudges the initiator toward the invitee immediately on invite (stock engine only walked on the next agent tick)
- **`INVITE_TIMEOUT` applies only to pending `invited` status** — once a player accepts (`walkingOver`), agents keep walking until they're in range; accepted invites no longer expire mid-walk

### Town chat turns (`patches/orion-town-chat-turns.patch`)

Fixes NPC-human chats where agents talk over the human, narrate scene prose instead of replying, or auto-leave mid-conversation:

- NPCs **wait for the human's first line** (no synthetic `start` message over Juniper)
- Replies are **in-character dialogue** with `clampTownReply` (no observation-lounge narration)
- NPCs **do not auto-leave** human conversations (message cap / duration only applies NPC-NPC)
- **3 minute grace** after an NPC speaks before it considers speaking again (`HUMAN_REPLY_GRACE_MS`)

Orion's external embodiment worker already walks on `walkingOver` via `approach_player` intents in `services/orion-embodiment/app/worker.py`.

### NPC cooldown tuning (`patches/orion-npc-cooldown-tuning.patch`)

Load-shedding, not a gameplay change. Every NPC conversation message is an `agentGenerateMessage` call routed through `orion-llm-gateway`; when that gateway's host is under load, throttling how often those calls happen is the direct lever. Raised from `convex/constants.ts` upstream defaults (2026-07-29):

- `CONVERSATION_COOLDOWN`: `15000` → `45000` (agents wait 3x longer after ending a conversation before starting another)
- `MESSAGE_COOLDOWN`: `2000` → `8000` (4x longer between messages within an active conversation — the main per-conversation LLM-call-rate lever)

Revert both to their upstream defaults if the gateway's host load profile changes and the throttling is no longer needed.

### Input-counter contention fix (`patches/orion-input-counter-contention.patch`)

Not a throttle -- a real contention fix. Investigation (2026-07-29/30) found the cooldown tuning above didn't touch the actual cause of laggy player movement: every `sendInput` call (movement, NPC actions, embodiment) allocated its `inputs.number` by reading the last input for the engine and incrementing, and separately read `worldStatus` to resolve `worldId` -> `engineId`. Both reads created wide Convex OCC conflict surfaces:

- The read-last-then-increment pattern made every `sendInput` call conflict with every other concurrent `sendInput` call, and with `saveWorld` patching `returnValue` onto recently-inserted input rows in the same index range.
- The `worldStatus` read collided with `heartbeatWorld`, which patches `lastViewed` on that same document on nearly every call.

Neither is a storage-engine issue (an earlier Postgres-backend experiment for this same symptom found no improvement, consistent with the conflicts being enforced by Convex's own OCC layer, not SQLite locking).

Fix: two new narrow tables that nothing else reads or writes:

- `inputCounters` (one row per engine) -- `engineInsertInput` allocates `number` from this instead of scanning `inputs`.
- `worldEngineMap` (one row per world) -- `insertInput` resolves `engineId` from this instead of reading `worldStatus`.

Both are lazily backfilled on first use per engine/world (falls back to the legacy read once, then seeds the new table), so this needed no separate migration step against the live world's existing data.

## MCP integration

Gameplay MCP lives in `mcp/orion_aitown_mcp/`. Hub fcc-claude includes it when `HUB_AITOWN_ENABLED=true` and MCP is enabled.

Secrets (`AITOWN_*`) live in `~/.fcc/.env`, not in this service `.env`.
