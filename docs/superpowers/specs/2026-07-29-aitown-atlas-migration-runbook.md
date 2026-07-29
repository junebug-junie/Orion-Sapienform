# AI Town → Atlas migration runbook

**Date:** 2026-07-29
**Status:** Runbook — for a Claude Code (or manual) session running directly on `atlas`
**Why**: live investigation this session found AI Town's 15-25s mutation latency is not caused by AI Town's own architecture, SQLite vs Postgres (isolated A/B test showed no meaningful difference — see `docs/superpowers/specs/2026-07-29-aitown-orion-conversational-quality-design.md`), or the embodiment worker's polling. It's host-wide CPU contention on `athena` — dominated by `orion-heartbeat` (a legitimate, deliberate tensor-network substrate, sustained 600-1600% CPU) and bursty spikes from `orion-athena-vector-host`. Rather than constrain that research substrate, the decision is to relocate AI Town to `atlas` — a separate mesh node (`100.121.214.30`, tailscale) already dedicated to GPU llamacpp inference workers, with far less competing CPU load from `athena`'s always-on cognition services.

This doc has two audiences:
- **Atlas-side session**: run the "On Atlas" steps below directly on that host.
- **Athena-side session** (this repo's usual host): run the "On athena (cutover)" steps once Atlas is confirmed healthy.

Do not skip the verification step at the end of each phase before moving to the next.

---

## Non-goals

- Not changing AI Town's game architecture, engine code, or the identity/social-memory work tracked separately in the conversational-quality design doc.
- Not touching `orion-heartbeat`/`orion-athena-vector-host` — this migration is the chosen mitigation instead of constraining them.
- Not migrating any *other* athena-hosted service to Atlas — this is AI Town only.

---

## What moves, what doesn't

**Moves to Atlas**: `services/orion-ai-town`'s Convex backend + dashboard + frontend (the actual game world/engine), and the compaction cron job for it.

**Stays on athena**: `services/orion-embodiment` (drives Orion's body — no reason to move it; it already talks to AI Town over the tailscale mesh, just needs its `AITOWN_CONVEX_URL` repointed), `services/orion-harness-governor` (currently `HARNESS_AITOWN_ENABLED=false`, not live, but its env should still be corrected), `services/orion-hub` (needs `HUB_AITOWN_UI_URL` repointed to proxy Atlas's frontend instead of localhost).

---

## Phase 1 — On Atlas: bootstrap a fresh deployment

### 1.0 Confirm the host is actually ready

```bash
nproc
free -h
df -h /
docker --version
docker compose version
tailscale ip -4   # should print 100.121.214.30
```

If Docker isn't installed or the `app-net` network doesn't exist yet (it's already used by the llamacpp workers on this host, so it likely does):

```bash
docker network create app-net >/dev/null 2>&1 || true
```

### 1.1 Get the repo and clone upstream ai-town

If this session doesn't already have `Orion-Sapienform` cloned on Atlas, clone it (same repo, same remote as athena). Then, per `services/orion-ai-town/README.md`:

```bash
cd services/orion-ai-town
git clone https://github.com/a16z-infra/ai-town.git upstream
git -C upstream checkout 7b242334bfbfef02f7718bded120d431e8f307df  # AITOWN_UPSTREAM_REF, confirm current value in .env_example
bash scripts/apply_upstream_patches.sh
```

### 1.2 Configure env for Atlas

```bash
cp .env_example .env
```

Edit `.env`:

```dotenv
AITOWN_UPSTREAM_REF=7b242334bfbfef02f7718bded120d431e8f307df   # match athena's current pin
PORT=3210
SITE_PROXY_PORT=3211
DASHBOARD_PORT=6791
FRONTEND_PORT=5173
URL_BASE=http://100.121.214.30
INSTANCE_SECRET=<generate a new long random string -- do NOT reuse athena's>
INSTANCE_NAME=orion-aitown
OLLAMA_PORT=11434
DATABASE_URL=
```

`INSTANCE_SECRET` must be a **fresh** secret, not copied from athena's `.env` — this is a distinct deployment, not a clone with the same identity.

### 1.3 Bring it up

```bash
docker compose --env-file .env up -d --build
docker compose exec backend ./generate_admin_key.sh
```

Save the printed admin key — you'll need it for both the deploy step below and the data pull in Phase 2.

### 1.4 Deploy Convex functions

```bash
cd upstream
cat > .env.local <<EOF
CONVEX_SELF_HOSTED_URL=http://127.0.0.1:3210
CONVEX_SELF_HOSTED_ADMIN_KEY="<admin-key-from-1.3>"
VITE_CONVEX_URL=http://127.0.0.1:3210
EOF
npm install
npx convex dev --once
```

### 1.5 Verify Atlas backend is healthy before touching any data

```bash
curl -fsS http://127.0.0.1:3210/version
docker compose ps
```

All three services (`backend`, `dashboard`, `frontend`) should show healthy/running before Phase 2.

---

## Phase 2 — Migrate the live world from athena to Atlas

This pulls data **directly from athena's live, currently-running deployment** over the tailscale mesh — no manual file transfer needed, since both hosts are on the same network.

### 2.1 From Atlas, export athena's current live data

You'll need athena's current admin key. It's in `services/orion-ai-town/upstream/.env.local` on athena — ask the athena-side session/operator for it, or read it there if you have access. **Do not** commit or paste this key into any tracked file.

```bash
cd services/orion-ai-town/upstream
mkdir -p /tmp/aitown-atlas-migration
CONVEX_SELF_HOSTED_URL=http://100.92.216.81:3210 \
CONVEX_SELF_HOSTED_ADMIN_KEY="<athena's admin key>" \
npx convex export --path /tmp/aitown-atlas-migration/export.zip
```

Sanity check the export isn't suspiciously small before proceeding (should be roughly the size reported by `du -sh` on athena's `db.sqlite3` at the time of the last compaction, or larger if it's grown since — currently around 240MB-a-few-hundred-MB as of the last compaction on 2026-07-29):

```bash
ls -lh /tmp/aitown-atlas-migration/export.zip
```

### 2.2 Import into Atlas's fresh backend

Switch back to targeting Atlas's *own* local backend for this step:

```bash
CONVEX_SELF_HOSTED_URL=http://127.0.0.1:3210 \
CONVEX_SELF_HOSTED_ADMIN_KEY="<Atlas's admin key from 1.3>" \
npx convex import --replace-all -y /tmp/aitown-atlas-migration/export.zip
```

Watch the import change-summary output — every table's `create` count should be non-zero and match athena's live row counts (players, agents, conversations, messages, memories, etc.). If any table shows unexpectedly `0`, stop and investigate before proceeding — do not delete athena's original data yet regardless of how this looks.

### 2.3 Bring the migrated world back to life on Atlas

The imported `worldStatus` will likely show `status: "inactive"` (nothing has viewed it yet on this fresh backend). Resolve the world ID and heartbeat it:

```bash
npx convex run world:defaultWorldStatus
# note the worldId from the output, then:
npx convex run world:heartbeatWorld '{"worldId": "<worldId>"}'
```

### 2.4 Verify

```bash
npx convex data worldStatus         # status should now be "running"
npx convex data playerDescriptions --limit 5   # spot-check real player data present
```

Also load `http://100.121.214.30:5173` in a browser to confirm the frontend renders the migrated world with the real cast (Mara, Nico, Orion, etc. — not a fresh/empty world).

---

## Phase 3 — Wire the LLM gateway on Atlas

Same as athena's setup (`services/orion-ai-town/README.md`, "Mesh LLM wiring" section) — NPC dialogue on Atlas should still route through `orion-llm-gateway` (wherever that runs; confirm whether it's athena-hosted or already available on Atlas):

```bash
bash scripts/wire_llm_gateway.sh
```

Confirm the resulting `LLM_API_URL` Convex env var actually resolves from *inside* the Atlas backend container (it's Dockerized, so `127.0.0.1`/`localhost` references need mesh-IP rewriting exactly as `wire_llm_gateway.sh` already handles — verify it produced a real mesh IP, not a loopback address, in its printed output).

---

## Phase 4 — On athena: cutover

Once Atlas's migrated world is confirmed healthy (Phase 2.4) and NPC chat is confirmed working (Phase 3), update every athena-side consumer to point at Atlas instead of athena's own (soon-to-be-retired) AI Town instance.

### 4.1 `~/.fcc/.env` (shared, mounted into `orion-embodiment`)

Update the three keys added earlier this session:

```dotenv
AITOWN_CONVEX_URL=http://100.121.214.30:3210
AITOWN_ADMIN_KEY=<Atlas's admin key from 1.3>
AITOWN_WORLD_ID=<worldId from Phase 2.3>
```

`AITOWN_ORION_PLAYER_ID` should **not** need to change — the import in Phase 2.2 preserves Orion's existing player ID (`p:24` as of this writing) since `--replace-all` restores documents with their original IDs. Confirm this directly rather than assuming:

```bash
# from athena, after updating ~/.fcc/.env:
docker compose -f services/orion-embodiment/docker-compose.yml restart embodiment
docker compose -f services/orion-embodiment/docker-compose.yml logs -f embodiment
# watch for embodiment_heartbeat player=p:24 nearby=N ... (not perception=none, not an invalid-player-id error)
```

If `AITOWN_ORION_PLAYER_ID` turns out stale, rerun `services/orion-embodiment/scripts/bootstrap_orion_agent.py` — **beware the dry-run footgun found this session**: running it without `--write` still performs a real `join_player` call against the live world (it only skips *persisting* the result, not the join itself). Check the player list before and after to catch an accidental duplicate join, same as this session's cleanup.

### 4.2 `services/orion-harness-governor/.env`

Not currently live (`HARNESS_AITOWN_ENABLED=false`), but correct it for whenever it's turned on:

```dotenv
HARNESS_AITOWN_CONVEX_URL=http://100.121.214.30:3210
```

### 4.3 `services/orion-hub/.env`

```dotenv
HUB_AITOWN_UI_URL=http://100.121.214.30:5173
```

Restart Hub to pick this up.

### 4.4 Compaction cron

The daily compaction cron entry (`crontab -l` on athena, installed 2026-07-29) references `services/orion-ai-town` **on athena** — `docker compose`/`scripts/compact_convex_data.sh` must run on whichever host actually holds the containers. Remove the athena crontab entry and install the equivalent one on Atlas instead (same script, same repo checkout, now pointed at Atlas's own local backend):

```bash
# on athena: remove the old entry
crontab -e   # delete the "AI Town Convex..." block

# on atlas: add the equivalent entry (adjust path to wherever the repo lives on atlas)
(crontab -l 2>/dev/null; echo "0 13 * * * cd <atlas-repo-path>/services/orion-ai-town && bash scripts/compact_convex_data.sh >> <atlas-repo-path>/logs/orion-aitown-compact.log 2>&1") | crontab -
```

### 4.5 Retire athena's old AI Town containers — don't delete data yet

```bash
cd services/orion-ai-town
docker compose stop
```

**Stop, don't remove**, and leave the SQLite volume in place for at least a few days as a rollback path in case something on Atlas needs debugging. Only run `docker compose down -v` (or delete the volume) once Atlas has been running cleanly for a while and you're confident the migration is durable.

### 4.6 Update repo docs

`services/orion-ai-town/README.md`'s bootstrap instructions currently assume `127.0.0.1`/local deployment. Add a note (or a dedicated section) that the live deployment now runs on `atlas` (`100.121.214.30`), with a pointer to this runbook, so a future session doesn't rediscover this from scratch.

---

## Acceptance checks

1. Atlas's world loads in a browser at `http://100.121.214.30:5173` with the real, migrated cast and history intact (not a fresh/empty world).
2. `docker stats` on Atlas shows the backend container's CPU/latency profile matching the isolated test's healthy baseline (~1.2-2.0s per mutation), not athena's pathological 15-25s.
3. Orion (`p:24`) perceives and can move/speak in the migrated world (embodiment logs show real `nearby=N` heartbeats, no `perception=none`, no invalid-player-id errors).
4. NPC-to-NPC chat still works (LLM gateway wiring confirmed reachable from Atlas's container).
5. `~/.fcc/.env`, `orion-harness-governor/.env`, `orion-hub/.env` all point at Atlas; athena's old containers are stopped (not necessarily deleted).
6. Compaction cron runs on Atlas, not athena.

## Rollback

Athena's stopped (not deleted) containers + SQLite volume are the rollback path: `docker compose start` on athena, revert the four env files in 4.1-4.3 back to athena's IP, restart embodiment/harness-governor/hub. No data was destroyed by this migration (export/import is additive; athena's original volume is untouched).
