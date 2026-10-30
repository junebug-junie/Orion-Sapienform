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

All three services use `restart: unless-stopped` so a Circe host reboot (or Docker daemon restart) brings town back without a manual `compose up`. A deliberate `docker compose stop` still stays stopped.

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

Env overrides — these are read from the **environment**, not from
`services/orion-ai-town/.env`; this is a host script and never sources that file,
so export them in the crontab entry or the invoking shell:

| var | default | purpose |
| --- | --- | --- |
| `AITOWN_COMPACT_THRESHOLD_BYTES` | `5368709120` (5GiB) | skip below this size |
| `AITOWN_COMPACT_HEALTH_TIMEOUT_SEC` | `180` | step 5 health wait |
| `AITOWN_COMPACT_JOB_DIR_BASE` | `/tmp` | where the job dir (and step 3's full db copy) lands |
| `AITOWN_COMPACT_SKIP_RAW_BACKUP` | `0` | skip step 3's raw copy and relax the preflight |
| `AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE` | `0` | proceed when `$HOME` free space is unreadable |

The step 0 preflight sums the step 3 backup and the step 5b npm-cache
requirements **when the job dir and `$HOME` are on the same filesystem** — which
is the common case (`/tmp`, `$HOME` and `/` are one device on circe). Checking
them independently is not enough: each can pass while their sum does not fit.

If the database has grown past roughly half its filesystem, a full-copy-sized
gate would refuse every run — including the one that shrinks it. Set
`AITOWN_COMPACT_SKIP_RAW_BACKUP=1` in that case: recovery then rests on
`export.zip` plus step 4's in-volume rename, which is two artifacts, not zero.

Each run writes a job dir under `<AITOWN_COMPACT_JOB_DIR_BASE>/aitown-compact-<timestamp>/`
containing the pre-compact export, a raw `db.sqlite3` backup, and a `report.md`
with before/after sizes — keep these until you've confirmed the town looks right.

### If a compaction fails: DO NOT re-run it

Step 4 renames the live database aside and step 5 starts the backend on a fresh
empty one. Past that point a second run of `compact_convex_data.sh` would export
that **empty** database over the good `export.zip` and reimport it — turning a
recoverable outage into permanent data loss. Use the resume script instead:

```bash
bash scripts/resume_compact_convex_data.sh /tmp/aitown-compact-<timestamp>
```

It replays steps 5b–7 (redeploy functions → restore env vars → reimport →
heartbeat the world) against the original job dir. `compact_convex_data.sh`
prints this instruction itself on **any** failure past step 4 — including
Ctrl-C, a dropped SSH session, and a cron timeout — so you should not have to
remember it.

The resume script refuses to run, and names the override, when:

- `export.zip` holds fewer than 1000 documents — it looks like an export *of* an
  already-emptied database, and importing it would make the loss permanent
- `export.zip` is more than 6h old — a stale export rolls the world back that
  far just as destructively (`AITOWN_RESUME_CONFIRM=1`)
- the deployment already answers `world:defaultWorldStatus` — functions and data
  are present, so this is not the broken state it repairs, and running anyway
  would `--replace-all` a healthy town (`AITOWN_RESUME_CONFIRM=1`)
- free space under `$HOME` or the job dir is below 15GiB
  (`AITOWN_RESUME_MIN_FREE_BYTES`)

**A "healthy" backend is not evidence the town is up.** The Convex healthcheck
proves the process answers HTTP, not that it has functions or data. Verify with
`world:defaultWorldStatus` (it only answers when both are back) and confirm the
engine is actually ticking — it rewrites `world`/`engines` every tick, so the
file grows:

```bash
watch -n5 'docker compose exec -T backend stat -c%s /convex/data/db.sqlite3'
```

**Incident, 2026-08-31.** A run wrote its 11GB `db.sqlite3` backup into `/tmp` on
a filesystem without 11GB free, filled the root disk to 100%, and died at step 5b
with `npm error nospc` — after the reset. AI Town served an empty 128KB database
for 33 hours while its container reported `healthy`, and nothing in the script
had looked at `df`. There is now a step 0 disk preflight that checks both the
job-dir filesystem and `$HOME` (where step 5b's npm cache lives) before step 1
touches anything. Recovery took `resume_compact_convex_data.sh`, which restored
all 117,679 documents from the original export; the compaction's own goal was
met in the process (10.2GB → 314MB).

A host crontab entry runs this daily (threshold-gated, so it's a no-op most
days — see `crontab -l` for the exact line, installed 2026-07-29). There is
brief real downtime for AI Town while a compaction actually runs (stop →
reset → restart → reimport), typically well under a minute once the backend
is healthy again, though function redeploy/reindexing can add several
minutes on a large table set.

## Cast cards (source of truth)

The live character set — **Mara Vale**, **Nico Sable**, **Sofia Bell**, and **Cam Lin**, plus **Juniper Feld** (human) and **Orion** (external join) — lives as authored cards in `cards/town_cards.yaml`. This is the single source of truth for identities. Retired cards (Juno Park, Tessa Quinn, Vale Moreno, Dr. Elian Cross) are archived at `cards/archived/2026-08-29-retired-cast.yaml` — do not delete them.

**Personal details stay local.** The repo is public, so the committed `cards/town_cards.yaml` is a redacted example. The real cards live in the gitignored `cards/town_cards.local.yaml`, and the generator reads that first. Its outputs follow the source: real cards write `cards/generated/*.local.txt` (gitignored, and the embodiment bootstrap reads `orion_town_card.local.txt` before `orion_town_card.txt`). The example cards write the tracked `*.txt` files. `patches/orion-human-juniper.patch` holds the token `__ORION_TOWN_HUMAN_DESCRIPTION__` in place of Juniper's join description; `scripts/apply_upstream_patches.sh` fills it in from the cards (`generate_descriptions.py --render-human-patch`). A fresh host needs `town_cards.local.yaml` copied over by hand, or it gets the redacted text. CI gate: `tests/test_no_personal_terms.py`.

Regenerate the AI Town artifacts from the cards with the deterministic generator (run from repo root):

```bash
python services/orion-ai-town/scripts/generate_descriptions.py
```

It rewrites, in `upstream/`: the `Descriptions` array in `data/characters.ts` (rich, prompt-injected NPC identities read by `convex/agent/conversation.ts`), `DEFAULT_NAME` in `convex/constants.ts`, and Juniper's join sprite + description in `convex/world.ts` (sprite from `cards.sprites.juniper_feld`, currently `f6`, the female redhead — not random). It also emits `cards/generated/orion_town_card.txt` (Orion's full blurb, consumed by the embodiment bootstrap) and `cards/generated/juniper_description.txt` (reference copy). After regenerating, refresh the tracked patches:

```bash
git -C upstream diff -- data/characters.ts > patches/orion-character.patch
git -C upstream diff -- convex/constants.ts convex/world.ts > patches/orion-human-juniper.patch
```

## Living world: simulation core

The town now has a world that changes on its own, in plain code with no model calls. It holds places, stock (coffee at the diner, parts at the workshop, everyone's cash), each resident's needs, trust between characters, facts, and quests. Every 5 minutes a Convex cron moves it forward by 5 game-minutes, but only while the town world is running. Design: `docs/superpowers/specs/2026-10-14-aitown-living-world-design.md` (PR #2666), sections 1–3 and 8. This is PR 2 of that design.

What one tick does:
- Needs drift by each resident's schedule. Needs are wants: 0 means satisfied, 1 means urgent. Rest builds up during work and recovers during sleep. Social builds up when awake. Purpose rises while the resident is in an open quest and slowly falls otherwise. Money is not drifted: it follows cash (no cash means 1, 40 cash or more means 0).
- Routines run at the top of the hour: coffee at the diner at 8, 12 and 18 (the coffee leaves stock, the cash goes to the diner's owner), and a wage at the end of each shift.
- Random events fire from a small template table (coffee and parts deliveries, the workshop generator breaking and getting fixed, gossip that moves trust, small loans). Each template has a rate per game-hour and fires as a seeded Poisson arrival.
- Quests resolve: a quest succeeds when its goal holds and fails once its deadline passes. Either way its reward or penalty effects apply, and a `quest.<id>` fact records the outcome.

Every random draw comes from a generator whose state is part of the saved world. The same world plus the same seed gives exactly the same next world (tested: one 24-hour tick equals 288 five-minute ticks with a save in between).

**Files.** The code ships as whole new files in `overlay/convex/sim/`, copied into `upstream/` after the patches:
- `core.ts`: the pure simulation (no imports at all, so CI runs it under Node).
- `schema.ts`, `store.ts`, `tick.ts`, `events.ts`, `state.ts`: the thin Convex layer.
- `seed.ts`: generated from the cards (see below).

`patches/orion-sim-core-hooks.patch` is the only upstream edit: it registers the two tables in `convex/schema.ts` and the cron in `convex/crons.ts`.

**Two tables, not ten.** The design listed ten tables. This PR uses two. `simState` holds one versioned document per world with the whole simulated world in it. `simEvents` is the append-only log of every event, with its kind, source, title, effects, generator seed, status and game-minute. One document means an event's effects (stock and trust and a fact together) commit in one write, and a replay starts from one value. The world is small (six characters, five places), so it fits one document easily. Scenes and predictions (PRs 3 and 4) are logs, so they get their own tables when they arrive.

**One apply path.** Every change goes through `core.applyEvent`: tick templates, routines, quest results and Juniper's events. An event carries a closed set of effects:

| op | target | value | rule |
|---|---|---|---|
| `stock_delta` | `<place or resident>.<item>` | integer | stock never goes below 0. A new key needs an item that owner kind already holds (places: coffee, parts; residents: cash), so `workshop.coffee` works and `diner.cash` is refused |
| `need_delta` | `<resident>.rest` / `.social` / `.purpose` | number | clamped to 0..1 (money follows cash) |
| `trust_delta` | `<from>\|<to>` | number | clamped to -1..1 |
| `fact_set` | fact key, e.g. `generator.fixed` | number, string or boolean | `rumor.*` and `quest.*` are written by the sim only; at most 100 facts |
| `place_flag` | `<place>.<flag>` | number, string or boolean | at most 16 flags per place |

An event with an unknown op, an unknown target or a bad value is refused whole, and nothing in it applies. Each effect is stored as exactly `{op, target, value}`; extra keys are dropped. An event with no effects becomes a rumor fact (`rumor.<event id>`, the 20 newest are kept) and moves no numbers. The world has sources `sim` (the tick only) and `juniper` (the only source `inject` takes). `mirror` and `orion` arrive with PRs 4 and 5.

**Size limits.** The whole world is one Convex document, so it stays bounded. New keys past a limit are refused with a reason, never evicted: 100 facts, 16 flags per place, 10 open quests, 16 effects per event, 15 per quest reward or penalty (the tick adds the `quest.<id>` result fact). Only the 20 newest rumors and the 20 newest resolved quests (with their `quest.<id>` facts) are kept.

**A bad quest cannot stop the tick.** If resolving a quest throws, the tick marks it `abandoned` with the reason, logs a `rejected` row, and carries on.

**The entity rule.** `core.checkEntities(state, names)` returns every name that is not a place, an item or a character in the world. The scene writer (PR 3) uses it to refuse a scene that names an invented person.

**Inspect** (from `upstream/`, with the admin key in `.env.local`):

```bash
npx convex run sim/state:snapshot                 # the world + the 20 newest events
npx convex run sim/state:snapshot '{"events": 100}'
npx convex run sim/tick:run '{"minutes": 60, "force": true}'   # advance by hand, even while the town is stopped (at most 1440)
```

The cron does nothing while the town is not running (`stoppedByDeveloper` or `inactive`). Since 2026-10-30 the town is paused by default and runs only in sessions: one Orion chose (capped at 2 hours a day), or one Juniper opened with Unfreeze. orion-embodiment's session controller heartbeats the world only during Orion's own sessions (see `services/orion-embodiment/README.md`, "AI Town sessions"). Game time pauses with the world and does not catch up afterwards, so after any pause game hours drift away from wall-clock hours: game 08:00 (breakfast) is not 08:00 on the wall. The first tick or event creates the world from the card seed. To reset it to the seed (the event log stays, and the game clock carries on):

```bash
npx convex run sim/state:bootstrap '{"replace": true}'
```

**Add an event as Juniper** (CLI only for now; the Hub form and the bus path arrive with PR 4):

```bash
npx convex run sim/events:inject '{"event": {"source": "juniper", "kind": "delivery", "title": "A crate of parts turns up at the workshop.", "place": "workshop", "effects": [{"op": "stock_delta", "target": "workshop.parts", "value": 5}]}}'
npx convex run sim/events:inject '{"event": {"source": "juniper", "kind": "rumor", "title": "Someone saw lights over the square."}}'
npx convex run sim/events:inject '{"event": {"source": "juniper", "kind": "quest", "title": "Fix the generator", "quest": {"goal": [{"kind": "fact", "target": "generator.fixed", "cmp": "==", "value": true}], "participants": ["cam_lin", "orion"], "deadline_in_minutes": 1440, "reward": [{"op": "trust_delta", "target": "cam_lin|orion", "value": 0.2}], "penalty": [{"op": "trust_delta", "target": "cam_lin|orion", "value": -0.1}]}}}'
```

Each call writes one `simEvents` row with `source: "juniper"`. A refused event writes a row with `status: "rejected"` and the reason, and changes nothing. Any other source (including `sim`) is refused without a row. Quest goals are an AND of conditions: `kind` is `stock`, `fact`, `need` or `trust`, and `cmp` is one of `>= <= > < == !=`. Participants are character ids.

**Cards to seed.** Each resident card in `cards/town_cards.yaml` has a `sim:` block: starting needs, schedule hours (0..23), work place, cash, wage per shift, purpose weights and lying rate. The work place decides who owns the diner (coffee money goes to them) and who fixes the workshop. Purpose weights pick who gossips (`trade_news` + `deceive`) and who asks for loans (`ask_favor`). Lying rate is how often gossip is a spun story, which can cost trust when caught. Every field is read by `core.ts`. The top-level `sim_world:` block holds the places, their starting stock, the starting facts and the seed. Juniper and Orion have no `sim:` block. They are visitors: they are valid names, quest participants and trust targets, but the tick never drifts them. `generate_descriptions.py` renders all of this into `overlay/convex/sim/seed.ts`. The committed copy renders the redacted example cards (a test fails when it drifts). `apply_upstream_patches.sh` renders it again into `upstream/` from `town_cards.local.yaml` when that file exists. A local cards file without `sim:` blocks borrows them from the example, because they hold no personal details.

**Tests.** `tests/test_sim_core.py` runs the shipped `core.ts` and `seed.ts` under Node >= 22.6 with type stripping (`AITOWN_TEST_NODE` points at another node). It covers acceptance A1 (a seeded 24 game-hour run moves stock, needs and trust, and replays exactly), clamping, refused effects, rumors, quests that succeed and fail, the entity rule, and A7's Juniper half at the core. CI runs it in `.github/workflows/aitown-patch-chain.yml`.

## Orion embodiment

`patches/orion-character.patch` is the tracked diff for AI Town's `Descriptions` array (applied by `scripts/apply_upstream_patches.sh` alongside the embed patch). **Live cards, `generate_descriptions.py`, and the checked-in patch seed four NPCs** — Mara Vale, Nico Sable, Sofia Bell, and Cam Lin. Retired cards (Dr. Elian Cross, Juno Park, Tessa Quinn, Vale Moreno) are archived at `cards/archived/2026-08-29-retired-cast.yaml`. Orion is **not** in `Descriptions`; Orion joins externally — its body created/updated by `services/orion-embodiment/scripts/bootstrap_orion_agent.py` (dry-run by default; `--write` persists `AITOWN_ORION_*` to `~/.fcc/.env`). Orion joins with its **authored town card** (`cards/generated/orion_town_card.txt`, from `town_cards.yaml`); if that file is unreachable the bootstrap falls back to the live self-model projection, then a minimal safe blurb. Juniper Feld is the **human player**, wired via `patches/orion-human-juniper.patch` (sets `DEFAULT_NAME = 'Juniper Feld'`, pins `character: 'f6'`, and her rich join description in `convex/world.ts`). Nico is `f2` (boy sheet). Orion and Cam stay on `f1`. A world wipe is still required after recasting (`testing:wipeAllTables` then `init`) — see Fresh game / reset below.

> Note: `patches/orion-character.patch` and `patches/orion-human-juniper.patch` are generated from real diffs against the cloned `upstream/`. On a node where `upstream/` is not yet cloned, the apply script skips a patch (with a message) rather than failing; generate the patches on a node that has `upstream/` before relying on the cast. If `upstream/` already has an older apply of these two patches, `apply_upstream_patches.sh` will refuse the new hunks — restore vanilla `data/characters.ts` and `convex/world.ts` from the pin (or re-clone), then re-apply, **or** run `generate_descriptions.py` on the dirty tree (it pins Juniper's sprite and rewrites Nico from the cards) and refresh the two patch files.

### Deploys: `scripts/pre_rebuild.sh` and `scripts/post_rebuild.sh`

`scripts/rebuild_services_from_git_diff.sh` runs `scripts/pre_rebuild.sh` before every post-pull rebuild of this stack. It runs `scripts/post_rebuild.sh` only after every compose step passed. If either fails, the stack is marked failed. Before 2026-10-23 nothing did this. The frontend image builds from `upstream/`, and the Convex functions live outside compose, so a pull rebuilt only an unpatched frontend. The live town ran 2026-08-30 code for almost two months.

`pre_rebuild.sh` does four things:
1. It saves `upstream/`'s changes to `~/.orion/aitown-upstream-backup-<time>/`: the diff against HEAD, and the contents of untracked files. It keeps the last 10.
2. It resets tracked files to `AITOWN_UPSTREAM_REF` from `.env_example` and deletes the files the patches create. It also deletes the files `overlay/` copies, and any overlay directory the pin does not have (`convex/sim/`) as a whole, so a file dropped from `overlay/` does not stay behind and deploy.
3. It applies the chain from scratch, then copies `overlay/` over `upstream/` and renders `convex/sim/seed.ts` from the cards in use.
4. It runs `npx convex codegen --typecheck disable`, which rewrites `convex/_generated/` for the patched tree. The image build type-checks against those files, and the pin's copy lacks every module the patches add: on 2026-10-29 the build failed because `crons.ts` called `internal.sim.tick.run`. Codegen does not change the deployment, but it reads the Convex backend, so the hook first starts the `backend` container (`up -d --wait backend`, a no-op when it runs).

It refuses in two cases:
- `cards/town_cards.local.yaml` is missing, because the Juniper join text would render from the redacted example. Copy that file from the operator host with owner-only permissions.
- Upstream's HEAD isn't the pin. A pin bump also needs `npm ci`, so do it by hand.

`post_rebuild.sh` pushes the functions with `npx convex dev --once`. Both hooks run Convex through `scripts/run_convex.sh`: on a host without Node (circe), it runs `npx convex` in `node:20-alpine`. `post_rebuild.sh` runs after `up`, so the backend is running, and `safe_docker_build.sh`'s main-checkout and migration gates have already passed.

To deploy by hand: `bash scripts/pre_rebuild.sh`, rebuild the frontend, then `bash scripts/post_rebuild.sh`.

### Fresh game / reset

Reseed the town from scratch (destructive — wipes all world/memory tables). Operator-run:

Live cards, `generate_descriptions.py`, and the tracked `orion-character.patch` seed **four** NPCs (see Cast cards above). This recipe applies the tracked patch first, so `init` seeds the four live identities in `Descriptions`.

```bash
cd services/orion-ai-town && bash scripts/apply_upstream_patches.sh  # tracked patch seeds four NPCs
cd upstream && npx convex dev --once            # redeploy Convex functions
npx convex run testing:stop
npx convex run testing:wipeAllTables            # internalMutation; wipes all world/memory tables
npx convex run init                             # seeds the four live Descriptions after patches
npx convex run testing:resume
# re-bootstrap Orion's external body:
cd ../../.. && python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write
```

### Session correlation id (`patches/orion-town-correlation-id.patch`)

Every gateway call from `convex/util/llm.ts` (chat, embeddings, moderation) sends `X-Orion-Correlation-Id` when the Convex env var `ORION_TOWN_CORRELATION_ID` is set. orion-embodiment's session controller sets it to the open session's id and removes it when the session ends (the same admin call as `npx convex env set|remove`). The LLM gateway records it in `llm_call_usage.correlation_id`, so AI Town tokens join to the session (and to the `play:aitown.session` act, whose correlation id is the same). Before this the town's ~1,100 calls a day carried no correlation id. A deploy with no session open sends no header.

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

### NPC answer-first (`patches/orion-npc-answer-first.patch`)

Human chats never call `startConversationMessage` — Juniper's first NPC line is `continue`. The old continue contract (`Name a person, object, or task`) plus Nico's tease-plan produced a secret-hook on "hi" and a lockbox on "spill the tea" before any topic had repeated.

This patch (rewritten 2026-09-15 so it actually applies after pair-turn continuity) does four things for **every** NPC, not just Juniper chats:

1. Continue: answer the last line, no quest hooks, break abstract circling or leave.
2. Start: talk like a person who ran into them, not a quest giver.
3. Relabel `plan` as role background (`Your role today (background only, not the topic of this chat)`), not `Your goals for the conversation`.
4. Drop the forced “mention a previous conversation” greeting callback.

Companion `orion-resync-agent-descriptions.patch` adds engine input `resyncAgentDescriptions` so live baked `agentDescriptions` pick up card/plan rewrites after deploy (`scripts/resync_agent_descriptions.py`). Without that one-shot, Circe keeps serving the old coffee/quest plans forever.

### Mechanical leave (`patches/orion-mechanical-leave.patch`)

When an NPC–NPC chat hits duration or message cap, leave immediately via `conversation.leave` — do **not** ask the LLM to write a goodbye. Live 2026-09-14 Mara↔Orion: leave-via-`agentGenerateMessage` hung on empty stubs against a bloated transcript, so the chat never ended. Orion joins without a Convex `human` flag today, so this gate applies to Mara↔Orion as well as NPC↔NPC. Human (Juniper) partners still stay until they leave. Embodiment's `EMBODIMENT_CONVERSATION_ABANDON_SEC` is the independent backstop if a chat still sticks.

### NPC no-self-repeat (`patches/orion-npc-no-self-repeat.patch`)

Prompt anti-repetition was not enough: live 2026-09-15 Sofia↔Juniper posted the exact same line twice after a goodbye. Continue now rejects an exact (normalized) self-repeat against the last few own lines, retries once, then returns a sentinel; `agentGenerateMessage` turns that sentinel into a leave (ordinary empty continues are unchanged). Embodiment has the same class of gate for Orion (`is_self_repeat` + one retry, then mark that partner line exhausted so perception ticks do not storm cortex).

### Walk-over across a wall (`patches/orion-walkover-meet.patch`)

Live 2026-10-23: both NPC pairs stood on either side of a wall, `walkingOver`, for good. Their meeting spot was the midpoint, which was the wall tile. `findRoute` failed every tick, and only closes straight-line distance, so it never went around. `orion-conversation-proximity.patch` had removed the invite timeout for accepted walk-overs, so nothing ended it. Orion's own invites to those NPCs were then refused ("already in a conversation") about once a second. Now, when the midpoint is a wall or off the map, the player with the smaller id walks to a free tile next to the other, and the other waits. An accepted walk-over that hasn't met after `WALK_OVER_TIMEOUT` (3 minutes) is abandoned. On the live map, the rule gave reachable routes of 45 and 12 steps for the two stuck pairs. An agent never waits for a human (Juniper only moves when she clicks). The 3-minute limit also covers two cases the rule can't solve: a pair with no path between them, and a partner with all four neighboring tiles taken. Such a pair may re-invite after the 60 s cooldown.

### Live-turn speaker roles and partner echo (`patches/orion-live-turn-roles.patch`)

Upstream sent every chat line to the model as a `user` turn with a `Name to Name:` prefix. Live 2026-09-17 Sofia↔Mara (`c:578946`), both NPCs then copied each other's last line word for word for five turns. The self-repeat gate missed it, because it only compares a speaker with their own lines. Now the speaker's own lines go in as `assistant` turns and the partner's lines go in as `user` turns, as plain text. A reply that shares at least 70% of its words with the partner's last line (lines under 4 words are exempt) goes through the same retry-then-leave path as a self-repeat. After a question, only an exact copy counts, because a plain answer restates the question. The retry instruction goes in the system turn, so the model does not read it as the partner speaking. The threshold flags 2.0% of 9,391 live town turns. `tests/test_live_turn_roles_patch.py` runs the shipped check on the real transcript (Node >= 22.6). It also applies the whole patch chain to the pinned upstream. CI runs both in `.github/workflows/aitown-patch-chain.yml`.

The same PR rebased `orion-npc-no-self-repeat.patch` onto the continuity/answer-first chain without changing its behavior. On `main`, a fresh apply had stopped at that patch.

### No human idle kick (`patches/orion-no-human-idle-kick.patch`)

Upstream `player.tick()` deleted the human after `HUMAN_IDLE_TOO_LONG` (5 minutes). `lastInput` was only written at join, so that was a session fuse, not idle detection. The patch removes the kick. Juniper stays in the world until she actually leaves.

Orion's external embodiment worker already walks on `walkingOver` via `approach_player` intents in `services/orion-embodiment/app/worker.py`.

### Town continuity ingest (`patches/orion-town-continuity-ingest.patch`)

NPC-to-NPC openers fetch `aitown-town` continuity in `startConversationMessage` (`GET /town-continuity` with `thread_id` + `speaker_id` → `Earlier with them:` / `From your other conversations:` when non-empty). Human (Juniper) conversations skip that start path — `orion-town-chat-turns.patch` waits for the human's first line — and GET `/town-continuity` on the first continue instead (`priorMessages.length <= 1`). Later continues do not fetch again. Topic bags are not injected. Embodiment still uses `/summary`. NPC continue/leave still `POST /ingest-turn` (fire-and-forget, 3s abort) when the other player is not Orion. Orion↔anyone stays embodiment-only. Ingest and continuity fetch fail-open (3s `AbortSignal.timeout`) and never block the speech return path. Slugs are the same six hardcoded names as `orion/town_cast.py`.

These are **Convex env vars** (operator `npx convex env set` from `upstream/`), not this service's Python `.env`:

| Convex env | Example | Meaning |
|------------|---------|---------|
| `SOCIAL_MEMORY_URL` | `http://<mesh-ip>:8765` | `orion-social-memory` base URL (`GET /town-continuity`, `POST /ingest-turn`; embodiment still uses `GET /summary`) |
| `SOCIAL_MEMORY_INGEST_TOKEN` | same token as social-memory | Bearer token for `POST /ingest-turn` |
| `AITOWN_ORION_NAME` | `Orion` | Other-player name that skips Convex ingest (embodiment owns that dyad) |

```bash
cd services/orion-ai-town/upstream
npx convex env set SOCIAL_MEMORY_URL http://<mesh-ip>:8765
npx convex env set SOCIAL_MEMORY_INGEST_TOKEN "<token>"
npx convex env set AITOWN_ORION_NAME Orion
```

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
