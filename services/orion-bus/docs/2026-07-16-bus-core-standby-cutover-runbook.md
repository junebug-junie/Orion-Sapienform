# bus-core cold-standby: cutover / cutback runbook

Status: operational runbook, manual procedure only. No automation, no
failover, no leader election. Companion to
`services/orion-bus/docker-compose.atlas-standby.yml`.

## Scope and non-goals

`bus-core` (`services/orion-bus/docker-compose.yml`) is a single
`redis:7-alpine` instance on `athena` and is a real single point of failure.
This runbook covers a **passive, cold-standby target** on a second mesh node
and the fully manual steps to point the mesh at it if `athena`'s `bus-core`
is down for an extended period (or its data directory is corrupted beyond
what the separate AOF-auto-repair effort can fix).

Explicitly out of scope, do not improvise these mid-incident:

- Automatic failover or health-based promotion.
- Leader election or Postgres advisory-lock coordination.
- Live/streaming replication (`bus-core-standby` is not a Redis replica of
  `bus-core` — there is no `REPLICAOF` wired up). Data only moves between
  the two by manual restore of an AOF/RDB snapshot, which is why cutover is
  "restore data, then repoint clients," not "flip a switch."
- Fixing AOF corruption on the primary itself — that is a separate, parallel
  effort (`fix/bus-core-aof-auto-repair`). This runbook assumes you already
  have a *valid* AOF/RDB snapshot to work with, whether that's the primary's
  own (repaired) data or the standby's own accumulated data.

## Node chosen: atlas, not circe

`config/biometrics/node_catalog.yaml`:

```yaml
atlas:
  role: inference_gpu
  expected_online: true
circe:
  role: burst_gpu
  expected_online: false
```

`circe` is marked `expected_online: false` in the repo's own node catalog —
it's a burst/training box that is not reliably up. A cold standby that
itself isn't reliably online defeats the purpose. `atlas` is
`expected_online: true`, live-reachable over Tailscale (`atlas.tail348bbe.ts.net`,
`100.121.214.30`, confirmed via `tailscale status`/`ping` from `athena` while
writing this runbook). Deploy `bus-core-standby` there.

## What was actually tested vs. not (read this before trusting anything below)

**Tested, on `athena`, with real data (do not skip this section):**

- Took a read-only snapshot of the *live* `bus-core` AOF/RDB data
  (`/mnt/telemetry/orion-athena/bus/data`) using a throwaway container with
  that path bind-mounted `:ro` — the live path itself was never opened for
  write, only read through a read-only mount and copied elsewhere.
- Started a `redis:7-alpine` container against that snapshot copy and
  confirmed the AOF+base-RDB restore sequence
  (`Reading RDB base file on AOF loading` → `Done loading RDB, keys loaded: 20`
  → `DB loaded from incr file ... `) completes cleanly, and that
  `DBSIZE`/`KEYS *` on the restored instance exactly match the live
  primary's `DBSIZE` (20/20) with real Orion mesh keys
  (`orion:pad:events`, `orion:spark_state:latest:global`,
  `mesh-guardian:state`, etc.) — this is real evidence the AOF format
  restores correctly, not a guess.
- Brought up `docker-compose.atlas-standby.yml` itself (not just raw
  `redis-server`) end-to-end against that same snapshot, using a synthetic
  `PROJECT`/`TELEMETRY_ROOT` pointed at a scratch directory so it never
  touched `/mnt/telemetry/orion-athena/bus/` — confirmed `DBSIZE=20`,
  healthy healthcheck, and a `bus-core-standby` network alias distinct from
  the live `bus-core` alias (no accidental collision).
- Confirmed the live primary's `bus-core` was untouched and still healthy
  (`Up ... (healthy)`, `DBSIZE=20`) after the standby container was torn
  down.

**Not tested, needs a human (or an agent with real access) to verify before
trusting this runbook in an actual incident:**

- Actually deploying `docker-compose.atlas-standby.yml` **on** the atlas
  host. This session runs on `athena` and does **not** have SSH access to
  `atlas` (`ssh atlas.tail348bbe.ts.net` → `Permission denied
  (publickey,password)` — confirmed live, not assumed). Everything above
  was tested on `athena` against a synthetic node identity, which proves
  the compose file and the AOF-restore mechanism work, but does **not**
  prove atlas-specific things: that atlas has a `TELEMETRY_ROOT` with
  enough free disk (Step 1 now includes a check command, but it was never
  run against real atlas disk), that atlas's Docker has the `app-net`
  network created (Step 1's deploy command now includes the idempotent
  create-if-missing step, matching services/orion-bus/README.md's own
  bootstrap), that atlas's own `.env`/`PROJECT`/`NODE_NAME` are what this
  runbook assumes, or that atlas is reachable from every other mesh node the
  way athena is.
- Whether `TELEMETRY_ROOT` (default `/mnt/telemetry`, root `.env_example`)
  is a node-local mount or a mesh-shared one on atlas specifically was never
  established this session — the root `.env_example` doesn't say either way,
  and this matters a lot here: if it turns out to be shared (e.g. an NFS/SMB
  mount visible from multiple nodes), the standby's `bus-standby/data`
  subdirectory could unintentionally collide with or shadow something else
  already using that path on atlas. Confirm with `df -h /mnt/telemetry` and
  `mount | grep telemetry` on atlas directly before trusting this runbook's
  path assumptions.
- Whether atlas's Tailscale ACLs/firewall allow inbound Redis traffic
  (`6381` by default, or `6379` if `BUS_STANDBY_PORT` is overridden for a
  real cutover — see Step 2) from the rest of the mesh the same way
  athena's does. Nothing in this session checked that.
- The full mesh-wide `ORION_BUS_URL` audit below (which services actually
  honor a changed root `.env` value) was done by grep against the current
  `main` branch, not by actually cutting a live service over and watching
  it reconnect. Treat the "51 vs. 3 vs. 20" breakdown as a snapshot of
  `main` as of 2026-07-16, not a guarantee — re-run the greps at actual
  cutover time.

## Step 1 — restore data onto the standby, and verify it

Before trusting the standby, prove its data is real, on the *actual* atlas
host (repeat the athena-side test above, on atlas, against real primary
data — not synthetic):

```bash
# --- Everything up to the rsync line below runs ON ATHENA. ---

# On athena, capture the primary's key list and DBSIZE BEFORE snapshotting,
# so there is a real point-in-time baseline to diff against later. Write it
# INTO the snapshot directory itself so the one rsync below carries both
# the data and the key list to atlas together -- no separate transfer step
# to forget:
mkdir -p /tmp/bus-core-snapshot
docker exec orion-orion-athena-bus-core redis-cli DBSIZE
docker exec orion-orion-athena-bus-core redis-cli KEYS '*' | sort > /tmp/bus-core-snapshot/primary_keys.txt

# On athena, snapshot bus-core's data without touching the live path directly:
docker run --rm \
  -v /mnt/telemetry/orion-athena/bus/data:/src:ro \
  -v /tmp/bus-core-snapshot:/dst \
  alpine sh -c "cp -a /src/. /dst/ && chown -R $(id -u):$(id -g) /dst"

# Copy the snapshot (data + primary_keys.txt together) to atlas (adjust
# path/user for the real atlas layout). This is the one and only file
# transfer between the two hosts this step needs -- everything below reads
# from the copy that lands here, nothing later reaches back across hosts:
rsync -a /tmp/bus-core-snapshot/ athena@atlas.tail348bbe.ts.net:/tmp/bus-core-snapshot/

# --- Everything below runs ON ATLAS, from atlas's own checkout's repo root. ---
cd /path/to/Orion-Sapienform  # atlas's own checkout

# Confirm atlas's own root .env has real, ATLAS-APPROPRIATE values BEFORE
# using them below -- do not assume, print them and read them:
grep -E "^(PROJECT|NODE_NAME|TELEMETRY_ROOT)=" .env
# A non-empty result here is NOT enough by itself: this repo's own root
# .env_example ships plausible-looking, non-placeholder defaults
# (PROJECT=orion-athena, NODE_NAME=athena) -- a stale or accidentally-copied
# athena .env on atlas would pass a bare "is it empty" check while still
# being wrong, producing a standby quietly named/pathed as if it were
# athena's own (a quieter, more confusing failure than the empty-string case
# below, which at least fails loudly). If this is genuinely atlas's own
# checkout, NODE_NAME must NOT read "athena" -- if it does, this is athena's
# .env, not atlas's, and every command below will target the wrong node.
# Substitute the real printed values for <PROJECT>/<TELEMETRY_ROOT> in every
# command below by hand. Do not rely on shell variable expansion here --
# these are docker-compose --env-file values, not exported shell vars, and
# treating them as the latter is exactly the mistake that produced a broken,
# root-bound volume path when this runbook's compose command was first
# drafted (caught in review, fixed by always chaining --env-file .env
# --env-file services/orion-bus/.env.atlas-standby below).

# Seed the standby's own data dir from the snapshot, then start it (first
# time only -- afterwards bus-core-standby keeps its own AOF and this step
# is not needed again unless re-seeding from a fresher primary snapshot).
# Exclude primary_keys.txt -- it rode along in the same rsync payload for
# convenience but is not Redis data and must not land inside the AOF/RDB
# data dir:
mkdir -p <TELEMETRY_ROOT>/<PROJECT>/bus-standby/data
rsync -a --exclude=primary_keys.txt /tmp/bus-core-snapshot/ <TELEMETRY_ROOT>/<PROJECT>/bus-standby/data/

# Disk space: the standby's data dir needs to hold at least the size of the
# snapshot just rsynced in, plus headroom for ongoing AOF growth. Check
# before seeding, not after a failed write mid-restore:
df -h "$(dirname <TELEMETRY_ROOT>/<PROJECT>/bus-standby/data)"
du -sh /tmp/bus-core-snapshot

cp services/orion-bus/.env_example services/orion-bus/.env.atlas-standby
# .env.atlas-standby only carries orion-bus-specific keys (BUS_STANDBY_PORT,
# etc). PROJECT/TELEMETRY_ROOT/NODE_NAME come from root .env, chained first:

# app-net must exist before `up` -- it's declared external:true in the
# compose file (same as every other compose file in this repo), not created
# by it. Idempotent create-if-missing, matches
# services/orion-bus/README.md's own bootstrap step:
docker network inspect app-net >/dev/null 2>&1 || docker network create app-net

docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env.atlas-standby \
  -f services/orion-bus/docker-compose.atlas-standby.yml \
  up -d --build

# Verify the restore actually happened -- do not skip this. Substitute the
# real <PROJECT> value; the container name is orion-<PROJECT>-bus-core-standby:
docker logs orion-<PROJECT>-bus-core-standby 2>&1 | grep -E "Done loading RDB|DB loaded from|Ready to accept"
docker exec orion-<PROJECT>-bus-core-standby redis-cli DBSIZE
docker exec orion-<PROJECT>-bus-core-standby redis-cli KEYS '*' | sort > /tmp/standby_keys.txt

# Compare against the primary's key list captured on athena BEFORE the
# snapshot (/tmp/bus-core-snapshot/primary_keys.txt, transferred here by the
# same rsync that carried the data -- see above). Both files are already on
# atlas at this point; this diff needs no further cross-host step:
diff /tmp/bus-core-snapshot/primary_keys.txt /tmp/standby_keys.txt   # expect no diff (modulo TTL expiry between snapshot and check)
```

If `DBSIZE` and `KEYS *` don't match (accounting for keys that legitimately
expired in the gap between snapshot and check — several of the live keys
have TTLs, per `expires=12` in `INFO keyspace`), stop. Do not proceed to
cutover with an unverified restore.

## Step 2 — cutover: what actually needs to change, and where

`ORION_BUS_URL` is not a single value referenced from one place. It is a
root `.env` key (`ORION_BUS_URL=redis://100.92.216.81:6379/0`) that most,
but not all, services pick up via a compose-level `${ORION_BUS_URL}`
pass-through. Verified against `main` on 2026-07-16 by grep:

- **51 services'** `docker-compose.yml` files pass it through explicitly:
  `environment: ORION_BUS_URL: ${ORION_BUS_URL}` (or the `=` form). For
  these, changing the root `.env` value and recreating the container is
  sufficient — Compose re-interpolates `${ORION_BUS_URL}` from whatever
  `--env-file .env` supplies at `up` time, and the `environment:` block
  always wins over anything in the service's own `env_file: .env`, so a
  stale literal sitting in a per-service `.env` does not matter for these.
- **3 services hardcode the primary's IP directly inside their own
  `docker-compose.yml`**, not via `${ORION_BUS_URL}`:
  `orion-llama-cola-host`, `orion-vision-retina`,
  `orion-llamacpp-neural-host`. Root `.env` changes do **nothing** for
  these — their compose file (or a compose override) must be edited
  directly.
- **20 services have no `ORION_BUS_URL` reference anywhere in their
  `docker-compose.yml`** at all: `orion-ai-town`, `orion-attention-runtime`,
  `orion-bus` (itself), `orion-consolidation-runtime`, `orion-fcc`,
  `orion-field-digester`, `orion-graphiti-adapter`, `orion-knowledge-forge`,
  `orion-llamacpp-host`, `orion-notify-digest`, `orion-ollama-host`,
  `orion-pageindex`, `orion-policy-runtime`, `orion-proposal-runtime`,
  `orion-rdf-store`, `orion-self-state-runtime`, `orion-sql-db`,
  `orion-vector-db`, `orion-vllm-host`, `orion-world-pulse`. Some of these
  are pure backing stores that genuinely don't talk to the bus
  (`orion-sql-db`, `orion-vector-db`, `orion-rdf-store`, `orion-ollama-host`,
  `orion-vllm-host`, `orion-pageindex`) and don't need anything. **The rest
  are real cognition/runtime services whose bus wiring is not visible from
  `docker-compose.yml` alone** — most have `100.92.216.81` hardcoded as a
  Python-level `settings.py` default (confirmed present in
  `orion-self-state-runtime`, `orion-world-pulse`, `orion-substrate-runtime`,
  `orion-spark-introspector`, `orion-fcc`, and ~20 others), used only if the
  process's actual env var ends up unset — which for these particular
  services is a real possibility since nothing visible in their compose
  file guarantees it gets set. **Do not assume these follow a root `.env`
  change.** Check each one individually before the incident, not during it:
  `docker exec <container> env | grep ORION_BUS_URL` against the running
  container to see what it actually resolved to.

Reproduce this breakdown yourself at actual cutover time (the mesh grows;
this list is a snapshot, not a promise):

```bash
grep -rl 'ORION_BUS_URL: \${ORION_BUS_URL}\|ORION_BUS_URL=\${ORION_BUS_URL}' services/*/docker-compose.yml
grep -rln 'ORION_BUS_URL.*100\.92\.216\.81\|ORION_BUS_URL.*100\.x\.x\.x' services/*/docker-compose.yml
for f in services/*/docker-compose.yml; do grep -q "ORION_BUS_URL" "$f" || echo "$f"; done
```

### Cutover steps

**Stop here before running any step below in a real incident.** This
repoints the live root `.env` of the whole mesh and recreates every
consuming service's container -- a production write with mesh-wide blast
radius, exactly the kind of action AGENTS.md §13 requires explicit sign-off
for ("if touching production, state the exact action and wait for
confirmation"). Get that sign-off first. Steps 1-2 above (restoring and
verifying the standby's data) are read-mostly against athena and don't need
this gate; the cutover steps below do.

1. Confirm Step 1's restore is verified (matching `DBSIZE`/`KEYS`).
2. Stop `bus-core` on athena, or confirm it is already down/crash-looping:
   ```bash
   docker compose --env-file .env --env-file services/orion-bus/.env \
     -f services/orion-bus/docker-compose.yml stop bus-core
   ```
3. **Before editing anything**, re-verify atlas's Tailscale IP is still
   `100.121.214.30` — it was only confirmed once, at the time this runbook
   was authored (Step 0 above), and a stale copy-pasted IP during a real
   incident would silently repoint the whole mesh at a dead or reassigned
   address:
   ```bash
   tailscale status | grep atlas   # or: ping -c1 atlas.tail348bbe.ts.net
   ```
   If the IP has changed, use the current one in place of
   `100.121.214.30` below — do not proceed on the value in this doc without
   checking.

   Which port to use: **this runbook's own recorded assumption is
   `BUS_STANDBY_PORT=6381`** (the template default in
   `services/orion-bus/.env_example` -- chosen instead of the originally
   drafted `6380` after review found that port already hardcoded by
   `services/orion-falkordb/` (shared FalkorDB operator stack on host port 6380).
   container; `6381` was confirmed free across every `services/*/docker-
   compose*.yml` as of 2026-07-16) — treat this as the deployed value unless
   you have direct, live-verified knowledge it was overridden to `6379` at
   initial deploy time on atlas (e.g. you did that deploy yourself, or a
   prior operator left a note recording it — check for one before assuming
   6381). If you can reach atlas via SSH, confirm directly instead of
   relying on this assumption: `docker port orion-<PROJECT>-bus-core-standby`
   on atlas. If you cannot reach atlas (this runbook was itself authored
   during a session where atlas SSH access was unavailable — don't assume
   it'll be available during a real incident either), fall back to the
   `6381` assumption above rather than being stuck with no path forward.

   Edit the root `.env` on athena (and anywhere else mesh services actually
   read their root `.env` from — e.g. any other node's own checkout):
   ```
   ORION_BUS_URL=redis://100.121.214.30:<BUS_STANDBY_PORT>/0
   ```
   (atlas's Tailscale IP, re-verified above; use the hostname
   `atlas.tail348bbe.ts.net` if DNS resolution over Tailscale is confirmed
   reliable for every consuming service's container network — IP is
   safer/simpler and matches how the existing value is a literal IP, not a
   hostname.)
4. For the 3 services that hardcode the IP directly in their own
   `docker-compose.yml` (Step 2's list), edit those files' `environment:`
   blocks directly to the same IP.
5. For the "no reference visible" services that turn out (per your
   `docker exec ... env | grep ORION_BUS_URL` check) to actually be running
   with the stale hardcoded default, override it explicitly — either add
   `ORION_BUS_URL=${ORION_BUS_URL}` to their compose `environment:` block
   (real fix, requires a rebuild) or set it in their own service `.env`
   directly for a same-day incident fix.
6. Recreate every affected container so the new env value actually takes
   effect (env vars are baked in at container start; editing `.env` alone
   does nothing to a running container):
   ```bash
   docker compose --env-file .env --env-file services/<svc>/.env \
     -f services/<svc>/docker-compose.yml up -d
   ```
7. Restart order: no strict dependency ordering is required for the bus
   swap itself — Redis client libraries generally reconnect on next use.
   **Not independently verified this session:** whether consumer groups
   actually resume delivery from the correct position on the new instance.
   Step 1 only confirmed `DBSIZE`/`KEYS` match, which proves the data
   restored correctly but says nothing about consumer-group cursor state
   (`XINFO GROUPS <stream>` on both primary and standby, before and after
   cutover, is the real check — not done here). Redis persists consumer
   group state in the same RDB/AOF as the keys themselves, so it should
   carry over with the snapshot, but this repo has no live evidence of that
   for this specific setup — treat it as a real check to run during the
   next test cutover, not a settled fact. As a practical sequencing
   preference, restart the central hub/orchestration layer first
   (`orion-hub`, `orion-cortex-orch`, `orion-cortex-gateway`) so downstream
   services reconnecting shortly after have something to talk to, then
   everything else in any order.
8. Verify: pick 2-3 services from each bucket in Step 2 and confirm
   `docker exec <container> env | grep ORION_BUS_URL` shows the new value,
   and that `docker logs` shows a fresh Redis connection with no repeated
   connection-refused errors.

## Cutting back once athena's primary is healthy again

This is a **cold** standby, not a live replica — there is no ongoing
replication, so whatever was written to `bus-core-standby` while it was
serving the mesh is **not** automatically on the recovered primary.
Choose one of two paths depending on why you cut over in the first place:

- **Primary's own data was never lost** (e.g. cut over for a hardware/network
  outage on athena, not data corruption): repoint `ORION_BUS_URL` back to
  `redis://100.92.216.81:6379/0` and repeat the recreate steps above. **There
  is no vetted merge procedure for writes that landed on the standby during
  the cutover window — they are lost when you cut back**, full stop. This is
  acceptable for most of this bus's traffic (ephemeral pub/sub, short-TTL
  state keys), a real loss for anything durable that was only appended to
  the standby's stream. Check `orion:evt:gateway`/`orion:bus:out` stream
  length on the standby against what's expected before discarding it — if
  the loss looks significant, a manual per-key `redis-cli --rdb`/`DUMP`+
  `RESTORE` copy of specific known keys from standby to the restored primary
  is possible in principle, but has not been built, tested, or written up
  as a runbook step here; treat it as a from-scratch incident-time task, not
  something this document has already solved. The real mitigation is
  avoiding writes during the cutover window in the first place, not
  recovering them after.
- **Primary's data was corrupted/lost** (the AOF-crash-loop scenario this
  standby exists for): do **not** cut back to the old, empty/corrupt
  primary data directory. Instead, promote the standby's data: snapshot
  `bus-core-standby`'s data dir the same read-only way Step 1 did, copy it
  onto athena, stop `bus-core`, replace
  `/mnt/telemetry/orion-athena/bus/data` with the promoted snapshot (back
  up the old corrupt directory first, don't just delete it, in case
  forensics on the corruption is still useful for the AOF-auto-repair
  effort), start `bus-core`, verify `DBSIZE`/`KEYS` match the standby, then
  repoint `ORION_BUS_URL` back to athena and recreate the affected
  containers.
- Either way, after cutting back, stop `bus-core-standby` (or leave it
  running as a cold standby again — it does no harm idle, it's just not
  wired to anything) and re-run Step 1's verification once more against
  fresh primary data so the standby stays a trustworthy target for the
  *next* incident instead of a silently stale one.

## Known gaps in this runbook (be honest about these)

- The "51 / 3 / 20" service breakdown in Step 2 was produced entirely from
  `main` on 2026-07-16 via `grep` against compose files — it was not
  exercised end-to-end against a running mesh. Some of the 20
  "no-reference" services may in practice already get `ORION_BUS_URL`
  correctly from a mechanism this grep didn't catch (e.g. a shared base
  compose fragment, a `.env` merge this session didn't trace). Verify with
  `docker exec ... env` before an incident, not during one.
- No test exercised deploying to atlas itself — see "What was actually
  tested" above. The compose file and AOF-restore mechanism are proven;
  the atlas-specific deployment (disk space, network config, Tailscale
  firewall) is not.
- This runbook does not address what happens to services that buffer
  outbound bus writes locally while the primary is down (if any do) — a
  full audit of at-least-once vs. best-effort publish semantics per service
  is out of scope here.
