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
  enough free disk, that atlas's Docker has the `app-net` network created,
  that atlas's own `.env`/`PROJECT`/`NODE_NAME` are what this runbook
  assumes, or that atlas is reachable from every other mesh node the way
  athena is.
- Whether atlas's Tailscale ACLs/firewall allow inbound Redis traffic
  (`6380` by default, or `6379` if `BUS_STANDBY_PORT` is overridden for a
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
# On athena, snapshot bus-core's data without touching the live path directly:
mkdir -p /tmp/bus-core-snapshot
docker run --rm \
  -v /mnt/telemetry/orion-athena/bus/data:/src:ro \
  -v /tmp/bus-core-snapshot:/dst \
  alpine sh -c "cp -a /src/. /dst/ && chown -R $(id -u):$(id -g) /dst"

# Copy the snapshot to atlas (adjust path/user for the real atlas layout):
rsync -a /tmp/bus-core-snapshot/ athena@atlas.tail348bbe.ts.net:/tmp/bus-core-snapshot/

# --- Everything below runs ON ATLAS, from atlas's own checkout's repo root. ---
cd /path/to/Orion-Sapienform  # atlas's own checkout

# Confirm atlas's own root .env has real values BEFORE using them below --
# do not assume, print them and read them:
grep -E "^(PROJECT|NODE_NAME|TELEMETRY_ROOT)=" .env
# Substitute the real printed values for <PROJECT>/<TELEMETRY_ROOT> in every
# command below by hand. Do not rely on shell variable expansion here --
# these are docker-compose --env-file values, not exported shell vars, and
# treating them as the latter is exactly the mistake that produced a broken,
# root-bound volume path when this runbook's compose command was first
# drafted (caught in review, fixed by always chaining --env-file .env
# --env-file services/orion-bus/.env.atlas-standby below).

# Seed the standby's own data dir from the snapshot, then start it (first
# time only -- afterwards bus-core-standby keeps its own AOF and this step
# is not needed again unless re-seeding from a fresher primary snapshot):
mkdir -p <TELEMETRY_ROOT>/<PROJECT>/bus-standby/data
cp -a /tmp/bus-core-snapshot/. <TELEMETRY_ROOT>/<PROJECT>/bus-standby/data/

cp services/orion-bus/.env_example services/orion-bus/.env.atlas-standby
# .env.atlas-standby only carries orion-bus-specific keys (BUS_STANDBY_PORT,
# etc). PROJECT/TELEMETRY_ROOT/NODE_NAME come from root .env, chained first:

docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env.atlas-standby \
  -f services/orion-bus/docker-compose.atlas-standby.yml \
  up -d

# Verify the restore actually happened -- do not skip this. Substitute the
# real <PROJECT> value; the container name is orion-<PROJECT>-bus-core-standby:
docker logs orion-<PROJECT>-bus-core-standby 2>&1 | grep -E "Done loading RDB|DB loaded from|Ready to accept"
docker exec orion-<PROJECT>-bus-core-standby redis-cli DBSIZE
docker exec orion-<PROJECT>-bus-core-standby redis-cli KEYS '*' | sort > /tmp/standby_keys.txt

# Compare against the primary's key count/names at snapshot time
# (run this on athena BEFORE the snapshot, save it, compare after):
docker exec orion-orion-athena-bus-core redis-cli DBSIZE
docker exec orion-orion-athena-bus-core redis-cli KEYS '*' | sort > /tmp/primary_keys.txt
diff /tmp/primary_keys.txt /tmp/standby_keys.txt   # expect no diff (modulo TTL expiry between snapshot and check)
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

1. Confirm Step 1's restore is verified (matching `DBSIZE`/`KEYS`).
2. Stop `bus-core` on athena, or confirm it is already down/crash-looping:
   ```bash
   docker compose --env-file .env --env-file services/orion-bus/.env \
     -f services/orion-bus/docker-compose.yml stop bus-core
   ```
3. Edit the root `.env` on athena (and anywhere else mesh services actually
   read their root `.env` from — e.g. any other node's own checkout):
   ```
   ORION_BUS_URL=redis://100.121.214.30:<BUS_STANDBY_PORT>/0
   ```
   `BUS_STANDBY_PORT` defaults to `6380` (`services/orion-bus/.env_example`)
   so the standby is safe to co-locate with the primary on one host for
   testing without a port collision. On the real, separate atlas node this
   default is unnecessary — set `BUS_STANDBY_PORT=6379` in
   `services/orion-bus/.env.atlas-standby` before deploying if you want
   cutover to be a pure IP swap; otherwise use whatever port the standby
   was actually deployed with (check `docker port
   orion-<PROJECT>-bus-core-standby` on atlas if unsure).
   (atlas's Tailscale IP; use the hostname `atlas.tail348bbe.ts.net` if DNS
   resolution over Tailscale is confirmed reliable for every consuming
   service's container network — IP is safer/simpler and matches how the
   existing value is a literal IP, not a hostname.)
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
   swap itself — Redis client libraries reconnect on next use, and
   consumer groups resume from where they left off as long as the stream
   data is present on the new instance (which Step 1 verified). As a
   practical sequencing preference, restart the central hub/orchestration
   layer first (`orion-hub`, `orion-cortex-orch`, `orion-cortex-gateway`)
   so downstream services reconnecting shortly after have something to
   talk to, then everything else in any order.
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
  `redis://100.92.216.81:6379/0` and repeat the recreate steps above. Any
  writes that landed on the standby during the cutover window are lost
  unless manually merged (see below) — acceptable for most of this bus's
  traffic (ephemeral pub/sub, short-TTL state keys), a real loss for
  anything durable that was only appended to the standby's stream. Check
  `orion:evt:gateway`/`orion:bus:out` stream length on the standby against
  what's expected before discarding it.
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
