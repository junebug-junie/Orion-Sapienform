# 🧠 Orion Bus — Mesh Message Backbone

The **Orion Bus** is the shared message-broker layer that connects every service in the Orion-Sapienform mesh.
It functions like the **nervous system** of the network — propagating events, state changes, and telemetry across all nodes.

---

## 🌐 Overview

| Component        | Purpose                                          | Port   |
| ---------------- | ------------------------------------------------ | ------ |
| **Bus Core**     | Primary message broker (temporary Redis backend) | `6379` |
| **Bus Exporter** | Prometheus metrics endpoint                      | `9121` |

All services in the mesh communicate through this bus for:

* Event emission (`orion:evt:*`)
* Internal messaging (`orion:bus:*`)
* Shared state and coordination
* Real-time introspection and health signals

---

## ⚙️ Prerequisites

* Docker / Docker Compose v2
* Shared Docker network (usually created once per node):

```bash
docker network inspect app-net >/dev/null 2>&1 || docker network create app-net
```

---

## 🚀 Bring Up / Down

```bash
# Start the bus stack
make up

# Show running containers
make ps

# Tail logs
make logs

# Stop the stack
make down
```

### Redis pubsub output buffers

`bus-core` sets a raised pubsub client output buffer limit so slow subscribers are less
likely to be disconnected by Redis during long RPC handlers:

```text
client-output-buffer-limit pubsub 64mb 32mb 120
```

If production uses an external Redis (for example the Tailscale-hosted bus node), apply
the same setting in `redis.conf` and restart Redis there. Recreating only the compose
`bus-core` container does not change an external broker.

---

## 🔍 Monitoring

Prometheus metrics are exposed at:

```
http://localhost:9121/metrics
```

To quickly view the top of the metrics stream:

```bash
make metrics
```

### Host-level crash-loop watchdog (survives bus AND Postgres both being down)

`bus-core` periodically crash-loops from AOF corruption (auto-repair for that is a
separate, parallel piece of work -- not covered here). The gap this closes:
detecting "bus is stuck in a crash loop" today depends on either a human noticing
cascading failures across many *other* services, or on signals that themselves
route through the bus or Postgres -- both of which can be down at the same time in
dev (confirmed, not hypothetical). `scripts/bus_core_health_watchdog.py` is a
detection floor that survives both being down simultaneously: it reads
`docker inspect`'s `.State.Health.Status` / `.RestartCount` for the real
`orion-${PROJECT}-bus-core` container only -- **no Redis connection, no Postgres
connection, ever**. It persists its own state as a local JSON file and, on a
crash-loop signature, writes a plain, impossible-to-miss marker file (this repo
has no `notify-send`/`osascript`/desktop-notification mechanism to reuse; the one
HTTP alert path that exists, `orion-notify`'s `/attention/request`, is itself
infrastructure that can plausibly be down in the same incident this watchdog
exists to catch).

Two independent crash-loop signatures, either fires an alert (both overridable):

- **Consecutive-unhealthy streak** -- default 3 consecutive watchdog runs where
  health was not `healthy` (`unhealthy`, the container missing, or the Docker
  daemon itself unreachable all count; `starting`/`none` are neutral).
- **Restart-count-in-window** -- default 3 restarts within a 10-minute rolling
  window, from `docker inspect`'s cumulative `.RestartCount`. Catches a crash
  loop that never settles into `unhealthy` between polls.

State file (default): `${TELEMETRY_ROOT}/${PROJECT}/bus/watchdog/health_watchdog_state.json`
-- a sibling of, not inside, the AOF data mount at `${TELEMETRY_ROOT}/${PROJECT}/bus/data`.

Alert marker (default, never auto-deleted, updated with a RESOLVED footer instead
of removed if the condition clears): `${TELEMETRY_ROOT}/${PROJECT}/bus/ORION_BUS_CRASH_LOOP_ALERT.txt`
-- **this is where Juniper should look** if bus-core is suspected to be
crash-looping and other symptoms (bus-dependent services failing, Postgres also
possibly down) are inconclusive.

```bash
python3 scripts/bus_core_health_watchdog.py
# or: make bus-core-health-watchdog
```

Both default to `$PROJECT`/`$TELEMETRY_ROOT` from `.env` (`orion-athena`/`/mnt/telemetry`
today) if unset; override with `--project`/`--telemetry-root`, or pin exact paths
with `--container`/`--state-file`/`--alert-marker`. See the script's own docstring
for the full flag list and exit-code contract (0 healthy, 1 crash-loop detected, 2
the `docker` binary itself could not be run).

The default state/marker paths live under `/mnt/telemetry/${PROJECT}/bus/`, which
on this host is owned by `root:root` (755) -- the first run needs the directory
either pre-created and chowned to whichever user runs the cron job, or run once
manually with `sudo` to create it:

```bash
sudo mkdir -p /mnt/telemetry/${PROJECT}/bus/watchdog
sudo chown "$(whoami)":"$(whoami)" /mnt/telemetry/${PROJECT}/bus/watchdog
```

#### Scheduled maintenance (Athena cron)

Install on the host that runs `bus-core` (`crontab -e` as `athena`), matching this
repo's established pattern for host-level scheduled checks (see
`services/orion-rdf-store/README.md`'s "Scheduled maintenance" section and
`services/orion-memory-consolidation/README.md`'s, both crontab-based). The one
systemd-timer precedent in this repo (`docs/superpowers/plans/2026-05-09-local-mnt-scripts-backup-implementation.md`,
units generated into a gitignored `.worktrees/` path, not present in a fresh
checkout) is for a single long-running nightly backup job, not this
repeated-fast-poll shape -- crontab is the better fit here, and matches the
dominant house convention:

```cron
# bus-core crash-loop watchdog -- docker-inspect only, no Redis/Postgres dependency
* * * * * cd /mnt/scripts/Orion-Sapienform && make bus-core-health-watchdog >> /mnt/scripts/Orion-Sapienform/logs/orion-bus-core-health-watchdog.log 2>&1
```

Every-minute cadence matches the default thresholds (3 consecutive unhealthy checks
~= 3 minutes; 3 restarts within a 10-minute window) -- fast enough to catch a real
crash loop within a few minutes, slow enough not to flap on a single transient
health-check blip (`bus-core`'s own Docker healthcheck already requires 5
consecutive internal failures at 5s intervals, ~25s, before it reports `unhealthy`
in the first place). If you change the cron cadence, re-derive the threshold
defaults (`--unhealthy-streak-threshold`, `--restart-count-threshold`,
`--restart-window-minutes`) so they stay a real multiple of the new interval,
same caveat noted in the concept-relation-digest precedent.

Not installed automatically -- per this repo's safety rules, host crontab is
host-level shared infrastructure and is not modified without explicit sign-off.
Install the line above by hand.

The `>> ... 2>&1` redirect to a log file is deliberate, not an alerting gap:
this script prints a status line on *every* run, healthy or not, so an
unredirected cron job would mail on every single invocation (every minute)
rather than only on failure. The real alert path for a human is the
`ORION_BUS_CRASH_LOOP_ALERT.txt` marker file described above (survives both
Redis and Postgres being down, which cron's own MTA-dependent mail delivery
does not) plus this exit code (1 = crash loop, 2/3 = watchdog itself
failed) -- a wrapper that needs to escalate further should key off those,
not off cron's mail-on-output behavior.

---

## 🧩 Interacting with the Bus

### CLI Access

```bash
make cli
```

Opens a shell into the broker for issuing commands or inspecting streams.

### Stream Inspection

Each Orion service publishes events to canonical stream keys:

| Stream              | Purpose                    |
| ------------------- | -------------------------- |
| `orion:evt:gateway` | internal mesh event stream |
| `orion:bus:out`     | outbound message bus       |

To preview messages:

```bash
make stream            # defaults to orion:evt:gateway
make stream STREAM=orion:bus:out
```

---

## 🧠 How It Fits the Mesh

* The **Bus** carries messages between higher-order services such as:

  * **Brain** (reasoning)
  * **RAG** (retrieval / context)
  * **Mirror** (memory / collapse logging)
  * **Meta** (tagging / semantic linking)
* Each node in the Orion mesh attaches to the same network and uses the same topic conventions.
* Replacing the current Redis backend with another protocol (e.g., NATS, Kafka) will not affect higher services—the Bus interface remains stable.

---

## Substrate transport traces

Optional `bus-observer` sidecar emits bounded periodic `GrammarEventV1` rollups on `orion:grammar:event` when `PUBLISH_ORION_BUS_GRAMMAR=true` (default **off**).

Observes transport health (`PING`), stream depth (`XLEN`), backpressure thresholds, and configured streams missing from `orion/bus/channels.yaml`. Does **not** emit per-message traces.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `PUBLISH_ORION_BUS_GRAMMAR` | `false` | Publish grammar traces to bus |
| `BUS_OBSERVER_STREAMS` | `orion:evt:gateway,orion:bus:out` | Streams to sample (not `orion:grammar:event` by default) |
| `BUS_OBSERVER_POLL_INTERVAL_SEC` | `10` | Rollup interval |

Smoke: `../../scripts/smoke_orion_bus_substrate_trace.sh`

Layer 3 `bus_transport_reducer` is deferred — see `LAYER_PIPELINE_PLAN.md`.

---

## 🧊 Cold standby (`bus-core-standby`)

`bus-core` is a single `redis:7-alpine` instance — a real single point of failure. A
passive, cold-standby target lives in a separate, node-scoped compose file:
`docker-compose.atlas-standby.yml`, deployed on the `atlas` mesh node (see
`config/biometrics/node_catalog.yaml` — `atlas` is `expected_online: true`; `circe`
is `expected_online: false` and was ruled out for that reason).

This is **not** automatic failover, **not** leader election, and **not** wired into
`ORION_BUS_URL` or any client. It exists so a human has a real place to restore AOF
data to and cut traffic over manually if `bus-core` is down for an extended period.
See `docs/notes/2026-07-16-bus-core-standby-cutover-runbook.md` for the full manual
cutover/cutback procedure.

```bash
# On the atlas host, from its own checkout's repo root:
cp services/orion-bus/.env_example services/orion-bus/.env.atlas-standby

# PROJECT/TELEMETRY_ROOT/NODE_NAME are root-level vars (root .env_example,
# not services/orion-bus/.env_example) -- confirm atlas's own root .env
# already has real values for these before deploying:
grep -E "^(PROJECT|NODE_NAME|TELEMETRY_ROOT)=" .env

# Chain root .env FIRST, then the service env file (AGENTS.md section 8
# pattern) -- a single --env-file here leaves PROJECT/TELEMETRY_ROOT empty
# and silently binds the data volume at the filesystem root instead of
# under /mnt/telemetry/.
docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env.atlas-standby \
  -f services/orion-bus/docker-compose.atlas-standby.yml \
  up -d
```

This follows the same node-scoped separate-compose-file convention as
`services/orion-llamacpp-host/docker-compose.atlas-workers.yml` rather than a
profile inside the primary `docker-compose.yml`, since the standby runs on
different physical hardware, not alongside `bus-core` on the same host.

---

## 🛠️ Future Roadmap

* Transition from Redis Streams to a custom event-bus abstraction.
* Introduce schema validation for inter-service messages.
* Enable persistent stream snapshots for replay and debugging.
* Integrate distributed tracing hooks for Mesh Observability.

---

### TL;DR

> The Orion Bus is not just storage—it’s the **connective tissue** that gives the Mesh a collective nervous system.
