# Service heartbeat standardization + node telemetry (docker/disk/iLO) — design

Status: design only, nothing implemented. Requested by Juniper as a side quest ahead of building
docker-downtime measurement, disk-capacity monitoring, and hardware telemetry — the working
theory being that a standard bus-native heartbeat needs to exist across `/services` first.

## Arsonist summary

The premise "we don't have a bus-native heartbeat pattern" is **half true**. The pattern already
exists and is live — `BaseChassis` in `orion/core/bus/bus_service_chassis.py` publishes a
`SystemHealthV1` envelope to `orion:system:health` every `heartbeat_interval_sec` (default 10s),
and there's already a real consumer (`orion-equilibrium-service`, container `orion-athena-equilibrium`,
confirmed `Up 19 minutes` right now) that turns those heartbeats into `down_for_ms`/`uptime_pct`/
`distress_score`/`zen_score` per service. This is not a proposal — it's running in production
today. What's real and missing is **adoption**, not the pattern itself:

- **25 of 84 services** actually import `BaseChassis` and publish real heartbeats.
- **59 services don't** — suspiciously close to Juniper's "60 services" estimate, which suggests
  the instinct here is correct even though the stated premise ("no pattern exists") wasn't.
- Of the 25 already publishing, the live equilibrium-service is only configured to *track*
  4 of them: `EQUILIBRIUM_EXPECTED_SERVICES=recall,state-journaler,sql-writer,vector-writer`
  (confirmed in both `.env` and `.env_example`). The other 21 services' heartbeats are being
  published onto the bus and going nowhere. This is a pure config change, zero code, and it's
  the single cheapest win available before any of the rest of this spec is built.
- Separately, **54 of 84 services** have *some* HTTP `/health` route (confirmed by grepping real
  route decorators, not just string mentions — spot-checked `orion-hub`, `orion-signal-gateway`,
  `orion-recall`, all real `@app.get("/health")`/`@router.get("/health")` handlers). This is a
  genuinely separate, parallel mechanism from the bus heartbeat — pull-based HTTP liveness vs.
  push-based bus telemetry — confirming Juniper's instinct that these are two independent tracks,
  not one.
- **Nothing today consumes `EquilibriumSnapshotV1`'s `distress_score`/`zen_score`/`down_for_ms`.**
  The `*_metacog_gate.py` files already living in `orion-equilibrium-service` (`chat_turn_metacog_gate.py`,
  `telemetry_anomaly_metacog_gate.py`, `transport_metacog_gate.py`, `substrate_metacog_gate.py`,
  `repair_pressure_metacog_gate.py`) are a *different* metacog-gating family — channel/signal
  quality gates, unrelated to per-service uptime. "Downstream consumers like metacog can use it"
  is aspirational, not wired. Same for organ-signal consumers: `orion-signal-gateway` and
  `orion-substrate-organs` don't read `orion:system:health` or the equilibrium snapshot channel
  today. Hub's `organ-signals-graph-ui.js` has exactly one line referencing equilibrium —
  `equilibrium: "infra"`, a taxonomy label with no data binding behind it. That's the textbook
  shape of a keyword-cathedral fragment per this repo's own CLAUDE.md §0A: a name with no
  producer/consumer/reducer attached yet.
- **Equilibrium's output is live/broadcast-only — no persisted history.** `service.py` publishes
  to a bus channel and nowhere else (no `sql_writer`/`INSERT INTO` in that file). So even once
  `EQUILIBRIUM_EXPECTED_SERVICES` is expanded, "downtime" as something you can query after the
  fact (a report, a dashboard trend line) doesn't exist yet — only a live rolling-window snapshot
  that resets whenever the equilibrium-service process itself restarts. Whether that matters
  depends on what Juniper actually wants from "measure docker downtime" — a live gauge vs. a
  historical record are different asks with different costs.
- **Docker sludge is real and current, not hypothetical**: `/mnt/docker` on athena is at 85%
  (68G free of 469G). `docker system df`: 8,233 images, 81 active, 138.4GB reclaimable.
- **Cross-node visibility (circe, atlas) is currently blocked.** Direct SSH from this session
  failed (`Permission denied (publickey,password)` to atlas, host-key issue on circe) — no
  established cross-node SSH convention exists anywhere in `scripts/` either (grepped, zero
  hits). But the bus itself already reaches every node — Tailscale confirms `athena`, `atlas`
  (`192.168.1.17`), `circe` (`192.168.1.22`) are all directly reachable, and a live subscribe to
  `orion:system:health` right now shows heartbeats arriving from non-athena sources already (e.g.
  `vision-edge`, a camera-adjacent service). The bus is proven cross-node transport that already
  exists; SSH is not. That materially changes the right design for node-level disk telemetry —
  see below.
- **Live anomaly spotted while subscribing (not investigated further, flagging only):**
  `vision-edge` is heartbeating at `heartbeat_interval_sec: 0.0667` (~15/sec) — matching its
  camera's `fps: 15.0` in the same payload. That looks like a heartbeat fired per video frame
  rather than a real periodic system pulse, which is either a meaningful bug (every camera
  instance spamming ~900 heartbeat envelopes/minute) or an intentional-but-undocumented
  repurposing of the field. Worth a look during the Phase 0 audit pass, not fixing now.
- **No iLO/BMC tooling installed** (`ilorest`, `ipmitool`, `redfish` python lib — all absent from
  athena). No documented iLO management IP/credentials found in the repo. The `prometheous`
  Tailscale peer Juniper flagged is confirmed a not-yet-stood-up dev cluster — disregarded per
  her direction, not treated as existing infra.

## Current architecture

**Bus-native heartbeat (exists, partially adopted):**
- `orion/schemas/telemetry/system_health.py` — `SystemHealthV1` (service, node, version,
  instance, boot_id, status: ok/degraded/down, last_seen_ts, heartbeat_interval_sec, details:
  dict). Also defines `EquilibriumServiceState` (adds down_for_ms, uptime_pct per window,
  boot_id) and `EquilibriumSnapshotV1` (aggregate: distress_score, zen_score, expected_services,
  services: list[EquilibriumServiceState]). Also `BusConsumerReadinessV1` — a third, separate
  readiness concept (bus_consumer_ready, subscriber_count, heartbeat_fresh, rpc_smoke_ok,
  dependency_status), implemented in `orion/bus/consumer_readiness.py`.
- `orion/core/bus/bus_service_chassis.py` — `BaseChassis`/`ChassisConfig`. Owns bus
  connect/disconnect, SIGTERM shutdown, a `_heartbeat_loop()` publishing `SystemHealthV1` to
  `cfg.health_channel` (default `orion:system:health`) every `cfg.heartbeat_interval_sec`
  (default 10s), and exception-to-`orion:system:error` wrapping. 25 services currently construct
  a `BaseChassis`/subclass: orion-actions, orion-agent-council, orion-biometrics,
  orion-chat-memory, orion-collapse-mirror, orion-cortex-exec, orion-cortex-orch, orion-dream,
  orion-equilibrium-service, orion-llama-cola-host, orion-llamacpp-neural-host,
  orion-llm-gateway, orion-memory-consolidation, orion-memory-crystallizer, orion-meta-tags,
  orion-rdf-writer, orion-recall, orion-signal-gateway, orion-spark-introspector,
  orion-sql-writer, orion-state-journaler, orion-state-service, orion-substrate-telemetry,
  orion-vector-host, orion-vector-writer.
- `orion/bus/channels.yaml` line 43 — `orion:system:health` is a registered channel (so this
  isn't an unregistered/undocumented event shape).
- `services/orion-equilibrium-service/app/service.py` — subscribes to `orion:system:health`,
  computes `uptime_pct`/`down_for_ms` per `settings.windows_sec`, publishes
  `EquilibriumSnapshotV1` to `settings.channel_equilibrium_snapshot` on a timer. Only tracks
  whatever's in `settings.expected_services()` (currently 4 names, hardcoded in `.env`/
  `.env_example`, or optionally a YAML file path — `EQUILIBRIUM_EXPECTED_SERVICES_PATH`, unset
  today). No Postgres write path in this file.
- **Separate, unrelated to service-uptime**: `orion/core/bus/rpc_health_publish.py` +
  `orion/schemas/telemetry/rpc_health.py` — a per-service RPC-call-latency snapshot on its own
  channel (`orion:rpc_health:snapshot`), built in
  `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md`. Don't confuse
  this with `orion:system:health` — different question (RPC call quality vs. process liveness).

**HTTP `/health` routes (exists, differently adopted, currently unrelated to the bus heartbeat):**
54 services have a real `/health` route handler (spot-checked, not string-matched). No evidence
found tying any of these routes to `BaseChassis`/`SystemHealthV1` — they're independently
hand-rolled per service, format/content unverified across all 54 (didn't audit response shape
for this pass — see Missing questions).

**Infra-layer restart/crash-loop detection (exists, container-level, orthogonal to both above):**
`scripts/bus_core_health_watchdog.py` — polls `docker inspect` directly (RestartCount,
unhealthy-streak thresholds), no bus involvement, writes an alert-marker file. This measures "is
the container alive" (infra layer) vs. `SystemHealthV1` measuring "did the app process itself
report a pulse" (app layer) — a hung-but-technically-running process would be caught by one and
missed by the other. Both are worth keeping; they answer different questions.

**Disk/docker capacity — no monitoring exists anywhere in the repo.** Confirmed via grep across
`scripts/` for `docker system prune`, `disk_usage`, `shutil.disk_usage`: nothing. Live numbers
gathered by hand this session (athena only): `/mnt/docker` 85% full (377G/469G used, 68G free);
`docker system df` shows 8,233 images / 81 active / 138.4GB reclaimable, 227.9GB build cache.
`/mnt/telemetry` 20%, `/mnt/scripts` 13%, `/mnt/postgres` 14%, `/mnt/graphdb` 1% — all fine on
athena today, but circe/atlas are unmeasured (SSH blocked, see below).

**iLO/BMC — nothing exists.** No tooling installed, no documented management IP, no code
references outside an unrelated string match in `orion-recall`'s README (false positive, not a
real hit).

## Missing questions

1. **Pilot-5 selection criteria.** Juniper said prove out with 5 services before going wider —
   which 5? Candidates worth considering: pick from the 59 non-adopted services based on (a)
   highest operational blast radius if silently down (e.g. `orion-hub`, `orion-harness-governor`,
   `orion-cortex-gateway` — currently no bus heartbeat despite being critical-path), or (b)
   services already in `EQUILIBRIUM_EXPECTED_SERVICES`'s natural next-tier (things adjacent to
   `recall`/`sql-writer` in the data path), or (c) simplest/lowest-risk first to validate the
   rollout mechanics before touching anything load-bearing. This needs Juniper's call, not an
   assumed default.
2. **Is "unlock #1" the live gauge, or a queryable history?** Expanding
   `EQUILIBRIUM_EXPECTED_SERVICES` to the already-heartbeating 21 services is a same-day,
   zero-code win for a *live* uptime/distress view. But equilibrium publishes to the bus only —
   no persisted history. If Juniper wants "how much was service X down last week," that's a real
   new patch (a consumer writing `EquilibriumServiceState`/downtime events to Postgres via
   `sql-writer`, following the same bus→sql-writer routing pattern already established elsewhere
   in this repo — see [[project_causal_geometry_v1_bus_routing_correction]] as the precedent for
   *not* writing Postgres directly from a producer). Which one is actually wanted changes the
   scope significantly.
3. **What should concretely consume `distress_score`/`zen_score`/per-service `down_for_ms`?**
   Juniper named "metacog and organ signals" as the intended consumers, but neither currently
   reads this channel, and the existing `*_metacog_gate.py` family in equilibrium-service is a
   different (channel-quality, not service-uptime) concept. Does this mean: (a) a new metacog
   gate alongside the existing 5, reacting to service distress specifically, and/or (b) wiring
   `orion-signal-gateway`/`orion-substrate-organs` to actually consume `orion:system:health` or
   the equilibrium snapshot for the first time? Needs a named target, not just "downstream
   consumers" left abstract — that's exactly the kind of unwired-name risk CLAUDE.md's
   keyword-cathedral gate exists to catch, and Hub's `equilibrium: "infra"` label is already a
   small live instance of it.
4. **Cross-node disk telemetry: bus-piggyback vs. SSH?** No cross-node SSH convention exists
   today. Two real options: (a) fix SSH access to atlas/circe (keys, host-key trust) and run a
   thin poller from athena, matching the "downtime" script's docker-events-polling shape; or (b)
   piggyback per-node disk usage onto each node's *existing* heartbeat `details: dict` field
   (`SystemHealthV1.details` is already a free-form dict, and the bus already proven to cross
   nodes live). (b) needs zero new network trust and reuses infra this spec is already
   standardizing — but it means picking (or building) at least one chassis-adopted service per
   node to carry it, and disk usage is node-level, not per-service, so reporting it from every
   chassis service on a node would be redundant. Needs Juniper's preference and a decision on
   which single per-node "carrier" service (if one exists on circe/atlas already) vs. a new thin
   per-node publisher.
5. **iLO reachability, concretely.** Does Juniper know the iLO management IP/VLAN for athena and
   atlas, and does she have credentials already, or does that need first-time setup research
   too? RedFish (HTTPS+JSON, no vendor tool required — should work with plain `requests` if the
   management interface is reachable) is the standard modern approach for HPE iLO 4/5 over
   `ilorest`/`hponcfg`, but "is the management NIC even on a network this host can reach" is
   unverified and can't be answered from code alone.
6. **`vision-edge`'s per-frame heartbeat rate** — confirm bug vs. intentional before folding
   vision-* services into any rollout tier; if it's a bug, standardizing on top of it just
   propagates the noise.
7. **HTTP `/health` route standardization — what does "standard" mean here?** Content/shape
   wasn't audited across all 54 (this pass only confirmed the route exists on each, not that they
   return comparable payloads). Worth deciding if this track needs a shared response schema
   (mirroring `BusConsumerReadinessV1`, which already exists for exactly this — `ok`,
   `http_alive`, `bus_consumer_ready`, etc.) or just presence-per-service is the near-term bar.

## Proposed schema / API changes

None yet — this pass is find-before-build. Once Juniper answers the missing questions above,
likely candidates (not committed):
- `EQUILIBRIUM_EXPECTED_SERVICES` env expansion in `services/orion-equilibrium-service/.env`/
  `.env_example` (config-only, no schema change) — the cheapest available win.
- If historical downtime is wanted: a new bus→sql-writer consumer path for
  `EquilibriumServiceState`/`EquilibriumSnapshotV1`, following the established
  bus-routing-not-direct-write pattern.
- If cross-node disk goes the bus-piggyback route: no new schema needed — `SystemHealthV1.details`
  already accepts arbitrary dict content; would just need a documented `details` key convention
  (e.g. `details.disk_usage_pct` per mount) rather than a new top-level field.
- iLO: likely a new `orion/schemas/telemetry/hardware_health.py`-shaped schema eventually, but
  not before reachability is confirmed — no point designing a payload shape for data we don't
  yet know we can fetch.

## Files likely to touch (once scoped)

- `services/<pilot-5>/app/main.py` (or equivalent) — wire `BaseChassis`, mirroring one of the 25
  already-adopted services as a template (`orion-signal-gateway` or `orion-recall` are good
  reference implementations — both real, both already reviewed-in-production).
- `services/orion-equilibrium-service/.env`, `.env_example` — expand `EQUILIBRIUM_EXPECTED_SERVICES`.
- `services/orion-equilibrium-service/app/service.py` — only if historical persistence or new
  metacog-gate wiring is in scope.
- `services/orion-signal-gateway/app/` or `services/orion-substrate-organs/app/` — only once a
  concrete organ-signal consumer is named (missing question #3).
- `services/orion-hub/static/js/organ-signals-graph-ui.js` — only once equilibrium data is
  actually wired to something real, replacing the current unwired `"infra"` label.
- `scripts/bus_core_health_watchdog.py` — reference pattern for any new disk/docker watchdog,
  not necessarily a file to edit.
- `orion/bus/channels.yaml` — only if a genuinely new channel is needed (unlikely for the
  heartbeat rollout itself, since `orion:system:health` already exists and is registered).

## Non-goals

- Not rebuilding the heartbeat pattern from scratch — it exists, this is an adoption + wiring
  gap, not a missing-pattern gap.
- Not rolling out to all 59 remaining services in one patch — 5-service pilot first, per
  Juniper's explicit instruction.
- Not building cross-node SSH infrastructure unless Juniper picks that option over the
  bus-piggyback approach in missing question #4.
- Not building any iLO integration in this phase — research/reachability-confirmation only,
  explicitly deferred per Juniper's "bake in that research" framing, not "build it."
- Not touching `vision-edge`'s heartbeat-rate anomaly in this pass — flagged, not fixed.
- Not inventing a new "downtime" concept parallel to `EquilibriumSnapshotV1` — extend/consume the
  existing one.

## Acceptance checks

- Pilot: 5 named services publish real `SystemHealthV1` heartbeats to `orion:system:health`,
  confirmed live (subscribe and see real envelopes, not just code review).
- `EQUILIBRIUM_EXPECTED_SERVICES` expansion (if approved as an immediate first step, independent
  of the pilot): live `EquilibriumSnapshotV1` broadcasts show `uptime_pct`/`down_for_ms` for the
  newly-added services, verified via a live subscribe, not just config presence.
- Whichever consumer gets named for missing question #3 actually reads real `distress_score`/
  `down_for_ms` data in a live trace — not a schema-valid-but-unused wiring (per CLAUDE.md's
  no-empty-shell-cognition rule, applies equally to infra telemetry).
- Disk/docker: whichever mechanism is chosen for cross-node visibility (SSH or bus-piggyback)
  produces real numbers from circe and atlas, not just athena — the current gap is exactly that
  athena-only data was all that was reachable this session.
- iLO: acceptance for this phase is answering "is the management interface reachable from
  athena/atlas at all" — not a working telemetry pull.

## Recommended next patch

Two independent, cheap, sequenceable pieces, both gated on Juniper's answers to the missing
questions above — not proposing to build either without her picking a direction first:

1. **Immediate, near-zero-cost**: expand `EQUILIBRIUM_EXPECTED_SERVICES` to cover the 21
   already-heartbeating-but-untracked services. Pure config, no code, live-verifiable same day,
   and directly informs how much the pilot-5 rollout actually matters once Juniper sees real
   current coverage.
2. **Pilot-5 chassis adoption**: once Juniper names the 5 services (missing question #1), wire
   `BaseChassis` into each following `orion-signal-gateway`'s existing pattern as the template,
   live-verify each publishes real heartbeats, and use that as the proof point before scoping the
   remaining 54.

Both are independent of the disk/docker and iLO tracks, which stay blocked on missing questions
#4 and #5 respectively until Juniper weighs in.
