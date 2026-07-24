# RPC-health signal-gateway wiring — design (Step 3 of the transport-domain RPC-health redesign)

Design-only. No code in this patch. Follow-up to
`docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` (Step 1: worker-path
logging fix + real baseline measurement, PR #1290; Step 2: `RpcHealthAggregator` built and wired
into `OrionBusAsync.rpc_request()`'s 4 real outcome sites, PR #1303). This spec covers Step 3: how
a small number of real, high-volume services get their in-process RPC-health snapshots onto the
bus and into `orion-signal-gateway` — the same organ-adapter machinery that already ingests the 24
existing organs (biometrics, recall, cortex_exec, sql_writer, etc.) — instead of inventing a
parallel path through `orion-substrate-runtime`.

## Arsonist summary

Every real cross-service call in this codebase (`OrionBusAsync.rpc_request()`, 37+ files) now
keeps an accurate, bounded, in-memory tally of its own success/timeout outcomes
(`orion/core/bus/rpc_health.py`, PR #1303). Nothing drains it. `get_rpc_health_snapshot()` exists
and works, but no periodic caller exists — by design, per that PR's own docstring, until this real
consumption path is proven out rather than assumed.

The user asked whether we should "bite the bullet and wire into every relevant service" the way
organ signals were wired. The honest answer, checked against real code this turn, is: **not the
same bullet**. The 24 existing `orion/signals/adapters/*.py` organs work by **eavesdropping** —
`orion-signal-gateway` subscribes to channel patterns (`services/orion-signal-gateway/app/settings.py`
`ORGAN_CHANNELS`) that organs already publish to for their own normal operation; no organ was ever
modified to "report" to the gateway. RPC calls publish nothing about their own outcome today — only
a log line (pre-#1290: nothing at all on success) and, since #1303, an in-memory counter nobody
reads. There is no free ride here. Something has to newly publish.

The good news, found while writing this spec, not assumed going in: that publish step does not
need to be invented from scratch. Every service that already uses the shared `BaseChassis`/`Hunter`
chassis (`orion/core/bus/bus_service_chassis.py`) — which includes both `orion-cortex-exec` and
`orion-cortex-orch` — already runs a periodic `_heartbeat_loop()` (every `heartbeat_interval_sec`,
default 10s) that publishes `SystemHealthV1` to `orion:system:health` via `self.bus.publish(...)`.
That loop already has `self.bus` — an `OrionBusAsync` instance — in scope. Adding one more publish
call per tick, draining `self.bus.get_rpc_health_snapshot()`, rides an existing seam instead of
adding a new timer/task per service.

**The catch, found by tracing the actual RPC call sites rather than assuming the chassis's `self.bus`
is the one making the real calls:** in `cortex-exec`, `cortex-orch`, and 7 other services
(`chat-memory`, `actions`, `context-exec` ×2 files, `hub` ×2 files, `vision-council`), the real
`rpc_request()` traffic does **not** go through `svc.bus` (the chassis instance running the
heartbeat loop). It goes through a separate instance created by `fork_rpc_client(svc.bus)`
(`orion/core/bus/rpc_fork.py`), stored as a module-level `_rpc_bus` and used by the actual
verb/workflow runtime. `OrionBusAsync.fork()` (confirmed by reading it directly,
`orion/core/bus/async_service.py:94-110`) constructs a brand-new `OrionBusAsync(...)` — meaning a
brand-new, independently-empty `RpcHealthAggregator` — because the *reason* `fork()` exists is to
give nested RPC callers their own Redis connection/subscriber, not because the health aggregator
specifically needs isolation. If the heartbeat loop naively snapshots `svc.bus`, it will report an
aggregator that never sees the service's real RPC traffic: a flat, always-empty signal that looks
identical to the exact kind of degenerate "measures nothing" problem this whole redesign exists to
kill. This must be resolved before wiring any of these 9 services — it is not a detail to fix later.

## Current architecture

- **Signal gateway ingestion**: `GatewayService.start()` (`services/orion-signal-gateway/app/service.py:51-72`)
  subscribes a `Hunter` to `settings.ORGAN_CHANNELS` — a fixed list of ~28 bus channel *patterns*
  (`orion:biometrics:*`, `orion:cortex:*`, `orion:exec:*`, etc., `services/orion-signal-gateway/app/settings.py:65-95`).
  Every matching envelope is handed to `SignalProcessor.handle_envelope()`, which runs it through
  `ORGAN_REGISTRY`-registered adapters (`orion/signals/adapters/*.py`, each implementing
  `can_handle(channel, payload)` / `adapt(...)`) to produce an `OrionSignalV1`.
- **Organ registry**: `orion/signals/registry.py`'s `ORGAN_REGISTRY: Dict[str, OrionOrganRegistryEntry]`
  — 24 entries today. Each entry names `organ_id`, `organ_class` (exogenous/endogenous/hybrid),
  `service`, `signal_kinds`, `canonical_dimensions`, `causal_parent_organs`, `bus_channels` (the
  channels the gateway subscribes to *for that organ*), `notes`. Closest existing precedent for an
  infra/plumbing signal (not a cognition organ): `sql_writer`/`rdf_writer`/`vector_writer` — all
  `OrganClass.exogenous`, `signal_kinds=["persist"]`, no causal parents, publishing bare
  presence/latency receipts rather than semantic content.
- **RPC health today**: `RpcHealthAggregator` (`orion/core/bus/rpc_health.py`) lives inside each
  `OrionBusAsync` instance (`self._rpc_health` in `__init__`), fed synchronously by
  `record_success()`/`record_timeout()` at `rpc_request()`'s 4 real outcome sites. Drained via
  `get_rpc_health_snapshot()` → `RpcHealthSnapshot` (success/timeout counts, p50/p95/max success
  latency, max timeout elapsed, per-request-channel counts, a `truncated` flag). Per-process,
  per-`OrionBusAsync`-instance, **not shared across `fork()`** — confirmed by
  `test_fork_gets_an_independent_rpc_health_aggregator` (PR #1303).
- **Existing periodic-publish seam**: `BaseChassis._heartbeat_loop()`
  (`orion/core/bus/bus_service_chassis.py:120-149`) already runs on every service built on this
  chassis, already holds `self.bus`, already publishes on a fixed interval. Both `orion-cortex-exec`
  and `orion-cortex-orch` use this chassis (`app/main.py` in each).
  `HEARTBEAT_INTERVAL_SEC` already exists as a per-service settings key (e.g.
  `services/orion-signal-gateway/app/settings.py:47` — used by the gateway's own chassis instance,
  not by cortex-exec/cortex-orch's; each service has its own copy of this setting).
- **The fork split** (see Arsonist summary): in `cortex-exec`
  (`services/orion-cortex-exec/app/main.py:897-1008`) and `cortex-orch`
  (`services/orion-cortex-orch/app/main.py:581-684`), a module-level `_rpc_bus` — obtained via
  `fork_rpc_client(svc.bus)` — is what verb/workflow runtime code actually calls `rpc_request()` on.
  `svc.bus` (the chassis's own instance, running `_heartbeat_loop()`) is not the same object.
  Same `fork_rpc_client(svc.bus)` pattern repeats in `chat-memory`, `actions`, `context-exec` (2
  files), `hub` (2 files), `vision-council` — a real, repo-wide pattern, not specific to these two
  services.
- **Baseline measurement's own service list**: `scripts/analysis/measure_rpc_health_baseline.py`'s
  `DEFAULT_CONTAINERS` (chosen in Step 1 for real cross-service traffic, not arbitrarily) is
  `orion-athena-cortex-exec`, `orion-athena-cortex-orch`, `orion-athena-hub`, `orion-athena-actions`,
  `orion-athena-thought`, `orion-athena-spark-introspector`, `orion-athena-chat-memory`. Of these,
  every one *except* `thought` and `spark-introspector` appears in the `fork_rpc_client` list above
  — i.e. most of the services we already know carry real RPC volume are exactly the ones with the
  fork gotcha.

## Missing questions

1. **Fork-aggregator resolution, per service.** For each service picked in this step: does its
   heartbeat-loop code snapshot `svc.bus` (wrong, silently empty) or the actual `_rpc_bus` /
   fork child (right)? This is not one repo-wide answer — `thought` and `spark-introspector` may
   call `rpc_request()` directly on `svc.bus` with no fork at all, making them simpler starting
   points than cortex-exec/cortex-orch despite lower measured volume. Needs a per-service read,
   not an assumption, before implementation.
2. **Does `fork()` need to start sharing the health aggregator?** Two real options: (a) each
   service's health-report code explicitly holds a reference to whichever `OrionBusAsync` instance
   does its real RPC calls (works today, zero core changes, but is per-service boilerplate to get
   right); (b) `OrionBusAsync.fork()` grows an optional `share_rpc_health: bool` (or always shares —
   there's no correctness reason found so far for per-fork isolation of *this* piece of state,
   unlike `_pending_rpc`/`_rpc_subscribed` which must stay instance-local for reply routing to work).
   Option (b) is a core `orion/core/bus/async_service.py` change and needs its own review pass; not
   assumed safe here.
3. **Publish cadence.** Ride the existing `heartbeat_interval_sec` (~10s) or a separate, coarser
   interval? A snapshot with `truncated=True` at high call volume within a 10s window is plausible
   for cortex-exec — real measured cadence data (Step 1's baseline) should settle this, not a guess.
4. **New channel name and schema fields.** Deliberately not fixed here. Placeholder direction only,
   below — exact field list needs the same "look at real snapshots first" discipline used for every
   prior step in this redesign.
5. **`organ_class` for this new organ.** Leaning `exogenous` (infra/plumbing, no causal parent
   organs, same class as `sql_writer`/`rdf_writer`/`vector_writer`) rather than treating it as
   downstream of any cognition organ — RPC health measures transport reliability, not cognitive
   content. Worth a second look once real snapshot data exists.
6. **One organ_id covering multiple services, or one per service?** `sql_writer`/`rdf_writer` are
   each scoped to one service. RPC health is structurally the same measurement across every
   service that has it. A single `rpc_health` organ_id with `service` differentiated via the
   signal's own payload/dimensions (vs. per-service organ_ids like `rpc_health_cortex_exec`) is the
   current lean but not decided — affects `ORGAN_REGISTRY` shape and adapter `can_handle()` logic.
7. **Old `transport_pressure`/`bus_health` family's fate.** Still explicitly out of scope for this
   step too, per the parent spec's own deferral.

## Proposed schema / API changes (directional only, not final)

- New bus channel, e.g. `orion:rpc_health:snapshot` (exact name TBD against Q6) — added to
  `orion-signal-gateway`'s `ORGAN_CHANNELS` list (`services/orion-signal-gateway/app/settings.py`)
  as a new pattern, e.g. `"orion:rpc_health:*"`.
  - Must also be added to `orion/bus/channels.yaml` per this repo's bus contract rules — a new
    published channel is a contract change regardless of which consumer reads it.
- New registry entry in `orion/signals/registry.py`'s `ORGAN_REGISTRY`, modeled on `sql_writer`/
  `rdf_writer`/`vector_writer`:
  ```python
  "rpc_health": OrionOrganRegistryEntry(
      organ_id="rpc_health",
      organ_class=OrganClass.exogenous,
      service="<multiple — see Q7>",
      signal_kinds=["rpc_transport_health"],
      canonical_dimensions=["level", "confidence"],  # placeholder; real fields pending Q4
      causal_parent_organs=[],
      bus_channels=["orion:rpc_health:snapshot"],
      notes=["Drains OrionBusAsync.get_rpc_health_snapshot() on a periodic publish loop; "
             "see docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md."],
  ),
  ```
- New adapter `orion/signals/adapters/rpc_health.py` (mirrors `CortexOrchAdapter`'s shape:
  `can_handle()` matching the new channel, `adapt()` mapping `RpcHealthSnapshot`'s fields into
  `OrionSignalV1.dimensions`).
- Either: (a) a small per-service snippet added to each target service's existing heartbeat-adjacent
  code, publishing a new envelope alongside (not instead of) `SystemHealthV1`; or (b) if Q2 resolves
  toward shared aggregators, a chassis-level opt-in (`ChassisConfig` gains an `rpc_health_channel`
  field, `_heartbeat_loop()` publishes it automatically when set) — this second option is more
  invasive (touches every chassis consumer's config surface) and should only be taken if the
  per-service snippet proves annoying to repeat 2-3 times.

## Files likely to touch

- `orion/signals/registry.py` — new `ORGAN_REGISTRY` entry.
- `orion/signals/adapters/rpc_health.py` — new adapter (new file).
- `orion/signals/adapters/__init__.py` or wherever adapters are collected/registered — wire the new
  adapter in.
- `services/orion-signal-gateway/app/settings.py` — `ORGAN_CHANNELS` gains the new pattern;
  `services/orion-signal-gateway/.env_example` if the pattern becomes configurable.
- `orion/bus/channels.yaml` — register the new channel.
- Per chosen service (`app/main.py` in each, exact set pending Q1): the periodic-publish snippet,
  reading from whichever `OrionBusAsync` instance is confirmed to be the real RPC caller.
- `orion/core/bus/async_service.py` + `orion/core/bus/rpc_fork.py` — only if Q2 resolves toward
  shared aggregators across `fork()`.
- Tests: `orion/signals/adapters/tests/` (new adapter unit tests, following the existing
  per-adapter test convention), a gateway-level test proving the new pattern is actually subscribed
  and an envelope on it produces a signal end-to-end.
- `services/orion-signal-gateway/README.md` and each touched service's `README.md` — document the
  new publish/consume path per section 9/16 of this repo's CLAUDE.md.

## Non-goals

- Not deciding the exact schema field list or dimension mapping — deferred to real snapshot data,
  same discipline as every prior step.
- Not deciding publish cadence with certainty — directional lean toward the existing heartbeat
  interval, not committed.
- Not implementing a fix for `fork()`'s aggregator-sharing question in this spec — Q2 needs its own
  small design-or-implement decision, likely as the literal first commit of the follow-up patch.
- Not touching `orion-substrate-runtime`'s tick loop, `transport_pressure`, `bus_health`, or any of
  the 5 `orion/substrate/prediction_error.py` domain instruments — this is an additive new organ,
  not a replacement, and the parent spec's non-goal about the old family's fate still stands.
- Not wiring all 37+ `rpc_request()` call sites — 2-3 services only, chosen by real measured volume
  and fork-simplicity (see Recommended next patch), consistent with "measure before minting" over
  "wire everything because we can."
- Not building the harness's bespoke RPC path (`HarnessGovernorClient`) into this — confirmed
  earlier in this redesign to use its own long-poll mechanism, structurally outside
  `OrionBusAsync.rpc_request()` entirely.

## Acceptance checks

- A real running instance of each chosen service publishes at least one `orion:rpc_health:snapshot`
  envelope per heartbeat-interval window, verified via `docker compose logs` or a direct Redis
  `SUBSCRIBE`, not just "code compiles."
- The published snapshot's `success_count`/`timeout_count` are demonstrably nonzero after the
  service handles real traffic (not a flat 0 — the exact failure mode this whole redesign exists to
  avoid repeating).
- `orion-signal-gateway`'s logs show `Gateway subscribing to N channel patterns` with N incremented
  by exactly the new pattern(s) added, and a resulting `OrionSignalV1` with `organ_id="rpc_health"`
  is observable on `orion:signals:*` (or the gateway's own output channel) after a real publish.
- Unit test proves the new adapter's `can_handle()`/`adapt()` in isolation (no bus/Docker needed),
  matching every existing adapter's test convention.
- Whichever fork-resolution path is chosen (Q1/Q2), a test proves the snapshot read actually reads
  the instance handling real `rpc_request()` traffic for that service — not the chassis's idle
  `svc.bus`. This is the single most important check in this whole step: it is exactly the kind of
  claim that must not be taken on faith per this repo's "runtime truth beats config truth" rule.

## Recommended next patch

Start with 2 services, chosen for the clearest signal-to-effort ratio rather than raw volume alone:

1. **Resolve Q1/Q2 first, for `orion-cortex-exec` and `orion-cortex-orch` specifically** (highest
   measured real volume per Step 1's baseline, and the two the user's own framing centered on).
   This is a short, self-contained investigation: read each service's heartbeat/RPC-runtime wiring,
   confirm exactly which `OrionBusAsync` instance is live, and decide whether a per-service snippet
   (referencing `_rpc_bus` directly) is enough or whether `fork()` needs the aggregator-sharing
   change from Q2. Do this as its own small commit/PR before touching the gateway side at all — if
   it turns up something unexpected (a third bus instance, a service that reconnects and replaces
   `_rpc_bus` mid-run, etc.), that changes this spec's Files-to-touch list materially.
2. Once resolved, build the registry entry + adapter + channel wiring + per-service publish snippet
   for those two services only, following the acceptance checks above.
3. Leave `thought` and `spark-introspector` (present in the baseline's container list, likely
   simpler — no confirmed fork split found yet) as the natural next 1-2 services for a follow-up
   patch, not this one.

## Related work

- `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` — parent spec (Steps
  1-2), Step 3 pointer added there in this patch.
- `docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md` — the
  measurement-first spec that started this whole thread.
- PR #1290 (worker-path logging fix + baseline script), PR #1299 (gated investigation + overhead
  benchmark), PR #1303 (`RpcHealthAggregator` built and wired) — the concrete Step 1-2 work this
  step builds on.
