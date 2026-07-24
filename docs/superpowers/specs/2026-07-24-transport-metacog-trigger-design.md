# Transport metacog trigger — design (not implemented)

Status: **design mode, not implemented.** A new `trigger_kind` touches the metacog/collapse-mirror
cognition loop, which CLAUDE.md §0A requires explicit proposal mode for before implementation. This
document proposes; it does not build.

## Arsonist summary

Juniper asked for a new metacog trigger kind, analogous to `chat_turn`, for "transport, based on
grammar." Tracing what that maps onto in the live repo surfaced a real conflict worth stating before
any schema gets designed:

**The only existing grammar-event source for "transport" is built on a signal family already found
narrow/misleading, and partially superseded this session.** `config/substrate-lattice/
grammar_producer_registry.v1.yaml` registers exactly one live "transport" lane producer
(`orion-bus`, `trace_prefixes: ["bus.transport:"]`, `trusted_channels: [bus_health, transport_pressure,
contract_pressure, catalog_drift_pressure, observer_failure_pressure, delivery_confidence]`). The only
code that emits those grammar events is `orion/substrate/transport_loop/` (`extract.py`'s
`compute_transport_pressures`, semantic role `bus_health_observed`). `docs/superpowers/specs/
2026-07-23-transport-domain-rpc-health-redesign.md` (this session, confirmed via live introspection)
found that this entire family derives from `BUS_OBSERVER_STREAMS`, which is configured to exactly one
real producer's Redis Stream (`orion:stream:world_pulse:run:result`, a periodic news-digest job) —
fully healthy, fully consumed, `XLEN` is lifetime count not backlog depth. Everything else on the bus
is Pub/Sub with no stream/backlog concept at all. **A `transport` metacog trigger built on
`bus.transport:*` grammar events today would mean "fires when the world_pulse job's queue looks
unhealthy," not "fires when the bus/transport layer is actually unhealthy"** — the exact
degenerate-signal shape CLAUDE.md's metric-quality-gate step 4 (live-data sanity check) exists to
catch, already caught once this session for the old family's sibling metrics.

**A better foundation now exists, freshly built and live-verified this session.**
`docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` (PR #1313 + #1315,
merged and live-verified in the last few hours) ships `RpcHealthSnapshotV1` — real
`OrionBusAsync.rpc_request()` success/timeout/latency counts, drained from `orion-cortex-exec` and
`orion-cortex-orch`, confirmed live with real nonzero `success_count`/`timeout_count`/`p95` under real
traffic. This is genuinely cross-service (37+ call sites feed the one shared client), not one
unrelated queue standing in for "the bus." It is **not** a `GrammarEventV1` today — it publishes its
own `RpcHealthSnapshotV1` schema directly on `orion:rpc_health:snapshot`, consumed by
`orion-signal-gateway`'s organ pipeline (`OrionSignalV1`), a different pipeline from
`MetacogTriggerV1`/`orion_metacog` entirely.

**A second, separate honest transport-reliability signal already exists and is grammar-based**:
`chat_turn`'s own gate (`services/orion-equilibrium-service/app/chat_turn_metacog_gate.py`) already
treats a governor-RPC-never-returned condition as terminal evidence via a real `GrammarEventV1`
(`semantic_role="exec_turn_timeout"`, Patch B / PR #1287) and a stance-react timeout
(`stance_react_timeout`). These are real, live, honest grammar events — but they are scoped to one
turn's harness call, not a standing transport-health signal, and they're already consumed inside
`chat_turn`, not a separate trigger.

So "transport based on grammar" as literally stated would build on the flawed lane. The honest options
are named in Missing Question 1 below — this doc does not pick one without Juniper's steer, per the
"measure before minting" discipline already used for `rpc_health`.

## Current architecture

- **`MetacogTriggerV1`/`orion_metacog` pipeline** (what a new `trigger_kind` would join):
  `orion-equilibrium-service` evaluates gate conditions per trigger kind (`baseline`, `dense`, `manual`,
  `pulse`, `relational`, `llm_surface_instability`, `telemetry_anomaly`, `chat_turn` — full list in
  `orion/schemas/telemetry/metacog_trigger.py`'s `trigger_kind` docstring), publishes
  `CHANNEL_EQUILIBRIUM_METACOG_TRIGGER`, `orion-cortex-exec` drafts a `CollapseMirrorEntryV2` via LLM
  (`_fallback_metacog_draft`/`MetacogDraftService`), `orion-sql-writer` persists to `orion_metacog`.
  `chat_turn` (this session, PR #1291/#1298/#1306/#1309, live-verified) is the closest and most recent
  precedent for how a new kind gets built: `orion/schemas/telemetry/metacog_trigger.py` docstring entry,
  a `<kind>_metacog_gate.py` module in `orion-equilibrium-service/app/` (correlator + terminal-evidence
  check + gate-condition evaluator + trigger builder), a `_run()` dispatch branch subscribing the real
  upstream channel(s), its own cooldown lane (`EQUILIBRIUM_METACOG_<KIND>_COOLDOWN_SEC`, separate from
  the shared one — a real bug fixed this session when `chat_turn` initially shared the global cooldown).
- **`bus.transport:*` grammar lane** (the literal "grammar" substrate the ask referenced):
  `orion/substrate/transport_loop/extract.py`/`reducer.py`/`pipeline.py`. Reads `GrammarEventV1`s with
  `trace_id` matching `bus.transport:*`, semantic role `bus_health_observed`, computes
  `bus_health`/`transport_pressure`/`stream_depth_pressure`/`backpressure` from
  `BUS_OBSERVER_STREAMS`-configured stream introspection. Feeds `orion-substrate-runtime`'s
  `_transport_tick()` (5-domain prediction-error pattern) and `substrate_transport_bus_projection`.
  Separately, `Bus Transport Substrate Proof Ladder` (`docs/transport_substrate_proof_ladder.md`, PR
  #648, pre-existing, unrelated to metacog) also reads this same grammar lane for its own Layers 1-11
  self-modeling ladder (`transport_bus_reducer`, `capability:transport` field vector entry, etc.) —
  **two different consumers of the same narrow-scope grammar data**, neither the metacog pipeline.
- **`rpc_health` organ pipeline** (the better-scoped alternative): `orion/core/bus/rpc_health.py`
  (in-process aggregator, PR #1303) → `orion/core/bus/rpc_health_publish.py` (periodic drain, PR #1313)
  → `orion:rpc_health:snapshot` → `orion-signal-gateway`'s `RpcHealthAdapter` →
  `OrionSignalV1(organ_id="rpc_health_cortex_exec"/"rpc_health_cortex_orch")`. Shadow-only, not wired
  into any live consumer yet, per that spec's own non-goals — this doc's proposal would be the first
  consumer if Option A below is chosen.
- **`chat_turn`'s existing timeout grammar events** (the other honest grammar substrate): harness
  governor publishes real lifecycle `GrammarEventV1`s (`harness_lifecycle_grammar_published`,
  observed roles include `exec_result_assembled`/`exec_result_emitted` in this session's live logs) and
  a governor-RPC-timeout role (`exec_turn_timeout`, Patch B/PR #1287) consumed today only by
  `chat_turn_metacog_gate.py`.

## Missing questions

1. **Which substrate does "transport" actually mean, concretely?** Three real, honest options exist,
   not yet chosen:
   - **(A) Ride `rpc_health`.** A `transport` trigger correlates `RpcHealthSnapshotV1` windows
     (already real, live-verified) — fires on e.g. `timeout_count > 0` or `success_count == 0 and
     window has real prior traffic` (degraded-but-not-obviously-flagged) or a p95 latency spike. Not
     grammar-event-based at all; would need its own subscription to `orion:rpc_health:snapshot`
     directly, parallel to how `chat_turn` subscribes `orion:thought:artifact`/`orion:harness:run:artifact`.
   - **(B) Fix the grammar lane first, then ride it.** `bus.transport:*`'s narrow-scope problem is a
     pre-existing, separately-tracked issue (`docs/superpowers/specs/2026-07-23-transport-domain-
     rpc-health-redesign.md` Missing Question 5, explicitly deferred: rename honestly vs. deprecate).
     Building a metacog trigger on it now means building on a signal this session already flagged as
     misleading. Not recommended without that fix landing first.
   - **(C) A genuinely new grammar substrate: cross-service RPC failure/timeout events during real
     turns**, generalizing `chat_turn`'s own `exec_turn_timeout`/`stance_react_timeout` pattern beyond
     one turn's harness call — e.g. any `GrammarEventV1` role indicating an RPC that didn't get a
     reply within its configured timeout, across any of the 37+ `rpc_request()` call sites, not just
     the two harness-adjacent ones `chat_turn` already watches. This doesn't exist as a distinct
     grammar-emitting mechanism yet; would be new producer work (a grammar-emit call inside
     `rpc_request()`'s existing timeout branch, alongside the already-there `RpcHealthAggregator.
     record_timeout()`), overlapping with the `rpc_health` scope but expressed as discrete events
     instead of periodic snapshots.

   **Recommendation, not a decision**: (A) is the thin seam — it reuses real, already-verified data
   with zero new producer code, at the cost of not literally being "grammar-based." (C) is the closest
   literal match to "transport based on grammar" but is new producer work needing its own
   metric-quality-gate pass (steps 1-4) before any schema gets designed, mirroring exactly the
   discipline `rpc_health` itself followed. (B) is explicitly not recommended right now.

2. **What gate conditions would actually fire?** Depends entirely on Question 1's answer. If (A):
   plausible conditions mirror `chat_turn`'s shape — `timeout_count > 0` (a real RPC never got a
   reply), `success_latency_ms_p95` above a threshold (needs a real baseline first, not a guessed
   default — `scripts/analysis/measure_rpc_health_baseline.py` already exists from the `rpc_health`
   work and could be rerun for this), `success_count == 0` in a window with real prior traffic
   (silent-degradation case). None of these have real threshold data yet.

3. **Correlation scope: per-service, or bus-wide?** `chat_turn` correlates by `correlation_id` (one
   trigger per turn). A transport trigger has no natural per-turn correlation key — it's a
   time-windowed aggregate (`RpcHealthSnapshotV1`'s `window_start`/`window_end`), closer in shape to
   `telemetry_anomaly`'s periodic-tick pattern than to `chat_turn`'s per-correlation-id pattern. Needs
   its own design, not a direct copy of either.

4. **Volume/cadence**: `rpc_health` already publishes every 30s per producer service (2 services
   today). A trigger evaluated on every publish would be a periodic-tick shape; needs its own cooldown
   lane regardless (same lesson as `chat_turn`'s cooldown-sharing bug, fixed this session).

5. **Real threshold data**: none of steps 2-4 in CLAUDE.md's metric-quality-gate have been run for
   *this* proposed trigger yet — this doc is step 0 (design), not step 4 (live-data sanity check).

## Proposed schema / API changes

Deliberately not fixed here — depends on Question 1's answer, which needs Juniper's steer, and on real
threshold data from Question 2/5, which doesn't exist yet. Sketched only for the recommended-lean
option (A), directional:

- `orion/schemas/telemetry/metacog_trigger.py`: add `transport` to the `trigger_kind` docstring.
- New `services/orion-equilibrium-service/app/transport_metacog_gate.py`, mirroring
  `chat_turn_metacog_gate.py`'s shape: gate-condition evaluator over an `RpcHealthSnapshotV1` payload
  (not a correlator — no per-turn correlation key, see Question 3), `build_transport_metacog_trigger()`.
- `orion-equilibrium-service/app/service.py`: new subscription to `orion:rpc_health:snapshot`, new
  dispatch branch, new `EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE` flag (default off) +
  `EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC` (separate lane from day one, not retrofitted).
- `orion/bus/channels.yaml`: no new channel — reuses `orion:rpc_health:snapshot`, but add
  `orion-equilibrium-service` to that channel's `consumer_services` list.

## Files likely to touch (at implementation time, not this patch)

- `orion/schemas/telemetry/metacog_trigger.py` — docstring only.
- `services/orion-equilibrium-service/app/transport_metacog_gate.py` (new).
- `services/orion-equilibrium-service/app/service.py` — subscription, dispatch, cooldown lane.
- `services/orion-equilibrium-service/.env_example`, `docker-compose.yml`, `README.md`.
- `orion/bus/channels.yaml` — consumer list update only.
- Tests mirroring `test_chat_turn_metacog_gate.py`'s structure.
- If Option A: no other services touched (reuses existing `rpc_health` producers as-is).
- If Option C: `orion/core/bus/async_service.py` (new grammar-emit call in the timeout branch),
  `orion/bus/channels.yaml` (new channel), plus everything above.

## Non-goals

- Not deciding Question 1 (A vs. B vs. C) — this doc surfaces the tradeoff, doesn't resolve it.
- Not fixing the old `bus.transport:*`/`transport_pressure` family's narrow-scope problem — tracked
  separately in `2026-07-23-transport-domain-rpc-health-redesign.md` Missing Question 5.
- Not touching the pre-existing Layers 1-11 self-modeling ladder (`docs/transport_substrate_proof_
  ladder.md`, PR #648) — unrelated system, different consumer of the same grammar lane.
- Not implementing any code in this patch — design mode only, per CLAUDE.md §0A.
- Not guessing gate thresholds — Question 2/5 need real baseline data first, same discipline as every
  metacog trigger built this session.

## Acceptance checks (for a future implementation pass, not this doc)

- Same shape as `chat_turn`'s own acceptance bar: a real, non-degenerate `orion_metacog` row with
  `trigger_kind="transport"` observed from genuine gate-condition evidence (a real RPC timeout or
  latency spike), not a forced/threshold-lowered test.
- Separate cooldown lane from day one (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC`), not shared with
  the global one — this session's `chat_turn` cooldown bug should not repeat.
- If Option A: real live data confirms `rpc_health` snapshots show non-degenerate variance across a
  representative window before any threshold gets hardcoded (mirrors `rpc_health`'s own acceptance
  checks, already partially satisfied by this session's live verification).
- If Option C: the new grammar-emit's overhead on `rpc_request()`'s hot path benchmarked at
  effectively zero, same as `rpc_health`'s own PR #1299 gate.

## Recommended next patch

1. **Get Juniper's steer on Question 1** (A/B/C) before writing any code — this is a real fork, not a
   routine implementation detail, per CLAUDE.md's confusion-protocol guidance.
2. If (A): the smallest real first patch is wiring `orion-equilibrium-service` to subscribe
   `orion:rpc_health:snapshot` and log-only (no trigger publish yet) for a real observation window,
   mirroring the `measure_rpc_health_baseline.py` discipline — see real threshold data before
   hardcoding gate conditions.
3. If (C): start with the grammar-emit instrumentation inside `rpc_request()`'s timeout branch alone,
   benchmarked, before any equilibrium-side consumer exists — same sequencing this whole redesign has
   used throughout.

## Related work

- `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` — parent metacog-trigger
  taxonomy design; `chat_turn` (Implemented 2026-07-23 section) is the direct precedent this doc
  follows the shape of.
- `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` +
  `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` — the `rpc_health`
  work (PR #1290, #1299, #1303, #1313, #1315, all merged and live-verified this session) that this doc
  leans on as the better-scoped alternative to the old grammar lane.
- `docs/transport_substrate_proof_ladder.md` (PR #648) — the pre-existing, separate consumer of the
  same `bus.transport:*` grammar lane; not this doc's concern, named only to avoid confusion between
  the two.
