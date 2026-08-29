# Capability absence: making Orion notice when part of its body is gone

Status: design + phase 1 implemented
Date: 2026-08-29
Incident that prompted it: circe dark 2026-08-29 00:01:16Z -> 00:47:04Z (~45 min)

## Arsonist summary

Orion's entire local GPU inference capacity vanished for 45 minutes. Four independent
detectors saw it within 1-2 minutes and correctly attributed it. Nothing changed behavior,
nothing reached Juniper, and Orion kept dispatching work into a dead host and storing the
resulting error strings as its own reasoning.

The bug is not "no alert". It is that **every layer of this system models load and never
models presence.** Four places, same mistake:

1. `pressure_organ.py::invoke_biometrics_pressure()` resolves its subject node from the
   incoming event (`_node_id_from_trigger`, L41-47). A node that stops reporting is never
   evaluated. The staleness rule can only fire for a node that is currently reporting.
2a. **Found while building this patch, pre-existing:** `build_pressure_candidate_events`
   derived `trace_id` from `{node_id}:{ts}` only. Every rule firing for the same node in
   the same tick therefore shared one trace_id, `group_candidate_events_by_trace()` merged
   them into a single trace, and the reducer -- which takes one atom per trace -- silently
   dropped all but the first. Reproduced against **shipping rules only**: a node with
   `gpu` hint 0.7 and a prior active pressure fires Rule C (`node_pressure_reinforced`)
   and Rule E (`node_capability_impact`) together, and the capability impact was discarded.
   So `node_capability_impact` had two independent reasons to never appear: it could not be
   reached by absence, and when it *was* reached by saturation it was thrown away.
2. Rule E (`pressure_organ.py:209-223`) is the only emitter of `node_capability_impact`,
   and it fires on `gpu_hint >= GPU_HINT_THRESHOLD` -- GPU **saturation**. A node that is
   gone has no hints at all, so capability impact is structurally unreachable by absence.
   Live proof: `grammar_atoms` has **0** rows with `semantic_role='node_capability_impact'`,
   ever, and `capability_impacts` is `[]` in every
   `substrate_active_node_pressure_projection` row ever written.
3. The field's `node_capability` edge `node:circe -> capability:llm_inference` carries
   `channel_map = {gpu_pressure: pressure, memory_pressure: pressure,
   reasoning_load: reasoning_pressure}`. `availability` and `staleness` are both real
   entries in `NODE_CHANNELS` (`services/orion-field-digester/app/tensor/channels.py`)
   and neither is mapped. Node presence does not propagate to capability state.
4. `orion-llm-gateway` routes off `get_route_targets()`, a static table of hardcoded IPs.
   `route_catalog.py` really does probe `/health` per target and cache it
   (`refresh_route_health_cache`, `_probe_health`) -- but that cache is referenced only
   inside `route_catalog.py`, serving the `GET /routes` display endpoint. It "fails open
   to unknown" and dispatches to a dead host regardless.

Because of (4) the gateway has no move available on failure except to fabricate prose:
`llm_backend.py:1339` returns `{"text": "[Error: llamacpp timed out after waiting]"}` as a
normal result. That is why patching the return value is the wrong fix -- it is the symptom
of a router with no inputs, not a bug in the error branch.

## Current architecture

```
node_catalog.yaml          circe: capabilities{local_llm_heavy, local_llm_quick, ...}
   |                       (declares what each node provides)
   v
orion_biometrics ----> NodeBiometricsProjectionV1.nodes{last_seen_at, expected_online}
   |                       (knows who is reporting)
   v
pressure_organ  ----> ActiveNodePressureStateV1{active_pressures, capability_impacts:[]}
   |                       (event-triggered; blind to absence; capability_impacts dead)
   v
field digester  ----> node channels (availability, staleness, ...) 
   |                  node_capability edges (channel_map omits availability)
   v
capability channels (pressure, confidence, available_capacity, ...)

           ... and, entirely disconnected from all of the above:

orion-llm-gateway  get_route_targets() -> {"quick": http://100.112.254.99:8013, ...}
```

The body model and the router have never been introduced. Everything needed to connect
them already exists as schema; the joins are missing.

## Missing questions

- What is the authoritative statement of "capability C is currently servable"? Today: none.
- Who is allowed to act on it -- router only, or cognition too?
- If a capability has multiple providers, is partial loss an incident or a non-event?
  (Design answer: non-event for alerting, real event for the field.)
- Does failing a request loudly when no provider exists regress the chat UX vs. today's
  fabricated `[Error: ...]` text? (It changes it; see Risks.)

## Proposed schema / API changes

### Phase 1 (this patch)

- **Rule F**, new, in `pressure_organ.py`: `expected_online AND stale` now also emits one
  `node_capability_impact` per capability the catalog says the node provides. Rule E
  (saturation) is untouched and keeps its meaning; Rule F is the absence sibling it never
  had. `node_capability_impact` is already in `ALLOWED_PRESSURE_ROLES` and already handled
  by `pressure_reducer.py:228` -- this makes existing dead code reachable, it does not mint
  a new concept.
- **`sweep_absent_nodes()`**, new, in `pressure_organ.py`: a pure function over the
  existing `NodeBiometricsProjectionV1` that returns the node_ids which are
  `expected_online` and stale. This is the missing trigger. It takes the projection that
  already carries `last_seen_at` and `expected_online` per node -- no new state, no new
  storage. The event-triggered path is left exactly as-is; this runs alongside it.
- **`trace_id` now includes `semantic_role`** (`candidate_events.py`), fixing 2a. Position
  is deliberate: `parse_pressure_trace_id()` recovers the node with
  `trace_id.split(":", 2)[1]`, so the role goes *after* the node id and that parser is
  untouched.
- **Reducer arm fixed** (`pressure_reducer.py`): `node_capability_impact` appended
  `f"capability:{pressure_kind}"`, and `pressure_kind` for this role is the constant
  `"capability"` -- so the only value it could ever write was the literal
  `"capability:capability"`. It now expands to the node's real declared capabilities from
  the catalog profile already resolved in that loop, truthy-only and sorted. This was never
  caught because the arm had never once executed.
- No new bus channel, no new table, no new env key in phase 1.

### Phase 2 (not this patch)

- **Sweep the catalog, not just the projection.** `sweep_absent_nodes()` iterates
  `node_bio.nodes`, which is built from received events, so a node that has never reported
  once is invisible to it. Live 2026-08-29 the projection holds only `atlas`, `circe`,
  `athena` -- while `prometheus` is catalogued `expected_online: true` with
  `monitoring/logs/metrics: true` and has never written an `orion_biometrics` row in the
  table's entire history. A declared node that never appears is the quietest possible
  failure and phase 1 does not catch it.
- `CapabilityAvailabilityProjectionV1`: `capability -> {providers, available_providers,
  degraded_since}`. Reducer over the pressure projection. The single authoritative answer
  to "is C servable".
- Field: add `availability`/`staleness` to the `node_capability` edge `channel_map` so node
  presence reaches `capability:*`. See "Substrate/field signals" below.

### Phase 3 (not this patch)

- `get_route_targets()` becomes capability-aware; a route whose capability has no available
  provider is not selectable. Reroute if another provider exists; typed no-provider failure
  if not. This is what retires the `[Error: ...]`-as-content hack without touching the
  payload contract.

## Substrate/field signals to include

Requested explicitly, and it turns out to be the cheapest high-value part -- the channels
already exist and are already wired, they are just not fed from presence.

1. **`availability` and `staleness` -> capability channels.** Both are already in
   `NODE_CHANNELS`. The `node_capability` edges already have a `channel_map` mechanism that
   propagates node channels to capability channels with an edge weight. Adding
   `{"availability": "confidence", "staleness": "pressure"}` to those edges makes capability
   confidence drop when its provider goes silent, using machinery that already runs every
   tick. No new reducer.
2. **`available_capacity`** (already in `CAPABILITY_CHANNELS`, already live -- it showed up
   as `top=available_capacity=0.20177` in a real `telemetry_anomaly` metacog trigger during
   the outage window). This is the natural home for "fraction of declared providers that are
   currently reachable". Today nothing computes it from provider presence.
3. **`reliability_pressure`** (already in `CAPABILITY_CHANNELS`). Capability-level
   counterpart to the node-level `failure_pressure`/`observer_failure_pressure`.
4. **Deliberately NOT added:** a new "capability_absence" channel. Every signal above
   already exists with the right semantics; minting a parallel name would be a keyword
   cathedral. If phase 2 finds these three genuinely cannot express absence, that is the
   moment to propose a new channel -- with a producer, consumer and test in the same patch.

Metric quality gate for the phase-2 signals, run per CLAUDE.md 0A:

1. *Provenance*: `availability`/`staleness` are written by the field digester's node
   reconcile from `NodeBiometricsProjectionV1.last_seen_at`; `available_capacity` would be
   computed from `NodeCatalog` capability declarations joined to that same presence. Traced
   to real producing code, not schema comments.
2. *Independence*: presence is **not** a monotonic transform of anything already in the
   capability vector. Today's capability channels derive from `gpu_pressure`,
   `memory_pressure`, `reasoning_load` -- all load metrics that go to 0 when a node dies.
   Presence is the orthogonal axis; that orthogonality is the whole finding.
3. *Theory anchor*: a capability is servable iff at least one declared provider is
   reachable. That is a definition, not a correlation.
4. *Live-data sanity*: **must be run before phase 2 ships.** Specifically check the failure
   mode this codebase has hit before -- confirm `availability` can return to a genuine 1.0
   rest state after recovery and is not ratcheted or decayed toward 0 by
   `NODE_DECAY_CHANNELS` in `services/orion-field-digester/app/digestion/decay.py`. Rule B'
   (`node_availability_recovered`) exists precisely because `availability` was a one-way
   ratchet once before (2026-07-22, node:atlas).
5. *Existing mechanism*: searched. `node_capability_impact` + `capability_impacts` +
   `CAPABILITY_CHANNELS` all already exist. This patch reuses them rather than adding.
6. *Reversibility*: phase 1 is one rule and one pure function, both removable without a
   schema or data migration. Phase 2's channel_map entries are config-shaped and reversible.

## Files likely to touch

- `orion/substrate/biometrics_loop/pressure_organ.py`: Rule F + `sweep_absent_nodes()` (phase 1)
- `orion/substrate/biometrics_loop/tests/`: new tests (phase 1)
- `orion/substrate/biometrics_loop/pipeline.py`: call the sweep (phase 2 wiring)
- `services/orion-field-digester/app/tensor/*`: channel_map (phase 2)
- `services/orion-llm-gateway/app/llm_backend.py`: capability-aware routing (phase 3)

## Non-goals

- Not changing `run_llm_chat()`'s return contract. Explicitly vetoed; phase 3 makes it
  unnecessary rather than patching it.
- Not building a notification rule. A rate threshold on this was measured and **rejected**:
  transport metacog-trigger rate is cooldown-capped at 120/hr
  (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC=30`), and over 315 hours p95=118.3 while the
  real outage hour scored **89** -- below the p95 of ordinary hours. No threshold separates
  them. Capability transitions are discrete state changes and need no threshold at all.
- Not fixing notify delivery. Separate, real, and a prerequisite for any alerting:
  `notify_attempts` has **0 rows ever** and `notify_requests` is **100% `status='pending'`**
  across 10,671 rows since 2026-07-20.
- Not deleting the 936 poisoned `orion_metacognitive_trace` rows.

## Acceptance checks

- [x] Phase 1: Rule F emits `node_capability_impact` for an `expected_online` node that has
      gone stale, one per declared capability, and emits nothing for an
      `expected_online: false` node (atlas must stay quiet forever).
- [x] Phase 1: `sweep_absent_nodes()` returns exactly the stale+expected_online node ids and
      is a pure function of the projection.
- [x] Phase 1: two rules firing for one node in one tick survive as two traces.
- [x] Phase 1: the reducer records real capability names, not `capability:capability`.
- [x] Phase 1: both fixes mutation-tested -- reverting either makes the matching test fail.
- [ ] Phase 2: `capability_impacts` becomes non-empty in a live projection row for the first
      time in its history.
- [ ] Phase 2: kill circe's llamacpp container -> `local_llm_quick` shows 0 available
      providers within `stale_after_sec`; restore -> Rule B' clears it.
- [ ] Phase 3: replaying 00:01-00:47Z produces **one** capability transition, not 663 timeouts.

## Risks

- Severity: medium. A wrong body model (false staleness) could take a healthy route out of
  service in phase 3. Mitigation: fail open on unknown, and Rule B' recovery already exists.
- Severity: medium. Phase 3 changes user-visible chat behavior on total capability loss
  (typed failure instead of fabricated prose). That is the point, but it is a real change.
- Severity: low. Rule F increases grammar-atom volume during a real outage (one per
  capability per stale node per tick). circe declares 5 capabilities; bounded and small,
  but it is a multiplier on an incident, which is when volume is least welcome. Phase 2
  should dedupe at the reducer, not the emitter.
