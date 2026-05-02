# Design: Orion Organ Signal Gateway — Phase 2 (adapters + Hub inspect)

**Date:** 2026-05-01  
**Status:** Draft — pending user review  
**Depends on:** [Organ Signal Gateway (phase 1)](2026-05-01-organ-signal-gateway-design.md), [Offboarding / inventory](../guides/2026-05-01-organ-signal-gateway-offboarding.md)  
**Execution model:** Milestoned depth-first for **Phase 2a**, then **Phase 2b** (Hub). This matches the agreed “A then B” shape with **option 2** from brainstorming: mergeable milestones, hardest adapters last, explicit acceptance for when **2a** is complete.

---

## What this is

Phase 2 turns the phase-1 **framework** into **production-grade signal paths** for the **primary causal chain**, then exposes those signals on the Hub for operators.

1. **Phase 2a — Vertical slice (milestones):** Replace stub adapters with real `can_handle` / `adapt` implementations and **verified** `bus_channels` for organs along the primary chain, in a sequence that keeps mainline shippable after each milestone.
2. **Phase 2b — Hub inspect:** Hub subscribes to gateway output channels, keeps an in-memory latest-per-organ window, and implements **`GET /api/signals/active`** per the phase-1 spec. Optional: trace cache + **`GET /api/signals/trace/{trace_id}`** in the same phase or as a fast follow (see §5).

---

## What this is NOT

- Not **orion-wearable-bridge** or custom OTEL wearable metrics (still out of scope; unchanged from phase 1).
- Not **signal persistence** (graph/SQL), alerting, or policy gates on signals.
- Not **full** migration of legacy `correlation_id` plumbing to OTEL across all services.
- Not a mandate to implement **every** organ adapter in phase 2; only the chain called out in §3 is in scope for **2a completion**. Other organs remain stubs until a later phase.

---

## Relationship to phase 1

Phase 1 delivered: `OrionSignalV1`, `ORGAN_REGISTRY`, normalization primitives, gateway subscription + processing + passthrough + OTEL emission, biometrics reference adapter, stub adapters, and tests. Phase 2 **extends** that code: same schema, same gateway chassis, same `orion:signals:{organ_id}` publish pattern described in the offboarding guide.

Errata for phase-1 doc contradictions (test tree, limitations) are **closed in the first pass** (next section) and may be mirrored back into `2026-05-01-organ-signal-gateway-design.md` as a short “Phase 2 errata” blurb if desired.

---

## First pass: operations & contract hardening (before production-ready)

**Intent:** Address gaps that block treating the gateway + Hub as a **reliable production contract** (per review: multi-instance, bus semantics, contradictions, OTEL tie-breaks, Hub trace completeness, privacy). This **first pass** is the **first slice of the implementation plan**: execute it **before or in tight lockstep with M1** so milestones do not encode ambiguous behavior.

**Production-ready gate:** Phase 2 work may not be labeled **production-ready** until **all Critical** and **all Important** items below are specified in repo docs (README + settings reference) **and** implemented or explicitly waived by operator config with documented risk. **Minor** items: at least document defaults; implement where trivial.

### Critical

| Issue | Hardening deliverable |
|--------|------------------------|
| **Multi-instance gateway** | Document and enforce **one** deployment model: **(A)** singleton writer (one replica processing organ bus → `signals.*`), **or (B)** N replicas with **idempotent consumer dedupe** downstream by `signal_id` **and** gateway-side **partitioning** / leader election so **at most one** replica adapts a given raw channel. **Unsupported:** multiple full subscribers republishing the same events with no dedupe/partitioning. Add operator checklist to gateway README and Helm/compose comments. |
| **Bus semantics** | Document assumed transport (e.g. Redis **pub/sub** vs **streams**): **ordering** (per connection / per channel / none across channels), **delivery** (at-most-once vs at-least-once for chosen transport), and **gateway under load** (bounded internal queue, drop policy vs backpressure, metrics). State impact on **lineage inside the signal window** (reordering/backlog can break parent resolution). |
| **Contradictory test layout** | **Canonical layout:** **`orion/signals/adapters/tests/`** — pure adapter tests (fixtures, `prior_signals` dict injection, no live Redis). **`services/orion-signal-gateway/app/tests/`** — gateway integration (processor, passthrough, OTEL, window, multi-envelope). Cross-import helpers live in `orion/signals/` test utilities only if shared. Update the **File map** section below and phase-1 §7 / file map by errata. |
| **`otel_trace_id` / parent when parents disagree** | **Deterministic rules:** (1) Build the list of parents contributing to this signal in **`ORGAN_REGISTRY[organ_id].causal_parent_organs` order**, retaining only parents actually linked for this emission. (2) **`causal_parents` signal_ids** on the emitted `OrionSignalV1` follow that **same order**. (3) **OTEL parent context:** walk that list in order; use the **first** parent with valid `otel_trace_id` + `otel_span_id` on the resolved `OrionSignalV1` from `prior_signals`. (4) If **trace_ids** disagree across parents with OTEL, keep **first-wins** for span parent; **structured log + `notes`** as in §3 (parent trace disagreement). |
| **Hub trace cache** | When **`GET /api/signals/trace/{trace_id}`** ships (2b stretch or follow-up), define **`TRACE_CACHE_MAX_TRACES`**, **`TRACE_CACHE_TTL_SEC`**, **`TRACE_CACHE_MAX_SIGNALS_PER_TRACE`** (defaults set in code + env). **API contract:** **`200`** with body `trace_id`, `chain` (ordered by `observed_at`), **`complete`: bool** (`false` if eviction/TTL/cap truncated), **`gaps`** (optional array describing missing organs or breaks). **`404`** only when the trace id was **never** ingested into the cache window. |
| **PII / sensitive dimensions on OTEL** | Add **security subsection** to gateway operator docs: span attributes use **`OTEL_DIMENSION_ALLOWLIST`** (or denylist + default deny) for `dim.*`; **never** attach `summary`, raw payload fields, or `notes` text to spans. Document **high-cardinality** policy (hash or omit). Adapters remain responsible for not putting secrets in `dimensions`. |

### Important

| Issue | Hardening deliverable |
|--------|------------------------|
| **16-hex `signal_id`** | Document collision probability; **prefer** widening to **32-hex** (full SHA-256 hex of the same preimage) for newly emitted signals where backward compatibility allows; otherwise document **merge/dedupe** behavior when collision detected (metric + log). |
| **Graduation duplicates** | Specify policy: **(preferred)** optional gateway behavior **`suppress_adapted_when_passthrough`** keyed by `(organ_id, source_event_id)` or dedupe window; **or** require **consumers** to dedupe by `signal_id` / `source_event_id`. Pick one default for Orion first-party consumers and document. |
| **Collector vs biometrics** | Document **single source of truth** for GPU narrative: organ-bus biometrics adapter vs DCGM/hostmetrics. Default stance: **mesh causal chain** uses **gateway-emitted biometrics signals**; collector metrics are **parallel infra telemetry** for dashboards, not alternate parents in the DAG unless explicitly wired. |
| **`notes` max 5** | Define **append priority** when adapter, gateway, and validation compete: e.g. **(1)** gateway safety / lineage, **(2)** adapter degradation, **(3)** validation; **drop from lowest priority** with one **`notes`** entry recording truncation (`notes_truncated: true` is not in schema — use a final slot text like `"[truncated]"` within the five strings) or merge last two. |
| **EWMA vs `[-1, 1]` dimensions** | In normalization docs: **`EwmaBand.normalize` → `[0, 1]`** for **level-style** metrics; **`trend`**, **`valence`**, and similar use **`clamp11`** on **derived** deltas or signed scores, **not** raw EWMA output unless explicitly documented per adapter. Per-adapter table in README or spec appendix. |
| **`spark` vs `spark_introspector`** | **Canonical `organ_id` is `spark_introspector`** everywhere (registry, adapters, docs, primary chain narrative). Replace informal “spark” in prose with **`spark_introspector`** or define alias explicitly once if legacy events require it. |
| **`world_pulse` parents** | Replace “see note” with a **definitive edge list** for `causal_parent_organs` (even if empty + external annotation) and add **one DAG validation test** that fails if registry edges regress. |

### Minor

| Issue | Hardening deliverable |
|--------|------------------------|
| **Passthrough vs adapted race** | Document **last-write-wins** per **`orion:signals:{organ_id}`** channel and interaction with dedupe keys; if both paths can emit for the same logical event, tie to **`signal_id`** / `source_event_id` policy from Important table. |
| **`GET /api/signals/active` shape** | Default remains **latest per `organ_id`**. Document limitation: **multiple concurrent `signal_kind`** collapse to one row. **Optional extension (2b+):** `signals[organ_id]` → map keyed by `signal_kind` or nested object; defer unless product requires. |
| **`ttl_ms` (15s) vs 30s window vs Hub** | Document semantics: **`ttl_ms`** = hint for consumers; **gateway window** = parent lookup horizon; **Hub cache TTL** ≥ window or aligned explicitly; table in README. |
| **File map** | After test layout fix, the **File map** section lists both canonical test roots explicitly. |

### Assessment (from review) — closure criteria

Implementation was **not** production-ready until Critical items had **written + coded** resolution; Important items should be **nailed before** treating the **plan** as production-ready. **First pass** in the implementation plan tracks these explicitly; milestone **M1** should not merge on a branch advertised as HA without meeting the multi-instance row.

---

## Phase 2a: Milestoned depth-first (option 2)

### Rationale

A strictly linear “no organ until the previous is perfect” plan risks blocking for a long time on **recall** (request-response semantics) and **chat_stance** (cortex/router payload shape). Option 2 instead defines **mergeable milestones** and a **single written acceptance bar** for declaring **2a complete**, so the team can ship value incrementally without pretending the whole chain is done before Hub work.

### Milestones

| ID | Organ(s) | Intent |
|----|-----------|--------|
| **M1** | `equilibrium` | First **hybrid** organ with a real bus contract: real payloads, registry `bus_channels` verified against deployment, dimensions and `signal_kind` aligned with `ORGAN_REGISTRY`. |
| **M2** | `collapse_mirror` | First **endogenous** organ in the slice that depends on **biometrics** + **equilibrium** in `prior_signals`; **causal_parents** and **shared OTEL trace** with upstream signals validated in tests or staging. |
| **M3** | `journaler` **or** `recall` (order TBD by contract clarity) | Implement whichever has the **clearer, documented** bus (or result) contract first; the other follows in the same phase or as **2a.1** if 2a acceptance (§4) is already met without it. |
| **M4** | `chat_stance` | **Last:** requires explicit **cortex / router payload contract** subsection (channel, field paths, degradation rules) before implementation. Adapter degrades to low `confidence` + `notes` when fields are missing, per phase-1 design rules. |

**Biometrics** remains the reference adapter; no redesign unless a measured gap appears during M1–M2.

### Channel verification (gating every milestone)

Before or alongside each adapter milestone:

1. For each organ in scope, record **actual** Redis channel names and **representative payloads** (fixtures in repo, or redacted captures referenced from runbooks).
2. Align `ORGAN_REGISTRY[*].bus_channels` and gateway `ORGAN_CHANNELS` (or equivalent subscription config) so the gateway **subscribes only** to channels that feed that organ’s adapter, avoiding duplicate processing and noise.
3. Document any **environment variance** (staging vs prod channel prefixes) in the milestone notes or registry `notes` field where appropriate.

### Adapter rules (unchanged from phase 1, reinforced)

- Adapters stay in **`orion/signals/adapters/`**; stateless; **`NormalizationContext`** owned by gateway.
- **Graceful degradation:** malformed or partial payloads → valid `OrionSignalV1` with low `confidence` (e.g. `0.1`) and a **`notes`** entry; drop only truly irrelevant events.
- **`signal_id`:** Follow phase-1 rules (deterministic from `source_event_id` when present; UUID otherwise). When touching stub code, align short-hex inconsistencies called out in offboarding.
- **OTEL on adapter path:** Adapters do not set `otel_*`; gateway continues to own span creation. Endogenous adapters still resolve causal parents from `prior_signals` for `causal_parents` IDs.

### Parent trace disagreement (small hardening in 2a)

**Parent ordering** for both `causal_parents` and OTEL parent pick is **deterministic** — see **First pass → Critical** (`causal_parent_organs` registry order). When multiple parents carry OTEL context and **trace_ids disagree**, keep **first-in-that-order with OTEL** as parent for span linkage. Phase 2 adds:

- **Structured log** (organ_id, signal_kind, ordered list of conflicting parent signal_ids / trace_ids), and  
- **`notes` append** when there is room under the five-note cap (truncation priority per **First pass → Important**),

so operators can see conflicts in signal inspect. Changing the winner algorithm beyond first-in-registry-order requires a new spec revision.

---

## Phase 2a completion criteria (acceptance for “A done”)

**2a is complete** when all of the following are true:

1. **M1 and M2 are merged** with fixture-backed adapter tests and updated channel documentation.
2. **End-to-end proof:** Under a **scripted test harness** (preferred) or **documented staging procedure**, the gateway produces hardened signals on **`orion:signals:{organ_id}`** for at least **three distinct organ_ids** in the primary chain (e.g. biometrics, equilibrium, collapse_mirror) such that **one OTEL `trace_id`** spans those signals in order (exogenous start → derived children), within the configured signal window. The procedure or test name is referenced in this spec’s implementation plan appendix.
3. **M3 and/or M4:** If **M3** is not merged by the time (1)–(2) hold, **2a** may still be marked complete only if the written proof in (2) explicitly lists which organs are covered and **M3/M4** are scheduled as **2a.1** with dates or ticket references. If the team prefers a stricter bar, **require M3 merged** before 2a complete; pick one interpretation in the implementation plan and do not leave it ambiguous.

Default interpretation for this spec: **(1)+(2) required**; **M3 strongly encouraged**; **M4 may slip to 2a.1** if chat_stance contract is not ready, **provided** (2) still holds for three organs without counting duplicate kinds on the same organ.

**Production-ready vs milestone progress:** Merging **M1** on a branch advertised as **multi-replica HA** without meeting **First pass → Critical (multi-instance)** is **out of spec**. The **functional** 2a bar above can be met on a **singleton** gateway while first-pass items are still in flight; the **production-ready gate** (first pass introduction) applies before any “GA” or customer-facing HA claim.

---

## Phase 2b: Hub inspect (“B”)

### Preconditions

- Gateway output channel pattern is stable (e.g. **`orion:signals:{organ_id}`** per offboarding).
- **`OrionSignalV1`** JSON is the on-wire contract; Hub should parse with **`orion.signals.models`** where practical to avoid schema drift.

### `GET /api/signals/active`

Matches the phase-1 design response shape:

- **`as_of`:** ISO-8601 timestamp of the snapshot.
- **`signals`:** Map of `organ_id` → latest `OrionSignalV1` (or null for missing organs if the API chooses to enumerate known registry keys; default is **sparse map**: only organs that have been seen at least once).

Implementation sketch:

- Hub process maintains a **subscriber** (async task or existing Hub Redis lifecycle) to the gateway output pattern.
- **In-memory store:** latest signal per `organ_id`; eviction optional (e.g. TTL aligned with `ttl_ms` or gateway window). No database.
- Route lives with existing Hub FastAPI patterns (e.g. `services/orion-hub/scripts/api_routes.py`), with **the same auth/session posture** as other operator-read routes (align with `/api/substrate/*` unless security review dictates stricter RBAC).

### `GET /api/signals/trace/{trace_id}` (optional in 2b)

- **Default for phase 2b:** Implement **`/api/signals/active` only** to bound scope.
- **Stretch:** Rolling in-memory cache keyed by `otel_trace_id`, ordered by `observed_at`. **Constants and response shape** are mandatory when this ships — see **First pass → Critical (Hub trace cache)** (`TRACE_CACHE_*`, `complete`, `gaps`, `404` vs partial **200**).

---

## Testing

**Canonical roots (first pass):** adapter-only tests under **`orion/signals/adapters/tests/`**; gateway tests under **`services/orion-signal-gateway/app/tests/`**. No duplicate competing locations for the same concern.

### Phase 2a

- Per-adapter tests: golden fixtures from captured/representative payloads; assert `organ_id`, `organ_class`, `signal_kind`, `dimensions` ranges, and `causal_parents` **order** (registry order) when parents are injected via `prior_signals`.
- Gateway integration: sequential envelopes within the signal window; OTEL trace equality where specified in milestone tests; tests for **multi-parent OTEL tie-break** and **disagreeing trace_id** log/notes path.
- First-pass tests where applicable: dimension allowlist for OTEL export (mock exporter), dedupe/partitioning behavior if code path exists.
- Regression: biometrics + passthrough + missed-parent behavior unchanged unless intentionally extended.

### Phase 2b

- Unit tests for merge/update of latest-per-organ map and JSON serialization.
- Optional: HTTP test for `GET /api/signals/active` with mocked Redis subscriber or pre-seeded store.
- If trace endpoint ships: tests for **200 + `complete: false`**, **`gaps`**, and **404** never-seen trace.

---

## Observability and ops

- Phase 1 OTEL env vars remain the operator contract; phase 2 does not require Jaeger/Tempo deployment but documents how to validate trace continuity when a backend exists.
- **Span attribute privacy** follows **First pass → Critical (PII)**; ship **`OTEL_DIMENSION_ALLOWLIST`** (or equivalent) with safe defaults.
- Collector DCGM / Prometheus sections remain optional; **precedence vs organ-bus biometrics** follows **First pass → Important**.
- **Multi-instance and bus semantics** are documented per **First pass → Critical**; deployment manifests should echo the chosen model.

---

## File map (expected deltas)

| Area | Expected changes |
|------|-------------------|
| `orion/signals/adapters/*.py` | Replace stubs for M1–M4 organs in scope. |
| **`orion/signals/adapters/tests/`** (canonical) | Adapter unit tests + golden fixtures only. |
| `orion/signals/registry.py` | Verified `bus_channels`, `notes`, definitive **`world_pulse`** parents, **`spark_introspector`** naming alignment. |
| **`services/orion-signal-gateway/app/tests/`** (canonical) | Processor, passthrough, OTEL, window, multi-envelope integration tests. |
| `services/orion-signal-gateway/` | Subscription/settings (incl. cache caps, allowlist envs), multi-instance operator docs, bus overload behavior, dedupe/suppress flags per first pass. |
| `services/orion-hub/` | Subscriber module or lifespan hook; `GET /api/signals/active`; optional trace cache with first-pass `TRACE_CACHE_*` constants; tests colocated under Hub test tree per Hub conventions. |

---

## Self-review (spec quality)

- **Placeholders:** Milestone **M3** order is explicitly “TBD by contract clarity” — resolved during implementation planning with evidence, not left open at merge time. **First pass** supplies default constants for trace cache once the endpoint ships; implementation may tune defaults but must not ship without them.
- **Consistency:** Aligned with phase-1 schema, Hub JSON shapes, and offboarding channel naming; **first pass** resolves phase-1 contradictions (tests, naming, `world_pulse` edges).
- **Scope:** Phase 2b defaults to **active only**; trace endpoint is optional stretch with **mandatory** first-pass contract when enabled.
- **First pass:** Critical/Important/Minor tables are actionable; production-ready gate is explicit.

---

## Next step after approval

Implementation planning uses the **writing-plans** skill from a clean session; this document is the **requirements source** for that plan.
