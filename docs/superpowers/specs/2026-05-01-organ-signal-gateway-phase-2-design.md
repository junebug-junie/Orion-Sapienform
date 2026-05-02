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

When multiple causal parents carry OTEL context and **trace_ids disagree**, today “first parent with OTEL wins.” Phase 2 adds:

- **Structured log** (organ_id, signal_kind, list of conflicting parent signal_ids / trace_ids), and  
- **`notes` append** when there is room under the five-note cap (truncate or merge per existing gateway behavior),

so operators can see conflicts in signal inspect without changing the winner-takes-first resolution algorithm unless a later phase revisits it.

---

## Phase 2a completion criteria (acceptance for “A done”)

**2a is complete** when all of the following are true:

1. **M1 and M2 are merged** with fixture-backed adapter tests and updated channel documentation.
2. **End-to-end proof:** Under a **scripted test harness** (preferred) or **documented staging procedure**, the gateway produces hardened signals on **`orion:signals:{organ_id}`** for at least **three distinct organ_ids** in the primary chain (e.g. biometrics, equilibrium, collapse_mirror) such that **one OTEL `trace_id`** spans those signals in order (exogenous start → derived children), within the configured signal window. The procedure or test name is referenced in this spec’s implementation plan appendix.
3. **M3 and/or M4:** If **M3** is not merged by the time (1)–(2) hold, **2a** may still be marked complete only if the written proof in (2) explicitly lists which organs are covered and **M3/M4** are scheduled as **2a.1** with dates or ticket references. If the team prefers a stricter bar, **require M3 merged** before 2a complete; pick one interpretation in the implementation plan and do not leave it ambiguous.

Default interpretation for this spec: **(1)+(2) required**; **M3 strongly encouraged**; **M4 may slip to 2a.1** if chat_stance contract is not ready, **provided** (2) still holds for three organs without counting duplicate kinds on the same organ.

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
- **Stretch:** Rolling in-memory cache of recent signals keyed by `otel_trace_id`, ordered by `observed_at`, serving **`GET /api/signals/trace/{trace_id}`** as in phase-1 spec. If included, cap cache size and document eviction (e.g. max N traces or M total signals).

---

## Testing

### Phase 2a

- Per-adapter tests: golden fixtures from captured/representative payloads; assert `organ_id`, `organ_class`, `signal_kind`, `dimensions` ranges, and `causal_parents` when parents are injected via `prior_signals`.
- Gateway integration: sequential envelopes within the signal window; OTEL trace equality where specified in milestone tests.
- Regression: biometrics + passthrough + missed-parent behavior unchanged unless intentionally extended.

### Phase 2b

- Unit tests for merge/update of latest-per-organ map and JSON serialization.
- Optional: HTTP test for `GET /api/signals/active` with mocked Redis subscriber or pre-seeded store.

---

## Observability and ops

- Phase 1 OTEL env vars remain the operator contract; phase 2 does not require Jaeger/Tempo deployment but documents how to validate trace continuity when a backend exists.
- Collector DCGM / Prometheus sections remain optional; no new requirement in phase 2.

---

## File map (expected deltas)

| Area | Expected changes |
|------|-------------------|
| `orion/signals/adapters/*.py` | Replace stubs for M1–M4 organs in scope; add fixtures under `orion/signals/adapters/tests/` (or adjacent) per organ. |
| `orion/signals/registry.py` | Verified `bus_channels` and `notes` for graduated organs. |
| `services/orion-signal-gateway/` | Subscription/settings tweaks; processor logging for parent trace conflict; tests under `app/tests/`. |
| `services/orion-hub/` | Subscriber module or lifespan hook; `GET /api/signals/active` route; tests. |

---

## Self-review (spec quality)

- **Placeholders:** Milestone **M3** order is explicitly “TBD by contract clarity” — resolved during implementation planning with evidence, not left open at merge time.
- **Consistency:** Aligned with phase-1 schema, Hub JSON shapes, and offboarding channel naming.
- **Scope:** Phase 2b defaults to **active only**; trace endpoint is optional stretch to avoid scope creep.

---

## Next step after approval

Implementation planning uses the **writing-plans** skill from a clean session; this document is the **requirements source** for that plan.
