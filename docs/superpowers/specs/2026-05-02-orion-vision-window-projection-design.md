# Orion Vision Window — perception projection service — design

**Date:** 2026-05-02  
**Status:** Draft for review  
**Scope:** `services/orion-vision-window` only (Hub/UI and downstream memory systems are consumers, not owners)

---

## 1. Purpose

Make **`orion-vision-window`** a **vision perception projection service**: it **normalizes live and recent visual state** for subscribers, without becoming a cognition engine, a detector, a durable vision database, or a UI layer.

It is the **“eyes open”** interface in the sense of a **contract and projection**—**not** HTML/JS ownership.

---

## 2. Positioning in the mesh

| This service **is** | This service **is not** |
|---------------------|-------------------------|
| A **projection** of recent perception into **snapshots, bus events, and HTTP-readable state** | The **vision capture** node (e.g. Retina) |
| **Adjacent to** capture, inference workers, graph/journal consumers, Hub debug surfaces | The **inference / detection** worker (e.g. vision host) |
| A **subscriber-oriented** view of “what’s in view lately” | **Canonical long-term memory** (RDF, journal, embeddings store) |

**Hub and debug pages** **subscribe to and render** this service’s outputs; they **do not** live inside `orion-vision-window`.

---

## 3. Hard guardrails

### 3.1 Canonical ownership

**`orion-vision-window` must never be the canonical owner of** raw visual media, long-term visual memory, embeddings, RDF assertions, or journaled interpretation.

Raw frames, object/artifact storage, graph/RDF, journals, and embedding stores remain in **downstream canonical systems**. This service may hold **references, cursors, and lightweight summaries** only, within explicit bounds.

### 3.2 Snapshots vs memory

**Snapshots are projection artifacts, not canonical memory.** The same rule applies to any **recovery** or **catch-up** records: they are **indexes and envelopes** for reconnect and debugging, not a second “memory organ.”

### 3.3 RPC and a single projection model (design rule)

**RPC must be a control/query surface over the same projection state, not a parallel ingestion path or second state model.**  
There is **one** logical projection: **ingestion → `live_state` → published views**. Any **VisionWindowService** (or equivalent) path **joins** that projection for query/reply; it does **not** maintain **RPC-only** buffers or a separate windowing model.

---

## 4. State model: `live_state` vs `recovery_state`

### 4.1 `live_state` (ephemeral)

- **In-memory** normalized “live window” (recent visual state for low-latency use).
- **Rebuilt** from current streams/events as needed.
- **Safe to lose on process restart**; correctness of the *projection rules* does not depend on permanence of this heap.

### 4.2 `recovery_state` (durable, tiny)

- **Bounded** durable **recovery index** so Hub and operators can **reconnect to something meaningful** after restarts.
- Stores **at most**: last **N** snapshot **metadata/envelopes**, **stream cursors / bus offsets**, **artifact IDs and URIs** (not heavy frame bytes), optional **detection summary** fields, **expiry/TTL**.
- **Not** a full history of vision; **not** query-the-world.

**Durability target (by tier):**

- **Redis:** preferred for last **N**, TTL, cursors, reconnect state.
- **SQL:** optional, only for a **lightweight** snapshot-envelope **history** table if ops need it—still **not** a vision database.
- **Object/artifact store:** holds **actual** media; this service stores **pointers**, not blobs.
- **Graph/RDF/journal:** **interpreted** memory; out of scope as storage **owned** here.

**Posture:** **Ephemeral-first** with **bounded durable recovery**; **persist references and cursors, not visual substance.**

---

## 5. Subscriber plane

### 5.1 v1 (required)

- **Bus-first:** publish **normalized vision-window** events on the bus (aligned with existing `orion:vision:windows` / `vision.window` usage where applicable).
- **HTTP read** contract: expose endpoints for **current live snapshot**, **last N envelope metadata**, and **cursor catch-up** (read models for subscribers after disconnect or cold start).
- **No required WebSocket** in v1: **not** a product requirement that this service own live push in the first release.
- **Optional** lightweight **debug/preview** HTTP handlers are allowed; **product UI** remains in Hub.

### 5.2 Later (not v1-gated)

- **WebSocket** or **SSE** fanout for Hub/debug **live** panels, **derived from the same projection state** as bus + HTTP—**not** a separate canonical stream and **not** a second state machine.

**v1 end-to-end shape:**

```text
camera / artifact / detection events
  → orion-vision-window (projection)
  → in-memory live_state
  → Redis recovery_state (bounded)
  → bus events + HTTP catch-up
  → Hub / debug subscribers (render; no UI ownership here)
```

**Principle:** **bus-native**, **HTTP-readable**; **projection service**, not UI owner; **ephemeral-first**, not a memory organ.

---

## 6. Implementation scope split

### 6.1 `v1_required` (must ship for “real” v1 in this spec’s sense)

- **Bus-native projection ingestion** (subscribe to vision artifact / relevant ingress; normalize into the projection).
- **In-memory `live_state`** with clear normalization rules.
- **Redis bounded `recovery_state`** (last N, TTL, cursors, artifact refs/summaries as agreed).
- **HTTP catch-up/read contract** (snapshot, last N metadata, cursors).
- **Snapshot envelope schema** (see §7) and **documented subscriber semantics** (what each channel/endpoint means, ordering expectations, stale/empty behavior).

### 6.2 `v1_design_reserved` (specified in this doc; implementation may follow when the same change set touches the right runtimes)

- **`VisionWindowService` RPC** contract and wiring **as a query/control path** over the **same** projection (see §3.3).
- **Rolling flush** semantics (time- and/or event-driven windows) as part of that **single** projection—not a duplicate aggregator.
- **RPC joins the same projection path as streaming**; **no separate RPC-owned state.**

### 6.3 `phase_2_implementation` (explicit follow-on)

- **Cortex / tool RPC integration** end-to-end (exec verbs, request/reply channels, catalog alignment).
- **Request/reply projection queries** (e.g. “window as of cursor,” explicit flush or bounded read).
- **Explicit rolling flush controls** (operator or API knobs) if needed once Cortex paths land.

**Branch rule:** Do **not** force **full** RPC/Cortex plumbing into v1 **unless** the active branch is already modifying Cortex and window together; still document **§3.3** and **§6.2** so partial implementations do not fork state models.

---

## 7. Snapshot envelope schema (v1)

- **v1** formalizes the **envelope** used for bus emission, HTTP responses, and recovery rows: stable IDs, time bounds, **artifact ID list**, **summary** (counts, labels, captions as today), **stream/source keys** where available, **cursor/offset** for catch-up, and **TTL/expiry** where stored.
- **Alignment:** Extend or version **`VisionWindowPayload`** (and related kinds in `orion/schemas/vision.py`) **only as needed** for projection and recovery fields; avoid breaking Council/Scribe consumers without a migration note.

*(Exact Pydantic fields and channel `kind` matrix belong in the implementation plan and registry updates.)*

---

## 8. Non-goals

- Owning **Hub** templates, **primary** WebSocket product surface in v1, or **canonical** storage of media or interpreted memory.
- Replacing **orion-vision-council**, **orion-vision-scribe**, **RDF writer**, or **journal/collapse** pipelines.
- Becoming a **durable vision database** or **embedding** store.

---

## 9. Success criteria

- After restart, a subscriber can use **HTTP catch-up** (and bounded recovery data) to obtain **meaningful** last snapshot metadata and cursors **without** requiring this service to hold raw media.
- **Bus** and **HTTP** describe the **same** projection; **no** second hidden buffer for “RPC only.”
- Documentation states **guardrails** (§3) so future features cannot silently turn this service into a memory organ.

---

## 10. Open items (for implementation plan, not blockers for this design)

- Exact **stream/source key** extraction from `VisionArtifactPayload` (`inputs` / `meta`).
- **Catalog** entries for new HTTP routes or schema versions (if any).
- Optional **SQL** history table: yes/no per deployment.

---

## 11. References

- `docs/vision_services.md` — pipeline placement and channels overview.
- `orion/schemas/vision.py` — `VisionWindowPayload`, RPC payloads.
- `services/orion-vision-window/app/main.py` — current ingestion and flush behavior (baseline to replace incrementally per scope split).
