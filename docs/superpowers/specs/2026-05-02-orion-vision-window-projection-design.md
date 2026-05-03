# Orion Vision Window — perception projection service — design

**Date:** 2026-05-02  
**Status:** Draft (post-review: flush scope, ordering, envelope, privacy, channels)  
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
| A **bounded reconnect surface** for recent visual projection state | A replayable event log or visual warehouse |

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

### 3.4 No hidden durability escalation

Any new field, endpoint, cache, table, or Redis structure that would materially increase retention, store raw visual substance, or make this service authoritative for visual history must be treated as a **scope change** and reviewed as a memory-system decision, not as a convenience cache.

---

## 4. State model: `live_state` vs `recovery_state`

### 4.1 `live_state` (ephemeral)

- **In-memory** normalized “live window” (recent visual state for low-latency use).
- **Rebuilt** from current streams/events as needed.
- **Safe to lose on process restart**; correctness of the *projection rules* does not depend on permanence of this heap.
- May maintain per-stream rolling accumulators, but only as projection buffers with explicit age/count bounds.

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

### 4.3 Suggested Redis recovery shape

Redis keys should make the retention boundary obvious:

```text
vision-window:stream:{stream_id}:latest
vision-window:stream:{stream_id}:last_n
vision-window:stream:{stream_id}:cursor
vision-window:global:last_n
vision-window:global:cursor
vision-window:health:last_ingest
```

Implementation notes:

- `latest` stores the most recent snapshot envelope for a stream.
- `last_n` stores a capped list or sorted set of snapshot envelopes/references.
- `cursor` stores the last processed bus/event cursor known to this projection.
- `global:last_n` is a convenience projection for operators/Hub, not canonical memory.
- All recovery keys must have explicit TTLs or capped lengths.
- Raw frame bytes, large base64 payloads, and unbounded detection arrays are forbidden in Redis recovery state.

---

## 5. Subscriber plane

### 5.1 v1 (required)

- **Bus-first:** publish **normalized vision-window** events on the bus under the **canonical v1 channel** **`orion:vision:windows`**. Legacy or alternate names (e.g. `vision.window`) must be documented in the bus catalog as **aliases** that carry the **same** envelope shape—**one** logical stream per deployment, not competing semantics.
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

## 6. HTTP read contract

The exact route names may be adjusted to match existing Hub/service conventions, but v1 should expose these semantic surfaces.

### 6.1 Health / readiness

```text
GET /healthz
GET /readyz
```

`/healthz` answers whether the process is alive. `/readyz` answers whether the projection can serve meaningful read models, including Redis connectivity if Redis-backed recovery is enabled.

### 6.2 Current projection

```text
GET /api/vision-window/current
GET /api/vision-window/streams/{stream_id}/current
```

Returns the current in-memory live projection if available; may fall back to bounded recovery if process-local state is cold.

Required response semantics:

- `status: "ok"` when current projection exists and is fresh.
- `status: "empty"` when no projection has been observed yet.
- `status: "stale"` when the latest projection exceeds the freshness threshold.
- Response includes `generated_at`, `stream_id` where applicable, `snapshot_id`, `cursor`, `age_ms`, and `source` (`live_state` or `recovery_state`).

**Freshness (v1 default):** use a **single global** `stale_after_ms` (and optional global default in config) for `stale` vs `ok` in §6.2. **Per-stream** stale overrides are **not** required in v1; if added later, responses should still expose the effective threshold used so clients do not assume global-only behavior silently.

### 6.3 Last-N catch-up

```text
GET /api/vision-window/recent?limit=20
GET /api/vision-window/streams/{stream_id}/recent?limit=20
```

Returns bounded snapshot envelopes only. Implementations must cap `limit` server-side.

### 6.4 Cursor catch-up

```text
GET /api/vision-window/catch-up?after_cursor={cursor}&limit=20
GET /api/vision-window/streams/{stream_id}/catch-up?after_cursor={cursor}&limit=20
```

Returns envelopes after a known cursor if available within the bounded recovery window.

If the requested cursor is older than the recovery window, return a structured response such as:

```json
{
  "status": "cursor_expired",
  "message": "Requested cursor is outside the bounded recovery window.",
  "latest_cursor": "...",
  "earliest_available_cursor": "..."
}
```

This avoids pretending the service is a replay log.

---

## 7. Bus contract and ordering semantics

### 7.1 Published event semantics

The service publishes normalized projection events when the live window changes materially or a **flush boundary** (§10) is reached.

Event names/channels should align with the bus catalog; **canonical v1** channel is **`orion:vision:windows`** (see §5.1 for alias policy).

Each emitted event should include:

- stable `snapshot_id`
- `stream_id` / source key if available
- `window_start` / `window_end`
- `generated_at`
- `cursor` / upstream event offset if known
- `artifact_ids` / `artifact_uris`
- detection/caption summary fields
- freshness/expiry metadata

### 7.2 Ordering

Ordering is **best-effort per stream**, not globally total across all cameras/nodes.

Rules:

- Per-stream cursors are preferred over global ordering assumptions.
- Consumers must tolerate out-of-order or duplicate envelopes.
- **Dedupe and supersession:** Treat **`snapshot_id` as the primary idempotency key** for a snapshot envelope. If the same `snapshot_id` is seen twice for the same `stream_id`, the consumer should treat it as a duplicate and keep one copy (prefer the last ingested envelope if payloads differ). **`cursor` is an opaque, per-stream catch-up token** produced by this projection for subscribers; it is **not** assumed globally monotonic across restarts or all sources. Do **not** use `(stream_id, cursor)` alone as a uniqueness key for deduplication—cursors may advance without a 1:1 mapping to `snapshot_id` when recovery is partial. When comparing “newer vs older” for the **same** `snapshot_id`, use `generated_at` (and ingest metadata if present) as a tie-breaker.
- If source timestamps and ingest timestamps differ, both should be retained.

`upstream_event_ids` (§9) is a **hint list** for tracing; **order is not a total ordering guarantee** unless a future schema documents otherwise for a specific source.

### 7.3 Idempotency

Projection writes and publishes should be idempotent where practical:

- Duplicate upstream artifacts/events should not produce unbounded duplicate recovery rows.
- `snapshot_id` should be deterministic or de-duplicable for a given flush boundary when feasible.
- Redis writes should replace/cap rather than append forever.

---

## 8. Implementation scope split

### 8.1 `v1_required` (must ship for “real” v1 in this spec’s sense)

- **Bus-native projection ingestion** (subscribe to vision artifact / relevant ingress; normalize into the projection).
- **Minimal bounded flush** (see §10.1): ingestion must **emit snapshot envelopes** on explicit bounds so `live_state` and recovery are never unbounded aggregations of raw events (required wall-time and/or artifact-count ceiling before emit). **Fine-grained rolling policy** (interval defaults, optional `flush_on_stream_change`, operator knobs) may follow under §8.2 but **must reuse the same flush code path**.
- **In-memory `live_state`** with clear normalization rules.
- **Redis bounded `recovery_state`** (last N, TTL, cursors, artifact refs/summaries as agreed).
- **HTTP catch-up/read contract** (snapshot, last N metadata, cursors).
- **Snapshot envelope schema** (see §9) and **documented subscriber semantics** (what each channel/endpoint means, ordering expectations, stale/empty behavior).
- **Health/readiness endpoints** distinguishing process liveness from projection readiness.
- **Metrics/logging hooks** sufficient to diagnose missing streams, stale windows, and Redis recovery failures.

### 8.2 `v1_design_reserved` (specified in this doc; implementation may follow when the same change set touches the right runtimes)

- **`VisionWindowService` RPC** contract and wiring **as a query/control path** over the **same** projection (see §3.3).
- **Rolling flush policy tuning** (defaults in §10.2: `flush_interval_ms`, `flush_by_count`, `max_artifacts_per_window`, optional `flush_on_stream_change`, etc.) as part of that **single** projection—not a duplicate aggregator. **Does not replace** §8.1 minimal bounded flush; it **extends** the same flush path.
- **RPC joins the same projection path as streaming**; **no separate RPC-owned state.**

### 8.3 `phase_2_implementation` (explicit follow-on)

- **Cortex / tool RPC integration** end-to-end (exec verbs, request/reply channels, catalog alignment).
- **Request/reply projection queries** (e.g. “window as of cursor,” explicit flush or bounded read).
- **Explicit rolling flush controls** (operator or API knobs) if needed once Cortex paths land.

**Branch rule:** Do **not** force **full** RPC/Cortex plumbing into v1 **unless** the active branch is already modifying Cortex and window together; still document **§3.3** and **§8.2** so partial implementations do not fork state models.

---

## 9. Snapshot envelope schema (v1)

- **v1** formalizes the **envelope** used for bus emission, HTTP responses, and recovery rows: stable IDs, time bounds, **artifact ID list**, **summary** (counts, labels, captions as today), **stream/source keys** where available, **cursor/offset** for catch-up, and **TTL/expiry** where stored.
- **Alignment:** Extend or version **`VisionWindowPayload`** (and related kinds in `orion/schemas/vision.py`) **only as needed** for projection and recovery fields. **Compatibility rule (v1):** new projection fields are **additive** to the wire shape where possible; **no removal or type change** of existing fields consumed by Council/Scribe in the same release without a **documented migration** (dual-read window, explicit version bump in `schema_version`, or coordinated release note). Prefer optional fields with defaults over breaking renames.

Suggested envelope shape:

```yaml
schema_version: vision_window_snapshot.v1
snapshot_id: string
stream_id: string | null
source_node: string | null
camera_id: string | null
window_start: datetime | null
window_end: datetime
generated_at: datetime
cursor: string | null
upstream_event_ids: list[string]   # order not guaranteed; tracing hint only (§7.2)
artifact_ids: list[string]
artifact_uris: list[string]
summary:
  label_counts: object
  top_labels: list[string]
  caption: string | null
  detection_count: integer | null
freshness:
  age_ms: integer | null
  stale_after_ms: integer | null
  expires_at: datetime | null
# meta: intentionally omitted in vision_window_snapshot.v1 (see Hard bounds below)
```

Hard bounds:

- `artifact_uris` must point to external artifact/media storage.
- `summary` must remain lightweight.
- No raw frame bytes.
- No unbounded per-detection payloads unless explicitly capped and justified.
- **`meta` is not part of v1 envelopes.** A future schema version may introduce a bounded `meta` (e.g. max serialized size on the order of 2 KiB and **whitelisted keys** only) so §3.4 is not bypassed via opaque bags.

### 9.1 Privacy, logging, and sensitive fields

Captions, labels, and artifact URIs may contain **PII or sensitive environment detail** (faces, license plates, on-screen text, internal hostnames). **Rules for v1:**

- **Structured logs and metrics must not** embed full `caption`, full URI lists, or raw `top_labels` by default; use **hashes, counts, truncated previews**, or explicit debug flags gated for operators.
- **Hub and debug UIs** remain responsible for display policy; this service documents that summaries are **not** redacted unless a downstream policy applies.
- Prefer **artifact IDs** over dereferenced content in any persistence path for this service (already required elsewhere).

### 9.2 `VisionWindowPayload` versioning

- Ship **`schema_version: vision_window_snapshot.v1`** (or aligned enum) on every envelope once formalized.
- **Additive changes:** bump minor or extend optional fields; Council/Scribe readers ignore unknown fields.
- **Breaking field renames or removals:** require coordinated release, dual publication, or explicit consumer migration—**not** a silent patch inside `orion-vision-window` alone.

---

## 10. Flush semantics (minimal v1 and rolling policy)

### 10.1 Minimal bounded flush (`v1_required`)

Without this, ingestion could hold arbitrary aggregates in memory—violating projection bounds. **v1 must:**

- Emit at least one snapshot envelope per stream within a **configurable maximum wall time** since the last emit for that stream, **and/or** when a **configurable maximum artifact count** per window is reached (either trigger is sufficient; both may be enabled).
- Route every emit through the **same** path: update `live_state`, bounded recovery, bus publish, and HTTP read models.

This is the **minimum** “rolling” behavior: time- and/or count-bounded windows. It is **not** optional relative to §8.1.

### 10.2 Rolling flush defaults (`v1_design_reserved` tuning)

Full **policy knobs** (defaults below) may ship when the change set touches flush configuration and ops surfaces; they **must not** fork a second aggregator.

Rolling flush converts incoming artifact/detection events into snapshot envelopes.

Recommended defaults when full policy is enabled:

```text
flush_by_time: yes
flush_interval_ms: configurable, default small enough for UI/debug freshness
flush_by_count: yes
max_artifacts_per_window: configurable
flush_on_stream_change: no, unless needed by source semantics
max_window_age_ms: hard cap
```

Rules:

- Flush output feeds the same `live_state`, Redis recovery, bus publish, and HTTP read model.
- Manual/RPC flush, if added later, must invoke the same flush path.
- A flush with no meaningful new input should not spam bus events.
- **Heartbeat / status on the bus (v1 default):** **do not** publish separate heartbeat envelopes in v1 unless an operator explicitly enables a **minimal** status envelope (e.g. `snapshot_id` absent or a dedicated `kind: status`, **no** caption/artifact payload, **no** recovery expansion). Prefer process **metrics** (`vision_window_*`, §12) and **`/healthz` / `/readyz`** for liveness. If heartbeats are added later, they must use the **same** envelope schema family with an explicit discriminator and remain **bounded**.

---

## 11. Failure modes and degraded behavior

### 11.1 Redis unavailable

If Redis is unavailable:

- `live_state` may continue in-memory.
- HTTP current endpoints may continue from live state.
- catch-up/recent endpoints should return degraded status if recovery is unavailable.
- logs/metrics must clearly show recovery persistence failures.

The service should not crash-loop solely because Redis is unavailable unless the deployment marks Redis recovery as required for readiness.

### 11.2 No active streams

Return `empty` or `stale` explicitly. Do not synthesize fake visual state.

### 11.3 Artifact store unreachable

The service may still publish envelopes containing artifact IDs if those were already known, but should surface artifact dereference status separately. It should not inline artifact content as a fallback.

### 11.4 Restart

After restart:

- `live_state` starts cold.
- `recovery_state` may seed current/recent read models.
- service should expose whether returned data came from live memory or recovery index.

---

## 12. Observability

Minimum logs/events:

- service startup config: recovery enabled, Redis URL redacted, max N, TTL, flush interval
- ingest event accepted/rejected with stream/source key
- flush event with snapshot ID, stream ID, artifact count, summary count, cursor
- Redis recovery write success/failure
- stale-window detection
- HTTP catch-up cursor expired

Minimum metrics if metrics infrastructure is present:

```text
vision_window_ingest_events_total
vision_window_snapshots_published_total
vision_window_recovery_writes_total
vision_window_recovery_write_failures_total
vision_window_live_streams
vision_window_snapshot_age_ms
vision_window_http_catchup_requests_total
vision_window_cursor_expired_total
```

---

## 13. Non-goals

- Owning **Hub** templates, **primary** WebSocket product surface in v1, or **canonical** storage of media or interpreted memory.
- Replacing **orion-vision-council**, **orion-vision-scribe**, **RDF writer**, or **journal/collapse** pipelines.
- Becoming a **durable vision database** or **embedding** store.
- Solving long-term visual memory, identity recognition, object permanence, or world modeling directly inside this service.

---

## 14. Success criteria

- After restart, a subscriber can use **HTTP catch-up** (and bounded recovery data) to obtain **meaningful** last snapshot metadata and cursors **without** requiring this service to hold raw media.
- **Bus**, **HTTP**, and later **RPC** describe the **same** projection; **no** second hidden buffer for “RPC only.”
- Documentation states **guardrails** (§3) so future features cannot silently turn this service into a memory organ.
- A consumer can tell whether data is `ok`, `empty`, `stale`, or `cursor_expired` without reading logs.
- Operators can diagnose whether the issue is upstream capture, ingestion, projection flush, Redis recovery, or subscriber catch-up.

---

## 15. Open items (for implementation plan, not blockers for this design)

- Exact **stream/source key** extraction from `VisionArtifactPayload` (`inputs` / `meta`).
- **Catalog** entries for new HTTP routes, **canonical channel** `orion:vision:windows`, and legacy **alias** rows (if any).
- Optional **SQL** history table: yes/no per deployment.
- Exact Redis key names and TTL defaults.
- Numeric defaults: global `stale_after_ms`, max wall time / max artifacts for §10.1, and `flush_interval_ms` when §10.2 is enabled.

---

## 16. References

- `docs/vision_services.md` — pipeline placement and channels overview.
- `orion/schemas/vision.py` — `VisionWindowPayload`, RPC payloads.
- `services/orion-vision-window/app/main.py` — current ingestion and flush behavior (baseline to replace incrementally per scope split).
