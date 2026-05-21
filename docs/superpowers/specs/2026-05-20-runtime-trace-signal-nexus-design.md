# Runtime Trace → Signal Nexus — Design Spec

**Status:** Draft — pending user review  
**Date:** 2026-05-20  
**Authors:** Operator + agent (brainstorming session)  
**Depends on:** [Organ Signal Gateway (phase 1)](2026-05-01-organ-signal-gateway-design.md), [Organ Signal Gateway — Phase 2](2026-05-01-organ-signal-gateway-phase-2-design.md), [Offboarding / inventory guide](../guides/2026-05-01-organ-signal-gateway-offboarding.md)

---

## Design principle: no new trace truth

This spec does **not** introduce a new canonical trace format. It joins existing truth surfaces:

- `CognitionTracePayload` for execution
- `OrionSignalV1` for causal summaries
- OTEL for distributed spans
- Chat stance / mind debug payloads for stance provenance
- Hub for visualization

**Any implementation that duplicates full execution truth into signals is out of scope.**

---

## 1. Executive summary

Orion already emits rich **execution truth** on every cognition run (`CognitionTracePayload` / `PlanExecutionResult` with step graphs, recall debug, latency, errors). The **signal nexus** (`orion-signal-gateway` + `OrionSignalV1` + Hub Organ Signals) exists structurally but is **under-instrumented for runtime cognition**: 20 of 21 adapters are stubs, the Hub graph shows one node per organ with missed causal links, and cognition traces flow to SQL/RDF/vector/spark consumers but **not** into the signal join layer.

This spec defines the **Runtime Trace Nexus extension** to the Organ Signal Gateway work, delivered in two shippable milestones:

| Milestone | Name | Outcome |
|-----------|------|---------|
| **A** | Runtime trace fusion | Multi-emission adapter contract, `CognitionTraceAdapter`, dual Hub caches, chat step timeline, correlation-aware Organ Signals |
| **B** | Live mesh expansion | Real organ/runtime adapters, layer filters, useful always-on runtime nervous-system graph |

**Naming:** `orion-signals` is the **causal spine / signal nexus** — the join layer for cross-service causality and runtime observability. It is **not** the Orion control plane (routing, policy, mutation) until downstream consumers explicitly act on signals.

---

## 2. Problem statement

### 2.1 What works today

| Capability | Location |
|------------|----------|
| Runtime verbs with declared plans | e.g. `orion/cognition/verbs/chat_general.yaml` — three-step plan, `recall_profile: chat.general.v1` |
| Step execution atoms | `ExecutionStep`, `StepExecutionResult` in `orion/schemas/cortex/types.py` |
| Plan execution + trace publish | `PlanRunner` → `CognitionTracePayload` on `orion:cognition:trace` (`services/orion-cortex-exec/app/main.py`) |
| Hub chat request normalization | `services/orion-hub/scripts/chat_request_builder.py` — mode, verb, recall profile, routing debug |
| Signal gateway chassis | `services/orion-signal-gateway` — adapters, OTEL enrichment, `orion:signals:{organ_id}` publish |
| Hub inspect (signal-gateway phase 2b) | `SignalsInspectCache`, `/api/signals/active`, `/api/signals/trace/{otel_trace_id}`, Organ Signals tab |
| Cognition trace consumers | sql-writer, rdf-writer, vector-writer, spark-introspector |

### 2.2 What's broken / missing

| Symptom | Root cause |
|---------|------------|
| Organ Signals shows ~4 nodes, many stubs | Only `BiometricsAdapter` is real; 20 adapters emit `level: 0.5` + `"stub adapter — not yet implemented"` |
| Missed causal links, disagreeing OTEL parents | Stub signals + static registry parents without live upstream signals in window |
| No turn drill-down | Hub has reasoning/thought UI but no step graph from `CognitionTracePayload` |
| Cognition trace bypasses nexus | No adapter handles `cognition.trace` payloads into `OrionSignalV1` |
| Organ graph ≠ execution graph | Hub UI keys nodes by `organ_id` (latest only); a chat turn spans multiple steps/services |

### 2.3 Architectural rule: truth ownership

| Layer | Owns |
|-------|------|
| `CognitionTracePayload` | Execution truth (steps, verb, recall, final text) |
| Chat stance / mind run payloads | Stance and projection truth |
| Recall debug | Retrieval truth |
| OTEL / Tempo | Distributed span truth |
| `OrionSignalV1` | Causal-link summary + normalized dimensions |
| Hub | Operator visualization / fusion surface |

**Do not make orion-signals own all truth. Make it own the join layer.**

---

## 3. Goals & non-goals

### 3.1 Goals

- Unify runtime causality under the signal nexus without replacing canonical trace contracts
- Enable **turn drill-down**: chat turn → step timeline → correlation-scoped Organ Signals graph
- Enable **live mesh** (Milestone B): always-on runtime nervous-system graph with real adapters and layer filters
- Preserve correlation across `correlation_id`, `request_id`, `session_id`, `message_id`, `trace_id`, `otel_trace_id` where available
- Exclude stub adapter emissions in runtime correlation mode by default

### 3.2 Non-goals

- Signal persistence (graph/SQL store for signals) — out of scope
- Policy gates, routing mutation, or control-plane behavior — future work
- Full OTEL migration of all services — incremental only
- Replacing Tempo/Grafana for distributed span debugging
- Implementing all 20 stub organ adapters in Milestone A — Milestone B workstream
- Duplicating full execution truth into `OrionSignalV1` dimensions or summaries

---

## 4. Architecture

### 4.1 Two planes, one fusion surface

```text
Execution truth plane          Join / summary plane           Operator surface
─────────────────────          ────────────────────           ────────────────
CognitionTracePayload    →     CognitionTraceAdapter    →     OrionSignalV1[]
PlanExecutionResult            (orion-signal-gateway)          (run + step signals)
StepExecutionResult
Recall debug               →   recall adapter (Milestone B) → recall_result signals
Mind run payloads          →   mind adapter (Milestone B)   → mind_handoff signals
OTEL spans (Tempo)         →   gateway OTEL enrichment    → otel_trace_id chains

Hub fusion:
  correlation_id index  ← cognition trace signals + chat turn context
  otel_trace_id index   ← existing signals_inspect_cache
  organ_id index        ← existing latest-per-organ (live mesh)
```

### 4.2 End-to-end flow (one `chat_general` turn)

```text
1. Hub.build_cortex_chat_request()
     → mode, verb, recall_profile, routing_debug
     → correlation_id assigned

2. cortex-orch → cortex-exec PlanRunner
     → Step 0: MetacogContextService
     → Step 1: LLMGatewayService (stance brief)
     → Step 2: LLMGatewayService (chat_general)
     → PlanExecutionResult

3. cortex-exec publishes CognitionTracePayload → orion:cognition:trace
     (metadata enriched per §5.2)

4. Fan-out (parallel consumers):
     a. sql-writer / rdf-writer / vector-writer / spark-introspector (unchanged)
     b. CognitionTraceAdapter → gateway → orion:signals:{organ_id} (1+N emissions)
     c. Hub CognitionTraceCache (full payload, correlation_id keyed)
     d. Hub SignalsInspectCache (signal chain by correlation_id + otel_trace_id)

5. Hub chat UI
     → step timeline from CognitionTraceCache
     → "View in Signals" from correlation index

6. Organ Signals graph
     → live: latest-per-organ (existing; node id = organ_id)
     → turn-scoped: correlation chain (new; node id = signal_id)
     → OTEL: /api/signals/trace/{otel_trace_id} (existing)
```

### 4.3 Relationship to prior signal-gateway specs

This spec extends the [Organ Signal Gateway (phase 1)](2026-05-01-organ-signal-gateway-design.md) chassis and [phase 2b Hub inspect](2026-05-01-organ-signal-gateway-phase-2-design.md) surfaces. **Milestone B** continues the organ adapter milestones from signal-gateway phase 2 M1–M4, plus additional runtime service adapters.

---

## 5. Components

### 5.1 Adapter contract change (multi-emission)

The existing adapter interface is single-signal shaped:

```python
adapt(...) -> OrionSignalV1 | None
```

`CognitionTraceAdapter` is the **first multi-emission adapter**. The gateway processor must accept:

```python
OrionSignalV1 | list[OrionSignalV1] | None
```

and **normalize adapter output to a list** before publishing each signal independently (OTEL span, `orion:signals:{organ_id}`, window update).

**Implementation order (Milestone A gate):**

1. Update `OrionSignalAdapter` base contract in `orion/signals/adapters/base.py`
2. Update `SignalProcessor` in `services/orion-signal-gateway/app/processor.py` to flatten list emissions
3. Add processor tests proving multi-emission publish path
4. **Then** land `CognitionTraceAdapter`

Do **not** jam multiple step signals into one summary blob.

### 5.2 `CognitionTracePayload` metadata enrichment

Before or alongside `CognitionTraceAdapter`, `cortex-exec` must preserve useful `PlanExecutionResult.metadata` into `CognitionTracePayload.metadata`.

**At minimum:**

| Field | Purpose |
|-------|---------|
| Routing metadata | mode, verb, route intent from Hub/cortex path |
| `recall_profile` | selected recall profile name |
| `chat_stance_debug_present` | bool — stance brief artifacts exist |
| `mind_handoff_quality` | enum/score if mind handoff occurred |
| `reasoning_present` | bool — reasoning/thinking captured |
| `thinking_present` | bool — provider thinking block captured |
| `canonical_final_step_name` | e.g. `llm_chat_general` |
| `canonical_thought_source` | e.g. `reasoning_trace`, `metacog_traces`, `inline_think` |
| `session_id`, `message_id` | when available from request context |
| `root_correlation_id` | only if a hop uses a different id than envelope (see §5.8) |

Raw PII/text should **not** be copied into metadata unless already present in the canonical trace payload and required for operator drill-down (prefer presence flags).

**Path:** `services/orion-cortex-exec/app/main.py` (trace publish block) + `PlanRunner` metadata assembly.

### 5.3 `CognitionTraceAdapter` (new, reference-quality)

**Path:** `orion/signals/adapters/cognition_trace.py`  
**Registered in:** `orion/signals/adapters/__init__.py` (before generic stubs that might false-match)

**Input:** Bus envelopes on `orion:cognition:trace` with `CognitionTracePayload` payload (validate via `orion.schemas.telemetry.cognition_trace.CognitionTracePayload`).

**Output:** `list[OrionSignalV1]` — 1 run summary + N step signals per trace:

| Signal | `organ_id` | `signal_kind` | Role |
|--------|-----------|---------------|------|
| Run summary | `cortex_exec` | `cognition_run` | Parent node for the turn |
| Per-step child | mapped from step (§5.3.1) | `cognition_step` | One per `StepExecutionResult` |

#### 5.3.1 Step → organ mapping convention

Apply in order:

1. **Step name table** (explicit):

   | `step_name` | `organ_id` |
   |-------------|------------|
   | `collect_metacog_context` | `graph_cognition` |
   | `synthesize_chat_stance_brief` | `chat_stance` |
   | `llm_chat_general` | `llm_gateway` |

2. **Service prefix match** on primary service (`services[0]` if present):

   | Service prefix | `organ_id` |
   |----------------|------------|
   | `RecallService` | `recall` |
   | `Mind` | `mind` |
   | `LLMGatewayService` | `llm_gateway` |
   | `MetacogContextService` | `graph_cognition` |
   | `AgentChainService` | `agent_chain` |
   | `PlannerReactService` | `planner` |

3. **Fallback:** `cortex_exec` with note `step_organ_fallback:{step_name}`

#### 5.3.2 Dimensions (normalized floats in `[0.0, 1.0]` unless noted)

**Run signal (`cognition_run`):**

| Key | Derivation |
|-----|------------|
| `success` | `1.0` if all steps succeeded and run status success; else `0.0` |
| `step_count` | `min(len(steps), 20) / 20.0` |
| `latency_level` | `min(total_latency_ms, 120_000) / 120_000.0` |
| `recall_used` | `1.0` if `recall_used` else `0.0` |
| `reasoning_present` | `1.0` if reasoning/metacog content present in metadata or steps |
| `final_text_present` | `1.0` if `final_text` non-empty |

**Step signal (`cognition_step`):**

| Key | Derivation |
|-----|------------|
| `success` | `1.0` if `status == "success"` else `0.0` |
| `latency_level` | `min(step.latency_ms or 0, 60_000) / 60_000.0` |
| `error_present` | `1.0` if `error` non-empty else `0.0` |
| `service_count` | `min(len(services), 5) / 5.0` |

#### 5.3.3 Provenance and IDs

- `source_event_id` = envelope `correlation_id` (prefer envelope over payload copy per schema docs)
- Run `signal_id` = `make_signal_id("cortex_exec", f"{correlation_id}:run")`
- Step `signal_id` = `make_signal_id(organ_id, f"{correlation_id}:step:{order}:{step_name}")`
- Run signal `causal_parents`: registry-ordered parents from `ORGAN_REGISTRY["cortex_exec"]` that exist in `prior_signals`
- Step signal `causal_parents`: `[run_signal_id]` plus optional upstream organ parent if resolvable from step mapping and `prior_signals`
- `organ_class`: `endogenous` for run and step signals
- `summary`: compact operator string, e.g. `"verb=chat_general mode=brain steps=3 recall=1 latency=842ms"` — **no raw text**

#### 5.3.4 Privacy (non-negotiable for signals)

Signal `summary`, `dimensions`, `notes`, and OTEL `dim.*` attributes **must not** contain:

- `final_text`, user messages, stance brief content, recall hit text, prompt templates, or LLM completions

Full payloads may exist in `CognitionTraceCache` subject to §5.5 security rules.

### 5.4 Registry additions (runtime nodes)

Add to `ORGAN_REGISTRY` in `orion/signals/registry.py`:

| `organ_id` | `organ_class` | `service` | `signal_kinds` | `causal_parent_organs` |
|------------|---------------|-----------|----------------|------------------------|
| `cortex_exec` | `endogenous` | `orion-cortex-exec` | `cognition_run`, `cognition_step` | `autonomy`, `graph_cognition` |
| `llm_gateway` | `endogenous` | `orion-llm-gateway` | `cognition_step`, `completion` | `chat_stance`, `cortex_exec` |
| `mind` | `endogenous` | `orion-mind` | `mind_handoff`, `cognition_step` | `chat_stance`, `recall` |
| `cortex_gateway` | `endogenous` | `orion-cortex-gateway` | `route_decision` | `[]` (Milestone B adapter) |
| `cortex_orch` | `endogenous` | `orion-cortex-orch` | `plan_resolution` | `cortex_gateway` (Milestone B) |
| `hub` | `exogenous` | `orion-hub` | `chat_turn` | `[]` (Milestone B adapter) |

Ensure `bus_channels` for `cortex_exec` includes `orion:cognition:trace`.

#### Preflight requirement (gateway subscription)

**Do not assume** `orion:cognition:*` is already in gateway channels.

Verify `services/orion-signal-gateway` currently subscribes to `orion:cognition:trace` or `orion:cognition:*` in `ORGAN_CHANNELS` (see `app/settings.py`). If not, add the channel explicitly as part of **Milestone A**. Confirm in staging that a live `chat_general` turn produces a gateway-handled envelope before accepting the adapter milestone.

### 5.5 Hub `CognitionTraceCache` (new)

**Path:** `services/orion-hub/scripts/cognition_trace_cache.py`

Subscribes to `orion:cognition:trace`. Indexes `CognitionTracePayload` (+ envelope `correlation_id`, optional `otel_trace_id` from linked signals) by `correlation_id`.

**Settings** (add to `services/orion-hub/app/settings.py` + `.env_example`):

| Setting | Default | Role |
|---------|---------|------|
| `COGNITION_TRACE_CACHE_ENABLED` | `true` | Master switch |
| `COGNITION_TRACE_CACHE_MAX` | `200` | Max correlation entries |
| `COGNITION_TRACE_CACHE_TTL_SEC` | `300` | TTL per entry |
| `COGNITION_TRACE_SUBSCRIBE_CHANNEL` | `orion:cognition:trace` | Bus channel |
| `COGNITION_TRACE_API_DEBUG` | `false` | When true, API may return full step payloads including text |

**API:** `GET /api/cognition/trace/{correlation_id}`

Default response (when `COGNITION_TRACE_API_DEBUG=false`):

```json
{
  "correlation_id": "...",
  "verb": "chat_general",
  "mode": "brain",
  "steps": [
    {
      "step_name": "collect_metacog_context",
      "order": 0,
      "status": "success",
      "latency_ms": 120,
      "services": ["MetacogContextService"],
      "error_present": false,
      "prompt_template_present": false
    }
  ],
  "recall_used": true,
  "final_text_present": true,
  "otel_trace_id": "32-char-hex-or-null",
  "metadata": {},
  "complete": true,
  "gaps": []
}
```

When `COGNITION_TRACE_API_DEBUG=true`, API may additionally return `recall_debug`, log tails, and redacted artifact keys — never expose to unauthenticated callers.

- `404` when correlation id never ingested or evicted
- `503` when cache disabled
- `complete: false` + `gaps` when truncated or partial

#### CognitionTraceCache security

The cache is **operator-local**, **TTL-bounded**, and **must not be exposed to unauthenticated users**. The API must follow the same auth posture as existing operator/debug routes (e.g. `/api/signals/trace/{id}`).

The cache **must not be persisted** by default (in-memory only, same class as `SignalsInspectCache`).

Because full trace payloads may contain `final_text`, prompts, logs, and recall debug:

- Default API returns **presence flags** and **redacted step summaries** only
- Full text fields require explicit `COGNITION_TRACE_API_DEBUG=true` (operator-only, never default in production compose)

### 5.6 Hub correlation index (extend `SignalsInspectCache`)

**Path:** `services/orion-hub/scripts/signals_inspect_cache.py`

Add secondary index: `correlation_id → List[OrionSignalV1]` ordered by `observed_at`, bounded by same TTL/window as trace cache.

**API:** `GET /api/signals/correlation/{correlation_id}`

Response mirrors `/api/signals/trace/{id}` shape but keyed by correlation and using `source_event_id` match:

```json
{
  "correlation_id": "...",
  "chain": [],
  "complete": true,
  "gaps": [],
  "hidden_stubs": 0
}
```

### 5.7 Chat turn step timeline

**Path:** `services/orion-hub/static/js/thought-process.js` (+ minimal template hook if needed)

New collapsible panel **Execution Steps** on chat turn detail:

- Per step: name, status badge, latency ms, services list
- Expand: truncated error, log tail (max 5 lines) — only when debug mode or step error present
- Footer link: **View in Organ Signals** → opens graph with `?correlation_id=...`

Data: fetch `GET /api/cognition/trace/{correlation_id}` using id from turn metadata (must match §5.8 propagation gate).

### 5.8 Correlation ID propagation (Milestone A acceptance gate)

**Milestone A cannot be accepted** unless Hub chat turn metadata exposes the **same** `correlation_id` used by:

1. Cortex request envelope (`PlanExecutionRequest` / chat dispatch)
2. `CognitionTracePayload` envelope (`correlation_id` on `BaseEnvelope`)
3. `CognitionTraceAdapter` `source_event_id`
4. `GET /api/cognition/trace/{correlation_id}`
5. `GET /api/signals/correlation/{correlation_id}`

If any hop uses a different id, add an explicit mapping field **`root_correlation_id`** on the downstream payload/metadata and document the mapping in Hub turn metadata. Chat UI and APIs resolve via `root_correlation_id` when present.

**Verification:** one live `chat_general` turn; assert single id (or documented mapping) across all five hops in logs or API responses.

### 5.9 Organ Signals UI (Milestone A minimal)

**Path:** `services/orion-hub/static/js/organ-signals-graph-ui.js`

#### Graph identity rule

| Mode | Node id | Label | Edges |
|------|---------|-------|-------|
| **Live** | `organ_id` | `organ_id` + latest `signal_kind` | `causal_parents` → resolve parent `organ_id` via latest signal map |
| **Correlation** | `signal_id` | `organ_id` + `signal_kind` | `causal_parents` signal ids directly |

Correlation mode **must not** key nodes by `organ_id` — a single turn may emit multiple signals for the same organ (e.g. run + step both on `cortex_exec`).

#### Stub exclusion (runtime correlation mode)

**Runtime correlation mode must exclude stub adapter emissions by default.**

A signal is considered **stubbed** if any of:

- `notes[]` contains `"stub adapter"`
- `dimensions` is exactly the generic placeholder shape `{level: 0.5, confidence: 0.5}` (and no other keys)
- `signal_kind` comes from a known stub adapter **and** lacks `source_event_id`

UI shows compact warning: **`Hidden stubs: N`** with toggle to reveal. Revealed stubs render with dashed border + muted color.

| Change | Behavior |
|--------|----------|
| Correlation mode | When `correlation_id` query param set, build graph from `/api/signals/correlation/{id}` with `signal_id` nodes |
| Stub hiding | Apply stub rules above; live mode may optionally hide stubs (same toggle) |
| Step chain layout | Linear left-to-right layout for correlation mode (run → steps) |

Milestone B adds layer filter dropdown: All / Runtime / Cognition / Memory / Infra / Social / Vision / Persistence.

---

## 6. Milestone boundaries

### 6.1 Milestone A — Runtime trace fusion

**Goal:** One real `chat_general` turn is fully traceable from Hub chat → step timeline → Organ Signals correlation graph.

| Deliverable | Path |
|-------------|------|
| Multi-emission adapter contract | `orion/signals/adapters/base.py`, `services/orion-signal-gateway/app/processor.py` |
| Gateway preflight: cognition channel subscription | `services/orion-signal-gateway/app/settings.py` |
| `CognitionTracePayload.metadata` enrichment | `services/orion-cortex-exec/app/main.py`, `PlanRunner` |
| `CognitionTraceAdapter` | `orion/signals/adapters/cognition_trace.py` |
| Registry runtime nodes | `orion/signals/registry.py` |
| Adapter + processor tests | `orion/signals/adapters/tests/`, `services/orion-signal-gateway/app/tests/` |
| Hub cognition trace cache + API | `services/orion-hub/scripts/cognition_trace_cache.py`, `api_routes.py`, `main.py` |
| Correlation index + API | `signals_inspect_cache.py`, `api_routes.py` |
| Correlation ID propagation | Hub websocket handler / turn metadata |
| Chat step timeline | `thought-process.js` |
| Organ Signals correlation view + stub rules | `organ-signals-graph-ui.js` |
| Hub tests | `services/orion-hub/tests/test_cognition_trace_api.py`, extend signals tests |

**Milestone A acceptance criteria:**

1. Multi-emission processor tests pass before adapter merges
2. Gateway preflight confirmed: live cognition trace envelopes reach processor
3. One `chat_general` turn → Hub shows 3-step timeline with status and latency
4. Organ Signals correlation view uses **`signal_id` nodes** and shows step chain (`graph_cognition → chat_stance → llm_gateway` or equivalent)
5. Stub signals hidden in correlation mode; **`Hidden stubs: N`** + toggle works
6. **Correlation ID gate (§5.8):** same id (or `root_correlation_id` mapping) across all five hops
7. No PII in emitted `OrionSignalV1` fields or OTEL span attributes
8. Cognition trace API default response does not leak full text without debug flag
9. Targeted tests pass: `./scripts/test_service.sh orion-signal-gateway` and `./scripts/test_service.sh orion-hub`

### 6.2 Milestone B — Live mesh expansion

**Goal:** Organ Signals becomes a useful always-on runtime nervous-system graph.

| Workstream | Scope |
|------------|-------|
| Real organ adapters (priority) | `equilibrium`, `recall`, `chat_stance`, `autonomy`, `spark_introspector` per signal-gateway phase 2 M1–M4; then remaining stubs |
| Runtime service adapters | `hub` (`chat_turn`), `cortex_gateway` (`route_decision`), `cortex_orch` (`plan_resolution`), persistence writers (`sql_writer`, `rdf_writer`, `vector_writer` persist events) |
| Layer filters | Organ Signals UI filter by layer (see §5.9) |
| Multi-signal-per-organ (optional) | Extend `/api/signals/active?layer=runtime` if product needs concurrent kinds per organ |

**Milestone B acceptance criteria:**

1. Live mesh shows ≥8 non-stub nodes during normal chat operation
2. Layer filter **Runtime** shows cognition run + gateway + exec + llm_gateway
3. Layer filter **Cognition** shows chat_stance + recall + mind + spark_introspector
4. Missed causal parent rate < 20% on primary chain during staging smoke

---

## 7. Error handling & degradation

| Failure | Behavior |
|---------|----------|
| Malformed `CognitionTracePayload` | Adapter returns `None`; gateway logs skip; no signal emitted |
| Empty `steps[]` | Emit run signal only (list of 1); `step_count=0`; note `"no_steps_in_trace"` |
| Unknown step mapping | `organ_id=cortex_exec`; note `"step_organ_fallback:{step_name}"` |
| Missing `correlation_id` | Use envelope field; if absent, synthetic id from `{verb}:{timestamp}`; note `"synthetic_correlation_id"` |
| Hub trace cache miss | API `404`; chat UI message + Tempo link if `otel_trace_id` known |
| Parent organ not in `prior_signals` | `with_missed_parent_notes()`; step emits with run parent only |
| Disagreeing parent OTEL trace_ids | Existing gateway first-wins + note (signal-gateway phase 1/2 spec) |
| Step failed | Step: `success=0`, `error_present=1`; run: `success=0` if any step failed |
| Recall empty | Run: `recall_used=0`; recall_debug in trace cache only (debug-gated API) |
| Cache eviction | `complete: false`, `gaps: ["trace_truncated"]` or `["correlation_truncated"]` |
| Adapter returns list | Processor publishes each signal independently; partial publish on mid-list failure is logged and counted |

---

## 8. Testing

| Test | Proves |
|------|--------|
| `test_gateway_processor_multi_emission` | List return → N independent publishes |
| `test_cognition_trace_adapter_chat_general` | 3-step fixture → list of 4 signals; organ mapping; deterministic ids |
| `test_cognition_trace_adapter_failed_step` | Failed step → degraded dimensions |
| `test_cognition_trace_adapter_no_pii_in_signal` | `final_text` absent from signal fields |
| `test_signals_inspect_correlation_index` | Multiple signals indexed by `correlation_id` |
| `test_cognition_trace_cache_api_redacted_default` | Default API omits full text |
| `test_cognition_trace_cache_api_debug_mode` | Debug flag returns expanded fields (auth-gated) |
| `test_organ_signals_correlation_uses_signal_id` | Graph nodes keyed by `signal_id` in correlation mode |
| `test_organ_signals_hides_stubs` | Stub signals excluded; count in `hidden_stubs` |
| `test_correlation_id_propagation` | End-to-end id match across hub → exec → adapter → APIs |
| Gateway processor integration | Cognition envelope through full multi-emission emit path |

**Staging smoke (closure for Milestone A):** One live `chat_general` turn with compose up; verify step timeline in Hub, signals on `orion:signals:cortex_exec`, and §5.8 correlation gate.

---

## 9. File map

| Area | Path |
|------|------|
| Adapter base contract | `orion/signals/adapters/base.py` |
| Cognition trace schema | `orion/schemas/telemetry/cognition_trace.py` |
| Step types | `orion/schemas/cortex/types.py` |
| Signal models + registry | `orion/signals/models.py`, `orion/signals/registry.py` |
| New adapter | `orion/signals/adapters/cognition_trace.py` |
| Gateway processor | `services/orion-signal-gateway/app/processor.py` |
| Gateway settings | `services/orion-signal-gateway/app/settings.py` |
| Hub signals cache | `services/orion-hub/scripts/signals_inspect_cache.py` |
| Hub cognition cache | `services/orion-hub/scripts/cognition_trace_cache.py` (new) |
| Hub APIs | `services/orion-hub/scripts/api_routes.py` |
| Chat UI | `services/orion-hub/static/js/thought-process.js` |
| Organ Signals UI | `services/orion-hub/static/js/organ-signals-graph-ui.js` |
| chat_general verb | `orion/cognition/verbs/chat_general.yaml` |
| Trace publish + metadata | `services/orion-cortex-exec/app/main.py` |

---

## 10. Self-review checklist

- [x] No unresolved placeholders in **Milestone A** requirements
- [x] Architecture consistent with components and data flow
- [x] Milestone A scoped to single implementation plan; Milestone B clearly deferred
- [x] PII boundary explicit (signals + cognition cache API)
- [x] Correlation id propagation gate explicit (§5.8)
- [x] Multi-emission adapter contract explicit (§5.1)
- [x] Graph identity rule explicit (§5.9)
- [x] Acceptance criteria testable
- [x] Relationship to May-1 signal-gateway specs documented
- [x] "No new trace truth" principle stated

---

## 11. Open questions (Milestone B only)

1. **Multi-signal-per-organ API shape:** nested map by `signal_kind` vs correlation-only endpoint (Milestone A uses correlation endpoint only).
2. **Hub chat_turn adapter:** emit exogenous `hub.chat_turn` signal at request start for clearer graph root.
3. **Signal persistence:** if/when signals land in SQL/RDF, CognitionTraceAdapter emissions become replay source — not in scope here.
