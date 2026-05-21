# Runtime Trace → Signal Nexus — Design Spec

**Status:** Draft — pending user review  
**Date:** 2026-05-20  
**Authors:** Operator + agent (brainstorming session)  
**Depends on:** [Organ Signal Gateway (phase 1)](2026-05-01-organ-signal-gateway-design.md), [Organ Signal Gateway — Phase 2](2026-05-01-organ-signal-gateway-phase-2-design.md), [Offboarding / inventory guide](../guides/2026-05-01-organ-signal-gateway-offboarding.md)

---

## 1. Executive summary

Orion already emits rich **execution truth** on every cognition run (`CognitionTracePayload` / `PlanExecutionResult` with step graphs, recall debug, latency, errors). The **signal nexus** (`orion-signal-gateway` + `OrionSignalV1` + Hub Organ Signals) exists structurally but is **mostly hollow**: 20 of 21 adapters are stubs, the Hub graph shows one node per organ with missed causal links, and cognition traces flow to SQL/RDF/vector/spark consumers but **not** into the signal join layer.

This spec defines **Phase 3 — Runtime trace nexus** in two shippable phases:

| Phase | Name | Outcome |
|-------|------|---------|
| **1** | Runtime trace fusion | `CognitionTraceAdapter`, dual Hub caches, chat step timeline, correlation-aware Organ Signals |
| **2** | Live mesh expansion | Real organ/runtime adapters, layer filters, useful always-on nervous-system graph |

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
| Hub inspect (phase 2b) | `SignalsInspectCache`, `/api/signals/active`, `/api/signals/trace/{otel_trace_id}`, Organ Signals tab |
| Cognition trace consumers | sql-writer, rdf-writer, vector-writer, spark-introspector |

### 2.2 What's broken / missing

| Symptom | Root cause |
|---------|------------|
| Organ Signals shows ~4 nodes, many stubs | Only `BiometricsAdapter` is real; 20 adapters emit `level: 0.5` + `"stub adapter — not yet implemented"` |
| Missed causal links, disagreeing OTEL parents | Stub signals + static registry parents without live upstream signals in window |
| No turn drill-down | Hub has reasoning/thought UI but no step graph from `CognitionTracePayload` |
| Cognition trace bypasses nexus | Gateway subscribes to `orion:cognition:*` but no adapter handles `cognition.trace` payloads |
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
- Enable **live mesh** (Phase 2): always-on runtime nervous-system graph with real adapters and layer filters
- Preserve correlation across `correlation_id`, `request_id`, `session_id`, `message_id`, `trace_id`, `otel_trace_id` where available
- Hide stub adapters by default so operators see signal, not scaffolding

### 3.2 Non-goals

- Signal persistence (graph/SQL store for signals) — out of scope
- Policy gates, routing mutation, or control-plane behavior — future work
- Full OTEL migration of all services — incremental only
- Replacing Tempo/Grafana for distributed span debugging
- Implementing all 20 stub organ adapters in Phase 1 — Phase 2 workstream

---

## 4. Architecture

### 4.1 Two planes, one fusion surface

```text
Execution truth plane          Join / summary plane           Operator surface
─────────────────────          ────────────────────           ────────────────
CognitionTracePayload    →     CognitionTraceAdapter    →     OrionSignalV1[]
PlanExecutionResult            (orion-signal-gateway)          (run + step signals)
StepExecutionResult
Recall debug               →   recall adapter (Phase 2)  →    recall_result signals
Mind run payloads          →   mind adapter (Phase 2)    →    mind_handoff signals
OTEL spans (Tempo)         →   gateway OTEL enrichment →    otel_trace_id chains

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

4. Fan-out (parallel consumers):
     a. sql-writer / rdf-writer / vector-writer / spark-introspector (unchanged)
     b. CognitionTraceAdapter → gateway → orion:signals:{organ_id}
     c. Hub CognitionTraceCache (full payload, correlation_id keyed)
     d. Hub SignalsInspectCache (signal chain by correlation_id + otel_trace_id)

5. Hub chat UI
     → step timeline from CognitionTraceCache
     → "View in Signals" from correlation index

6. Organ Signals graph
     → live: latest-per-organ (existing)
     → turn-scoped: correlation chain (new)
     → OTEL: /api/signals/trace/{otel_trace_id} (existing)
```

### 4.3 Relationship to prior signal-gateway specs

This spec is **Phase 3** — it builds on the phase-1 chassis and phase-2b Hub inspect surfaces. Phase 2 of *this* spec (live mesh) continues the organ adapter milestones from [Organ Signal Gateway — Phase 2](2026-05-01-organ-signal-gateway-phase-2-design.md) M1–M4, plus additional runtime service adapters.

---

## 5. Components

### 5.1 `CognitionTraceAdapter` (new, reference-quality)

**Path:** `orion/signals/adapters/cognition_trace.py`  
**Registered in:** `orion/signals/adapters/__init__.py` (before generic stubs that might false-match)

**Input:** Bus envelopes on `orion:cognition:trace` with `CognitionTracePayload` payload (validate via `orion.schemas.telemetry.cognition_trace.CognitionTracePayload`).

**Output:** 1 + N `OrionSignalV1` records per trace:

| Signal | `organ_id` | `signal_kind` | Role |
|--------|-----------|---------------|------|
| Run summary | `cortex_exec` | `cognition_run` | Parent node for the turn |
| Per-step child | mapped from step (§5.1.1) | `cognition_step` | One per `StepExecutionResult` |

#### 5.1.1 Step → organ mapping convention

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

#### 5.1.2 Dimensions (normalized floats in `[0.0, 1.0]` unless noted)

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

#### 5.1.3 Provenance and IDs

- `source_event_id` = envelope `correlation_id` (prefer envelope over payload copy per schema docs)
- Run `signal_id` = `make_signal_id("cortex_exec", f"{correlation_id}:run")`
- Step `signal_id` = `make_signal_id(organ_id, f"{correlation_id}:step:{order}:{step_name}")`
- Run signal `causal_parents`: registry-ordered parents from `ORGAN_REGISTRY["cortex_exec"]` (or autonomy/graph_cognition chain) that exist in `prior_signals`
- Step signal `causal_parents`: `[run_signal_id]` plus optional upstream organ parent if resolvable from step mapping and `prior_signals`
- `organ_class`: `endogenous` for run and step signals
- `summary`: compact operator string, e.g. `"verb=chat_general mode=brain steps=3 recall=1 latency=842ms"` — **no raw text**

#### 5.1.4 Privacy (non-negotiable)

Signal `summary`, `dimensions`, `notes`, and OTEL `dim.*` attributes **must not** contain:

- `final_text`, user messages, stance brief content, recall hit text, prompt templates, or LLM completions

Full payloads remain in `CognitionTraceCache` only (operator-local, TTL-bounded).

### 5.2 Registry additions (runtime nodes)

Add to `ORGAN_REGISTRY` in `orion/signals/registry.py`:

| `organ_id` | `organ_class` | `service` | `signal_kinds` | `causal_parent_organs` |
|------------|---------------|-----------|----------------|------------------------|
| `cortex_exec` | `endogenous` | `orion-cortex-exec` | `cognition_run`, `cognition_step` | `autonomy`, `graph_cognition` |
| `llm_gateway` | `endogenous` | `orion-llm-gateway` | `cognition_step`, `completion` | `chat_stance`, `cortex_exec` |
| `mind` | `endogenous` | `orion-mind` | `mind_handoff`, `cognition_step` | `chat_stance`, `recall` |
| `cortex_gateway` | `endogenous` | `orion-cortex-gateway` | `route_decision` | `[]` (Phase 2 adapter) |
| `cortex_orch` | `endogenous` | `orion-cortex-orch` | `plan_resolution` | `cortex_gateway` (Phase 2) |
| `hub` | `exogenous` | `orion-hub` | `chat_turn` | `[]` (Phase 2 adapter) |

Ensure `bus_channels` for `cortex_exec` includes `orion:cognition:trace`. Gateway already lists `orion:cognition:*` in `ORGAN_CHANNELS`.

### 5.3 Hub `CognitionTraceCache` (new)

**Path:** `services/orion-hub/scripts/cognition_trace_cache.py`

Subscribes to `orion:cognition:trace`. Indexes full `CognitionTracePayload` (+ envelope `correlation_id`, optional `otel_trace_id` from linked signals) by `correlation_id`.

**Settings** (add to `services/orion-hub/app/settings.py` + `.env_example`):

| Setting | Default | Role |
|---------|---------|------|
| `COGNITION_TRACE_CACHE_ENABLED` | `true` | Master switch |
| `COGNITION_TRACE_CACHE_MAX` | `200` | Max correlation entries |
| `COGNITION_TRACE_CACHE_TTL_SEC` | `300` | TTL per entry |
| `COGNITION_TRACE_SUBSCRIBE_CHANNEL` | `orion:cognition:trace` | Bus channel |

**API:** `GET /api/cognition/trace/{correlation_id}`

Response:

```json
{
  "correlation_id": "...",
  "verb": "chat_general",
  "mode": "brain",
  "steps": [],
  "recall_debug": {},
  "recall_used": true,
  "final_text_present": true,
  "otel_trace_id": "32-char-hex-or-null",
  "metadata": {},
  "complete": true,
  "gaps": []
}
```

- `404` when correlation id never ingested or evicted
- `503` when cache disabled
- `complete: false` + `gaps` when truncated or partial

### 5.4 Hub correlation index (extend `SignalsInspectCache`)

**Path:** `services/orion-hub/scripts/signals_inspect_cache.py`

Add secondary index: `correlation_id → List[OrionSignalV1]` ordered by `observed_at`, bounded by same TTL/window as trace cache.

**API:** `GET /api/signals/correlation/{correlation_id}`

Response mirrors `/api/signals/trace/{id}` shape but keyed by correlation and using `source_event_id` match:

```json
{
  "correlation_id": "...",
  "chain": [],
  "complete": true,
  "gaps": []
}
```

### 5.5 Chat turn step timeline

**Path:** `services/orion-hub/static/js/thought-process.js` (+ minimal template hook if needed)

New collapsible panel **Execution Steps** on chat turn detail:

- Per step: name, status badge, latency ms, services list
- Expand: truncated error, log tail (max 5 lines), recall profile if present
- Footer link: **View in Organ Signals** → opens graph with `?correlation_id=...`

Data: fetch `GET /api/cognition/trace/{correlation_id}` using id from turn metadata (`correlation_id` already propagated via websocket handler / routing debug).

### 5.6 Organ Signals UI (Phase 1 minimal)

**Path:** `services/orion-hub/static/js/organ-signals-graph-ui.js`

| Change | Behavior |
|--------|----------|
| Stub hiding | Default: exclude nodes where any `notes` entry contains `"stub adapter"` |
| Show stubs toggle | Checkbox restores stub nodes with dashed border + muted color |
| Correlation mode | When `correlation_id` query param set, build graph from `/api/signals/correlation/{id}` instead of `/api/signals/active` |
| Step chain layout | Linear left-to-right layout for correlation mode (run → steps) |

Phase 2 adds layer filter dropdown: All / Runtime / Cognition / Memory / Infra / Social / Vision / Persistence.

---

## 6. Phase boundaries

### 6.1 Phase 1 — Runtime trace fusion

**Goal:** One real `chat_general` turn is fully traceable from Hub chat → step timeline → Organ Signals correlation graph.

| Deliverable | Path |
|-------------|------|
| `CognitionTraceAdapter` | `orion/signals/adapters/cognition_trace.py` |
| Registry runtime nodes | `orion/signals/registry.py` |
| Adapter registration + tests | `orion/signals/adapters/__init__.py`, `orion/signals/adapters/tests/test_cognition_trace_adapter.py` |
| Hub cognition trace cache + API | `services/orion-hub/scripts/cognition_trace_cache.py`, `api_routes.py`, `main.py` |
| Correlation index + API | `signals_inspect_cache.py`, `api_routes.py` |
| Chat step timeline | `thought-process.js` |
| Organ Signals stub hide + correlation view | `organ-signals-graph-ui.js` |
| Hub tests | `services/orion-hub/tests/test_cognition_trace_api.py`, extend signals tests |

**Phase 1 acceptance criteria:**

1. One `chat_general` turn → Hub shows 3-step timeline with status and latency
2. Organ Signals correlation view shows `graph_cognition → chat_stance → llm_gateway` (or equivalent mapping) chained under `cortex_exec` run node
3. Stub organs hidden by default; toggle reveals them marked as stubs
4. Chat panel **View in Signals** opens correlation-scoped graph
5. No PII in emitted `OrionSignalV1` fields or OTEL span attributes
6. Targeted tests pass: `./scripts/test_service.sh orion-signal-gateway` and `./scripts/test_service.sh orion-hub`

### 6.2 Phase 2 — Live mesh expansion

**Goal:** Organ Signals becomes a useful always-on runtime graph.

| Workstream | Scope |
|------------|-------|
| Real organ adapters (priority) | `equilibrium`, `recall`, `chat_stance`, `autonomy`, `spark_introspector` per May-1 M1–M4; then remaining stubs |
| Runtime service adapters | `hub` (`chat_turn`), `cortex_gateway` (`route_decision`), `cortex_orch` (`plan_resolution`), persistence writers (`sql_writer`, `rdf_writer`, `vector_writer` persist events) |
| Layer filters | Organ Signals UI filter by layer (see §5.6) |
| Multi-signal-per-organ (optional) | Extend `/api/signals/active?layer=runtime` if product needs concurrent kinds per organ |

**Phase 2 acceptance criteria:**

1. Live mesh shows ≥8 non-stub nodes during normal chat operation
2. Layer filter **Runtime** shows cognition run + gateway + exec + llm_gateway
3. Layer filter **Cognition** shows chat_stance + recall + mind + spark_introspector
4. Missed causal parent rate < 20% on primary chain during staging smoke

---

## 7. Error handling & degradation

| Failure | Behavior |
|---------|----------|
| Malformed `CognitionTracePayload` | Adapter returns `None`; gateway logs skip; no signal emitted |
| Empty `steps[]` | Emit run signal only; `step_count=0`; note `"no_steps_in_trace"` |
| Unknown step mapping | `organ_id=cortex_exec`; note `"step_organ_fallback:{step_name}"` |
| Missing `correlation_id` | Use envelope field; if absent, synthetic id from `{verb}:{timestamp}`; note `"synthetic_correlation_id"` |
| Hub trace cache miss | API `404`; chat UI message + Tempo link if `otel_trace_id` known |
| Parent organ not in `prior_signals` | `with_missed_parent_notes()`; step emits with run parent only |
| Disagreeing parent OTEL trace_ids | Existing gateway first-wins + note (phase-1/2 spec) |
| Step failed | Step: `success=0`, `error_present=1`; run: `success=0` if any step failed |
| Recall empty | Run: `recall_used=0`; recall_debug in trace cache only |
| Cache eviction | `complete: false`, `gaps: ["trace_truncated"]` or `["correlation_truncated"]` |

---

## 8. Testing

| Test | Proves |
|------|--------|
| `test_cognition_trace_adapter_chat_general` | 3-step fixture → 1 run + 3 step signals; organ mapping; deterministic ids |
| `test_cognition_trace_adapter_failed_step` | Failed step → degraded dimensions |
| `test_cognition_trace_adapter_no_pii_in_signal` | `final_text` absent from signal fields |
| `test_signals_inspect_correlation_index` | Multiple signals indexed by `correlation_id` |
| `test_cognition_trace_cache_api` | Round-trip cognition trace API |
| `test_organ_signals_hides_stubs` | Stub nodes excluded by default |
| Gateway processor integration | Cognition envelope through full emit path |

**Staging smoke (closure for Phase 1):** One live `chat_general` turn with compose up; verify step timeline in Hub and signals on `orion:signals:cortex_exec`.

---

## 9. File map

| Area | Path |
|------|------|
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
| Trace publish | `services/orion-cortex-exec/app/main.py` |

---

## 10. Self-review checklist

- [x] No TBD/TODO placeholders in requirements
- [x] Architecture consistent with components and data flow
- [x] Phase 1 scoped to single implementation plan; Phase 2 clearly deferred
- [x] PII boundary explicit
- [x] Correlation id propagation rule stated
- [x] Acceptance criteria testable
- [x] Relationship to May-1 signal-gateway specs documented

---

## 11. Open questions (Phase 2 only)

1. **Multi-signal-per-organ API shape:** nested map by `signal_kind` vs separate correlation-only endpoint (Phase 1 uses correlation endpoint only).
2. **Hub chat_turn adapter:** emit exogenous `hub.chat_turn` signal at request start (Phase 2) for clearer graph root.
3. **Signal persistence:** if/when signals land in SQL/RDF, CognitionTraceAdapter emissions become replay source — not in scope here.
