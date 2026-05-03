# Hub — OpenTelemetry traces and metrics (unified observability) — design

**Date:** 2026-05-03  
**Status:** Draft — ready for review; **no implementation committed** until a follow-up implementation plan exists.  
**Scope:** How operators (and later power users) **see OTEL traces alongside metrics**, with Hub as the primary **entry surface**; backend stores and collectors are specified at the integration boundary only.

---

## 1. Purpose

Deliver a **credible path** from today’s fragmented signals (OTLP export, Prometheus scrape on the collector, Hub signal cache keyed by `otel_trace_id`) to a **single operator story**: open Hub (or Hub-linked surfaces), navigate by **trace id** / **time** / **correlation context**, and inspect **distributed traces** next to **relevant metrics**.

This document **does not** redefine signal-gateway span semantics (see existing gateway README and organ offboarding guide); it defines **where observability lands in product** and **what is explicitly deferred** for a later brainstorm.

---

## 2. Current state (anchor)

| Concern | Where it lives today |
|--------|----------------------|
| Span export from gateway app | `services/orion-signal-gateway/app/instrumentation.py` — OTLP gRPC if `OTEL_EXPORTER_OTLP_ENDPOINT` set; else console or drop. |
| Span attributes + ids on mesh signals | `services/orion-signal-gateway/app/processor.py` — `otel_trace_id`, `otel_span_id`, parent linkage on `OrionSignalV1`. |
| Collector sidecar | `services/orion-signal-gateway/otel/collector-config.yaml` — traces → `debug` exporter; metrics → `debug` + **Prometheus** on `0.0.0.0:8889` (`namespace: orion`). OTLP to Jaeger **commented**. |
| Hub — causal signal view | `GET /api/signals/active`, `GET /api/signals/trace/{trace_id}` — in-memory / bounded cache; **Orion signal chain**, not full OTEL span trees (`services/orion-hub/scripts/api_routes.py`, `signals_inspect_cache.py`). |
| Hub — other “trace” UI | Agent / reasoning traces (cortex), not OTEL (`templates/index.html`, `static/js/app.js`, `agent-trace.js`). |
| Hub — “telemetry” named APIs | `/api/substrate/telemetry-summary` etc. — **graph review runtime** telemetry (SQL), **not** OpenTelemetry. |

**Join key:** `otel_trace_id` (hex) is the natural link between Hub signal payloads and a trace backend once OTLP pipelines persist traces.

### 2.1 ID taxonomy

Several concepts are named “trace” in this repo. **Do not** conflate them in URLs, caches, or UI wiring.

| Identifier | Meaning |
|------------|---------|
| **`otel_trace_id`** | 32-character **lowercase hex** OpenTelemetry trace id. Used for **Tempo** / **Jaeger** lookup. |
| **`otel_span_id`** | 16-character **lowercase hex** OpenTelemetry span id for the current span. |
| **`otel_parent_span_id`** | Parent OTEL span id when inferred from prior signal state (gateway processor). |
| **Bus `correlation_id` / envelope “trace” fields** | Orion envelope / runtime **correlation** id (often UUID-shaped). **Not** an OTEL trace id unless explicitly propagated and documented as such. |
| **Agent / reasoning “trace”** | Cortex / debug **execution** trace (steps, reasoning payloads). **Not** an OTEL span tree. |

**Failure mode to avoid:** wiring a UUID-shaped bus `correlation_id` into a Tempo/Grafana trace-id URL and expecting a hit. Join paths must use **`otel_trace_id`** for OTEL backends unless a separate, documented mapping exists.

---

## 3. Goals and non-goals

### 3.1 Goals

1. **End-to-end OTEL traces** persisted in **Grafana Tempo** (default trace backend), with **Prometheus** (or Mimir later) for metrics and **Grafana Explore / dashboards** as the default operator UI for “trace + metrics in one surface.” **Jaeger** remains **optional / dev-only** for one-box trace viewing when faster than wiring Grafana.
2. **Metrics** remain scrape- or remote-write–based; **same time window** as trace investigation should be discoverable from Hub (link, embed, or API-backed chart — phased).
3. **Hub** becomes the **default entry** for operators: at minimum **deep links** into Grafana (Explore or dashboard); at maximum **embedded or native** combined views (phased).
4. **Preserve** `GET /api/signals/trace/{trace_id}` as the **Orion causal** debugging surface; **do not** conflate it with OTEL span completeness (see §2.1).

### 3.2 Non-goals (this spec)

- Replacing Grafana/Tempo with a **full** custom observability product inside Hub.
- Changing **substrate graph review** telemetry schemas or merging them into OTEL without a separate design.
- Defining **SLOs and alerting** policy (only noted as backlog).

### 3.3 Hub must not mirror OTEL span stores (Phases 0–2)

**Hard guardrail:** Through **Phase 2**, Hub **must not** persist or mirror **full OTEL span payloads** (no second observability database). Hub **may** store **URL templates**, **trace ids**, **small summaries**, and **time windows** only. Full span trees stay in **Tempo** (and Grafana). Deferred expansion of Hub-side span caching is listed in §7.3 with the same boundary.

---

## 4. Architecture options (recap)

| Option | Idea | Pros | Cons |
|--------|------|------|------|
| **A — Embed / deep link** | Grafana Explore (default) or Tempo/Jaeger UI in iframe; Hub supplies **otel** trace id, time range, `service.name`. | Fast, full-featured UI, exemplars if enabled; **metrics + traces** in Grafana. | Auth, CORS, cookie domains, “two apps” feel. |
| **B — Hub links only** | Hub shows “View trace in Grafana Explore” with URL templates; no iframe. | Minimal code; clear security boundary. | Context switching; no single-pane chrome. |
| **C — Native Hub** | Hub backend proxies Tempo + Prom; custom timeline + charts. | Unified auth and layout. | Ongoing UX and API surface ownership. |

**Recommendation:** **B first**, then **A** where security review allows, then **C** only for **narrow** golden paths (e.g. one trace summary + fixed metric panels).

---

## 5. Phased delivery (in scope by phase)

### Phase 0 — Documentation and wiring truth

- Document **required** env vars and URLs: gateway OTLP → collector → **Tempo**; Prometheus scrape target for collector **`:8889`** (or service discovery equivalent); Grafana URLs for Explore/dashboards.
- Enable (in dev/staging) **OTLP exporter** on the collector to **Tempo** (replace or extend the commented Jaeger OTLP block in `collector-config.yaml` as appropriate for the chosen stack).

#### Phase 0 — Truth gap: compose vs application OTLP

The gateway **compose** file defines the **otel-collector** sidecar, and `.env.example` documents `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_CONSOLE_EXPORT` / `OTEL_DIMENSION_ALLOWLIST`, but the **`orion-signal-gateway` service** environment in compose **does not** currently pass those variables into the app container—only service / bus / log style settings are wired. **Result:** a developer can `docker compose up`, see a collector, and still get **no OTLP export from the gateway app** unless they inject env another way. That is the gap between **“docs say it works”** and **“span export actually happens when we run compose.”**

**Explicit Phase 0 task — wire gateway compose env through** (defaults match a typical local mesh: app → sidecar OTLP gRPC):

```yaml
# Example intent for orion-signal-gateway service environment (exact YAML placement is implementation detail):
OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://otel-collector:4317}
OTEL_CONSOLE_EXPORT=${OTEL_CONSOLE_EXPORT:-false}
OTEL_DIMENSION_ALLOWLIST=${OTEL_DIMENSION_ALLOWLIST:-["level","trend","volatility","confidence"]}
```

If the compose parser or shell strips the JSON allowlist default, **quote** the default value so `OTEL_DIMENSION_ALLOWLIST` remains valid JSON for `app/settings.py`.

### Phase 1 — Trace persistence + operator links from Hub

- Deploy **Tempo** + **Grafana** (+ Prometheus as already aligned with collector metrics); record any deviation in a short ADR.
- Add Hub UI or debug affordance: given **`otel_trace_id`** (§2.1), **open trace** via a **stable Grafana Explore URL template** (Jaeger UI optional for local/dev-only shortcuts).
- Ensure **32-char lowercase hex** `otel_trace_id` matches what Tempo/Grafana Explore expect (normalize in one place).

### Phase 2 — Unified operator surface (metrics + traces)

- Single **operator page** (new template or Hub route): **trace** (iframe or linked panel) + **metrics** (second iframe, Grafana dashboard uid, or static PromQL/Grafana link with time range).
- Time range: default **last 15m** or derived from signal `as_of` when available.

### Phase 3 — Optional native Hub panels (narrow)

- Hub backend routes that **proxy** read-only Tempo trace-by-id + Prometheus query_range (or Grafana API if org standardizes on it).
- **Fixed** metric set and **simplified** span timeline — not feature parity with Grafana.

---

## 6. In-scope vs out-of-scope per phase

| Phase | In scope | Explicitly out of scope (see §7) |
|-------|----------|-----------------------------------|
| 0 | Docs; compose/k8s snippets; **gateway compose passes OTEL_* into app** (§5 Phase 0 truth gap); collector OTLP to **Tempo** in non-prod | Prod hardening, SSO, retention |
| 1 | Tempo + Grafana; **Grafana Explore** deep links from Hub | Iframes, PromQL proxy, custom charts |
| 2 | Combined operator page with trace + metrics (embed or dual link) | Native waterfall renderer, full RBAC model |
| 3 | Narrow proxy + custom UI | Full Grafana replacement, alerting |

---

## 7. Future brainstorm backlog (deferred / pick up later)

Use this list as the **agenda source** for a follow-on brainstorming session. Items are **not rejected**; they are **out of scope for the phased plan above** until explicitly pulled in.

### 7.1 Product and UX

- **Full native OTEL waterfall** in Hub (span parent/child, async links, service graph) with **Grafana-level** interactions.
- **Unified chrome**: Hub header/nav on embedded Grafana; postMessage theming sync.
- **Per-turn chat integration**: auto-surface “view trace” on every assistant message when `otel_trace_id` propagates from cortex path (requires **correlation** design: chat `correlation_id` ↔ trace id guarantees).
- **Mobile / low-bandwidth** observability mode (summary-only traces).
- **End-user vs operator** surfaces: whether any OTEL view is ever shown to non-operator roles.

### 7.2 Security, identity, and tenancy

- **RBAC** matrix: who may hit `/api/signals/*`, substrate APIs, and future trace/metrics proxies.
- **SSO / OIDC** for **Grafana** (and optionally Jaeger **only** if retained as dev UI) aligned with Hub auth; **iframe** cookie and **Content-Security-Policy** strategy.
- **Multi-tenant** trace isolation (namespace, headers, org id on spans).
- **Audit log** of “who opened which trace id” for regulated environments.

### 7.3 Backend and data plane

- **Hub read proxy** to Tempo + Prometheus with **rate limits**, **query budgets**, and **timeout** semantics (full design of §5 Phase 3).
- **Mimir** or **long-term** metrics store vs raw Prometheus only.
- **OpenTelemetry Logs** pipeline and correlation with traces (`trace_id` in log records).
- **Tail sampling** and **head sampling** policies in collector; **sampling overrides** per organ or per route.
- **Exemplars**: Prom metrics linking to `trace_id` for click-through from metric spike to trace.
- **Hub trace cache** storing **full OTEL span payloads** (beyond §3.3): today bounded **Orion signal** cache only; any expansion needs retention, PII, cost, and **explicit** repeal of the Phase 0–2 guardrail.

### 7.4 Infra and mesh

- **GPU / DCGM** and **mesh Prometheus** scrapes (commented blocks in `collector-config.yaml`) wired into the **same** operator dashboards as Phase 2.
- **Service mesh** metrics (if mesh is introduced) in the same Grafana folder as Orion services.
- **Synthetic probes** from Hub or CI publishing **known** traces for **drift detection**.
- **Cross-region** and **federated** Grafana (Mimir/Tempo) for multi-cluster.

### 7.5 SRE and policy

- **SLO definitions** (latency, error rate, saturation) and **PrometheusRule** / Alertmanager routing.
- **Cost controls**: retention UI, sampling UI in Hub, cardinality dashboards.
- **Production** removal or gating of collector **debug** exporter verbosity.

### 7.6 Semantic and cross-domain joins

- **Substrate graph review telemetry** (`GraphReviewTelemetry*`) **on one page with OTEL** — shared time slider, **no schema merge** without an explicit “substrate observability” spec.
- **Memory graph** / journal / annotator events **linked** to `otel_trace_id` for “what cognition happened in this trace” (depends on other memory-graph specs).
- **Trace-driven regression**: golden trace IDs or span shapes after releases.

### 7.7 ADRs and defaults (repo stance)

**Default stack for implementation planning** (override only with cause):

| Layer | Default |
|-------|---------|
| Trace backend | **Grafana Tempo** |
| Metrics | **Prometheus** (existing collector `:8889` scrape path; Mimir etc. deferred) |
| Operator UI for trace + metrics | **Grafana Explore** and/or **dashboard deep links** |
| Optional local / dev trace UI | **Jaeger** (or bare Tempo query UI) if faster for **one-box** debugging—does **not** satisfy “metrics beside traces” by itself |

**Still worth an ADR when choosing vendors or cloud:** managed **Grafana Cloud** (Tempo + Mimir), **Honeycomb**, **Datadog**, vs self-hosted **Grafana + Tempo + Prometheus**. The **default in-repo compose/docs** remains **Tempo + Grafana + Prometheus** unless that ADR moves the default.

- **Remote write** for metrics (OTLP metrics vs Prometheus remote write) vs scrape-only — scheduling only.

---

## 8. Testing and verification (when implemented)

### 8.1 Phase 1 acceptance (concrete)

1. Start **gateway + collector + Tempo** (+ Grafana as needed for links).
2. Emit **one** known gateway-adapted signal path that creates a span (integration smoke or runbook).
3. Assert emitted **`OrionSignalV1.otel_trace_id`** is **32-character lowercase hex**.
4. Assert the **same** trace id is **queryable in Tempo** (and visible in Grafana Explore if that is the link target).
5. Assert **`GET /api/signals/trace/{otel_trace_id}`** on Hub returns the **Orion causal signal chain**, **not** an OTEL span tree (behavior unchanged from today’s contract).
6. Assert the **Hub-generated external trace link** (Grafana Explore or approved dev UI) opens the backend to **that** trace id.

Existing Hub tests that round-trip the signals inspect cache by `otel_trace_id` remain valid for step 5’s **Hub** slice; extend coverage only where the smoke path is not already exercised.

### 8.2 General

- **Hub:** operator page route returns **200** where applicable; iframe **CSP** documented if Phase 2 uses embeds.

---

## 9. References (in-repo)

- `services/orion-signal-gateway/README.md` — OTEL env and behavior.
- `services/orion-signal-gateway/otel/collector-config.yaml` — pipelines and Prometheus port.
- `docs/superpowers/guides/2026-05-01-organ-signal-gateway-offboarding.md` — Hub `/api/signals/*` and gateway `/signals/active`.
- `orion/core/schemas/substrate_review_telemetry.py` — substrate “graph review” telemetry (distinct from OTEL).

---

## 10. Spec self-review

- **Placeholders:** Default stack is **Tempo + Grafana + Prometheus** (§3.1, §7.7); vendor ADRs only for managed/cloud overrides.
- **Consistency:** Orion signal trace cache and OTEL trace store are **two systems**; **§2.1** IDs vs bus correlation; **§3.3** forbids Hub span payload mirror through Phase 2.
- **Scope:** Single implementation plan should cover **Phase 0–1** first (including **compose OTEL_* wiring**); Phase 2–3 may be separate plans if security review splits embed vs proxy.

---

**Next step:** User review of this file; then invoke **writing-plans** to produce `docs/superpowers/plans/…` for Phase 0–1 only unless user directs otherwise.
