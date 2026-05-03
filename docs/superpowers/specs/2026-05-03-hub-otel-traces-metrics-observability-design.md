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

---

## 3. Goals and non-goals

### 3.1 Goals

1. **End-to-end OTEL traces** persisted in a **trace backend** (Tempo or Jaeger — decision in implementation plan), reachable from deployment docs and compose/k8s manifests.
2. **Metrics** remain scrape- or remote-write–based; **same time window** as trace investigation should be discoverable from Hub (link, embed, or API-backed chart — phased).
3. **Hub** becomes the **default entry** for operators: at minimum **deep links**; at maximum **embedded or native** combined views (phased).
4. **Preserve** `GET /api/signals/trace/{trace_id}` as the **Orion causal** debugging surface; **do not** conflate it with OTEL span completeness.

### 3.2 Non-goals (this spec)

- Replacing Grafana/Jaeger/Tempo with a **full** custom observability product inside Hub.
- Changing **substrate graph review** telemetry schemas or merging them into OTEL without a separate design.
- Defining **SLOs and alerting** policy (only noted as backlog).

---

## 4. Architecture options (recap)

| Option | Idea | Pros | Cons |
|--------|------|------|------|
| **A — Embed / deep link** | Grafana Explore or Jaeger UI in iframe; Hub supplies trace id, time range, `service.name`. | Fast, full-featured UI, exemplars if enabled. | Auth, CORS, cookie domains, “two apps” feel. |
| **B — Hub links only** | Hub shows “View trace in Grafana” with URL templates; no iframe. | Minimal code; clear security boundary. | Context switching; no single-pane chrome. |
| **C — Native Hub** | Hub backend proxies Tempo + Prom; custom timeline + charts. | Unified auth and layout. | Ongoing UX and API surface ownership. |

**Recommendation:** **B first**, then **A** where security review allows, then **C** only for **narrow** golden paths (e.g. one trace summary + fixed metric panels).

---

## 5. Phased delivery (in scope by phase)

### Phase 0 — Documentation and wiring truth

- Document **required** env vars and URLs: gateway OTLP → collector → trace backend; Prometheus scrape target for `:8889` (or service discovery equivalent).
- Enable (in dev/staging) **OTLP exporter** on the collector to the chosen trace backend (today Jaeger OTLP block is commented in `collector-config.yaml`).

### Phase 1 — Trace persistence + operator links from Hub

- Choose and deploy **Tempo or Jaeger** (implementation plan records ADR).
- Add Hub UI or debug affordance: given `otel_trace_id` from signals or logs, **open trace** via **stable URL template** (Grafana Explore or native Jaeger/Tempo UI).
- Ensure **hex trace id** format matches what Tempo/Jaeger expect (normalize in one place).

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
| 0 | Docs, compose/k8s snippets, collector OTLP to trace backend in non-prod | Prod hardening, SSO, retention |
| 1 | Trace store, deep links from Hub | Iframes, PromQL proxy, custom charts |
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
- **SSO / OIDC** for Grafana (or Jaeger) aligned with Hub auth; **iframe** cookie and **Content-Security-Policy** strategy.
- **Multi-tenant** trace isolation (namespace, headers, org id on spans).
- **Audit log** of “who opened which trace id” for regulated environments.

### 7.3 Backend and data plane

- **Hub read proxy** to Tempo + Prometheus with **rate limits**, **query budgets**, and **timeout** semantics (full design of §5 Phase 3).
- **Mimir** or **long-term** metrics store vs raw Prometheus only.
- **OpenTelemetry Logs** pipeline and correlation with traces (`trace_id` in log records).
- **Tail sampling** and **head sampling** policies in collector; **sampling overrides** per organ or per route.
- **Exemplars**: Prom metrics linking to `trace_id` for click-through from metric spike to trace.
- **Hub trace cache** storing **full span payloads** (today bounded **signal** cache only) — retention, PII, and cost implications.

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

### 7.7 ADRs to schedule

- **Tempo vs Jaeger** (and managed alternatives: Grafana Cloud, Honeycomb, Datadog) for this repo’s default.
- **Grafana vs bare Jaeger/Tempo UI** as the embedded surface.
- **Remote write** for metrics (OTLP metrics vs Prometheus remote write) vs scrape-only.

---

## 8. Testing and verification (when implemented)

- **Smoke:** emit a known span from gateway (or integration test), assert it appears in trace backend, assert Hub link URL resolves (automated or runbook).
- **Contract:** `otel_trace_id` on `OrionSignalV1` matches trace id query in Tempo/Jaeger for the same request path.
- **Hub:** route returns 200 for operator page; iframe CSP documented if used.

---

## 9. References (in-repo)

- `services/orion-signal-gateway/README.md` — OTEL env and behavior.
- `services/orion-signal-gateway/otel/collector-config.yaml` — pipelines and Prometheus port.
- `docs/superpowers/guides/2026-05-01-organ-signal-gateway-offboarding.md` — Hub `/api/signals/*` and gateway `/signals/active`.
- `orion/core/schemas/substrate_review_telemetry.py` — substrate “graph review” telemetry (distinct from OTEL).

---

## 10. Spec self-review

- **Placeholders:** None intentional; backend product names left as “Tempo or Jaeger” where a choice is explicitly backlog (§7.7).
- **Consistency:** Orion signal trace cache and OTEL trace store are **two systems**; join key and non-conflation are stated in §3.1 and §6.
- **Scope:** Single implementation plan should cover **Phase 0–1** first; Phase 2–3 may be separate plans if security review splits embed vs proxy.

---

**Next step:** User review of this file; then invoke **writing-plans** to produce `docs/superpowers/plans/…` for Phase 0–1 only unless user directs otherwise.
