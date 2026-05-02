# Organ Signal Gateway — offboarding / next-phase guide

**Audience:** Engineers picking up after the `feature/organ-signal-gateway` work (shared library + gateway service).  
**Design source of truth (phase 1 chassis):** [Organ Signal Gateway design](../specs/2026-05-01-organ-signal-gateway-design.md).  
**Phase 2 (adapters + Hub inspect):** [Organ Signal Gateway — Phase 2](../specs/2026-05-01-organ-signal-gateway-phase-2-design.md) — milestones, acceptance for **2a done**, first-pass Critical / Important / Minor tables, and 2b scope.

### Authority / read order (next-phase handoff)

1. **Phase 2 spec** — [2026-05-01-organ-signal-gateway-phase-2-design.md](../specs/2026-05-01-organ-signal-gateway-phase-2-design.md): **§ Phase 2a milestones**, **§ Phase 2a completion criteria**, **§ Phase 2b**, **First pass** tables (production-ready gate), **File map**, **Self-review**.
2. **Phase 1 spec** — schema, gateway chassis, Hub JSON shapes: [2026-05-01-organ-signal-gateway-design.md](../specs/2026-05-01-organ-signal-gateway-design.md).
3. **This guide** — inventory, channel pattern `orion:signals:{organ_id}`, what exists vs stub.

If the active program is **memory cards v1** instead of signal-gateway phase 2, use [Orion memory cards v1](../specs/2026-05-01-orion-memory-cards-v1-design.md) as the requirements doc for that track.

This document maps **what was created**, **what is real vs stub**, **how it wires into the mesh**, and **suggested next-phase workstreams** so planning does not rediscover context from scratch.

---

## One-sentence recap

A **shared** `orion/signals/` package defines `OrionSignalV1`, the static `ORGAN_REGISTRY`, normalization primitives, and per-organ **adapters**; **`services/orion-signal-gateway`** subscribes to the bus, runs adapters (or validates passthrough `signal.*` kinds), enriches causal notes, attaches **OTEL** spans, and republishes hardened signals to `orion:signals:{organ_id}`.

---

## Architecture (current)

```text
Redis bus (organ channels, see settings.ORGAN_CHANNELS)
  → Hunter (orion-signal-gateway)
      → SignalProcessor.handle_envelope
          → adapter.adapt()  OR  passthrough (kind starts with "signal.")
          → with_missed_parent_notes()
          → _emit_traced() (OTEL span + signal.otel_* fields)  OR  _emit_passthrough()
          → publish orion:signals:{organ_id}, kind signal.{organ}.{kind}
```

**Read-only:** The gateway does not write back to organ services or alter their legacy channels.

---

## File inventory

### Shared library: `orion/signals/`

| Path | Role |
|------|------|
| `__init__.py` | Package marker (minimal). |
| `models.py` | `OrganClass`, `OrionSignalV1`, `OrionOrganRegistryEntry` (Pydantic); `notes` capped at 5. |
| `registry.py` | `ORGAN_REGISTRY` — causal DAG + **`bus_channels` filled from service defaults** (verify against your deployment). |
| `normalization.py` | Canonical **`EwmaBand`**, **`InductionTracker`**, `clamp01`; `clamp11`; `NormalizationContext` (`get_band`, `get_tracker` per `(organ_id, metric_key)`). |
| `causal_helpers.py` | `with_missed_parent_notes(signal, prior, registry)` for §7.B-style missed-parent audit lines. |
| `adapters/base.py` | `OrionSignalAdapter` ABC. |
| `adapters/biometrics.py` | **Reference adapter:** `biometrics.induction.v1`-style payloads → one `biometrics_state` signal; tracker-driven dimensions; **does not** set `otel_*` (gateway does). |
| `adapters/*.py` (except `biometrics`) | **Stubs:** minimal `can_handle` / `adapt`, placeholder dimensions, stub `notes`; causal parent IDs when `prior_signals` has that organ. |
| `adapters/__init__.py` | Exports adapters + `ADAPTERS` list (gateway iterates in order — **order matters** for first-match dispatch). |
| `adapters/tests/` | Spec-aligned adapter tests location: causal helper + biometrics OTEL-absence contract. |

### Telemetry shim: `orion/telemetry/biometrics_pipeline.py`

| Change | Role |
|--------|------|
| Imports `EwmaBand`, `InductionTracker`, `clamp01` from `orion.signals.normalization` | Keeps a single implementation per design file map; existing services importing from `biometrics_pipeline` keep working. |

### Service: `services/orion-signal-gateway/`

| Path | Role |
|------|------|
| `app/main.py` | FastAPI lifespan: `configure_tracing()` then `GatewayService`; routes `/health`, `/signals/active`. |
| `app/instrumentation.py` | Global `TracerProvider`: OTLP if `OTEL_EXPORTER_OTLP_ENDPOINT`, console if `OTEL_CONSOLE_EXPORT`, else silent drop exporter (real span IDs). |
| `app/settings.py` | Bus URL, `ORGAN_CHANNELS` glob list, `SIGNAL_WINDOW_SEC`, `SIGNALS_OUTPUT_CHANNEL`, OTEL envs. |
| `app/service.py` | `Hunter` + lazy `SignalProcessor` wiring. |
| `app/processor.py` | Envelope dispatch; passthrough guard (`signal.` prefix, skip self-echo); `with_missed_parent_notes`; OTEL parent context from first registry parent with valid `otel_*` on **non-exogenous** signals. |
| `app/signal_window.py` | Latest `OrionSignalV1` per `organ_id`, TTL eviction. |
| `app/normalization_state.py` | Per-organ `NormalizationContext` instances for adapters. |
| `app/passthrough.py` | Validates self-hardened payloads (`OrionSignalV1` + known `organ_id`). |
| `app/tests/` | Gateway + normalization + passthrough + OTEL (in-memory exporter) + adapter integration-style tests. |
| `otel/collector-config.yaml` | Sidecar template (OTLP, hostmetrics, prometheus/dcgm **commented**). |
| `Dockerfile` | Build context = **repo root** (see `docker-compose.yml`). |
| `docker-compose.yml`, `requirements.txt`, `.env_example`, `README.md`, `pytest.ini` | Ops and developer entrypoints. |

---

## What is “done” vs “next phase”

### Done in this phase (framework)

- Schema + registry + mesh-oriented `bus_channels` (still **verify** per environment).
- Gateway bus subscription shell, window, passthrough rules, missed-parent notes, OTEL span emission and parent trace inheritance for **adapter** path.
- Biometrics reference adapter + tests.
- Stub adapters for every organ in the design tree (contract placeholders).
- Collector config skeleton.

### Explicitly still next phase (per design “out of scope” + gaps)

| Area | Notes |
|------|--------|
| **Real adapters** | Replace stubs with payload shapes from each service; align `can_handle` with actual `env.kind` + channel patterns. |
| **Hub** | Spec’s `GET /api/signals/active` and trace explorer live on Hub; gateway exposes `/signals/active` only. |
| **End-to-end mesh** | Subscribe patterns vs real traffic; avoid duplicate processing; confirm no channel gaps. |
| **Parent trace conflicts** | Spec: if parents disagree on `trace_id`, log / `notes` — only “first parent with OTEL” wins today. |
| **Stub `signal_id`** | Biometrics matches spec when `source_event_id` present; Phase 2 targets **64-hex** (full SHA-256 of preimage) for deterministic ids where backward compatibility allows — audit stubs and consumers for fixed-width assumptions. |
| **Wearables / DCGM** | Collector comments only; no `orion-wearable-bridge`. |
| **CI / image** | Ensure pipeline installs `services/orion-signal-gateway/requirements.txt` (includes OTEL + `pytest-asyncio` for tests). |

---

## Phase 2a — milestones and acceptance (from spec; do not drift)

The following is aligned with **§ Phase 2a** and **§ Phase 2a completion criteria** in [Phase 2 design](../specs/2026-05-01-organ-signal-gateway-phase-2-design.md). Treat the Phase 2 spec tables (**First pass** Critical / Important / Minor) as the **checklist** for production-ready vs partial implementation.

### Milestones (depth-first, mergeable)

| ID | Organ(s) | Intent |
|----|----------|--------|
| **M1** | `equilibrium` | First **hybrid** organ with a real bus contract: real payloads, registry `bus_channels` verified against deployment, dimensions and `signal_kind` aligned with `ORGAN_REGISTRY`. |
| **M2** | `collapse_mirror` | First **endogenous** organ in the slice that depends on **biometrics** + **equilibrium** in `prior_signals`; **causal_parents** and **shared OTEL trace** with upstream signals validated in tests or staging. |
| **M3** | `journaler` **or** `recall` (order **TBD by contract clarity**) | Implement whichever has the **clearer, documented** bus (or result) contract first; the other follows in the same phase or as **2a.1** if **2a** acceptance is already met without it. Resolve with evidence (actual bus payloads / runbooks), not merge-time guesswork. |
| **M4** | `chat_stance` | **Last:** requires an explicit **cortex / router payload contract** subsection (channel, field paths, degradation rules) before implementation. Adapter degrades to low `confidence` + `notes` when fields are missing, per phase-1 design rules. |

**Biometrics** remains the reference adapter for the slice. **Channel verification** (actual Redis channel names, representative payloads, `bus_channels` + `ORGAN_CHANNELS` alignment) gates **every** milestone — see Phase 2 spec § “Channel verification”.

### When **2a** is “done” (acceptance bar)

**2a is complete** when all of the following hold (verbatim structure from Phase 2 §4):

1. **M1 and M2 are merged** with fixture-backed adapter tests and updated channel documentation.
2. **End-to-end proof:** Under a **scripted test harness** (preferred) or **documented staging procedure**, the gateway produces hardened signals on **`orion:signals:{organ_id}`** for at least **three distinct organ_ids** in the primary chain (e.g. biometrics, equilibrium, collapse_mirror) such that **one OTEL `trace_id`** spans those signals in order (exogenous start → derived children), within the configured signal window. The procedure or **test name** must be referenced in the **implementation plan appendix** for this spec (Phase 2 doc).
3. **M3 and/or M4:** If **M3** is not merged when (1)–(2) hold, **2a** may still be marked complete only if the written proof in (2) lists which organs are covered and **M3/M4** are scheduled as **2a.1** with dates or ticket references. If the team prefers a stricter bar, **require M3 merged** before 2a complete; **pick one interpretation in the implementation plan and do not leave it ambiguous.**

**Default interpretation in the Phase 2 spec:** **(1)+(2) required**; **M3 strongly encouraged**; **M4 may slip to 2a.1** if the `chat_stance` contract is not ready, **provided** (2) still holds for three organs **without** counting duplicate kinds on the same organ.

**Production-ready vs functional milestones:** Merging **M1** on a branch advertised as **multi-replica HA** without meeting **First pass → Critical (multi-instance)** is **out of spec**. The **functional** 2a bar above can be met on a **singleton** gateway while first-pass items are still in flight; the **production-ready gate** (Phase 2 “First pass” introduction) applies before any GA / customer-facing HA claim.

## Phase 2b — Hub inspect (from spec)

- **Preconditions:** Stable gateway output pattern (e.g. **`orion:signals:{organ_id}`**); Hub parses **`OrionSignalV1`** JSON (prefer **`orion.signals.models`**).
- **Ship by default:** **`GET /api/signals/active`** — in-memory latest-per-organ, response shape per phase-1 spec (`as_of`, sparse `signals` map), **same auth posture** as other operator-read Hub routes (e.g. align with `/api/substrate/*` unless security review dictates stricter RBAC). Implementation sketch: subscriber on Hub Redis lifecycle, route colocated with Hub FastAPI patterns (e.g. `services/orion-hub/scripts/api_routes.py`).
- **Stretch:** **`GET /api/signals/trace/{trace_id}`** — only with **mandatory** `TRACE_CACHE_MAX_TRACES`, `TRACE_CACHE_TTL_SEC`, `TRACE_CACHE_MAX_SIGNALS_PER_TRACE` and full **200** / `complete` / `gaps` / **404** semantics per Phase 2 **First pass → Critical (Hub trace cache)**.

## Planning the next phase (workstreams — summary)

1. **First-pass / production-ready gate** — Close gaps in Phase 2 **First pass** tables (Critical / Important / Minor): code, doc-only, or **waived + documented risk**; do not treat rows as done until spec + README / Helm match.
2. **M1 → M2 → M3 (order by contract) → M4 (after contract)** — As in the milestone table above; use Phase 2 **Channel verification** before each merge.
3. **Hub (2b)** — After gateway output is stable: subscribe to `orion:signals:{organ_id}` pattern, implement **`GET /api/signals/active`**; trace endpoint only if stretch criteria are met.
4. **Downstream consumers** — `orion-heartbeat`, `orion-cortex-exec`, research harness: consume `OrionSignalV1` from signals channels or import `orion.signals.models` for typed reads.
5. **Hardening graduation** — When an organ emits valid `signal.*` kinds itself, registry `notes` can track status; gateway passthrough path is already there.
6. **Observability** — Wire `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector; span attribute privacy per Phase 2 **OTEL_DIMENSION_ALLOWLIST** (or equivalent); collector vs organ-bus biometrics precedence per Phase 2 **First pass → Important**.

**Risks to carry forward (from phase 2 work):** deterministic **`signal_id`** length (**64-hex** full SHA-256 when preimage-based) — audit consumers (Hub, substrate, logging) for fixed-width assumptions; OTEL **`dim.*`** suffix rule (`*_level` / `*_trend` / `*_volatility`) — review adapters for sensitive or high-cardinality keys matching those patterns.

---

## Commands (verification)

From repo root, with **`PYTHONPATH=.`** after installing gateway deps (Phase 2 **canonical test roots**):

```bash
pip install -r services/orion-signal-gateway/requirements.txt
PYTHONPATH=. python3 -m pytest -q services/orion-signal-gateway/app/tests orion/signals/adapters/tests
```

Optional: include `tests/test_biometrics_pipeline.py` if you are validating the telemetry shim against shared normalization.

Local API (adjust `PYTHONPATH` / `--app-dir` as in `README.md`):

```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir services/orion-signal-gateway
```

Docker build expects **context = repository root** (`services/orion-signal-gateway/docker-compose.yml`).

---

## Order of reading for a new owner

1. Design spec (linked above).  
2. `services/orion-signal-gateway/README.md`.  
3. `orion/signals/registry.py` (DAG + channels).  
4. `services/orion-signal-gateway/app/processor.py` (dispatch + OTEL + notes).  
5. `orion/signals/adapters/biometrics.py` (reference adapter).  
6. `app/tests/test_otel_propagation.py` (OTEL behavior contract).

---

## Branch / history

Implementation landed on branch **`feature/organ-signal-gateway`** (pushed to `origin`). Use `git log --oneline --follow -- orion/signals services/orion-signal-gateway` for the exact commit range if the branch is merged or renamed later.
