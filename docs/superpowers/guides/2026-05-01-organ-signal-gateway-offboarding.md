# Organ Signal Gateway — offboarding / next-phase guide

**Audience:** Engineers picking up after the `feature/organ-signal-gateway` work (shared library + gateway service).  
**Design source of truth:** [Organ Signal Gateway design](../specs/2026-05-01-organ-signal-gateway-design.md).

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
| **Stub `signal_id`** | Biometrics matches spec when `source_event_id` present; some stubs still use short hex when id missing. |
| **Wearables / DCGM** | Collector comments only; no `orion-wearable-bridge`. |
| **CI / image** | Ensure pipeline installs `services/orion-signal-gateway/requirements.txt` (includes OTEL + `pytest-asyncio` for tests). |

---

## Planning the next phase (suggested workstreams)

1. **Channel verification pass** — For each `ORGAN_REGISTRY[*].bus_channels`, confirm payloads on those channels match what the corresponding adapter expects; tighten `ORGAN_CHANNELS` in gateway `settings.py` to avoid noise and double-handling.

2. **Adapter implementation order** — Biometrics is the template. High-value next organs from the design: **equilibrium**, **collapse_mirror**, **recall**, **chat_stance** (hardest — cortex cooperation), then the rest.

3. **Hub inspect** — Subscribe Hub to `orion:signals:*`, keep latest-per-organ window, implement `/api/signals/active` and optional trace cache for `/api/signals/trace/{trace_id}` as in the spec.

4. **Downstream consumers** — `orion-heartbeat`, `orion-cortex-exec`, research harness: consume `OrionSignalV1` from signals channels or import `orion.signals.models` for typed reads.

5. **Hardening graduation** — When an organ emits valid `signal.*` kinds itself, registry `notes` can track status; gateway passthrough path is already there.

6. **Observability** — Wire `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector; enable commented DCGM/Prometheus receivers when exporters exist.

---

## Commands (verification)

From repo root (with `.orion_dev` or equivalent venv that has gateway deps + OTEL):

```bash
pip install -r services/orion-signal-gateway/requirements.txt
pytest orion/signals/adapters/tests services/orion-signal-gateway tests/test_biometrics_pipeline.py -q
```

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
