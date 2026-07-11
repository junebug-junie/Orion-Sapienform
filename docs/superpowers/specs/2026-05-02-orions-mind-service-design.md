# Design: Orion’s Mind (control-plane service)

**Date:** 2026-05-02  
**Status:** Draft for review  
**One-line charter:** Orion’s Mind is a **bounded multi-loop cognition process** that produces a **stance trajectory**, a **routed control decision**, and a **handoff brief**; **execution is downstream and contract-driven**.

**Hard rule — service, not a module:** **Orion’s Mind is only the `services/orion-mind` deployable** (container/process with HTTP API, own lifecycle, own scaling). **Cognition loops, snapshot merge, router/governor, and LLM calls run exclusively inside that service.** The **`orion/mind/`** package is **contracts only** (Pydantic models, JSON validation, shared constants)—**no** loop runner, **no** router implementation, **no** “import Mind from orion” in Orch or Exec. Orch **HTTP-calls** the service; it does not embed Mind as a Python module.

**Related work:** AutonomyStateV2 reducer (`docs/superpowers/specs/2026-05-01-autonomy-state-v2-reducer-design.md`) is an **appraisal spine** for graph-backed autonomy on a turn; Mind **consumes** autonomy (and other universe slices) as **inputs**. It does **not** replace the reducer unless explicitly scoped to do so later.

---

## What this is

A **deployable control-plane service** (`services/orion-mind`) that:

1. **Aggregates** a versioned, bounded **universe snapshot** of Orion-internal state and evidence pointers (substrate pressures, collapse mirror dimensions, concept-induction artifacts, autonomy summaries / optional V2, recall digest summaries, social/room signals, metacog, tool outcomes—**subject to caps and trust tags**).
2. Runs **N** (configurable, capped) **cognition loops**. Each loop: **attend → reason → optional hypotheses / simulations → emit a stance patch** (structured + optional LLM assistance behind explicit flags).
3. **Feeds forward**: loop *k + 1* receives **prior stance trajectory** plus snapshot (and optional scratch from prior loops).
4. Runs a **governor / router** step that consumes **final stance + trajectory** and emits a **control decision**: which downstream **route** (e.g. `brain` single-plan, `agent` chain, `workflow_only`, `no_chat`), **capability allowlist** (verb names / skill ids / packs), **budgets** (latency, max tools, max loops already used), and **correlation continuity** fields.
5. Emits a **handoff brief** (`MindHandoffBriefV1`): compact, typed, intended for **Orch / Exec / Actions** to execute **without** re-deriving the universe. Downstream may ignore fields only where the brief explicitly marks them **advisory**; **mandatory** fields must be honored or the consumer returns a typed refusal.

Mind **does not** execute user-facing chat itself in v1. It **ends** at the brief + decision boundary.

---

## What this is NOT

- Not a claim of consciousness or sentience.
- Not a replacement for **GraphDB**, **substrate mutation workers**, **orion-recall**, or **orion-actions**—Mind **reads** and **routes**; it does not own durable graph writes in v1 (see **Deferred** under Phased delivery).
- Not an unbounded “think forever” loop: **hard caps** on loops, tokens, wall time, and snapshot bytes.
- Not a second Exec: Mind does **not** run `PlanExecutionRequest` steps; it **produces** artifacts that **Exec** (or workflows) consume.
- Not **cognition logic packaged as `orion.mind` for in-process import** from Orch, Exec, or Hub. If code is not running inside **`services/orion-mind`**, it is not the Mind runtime.

---

## Goals

| Goal | Detail |
|------|--------|
| **Single locus of “what Orion attends to now”** | The **`services/orion-mind`** process owns aggregation + loop sequencing + routing + brief; other services only supply inputs or consume outputs. |
| **Auditable interior** | Trajectory and decisions are **JSON-serializable**, loggable, replayable with redaction. |
| **Contract-driven handoff** | Downstream services take **MindHandoffBriefV1** + **MindControlDecisionV1** as primary contract inputs. |
| **Bounded cost** | Enforce N_max, T_max, snapshot size, per-loop LLM budget. |
| **Durable audit in Postgres** | Every completed Mind run (including `trigger=operator` introspection via Orch) writes an **artifact row** for replay, compliance, and Hub fetch-by-id. |

---

## Non-goals (v1)

- Hub UI dashboards for Mind (reuse logs / metadata).
- Persisting full Mind state to GraphDB (optional later: store **brief hash + correlation** only).
- Replacing `chat_stance_brief` / `chat_general` prompts in Exec entirely—v1 may **feed** them via brief fields while Exec remains executor of plans.

---

## Canonical entrypoint: Hub vs Orch

**Orch is the only canonical entrypoint** for any Mind run whose outputs may **bind** execution (chat plans, agent chains, workflow dispatch). Exec and downstream consumers see Mind artifacts **only** as merged by Orch into the `PlanExecutionRequest` / correlation they already own.

**Hub** may:

- **Trigger** a Mind-inclusive path by sending a **normal Orch request** with **`context.metadata["mind_enabled"]=true`** (and optional introspection-only keys per implementation plan)—Hub does **not** call `orion-mind` HTTP directly.
- **Introspect** by asking Orch for a **read-only or diagnostic** response (same service boundary: Orch calls Mind with `trigger=operator` or a dedicated introspection profile and returns artifacts in metadata without advancing chat execution, if that mode is defined).

**Rule:** If Hub appears to “run Mind,” it is **always** mediated by Orch. Two HTTP hops (Hub → Orch → Mind) are acceptable; **Hub → Mind → Exec** as a shortcut for canonical chat is **not**.

**Persistence (resolved):** Every completed Mind run (including `mind_introspect` / `trigger=operator`) **must** produce a durable **`mind_runs`** row. **Orch** publishes **`mind.run.artifact.v1`** after receiving the HTTP response from `orion-mind`; **`orion-sql-writer`** performs the `INSERT`/`UPSERT`. Hub reads via Hub API backed by the same table. Persistence default **on**; disabling it is operator-only and out of v1 scope.

**Postgres (`mind_runs`) — locked:** table `mind_runs` with columns `mind_run_id` (uuid PK), `correlation_id`, `session_id`, `trigger`, `ok`, `error_code`, `snapshot_hash`, `router_profile_id`, `result_jsonb`, `request_summary_jsonb` (bounded), `redaction_profile_id`, `created_at_utc`; indexes `(correlation_id)`, `(created_at_utc DESC)`.

**Writer — locked:** **Orch never INSERTs into Postgres for Mind.** After Orch receives `MindRunResultV1` from `orion-mind` over HTTP, it publishes **`mind.run.artifact.v1`** (name finalized in implementation plan) on the bus with an idempotent payload keyed by `mind_run_id`. **`orion-sql-writer`** is the sole component that applies `INSERT`/`UPSERT` into `mind_runs` (same operational pattern as journal / other append-only artifacts). If the bus publish fails, Orch surfaces `mind_artifact_persist_failed` on the correlation and does not claim persistence succeeded.

---

## Resolved architecture (v1) — no forks

| Decision | Choice |
|----------|--------|
| **Mind runtime** | **Only** **`services/orion-mind`** (deployable). No in-process Mind library in Orch or Exec; no “optional” embed of the same loops inside `orion/`. |
| **Transport** | **HTTP only:** `POST /v1/mind/run` from **Orch → orion-mind** (JSON body + `correlation_id`). Bus-RPC for Mind is **out of scope** until v2. |
| **Canonical entry** | **Orch** only for binding + persistence trigger; Hub → Orch only. |
| **Postgres** | **`mind_runs`** table; writes **only** via **`orion-sql-writer`** consuming `mind.run.artifact.v1`. |
| **`merged_stance_brief` validity** | For **`ok=true`**, `merged_stance_brief` **MUST** validate as a full **`ChatStanceBrief`**. If merge cannot produce a valid brief → **`ok=false`**, `error_code=stance_merge_invalid`, no “partial ok” for chat-binding paths unless a separate operator-only `force_advisory_handoff` exists (default **off**). |

Shared **contract types** live in **`orion/mind/`** (Pydantic only). The **`orion-mind` service** imports that package; Orch/Hub/Exec import **types only**—they never import Mind runtime modules.

---

## High-level data flow

```
Hub --(always)--> Orch  --HTTP POST-->  services/orion-mind
                      │
                      │  ┌──────────────────────────────────────┐
                      └──│ 1. Build MindUniverseSnapshotV1      │
                         │ 2. For i in 1..N: CognitionLoopV1     │
                         │      (in ← snapshot + stance_so_far) │
                         │      (out ← stance_patch_i)           │
                         │ 3. Merge → MindStanceTrajectoryV1     │
                         │ 4. Route → MindControlDecisionV1      │
                         │ 5. Brief → MindHandoffBriefV1           │
                         └──────────────────────────────────────┘
        Actions / schedules ──(canonical path)──> Orch ──> … same Mind box …
        -->
Orch merges brief into plan context --> Exec / workflows execute under contract
```

---

## Contract package (`orion/mind/`) — **types only**

Pydantic models and (optional) small **pure validators** shared across repo. **Does not** host the cognition engine, HTTP server, or router implementation.

Place models here so Hub, Orch, Exec, and **`services/orion-mind`** share identical wire types without the monorepo importing the service’s internal application code.

### `MindRunRequestV1`

- `schema_version` (e.g. `mind.run.v1`)
- `correlation_id`, `session_id`, `trace_id` (optional)
- `trigger`: enum `user_turn | scheduled | operator | replay`
- `snapshot_inputs`: structured blobs **or** **refs** (URLs, bus reply channels, opaque ids) Mind is allowed to resolve—**v1 prefers inline bounded JSON** from caller to avoid Mind needing 10 database drivers.
- `policy`: `n_loops_max`, `wall_time_ms_max`, `llm_enabled_per_loop[]`, `router_profile_id`
- `upstream_artifacts`: optional handles to autonomy V2, collapse entry ids, etc.

### `MindUniverseSnapshotV1`

- Versioned wrapper with **facet** map: `autonomy`, `substrate`, `collapse`, `concept_induction`, `recall_digest`, `social`, `metacog`, `tool_outcomes`, …
- Each facet: `{ trust: low|med|high, source, compact_json, bytes_approx }`
- **Hard cap** on total serialized size; **deterministic truncation order** documented.

### `MindStancePatchV1` (per loop)

- `loop_index`
- `structured`: facets compatible with or toward `ChatStanceBrief` (may be partial / delta)
- `narrative_notes` (optional, internal-only)
- `hypotheses[]` (optional, tagged `assumption|hypothesis|simulation_result`)
- `provenance`: model id, temperature, hash of inputs

### `MindStanceTrajectoryV1`

- `patches: list[MindStancePatchV1]`
- `merged_stance_brief`: JSON object; for **`ok=true` Mind runs** must satisfy full **`ChatStanceBrief`** validation (see Resolved architecture).
- `merge_policy`: v1 implementation uses **`deterministic_merge`** for known keys; unknown keys last-wins. If result fails `ChatStanceBrief` validation, run is **`ok=false`** (`stance_merge_invalid`).

### `MindControlDecisionV1`

- `route_kind`: enum (extensible string allowed with registry)
- `allowed_verbs: list[str]` or `allowed_skill_ids: list[str]` (one primary list for v1)
- `recall_profile_override: str | null`
- `mode_suggestion: brain | agent | ...` (advisory vs mandatory—field `binding: advisory|mandatory`)
- `budgets`: remaining wall ms, max tools
- `refusals[]`: typed reasons Mind refuses to route (e.g. snapshot untrusted)

### `MindHandoffBriefV1`

- `summary_one_paragraph` (operator-facing optional)
- `machine_contract`: fields Exec/Orch **must** pass through to `ctx` / metadata (namespaced keys under `mind.*`)
- `mandatory_keys: list[str]` vs `advisory_keys: list[str]`
- `stance_payload`: canonical `ChatStanceBrief`-compatible JSON for downstream chat assembly
- `next_envelopes[]` (optional v2): suggested bus kinds—**out of scope for v1** unless trivial

### `MindRunResultV1`

- `ok`, `error_code`, `diagnostics`
- `snapshot_hash`, `trajectory`, `decision`, `brief`
- `timing_ms_by_phase`

---

## Cognition loop semantics (**implemented in `services/orion-mind` only**)

Each loop **must**:

1. Read **snapshot** + **stance_so_far** (from trajectory merge to date).
2. Produce **at least one** structured delta unless loop is no-op (explicit `noop_reason`).
3. Respect budgets; on exceed, emit **degraded** trajectory with `error_code=loop_budget_exceeded`.

**LLM usage:** Optional per loop behind `policy.llm_enabled_per_loop[i]`. When disabled, loops may run **deterministic** merge rules only (useful for tests and cheap paths).

**Simulations:** v1 may store **structured** “if-then” records without executing tools; tool-using simulation is **non-goal** for v1.

---

## Router / governor (**implemented in `services/orion-mind` only**)

**Inputs:** `MindStanceTrajectoryV1.merged_stance_brief`, hazards from trajectory, snapshot trust flags.

**Outputs:** `MindControlDecisionV1`.

**Routing logic v1:**

- Table-driven **`router_profile_id`** → rules (YAML or DB later); code loads default profile in repo.
- **No** free-form LLM router in v1 unless explicitly enabled by profile (default off).

---

## Integration points (concrete)

| Caller | When | Behavior |
|--------|------|----------|
| **Orch** | Before `call_cortex_exec` when **`context.metadata["mind_enabled"]` is true** | Builds `MindRunRequestV1`; **HTTP `POST`** to **`services/orion-mind`**; merges `MindHandoffBriefV1.machine_contract` into plan context. **Only Orch** performs this merge for canonical chat/agent paths. |
| **Hub** | Operator or UI wants Mind or introspection | Sends request **to Orch** with **`context.metadata["mind_enabled"]=true`** (or introspection-only mode key defined in implementation plan) so Orch invokes Mind; Hub does **not** call `orion-mind` directly. |
| **Actions** | Scheduled “think without chat” | Publishes or schedules work that is **handled through the same Orch workflow entry** (or a dedicated Orch-handled envelope), so Mind still sits behind Orch for any result that must align with execution contracts. |

**Exec changes (v1 minimal):** read `ctx["mind"]` or `metadata["mind_handoff"]` if present and map into existing stance pipeline **or** pass `stance_payload` directly to skip duplicate stance synthesis (feature-flagged, single code path).

---

## Error handling

- **Snapshot build failure** → Mind returns `ok=false`, `error_code=snapshot_incomplete`, still may emit **safe** control decision `route_kind=no_chat` or `mandatory` conservative allowlist.
- **Loop partial failure** → record in trajectory, continue if policy says continue; else stop.
- **Downstream refusal** → not Mind’s problem; caller logs contract violation.

---

## Security & trust

- Mind must not **exfiltrate** secrets from refs beyond declared facet contracts.
- **Operator-only** facets gated by auth on HTTP API.
- **Proxy / phi** facets: `trust=low`, cannot alone justify `mandatory` aggressive routing.

---

## Observability

- Structured logs: `mind_run_start`, `mind_run_end`, per-loop timing, snapshot hash, decision summary.
- Metrics: loop count used, truncation happened, router profile id.

---

## Deployment & configuration provenance (repo conventions)

Orion’s Mind follows the **same service layout as other `services/orion-*` processes** (e.g. `orion-notify`, `orion-vision-host`): **Dockerfile + service-local `docker-compose.yml` + `.env` / `.env_example` + `app/settings.py`**, with a clear **`.env` → compose `environment:` → container env → Pydantic settings** chain.

### Directory layout (`services/orion-mind/`)

| Path | Role |
|------|------|
| `Dockerfile` | OCI image; build context is the **monorepo root** (see compose `context`). |
| `docker-compose.yml` | Stack wiring for this service; `env_file` + explicit `environment` pass-through. |
| `.env` | Operator overrides; **gitignored**. |
| `.env_example` | **Committed** contract; must stay in **lockstep** with real `.env` keys (repo rule: change meaning or add a key in `.env` → update `.env_example` in the same change set). |
| `requirements.txt` | Service-only Python deps. |
| `app/main.py` | FastAPI app entry. |
| `app/settings.py` | `pydantic_settings.BaseSettings` — single source of truth for tunables. |
| `app/config/router_profiles.yaml` (or `config/...` **COPY**’d into the image) | Default router table for v1. |

### Dockerfile (match lightweight services)

- **`FROM python:3.12-slim`** (or the repo’s agreed Python image for HTTP microservices).
- **`WORKDIR /app`**
- Minimal **apt** (`curl`, `ca-certificates`) if healthcheck uses `curl`.
- **Layer cache:** `COPY services/orion-mind/requirements.txt` → `pip install -r requirements.txt` **before** copying app code.
- **`COPY services/orion-mind/app ./app`**
- **`COPY orion ./orion`** so the image includes **`orion/mind/`** contract types (and any other `orion.*` imports the service uses).
- **`EXPOSE`** the HTTP port defined in settings (internal port, e.g. `6611`).
- **`CMD`:** `uvicorn app.main:app --host 0.0.0.0 --port <PORT>` (same pattern as `orion-notify`).

### docker-compose.yml

- **`build.context: ../..`** and **`dockerfile: services/orion-mind/Dockerfile`** (repo root context is **required** for `COPY orion`).
- **`env_file: [.env]`** beside the compose file.
- **`environment:`** — list **every** variable the container must see using **`VAR=${VAR}`** so values flow from `.env` into the container **explicitly** (provenance: `docker compose config` shows the full bridge). Do not rely on compose inheriting host env without listing it.
- **`ports`:** `"${HOST_PORT}:<internal_port>"` (internal port matches Dockerfile/uvicorn).
- **`networks`:** `app-net` **external: true`** when joining the shared stack (same as other services).
- **`healthcheck`:** e.g. `curl -f http://localhost:<PORT>/health` with `start_period`, `interval`, `timeout`, `retries` (match `orion-notify` style).

### `app/settings.py` (pydantic-settings)

- `model_config = SettingsConfigDict(env_file=".env", extra="ignore")` for local dev; in Docker, **environment variables set by compose override** `.env` as usual for Pydantic.
- **Field names = env names:** `UPPER_SNAKE` on both sides. Use `Field(..., validation_alias=AliasChoices(...))` **only** when renaming for migration—and document every alias in **`.env_example`**.
- Group fields logically: service identity, HTTP server, Mind policy defaults, optional LLM (later phase).

### Caller service (Orch) provenance

Orch is the HTTP client to Mind. **`services/orion-cortex-orch/.env_example`** gains e.g. **`ORION_MIND_BASE_URL`** (and optional **`ORION_MIND_TIMEOUT_SEC`**) with a compose-friendly default (`http://orion-${PROJECT}-mind:6611` or the chosen service name). Same **`.env` / `.env_example` parity** applies to Orch.

### Example env keys (`services/orion-mind/.env_example`)

Document at least: `SERVICE_NAME`, `SERVICE_VERSION`, `NODE_NAME`, `LOG_LEVEL`, `PORT` (or `MIND_HTTP_PORT`), `MIND_SNAPSHOT_MAX_BYTES`, `MIND_WALL_MS_DEFAULT`, `MIND_N_LOOPS_DEFAULT`, `MIND_ROUTER_PROFILES_PATH`. Later phases add LLM-related keys with **empty or placeholder** values only.

### Local `.env` (gitignored) vs `.env_example`

- **`services/orion-mind/.env`** remains **gitignored** like other services.
- **Convention:** whenever `.env_example` changes, operators/devs **refresh** local `.env` from it (`cp .env_example .env` or equivalent merge). First-time bootstrap is **`cp .env_example .env`** then edit secrets.
- **Committed repos:** do not commit `.env`; CI uses compose/env injection only.

---

## Bus catalog & schema registry (mandatory for any new envelope/channel)

Any **new bus traffic** introduced for Mind (e.g. **`mind.run.artifact.v1`** envelope payload written by Orch and consumed by **`orion-sql-writer`**) **must** be registered in the **same delivery** as the code that sends/receives it:

1. **`orion/bus/channels.yaml`** — add the channel (and **`message_kind`** / **`schema_id`** fields per existing patterns); include **`producer_services`** / **`consumer_services`** (`orion-cortex-orch`, `orion-sql-writer`, …).
2. **`orion/schemas/registry.py`** — import the Pydantic payload model(s), add **`schema_id` → model class** entries to **`_REGISTRY`**, so **`resolve(schema_id)`** works for catalog enforcement (`ORION_BUS_ENFORCE_CATALOG`).
3. Payload types live under **`orion/schemas/`** (e.g. `orion/schemas/mind/…`) or **`orion/mind/`** if they are shared contracts—implementation picks one namespace but **registry must list every `schema_id`** referenced from `channels.yaml`.

HTTP-only Mind runtime does **not** require bus registration until Orch publishes **`mind.run.artifact.v1`** (Phase **P3**); add channels + registry in that phase’s merge.

---

## Testing strategy

- **Unit:** merge policies, router table, snapshot caps — **`services/orion-mind/tests/`** (service owns behavior). **`orion/mind/tests/`** only for model/schema validation.
- **Contract:** golden `MindRunRequestV1` → `MindRunResultV1` with LLM disabled in **`services/orion-mind/tests/`** (uses `orion/mind` models only).
- **Integration:** Orch test with fake Mind HTTP server (httpx mock) verifying metadata merge **and** that a **`mind_runs`** row is written (or the chosen writer is invoked) for both `user_turn` and `operator` triggers.

---

## Phased delivery (P0–P6)

Each phase has a **single acceptance bar**; no phase “completes” without it.

| Phase | Deliverables | Acceptance |
|-------|----------------|------------|
| **P0** | **`orion/mind/`** Pydantic models only (`MindRunRequestV1`, snapshot, trajectory, decision, brief, result); `orion/mind/tests/` for schema/roundtrip only. **No** `FakeMindRunner` cognition in `orion/` — stubs live beside **Orch** or **Mind service** tests as HTTP mocks. | `pytest orion/mind/tests` green; models serialize/deserialize. |
| **P1** | **`services/orion-mind`:** FastAPI `GET /health`, `POST /v1/mind/run`; **N=1** deterministic loop; router YAML; **`Dockerfile`**, **`docker-compose.yml`**, **`.env_example`**, **`app/settings.py`** per **Deployment & configuration provenance**; image builds from repo root with `COPY orion`. | Image `docker build -f services/orion-mind/Dockerfile` succeeds; compose `healthcheck` passes; HTTP test as today. |
| **P2** | **Orch:** when **`context.metadata["mind_enabled"]` is exactly `true`**, build `MindRunRequestV1` from `CortexClientRequest` + optional `orion_state`, HTTP `POST` to `orion-mind`, merge `MindHandoffBriefV1.machine_contract` into `PlanExecutionRequest.context.metadata` under `mind.*`; **default `mind_enabled` absent or false** so production is unchanged until Hub sets the flag. | Orch integration test with Mind mocked: metadata contains merged keys; when `mind_enabled` absent, zero HTTP calls to Mind. |
| **P3** | **Bus + `orion-sql-writer`:** add **`orion/bus/channels.yaml`** entry + **`orion/schemas/registry.py`** registration for the artifact payload **`schema_id`**; **`orion-sql-writer`** handler for **`mind.run.artifact.v1`** (or aligned `kind`); **`INSERT` into `mind_runs`**; migration; idempotency on `mind_run_id`. | Catalog validation passes for published envelopes; publish → row exists; duplicate publish → safe idempotent behavior. |
| **P4** | **Hub:** `GET /api/mind/runs/{mind_run_id}` and `GET /api/mind/runs?correlation_id=` (read-only), same PG pool pattern as other Hub routes; auth/session as Hub norms. | Hub test: seed row → API returns expected JSON; 404 when missing. |
| **P5** | **Exec:** read `metadata["mind"]` / `mind_handoff` (final key in plan); map `stance_payload` into ctx for `chat_general` **or** feature-flag **skip duplicate** `synthesize_chat_stance_brief` when brief supplies full stance. | Exec test: with flag, one fewer stance call or equivalent ctx assertion. |
| **P6** | **N>1** loops with feedforward; optional LLM per loop behind `policy.llm_enabled_per_loop[]`; Orch **snapshot builder v2** pulling autonomy / collapse / substrate facets into snapshot (bounded, trust-tagged). | E2E test: N=2 with LLM disabled still produces valid trajectory + brief; snapshot size cap enforced. |

**Deferred (not P0–P6):** Mind-invoked **bus RPC**; **GraphDB** persistence of Mind state; **Actions** scheduling without Orch (Actions must enqueue Orch work item per canonical rule).

---

## Approval

After review, set **Status** to *Approved for implementation planning* and add **`docs/superpowers/plans/YYYY-MM-DD-orions-mind-service-implementation.md`** (task checklist) — same workflow as other Orion specs.
