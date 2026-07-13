# Execution dispatch motor nerve (P1 of the motor-nerve spec) — design

**Date:** 2026-07-13
**Status:** PROPOSAL — implementation gated on Juniper sign-off before any subagent spawns
**Mode:** Design (dense). Every claim below was checked against the live worktree (`feat/execution-dispatch-motor-nerve-p1`, off fresh `origin/main` post-P0 merge), not inferred from the original brainstorm-level P1 section of `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md`. Several concrete gaps below were not known when that doc was written.

## Arsonist summary

P0 made the status vocabulary honest: `prepared_for_dispatch` is now the correct name for "cleared every gate, envelope built, never sent." P1 is the patch that makes something real happen to that envelope. Today, `orion-execution-dispatch-runtime`'s worker never touches the bus at all — no `redis` import, no `redis` dependency in `requirements.txt`, and a `CORTEX_EXEC_CHANNEL` setting that's wired into `settings.py` but referenced nowhere else and defaults to a channel name (`orion:cortex:request`) that doesn't exist in `orion/bus/channels.yaml`. This isn't a small wiring gap — it's an empty seam. P1 fills it: build a real cortex-exec RPC client (mirroring an existing one, not inventing a new pattern), send `prepared_for_dispatch` envelopes over a bus channel that already has multi-producer precedent, persist what comes back, feed it into feedback scoring (currently a hardcoded stub returning `[]`), and gate all of it behind a deterministic tripwire so a live loop producing empty results reverts itself before anyone has to notice by hand.

## Current architecture

- **Worker** (`services/orion-execution-dispatch-runtime/app/worker.py`, 119 lines): `_tick()` loads a policy frame, its proposal, its self-state, calls `build_execution_dispatch_frame(...)`, and persists the result. No bus code anywhere in the file. No `redis` in `requirements.txt` (fastapi, uvicorn, pydantic, pydantic-settings, sqlalchemy, psycopg2-binary, PyYAML only) — `OrionBusAsync` cannot even import in this service today.
- **Envelope** (`orion/execution_dispatch/envelopes.py`, `build_cortex_request_envelope`): returns a plain dict `{"verb", "mode", "source", "dry_run", "context", "constraints"}` — no `correlation_id`, no `reply_to`, no `kind`. Not `BaseEnvelope`-shaped. Stashed onto `ExecutionDispatchCandidateV1.request_envelope`/`.constraints` and persisted as part of the frame; never read again by anything that sends it.
- **Cortex verbs**: `config/execution_dispatch/execution_dispatch_policy.v1.yaml` routes `inspect`/`summarize`/`observe` proposal kinds to cortex verbs `substrate.inspect`/`substrate.summarize`/`substrate.observe` (`cortex_mode: brain`). **No verb YAML exists for any of these three** under `orion/cognition/verbs/` (96 other verb files exist; none named `substrate.*`). The closest structural precedent is the `skills.*` family (e.g. `skills.docker.ps_status.v1.yaml`: `category: ExecutiveControl`, `priority: low`, `interruptible: true`, `services: []`, plan-based, no LLM generation step) — a read-only system-probe shape, not the `Generative`/LLM-answer shape of verbs like `answer_direct.yaml`.
- **Empty-output-fail gate**: `services/orion-cortex-exec/app/router.py:445-458`, `_should_fail_empty_structured_verb_output`, fires only for verbs in the `_structured_output_expected` allowlist (`router.py:82-91`: `journal.compose`, `concept_induction_journal_synthesize`, `github_compactor_digest_v1`, `chat_history_compactor_digest_v1`, `memory_graph_suggest`, `stance_react`, `harness_finalize_reflect`). None of `substrate.*` are in this set — today, an empty structured response from a future `substrate.inspect` verb would not be caught by this existing mechanism.
- **Bus channels**: `orion:cortex:exec:request` (`channels.yaml:135-144`) is `single_consumer: true`, `producer_services: [orion-cortex-orch, orion-thought]`, consumed solely by `orion-cortex-exec`. A sibling channel, `orion:cortex:exec:request:background` (`channels.yaml:164-171`), already has `producer_services: [orion-cortex-orch, orion-actions, orion-harness-governor]` — this is the live precedent for "a non-orchestrator service producing onto a cortex-exec request queue," and is a closer structural match for what P1 needs than the primary `:request` channel.
- **Single-consumer gate** (`scripts/check_single_consumer_channels.py`): validates only that `single_consumer: true` channels have exactly one live Redis subscriber via `PUBSUB NUMSUB` — it does not validate `producer_services` membership at all. Producer-list correctness is enforced ad hoc by per-feature "bus catalog" tests (e.g. `tests/test_autonomy_goals_bus_catalog.py:64`). Adding `orion-execution-dispatch-runtime` as a producer on `:background` is safe with respect to the single-consumer gate (consumer count doesn't change) but needs its own catalog-membership test, following the established per-feature pattern.
- **RPC pattern already live**: `orion/harness/cortex_client.py`'s `HarnessCortexClient.execute_plan` — constructs a `BaseEnvelope(kind=req.kind, source=ServiceRef(...), correlation_id=..., reply_to=f"{result_prefix}:{uuid4()}", payload=req.model_dump(mode="json"))`, then calls `OrionBusAsync.rpc_request(request_channel, env, reply_channel=..., timeout_sec=...)` — publish, subscribe-to-reply, bounded wait, all in one primitive. Nothing in this class is harness-specific in its constructor signature (`bus`, `request_channel`, `result_prefix`, `source_name`, `timeout_sec`) — it lives in `orion/harness/` (a shared library path, not a `services/` package), so it is a candidate for direct reuse rather than a pattern to re-derive from scratch.
- **Result channel**: `orion:exec:result:*` (`channels.yaml:245-256`, schema `CortexExecResultPayload`, `consumer_services: ["*"]`) — glob-consumed, so a new consumer needs no catalog change.
- **Feedback evidence**: `orion/feedback/extractors.py:52-67`, `normalize_cortex_result_evidence(result: dict) -> dict` — normalizes `status`/`ok`, `result_id`/`correlation_id`, `dispatch_id`, `evidence_refs` into a 4-key dict, consumed by `orion/feedback/builder.py:221-239` to build `cortex_result`-kind observations. **The only real-world caller of `build_feedback_frame` that supplies `cortex_results` is `services/orion-feedback-runtime/app/worker.py`**, which sources them from `FeedbackRuntimeStore.load_cortex_result_evidence(dispatch_frame)` — currently a two-line stub that ignores its argument and returns `[]` (`store.py:259-263`). This is the second dead end P1 must close, symmetric to the worker's missing send step.
- **Migration mechanism**: no Alembic anywhere in the repo. ~47 `manual_migration_*.sql` files under `services/orion-sql-db/`, applied by hand/ops script. `manual_migration_execution_dispatch_frame_v1.sql` is the exact template shape: single JSONB payload column, text primary key, `created_at default now()`, `generated_at desc` index.
- **Notification rail**: NOT a direct bus publish for arbitrary services. `orion-notify` is the sole publisher of `HubNotificationEvent` onto `orion:notify:in_app` (consumed by Hub's `NotificationCache`). Other services integrate via HTTP: `orion/notify/client.py`'s `NotifyClient.send(NotificationRequest)` → `POST {base_url}/notify`. This is simpler than the original brainstorm assumed (no bus schema work needed for notifications).
- **`CORTEX_EXEC_CHANNEL`**: already declared in `services/orion-execution-dispatch-runtime/app/settings.py:31` (`Field("orion:cortex:request", alias="CORTEX_EXEC_CHANNEL")`) and in `.env_example`, but referenced nowhere in `worker.py` — dead config, and its default value doesn't match any real channel name. Must be fixed regardless of which channel P1 targets.
- **Tests**: `tests/test_execution_dispatch_transport_dry_run.py` despite its name does not touch the bus — "transport" refers to a verb-name test fixture (`substrate.inspect.transport`), not networking. No bus-mocking precedent exists in this test suite; `orion/harness/tests/test_cortex_client_finalize_timeouts.py` is the real precedent (`AsyncMock` bus, `bus.rpc_request = AsyncMock(return_value=...)`).

## Missing questions

1. **Reuse `HarnessCortexClient` directly, or fork a thin `ExecutionDispatchCortexClient`?** The constructor is generic. Reuse avoids duplicating the RPC primitive; a fork keeps `orion/harness/` from acquiring an execution-dispatch-shaped caller in its import graph, which is arguably a cleaner service boundary (`orion/harness/` is conceptually "the FCC/Claude harness's cortex access," not a generic cortex-RPC library, even though nothing currently enforces that). **Leaning fork** — a ~40-line class in `orion/execution_dispatch/cortex_client.py` mirroring `HarnessCortexClient`'s shape exactly, since CLAUDE.md's service-boundary rules favor explicit, narrow ownership over cross-package reuse when the reused class isn't already positioned as a shared library. Recommend deciding this explicitly before implementation, not silently.
2. **Target `orion:cortex:exec:request:background` or add a new channel?** The `:background` channel already has non-orchestrator producers and is described (by naming convention) for exactly this kind of lower-priority, non-interactive request. Recommend targeting it — zero new channel, minimal catalog diff, matches an existing pattern instead of inventing one. Confirm consumer-side (`orion-cortex-exec`) actually treats `:background` requests identically to `:request` ones (same verb-routing logic) before committing — if `:background` has different priority/queueing semantics inside cortex-exec, that changes the choice.
3. **Which verb template family?** Recommend modeling `substrate.inspect/summarize/observe.yaml` on the `skills.*` shape (`category: ExecutiveControl`, no `services:` list, plan-based, non-generative) rather than `answer_direct`'s LLM-generation shape — these are read-only system probes producing structured JSON, not conversational answers. Needs one clarifying look at how `skills.*` verbs actually produce their output (plan step mechanics) before the exact YAML shape is final — flagged as a first-half-hour implementation task, not resolved here.
4. **Add `substrate.*` to `_structured_output_expected`'s allowlist, or build P1's own empty-check?** The existing allowlist mechanism in `router.py` is the established, already-tested empty-shell gate. Recommend extending it rather than duplicating its logic in the dispatch worker — one mechanism for "structured verb output must not be empty," not two.

## Proposed schema / API changes

**New verb files** (`orion/cognition/verbs/substrate.inspect.yaml`, `substrate.summarize.yaml`, `substrate.observe.yaml`) sharing one Jinja template (`orion/cognition/prompts/substrate_probe.j2`). Output contract: `{"observation": str, "salient_facts": list[str], "confidence": float}`. Inputs to the template: proposal target (`target_kind`, `target_id`), the triggering `SelfStateV1` dimension snapshot, `proposed_effect`. Add all three verb names to `router.py`'s `_structured_output_expected` allowlist.

**`orion/execution_dispatch/envelopes.py`**: `build_cortex_request_envelope` gains `correlation_id`/`origin: "endogenous.dispatch"` fields in its returned dict so the eventual `BaseEnvelope` wrapping carries dispatch attribution through to cortex-exec's logs and the grammar trace it publishes.

**`orion/execution_dispatch/cortex_client.py`** (new, pending Missing Question 1): thin RPC client mirroring `HarnessCortexClient`, targeting `orion:cortex:exec:request:background` (pending Missing Question 2) with `result_prefix` matching `orion:exec:result:*`.

**`services/orion-execution-dispatch-runtime/app/worker.py`**: `_tick()` gains a step after `build_execution_dispatch_frame` returns: for each candidate in `frame.candidates` with `dispatch_status == "prepared_for_dispatch"`, RPC-send its `request_envelope`, await bounded, record success/failure, and only then construct a new `ExecutionDispatchCandidateV1` copy with `dispatch_status="dispatched"`, `dispatched_at=now`, and `result_ref`/`dispatch_error` set per P0's validator — before persisting.

**New table** `substrate_dispatch_results` (migration `services/orion-sql-db/manual_migration_substrate_dispatch_results_v1.sql`, modeled on the execution-dispatch-frame migration): `result_id text primary key`, `dispatch_id text not null`, `frame_id text not null`, `status text not null`, `result_json jsonb not null`, `raw_len int not null`, `created_at timestamptz not null default now()`. `raw_len=0` or empty `observation` stored with `status='empty'` — never `'success'`.

**`services/orion-feedback-runtime/app/store.py`**: `load_cortex_result_evidence` stops being a stub — real query against `substrate_dispatch_results` filtered by the dispatch frame's candidate `dispatch_id`s, normalized to the shape `normalize_cortex_result_evidence` expects.

**Config**: `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: `mode.allow_dispatch_read_only: true`. `services/orion-execution-dispatch-runtime/.env_example` + `.env`: fix `CORTEX_EXEC_CHANNEL` default to `orion:cortex:exec:request:background` (pending Q2), add `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC=120`, `ORION_DISPATCH_MAX_PER_DAY=24`. `requirements.txt`: add `redis` (pinned to match sibling services, e.g. `redis==5.0.*`).

**Theater tripwire**: in the worker, track the trailing 10 dispatch results (new small store method or in-memory-plus-DB-query); if more than half have `status='empty'`, self-revert `EXECUTION_DISPATCH_MODE` behavior to dry-run-equivalent for subsequent ticks (in-process flag, not an env rewrite) and call `NotifyClient.send(...)` with a `theater_tripwire_active` event. Expose the flag on the existing `/latest` debug endpoint. Re-arm requires a restart (env flip), matching the parent spec's stated design.

## Files likely to touch

- `orion/cognition/verbs/substrate.inspect.yaml`, `substrate.summarize.yaml`, `substrate.observe.yaml` (new)
- `orion/cognition/prompts/substrate_probe.j2` (new)
- `services/orion-cortex-exec/app/router.py` — extend `_structured_output_expected`
- `orion/execution_dispatch/envelopes.py` — correlation/origin fields
- `orion/execution_dispatch/cortex_client.py` (new, pending Q1)
- `services/orion-execution-dispatch-runtime/app/worker.py` — send-and-wait step, tripwire tracking
- `services/orion-execution-dispatch-runtime/requirements.txt` — add `redis`
- `services/orion-execution-dispatch-runtime/.env_example` (+ local `.env` sync) — fixed `CORTEX_EXEC_CHANNEL`, new timeout/cap keys
- `services/orion-sql-db/manual_migration_substrate_dispatch_results_v1.sql` (new)
- `services/orion-feedback-runtime/app/store.py` — real `load_cortex_result_evidence`
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` — `allow_dispatch_read_only: true`
- `orion/bus/channels.yaml` — add `orion-execution-dispatch-runtime` to `:background`'s `producer_services` (pending Q2)
- New bus-catalog test asserting that producer membership, mirroring `tests/test_autonomy_goals_bus_catalog.py`'s pattern
- New worker tests using the `AsyncMock`-bus pattern from `orion/harness/tests/test_cortex_client_finalize_timeouts.py`
- `services/orion-execution-dispatch-runtime/README.md` — document the send step, tripwire, rollback
- `scripts/smoke_execution_dispatch_live.sh` (new) — one real end-to-end dispatch smoke, per the parent spec's evidence bar

## Non-goals

- No mutating dispatch (`allow_mutating_dispatch` stays `false`).
- No LLM-driven action selection, no planner, no scoring model beyond the existing deterministic policy gates.
- No origination enablement (`ORION_ENDOGENOUS_ORIGINATION_ENABLED` stays `false` — that's P7, gated on 2 weeks of clean P1–P3 burn-in per the parent spec).
- No experience-loop wiring (episode journal, felt-state lane) — that's P2.
- No satisfaction/drive-pressure relief — that's P3.
- No new capability vocabulary (recall-first, self-experiments) — that's P4.
- No attention-bound proposal template — that's P5.
- Resolving Missing Questions 1–4 happens at the start of implementation, not deferred into the patch as open TODOs.

## Acceptance checks

Per the parent spec's P1 evidence bar, all four required:
1. A `substrate_dispatch_results` row with `raw_len > 0` and non-empty `observation`, linked from a candidate with `dispatch_status="dispatched"`.
2. A `FeedbackFrameV1` whose evidence references that result (i.e. `load_cortex_result_evidence` actually returns something real, not `[]`).
3. A cortex-exec log line carrying the dispatch correlation ID.
4. `curl :8121/latest` shows the frame with `dispatch_count=1`.

Plus, specific to this session's grounding:
5. `services/orion-execution-dispatch-runtime`'s test suite passes with `redis` importable and a real `AsyncMock`-based test exercising the send-and-wait step (not just the existing pure-function builder tests).
6. Theater tripwire unit-tested: 10 trailing results, >5 empty → tripwire fires, `/latest` reflects it.
7. `make agent-check SERVICE=orion-execution-dispatch-runtime` (or the closest equivalent) passes, including env parity sync.

## Recommended next patch

This spec itself — pause here for sign-off on Missing Questions 1–4 before any implementation subagent spawns. This is a materially larger and riskier patch than P0 (new dependency, new bus wiring, new DB table, policy flip enabling real dispatch, first real autonomous action any Orion service will have taken). Once the four questions are answered, implementation should slice into the same kind of independent-track parallel sprint used for P0: roughly (a) verb files + template + router allowlist extension, (b) cortex client + worker send step + envelope fields, (c) results table + feedback-runtime wiring, (d) tripwire + notify + policy/env flip — with (a)–(c) largely parallel and (d) landing last since it depends on the others existing to have something to gate.
