# Orch route-grammar lane — design spec

Status: implementing. **Second correction, this one load-bearing on Patches A/B: they are void.**

Deeper trace (found while re-verifying line anchors for implementation) shows `handle_verb_request` → `VerbRuntime.handle_request` dispatches `trigger="legacy.plan"` to `LegacyPlanVerb.execute()` in `services/orion-cortex-exec/app/verb_adapters.py:148-276` — **not** a dead end. That verb builds its own `ctx_payload` and calls `router.run_plan(..., ctx=ctx_payload)` (verb_adapters.py:236), and `PlanRouter.run_plan` calls `begin_plan_grammar(ctx, ...)` unconditionally near its top (`router.py:931`), which lazily creates a `CortexExecGrammarCollector` and writes it into `ctx["_cortex_exec_grammar_collector"]` (`grammar_emit.py:389-407`, `get_or_create_collector`). Every step call records onto that same collector (`router.py:1079-1342`). `LegacyPlanVerb.execute()` then explicitly flushes it (`verb_adapters.py:253-275`, `flush_cortex_exec_grammar`). Trace id is `cortex_exec_trace_id(node_name, correlation_id, lane=...)` = `"cortex.exec:{node}:{corr_id}"` (`orion/substrate/execution_loop/ids.py`) — the exact prefix `EXECUTION_TRACE_PREFIX` the `execution_grammar_reducer` cursor already filters on, and `provenance.source_service` is hardcoded `"orion-cortex-exec"`, already in `EXECUTION_SOURCE_SERVICES`.

**Conclusion: chat-lane `legacy.plan` requests already produce full step-level `GrammarEventV1` traces, already land in the `execution_trajectory` projection.** There is no chat-lane step/plan-mechanics gap. Patch A (make `handle_verb_request` publish grammar) is deleted — it would duplicate an existing, working path. Patch B (failing-test proving the gap) is deleted — the premise was false, nothing to prove. The prior "Ground truth" correction in this doc (the one about `handle_verb_request` never calling grammar publish) was itself wrong — it looked at the wrong file (`main.py`) and missed that the actual publish call lives in `verb_adapters.py`.

**What's still real and still being built**: orch's own arbitration (lane pick, mind-gate, output-mode) is still genuinely unpublished anywhere — nothing above touches that, it's a fact about orch's pre-dispatch code, not cortex-exec's execution. Patches C (orch shadow producer), D (route reducer), and E (surface arbitration metadata on the result) stand as designed below and are what's being implemented in this pass.

## Arsonist summary (superseded — kept for record)

~~Cortex-exec already has two live grammar-emitting lanes (`execution_grammar_reducer` via cortex-exec plan/step trace, `chat_grammar_consumer` via hub's turn-level trace). Orch's own arbitration — which lane got picked, whether mind fired, output mode — is emitted nowhere. Chat-lane requests that route through `legacy.plan` execute the full plan/step machinery in cortex-exec but never hit the code path that publishes `GrammarEventV1`, so their step-level mechanics (memory_gate, step, result, egress) are invisible even though hub's coarser turn-level trace (session/utterance/repair-signal) does capture them. Two independent, real gaps, not one.~~ — wrong, see Status above.

## Current architecture (ground truth, verified against HEAD)

**Producer paths into `orion:grammar:event`:**

| Producer | Trigger | trace_id prefix | Layers emitted | Flag |
|---|---|---|---|---|
| `orion-cortex-exec` `handle()` (main.py:457) | direct RPC via `call_cortex_exec`, only when `exec_lane_routing_enabled=True and lane != "chat"` | `cortex.exec:` | intake, plan, memory_gate, step, result, egress, execution | `PUBLISH_CORTEX_EXEC_GRAMMAR` (default `True` since `044d5318`) |
| `orion-cortex-exec` `handle_verb_request()` (main.py:785) | `orion:verb:request`, all lanes incl. chat, `trigger="legacy.plan"` for non-`_DIRECT_VERB_TRIGGERS` verbs | — | **none** — only `_publish_cognition_trace_for_plan_result` (→ `cognition.trace`, not grammar) | n/a — no grammar call exists here |
| `orion-hub` `grammar_emit.py` (`build_chat_turn_grammar_events`) | per chat turn, hub-side, post-hoc | `hub.chat:` | chat (root), context, raw_input, organ_signal (repair) | `PUBLISH_HUB_CHAT_GRAMMAR` (default `True` since `044d5318`) |
| `orion-cortex-orch` | — | — | **does not exist** | n/a |

**Consumer/reducer framework** (`services/orion-substrate-runtime`), fully generic and already mature — do not reinvent any part of this:

- `GRAMMAR_CURSOR_REGISTRY: dict[cursor_name, (source_services: tuple, trace_prefix: str)]` in `app/store.py:48` — the single place a new lane registers.
- One `orion/substrate/<name>_loop/{constants,pipeline,reducer}.py` package per lane (`biometrics_loop`, `execution_loop`, `transport_loop`, `chat_loop`). `pipeline.py` groups events by `trace_id`, filters by `trace_id.startswith(<PREFIX>)`, folds into a projection, emits a receipt.
- `app/worker.py` `_REDUCER_SPECS` (~line 124-154): `(reducer_key, cursor_name, enabled_fn, batch_limit_fn)` tuples, dispatched in `_grammar_reducer_poll_loop`.
- `app/grammar_truth.py`: `REDUCER_KEY_BY_CURSOR` and `ENABLED_BY_REDUCER_KEY` — the "friendly key" remap. **Commit `54997e89` fixed a phantom-lane bug here** — a cursor existing without a correct entry in this map produced ghost lanes in the health snapshot. Any new cursor must add correct entries to both dicts in the same patch.
- `app/reducer_health.py`: backlog/lag/quarantine/heartbeat classification is automatic per `reducer_key` once registered — no bespoke health code needed for a new lane.

**Known landmine — commit `8daeecf7`** (2026-07-09, one day after chat/execution reducers were flipped default-on): `active_execution_trajectory.runs` grew unbounded (29,165 runs / 25MB) because every trace_id got a key that was never evicted. Fixed with LRU-by-`last_updated_at` eviction (`EXECUTION_TRAJECTORY_MAX_RUNS=2000`, `EXECUTION_TRAJECTORY_MAX_AGE_SEC=86400`) plus a `protected_trace_id` guard (the just-written run is excluded from eviction candidates by identity, not inferred from timestamp, because pipeline.py shares one clock across a batch and ties are possible). This is the same unbounded-accumulation shape already hit twice before per memory (`[[feedback_substrate_performance]]`, `[[feedback_execution_merge_cap]]`) — **third occurrence, same bug class.** Any new projection in this design ships the cap in its first commit, not as a follow-up.

## Missing questions (answer before Patch A ships)

1. Is `EXEC_LANE_ROUTING_ENABLED` actually `true` in the live environment? If not, the chat/non-chat split is currently moot and Patch A's urgency is lower. `UNVERIFIED` — check runtime env, not code default.
2. What is chat-lane request rate vs. spark/background, live? Determines whether Patch A needs a sampling knob on day one or can reuse the existing `EXECUTION_TRAJECTORY_MAX_RUNS` cap headroom as-is.
3. Does anything already read `hub.chat:` trace events for step/memory-gate-level detail, making Patch A partially redundant? (Best current answer: no — hub's trace has no step layer at all, only turn-level atoms.)

## Proposed schema / API changes

No changes to `GrammarEventV1`/`GrammarAtomV1`/`GrammarEdgeV1` (orion/schemas/grammar.py) — reused as-is by all patches.

New in Patch C/D only:
- `orion/substrate/route_loop/constants.py`: `ROUTE_GRAMMAR_CURSOR_NAME = "route_grammar_consumer"`, `ROUTE_SOURCE_SERVICE = "orion-cortex-orch"`, `ROUTE_TRACE_PREFIX = "orch.route:"`, `ROUTE_REDUCER_ID`, `ROUTE_PROJECTION_ID = "active_route_arbitration"`, `ROUTE_MAX_RUNS = 2000`, `ROUTE_MAX_AGE_SEC = 86400` (copy execution_loop's cap constants verbatim — same order of magnitude of traffic).
- `orion/schemas/route_projection.py`: `RouteArbitrationProjectionV1` — mirror `ExecutionTrajectoryProjectionV1`'s shape (dict of `trace_id -> run` with `last_updated_at`), fields: `lane`, `lane_reason`, `mind_requested`, `mind_skip_reason`, `output_mode`, `reached_execution_trace: bool | None` (populated later by Patch D's read-time join, not by this reducer — leave `None` at write time, don't cross-query another cursor's store from inside a reducer).
- `GRAMMAR_CURSOR_REGISTRY[ROUTE_GRAMMAR_CURSOR_NAME] = ((ROUTE_SOURCE_SERVICE,), ROUTE_TRACE_PREFIX)` in `store.py`.
- `Settings.enable_route_reducer: bool = Field(False, alias="ENABLE_ROUTE_REDUCER")`, `route_grammar_batch_limit: int = Field(100, alias="ROUTE_GRAMMAR_BATCH_LIMIT")` in substrate-runtime `app/settings.py`.
- `PUBLISH_CORTEX_ORCH_GRAMMAR: bool = Field(False, alias="PUBLISH_CORTEX_ORCH_GRAMMAR")` in orch `app/settings.py`.

**Explicit non-collision constraint:** `orch.route:` must never share a prefix with `cortex.exec:` or `hub.chat:`, and `source_service="orion-cortex-orch"` must never appear in `EXECUTION_SOURCE_SERVICES` or `CHAT_SOURCE_SERVICE`. `GRAMMAR_CURSOR_REGISTRY` matches on `(source_service, trace_prefix)` pairs — get either wrong and route events either vanish (matched by nobody) or get folded into the wrong projection (matched by the wrong cursor). This is the concrete form of "multicollinear" risk from the original brainstorm, and it's a config-correctness risk, not an architectural one — cheap to get right, expensive to debug if wrong.

## Files likely to touch

| Patch | Files |
|---|---|
| B (test-first) | `services/orion-cortex-exec/tests/test_grammar_truth_gate.py` or new file — assert chat_general turn produces a `cortex.exec:`-prefixed grammar event |
| A (chat→execution_trajectory) | `services/orion-cortex-exec/app/main.py` (extract grammar publish helper from `handle()`, call from `handle_verb_request` around line 868), `services/orion-cortex-exec/app/grammar_emit.py` |
| E (result metadata surfacing) | `services/orion-cortex-orch/app/orchestrator.py` (attach lane/mind/output_mode metadata to returned `VerbResultV1`), orion-hub's result/mind-artifact renderer (find via `rg VerbResultV1 services/orion-hub`) |
| C (orch producer, shadow) | `services/orion-cortex-orch/app/orchestrator.py`, new `services/orion-cortex-orch/app/grammar_emit.py` (mirror hub's `grammar_emit.py` pattern — pure builder, no I/O), `app/settings.py` |
| D (route reducer) | `orion/substrate/route_loop/{constants,pipeline,reducer}.py` (new, mirror `chat_loop/`), `orion/schemas/route_projection.py` (new), `services/orion-substrate-runtime/app/store.py` (registry + `fetch_route_grammar_events`), `app/settings.py`, `app/worker.py` (`_REDUCER_SPECS` entry + poll loop dispatch), `app/grammar_truth.py` (`REDUCER_KEY_BY_CURSOR`, `ENABLED_BY_REDUCER_KEY`), `config/substrate-lattice/grammar_producer_registry.v1.yaml` (add `orion-cortex-orch` entry; **also fix `orion-cortex-exec`'s stale `status: planned` → `live` in the same patch** — found stale during ground-truth pass, don't leave it) |

## Non-goals

- No unification of `chat_grammar` and `execution_trajectory` reducers/projections into one — they answer different questions (turn-level conversational context vs. plan/step mechanics) and merging them would be the actual multicollinearity risk the original brainstorm worried about.
- No re-instrumentation of plan/step internals at the orch layer — that's cortex-exec's job and already exists; orch only ever emits pre-dispatch arbitration facts.
- No synchronous/blocking coupling between orch and the bus for any of this — stays fire-and-forget, fail-open, exactly like `PUBLISH_CORTEX_EXEC_GRAMMAR`/`PUBLISH_HUB_CHAT_GRAMMAR` today (wrap in try/except, log-and-continue on publish failure).
- No cross-cursor coverage/reconciliation job that re-reads raw events from multiple streams — coverage answers ("did this turn produce ladder evidence anywhere") come from a read-time join over `grammar_truth.py`'s already-materialized per-cursor snapshot, not a new ingestion pipeline.
- Patch D's `reached_execution_trace` field is a placeholder for future read-time enrichment, not something Patch D itself computes — do not build the join in this pass.

## Acceptance checks

- **B**: new test fails on current `main`, passes after Patch A. Run: `pytest services/orion-cortex-exec/tests/test_grammar_truth_gate.py -q` (or wherever it lands).
- **A**: publish a `chat_general` request through the real `orion:verb:request` path (bus harness or existing integration test); assert a `cortex.exec:`-prefixed `GrammarEventV1` with `layer` in `{intake, plan, step, result}` lands on `orion:grammar:event` within the poll interval. Assert `execution_trajectory` cursor backlog/lag in `grammar_truth.py`'s snapshot does not exceed `max_lag_sec` after a burst of N chat turns (N = whatever the live rate answer from Missing Question 2 turns out to be) — this is the volume regression check for the `8daeecf7`-class bug.
- **C**: with `PUBLISH_CORTEX_ORCH_GRAMMAR=true` in a dev/shadow environment, confirm `orch.route:`-prefixed events appear on `orion:grammar:event` and are picked up by **neither** the `chat_grammar` nor `execution_trajectory` cursor's backlog (proves non-collision) while `route_grammar_consumer` is still unregistered/disabled.
- **D**: `route_grammar_consumer` cursor appears in `grammar_truth.py`'s `enabled_reducers`/`cursor_positions`/`reducer_health_by_name` with a correct friendly key (no phantom lane — diff against `54997e89`'s test pattern in `tests/test_brain_frame_worker.py`). Load-test the reducer with >2000 synthetic trace_ids and assert `active_route_arbitration` projection size stays capped at `ROUTE_MAX_RUNS`, and that the just-written run in a batch is never evicted in the same batch (regression test modeled on `8daeecf7`'s `protected_trace_id` test).
- **E**: manually drive one chat turn through the real UI, confirm `execution_lane`, `mind_requested`, `output_mode` are visible on the client-facing result without any new bus/reducer involvement.

## Recommended sequencing

B → A → E (E is independent, ship whenever) → C → D (D strictly needs C's `orch.route:` prefix to exist and needs `8daeecf7`'s cap pattern copied verbatim before any projection code is written).

Do not start D's projection schema before re-reading `orion/substrate/execution_loop/reducer.py`'s eviction code line-by-line — it is the exact template, not just prior art.
