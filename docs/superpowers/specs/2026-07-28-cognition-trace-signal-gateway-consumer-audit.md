# Cognition trace consumer audit + dead/redundant-consumer cleanup

**Status**: Implemented. Branch `fix/kill-cognition-trace-dead-consumers`.

**Trigger**: chat-session investigation starting from "does the Landing Pad still do anything
meaningful" and widening into "what actually consumes `orion:cognition:trace`, and is any of it
duplicating substrate/digester/organ-signal work." Every claim below was checked against live
code and, where noted, live running state (Redis, Postgres) — not assumed.

**Ground rule, per `AGENTS.md` 0A**: kill means kill, no fallback to the thing being killed.
Where a consumer is genuinely redundant with a real, already-live mechanism, remove it rather
than leave a partial exclusion in place.

---

## 1. What `orion:cognition:trace` is

`CognitionTracePayload` (`orion/schemas/telemetry/cognition_trace.py`), published by
`orion-cortex-exec` at the end of every plan execution (`main.py::_publish_cognition_trace_for_plan_result`).
Full-fidelity per-turn record: `final_text`, full `steps: List[StepExecutionResult]` (status,
verb_name, result dict, spark_vector, artifacts, latency_ms, error, logs), `recall_used`/
`recall_debug`, `metadata`. Contract doc: `docs/cognition_trace_contracts.md` (also updated by
this change — it described a design that was never fully real, or has since drifted).

## 2. Consumer-by-consumer verdict

| consumer | verdict | evidence |
|---|---|---|
| **orion-sql-writer** (`cognition_traces` table) | **keep — real, live, full-fidelity** | Live query 2026-07-28: 7393+ rows, newest seconds old. The only place with the raw record (final_text, full step results, raw errors). Nothing else in this tree has this fidelity. |
| **orion-rdf-writer** | already dead, no action needed | Unsubscribed 2026-07-22 as part of the Fuseki decommission campaign (`services/orion-rdf-writer/app/settings.py:82-93`) — Fuseki traffic was ~750 writes/6h of pure redundancy against sql-writer. Handler functions already deleted, only an explanatory comment remains. |
| **orion-vector-writer** | **dead code — removed this change** | `elif kind == "cognition.trace":` branch in `main.py` embedded `final_text` into a `orion_cognition` Chroma collection, but `VECTOR_WRITER_SUBSCRIBE_CHANNELS` (default and live `.env`) has never included `orion:cognition:trace`. Never reachable. Removed, comment left explaining why (mirrors the rdf-writer retirement style). |
| **orion-landing-pad** | **catalog drift — fixed this change** | `channels.yaml` listed it as a consumer, but `PAD_INPUT_ALLOWLIST_PATTERNS` (`orion:telemetry:*,orion:cortex:*,orion:spark:*`) has never matched `orion:cognition:trace`. Never a real consumer; catalog entry was simply wrong. Removed from `consumer_services`. |
| **orion-signal-gateway** (`CognitionTraceAdapter` → `cognition_run`/`cognition_step` `OrionSignalV1`) | **redundant with a real mechanism — removed this change** | See §3. |
| **orion-spark-introspector** (`handle_trace()`, real non-heartbeat turns) | **left alone — separate track** | Real subscription, but the step-derived heuristic (valence/arousal from success/fail counts) is computed then immediately overwritten by `_get_phi_stats()` before use — dead in effect, not in code. This service's future is being decided independently in `docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-substrate-convergence.md` (phi/EKG found largely theater on separate grounds). Not touched here to avoid stepping on that track; `docs/cognition_trace_contracts.md` updated to describe the discard bug accurately in the meantime. |
| **orion-substrate-runtime** | not a consumer of this channel at all | Has its own, doctrinally-designated mechanism instead — see §3. |
| **orion-field-digester** | not a consumer, not relevant | Zero references anywhere in the service; different domain (biometrics/telemetry decay). |

## 3. Why the Signal Gateway path was redundant, not just weak

`orion-signal-gateway`'s own README states its job plainly: normalize raw organ-bus events into
`OrionSignalV1` via hand-written per-organ adapters — an **inferred** interpretation layer
(e.g. `CognitionTraceAdapter`'s `"reasoning_present": bool(meta.get(...)) or bool(recall_debug)`
is a judgment call baked into adapter code, not an observation).

Two other mechanisms already do a version of this job, one of them for real:

- **`orion-bus-mirror`** mechanically records real observed bus traffic — actual publish edges
  and actual causal follow-on (via `correlation_id` matching across organs within a TTL window)
  — into FalkorDB (`orion_bus_synapse`), with real statistics (`gap_zscore`, `latency_zscore`)
  off real inter-arrival times. Not inferred. Left untouched by this change (different scope,
  transport-layer rather than per-turn-semantic), but it's the more honest version of "what
  happened, causally" at the mesh level.
- **`orion-substrate-runtime`'s `execution_grammar_reducer`** consumes `GrammarEventV1` (the
  doctrinal "substrate trace" — `docs/context-engineering/00_substrate_trace_doctrine.md` —
  emitted independently by `orion-cortex-exec`'s `grammar_emit.py` alongside, not from,
  `cognition_trace`) and materializes a live, HTTP-queryable `/projections/execution_trajectory`
  (`enable_execution_trajectory_reducer` defaults `True`). Field-verified: atoms carry
  `atom_type`, `semantic_role`, `confidence`, `salience`, a generated summary string, and causal
  edges (`temporal_successor`/`derived_from`/`contains`) — a real causal DAG, not raw payload.
  **This already has a real consumer**: `orion-spark-introspector/app/inner_state.py`
  (docstring: *"execution_trajectory/reasoning_activity projections — independently real"`)
  HTTP-polls this projection to build `InnerStateFeaturesV1` (execution_load, reasoning_load,
  recall_gate_fired).

So `CognitionTraceAdapter`'s `cognition_run`/`cognition_step` signals were a **third**,
independent re-derivation of "how did this turn go, causally" — built from raw `cognition_trace`
instead of from the doctrine's own designated substrate trace — and its only consumer
(`orion-hub`'s `SignalsInspectCache`, a debug/inspection cache for a UI panel) never acted on it.
Confirmed via `orion:signals`/`orion:signals:*` subscriber search: the one other real consumer,
`orion-spark-concept-induction`, only takes the `biometrics`/`spark`/`equilibrium` per-organ
sub-channels for homeostatic drives — cortex_exec/cognition signals were explicitly excluded
from that even before this change.

**GrammarEventV1 is not a superset of `cognition_trace`, by design** (doctrine: "not a raw
payload dump... not a full prompt/completion store") — atoms carry summary strings and pointer
refs (`payload_ref`), not `final_text`, raw `step.result`, `artifacts`, `spark_vector`, `logs`,
or raw error text (only a short classified `error_kind`). This is why sql-writer's
`cognition_traces` table stays — it's the only full-fidelity record — while the *shadow/causal*
representation of the same data should come from one place (`execution_trajectory`), not two.

## 4. What changed

- Deleted `orion/signals/adapters/cognition_trace.py` (`CognitionTraceAdapter`).
- Unregistered it from `orion/signals/adapters/__init__.py` (`ADAPTERS` list, imports, `__all__`).
- `orion/signals/registry.py`: `cortex_exec` organ entry kept (still referenced as
  `causal_parent_organs` by `llm_gateway`), but `signal_kinds`/`bus_channels` emptied and a note
  added explaining no adapter currently produces this organ's signals.
- Removed `orion:cognition:*` from `orion-signal-gateway`'s `ORGAN_CHANNELS` default
  (`app/settings.py`) — no adapter handles it anymore, so subscribing was pure overhead.
- Removed the dead `elif kind == "cognition.trace":` branch in
  `services/orion-vector-writer/app/main.py`.
- `orion/bus/channels.yaml`: removed `orion-landing-pad` from `orion:cognition:trace`'s
  `consumer_services` (catalog drift fix); added an explanatory comment for both this and the
  signal-gateway removal.
- Updated `docs/cognition_trace_contracts.md` §3 (Consumer Behavior) to describe live reality
  instead of the original, partially-never-real design.
- Updated `services/orion-signal-gateway/README.md`'s "Cognition trace preflight" section.

## 5. Non-goals / left alone

- `orion-spark-introspector`'s `handle_trace()` subscription and discarded-heuristic bug — left
  for the separate spark-introspector retirement track to decide.
- `orion-bus-mirror` — not touched; noted as the more honest mechanical alternative, not folded
  into this change.
- No new consumer was wired up for `execution_trajectory` in `orion-substrate-runtime` or
  `orion-field-digester` — `inner_state.py` already exists as a real consumer; this change didn't
  add scope there, it just stopped growing a redundant fourth path.
- `orion/substrate/signal_bridge.py` (`SubstrateSignalBusWorker`'s supported-inputs list still
  names `cortex_exec`/`cognition_run` and `cortex_exec`/`cognition_step`, now permanently
  unreachable) was found in review but **not cleaned up here** — that worker class is itself
  never instantiated anywhere in the repo, a separate, pre-existing dead-code question orthogonal
  to this audit. Docstring updated to say so; the worker itself is a follow-up, not this change's
  scope.

## 6. Acceptance checks

- `orion/signals/adapters/__init__.py` imports and instantiates cleanly with `CognitionTraceAdapter`
  removed (no dangling references).
- `orion-signal-gateway` and `orion-hub` test suites pass unchanged — confirmed the two test
  files touching `cognition_run`/`cognition_step` (`test_processor_multi_emission.py`,
  `test_signals_inspect_api.py`, `test_correlation_chain_fallback.py`) use local fixture
  adapters / fabricated `OrionSignalV1` objects, not `CognitionTraceAdapter` itself — unaffected
  by its removal.
- `orion-vector-writer` test suite has no tests referencing the removed branch (confirmed via
  grep before removal).
- `python scripts/check_bus_channels.py` / `check_schema_registry.py` pass against the edited
  `channels.yaml`.
