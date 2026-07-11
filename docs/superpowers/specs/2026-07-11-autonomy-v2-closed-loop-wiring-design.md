# Design: AutonomyStateV2 closed-loop wiring — persistence + orion-unified consumers

**Date:** 2026-07-11
**Status:** Draft — pending implementation go-ahead
**Scope:** `orion/schemas/thought.py`, `orion/harness/prefix.py`, `orion/cognition/prompts/stance_react.j2`, `services/orion-cortex-exec/app/chat_stance.py`, a new Postgres-backed `AutonomyStateV2` store, focused tests.
**Builds on:** `2026-07-10-autonomy-v2-evidence-signal-tension-design.md` (PR #939/#941). That design fixed the reducer's evidence contract and drive-pressure math. This design fixes the fact that even correct output never reaches anything that changes behavior, and that the reducer forgets its own output every turn.
**Does not touch:** `chat_general` verb / Lane A (`chat_stance_brief.j2`, `chat_general.j2`) — deliberately excluded, see Non-goals.

---

## Arsonist summary

`AUTONOMY_STATE_V2_REDUCER_ENABLED=true` was verified live this session: config loads, reducer runs, zero errors, `ctx["chat_autonomy_state_v2"]` populates every turn. That's necessary but not sufficient — it's still empty-shell cognition, just one layer deeper than the config bug that was blocking it.

Two independent gaps, confirmed by reading the actual render/consumption code, not inferred:

1. **No persistence.** `reduce_autonomy_state()` does real fold logic (`previous_state + evidence → new_state`), but nothing writes `result.state` back anywhere. `_load_autonomy_state` re-reads the same V1/graph baseline every turn. The reducer computes a real conclusion and forgets it immediately — a fold with amnesia.
2. **No consumer that changes behavior.** `inputs["autonomy"]["state_v2"/"delta"]` lands in `chat_stance_inputs`, which is read by exactly two things: `router.py` (copies it into a Hub UI debug preview — observability only) and `chat_stance_brief.j2` (doesn't reference `autonomy` at all — grepped, zero hits). The one template on the orion-unified path that *does* touch `chat_stance_inputs`, `stance_react.j2`, only dumps the whole dict raw and unstructured (`{{ stance_inputs }}`) with no instructions — a much weaker signal than the codebase's own precedent for exactly this problem (`mind_coloring`, which gets a named, instructed block).

Net effect: right now, nothing Orion does or says can be traced to its own drive/tension state, on any path.

---

## Goals

- AutonomyStateV2 persists turn-to-turn (non-amnesiac): turn N's `state` becomes turn N+1's `previous_state`.
- The state is visible to the orion-unified stance decision (`stance_react.j2`) as a structured, explicitly-instructed block — not a buried raw dump.
- The state is visible to **every FCC/Claude harness turn** via `compile_harness_prefix()` — this is the actual "real autonomy" lever: it can bias what an agentic turn chooses to work on/prioritize, not just chat tone.
- While in the neighborhood: relocate the curiosity/attention-frame prompt injection (`ORION_CURIOSITY_FRAME_ENABLED`, never turned on, currently Lane-A-only) to the same two Lane B injection points. The detector (`orion/substrate/attention_frame.py`) is already lane-agnostic — this is template wiring only, no detector changes.

## Non-goals

- **No goal-pipeline integration.** Considered and rejected: it's Fuseki/SPARQL-backed (slow) and gated behind a promotion/HITL step — the opposite of "real autonomy" as scoped for this patch. May revisit later as a separate design.
- **No unification of the graph-backed homeostatic-drives system with AutonomyStateV2.** They stay parallel, independently evolving, until a separate decision is made about whether they unite or one dies.
- **No changes to `chat_general.j2` / `chat_stance_brief.j2`.** Lane A is left exactly as-is, including its own (already-present, never-enabled) attention-frame wiring. Explicit choice, not an oversight — leave it, see what happens.
- **No removal of the existing `router.py` `autonomy_state_v2_preview`/`autonomy_state_delta` debug surface.** Keep it for Hub UI observability alongside the new real consumers.

---

## Current architecture (verified live this session)

| Piece | Today |
|---|---|
| Reducer | `chat_stance.py:_run_autonomy_reducer()` → `compile_autonomy_evidence()` + `reduce_autonomy_state()`. Real fold logic, correct math (per 2026-07-10 design), confirmed running with zero errors after the Docker config fix. |
| Read side | `previous_state` comes from `_load_autonomy_state`/`_project_autonomy_from_beliefs` — a V1/graph-backed baseline, re-read fresh every turn. |
| Write side | **Does not exist.** No SQL, no graph mutation, no bus publish of `result.state`/`result.delta`/`result.tensions_minted` anywhere in the repo. |
| Lane A (`chat_general` verb) | Two LLM calls: `chat_stance_brief.j2` (stance synthesis — zero `autonomy` references anywhere in the file) → `chat_general.j2` (reply text — same). Out of scope for this patch. |
| Lane B (orion-unified, `stance_react` verb) | `orion-thought`'s `bus_listener.py` builds a `stance_react` plan (mode=brain), RPCs to cortex-exec, which runs `prepare_brain_reply_context` → `build_chat_stance_inputs(ctx)` → renders `stance_react.j2` → produces `ThoughtEventV1`, described by its own module docstring as "the sole author" of that object for the unified turn. |
| `ThoughtEventV1` | No autonomy field. Flows past chat: `turn_orchestrator.py` passes `thought_event=thought` into `HarnessRunRequestV1`, so it's already the one object both chat and the harness see. |
| `stance_react.j2` | Has a structural precedent for exactly this problem: `{% if mind_coloring %}` — named sub-fields, "advisory — reconcile, do not obey," explicit usage instructions, "does not add output keys." No equivalent block for autonomy or attention_frame today. |
| `orion/harness/prefix.py:compile_harness_prefix()` | Builds the literal system prompt sent to every FCC/Claude harness turn (the agentic/tool-use path, not just chat). Currently renders `imperative`, `tone`, `stance_harness_slice`, `strain_refs`, `grounding_capsule`. No autonomy awareness — this is the real "does it change what Orion does" seam. |
| `orion/substrate/attention_frame.py` | Deterministic (detectors + scoring + templated questions, no LLM call). Gated by `ORION_CURIOSITY_FRAME_ENABLED` (default false, never enabled). Computed inside `build_chat_stance_inputs` regardless of verb — already lane-agnostic, just not rendered anywhere on Lane B. |
| Goal pipeline (`orion/autonomy/goal_actions.py`) | Real, graph-backed (`AUTONOMY_GOALS_GRAPH`, SPARQL via `GraphQueryClient`), lifecycle promote/dismiss/complete, `PromotionEngine` + `hitl_satisfied` gate. Already reaches chat via `goal_hint:{headline}` in `response_priorities` (`chat_stance.py:2725-2737`) and real action via `planned_task_id` → execution-dispatch-runtime. Deliberately not integrated here — see Non-goals. |

---

## Proposed architecture

```text
[Every turn, Lane B only]

stance_react verb (cortex-exec, mode=brain)
  └─ build_chat_stance_inputs(ctx)
       └─ _run_autonomy_reducer()
            ├─ READ  previous_state  ← AutonomyStateV2 store (Postgres, by subject)   [NEW: was V1/graph baseline only]
            ├─ compile_autonomy_evidence() + reduce_autonomy_state()   [unchanged, 2026-07-10 design]
            └─ WRITE result.state → AutonomyStateV2 store                              [NEW: closes the fold]
       └─ ctx["chat_autonomy_state_v2"], ctx["chat_autonomy_state_delta"]  (unchanged)
       └─ autonomy_slice = compact(state, delta)                                       [NEW]

stance_react.j2 render
  ├─ {% if mind_coloring %}   ...                          (existing, unchanged)
  ├─ {% if autonomy_slice %}  dominant_drive / top tensions / trend  [NEW block, mind_coloring-styled]
  └─ {% if chat_attention_frame %}  selected_action / suppressions / open_loops  [NEW block, relocated from Lane A]
       ↓
  ThoughtEventV1{ ..., autonomy_slice }                                                [NEW field]
       ↓
  ┌─────────────────────────────┬───────────────────────────────────────────┐
  │ Hub chat reply (existing)   │ HarnessRunRequestV1(thought_event=thought) │
  └─────────────────────────────┴───────────────────────────┬───────────────┘
                                                              ↓
                                          orion/harness/prefix.py:compile_harness_prefix()
                                            ├─ _format_stance_slice()        (existing)
                                            ├─ _format_autonomy_slice()      [NEW]
                                            └─ _format_attention_frame()     [NEW, relocated from chat_general.j2]
                                                              ↓
                                          FCC/Claude harness system prompt — every agentic tool-use turn
```

---

## Schema / API changes

### `AutonomySliceV1` (new model, `orion/schemas/thought.py`)

```python
class AutonomySliceV1(BaseModel):
    schema_version: Literal["autonomy.slice.v1"] = "autonomy.slice.v1"
    dominant_drive: str | None = None
    active_tensions: list[str] = Field(default_factory=list)   # top 2-3, compact labels only
    pressure_trend: str | None = None    # short derived note, e.g. "reasoning_pressure rising 3 turns"
    confidence: float | None = None
```

Added to `ThoughtEventV1` as `autonomy_slice: AutonomySliceV1 | None = None`.

Raw `drive_pressures` / full evidence never leave the reducer/store — only this compact projection reaches `ThoughtEventV1`. Same narrowing philosophy as `mind_coloring`'s allow-list.

### AutonomyStateV2 store (new)

- Postgres table, subject-keyed, holding latest `AutonomyStateV2` (state blob + updated_at). Exact DSN home is an open question for the implementer — reuse an existing shared Postgres connection (there's already one wired for action-outcomes/memory-cards) vs. a dedicated one.
- Read/write functions live near `orion/autonomy/reducer.py` or a new sibling module — `_run_autonomy_reducer` calls read before reducing, write after.

---

## Files likely to touch

- `orion/schemas/thought.py` — `AutonomySliceV1` + `ThoughtEventV1.autonomy_slice`
- `orion/autonomy/reducer.py` (or new store-client module) — persistence read/write-back
- new SQL migration — `AutonomyStateV2` table DDL
- `services/orion-cortex-exec/app/chat_stance.py` — `_run_autonomy_reducer()` wiring, `autonomy_slice` construction, pass into `stance_react.j2` render context
- `orion/cognition/prompts/stance_react.j2` — `{% if autonomy_slice %}` and `{% if chat_attention_frame %}` blocks
- `orion/harness/prefix.py` — `_format_autonomy_slice()`, `_format_attention_frame()`, wired into `compile_harness_prefix()`
- `.env_example` for whichever service owns the new store connection, synced to `.env`
- tests: reducer round-trip (turn N state → turn N+1 previous_state), template render includes real values, harness prefix includes the rendered block

## Open questions for the implementer

1. Postgres DSN: reuse existing shared connection or provision a dedicated one for `AutonomyStateV2`.
2. Exact derivation of `pressure_trend` — needs at least two persisted states to compute a trend; define behavior for the first-ever turn (no prior state).
3. Whether `autonomy_slice` should be omitted entirely (not just empty) when the reducer has no meaningful signal yet, matching the "omit-when-empty" gate philosophy from the 2026-07-10 design.

## Acceptance checks

- Two consecutive turns for the same subject: turn 2's `previous_state` equals turn 1's `result.state` (persistence round-trip, not re-derived from the V1 graph baseline).
- A live `stance_react.j2` render (inspectable via existing grammar-step logging) contains the `autonomy_slice` block with non-empty real values when the reducer has signal.
- A live harness turn's compiled system prefix contains the rendered autonomy block.
- `chat_general` lane files are byte-identical to before this patch (explicit non-regression check, since Non-goals excludes them).
- Existing `router.py` `autonomy_state_v2_preview` debug surface still works unchanged.
