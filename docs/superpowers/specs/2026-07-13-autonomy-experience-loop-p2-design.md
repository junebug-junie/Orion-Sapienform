# Autonomy experience loop (P2 of the motor-nerve spec) — design

**Date:** 2026-07-13
**Status:** Implementation, this session
**Mode:** Thin patch. Implements P2 of the motor-nerve series (P0/P1/P6 merged) — re-grounded against live code, not the original brainstorm-level P2 section, which assumed a `felt_state_reader` lane and a flag-flip that turn out to be wrong.

## Arsonist summary

P1 gave Layer 9 a real send path: `orion-execution-dispatch-runtime` now dispatches real cortex-exec calls and records results in `substrate_dispatch_results`. Nothing downstream of that knows it happened. Zero references to `episode_journal`, `action_outcome`, or any journaling/memory call exist anywhere in `services/orion-execution-dispatch-runtime/`. An autonomous action today produces a database row nothing reads and nothing narrates.

Two real mechanisms already exist to fix this, both currently orphaned from Layer 9:
1. **`ActionOutcomeEmitV1` bus-emit → sql-writer → `action_outcomes` table** — an always-on, no-feature-toggle route (`orion:autonomy:action:outcome` channel). Currently fired from exactly one call site: `orion/spark/concept_induction/bus_worker.py:1048`, on world-pulse curiosity-fetch completion. Layer 9 never calls it.
2. **`load_action_outcomes(subject)` → `AutonomyReducerInputV1.action_outcomes` → `AutonomyStateV2.last_action_outcomes`** — already wired at `services/orion-cortex-exec/app/chat_stance.py:2284`. This data already reaches chat-turn state. It is never rendered into any prompt template (`grep` confirms zero `.j2` references to `last_action_outcomes` or any autonomy-state action field).

So the real gap isn't "build a new pipe" — it's "connect Layer 9 to pipe #1's producer side, and connect pipe #2's already-flowing data to a template." Two thin patches, not the three-part felt-state-lane design the original brainstorm assumed before this session re-verified the code.

**Explicitly not in scope, corrected from the brainstorm:** flipping `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`. That flag is documented (`.env_example:127-135`, `orion/journaler/worker.py:219-223`) as retired — it gates a duplicate-email path that was deliberately folded into the world-pulse journal to stop sending two emails a day. Flipping it back on would resurrect that regression, not enable anything Layer-9-related. If Layer 9 dispatch results should also produce an episode-journal entry (narrating "I looked into X"), that needs its own, distinctly-named flag and its own call site — scoped as a explicit non-goal for this patch (see below) since it's a second, separable piece of surface area with its own privacy/tone considerations (mirroring `dispatch_autonomy_episode_journal`'s existing machinery, per `orion/autonomy/episode_journal.py`), better done as its own reviewed slice than bundled in.

## Current architecture

- **`services/orion-execution-dispatch-runtime/app/worker.py`** (P1): `_send_one` calls `client.dispatch(...)`, persists a result via `self._store.save_dispatch_result(...)`, and returns a promoted candidate. No bus publish beyond the cortex RPC itself. No `orion.autonomy` imports at all.
- **`orion/autonomy/action_outcomes.py`**: `append_action_outcome(subject, outcome)` writes only to a local file store (`/tmp/orion-action-outcomes.json` by default) — this is a dev/fallback path, not the durable one. The durable path is the bus-emit (`ActionOutcomeEmitV1` → `orion:autonomy:action:outcome` → sql-writer → `action_outcomes` SQL table), fired only from `bus_worker.py:1048`. `load_action_outcomes(subject)` reads SQL first (if `ORION_ACTION_OUTCOME_DB_URL` set) else falls back to the file store — so Layer 9's new outcomes must reach SQL via the bus-emit route to be visible to `load_action_outcomes` in the live (non-dev) config.
- **`orion/autonomy/models.py`**: `ActionOutcomeEmitV1` (flat: `subject, action_id, kind, summary, success, surprise, observed_at`) is what travels the bus; `ActionOutcomeRefV1` (richer: adds `query, articles, salience`) is what `load_action_outcomes` returns after `.to_outcome()` converts back. Layer 9 has no `query`/`articles` to report — the flat `ActionOutcomeEmitV1` shape is a complete fit, no schema change needed.
- **Bus channel**: `orion:autonomy:action:outcome` (`orion/bus/channels.yaml:1738-1744`), `kind: event`, `producer_services: ["orion-spark-concept-induction"]`, `consumer_services: ["orion-sql-writer", "*"]`. Needs `orion-execution-dispatch-runtime` added as a second producer.
- **`BaseEnvelope.correlation_id` must be a real UUID** (confirmed the hard way during P1) — the existing call site passes `env.correlation_id` (already a UUID from the triggering envelope). Layer 9 has no triggering envelope with a UUID correlation_id readily available per-candidate; generate a fresh `uuid4()` per emit, same pattern as P1's `cortex_client.py`.
- **`services/orion-cortex-exec/app/chat_stance.py`**: `_project_reverie_glimpse` (lines 1336-1382) is the established pattern for "typed, bounded, privacy-safe projection from raw ctx into a template-visible field" — reads a raw lane, validates it, extracts only the narration-safe field, documents explicitly what it does NOT pass through. `ctx["chat_autonomy_state"]` already carries `last_action_outcomes: list[ActionOutcomeRefV1]` by the time stance-building reaches template rendering (populated at line 2284, attached to ctx around line 2474-2475).
- **Templates**: `orion/cognition/prompts/chat_general.j2` has an explicit "EVIDENCE-GATED CLAIMS" section (lines 105-114): *"Evidence sources are current-turn inputs only... If evidence is absent, do not claim they occurred."* — the correct, already-designed landing spot for a bounded "recent autonomous actions" reference, so Orion can truthfully say "I looked into X" only when real evidence is present.

## Proposed schema / API changes

**No new schemas.** `ActionOutcomeEmitV1` and `ActionOutcomeRefV1` already exist and already fit Layer 9's shape.

**`services/orion-execution-dispatch-runtime/app/worker.py`**: after `_send_one` computes a result (success, empty, or failed), publish an `ActionOutcomeEmitV1` onto `orion:autonomy:action:outcome`:
- `subject="orion"` (the established self-subject convention — confirmed via `orion/autonomy/reducer.py:237`, `substrate_metabolism.py:41`, `signal_tension.py:64`, all default/use `subject="orion"`).
- `action_id=candidate.dispatch_id`.
- `kind=candidate.dispatch_kind` (already one of `inspect`/`summarize`/`observe`/`noop`).
- `summary=` the `observation` field from `parse_structured_observation`'s result, truncated to a bounded length (mirror `episode_fetch.py`'s discipline — never pass raw multi-KB text through; a short truncation constant, e.g. 280 chars, matches the "typed ref not raw content" rule).
- `success=(status == "success")`, `None` for `status == "failed"` (unknown, not false — an RPC failure isn't evidence the action itself didn't happen) — actually: `success=True` iff `status=="success"`, `success=False` iff `status in ("empty", "failed")` (both are real, evidenced non-successes; `None` is reserved for genuinely unknown, which doesn't occur here since every candidate reaching this point has a determined status).
- `surprise=0.0` (no computed surprise signal exists for this action type yet; explicit `0.0` rather than inventing one).
- `observed_at=dispatched_at` (the same timestamp already stamped on the candidate).

Failure to publish this event must never fail the tick — wrap in the same `try/except: logger.warning(...)` discipline already used for `_notify_tripwire`.

**`orion/bus/channels.yaml`**: add `orion-execution-dispatch-runtime` to `orion:autonomy:action:outcome`'s `producer_services`.

**`services/orion-cortex-exec/app/chat_stance.py`**: new `_project_recent_dispatch_actions(ctx)` function, mirroring `_project_reverie_glimpse`'s shape exactly:
- Reads `ctx["chat_autonomy_state"]` (already populated), extracts `.last_action_outcomes` if present.
- Filters to a small bounded window (reuse the existing `_MAX_OUTCOMES = 12` convention from `action_outcomes.py`, but only surface the most recent 1-3 in the prompt — a wall of 12 actions is not what "truthfully reference what I did" needs; cap at 3, newest-first).
- Projects each to `{kind, summary, success, observed_at}` only — never `query`/`articles`/`salience`/`action_id` (internal correlation data, matches `_project_reverie_glimpse`'s "never `evidence_refs`/`coalition`/`chain_id`" precedent).
- Result → `ctx["chat_recent_dispatch_actions"]`, a list (possibly empty).

**`orion/cognition/prompts/chat_general.j2`**: render `chat_recent_dispatch_actions` inside the existing "EVIDENCE-GATED CLAIMS" section, following that section's existing tone/format — only present if non-empty, formatted as bounded evidence lines, not prose Orion is meant to copy verbatim.

## Files likely to touch

- `services/orion-execution-dispatch-runtime/app/worker.py` — emit `ActionOutcomeEmitV1` after each real send.
- `orion/bus/channels.yaml` — producer registration.
- `services/orion-cortex-exec/app/chat_stance.py` — `_project_recent_dispatch_actions`.
- `orion/cognition/prompts/chat_general.j2` — render the new ctx key in the evidence-gated section.
- New bus-catalog test for the producer registration (mirrors P1's pattern).
- New/extended tests: `services/orion-execution-dispatch-runtime`'s worker test file (emit happens on success/empty/failed, never raises the tick); `services/orion-cortex-exec/tests`' chat_stance test file (projection bounds/truncates/never leaks raw fields, empty-safe).

## Non-goals

- **Episode journal integration for Layer 9 dispatch results** — a real, separate piece of surface area (new flag, new call site reusing `dispatch_autonomy_episode_journal`, its own narrative-seed privacy discipline). Explicitly deferred to its own patch, not bundled here — see Arsonist summary. `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED` stays untouched.
- **No `felt_state_reader.py` lane.** The original brainstorm's plan assumed one was needed; re-grounding found the data already reaches chat-turn state via a different, already-wired path (`AutonomyStateV2.last_action_outcomes`). Adding a redundant lane would be exactly the kind of duplicate-pipe cathedral this project's mandate bans.
- **No new `success`/`surprise` scoring model.** `success` is a direct boolean from dispatch status; `surprise=0.0` is an honest placeholder, not a computed signal — computing real surprise for this action type is future work if it turns out to matter, not invented here.
- **No mutating dispatch, no flag flips on `EXECUTION_DISPATCH_MODE`.** This patch only wires what happens *after* a real dispatch already occurred (which today only happens with `EXECUTION_DISPATCH_MODE=dispatch_read_only`, still not the default anywhere).

## Acceptance checks

1. `_send_one`'s three outcome paths (success, empty, failed) each produce exactly one `ActionOutcomeEmitV1` publish attempt, verified via a mocked bus in the worker test suite.
2. A publish failure (bus down) logs a warning and does not raise out of `_send_one`/`_send_prepared_candidates` — the tick completes and the frame still gets persisted.
3. `_project_recent_dispatch_actions` on an empty/missing `last_action_outcomes` returns `[]`, never raises.
4. `_project_recent_dispatch_actions` never includes `query`/`articles`/`salience`/`action_id` in its output — a test constructs an `ActionOutcomeRefV1` with all fields populated and asserts the projected dict's keys are exactly `{kind, summary, success, observed_at}`.
5. `chat_general.j2` renders without error whether `chat_recent_dispatch_actions` is present, empty, or absent from ctx.
6. Bus-catalog test confirms `orion-execution-dispatch-runtime` is a registered producer on `orion:autonomy:action:outcome`.

## Recommended next patch

The deferred episode-journal-for-Layer-9-dispatch piece named in Non-goals, once this lands and (per P1's own burn-in note) real dispatch data exists to journal. P3 (drive satisfaction + operator inform) is otherwise the next item in the parent series' dependency order.
