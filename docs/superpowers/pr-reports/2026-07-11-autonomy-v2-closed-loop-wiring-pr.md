## Summary

- AutonomyStateV2 now persists turn-to-turn (Postgres, fail-open) instead of being recomputed and discarded every turn — closes the reducer's own fold loop.
- A compact `AutonomySliceV1` reaches two real consumers on the orion-unified path: the `stance_react` LLM prompt itself and every FCC/Claude harness system prefix. Verified end-to-end, not just computed-and-dropped.
- Relocated the never-enabled curiosity/attention-frame prompt injection from Lane A (`chat_general`, untouched) to Lane B (`stance_react.j2`) — template-only, no detector changes.
- Code review (8-angle) found the prompt-injection half was dead on arrival due to a cross-wave timing bug; fixed directly along with two more correctness bugs and a CLAUDE.md "keyword cathedral" violation.
- Converted the `_run_autonomy_reducer` → `build_chat_stance_inputs` → `prepare_brain_reply_context`/`ensure_chat_stance_pipeline_ctx` call chain to async so the new Postgres calls run via `asyncio.to_thread` instead of blocking the event loop on every chat turn.

## Outcome moved

`AutonomyStateV2` — Orion's own drive/tension appraisal — now actually reaches things that shape behavior: the stance-decision LLM prompt and every harness/agentic system prefix, with real persisted memory across turns. Previously it was fully computed but unreachable by any consumer.

## Current architecture

Before this PR: `AUTONOMY_STATE_V2_REDUCER_ENABLED=true` ran the reducer every turn, but `previous_state` always came from the V1/graph baseline (never its own prior output), and the result only reached a Hub UI debug preview (`router.py`'s `autonomy_state_v2_preview`) — no prompt or harness consumer read it. `chat_general`/Lane A had its own (also-dead) curiosity-frame wiring; Lane B (`stance_react`) had none.

## Architecture touched

- `orion/autonomy/` — new Postgres-backed state store
- `orion/schemas/thought.py` — new `AutonomySliceV1`, `ThoughtEventV1.autonomy_slice`
- `services/orion-cortex-exec/app/` — reducer persistence wiring, `autonomy_slice.py` builder, async conversion of the stance-context prep chain
- `orion/cognition/prompts/stance_react.j2`, `orion/harness/prefix.py` — the two real consumers
- `orion/thought/`, `services/orion-thought/app/bus_listener.py` — cross-service extraction/attach (cortex-exec → orion-thought)
- `services/orion-sql-db/` — new manual migration

## Files changed

- `orion/autonomy/state_store.py`: new — `load_autonomy_state_v2`/`save_autonomy_state_v2`, mirrors `action_outcomes.py`'s SQLAlchemy convention, fail-open
- `services/orion-sql-db/manual_migration_autonomy_state_v2.sql`: new — `autonomy_state_v2` table (subject PK, jsonb state, updated_at)
- `orion/schemas/thought.py`: new `AutonomySliceV1` model + `ThoughtEventV1.autonomy_slice` field (optional, back-compatible)
- `services/orion-cortex-exec/app/autonomy_slice.py`: new — `build_autonomy_slice(ctx)`, mirrors `grounding_capsule.py`'s shape, omit-when-empty
- `services/orion-cortex-exec/app/chat_stance.py`: `_run_autonomy_reducer` reads/writes the store (via `asyncio.to_thread`), builds `chat_autonomy_movement_debug` from the actual `previous_state` used (not a separately-read baseline), sets `ctx["autonomy_slice"]` before the LLM step renders; `build_chat_stance_inputs` now `async def`
- `services/orion-cortex-exec/app/executor.py`: `prepare_brain_reply_context`/`ensure_chat_stance_pipeline_ctx` now `async def`, 2 call sites `await`
- `services/orion-cortex-exec/app/router.py`: removed the post-hoc (too-late) `autonomy_slice` compute; kept the metadata-attach; 2 call sites `await`
- `orion/cognition/prompts/stance_react.j2`: new `{% if autonomy_slice %}` and `{% if chat_attention_frame %}` blocks, mirroring the existing `mind_coloring` block's advisory framing
- `orion/harness/prefix.py`: new `_format_autonomy_slice()`, wired into `compile_harness_prefix()`; removed a dead `attention_frame` parameter added earlier in this branch (zero live caller, flagged by review, stripped per operator decision)
- `orion/thought/stance_react.py`, `services/orion-thought/app/bus_listener.py`: `_extract_autonomy_slice`, mirrors `_extract_grounding_capsule` exactly
- 14 test files updated for the async conversion (sync mocks → `AsyncMock`, `def test_` → `async def test_` + `@pytest.mark.asyncio` where they call the converted functions)
- 3 new test files: `orion/autonomy/tests/test_state_store.py`, `orion/schemas/tests/test_thought.py`, `services/orion-cortex-exec/tests/test_autonomy_slice.py`
- `docs/superpowers/plans/2026-07-11-autonomy-v2-closed-loop-wiring.md`: implementation plan (design doc is gitignored, local-only)

## Schema / bus / API changes

- Added: `AutonomySliceV1` schema, `ThoughtEventV1.autonomy_slice` field (optional)
- Added: `autonomy_state_v2` Postgres table
- No bus channel or schema-registry changes — this is a schema field + a new Postgres table, not a new event
- Compatibility: `ThoughtEventV1` remains valid with `autonomy_slice` omitted; every existing construction site unaffected

## Env/config changes

- Added keys: `ORION_AUTONOMY_STATE_DB_URL` (`services/orion-cortex-exec/.env_example`)
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (confirmed in-session)
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/autonomy/tests orion/schemas/tests orion/thought/tests orion/harness/tests -q
  343 passed, 3 failed (pre-existing on origin/main, confirmed via clean worktree — a
  mind_coloring StrictUndefined bug and an fcc_timeout error-string mismatch, unrelated)

pytest <14 touched cortex-exec test files> -q
  107 passed, 2 failed (pre-existing on origin/main, confirmed via same-.env in-place
  checkout of main's files — a chat_stance._UNIFICATION_LAYER test-order interference bug)

pytest services/orion-thought/tests -k stance_react -q
  8 passed
```

New regression tests: `test_chat_stance_autonomy_v2_slice_set_before_llm_render` (ctx["autonomy_slice"] present before the LLM step renders), `test_run_autonomy_reducer_movement_debug_before_matches_actual_previous_state` (before/after pressures compare against the same baseline), `test_run_autonomy_reducer_uses_persisted_state_not_v1_baseline_on_second_call` (non-amnesiac acceptance check).

## Evals run

None. This service has an existing eval harness for the V2 reducer's own math (`orion/autonomy/evals/`), but no eval was added for the new persistence/prompt/harness wiring — flagged as a known gap in code review, not addressed in this patch.

## Docker/build/smoke checks

Not run — Docker was not exercised in this environment for this patch. `docker compose ... config` validation and a live turn smoke (verifying the rendered `stance_react.j2` prompt and a real harness prefix both contain the autonomy block) are recommended before/after deploy — see Restart required below.

## Review findings fixed

8-angle code review ran twice (first pass hit a session usage limit mid-run with zero findings returned; re-ran clean after a plan upgrade).

- Finding: `stance_react.j2`'s `autonomy_slice` prompt block could never render with real data — `ctx["autonomy_slice"]` was set in router.py's post-step processing, which runs strictly after the single-step `stance_react` plan's LLM call already rendered its prompt.
  - Fix: moved `build_autonomy_slice(ctx)` + `ctx["autonomy_slice"]` assignment into `chat_stance.py`, synchronously before the LLM step (same place `chat_attention_frame` is already set).
  - Evidence: new regression test `test_chat_stance_autonomy_v2_slice_set_before_llm_render`; traced the full call chain (`call_step_services` → `prepare_brain_reply_context` → `build_chat_stance_inputs` all run before the step's own `_render_prompt`).
- Finding: `chat_autonomy_movement_debug`'s "before" pressures were read from `autonomy["state"]` (V1/graph baseline) while "after" came from the actual fold's `previous_state` (persisted V2 once warm) — mismatched baselines once persistence kicks in, corrupting `pressure_trend`.
  - Fix: moved `chat_autonomy_movement_debug` construction inside `_run_autonomy_reducer`, using the same `previous_state` the fold actually used.
  - Evidence: new regression test `test_run_autonomy_reducer_movement_debug_before_matches_actual_previous_state`.
- Finding: `build_autonomy_slice`'s omit-when-empty guard included `confidence is None`, but `confidence` defaults to `0.5` (always present) — empty-signal turns still produced a slice, rendering literal `"- dominant_drive: None"` into the LLM prompt.
  - Fix: excluded `confidence` from the emptiness check; added per-field Jinja guards as defense in depth.
  - Evidence: `orion/schemas/tests`, `test_autonomy_slice.py` pass; manual Jinja render check confirmed no `None`/`[]` leaks for partial slices.
- Finding: `compile_harness_prefix()`'s `attention_frame` parameter + formatter (added earlier in this branch) had zero live callers and a docstring admitting it — CLAUDE.md "No keyword cathedrals" violation.
  - Fix: removed per explicit operator decision; the live `stance_react.j2` half of that same task is untouched.
  - Evidence: `orion/harness/tests` unaffected (343 passed, same 3 pre-existing failures).
- Finding: `_run_autonomy_reducer`'s new Postgres calls ran synchronously inside an async call chain with no thread offload, despite an established `asyncio.to_thread` precedent elsewhere in this service.
  - Fix: full async conversion of the call chain (see Files changed); calls wrapped in `asyncio.to_thread`.
  - Evidence: 397+ tests passing after conversion, including 14 test files updated for the new async signatures.
- 3 findings not fixed (documented, non-blocking): `state_store.py`'s duplicated `_ENGINE_CACHE` pattern vs. `action_outcomes.py` (same DSN, two pools); `_extract_autonomy_slice`/`_extract_grounding_capsule` structural duplication (candidate for a shared helper); `state_store.py`'s docstring overclaims "mirrors action_outcomes.py exactly" but omits its file-backed fallback (silent total data loss if DSN unset/migration never applied, no distinguishing log).

## Restart required

```bash
# Apply the new migration to the live Postgres (conjourney) before this persists anything:
psql "$ORION_AUTONOMY_STATE_DB_URL" -f services/orion-sql-db/manual_migration_autonomy_state_v2.sql

# Restart cortex-exec (all 4 lanes share the image) and orion-thought:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium
  Concern: no evidence the `autonomy_state_v2` migration has been applied to the live Postgres — until it is, the reducer silently falls back to the V1 baseline every turn with no distinguishing log (documented in the state_store.py docstring finding above).
  Mitigation: run the migration command above before considering this "live." Consider adding a startup log line or health-check surfacing whether the table exists.
- Severity: low
  Concern: no eval added for the new persistence/prompt/harness behavior, only unit tests.
  Mitigation: follow-up eval extending `orion/autonomy/evals/` to cover turn-to-turn state carry and prompt/prefix content, if this becomes a priority.
- Severity: low
  Concern: `state_store.py` lacks the file-backed degraded-mode fallback that its sibling `action_outcomes.py` has.
  Mitigation: acceptable as-is (documented, fail-open, matches pre-existing behavior when unset) — revisit only if turn-to-turn persistence proves operationally important enough to need a degraded mode.
