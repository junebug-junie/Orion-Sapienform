# PR report: surface recent dispatch-action evidence into stance_react's pre-motor prompt

Branch: `feat/stance-react-dispatch-evidence`
Spec: `docs/superpowers/specs/2026-07-14-stance-react-dispatch-evidence-design.md`

## Summary

- `AutonomySliceV1` (`orion/schemas/thought.py`) gains `recent_actions: list[str]` — compact, one-line summaries of recent successful Layer-9 dispatch-action outcomes.
- Reuses an existing, live, already-proven pipe end to end (`chat_stance.py` → `router.py`'s metadata map-on → `orion-thought`'s `bus_listener.py` extraction → `ThoughtEventV1.autonomy_slice` → `stance_react.j2`'s rendered prompt) rather than building a parallel one — zero changes needed to either `router.py` or `bus_listener.py`.
- `orion/cognition/prompts/stance_react.j2` renders the new field inside its existing `autonomy_slice` advisory block, shaping `imperative`/`tone` before the FCC motor runs.
- Fixed a real trap along the way: `build_autonomy_slice`'s old early return (`if not state: return None`) ran before any recent-actions consideration and would have silently dropped the whole feature whenever the V2 reducer state was empty.
- Code review (self-run, high effort) found one real latent bug in the new code (cap-limit off-by-one), fixed and regression-tested.

## Outcome moved

Real Layer-9 dispatch-action evidence (proven live earlier today: a real `inspect` action produced a real observation, a real durable `action_outcomes` row, and a real drive-pressure relief) previously only reached `chat_general.j2` — the Brain-mode chat path, which is on a documented sunset track (`docs/superpowers/checklists/2026-07-05-unified-turn-sunset.md`). `ORION_UNIFIED_TURN_ENABLED=true` is live right now, meaning unified-turn is the actual active default chat path, and it never saw this evidence at all. Now it does, via the same pre-motor mechanism already carrying drive/tension state (`autonomy_slice`) into that path today.

## Current architecture

`build_chat_stance_inputs` (`services/orion-cortex-exec/app/chat_stance.py`) is a single shared context builder invoked for both the `chat_general` verb (Brain mode) and the `stance_react` verb (unified-turn's pre-motor cognition pass). It already computed `ctx["chat_recent_dispatch_actions"]` (via `_project_recent_dispatch_actions`, querying `load_action_outcomes(subject="orion")` directly, capped at `_MAX_RECENT_DISPATCH_ACTIONS=3`) and `ctx["autonomy_slice"]` (via `build_autonomy_slice`, from `AutonomyStateV2` reducer output) — but only the latter had a pipe reaching `stance_react.j2`. The former was rendered exclusively by `chat_general.j2`.

The full pre-motor pipe `autonomy_slice` already used, traced hop by hop in the design spec:

1. `chat_stance.py` builds `ctx["autonomy_slice"]`.
2. `router.py` (gated on `plan.verb_name == "stance_react"` and `step.step_name == "llm_stance_react"`) copies `ctx["autonomy_slice"]` into that step's result `metadata["autonomy_slice"]`.
3. `services/orion-thought/app/bus_listener.py`'s `_extract_autonomy_slice` reads it back out of that metadata and validates it as `AutonomySliceV1`.
4. `orion/thought/stance_react.py`'s `parse_stance_react_payload` strips any LLM-authored `autonomy_slice` from the raw JSON first (documented: "assembled deterministically in cortex-exec... never authored by the stance LLM"), then the trusted value from step 3 is attached onto the final `ThoughtEventV1.autonomy_slice`.
5. `stance_react.j2` renders it directly into the stance LLM's own prompt as a "PRIOR SELF-SIGNAL (advisory)" block, explicitly instructing the model to use it to color `imperative`/`tone`.
6. That `thought_event` is what `orion.hub.turn_orchestrator.execute_unified_turn` hands to `HarnessRunRequestV1`, which reaches the FCC motor.

## Architecture touched

- `orion/schemas/thought.py` — one new field on `AutonomySliceV1`.
- `services/orion-cortex-exec/app/autonomy_slice.py` — `build_autonomy_slice` now folds in recent dispatch-action evidence; omit-check extended.
- `services/orion-cortex-exec/app/chat_stance.py` — reordered `_project_recent_dispatch_actions` to run before the gated V2-reducer block so its output is available when `build_autonomy_slice` runs. `chat_general.j2`'s existing direct read of `ctx["chat_recent_dispatch_actions"]` is unaffected (same key, same content, function is side-effect-free and independent of `ctx` per its own docstring).
- `orion/cognition/prompts/stance_react.j2` — one new template line.

Not touched (deliberately, per spec's non-goals): `router.py`'s metadata map-on, `services/orion-thought/app/bus_listener.py`'s extraction, `chat_general.j2`.

## Files changed

- `orion/schemas/thought.py`: `AutonomySliceV1.recent_actions: list[str]` + updated docstring.
- `services/orion-cortex-exec/app/autonomy_slice.py`: new `_format_recent_actions` helper (success-only filter, `"{kind}: {summary}"` formatting, 160-char truncation budget, cap via `max_recent_actions` param); `build_autonomy_slice` reads `ctx["chat_recent_dispatch_actions"]`, no longer early-returns on empty reducer state (state defaults to `{}` instead), omit-check now also considers `recent_actions`.
- `services/orion-cortex-exec/app/chat_stance.py`: moved the `_project_recent_dispatch_actions` call earlier in `build_chat_stance_inputs`; `build_autonomy_slice` call site now passes `max_recent_actions=_MAX_RECENT_DISPATCH_ACTIONS` explicitly (single source of truth for the cap, not a second hardcoded `3`).
- `orion/cognition/prompts/stance_react.j2`: one new `{% if autonomy_slice.recent_actions %}` line inside the existing advisory block, matching the `active_tensions` pattern exactly.
- `services/orion-cortex-exec/tests/test_autonomy_slice.py`: 8 new tests — cap-at-3 with 5 successes, failed/`None`-success exclusion, char-budget truncation, the omit-check trap (slice emitted when only recent_actions has signal), fail-open on missing/empty/malformed input, full-empty `None` case, and the review-fix regression (`max_recent_actions<=0` yields zero entries, not one).
- `orion/schemas/tests/test_thought.py`: extended the exact-dict-equality round-trip test for the new field; added an explicit acceptance-check test proving `AutonomySliceV1` with `recent_actions` populated round-trips through `model_dump`/`model_validate` unchanged (the proof that `router.py`/`bus_listener.py` need zero code changes).
- `services/orion-cortex-exec/README.md`: not touched — no existing section documents `AutonomySliceV1`'s field shape, so none was invented.

## Schema / bus / API changes

- Added: `AutonomySliceV1.recent_actions: list[str] = []`. Additive, safe default — `schema_version` unchanged (`"autonomy.slice.v1"`), no version bump. Old payloads without the field still validate; new payloads with it round-trip through every existing hop with zero code changes elsewhere.
- Removed: none.
- Renamed: none.
- Behavior changed: `build_autonomy_slice` no longer returns `None` immediately when `AutonomyStateV2` reducer state is empty — it now falls through to consider `recent_actions` before deciding whether to omit. Net effect is strictly additive (a case that previously always omitted now sometimes doesn't, only when there's real recent-action signal to report); no existing non-empty-state behavior changes.
- Compatibility notes: none needed.

## Env/config changes

None. No new env vars; the feature inherits the existing `AUTONOMY_STATE_V2_REDUCER_ENABLED` and `settings.orion_unified_grounding_enabled` gates already governing `autonomy_slice`'s construction and delivery.

## Tests run

```text
$SCRATCH/p3p5-venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_autonomy_slice.py \
  services/orion-cortex-exec/tests/test_chat_stance_recent_dispatch_actions_projection.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py \
  services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py \
  orion/schemas/tests/test_thought.py -q
58 passed
```

The spec's literal `pytest services/orion-cortex-exec/tests/ -k "..."` commands hit a pre-existing, unrelated collection error (`ValueError: Verb already registered: legacy.plan`) across 12 test files when the whole `tests/` directory is collected. Verified identical on a clean `origin/main` tree with this patch stashed out — confirmed pre-existing, not introduced by this change. Ran the specific affected files directly instead (command above), which is the standard workaround already established earlier this session for this exact class of cross-service collection collision.

`git diff --check` clean throughout.

## Evals run

None applicable — this is a small, deterministic context-plumbing change with full unit coverage. No eval harness exists for `AutonomySliceV1`/`stance_react` specifically. Live verification of the end-to-end chat effect (does `ThoughtEventV1.autonomy_slice.recent_actions` actually show up non-empty on a real unified-turn chat turn shortly after a real Layer-9 dispatch) was not performed in this session — recommend a short manual check post-merge using the same technique used earlier today to verify the P3 satisfaction-tension pipeline (temporary diagnostic log, live watch during a real chat turn, then remove).

## Docker/build/smoke checks

Not run. This patch touches shared Python modules loaded by `orion-cortex-exec` and (transitively, via the schema) `orion-thought`; no compose/env/Dockerfile changes, so no rebuild is strictly required by this patch's own contract, but both services should be restarted to pick up the new code before the live-verification step above is meaningful.

## Review findings fixed

- Finding: `_format_recent_actions`'s cap check (`if len(out) >= limit: break`) ran *after* appending the current entry, not before — so `max_recent_actions<=0` still let exactly one entry through instead of zero.
  - Fix: moved the limit check to the top of the loop body (before any work happens on the current item) and added an explicit `limit <= 0` guard at the top of the function.
  - Evidence: new regression test `test_recent_actions_respects_zero_or_negative_limit` asserts `build_autonomy_slice(ctx, max_recent_actions=0)` and `max_recent_actions=-1` both return `None` (no entries produced); 58/58 tests pass including this one.

Self-run code review (high effort, 8-angle process) surfaced no other material findings. Not currently reachable via the sole production call site (which always passes the real constant, `3`), but real, latent, and untested prior to the fix.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

`orion-thought` does not need a restart — it consumes `AutonomySliceV1` structurally (via `model_validate`), and the new field is optional/additive, so no code change is required there for the new field to pass through correctly once `orion-cortex-exec` starts populating it.

## Risks / concerns

- Severity: LOW
  Concern: live end-to-end effect on real chat turns not manually verified in this session (see Evals run above).
  Mitigation: full unit coverage proves every hop of the pipe individually; the one missing step is an observational live check, not a code-correctness gap. Recommend running it once post-deploy.

- Severity: LOW
  Concern: `_DEFAULT_MAX_RECENT_ACTIONS = 3` in `autonomy_slice.py` is a second literal `3`, kept in sync with `chat_stance.py`'s `_MAX_RECENT_DISPATCH_ACTIONS` only by comment, not by import (avoiding a circular import between the two modules, since `chat_stance.py` already imports from `autonomy_slice.py`). The real production call site always passes the constant explicitly, so this default is only reachable via test/future callers.
  Mitigation: documented in-line at the constant's definition; low cost to keep in sync if `_MAX_RECENT_DISPATCH_ACTIONS` ever changes, since it's a single grep away.

## PR link

Not opened via `gh` this cycle (leaving that to you, consistent with this session's established pattern of pushing branch + PR link rather than opening PRs unprompted for review-gated work). Paste-ready:

**Title:**
```
feat(cortex-exec): surface recent dispatch-action evidence into stance_react's pre-motor prompt
```

**Body:** this file's Summary through Risks/concerns sections, verbatim.

Compare URL: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/stance-react-dispatch-evidence
