## Summary

- Root cause of "Orion asked how my evening was going at 12:45pm MDT" (2026-08-22): the live chat path (`orion/hub/turn_orchestrator.py::execute_unified_turn`) never called the existing situational-awareness builder at all. `services/orion-cortex-exec/app/situation.py` computed correct time-of-day/day-phase/conversation-phase/presence context, but was only ever wired into cortex-exec's own legacy chat-verb dispatch lane -- a different code path from the one real chat turns take (Hub -> `HarnessGovernorClient` -> harness-governor's FCC motor -> `compile_harness_prefix`). That prompt compiler had zero time/date content of any kind.
- Relocated `situation.py`/`perception_reader.py`/`session_turn_phase.py` from cortex-exec's private `app/` into a shared `orion/situational/` package (AGENTS.md §5: cross-service seams must be `orion/`-package or documented contracts, not reaching into another service's internals).
- Added `HarnessRunRequestV1.situation_prompt_fragment`, threaded Hub -> harness-governor -> `compile_harness_prefix`, same treatment as the existing `current_served_model` self-context line: omitted entirely when `None`, so a turn with no situation context renders byte-identical to before.
- Added `hub_settings_to_runtime_namespace()` to bridge orion-hub's UPPERCASE `Settings` attrs into the lowercase shape the existing `settings_from_runtime()` expects -- avoids a silent case-sensitive `getattr` miss that would have fallen back to hardcoded defaults with no visible error.
- Fixed a real event-loop-blocking bug found while moving this code: the runtime self-model probe's blocking `urlopen` now runs via `asyncio.to_thread` -- harmless on cortex-exec's per-turn dispatch, but a real stall risk once awaited from Hub's single shared WebSocket event loop.
- Also found and fixed: the unified-turn path never called `inject_session_presence`, so a presence set via `/api/presence` was invisible to every Orion-mode turn (both the WebSocket and HTTP call sites) -- confirmed by a review pass, not just guessed.

## Outcome moved

Real local time-of-day, day-phase, conversation-phase (same-breath/short-pause/long-gap/stale-thread), and presence context now reach the actual prompt behind live chat. Previously: none of it did, for any unified-turn (`mode="orion"`) conversation, ever.

## Current architecture

`services/orion-cortex-exec/app/situation.py` built a full `SituationBriefV1`/`SituationPromptFragmentV1` and was consumed only by `executor.py`'s `chat_general`/`chat_quick`/`chat_kids_story` verb-plan lane -- a separate, apparently-legacy agentic dispatch path through cortex-exec's own `router.py`. The actual production chat path (`orion_journal` sessions, `chat_route: "unified_turn_harness"`) is `orion/hub/turn_orchestrator.py::execute_unified_turn`, which hands off directly to the harness-governor's FCC motor and never touched cortex-exec's situation module at all. Separately, `services/orion-hub/scripts/api_routes.py` already had a *third*, independent, UI-facing re-derivation of the same time-of-day bucketing logic behind `/api/situation/status`/`/api/situation/brief` -- also not wired to the chat prompt. That third copy is untouched by this patch (out of scope; flagged as a follow-up below).

## Architecture touched

- `orion/situational/` (new shared package): `context.py` (formerly `situation.py`), `perception_reader.py`, `session_turn_phase.py`.
- `orion/schemas/harness_finalize.py`: `HarnessRunRequestV1` schema.
- `orion/harness/prefix.py`, `orion/harness/runner.py`: prompt compilation.
- `orion/hub/turn_orchestrator.py`: unified-turn orchestration.
- `services/orion-cortex-exec/app/{main,executor,router}.py`: import-path updates only, no behavior change to that service's own lane.

## Files changed

- `orion/situational/context.py` (renamed from `services/orion-cortex-exec/app/situation.py`): relocation; `_build_runtime_context` made async, its blocking `urlopen` now runs via `asyncio.to_thread`; added `hub_settings_to_runtime_namespace()`.
- `orion/situational/perception_reader.py`, `orion/situational/session_turn_phase.py`: pure relocation, no logic change.
- `orion/situational/__init__.py`, `orion/situational/tests/__init__.py`, `orion/situational/tests/test_hub_settings_adapter.py`: new package + adapter tests.
- `orion/schemas/harness_finalize.py`: added `HarnessRunRequestV1.situation_prompt_fragment: str | None`.
- `orion/harness/prefix.py`: `compile_harness_prefix` gains `situation_prompt_fragment` param, rendered as one more prefix block.
- `orion/harness/runner.py`: `build_harness_prompt` and `HarnessRunner.run()` thread the field through.
- `orion/hub/turn_orchestrator.py`: new `_build_situation_prompt_fragment()` helper (calls `build_situation_for_ctx`, merges stored presence via `inject_session_presence`, fail-open); wired into `execute_unified_turn` and `HarnessRunRequestV1` construction.
- `orion/schemas/situation.py`, `services/orion-thought/app/vision_reader.py`: stale doc-comment path fixes (caught by review).
- `services/orion-cortex-exec/app/{main,executor,router}.py`: import path updates (`.situation`/`.session_turn_phase` -> `orion.situational.*`).
- `scripts/smoke_situation_grounding.py`: import path updates.
- `services/orion-cortex-exec/tests/test_{situation_conversation_phase,situation_perception_context,situation_provider,session_turn_phase}.py`: import path updates; one test converted to `async`/`await` for the `_build_runtime_context` signature change.
- `orion/harness/tests/test_harness_{prefix,runner}.py`: new tests for the situation-fragment rendering/threading, including explicit "byte-identical when absent" regression guards.
- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`: new tests for `execute_unified_turn`'s situation-context wiring and fail-open behavior.

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.situation_prompt_fragment: str | None = None` (additive, optional, backward compatible; already registered via `orion/schemas/registry.py`'s existing `HarnessRunRequestV1` entry, no version bump needed).
- Removed: none.
- Renamed: none (deliberately did NOT rename orion-hub's existing `ORION_SITUATION_*`/`ORION_PRESENCE_*` `Settings` attrs, which are already live behind `/api/situation/*` -- see `hub_settings_to_runtime_namespace()`'s docstring for why).
- Behavior changed: unified-turn chat prompts now include a "Situation:" block (local time/day-phase/conversation-phase/presence) when situation context resolves; omitted entirely on failure or when disabled -- same as every other optional prefix block in `compile_harness_prefix`.
- Compatibility notes: `services/orion-cortex-exec`'s own `chat_general`/`chat_quick`/`chat_kids_story` verb lane is unaffected -- same behavior, new import path only.

## Env/config changes

- Added keys: none. `hub_settings_to_runtime_namespace()` reuses orion-hub's already-configured `ORION_SITUATION_ENABLED`/`ORION_SITUATION_TTL_SECONDS`/`ORION_SITUATION_TIMEZONE`/`ORION_PRESENCE_DEFAULT_REQUESTOR`/`ORION_PRESENCE_PERSIST_ALLOWED`/`HUB_LLM_GATEWAY_URL` -- all already present in `services/orion-hub/.env`/`.env_example` from an earlier, never-finished wiring attempt.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no (nothing added).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran; confirmed no changes needed (only pre-existing, unrelated "diverged" entries reported, none touched by this patch).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-cortex-exec/tests/test_{situation_conversation_phase,situation_perception_context,situation_provider,session_turn_phase}.py -q
  73 passed

orion/situational/tests/test_hub_settings_adapter.py
orion/harness/tests/test_harness_prefix.py
orion/harness/tests/test_harness_runner.py
services/orion-hub/tests/test_turn_orchestrator_ws_frames.py
services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py -q
  93 passed, 1 failed (test_harness_runner_surfaces_fcc_error_code:
  verified byte-identical failure on main HEAD, unrelated string-format
  assertion, pre-existing)

services/orion-cortex-exec/tests -q (full suite, minus 14 files that
already fail to COLLECT on main HEAD with "ValueError: Verb already
registered: legacy.plan" -- a pre-existing global verb-registry bug)
  111 failed, 726 passed on this branch vs 107 failed, 730 passed on
  main HEAD under the identical command. Diffed both failure sets
  directly: re-running with the situation-related test files excluded
  from BOTH trees converges to the exact same 107-failure set, confirmed
  via `comm` set-diff. The remaining +4/-1 delta is deterministic
  test-collection-order sensitivity from a pre-existing cross-test
  global-state hygiene issue in this suite (same class of bug already
  evidenced by the 14 excluded "Verb already registered" collection
  errors) -- none of the affected tests (test_chat_reply_logprobs_gate.py,
  test_chat_stance_brief.py, test_endogenous_runtime_phase8.py) reference
  situation/session_turn_phase/perception_reader/turn_orchestrator in any
  way, and each passes cleanly in isolation on this branch.
```

## Evals run

No dedicated eval harness exists for `orion-cortex-exec`'s situation module or `orion-hub`'s turn orchestrator; this PR adds regression tests in the gate-test lane instead (see above).

## Docker/build/smoke checks

Not run -- this patch is Python import/schema/prompt-compiler wiring only, no Dockerfile/compose/dependency changes (`orion.situational.perception_reader`'s `sqlalchemy` import was already a proven-live dependency in `orion-hub` via `orion.schemas.situation`'s pre-existing use in `api_routes.py`; `orion-hub/requirements.txt` already has `SQLAlchemy==2.0.43`). Restart required for the fix to take effect in production (see below).

## Review findings fixed

- Finding: an early docstring draft in `orion/hub/turn_orchestrator.py` contained the literal substring `chat_general`, tripping `test_unified_orion_turn_pollution_firewall` (a real guard against Brain-lane vocabulary leaking into the unified-turn path).
  - Fix: reworded to describe the same thing without the literal forbidden token.
  - Evidence: `test_unified_orion_turn_pollution_firewall` passes.
- Finding: two doc comments elsewhere in the repo (`orion/schemas/situation.py`, `services/orion-thought/app/vision_reader.py`) still pointed at the old `services/orion-cortex-exec/app/{situation,perception_reader}.py` paths after the relocation.
  - Fix: updated both to the new `orion/situational/` paths; noted the relocation as a genuine future-consolidation opportunity for `orion-thought`'s own separate perception reader, without changing that service's actual imports (separate call, out of scope here).
  - Evidence: `git diff` on both files; both still parse (`ast.parse`).
- Everything else in the review (adapter attribute-name correctness, presence-merge signature/behavior, async conversion correctness, end-to-end schema wiring, Docker packaging, test meaningfulness) verified correct with no changes needed -- see the review's live test runs cited above.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d --build
```

(Use `scripts/safe_docker_build.sh` from a worktree, not a bare `docker compose`, per repo convention. Juniper should run these -- I did not run them myself.)

## Risks / concerns

- Severity: low
  Concern: the runtime self-model probe (`_build_runtime_context`) now runs on Hub's own event loop via `asyncio.to_thread` on every situation-cache-miss unified turn (TTL 120s), adding up to `runtime_probe_timeout_sec` (2.0s) of thread-pool latency on a cold cache.
  Mitigation: bounded by an existing timeout, cached for 120s, and offloaded to a thread specifically so it cannot block other concurrent WebSocket clients. Fail-open on any error.

- Severity: low
  Concern: `services/orion-hub/scripts/api_routes.py`'s `/api/situation/status`/`/api/situation/brief` routes contain a third, independent re-derivation of the same time-of-day bucketing logic, still un-wired to the chat prompt and now redundant with the shared `orion/situational/context.py` module.
  Mitigation: deliberately left untouched -- it's a different feature (a UI status widget, not the chat path) and consolidating it wasn't part of the approved design. Flagged here as a follow-up, not silently left as a surprise.

- Severity: none (informational)
  Concern: `_build_situation_prompt_fragment` runs strictly serially in `execute_unified_turn` (after the Thought stance RPC, before harness dispatch) rather than concurrently with it.
  Mitigation: not fixed -- current placement means a deferred/refused turn skips building it entirely, which is the better default; the serial cost is small and cache-bounded.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1845

🤖 Generated with [Claude Code](https://claude.com/claude-code)
