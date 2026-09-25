## Summary

- **Memory consolidation compared against the wrong "previous turn".** When a turn's change-appraisal degraded, the retry job re-scored it against the 20th-oldest turn of the whole conversation instead of the turn right before it. It now uses the most recent 20, oldest first — the same shape the live path uses. (`services/orion-memory-consolidation/app/retry_degraded_classifies.py`)
- **Orion-mode chat never knew how long it had been since you last spoke.** Hub builds the unified turn's situation brief in its own process but only hooked one of the three Redis-backed situation stores up at startup, so every unified turn's prompt said "Conversation phase: unknown" and your turn was never recorded. A new shared helper (`orion/situational/state_buses.py`) binds all three; Hub and cortex-exec both call it.
- **The stance step lost the session.** orion-thought tucked `session_id` inside `metadata`, but cortex-exec reads it from the top level, so every unified turn recorded "Orion spoke" under a shared `"global"` bucket and emitted stance traces with no session. The stance context now carries it top level, like cortex-orch already does.
- **Only your own messages count as "you spoke".** Orion's unprompted outreach runs through the same unified turn in your live session; with the store now live, it would have stamped "Juniper just spoke" and made your real reply after a long absence read as a short pause. Turns now record the user timestamp only when `utterance_origin == "juniper"`; Orion-authored turns (outreach, curiosity, world-pulse) read the phase without moving it.
- **11 stale consolidation tests fixed.** They still expected the pre-2026-09-07 `metacog` classify route; no CI job runs that suite, so they rotted unnoticed. README row corrected too.
- **New CI gate** (`.github/workflows/session-scope-tests.yml`) runs these seams on every relevant change, so the store-binding gate and regression tests aren't just local.

## Outcome moved

- An Orion-mode turn after a gap now gets real framing ("Conversation phase: resumed_thread; continuity=lightly_resume", plus the "thread resumed, reorient" hint after long gaps) instead of "unknown" every time.
- Your turn and Orion's turn timestamps land on your conversation's own phase key instead of the shared `global` key.
- Degraded turn-change appraisals are re-scored against the turn that actually preceded them (proved against a real Postgres 16: old SQL picked `c19` for a 31st turn, new SQL picks `c29`).

## Current architecture

- `orion.situational.context.build_situation_for_ctx` reads three module-level Redis stores (conversation phase, Juniper's affect read, identity-ask cooldown), each needing a per-process `bind_*_bus` call. cortex-exec bound all three; Hub (which calls the builder on every unified turn via `orion/hub/turn_orchestrator.py::_build_situation_prompt_fragment`) bound only the affect store. Unbound reads fail open to "unknown" and skip the write.
- Unified turn order: stance RPC (orion-thought → cortex-exec `stance_react`, which ends in `router.py`'s `mark_orion_turn(ctx["session_id"] or "global")`) completes, then Hub builds the situation brief (records the user turn). Thought only put `session_id` under `context["metadata"]`.
- `execute_unified_turn` serves Hub chat (`utterance_origin="juniper"`, WS and HTTP) and Orion-authored turns: endogenous outreach (Juniper's live session, no origin), curiosity (`"orion"`), world-pulse reads, and Collapse Mirror replies (no origin).
- The consolidation retry loop fetched prior turns with `ORDER BY created_at ASC LIMIT 20`; `classify._prior_turn_baseline` reads `prior_turns[-1]`.

## Architecture touched

- orion-hub startup, orion-cortex-exec startup, orion-thought stance plan context, orion-memory-consolidation retry loop, shared `orion/situational`.
- No bus, schema, channel, or env contract changes.

## Files changed

- `orion/situational/state_buses.py`: new one-call binder for all situation stores.
- `orion/situational/tests/test_state_buses.py`: gate (every `bind_*_bus` in the package must be covered) + unified-turn sequence test + unbound-reads-unknown test.
- `services/orion-hub/scripts/main.py`: startup calls the helper instead of binding only the affect store.
- `services/orion-hub/tests/test_situation_state_buses_bound.py`: startup wiring check (source-level, same precedent as `test_memory_pg_pool.py`).
- `services/orion-cortex-exec/app/main.py`: three separate binds replaced by the helper (same bus, same behavior).
- `services/orion-cortex-exec/README.md`: phase-store paragraph updated (helper, moved module path, Hub/unified-turn writer).
- `services/orion-thought/app/bus_listener.py`: top-level `session_id` in the stance context.
- `services/orion-thought/tests/test_stance_context_session_id.py`: regression tests incl. cortex-exec's merge order.
- `services/orion-memory-consolidation/app/retry_degraded_classifies.py`: newest-20 prior turns, chronological.
- `services/orion-memory-consolidation/tests/test_retry_degraded_classifies.py`: regression test with an ORDER BY/LIMIT-honoring fake pool.
- `services/orion-memory-consolidation/tests/test_classify_turn_change.py`, `services/orion-memory-consolidation/README.md`: stale `metacog` expectations → `metacog_background`.
- `orion/situational/context.py`: `_records_user_turn(ctx)`; `_build_conversation_phase` writes the user timestamp only when it's true; read-only builds get their own situation-cache key.
- `orion/hub/turn_orchestrator.py`: `_build_situation_prompt_fragment(record_user_turn=False)` default; `execute_unified_turn` passes `utterance_origin == "juniper"`.
- `orion/situational/tests/test_conversation_phase_user_turn.py`: outreach-tick-then-reply regression, read-without-record, legacy default still records, cache-key separation, and an end-to-end run of the real `build_situation_for_ctx` cache (outreach brief cached, her reply inside the TTL still records; fails if the key split is removed).
- `services/orion-hub/tests/test_unified_turn_records_user_turn.py`: drives the real `execute_unified_turn` (cockpit-hops harness pattern) for `juniper` / `None` / `orion` origins.
- `orion/schemas/situation.py`: comment on what `last_orion_turn_at` means on unified turns (no field change).
- `orion/situational/tests/test_hub_settings_adapter.py`: stale comment naming the old bind.
- `.github/workflows/session-scope-tests.yml`: new path-triggered CI gate (slim deps, replayed in a clean venv).
- `config/metrics/metric_definitions.lock.json`: routine re-lock against the new merge base (no definition changes).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the stance_react plan context carries a top-level `session_id` (additive; `metadata.session_id` unchanged). Hub now reads the conversation-phase Redis key on every unified turn and writes the user timestamp only on Juniper-authored turns. Situation ctx gains an optional `record_user_turn` key (absent = true, so cortex-exec's legacy chat verbs are unchanged).
- Compatibility notes: none needed; all three services tolerate either side deploying first (fail-open reads, additive context key).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
# New/changed tests on the fixed code. Every new test was also run against the
# pre-fix file swapped back in and FAILS there (retry: c19 != c29; thought:
# KeyError 'session_id'; hub startup: 2 failed; binding gate with one store
# removed from the helper: "does not bind: ['orion.situational.identity_ask_cooldown']";
# outreach guard: 4/5 + 5/5 failed pre-fix; the end-to-end cache test also
# fails with the cache-key split removed).
orion/situational/tests                                          82 passed
services/orion-thought  tests/test_stance_context_session_id.py   3 passed
services/orion-memory-consolidation  tests (full suite)         119 passed  (11 failed on origin/main)
services/orion-hub  test_unified_turn_records_user_turn.py, test_situation_state_buses_bound.py,
                    test_unified_turn_surface_context.py, test_turn_orchestrator_cockpit_hops.py   32 passed

# Regression check: full suites vs a clean origin/main worktree, same venv,
# --continue-on-collection-errors, failure lists diffed.
orion-cortex-exec  954 passed / 124 failed / 14 errors   failure set IDENTICAL to origin/main
orion-hub         2261 passed / 111 failed / 63 errors   failure set IDENTICAL to origin/main (+7 new passes)
orion-thought      444 passed / 1 failed                 test_mind_enrichment_defaults_off also fails on origin/main
(The pre-existing failures are this container's environment -- missing spaCy etc. -- plus the one stale thought test.)

# New CI workflow replayed in a fresh venv with exactly its install commands:
session-scope-tests: 82 + 3 + 28 + 16 passed

# orion-static-gates.yml steps, run locally: all PASS
# (check_definition_drift needed the routine re-lock against the new merge base; 0 definition changes)
```

## Evals run

```text
No eval harness covers these seams (orion-memory-consolidation, orion-thought stance context, situation store binding have no evals/). Regression tests above prove each fix fails on the old code and passes on the new. Live behavior: UNVERIFIED (no access to the live bus/DB from this environment) -- see "Restart required" for the exact checks.
```

## Docker/build/smoke checks

```text
Docker daemon not available in this environment (docker CLI present, /var/run/docker.sock missing) -- no image build or container smoke was run.
Real-SQL check instead: local Postgres 16, 31-turn session + noise rows, old vs new _prior_turns_for:
  NEW  prior_turns: c10 ... c29 (n=20) baseline -> c29
  OLD  prior_turns: c00 ... c19 (n=20) baseline -> c19
```

## Review findings fixed

Review ran in a subagent (read-only, adversarial) against the first three commits.

- Finding (blocking): once Hub bound the phase store, Orion-authored unified turns in Juniper's live session (endogenous outreach, `endogenous_outreach.py:2541`) stamped `last_user_turn_at` -- reproduced: 6h absence, outreach tick, her reply 30 min later read `resumed_thread` instead of `long_gap`.
  - Fix: record only when `utterance_origin == "juniper"` (Hub chat, WS + HTTP); read-only builds use a separate situation-cache key so an outreach-built entry can't swallow her next real turn's record.
  - Evidence: `test_conversation_phase_user_turn.py` (4 of 5 fail on the pre-fix code; the one that passes is the legacy-default check) and `test_unified_turn_records_user_turn.py` (5 of 5 fail pre-fix) now pass.
- Finding (should-fix): the sequence test left `juniper_affect_state._BUS` / `identity_ask_cooldown._BUS` bound to a fake for the rest of the pytest process.
  - Fix: `unbound_stores` fixture monkeypatches every store's `_BUS` (restored after each test).
  - Evidence: `orion/situational/tests` 82 passed.
- Finding (should-fix): nothing automated ran the new binding gate.
  - Fix: `.github/workflows/session-scope-tests.yml`; docstring points at it.
  - Evidence: every workflow step replayed in a fresh venv with exactly the workflow's installs: 82 + 3 + 28 + 16 passed.
- Finding (nit): `last_orion_turn_at` on unified turns is stamped when the stance step ends, before the reply exists.
  - Fix: documented on the schema field; nothing reads it for bucketing (phase uses `last_user_turn_at` only).
  - Evidence: grep -- only the brief schema and `context.py` touch it.
- Finding (nit): vacuous `"global" not in store` assertion; the unbound test never checked its claim; stale bind comment in `test_hub_settings_adapter.py`.
  - Fix: assertion dropped (the Thought test covers the top-level session); unbound test now asserts the exact `session_turn_phase_read_bus_unbound` warning Hub has been logging; comment updated.
  - Evidence: tests pass.
- Finding (nit, not changed): session-less turns use `"anonymous"` in Hub and `"global"` in cortex-exec. Pre-existing, no live caller hits it on the unified path; left as is.
- Finding (nit): uncommitted README/comment edits at review time -- committed.
- Second review pass (on the fix commits): no blocking issues. Taken:
  - should-fix: CI didn't run the test pinning `utterance_origin="juniper"` on the WS path -- the thing that now makes her turns record. Added by node id (its sibling test also fails on origin/main).
  - nit: workflow triggers missed `orion/schemas/situation.py`, `requirements-dev.txt` and the Hub `*_client.py` modules the test imports -- added.
  - nit: the cache split was only tested structurally -- added the end-to-end `build_situation_for_ctx` test above.
  - nit: the workflow header cited this report before it was committed -- committed.
  - Not taken: passing `utterance_origin="juniper"` from Collapse Mirror replies (see Risks) -- it also changes the Mind's origin note, so it belongs in its own change.
- Reviewer also confirmed clean: the retry SQL fix and its test (also fails a DESC-without-reverse mutation); top-level `session_id` changes nothing in cortex-exec's stance path except the `mark_orion_turn` key plus traces now carrying the session (prompt unchanged, recall ignores session, situation/prior-stance only run for chat verbs); the stance-before-situation ordering; no env/schema/bus contract change.

## Restart required

All four images bake in `orion/` (`COPY orion`), so rebuild, from a worktree of main after merge:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

Live checks after the rebuild (send one Orion-mode message first):

```bash
# Hub stops warning that the phase store is unbound, and reads your session's key:
docker compose -f services/orion-hub/docker-compose.yml logs hub-app --since 10m | grep session_turn_phase
#   expect: session_turn_phase_read key=orion:cortex-exec:session_turn_phase:<your orion_sid> found=True
#   not:    session_turn_phase_read_bus_unbound
# cortex-exec's stance step writes to your session, not "global":
docker compose -f services/orion-cortex-exec/docker-compose.yml logs --since 10m | grep session_turn_phase_write
# Both timestamps on your conversation's key:
redis-cli -u "$ORION_BUS_URL" GET "orion:cortex-exec:session_turn_phase:<your orion_sid>"
```

## Risks / concerns

- Severity: low
  - Concern: Orion-mode prompts change. "Conversation phase: unknown; continuity=continue_directly" becomes real framing, including the "Reorient before acting on stale operational context" line and `temporal_resume` affordance after long gaps. This is the designed behavior (it already runs on the legacy chat path), but unified turns have never had it.
  - Mitigation: watch the first few replies after a long absence; revert is a single-commit revert of the Hub bind.
- Severity: low
  - Concern: the situation brief is cached for 5 minutes per session, and the phase is frozen with it. Right after a long gap, follow-ups within 5 minutes keep the "resumed/long_gap" framing, and quick back-and-forth mostly reads as `short_pause` rather than `same_breath`. Pre-existing on the legacy path; now visible on unified turns too.
  - Mitigation: follow-up task queued ("Stop situation cache freezing conversation phase").
- Severity: low
  - Concern: Collapse Mirror replies don't record a user turn (they pass no `utterance_origin`), though they're Juniper-authored. Same as before this PR (Hub never recorded anything).
  - Mitigation: set `utterance_origin="juniper"` there in a separate change if wanted -- it also changes the Mind's origin note, so it's not folded in here.
- Severity: low
  - Concern: unified turns still don't copy the phase into `spark_meta`, so memory consolidation's episode boundaries see "unknown" for them.
  - Mitigation: follow-up task queued (proposal first -- it changes episode grouping).
- Severity: info
  - Concern: live behavior is UNVERIFIED from this environment (no bus/DB/Docker access).
  - Mitigation: exact log/Redis checks under "Restart required".

## PR link

PR_LINK_PLACEHOLDER
