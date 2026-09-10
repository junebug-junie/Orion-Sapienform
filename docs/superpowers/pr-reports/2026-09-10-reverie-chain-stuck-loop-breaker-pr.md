## Summary

- The text reverie chain (`services/orion-thought/app/chain.py`) never armed its 15-minute refractory cooldown when a chain's first step came back hollow (`terminal_reason="no_coalition"`) — so a theme that can't ground a first thought re-fires on the very next tick, indefinitely.
- Added `resolve_reverie_chain_stuck_loop`, mirroring `visual_chain.py`'s existing `resolve_visual_chain_continuity` (the "still doing the same images of Roman aqueducts" fix): a deterministic streak counter that forces the cooldown once a theme repeats `no_coalition` `ORION_REVERIE_NO_COALITION_MAX_REPEATS` (default 3) times in a row.
- The streak is persisted in the chain's own `chain_json.no_coalition_streak` (new additive field on `ReverieChainV1`) and read back per-theme via a new `store.load_latest_no_coalition_streak`, the same technique the visual chain already uses for `continuity_streak` — no new table, no migration.
- New settings key `ORION_REVERIE_NO_COALITION_MAX_REPEATS` (default 3), `.env_example` updated, local `.env` synced.

## Outcome moved

A theme stuck unable to ground a first thought now gets suppressed for `refractory_sec` after 3 consecutive failures, instead of re-triggering every ~90s forever. This directly addresses the live escalation for `open-loop-7376a3da4050`: 47 of its 63 logged resonance violations (13 days / 678 chains) directly followed a `no_coalition` row with nothing ever breaking the cycle.

## Current architecture

`run_reverie_chain` (chain.py) arms a refractory cooldown only for `terminal_reason` in `("pressure_discharged", "max_steps", "low_salience")`. `"no_coalition"` — which fires whenever the very first step of a chain returns no thought, for any reason (hollow narration, exception, timeout, low salience, semantic-gate skip) — was excluded, so a chain landing there left the theme fully un-suppressed.

## Architecture touched

- `orion/schemas/reverie.py` — additive field `ReverieChainV1.no_coalition_streak: int = 0`.
- `services/orion-thought/app/chain.py` — new pure function + wiring in `run_reverie_chain`/`run_reverie_chain_worker`.
- `services/orion-thought/app/store.py` — new read `load_latest_no_coalition_streak(theme_key)`.
- `services/orion-thought/app/settings.py` + `.env_example` — new `ORION_REVERIE_NO_COALITION_MAX_REPEATS`.

## Files changed

- `orion/schemas/reverie.py`: additive `no_coalition_streak` field on `ReverieChainV1`, persisted inside the existing `chain_json` JSONB column (no migration).
- `services/orion-thought/app/chain.py`: `resolve_reverie_chain_stuck_loop` (pure, mirrors `visual_chain.py::resolve_visual_chain_continuity`), wired into `run_reverie_chain`'s refractory-arming decision and `run_reverie_chain_worker`'s call site.
- `services/orion-thought/app/store.py`: `load_latest_no_coalition_streak` — best-effort read of the latest chain row's own streak for a theme, degrades to 0 on any error/missing data (same direction as the visual chain's sibling reader).
- `services/orion-thought/.env_example`: `ORION_REVERIE_NO_COALITION_MAX_REPEATS=3`.
- `services/orion-thought/app/settings.py`: matching `ThoughtSettings` field.
- `services/orion-thought/tests/test_reverie_chain.py`: pure-function tests (below/at threshold, non-`no_coalition` reset) + wiring tests (streak carried through `run_reverie_chain`, forced suppression at threshold, loader-failure degrades to 0 not a raise).
- `services/orion-thought/tests/test_store.py`: `load_latest_no_coalition_streak` tests (reads value, missing key, non-dict `chain_json`, no rows, DB failure) mirroring the existing `load_latest_visual_chain_continuity_state` test block; extended the shared `_connect_result_engine` test fake to accept a params argument (this function's query is parameterized by `theme_key`, unlike its no-params sibling).

## Schema / bus / API changes

- Added: `ReverieChainV1.no_coalition_streak: int = 0` (additive, default 0, no consumer-first migration needed — read via `chain_json.get(...)`, never strict-revalidated elsewhere; confirmed by reading every `ReverieChainV1` consumer, including orion-hub's `reverie_routes.py`, which already reads `chain_json` as a raw dict).
- Removed: none.
- Renamed: none.
- Behavior changed: a theme whose chain ends `"no_coalition"` 3 times running now gets the refractory cooldown forced; below that, behavior is unchanged (matches the old test `test_chain_step_none_terminates_without_raise`, still passing).
- Compatibility notes: old chain rows with no `no_coalition_streak` key read back as streak 0 (honest "nothing recorded yet").

## Env/config changes

- Added keys: `ORION_REVERIE_NO_COALITION_MAX_REPEATS=3` (orion-thought).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (ran from primary checkout via the venv interpreter, then mirrored into this worktree's own `.env` per the known "sync writes to primary checkout, not worktree" gap).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=<worktree> .venv/bin/python -m pytest services/orion-thought/tests/test_reverie_chain.py services/orion-thought/tests/test_store.py services/orion-thought/tests/test_visual_chain.py -q
124 passed

PYTHONPATH=<worktree> .venv/bin/python -m pytest services/orion-thought/tests -q
379 passed, 3 failed

  Failed (pre-existing, unrelated to this patch -- verified they fail the same
  way with the offending vars unset, i.e. driven by this host's local .env
  content and an unreachable DB hostname outside the docker network, not by
  code):
  - test_reverie_spontaneous_thought.py::test_tick_publishes_grounded_thought
    (DB hostname "-sql-db" unresolvable outside container network; local .env
    already has ORION_ATTENTION_SALIENCE_V2_ENABLED=true, adding a 2nd publish
    call the test doesn't expect)
  - test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off
    (local .env already sets a non-default ORION_MIND_BASE_URL)
  - test_settings_salience_flags.py::test_salience_flags_default_off
    (local .env already sets ORION_ATTENTION_SALIENCE_V2_ENABLED=true)

PYTHONPATH=<worktree> .venv/bin/python -m pytest orion/reverie/tests/test_ouroboros_invariants.py \
  tests/test_attention_schema_surface.py services/orion-thought/tests/test_reverie_compaction_request.py -q
33 passed
  (every other file importing ReverieChainV1, to check blast radius of the new field)
```

## Evals run

No eval harness exists for this seam (services/orion-thought/evals is scoped to visual-chain/narration quality, not refractory timing). This is deterministic control logic (pure function + DB read), fully covered by the gate tests above.

## Docker/build/smoke checks

Not run — no runtime behavior change requiring a live container to observe (default-off flag behavior unchanged; new setting has a safe default). `scripts/check_env_key_single_source.py`, `scripts/check_env_template_parity.py`, and `scripts/check_service_env_compose_parity.py orion-thought` all pass (orion-thought uses `env_file:` passthrough, so no compose key wiring was needed).

## Review findings fixed

Code-review skill (medium effort) ran clean: no correctness bugs, no dangerous removed behavior, no cross-file breakage, no CLAUDE.md violations. No findings to fix.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml \
  up -d --build orion-thought
```

## Risks / concerns

- Severity: low
- Concern: `ORION_REVERIE_NO_COALITION_MAX_REPEATS=3` is a guessed default, not calibrated against live data (mirrors the visual chain's own `visual_chain_continuity_max_runs` pattern, but that value isn't necessarily right for this different failure mode).
- Mitigation: it's a plain env var — adjust after watching a few real streak-forced suppressions in the Hub Reverie tab / `substrate_reverie_chain.chain_json.no_coalition_streak`.

- Severity: low (separate, not fixed here)
- Concern: `terminal_reason` is a second, adjacent mislabeling I found live-debugging this (2026-09-10): a hollow-drop at chain step index ≥1 gets recorded as `"pressure_discharged"` even when pressure was never read (e.g. `services/orion-thought` container logs, chain `8a44f905...`, 2026-09-10 00:06:23 — logged a hollow drop `reason=zero_grounding` immediately before `terminal=pressure_discharged`). Doesn't cause the resonance bug (that terminal already arms the cooldown) but it does mean some `pressure_discharged` rows in the Hub are misleading about why the chain actually ended. Left out of this patch to keep it thin — flagging as a follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2182

🤖 Generated with [Claude Code](https://claude.com/claude-code)
