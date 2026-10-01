# fix(gpu-pool): report a blocked swap once per episode, not every tick (stage 6.1)

## Summary

- When the gpu2 seat's unload was blocked by a cooldown, the pool wrote a `swap_requested` event on every 1 s scheduler tick. It should write one for the whole block.
- Root cause: the pool changes an unload it may not send yet into a "blocked by cooldown" report, and it remembered that it had reported it. But the tick loop then kept only the *original* unload in its memory and threw the blocked one away. So on the next tick it looked new, and was reported again. This happened every tick for the whole cooldown.
- Fix: `_swap` now returns the key it actually reported under, and the tick keeps that key. The key is seat + action + the reason reported + the scheduler's own reason. It never includes guard state text or anything else that changes from tick to tick.
- Second bug on main, fixed here: every storm row had `detail.action: "load"` even though the blocked action was an unload. The action was read after the decision had been rewritten. So any past rows, or any reader grouping on `detail.action`, got this wrong. Now it says `"unload"`.
- The key is now saved only after the event is built and sent. So if building the event raises, the pool tries again on the next tick. Publish errors are already swallowed further down, so they cannot cause a retry storm.
- Spec: `docs/superpowers/specs/2026-09-30-gpu-pool-stage6-telemetry-reducers-lockdown.md` (PR #2439), item 6.1.

## Outcome moved

- The event history and the grammar table stop filling with copies of one fact. Live before: 628 rows from 08:45 to 08:55 UTC on 2026-09-30, plus 1,320 on 09-26 and 849 on 09-28. After: one row per blocked episode.
- A new reason still gets reported. So does the block clearing and then coming back. The actuator events (`swap_started`, `swapped`, `swap_failed`) never go through this dedupe.

## Current architecture

- `PoolRuntime.tick()` (`services/orion-gpu-pool/app/runtime.py`) runs the scheduler and gets a list of decisions. For each swap decision it added `_swap_key(d)` to a keep-set `swaps` and called `_swap(d)`. After the loop, `self._swap_requested &= swaps` drops any episode that did not come up again this tick.
- `_swap(d)` handles an actuated seat that is still in cooldown by replacing `d` with `SwapBlocked(role, "cooldown", ...)`. It then checks and adds the key of that *new* decision. The scheduler emits its own `SwapBlocked` for blocked loads, and those keys matched. The unload rewrite path never matched. That path is the live storm: `detail.guard_state="unload"` on every storm row, and it came right after `swap_failed upstream_not_idle:agent-gpu2`.

## Architecture touched

- orion-gpu-pool runtime only: how swap reports are deduped. No contract, channel, schema or config change.

## Files changed

- `services/orion-gpu-pool/app/runtime.py`: `_swap` returns the episode key it reported under. `_swap_key(role, action, reported_reason, wanted_reason)`. `tick()` keeps the returned key. The dedupe key is saved after emit. `detail.action` is taken before the rewrite.
- `services/orion-gpu-pool/tests/test_holds_and_actuation.py`: regression tests.
- `services/orion-gpu-pool/README.md`: documents the once-per-episode rule.
- `docs/superpowers/pr-reports/2026-09-30-gpu-pool-stage6-1-swap-requested-once-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `swap_requested` for blocked swaps is sent once per episode (seat + action + reason). For cooldown-blocked unloads, `detail.action` is now `"unload"` (it was wrongly `"load"`).
- Compatibility notes: the event shape is unchanged. No consumer reads `swap_requested` per tick. The Hub panel and the eval count it.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-gpu-pool && PYTHONPATH=<worktree> python -m pytest tests -q -p no:cacheprovider
  126 passed, 9 skipped
PYTHONPATH=<worktree> python -m pytest orion/gpu_pool/tests -q -p no:cacheprovider
  326 passed
New: test_cooldown_blocked_unload_is_reported_once_per_episode_not_every_tick
  - 60 one-second ticks inside a cooldown after a refused unload -> exactly 1 swap_requested and 1 grammar atom
  - pause (reason -> actuation_paused) -> 1 new event; resume while still cooling -> 1 new cooldown event
  - cooldown lapses -> unload really sent (swap_started +1), refused again -> 1 new cooldown event
  FAILS on origin/main (many events, labelled action=load); passes on this branch.
New: test_a_paused_swap_whose_wanted_reason_changes_is_a_new_episode
  - paused: idle unload x2 -> 1 event; then max_hold unload -> a 2nd event. FAILS on origin/main.
New: test_scheduler_blocked_load_is_still_reported_once_per_episode
  - refused load, 60 ticks -> exactly [("cooldown", "load")]. Guards the scheduler-side path; passes on main too.
Both "FAILS on main" claims were checked by running the branch tests against origin/main's runtime.py.
```

## Evals run

```text
PYTHONPATH=<worktree> python services/orion-gpu-pool/evals/run_pool_day_eval.py
  VERDICT: PASS
```

## Docker/build/smoke checks

```text
Not run. Juniper asked for no deploy. The runtime change is covered by the runtime tests above,
which use the real scheduler, the real lease graph and a fake bus.
```

## Review findings fixed

The code review ran in a subagent. It found no must-fix issues, and it independently confirmed that the regression test fails on main (60 events in 60 ticks).

- Finding (should): the key did not really follow "seat + action + reason". For paused and non-actuated swaps, the reason part of the key was `None`, so a change in the scheduler's reason while paused was not treated as a new episode, even though the README now promises it would be.
  - Fix: the key is now `(role, action, reported_reason, wanted_reason)` and is worked out after the reported reason is known.
  - Evidence: `test_a_paused_swap_whose_wanted_reason_changes_is_a_new_episode` passes on the branch and fails on main.
- Finding (nit): the comment "a failed publish is retried next tick" was false, because publish errors are swallowed.
  - Fix: the comment now says "an emit that raises is retried next tick (publish errors are swallowed inside)".
  - Evidence: `runtime.py`, in `_swap`.
- Finding (nit): the test name "...and again on a new reason" promised more than the test checks.
  - Fix: renamed it to `test_scheduler_blocked_load_is_still_reported_once_per_episode`.
  - Evidence: the test file.
- Finding (nit): the report should name the `detail.action` mislabel.
  - Fix: it is named in the Summary.
  - Evidence: this report.

## Restart required

Deploy only after #2438 (pool boot schema self-heal) is live.

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

Live acceptance: after the next cooldown-blocked swap, this count should grow by only a few rows per
episode, not about 600:
`select count(*) from gpu_pool_events where event='swap_requested' and reason='cooldown' and created_at > '<deploy ts>'`.

## Risks / concerns

- Severity: low
- Concern: a blocked episode that lasts a long time now shows as one row, so "how long was it blocked" can no longer be read from the row count.
- Mitigation: the row count was never a reliable duration anyway (its spacing depended on the tick). The next `swap_started` or `swapped` marks when the block ended.
- Live verification: UNVERIFIED until deployed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2441

🤖 Generated with [Claude Code](https://claude.com/claude-code)
