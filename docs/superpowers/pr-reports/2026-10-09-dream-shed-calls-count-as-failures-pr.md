# fix(dream): a shed model call is a failed call, not an unreadable answer

## Summary

- When the GPU pool turns a dream's model call away (for example, the cabinet is running hot), the dream now records that as a **failed call** instead of "the model said something unreadable."
- A sleep in which every call was turned away is now stored as **failed**, not **completed**. Its leftover items stay in the next sleep's replay window instead of being skipped forever.
- The dream service keeps its bus connection when the gateway refuses on purpose. It only reconnects for real transport errors.
- README: the story dream (`dreams` table) was described as "nightly", but nothing schedules it. Corrected.

## Outcome moved

Live, 2026-10-08 at 06:27 and 18:27 UTC: every recombination call was refused with `shed:cabinet_hot` (`gpu_pool_events`). Both sleeps were stored `completed` with `unparseable_count=4` and 0 hypotheses. Because the replay window starts at the last non-failed sleep (`cycle_store.py:69`), the next sleep started after them, and those two windows' backlog was never looked at by a model. That is 2 of the last 8 sleeps. The design already intended "a gateway outage does not throw the backlog away" (`cycle_store.py:63-66`). This bug defeated it.

A replay of the failed cycle `dc-f4b079db88f7`'s exact pairs through the live gateway (2026-10-09 ~02:00 UTC) parsed 4/4. The prompt and parser were never the problem.

## Current architecture

`app/llm.py complete()` returned `payload.content` as-is. The gateway's refusal shape (`orion-llm-gateway/app/main.py _pool_unavailable_result`, `_revoked_result`) is empty text plus `raw.error`. `recombine()` counted `""` as unparseable, and `cycle.py` marks `failed` only when every call *raised*.

## Architecture touched

`services/orion-dream` only. No bus, schema, or env change.

## Files changed

- `services/orion-dream/app/llm.py`: `GatewayRefused`, raised on `raw.error`, an `[Error:` text, or blank text.
- `services/orion-dream/app/main.py`: do not drop the bus on `GatewayRefused`.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: two regression tests using the gateway's real shed reply.
- `services/orion-dream/README.md`: story dream is hand-started; shed calls are failures.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: an all-shed sleep is `failed` (counted in `llm_failures`), not `completed`/`unparseable`.
- Compatibility notes: none.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-dream/tests -q
111 passed

Against the pre-fix llm.py/main.py (same tests):
FAILED test_gateway_error_reply_raises_instead_of_returning_empty_text
FAILED test_shed_sleep_is_failed_not_completed_so_the_window_does_not_advance
  AssertionError: assert 'completed' == 'failed'
```

## Evals run

```text
None. orion-dream has no eval harness for recombination quality. The arm scorecard
(scripts/dream_hypothesis_scorecard.py) is the closest thing. This fix changes failure
accounting, not hypothesis quality.
```

## Docker/build/smoke checks

```text
Live replay probe of dc-f4b079db88f7 inside orion-athena-dream: 4/4 replies parsed.
Post-deploy proof: UNVERIFIED until the next heat-shed sleep. Expect a dream_cycle row with
status='failed', llm_failures>0, unparseable_count=0.
```

## Review findings fixed

- Finding (medium): some gateway errors come back as `[Error: ...]` text with an empty `raw` (backend URL unset, `llm_backend.py:935/1018/1172`). These would still be read as unparseable, and the sleep stored `completed`.
  - Fix: `complete()` also raises on a `[Error:` prefix, the same check the gateway's own `_result_error` uses (`main.py:263`).
  - Evidence: new test case `match="upstream_error"`. 111 passed.
- Finding (low): nothing tested the line that keeps the bus connection on a refusal.
  - Fix: `test_gateway_refusal_keeps_the_bus_but_a_transport_error_drops_it`.
  - Evidence: deleting the `except llm.GatewayRefused: raise` line makes that test fail (1 failed, 25 passed).
- Finding (nit): a test name claimed "window does not advance", but the test only checks status.
  - Fix: renamed to `test_all_shed_sleep_is_stored_failed_not_completed`. The docstring points at the window SQL it relies on.
- Finding (low, not changed): a partly-shed sleep (some calls fail, some succeed) is still `completed`, and the shed pairs' items are not retried. Listed under risks.

## Restart required

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: a partly-shed sleep is stored `completed`. Items from the shed pairs are not retried.
  - Mitigation: none here. Retrying per pair would need per-item replay state. Revisit if `llm_failures` between 1 and 3 shows up live.

- Severity: low
  - Concern: a failed sleep still waits `DREAM_MIN_INTERVAL_HOURS` (6 h) before retrying, so a hot cabinet delays the dream rather than dropping it.
  - Mitigation: intended (`cycle_store.py:67-68`). Items are kept.
- Severity: medium (separate, not fixed here)
  - Concern: the gateway maps `llm_lane: "background"` to the `metacog` route (system priority) and ignores the caller's route (`orion-llm-gateway/app/lane_routes.py:111-114`). `DREAM_LLM_ROUTE=metacog_background` is dead config, and dream calls lease at `system`, not `background`. This affects every background-lane caller.
  - Mitigation: follow-up. A gateway-wide priority change needs its own blast-radius check.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
