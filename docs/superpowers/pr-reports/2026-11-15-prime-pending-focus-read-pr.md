## Summary

Fixes #3366: Prime's questions to curiosity were refused because Prime asked a slow page a yes/no question with a short timeout.

- Prime decides whether it may ask curiosity a question by checking whether a question is already waiting (the "pending focus"). It read that answer out of Hub's whole curiosity dashboard, which takes 4-44 s, with a 4 s timeout. Every read on 11-14 timed out, so the move was refused and the CURIOSITY part of Prime's frame was empty.
- Hub gets a cheap endpoint, `GET /curiosity/api/focus/pending`, that returns only the pending focus (one row read).
- Prime's move check reads that endpoint (4 s timeout). The dashboard read, still used for the frame's CURIOSITY section and for settling questions, gets its own 30 s timeout (`PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC`).
- Never fail open: a timed-out, failed, malformed or `available: false` pending-focus read still refuses the move. A timeout now says "timed out", and the frame names the gap.
- The frame shows a waiting focus from the cheap read, so it shows even when the dashboard is down.

## Outcome moved

Prime's only outward act (ask curiosity) is no longer blocked by dashboard latency. Measured: the dashboard route took 3.70-5.82 s from inside orion-mind (2 of 3 over Prime's 4 s); the new route answered in 43-111 ms in-process against the live database, and its query runs in 0.4-0.6 ms.

## Current architecture

`services/orion-mind/app/prime_loop.py` `_ask` read `GET {PRIME_HUB_URL}/curiosity/api/runs?days=4&line=investigate` with `HTTP_TIMEOUT_SEC = 4.0`. `_conclude` took `schedule.pending_focus` from that answer; a failed read gave `focus_pending=None`, and `orion/prime/thought.py` `apply_answer` refused the move.

## Architecture touched

- orion-hub: one new GET route on the curiosity router.
- orion-mind (Prime): a second read per thought; a separate timeout for the dashboard read.
- Shared: `orion/curiosity/focus.py` `PENDING_FOCUS_PATH` (both sides import it).

## Files changed

- `orion/curiosity/focus.py`: `PENDING_FOCUS_PATH` constant and its payload contract.
- `services/orion-hub/scripts/curiosity_routes.py`: `GET /curiosity/api/focus/pending`.
- `services/orion-mind/app/prime_loop.py`: `_read_pending_focus` (cheap read, short timeout, never fails open); `_read_curiosity` uses `get_json_long` (30 s); `_conclude` uses the cheap read.
- `orion/prime/thought.py`: `FrameInputs.pending_focus*`; the frame shows the cheap read and its gap; `apply_answer(focus_unread=...)` carries the refusal sentence.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC`.
- `services/orion-hub/README.md`: the new route.
- `services/orion-mind/tests/test_prime_loop.py`: regression tests; the fake Hub serves the new route.
- `services/orion-hub/tests/test_curiosity_focus.py`: route test.
- `orion/prime/evals/run_prime_frame_eval.py`: `focus_read_split` check.

## Schema / bus / API changes

- Added: `GET /curiosity/api/focus/pending` (Hub). Body `{"available": bool, "focus": <focus payload> | null, "enabled": bool | null}`, the same payload the dashboard's `schedule.pending_focus` already carries.
- Removed: none. Renamed: none. No bus channel, no schema registry entry.
- Behavior changed: Prime's move check reads the new route instead of the dashboard.
- Compatibility notes: a Hub without the route answers 405 (the `DELETE /api/focus/{focus_id}` path matches). Prime treats that as unreadable and refuses the move, which is no worse than today. Deploy Hub first.

## Env/config changes

- Added keys: `PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC=30` (orion-mind).
- Removed / renamed keys: none.
- `.env_example` updated: yes (`services/orion-mind/.env_example`), plus `settings.py`, `docker-compose.yml`, README table.
- local `.env`: added by hand to `/mnt/scripts/Orion-Sapienform/services/orion-mind/.env` after `PRIME_HUB_URL` (a comment line plus `PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC=30`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-mind: pytest tests -> 396 passed
pytest orion/prime -> 244 passed
pytest orion/mindloop + replica pure tests + refs contract -> 78 passed
pytest services/orion-hub/tests/test_curiosity_focus.py test_curiosity_routes_runs.py
       services/orion-hub/evals/test_curiosity_focus_eval.py orion/curiosity/tests/test_focus_lifecycle.py
       tests/test_curiosity_atlas.py -> 145 passed, 29 skipped
pytest tests/test_no_personal_terms.py -> 7 passed
scripts/check_env_template_parity.py -> PASS; check_env_key_single_source.py -> OK;
check_service_env_compose_parity.py orion-mind -> N/A (env_file)
git diff --check -> clean

Regression proof: the new Prime tests run against origin/main code (test file overlaid on
`git archive origin/main`) -> 10 failed. The slow-dashboard test fails there because no move is
published (the move is refused).
```

## Evals run

```text
orion/prime/evals/run_prime_frame_eval.py -> all PASS, new: focus_read_split: none->sent; waiting->refused; unread->refused
orion/prime/evals/run_prime_block_eval.py -> PASS
orion/prime/evals/run_prime_wake_replay.py -> PASS
python -m orion.prime.evals.run_intention_take_on_replay -> ok
orion/curiosity/replica/evals/run_replica_frame_eval.py -> PASS
```

## Docker/build/smoke checks

```text
Old route, live, from inside orion-mind (2026-11-15 ~05:27Z), Prime's exact URL:
  /curiosity/api/runs?days=4&line=investigate  5.82 s, 4.55 s, 3.70 s
  /curiosity/api/focus/pending on deployed Hub -> HTTP 405 (route not deployed yet)

New route, worktree code in-process (FastAPI + httpx ASGI transport) against the live database,
read-only: 200 in 58.9, 104.2, 111.0, 44.6, 43.1 ms. Upper bound: outside the Hub container the
`enabled` settings import fails and is retried on every call.
The query itself, inside orion-athena-hub (deployed read_pending): 7.1, 0.5, 0.4, 0.4, 0.6 ms;
EXPLAIN ANALYZE 0.068 ms (seq scan; the table has 16 rows, the partial unique index exists).

The new route over HTTP from orion-mind: UNVERIFIED until Hub is deployed.
```

## Review findings fixed

Review subagent (target: `git diff origin/main...HEAD`): no MUST-FIX. It found no path that sends the move without a real, successful pending-focus read. A 405 from an older Hub fails closed. GET and DELETE on `/api/focus/...` do not shadow each other. The new tests fail on origin/main.

- Finding (SHOULD): the 30 s dashboard read now sits inside a funded thought, before the model call, and it ran after the 4 s focus read.
  - Fix: the two reads now run at the same time (`asyncio.gather`), so the thought waits for the slower one only. The remaining trade-off is in Risks below. Caching the dashboard from settling was not done, because patch 3 of the spec removes this read.
  - Evidence: orion-mind tests 396 passed after the change.
- Finding (SHOULD): 30 s is still shorter than the slowest live read (44 s).
  - Fix: kept 30 s (the spec default, accepted by Juniper on 2026-11-15). `settings.py` and `.env_example` now say the slowest reads still fail by design: the frame shows a named gap, and settling tries again at the next interval.
- Finding (NIT): the `focus_read_split` eval check passes the runtime's sentence in by hand.
  - Fix: its docstring now says it does not run the runtime's mapping. `test_prime_loop.py` covers that mapping.
- Finding (NIT): the `refusals` eval docstring still named the dashboard as the source.
  - Fix: it now says the payload shape is the same, and since #3366 that payload comes from the cheap route.
- Finding (NIT): the settings comment did not say that both dashboard reads can fall in one tick.
  - Fix: the comment now says so.
- Finding (NIT): the PR report was not committed.
  - Fix: it is committed with this branch.

## Restart required

Deploy from the clean primary main checkout after merge, Hub first:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Live check after both are up: `docker exec orion-mind python3 -c "import urllib.request,time;t=time.monotonic();print(urllib.request.urlopen('http://100.92.216.81:8080/curiosity/api/focus/pending',timeout=4).read()[:200],time.monotonic()-t)"`. Then the next Prime thought that picks a move should close with `move curiosity_focus ...`, and no "could not read whether curiosity" refusal should appear for 3 days (#3366 done-when).

## Risks / concerns

- Severity: low. Concern: a funded thought can now wait up to 30 s for the dashboard before it asks for a model seat. Before, that read failed at 4 s. The lane reservation is priced for 60 s of model time. Mitigation: both reads run at the same time. The 600 s seat wait absorbs the extra time. `PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC` can lower it. Patch 3 of the spec removes the dashboard read from the frame.
- Severity: low. Concern: settling questions now wait up to 30 s on the dashboard inside Prime's tick (only while a question is open, once per settle interval). Both dashboard reads can fall in the same tick. Mitigation: before, settling failed at 4 s, so it only gains. Reads slower than 30 s (live max 44 s) still fail and are retried at the next interval.
- Severity: low. Concern: the live move publish is UNVERIFIED until deploy. Mitigation: #3366 stays open until the done-when holds.
- Issues opened: #3438 (curiosity replica reads the same dashboard with the same 4 s timeout; UNVERIFIED that its frames lose runs).
- Issues closed: #3366 (on merge, `Fixes #3366`; live proof still owed per its done-when).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
