# PR report: cap the length of a background GPU hold (#3391)

Patch E of `docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md` (PR #3439).

## Summary

- A background GPU hold now has a time limit. After 9,000 s (2.5 h) of holding a seat without a
  break, the pool takes the seat back.
- The pool first asks for the seat back (recall reason `background_max_hold`). The holder gets the
  normal 600 s to finish its current step. Then the pool ends the hold.
- A hold that can be retried (every durable run of Orion's) goes back in the queue in its old
  place. A hold nobody retries (the trial that caused the ticket) is released for good.
- The limit is ON by default. The default lives in code (`Defaults.background_max_hold_sec`).
- No caller moves to another lease class. PR #3404 / #3417 and the graphiti trial session do that.

## Outcome moved

On 2026-11-13 one trial hold (`graphiti-trial:sample100`) kept the agent lane for 25,126 s (7 h).
Orion recorded 215 acts waiting that day. With this change the same hold is taken back at 9,000 s
and gone by 9,600 s.

## Current architecture

Only one seat had a clock: the gpu2 swap seat (`agent-gpu2`, `max_hold_sec: 9000`), counted from
when the seat was loaded. A hold on the home agent seat (`agent`, gpu1) had no limit at all. A test
pinned that on purpose (`test_a_long_hold_on_the_home_seat_is_never_capped`, 2026-09-25).

## Architecture touched

- `orion/gpu_pool` only: one config default and one scheduler rule (H6). No service code, no bus
  channel, no schema, no table.

## Files changed

- `orion/gpu_pool/config.py`: `Defaults.background_max_hold_sec` (9000, `None` = no cap).
- `orion/gpu_pool/scheduler.py`: rule H6 and the reason constant `BACKGROUND_MAX_HOLD`.
- `config/gpu_pool.yaml`: a comment only. See "Contradiction with the spec" below.
- `orion/gpu_pool/tests/test_scheduler_holds.py`: lifecycle tests for the cap.
- `orion/gpu_pool/tests/test_lease_graph.py`: a capped hold nobody retries is released; a capped
  retryable hold is re-queued in place (these two pass on old code too; they pin the lease graph).
- `services/orion-gpu-pool/README.md`: the rule and a live check.
- `services/orion-gpu-pool/evals/run_controller_stale_eval.py`: runs with the cap off (see Evals).

## Schema / bus / API changes

- Added: recall reason string `background_max_hold` on pool lease events (reasons are free text).
- Removed: none.
- Renamed: none.
- Behavior changed: a background-priority hold is recalled after 9,000 s granted.
- Compatibility notes: `Defaults` gains a field with a code default. No reader of
  `config/gpu_pool.yaml` sees a new key.

## Env/config changes

- Added keys: none in any `.env_example`. One code default, `Defaults.background_max_hold_sec`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Why 9,000 s: the real hold lengths

Read from `gpu_pool_events` joined to `gpu_pool_leases` (background holds, not operator), window
2026-11-05 to 2026-11-15 (`held_ms` exists since 11-05).

| Holder | Holds | Median | 90% | 99% | Longest | Over 9,000 s |
|---|---|---|---|---|---|---|
| Orion's durable runs | 1,650 | 337 s | 2,213 s | 5,761 s | 8,545 s | 0 |
| Graphiti trial | 30 | 287 s | 1,233 s | 18,352 s | 25,126 s | 1 |
| Operator tools | 28 | 44 s | 689 s | 2,974 s | 3,258 s | 0 |

Orion's own holds by seat: `agent` 967 holds, longest 3,538 s. `agent-gpu2` 489 holds, longest
8,545 s (that seat already has the 9,000 s seat clock).

9,000 s is the limit Juniper set for the gpu2 seat on 2026-09-29. It is above every hold Orion's
own work took in the window. It is far below the trial hold. So the cap cuts the trial and cuts
none of Orion's 1,650 holds.

Honest limit of this evidence: 10 days of data. A durable run of Orion's that one day holds a seat
longer than 9,000 s is recalled, gets 600 s, and is re-queued in its place. It is not failed.

## Contradiction with the spec

The spec says "new key in `config/gpu_pool.yaml`". I did not add a key to that file. The file
itself says why (lines 30-36, about `background_drain_grace_sec`): the `Defaults` model refuses
unknown keys, and circe's lane controller parses the same file, so a new key breaks any reader
still on older code. I followed that precedent: the default is in code, and the YAML has a comment
that says when the key can be added. The cap still ships ON.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-background-hold-cap
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/gpu_pool/tests -q -p no:cacheprovider
  584 passed
(cd services/orion-gpu-pool && ... -m pytest tests -q -p no:cacheprovider)
  202 passed, 15 skipped
(cd services/orion-gpu-lane-controller && ... -m pytest tests -q -p no:cacheprovider)
  110 passed
python scripts/check_gpu_pool_config.py
  ok (6 cards, 9 roles, 17 classes, 2 launch blocks)
git diff --check
  clean
```

Old code check: `test_a_background_hold_is_recalled_at_the_cap_and_not_before` and
`test_the_cap_is_for_any_work_class_and_a_hold_nobody_retries` fail without the scheduler rule
(no `Recall` is emitted).

## Evals run

```text
python services/orion-gpu-pool/evals/run_controller_stale_eval.py    7 of 7 PASS
python services/orion-gpu-pool/evals/value_queue_replay.py           PASS (9/9 queues)
services/orion-gpu-pool/evals/run_pool_day_eval.py                   passed in CI
services/orion-durable-runs/evals/hold_fairness.py                   passed in CI (needs a test database; not run locally)
```

CI finding: the stale-controller replay first failed ("B refusals replayed: 43", expected 136).
That eval stands one background hold on the agent seat for 26 simulated hours to mean "agent is
busy". The cap took that hold back at 9,000 s. The eval is about controller alerts, not hold
length, so it now runs with the cap off (`REPLAY_CFG`). No eval runs a hold past the cap with the
cap on; the unit lifecycle tests cover that.

## Docker/build/smoke checks

```text
Not run. No deploy was allowed for this task. The live check is UNVERIFIED until deploy.
```

## Review findings fixed

Review ran in a subagent on the branch diff against `origin/main`. Verdict: nothing material.
It confirmed by reading and by running the scheduler: no holder branches on the new reason; the
grant clock restarts on every grant; no repeat recall each tick; system holds, child calls and the
Hub button are untouched; keeping the default out of the YAML is still correct.

- Finding: one new test said the waiting run gets the seat after a capped retryable hold. It
  passed only because the capped hold was left out of the next tick. In truth a capped retryable
  hold is re-queued in its old place and wins the seat back from a younger background hold.
  - Fix: the test now uses a hold nobody retries (the trial case). A second test pins the real
    behaviour: the re-queued hold beats a younger background hold and loses to a chat turn.
  - Evidence: `test_a_capped_retryable_hold_keeps_its_place_in_the_queue`.
- Finding: the README check query had no priority filter, no time filter, and a threshold that a
  legal `draining_background` grace can pass.
  - Fix: query filters on `priority = 'background'` and deploy time; threshold 12,700 s.
- Finding: the "no cap" test skipped validation of `None`.
  - Fix: it now also checks `Defaults.model_validate({"background_max_hold_sec": None})`.
- Finding: the config comment said operator holds are bounded by their seat's own limit. Only two
  seats have one.
  - Fix: the comment now says operator holds are exempt, nothing more.
- Finding (no code change): a hold in its 600 s cap grace makes an urgent run wait up to 600 s,
  not 5 s. The same is already true for every other recall reason. Listed under risks.
- Finding (no code change): the two new `test_lease_graph.py` cases pass on old code. They pin how
  the lease graph treats the new reason; the scheduler tests are the regression tests.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

Run it from the clean primary `main` checkout after merge (the post-pull rebuild does this). The
lane controller on circe does not enforce the cap; it needs no restart for it.

## Risks / concerns

- Severity: low
- Concern: a durable run of Orion's that holds a seat past 9,000 s loses its current step after
  the 600 s grace and is re-queued.
- Mitigation: 0 of 1,650 holds reached 9,000 s in 10 days. Rollback is one line (`None`).

- Severity: low
- Concern: the cap does not move Orion's own capped run to the back of the line. A retryable hold
  keeps its old place and can be granted again at once when only younger background work waits.
  The trial case (not retryable) does end for good.
- Mitigation: none needed for #3391. If same-priority fairness is wanted, it is a separate change.

- Severity: low
- Concern: an urgent run that arrives while a hold is in its 600 s cap grace waits for that grace,
  not for the 5 s urgent grace. Existing behaviour for every other recall reason.
- Mitigation: rare (a hold at the cap plus an urgent arrival inside 600 s). Not changed here.

- Severity: low
- Concern: UNVERIFIED live. Acceptance check: start a trial hold longer than the cap and see it
  released at the cap; Orion's day shows no "agent lane full" arc from it.
- Mitigation: after deploy run the check query in `services/orion-gpu-pool/README.md`
  (Background hold cap). It must return no row.

- Issues opened: none. Issues closed: Fixes #3391.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3480
